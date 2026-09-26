"""Incremental FastF1 sync into a parquet store.

Every finished session of every requested season lands in
    data/store/<year>/<rr>_<CODE>/{laps,results,weather,rcm}.parquet + meta.json
Sessions already stored with status 'ok' are skipped, so `sync` is cheap to re-run
after every session of a weekend.
"""
import json
import logging
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from .config import CACHE, SESSION_CODES, STORE

TD_COLS = ('Time', 'LapTime', 'PitOutTime', 'PitInTime', 'Sector1Time', 'Sector2Time',
           'Sector3Time', 'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime',
           'LapStartTime', 'Q1', 'Q2', 'Q3')

# Wait this long after a session's scheduled start before trying to pull it.
SESSION_SETTLE = {'R': timedelta(hours=4), 'S': timedelta(hours=2)}
DEFAULT_SETTLE = timedelta(hours=3)


def _fastf1():
    import fastf1
    logging.getLogger('fastf1').setLevel(logging.ERROR)
    fastf1.Cache.enable_cache(str(CACHE))
    return fastf1


def _to_seconds(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = df[c].astype(str)
        elif c in TD_COLS or pd.api.types.is_timedelta64_dtype(df[c]):
            df[c] = pd.to_timedelta(df[c], errors='coerce').dt.total_seconds()
    return df


def session_dir(year: int, rnd: int, code: str):
    return STORE / str(year) / f'{rnd:02d}_{code}'


def load_schedule(year: int) -> pd.DataFrame:
    path = STORE / str(year) / 'schedule.parquet'
    ff1 = _fastf1()
    try:
        sch = ff1.get_event_schedule(year, include_testing=False)
        rows = []
        for _, ev in sch.iterrows():
            sessions = {}
            for i in range(1, 6):
                name = ev.get(f'Session{i}')
                code = SESSION_CODES.get(name)
                when = ev.get(f'Session{i}DateUtc')
                if code and pd.notna(when):
                    sessions[code] = pd.Timestamp(when).tz_localize('UTC').isoformat()
            rows.append(dict(year=year, round=int(ev.RoundNumber), event=ev.EventName,
                             location=ev.Location, country=ev.Country,
                             format=ev.EventFormat, date=str(pd.Timestamp(ev.EventDate).date()),
                             sessions=json.dumps(sessions)))
        df = pd.DataFrame(rows)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        return df
    except Exception:
        if path.exists():
            return pd.read_parquet(path)
        raise


def all_schedules() -> pd.DataFrame:
    frames = [pd.read_parquet(p) for p in STORE.glob('*/schedule.parquet')]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _pull_session(ff1, year, rnd, code, out, light=False):
    s = ff1.get_session(year, rnd, code)
    extras = not light or code in ('R', 'S')
    s.load(laps=True, telemetry=False, weather=extras, messages=extras)
    out.mkdir(parents=True, exist_ok=True)
    laps = pd.DataFrame(s.laps)
    if len(laps) == 0 and code != 'Q':
        raise RuntimeError('no laps returned')
    _to_seconds(laps).to_parquet(out / 'laps.parquet', index=False)
    _to_seconds(pd.DataFrame(s.results)).to_parquet(out / 'results.parquet', index=False)
    if extras and s.weather_data is not None and len(s.weather_data):
        _to_seconds(pd.DataFrame(s.weather_data)).to_parquet(out / 'weather.parquet', index=False)
    if extras and s.race_control_messages is not None and len(s.race_control_messages):
        _to_seconds(pd.DataFrame(s.race_control_messages)).to_parquet(out / 'rcm.parquet', index=False)
    total_laps = getattr(s, 'total_laps', None)
    return dict(n_laps=len(laps), n_drivers=int(laps.Driver.nunique()) if len(laps) else 0,
                total_laps=int(total_laps) if total_laps else None,
                event=s.event.EventName)


HISTORY_CODES = ('R', 'Q', 'S', 'SQ', 'FP2')


def sync(years, codes=None, force=False, max_retries=2, full_history=False, log=print):
    """Pull every finished session for `years` that is not stored yet.

    Past seasons only pull what training uses (race, quali, sprint sessions, FP2,
    and FP1 on sprint weekends) to stay inside FastF1's 500 calls/hour limit.
    Hitting the limit pauses and resumes instead of failing.
    """
    ff1 = _fastf1()
    now = datetime.now(timezone.utc)
    cur_year = now.year
    pulled, failed = 0, 0
    for year in years:
        sch = load_schedule(year)
        for _, ev in sch.iterrows():
            sess = json.loads(ev.sessions)
            for code, start in sess.items():
                if codes and code not in codes:
                    continue
                light = year < cur_year and not full_history
                if light and code not in HISTORY_CODES and not (code == 'FP1' and 'SQ' in sess):
                    continue
                start = datetime.fromisoformat(start)
                if start + SESSION_SETTLE.get(code, DEFAULT_SETTLE) > now:
                    continue
                out = session_dir(year, ev['round'], code)
                meta_p = out / 'meta.json'
                if meta_p.exists() and not force:
                    meta = json.loads(meta_p.read_text())
                    if meta.get('status') == 'ok' or meta.get('attempts', 0) >= max_retries:
                        continue
                else:
                    meta = {}
                t0 = time.time()
                while True:
                    try:
                        info = _pull_session(ff1, year, int(ev['round']), code, out, light=light)
                        break
                    except Exception as e:
                        if 'RateLimit' not in type(e).__name__ and '500 calls' not in str(e):
                            info = e
                            break
                        log(f'    rate limit reached - pausing 10 min ({datetime.now():%H:%M})')
                        time.sleep(600)
                try:
                    if isinstance(info, Exception):
                        raise info
                    meta = dict(status='ok', pulled_at=now.isoformat(), **info)
                    pulled += 1
                    log(f"  + {year} R{ev['round']:02d} {code:3s} {ev.event:<28s} "
                        f"{info['n_laps']:5d} laps  ({time.time() - t0:4.0f}s)")
                except Exception as e:  # keep going; one broken session must not stop the backfill
                    out.mkdir(parents=True, exist_ok=True)
                    meta = dict(status='failed', error=str(e)[:300],
                                attempts=meta.get('attempts', 0) + 1, pulled_at=now.isoformat())
                    failed += 1
                    log(f"  ! {year} R{ev['round']:02d} {code:3s} {ev.event:<28s} FAILED: {str(e)[:80]}")
                meta_p.write_text(json.dumps(meta, indent=1))
    log(f'  sync done: {pulled} pulled, {failed} failed')
    return pulled


# ---------------------------------------------------------------- readers

def read(year, rnd, code, table='laps'):
    p = session_dir(year, rnd, code) / f'{table}.parquet'
    return pd.read_parquet(p) if p.exists() else None


def available_sessions(year, rnd):
    base = STORE / str(year)
    return sorted(p.name.split('_', 1)[1] for p in base.glob(f'{rnd:02d}_*')
                  if (p / 'meta.json').exists()
                  and json.loads((p / 'meta.json').read_text()).get('status') == 'ok')


def stored_events():
    """(year, round) pairs that have a stored race."""
    out = []
    for p in STORE.glob('*/*_R'):
        m = p / 'meta.json'
        if m.exists() and json.loads(m.read_text()).get('status') == 'ok':
            out.append((int(p.parent.name), int(p.name.split('_')[0])))
    return sorted(out)
