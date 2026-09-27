"""Bundle everything the website shows into web/data/v3/site.json.

Called automatically after predict / evaluate / backtest / ratings / weekend, or
directly with `python -m flatout export`.
"""
import json
from datetime import datetime

import numpy as np
import pandas as pd

from . import pipeline, race
from .config import MODELS, OUTPUT, ROOT

WEB = ROOT / 'web' / 'data' / 'v3'


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 5)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


def _records(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient='records'))


def _next_event(hist):
    """(year, round) of the first scheduled event without a stored race result, or None."""
    from . import ingest
    raced = set(map(tuple, hist[['year', 'round']].drop_duplicates().astype(int).values.tolist()))
    top = int(hist.year.max())
    for y in (top, top + 1):
        sp = ingest.STORE / str(y) / 'schedule.parquet'
        if not sp.exists():
            continue
        for r in sorted(pd.read_parquet(sp)['round'].astype(int)):
            if (y, r) not in raced:
                return y, r
    return None


def _latest_prediction(hist=None):
    """Forecast for the next unraced event only. A stored forecast for an event that has already been
    raced is history, not 'next'; superseded forecasts live in superseded/ and are never read here."""
    nxt = _next_event(hist) if hist is not None else None
    if nxt is None:
        return None
    d = OUTPUT / str(nxt[0]) / f'R{nxt[1]:02d}'
    if not (d / 'pre_race_summary.csv').exists():
        return None
    meta = json.loads((d / 'pre_race_meta.json').read_text())
    if meta.get('location') != (meta.get('identity') or {}).get('circuit', meta.get('location')):
        raise ValueError(f'{d}: forecast location {meta.get("location")} != verified circuit; refusing to export')
    summ = pd.read_csv(d / 'pre_race_summary.csv')
    dist = pd.read_csv(d / 'pre_race_distribution.csv')
    c = meta.get('circuit', {})
    try:
        from . import ingest
        sch = pd.read_parquet(ingest.STORE / str(meta['year']) / 'schedule.parquet')
        ev = sch[sch['round'] == meta['round']].iloc[0]
        meta['date'], meta['country'], meta['format'] = ev.date, ev.country, ev.format
        meta['sessions'] = json.loads(ev.sessions)
    except Exception:
        pass
    ident = meta.get('identity') or {}
    if ident.get('race_start_utc'):
        meta.setdefault('sessions', {})['R'] = ident['race_start_utc']
    reco = None
    if (d / 'recommendations.json').exists():
        r = json.loads((d / 'recommendations.json').read_text())
        if r.get('run') == meta.get('run'):          # only if computed for this exact forecast run
            reco = r
    return dict(meta={k: v for k, v in meta.items() if k not in ('circuit', 'provenance')}, recommendations=reco,
                provenance=meta.get('provenance'),
                circuit={k: c.get(k) for k in ('location', 'event', 'n_races', 'n_laps', 'base_lap', 'pit_loss',
                                                'fuel', 'lap_sd', 'sc_per_race', 'vsc_per_race', 'overtake_factor',
                                                'offset', 'deg', 'stops_mu', 'stops_dist', 'stint_frac', 'dnf_rate',
                                                'max_stint', 'red_share')},
                drivers=_records(summ),
                dist={r[0]: [round(float(x), 5) for x in r[1:]] for r in dist.itertuples(index=False)})


def _country(year, rnd):
    from . import ingest
    try:
        sch = pd.read_parquet(ingest.STORE / str(year) / 'schedule.parquet')
        return str(sch[sch['round'] == rnd].country.iloc[0])
    except Exception:
        return ''


def _past_races(hist):
    out = []
    for p in sorted(OUTPUT.glob('*/R*')):
        year, rnd = int(p.parent.name), int(p.name[1:])
        act = hist[(hist.year == year) & (hist['round'] == rnd)].set_index('Driver')
        if not len(act):
            continue
        entry = dict(year=year, round=rnd, location=act.location.iloc[0], country=_country(year, rnd), predictions={})
        for tag in ('pre_race', 'backtest_post_quali', 'backtest_pre_weekend'):
            sp, ep = p / f'{tag}_summary.csv', p / f'{tag}_evaluation.json'
            if not (sp.exists() and ep.exists()):
                continue
            s = pd.read_csv(sp)
            s['actual'] = s.Driver.map(act.finish)
            s['actual_grid'] = s.Driver.map(act.grid)
            s['actual_dnf'] = s.Driver.map(act.dnf)
            s['actual_strategy'] = s.Driver.map(act.strategy)
            s['actual_stops'] = s.Driver.map(act.n_stops)
            s['actual_first_stop'] = s.Driver.map(act.first_stop)
            s['actual_pace'] = s.Driver.map(act.pace_pct)
            cols = ['Driver', 'Team', 'grid', 'win', 'podium', 'points', 'dnf', 'exp_pos', 'p10', 'p90', 'pace_pct',
                    'stop1', 'stop2', 'stop3p', 'strategy', 'first_stop_p50', 'actual', 'actual_grid', 'actual_dnf',
                    'actual_strategy', 'actual_stops', 'actual_first_stop', 'actual_pace']
            entry['predictions'][tag] = dict(metrics=json.loads(ep.read_text()),
                                             drivers=_records(s[[c for c in cols if c in s]]))
        if entry['predictions']:
            out.append(entry)
    return out


def _reliability(hist, tag='backtest_post_quali'):
    rows = []
    for p in OUTPUT.glob(f'*/R*/{tag}_summary.csv'):
        year, rnd = int(p.parent.parent.name), int(p.parent.name[1:])
        act = hist[(hist.year == year) & (hist['round'] == rnd)].set_index('Driver')
        s = pd.read_csv(p)
        s = s[s.Driver.isin(act.index)]
        fin = s.Driver.map(act.finish)
        for kind, col, cut in (('win', 'win', 1), ('podium', 'podium', 3), ('points', 'points', 10)):
            rows += [(kind, pp, int(f <= cut)) for pp, f in zip(s[col], fin)]
    df = pd.DataFrame(rows, columns=['kind', 'p', 'y'])
    out = {}
    bins = [0, .02, .05, .1, .2, .3, .45, .6, .75, .9, 1.0001]
    for kind, g in df.groupby('kind'):
        b = pd.cut(g.p, bins, right=False)
        t = g.groupby(b, observed=True).agg(n=('y', 'size'), predicted=('p', 'mean'), observed=('y', 'mean'))
        out[kind] = _records(t.reset_index(drop=True))
    return out


def _season_summary():
    out = {}
    for p in OUTPUT.glob('backtest_*_*.csv'):
        if 'driver_errors' in p.name:
            continue
        _, year, *mode = p.stem.split('_')
        mode = '_'.join(mode)
        res = pd.read_csv(p)
        keys = ['log_loss', 'rps', 'brier_win', 'brier_podium', 'brier_points', 'spearman', 'mae_pos', 'p_winner',
                'top3_hit', 'top10_hit', 'winner_correct']
        avg = {src: {k: float(res[f'{src}_{k}'].astype(float).mean()) for k in keys if f'{src}_{k}' in res}
               for src in ('model', 'grid', 'pace')}
        strat = {k[6:]: float(res[k].mean()) for k in res.columns if k.startswith('strat_')}
        per = res[['round', 'event', 'model_rps', 'grid_rps', 'pace_rps', 'model_log_loss', 'grid_log_loss',
                   'model_spearman', 'model_p_winner', 'grid_p_winner', 'model_winner_correct',
                   'strat_stops_acc', 'strat_first_stop_mae']]
        out.setdefault(year, {})[mode] = dict(avg=avg, strategy=strat, per_race=_records(per), n=len(res))
    return out


def _benchmarks():
    """Latest nested walk-forward experiment per season (artifacts/experiments/ledger.jsonl), with per-race
    rows. These are the only fully walk-forward numbers; 'season_eval' (backtest) is in-sample for the
    simulator parameters and is kept for the race explorer only."""
    from .config import ROOT
    ledger = ROOT / 'artifacts' / 'experiments' / 'ledger.jsonl'
    if not ledger.exists():
        return {}
    latest = {}
    for line in ledger.read_text(encoding='utf-8').splitlines():
        try:
            e = json.loads(line)
        except ValueError:
            continue
        d = ROOT / 'artifacts' / 'experiments' / e['id']
        if not e['id'].startswith('nested_') or not (d / 'per_race.csv').exists():
            continue
        cfg = json.loads((d / 'config.json').read_text())
        latest[str(cfg['year'])] = (e, d, cfg)       # later lines win
    out = {}
    for year, (e, d, cfg) in latest.items():
        per = pd.read_csv(d / 'per_race.csv')
        keep = [c for c in ('year', 'round', 'event', 'mode', 'n', 'model_rps', 'grid_rps', 'pace_rps',
                            'model_log_loss', 'grid_log_loss', 'pace_log_loss', 'model_p_winner', 'grid_p_winner',
                            'model_winner_correct', 'grid_winner_correct', 'model_top3_hit', 'new_venue')
                if c in per]
        out[year] = dict(id=e['id'], created_utc=e['created_utc'], summary=e['summary'],
                         config={k: cfg[k] for k in ('eval_sims', 'calib_sims', 'calib_last_n', 'rounds', 'first', 'last')},
                         code=e.get('code', {}), per_race=_records(per[keep]),
                         status='development', note=('Nested walk-forward: simulator parameters re-tuned on the '
                                                     'races before each scored race. These seasons were inspected '
                                                     'during development, so this is not an untouched holdout.'))
    return out


def _standings(analyses, year):
    pts = {}
    team = {}
    results = {}
    for a in analyses:
        s = a['summary']
        if s['year'] != year:
            continue
        for d in a['drivers'].to_dict('records'):
            pts[d['Driver']] = pts.get(d['Driver'], 0) + (d.get('points') or 0)
            team[d['Driver']] = d['Team']
            if s['code'] == 'R':
                results.setdefault(d['Driver'], []).append(dict(round=s['round'], finish=d['finish'],
                                                                dnf=bool(d['dnf']), grid=d['grid']))
    rows = [dict(Driver=k, Team=team[k], points=v, results=sorted(results.get(k, []), key=lambda r: r['round']))
            for k, v in pts.items()]
    rows.sort(key=lambda r: -r['points'])
    teams = {}
    for r in rows:
        teams[r['Team']] = teams.get(r['Team'], 0) + r['points']
    return dict(drivers=rows, teams=[dict(Team=k, points=v) for k, v in sorted(teams.items(), key=lambda x: -x[1])])


F1_CDN = 'https://media.formula1.com/image/upload'


def _slug(team):
    return team.lower().replace(' ', '')


def _people(year):
    """Official names, photos, team colours and logos (F1 media CDN via FastF1 driver ids)."""
    import re
    from . import ingest
    from .features import team_norm
    people, colors = {}, {}
    for y in (year - 1, year):
        for rnd in range(1, 30):
            for code in ('Q', 'R'):
                res = ingest.read(y, rnd, code, 'results')
                if res is None:
                    continue
                for _, r in res.iterrows():
                    team = team_norm(r.TeamName)
                    url = r.get('HeadshotUrl') if isinstance(r.get('HeadshotUrl'), str) else ''
                    m = re.search(r'/([a-z]{6}\d\d)\.png', url)
                    p = dict(name=r.get('FullName') or r.Abbreviation, team=team, number=str(r.get('DriverNumber') or ''),
                             headshot=url.replace('/1col/', '/2col/') if url.startswith('http') else None)
                    if m and y == year:
                        # current-season photos: face crop for avatars, head-and-shoulders for hero cards
                        base = f"v1740000000/common/f1/{y}/{_slug(team)}/{m.group(1)}/{y}{_slug(team)}{m.group(1)}right.webp"
                        p['photo'] = f"{F1_CDN}/c_thumb,g_face,z_0.7,w_160,h_160/q_auto/{base}"
                        p['cutout'] = f"{F1_CDN}/c_thumb,g_face,z_0.42,w_480,h_480/q_auto/{base}"
                    people[r.Abbreviation] = p
                    if isinstance(r.get('TeamColor'), str) and len(r.TeamColor) == 6:
                        colors[team] = '#' + r.TeamColor
    return people, colors


def _team_logos(teams, year):
    return {t: f"{F1_CDN}/c_fit,h_96/q_auto/v1740000000/common/f1/{year}/{_slug(t)}/{year}{_slug(t)}logowhite.webp" for t in teams}


def export(log=print):
    from . import features, ratings
    analyses = race.analyze_all(log=lambda *a: None)
    hist = features.history_table(analyses)
    year = int(hist.year.max())
    site = dict(generated=datetime.now().isoformat(timespec='seconds'), season=year)
    site['next'] = _latest_prediction(hist)
    if site['next']:
        try:
            from . import tracks
            site['next']['track'] = tracks.outline(site['next']['meta']['location'])
        except Exception as e:  # the map is decoration; never block the export
            log(f'  ! track outline unavailable: {e}')
    site['races'] = _past_races(hist)
    site['season_eval'] = _season_summary()
    site['benchmarks'] = _benchmarks()
    site['reliability'] = _reliability(hist)
    site['standings'] = _standings(analyses, year)
    site['people'], site['team_colors'] = _people(year)
    site['team_logos'] = _team_logos(site['team_colors'].keys(), year)
    site['constructors'] = _records(ratings.constructor_ratings(hist, year))
    site['driver_vs_car'] = _records(ratings.driver_vs_car(hist, year))
    site['driver_vs_car_note'] = ratings.DVC_NOTE
    r = ratings.driver_ratings(hist)
    latest = r.last_race.value_counts().index[0]
    site['ratings'] = _records(r[r.last_race == latest])
    model = {}
    for f in ('pace_models_meta.json', 'sim_params.json'):
        if (MODELS / f).exists():
            model[f.split('.')[0]] = json.loads((MODELS / f).read_text())
    try:
        from .model import PaceModels
        model['importance'] = _records(PaceModels.load().importance())
    except Exception:
        pass
    if (MODELS / 'calibration_history.csv').exists():
        model['calibration_history'] = _records(pd.read_csv(MODELS / 'calibration_history.csv'))
    p = OUTPUT / 'performance_log.csv'
    if p.exists():
        model['performance_log'] = _records(pd.read_csv(p))
    site['model'] = model
    WEB.mkdir(parents=True, exist_ok=True)
    (WEB / 'site.json').write_text(json.dumps(_clean(site), separators=(',', ':')))
    log(f"  website data -> {WEB / 'site.json'} ({(WEB / 'site.json').stat().st_size / 1024:.0f} KB)")
