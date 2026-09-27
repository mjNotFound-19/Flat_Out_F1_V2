"""Analyse one stored race (or sprint) into clean, reusable facts.

The core is a per-race regression on clean green-flag laps:

    LapTime = driver_d + fuel * LapNumber + offset_compound + deg_compound * TyreLife + e

so a single fit yields fuel-and-tyre-corrected race pace per driver (the model's
training target), the circuit's degradation curves, and per-driver consistency.
Pit loss, SC/VSC/red-flag counts, on-track overtakes, lap-1 gains and the
executed strategies are extracted alongside.
"""
import json

import numpy as np
import pandas as pd

from . import ingest
from .config import DERIVED, DRY, venue

RACES_DIR = DERIVED / 'races'
RACES_DIR.mkdir(parents=True, exist_ok=True)
ANALYSIS_VERSION = 8   # 8: explicit outcome (dsq/dns are not retirements), dsq laps from lap data

DNF_CODES = {'R', 'D', 'E', 'W', 'F', 'N'}


def _green(ts):
    s = str(ts)
    return s != 'nan' and set(s) <= {'1'}


def _status_flags(laps, rcm, n_laps):
    """Per-lap SC / VSC / red flags plus deployment counts."""
    by_lap = laps.groupby('LapNumber').TrackStatus.agg(lambda s: ''.join(map(str, s.dropna())))
    sc = {int(l) for l, s in by_lap.items() if '4' in s}
    vsc = {int(l) for l, s in by_lap.items() if ('6' in s or '7' in s) and '4' not in s}
    red = {int(l) for l, s in by_lap.items() if '5' in s}
    counts = dict(sc=0, vsc=0, red=0)
    lap1_sc = False
    if rcm is not None and 'Message' in rcm:
        msg = rcm.Message.fillna('').str.upper()
        dep_sc = rcm[msg.str.contains('SAFETY CAR DEPLOYED') & ~msg.str.contains('VIRTUAL')]
        dep_vsc = rcm[msg.str.contains('VIRTUAL SAFETY CAR DEPLOYED')]
        flag = rcm.Flag.fillna('') if 'Flag' in rcm else pd.Series('', index=rcm.index)
        dep_red = rcm[(flag == 'RED') | msg.str.startswith('RED FLAG')]
        red_runs = sum(1 for l in red if l - 1 not in red)
        counts = dict(sc=len(dep_sc), vsc=len(dep_vsc), red=max(len(dep_red), red_runs))
        if 'Lap' in dep_sc and len(dep_sc):
            lap1_sc = bool((dep_sc.Lap.fillna(99) <= 1).any())
    else:  # fall back to contiguous runs of status laps
        for key, s in (('sc', sc), ('vsc', vsc), ('red', red)):
            counts[key] = sum(1 for l in s if l - 1 not in s)
    return sc, vsc, red, counts, lap1_sc


# prior sd: fuel 0.03 s/lap, deg 0.05 s/lap/lap, offsets 1 s  (weight = residual sd 0.5 / prior sd)
PRIOR_W = dict(fuel=0.5 / 0.03, deg=0.5 / 0.05, offset=0.5 / 1.0)
PRIOR_MEAN = dict(fuel=-0.055, deg=0.05)


def _tyre_stints(laps):
    """Stint id per lap from actual tyre changes (compound change or tyre age reset).
    FastF1's Stint counter also ticks on phantom pit entries (e.g. finishing behind the SC)."""
    comp = laps.groupby('Driver').Compound.ffill()
    life = laps.TyreLife
    prev_c = comp.groupby(laps.Driver).shift()
    prev_l = life.groupby(laps.Driver).shift()
    new = prev_c.isna() | (comp != prev_c) | (life < prev_l)
    return new.astype(int).groupby(laps.Driver).cumsum()


def _fit_pace(clean):
    """Robust dummy-variable least squares. Returns driver effects, fuel, compound params, residuals."""
    drivers = sorted(clean.Driver.unique())
    comps = [c for c in clean.Compound.value_counts().index]
    ref = comps[0]
    didx = {d: i for i, d in enumerate(drivers)}
    keep = np.ones(len(clean), bool)
    for _ in range(3):
        c = clean[keep]
        X = np.zeros((len(c), len(drivers) + 1 + (len(comps) - 1) + len(comps)))
        X[np.arange(len(c)), c.Driver.map(didx).values] = 1
        X[:, len(drivers)] = c.LapNumber.values
        col = len(drivers) + 1
        for comp in comps[1:]:
            X[:, col] = (c.Compound == comp).values
            col += 1
        for comp in comps:
            X[:, col] = np.where(c.Compound == comp, c.TyreLife.values, 0)
            col += 1
        # Weak priors on fuel / offsets / deg keep the fit stable when lap number and tyre age are
        # nearly collinear (one-stop races); with normal data they are swamped by ~1000 laps.
        P = np.zeros((1 + (len(comps) - 1) + len(comps), X.shape[1]))
        t = np.zeros(len(P))
        P[0, len(drivers)], t[0] = PRIOR_W['fuel'], PRIOR_W['fuel'] * PRIOR_MEAN['fuel']
        for i in range(len(comps) - 1):
            P[1 + i, len(drivers) + 1 + i] = PRIOR_W['offset']
        for i, comp in enumerate(comps):
            j = len(comps) + i
            P[j, len(drivers) + len(comps) + i] = PRIOR_W['deg']
            t[j] = PRIOR_W['deg'] * PRIOR_MEAN['deg']
        beta, *_ = np.linalg.lstsq(np.vstack([X, P]), np.r_[c.LapTime.values, t], rcond=None)
        resid_all = clean.LapTime.values - _predict(clean, beta, drivers, comps, didx)
        mad = np.median(np.abs(resid_all[keep] - np.median(resid_all[keep]))) * 1.4826
        new_keep = np.abs(resid_all) < max(3 * mad, 0.6)
        if (new_keep == keep).all():
            break
        keep = new_keep
    nd = len(drivers)
    fx = dict(zip(drivers, beta[:nd]))
    fuel = beta[nd]
    offset = {ref: 0.0}
    for i, comp in enumerate(comps[1:]):
        offset[comp] = beta[nd + 1 + i]
    deg = {comp: beta[nd + len(comps) + i] for i, comp in enumerate(comps)}
    resid = clean.LapTime.values - _predict(clean, beta, drivers, comps, didx)
    return fx, fuel, offset, deg, resid, keep, ref


def _predict(df, beta, drivers, comps, didx):
    nd = len(drivers)
    out = beta[df.Driver.map(didx).values] + beta[nd] * df.LapNumber.values
    for i, comp in enumerate(comps[1:]):
        out = out + np.where(df.Compound == comp, beta[nd + 1 + i], 0)
    for i, comp in enumerate(comps):
        out = out + np.where(df.Compound == comp, beta[nd + len(comps) + i] * df.TyreLife.values, 0)
    return out


def _overtakes(laps, pit_laps, green_laps):
    """Count on-track position swaps between cars that were not pitting, lap by lap."""
    pos = laps.pivot_table(index='LapNumber', columns='Driver', values='Position')
    total, per_driver = 0, {}
    for lap in pos.index[1:]:
        if lap not in green_laps or lap - 1 not in green_laps:
            continue
        prev, cur = pos.loc[lap - 1], pos.loc[lap]
        ok = [d for d in pos.columns if pd.notna(prev[d]) and pd.notna(cur[d])
              and (d, lap) not in pit_laps and (d, lap - 1) not in pit_laps]
        if len(ok) < 2:
            continue
        p0 = prev[ok].rank().values
        p1 = cur[ok].rank().values
        for i in range(len(ok)):
            for j in range(len(ok)):
                if p0[i] > p0[j] and p1[i] < p1[j]:   # i was behind j and is now ahead
                    total += 1
                    per_driver[ok[i]] = per_driver.get(ok[i], 0) + 1
    return total, per_driver


def analyze(year, rnd, code='R', force=False):
    """Return dict(summary=..., drivers=DataFrame, stints=DataFrame) for a stored race/sprint."""
    key = f'{year}_R{rnd:02d}_{code}'
    cache = RACES_DIR / f'{key}.json'
    if cache.exists() and not force:
        blob = json.loads(cache.read_text())
        if blob.get('version') == ANALYSIS_VERSION:
            st = pd.DataFrame(blob['stints'])
            if len(st):
                st['sc_stop'] = st.sc_stop.astype(bool)
            blob['summary']['location'] = venue(blob['summary']['location'])
            return dict(summary=blob['summary'], drivers=pd.DataFrame(blob['drivers']), stints=st)
    laps = ingest.read(year, rnd, code, 'laps')
    res = ingest.read(year, rnd, code, 'results')
    if laps is None or res is None or len(laps) == 0:
        return None
    rcm = ingest.read(year, rnd, code, 'rcm')
    weather = ingest.read(year, rnd, code, 'weather')
    sched = ingest.load_schedule(year) if not (ingest.STORE / str(year) / 'schedule.parquet').exists() \
        else pd.read_parquet(ingest.STORE / str(year) / 'schedule.parquet')
    ev = sched[sched['round'] == rnd].iloc[0]

    laps = laps[laps.LapNumber.notna()].copy()
    laps['LapNumber'] = laps.LapNumber.astype(int)
    n_laps = int(max(laps.LapNumber.max(), res.Laps.max() if 'Laps' in res else 0))
    sc, vsc, red, counts, lap1_sc = _status_flags(laps, rcm, n_laps)
    laps['green'] = laps.TrackStatus.map(_green)
    laps = laps.sort_values(['Driver', 'LapNumber'])
    laps['prev_green'] = laps.groupby('Driver').green.shift(1).astype('boolean').fillna(False).astype(bool)
    laps['pit'] = laps.PitInTime.notna() | laps.PitOutTime.notna()
    laps['Stint'] = _tyre_stints(laps)

    comp_share = laps.Compound.value_counts(normalize=True)
    wet_share = comp_share.get('INTERMEDIATE', 0) + comp_share.get('WET', 0)
    rain = bool(weather is not None and 'Rainfall' in weather and weather.Rainfall.astype(bool).mean() > 0.1)
    wet = wet_share > 0.10 or rain

    # ---- pace regression on clean laps
    clean = laps[(laps.LapNumber > 1) & laps.LapTime.notna() & ~laps.pit & laps.green & laps.prev_green
                 & laps.Compound.notna() & laps.TyreLife.notna()].copy()
    if 'Deleted' in clean:
        clean = clean[~clean.Deleted.astype('boolean').fillna(False).astype(bool)]
    med = clean.LapTime.median()
    clean = clean[(clean.LapTime < med * 1.10)]
    clean = clean[clean.groupby('Driver').LapTime.transform(lambda s: s < s.median() * 1.05)]
    cnt = clean.Driver.value_counts()
    clean = clean[clean.Driver.isin(cnt[cnt >= 5].index)]
    fx, fuel, offset, deg, resid, keep, ref_comp = _fit_pace(clean) if len(clean) > 50 else ({}, np.nan, {}, {}, np.array([]), np.array([], bool), None)
    ref_fx = np.median([v for d, v in fx.items() if cnt.get(d, 0) >= 10]) if fx else np.nan
    # Pace is relative (driver effect minus field median) scaled by the real median lap, so it
    # stays correct even when fuel and tyre age are collinear and the intercept is meaningless.
    med_lap = float(clean.LapTime.median()) if len(clean) else np.nan
    fit_ok = bool(fx) and np.isfinite(fuel) and -0.25 < fuel < 0.08 and         all(-0.08 < v < 0.6 for v in deg.values()) and all(-3 < v < 3 for v in offset.values())
    base_fx = med_lap
    clean = clean.assign(resid=resid if len(resid) else np.nan, used=keep if len(keep) else False)
    lap_sd = float(np.std(resid[keep])) if len(resid) else np.nan

    # ---- pit stops and pit loss
    pit_laps = set()
    pit_losses = []
    pit_by = {}
    for drv, g in laps.groupby('Driver'):
        g = g.set_index('LapNumber')
        ref = g[~g.pit & g.green & g.LapTime.notna()].LapTime
        for lap in g.index[g.PitInTime.notna()]:
            pit_laps.update({(drv, lap), (drv, lap + 1)})
            real = lap + 1 in g.index and g.loc[lap + 1, 'Stint'] != g.loc[lap, 'Stint']
            if real and lap + 1 in g.index and lap > 1 and g.loc[lap, 'green'] and g.loc[lap + 1, 'green'] \
                    and pd.notna(g.loc[lap, 'LapTime']) and pd.notna(g.loc[lap + 1, 'LapTime']):
                near = ref[(ref.index > lap - 6) & (ref.index < lap + 7)]
                if len(near) >= 3:
                    pit_losses.append(g.loc[lap, 'LapTime'] + g.loc[lap + 1, 'LapTime'] - 2 * near.median())
                    pit_by.setdefault(drv, []).append(pit_losses[-1])
    pit_loss = float(np.median(pit_losses)) if len(pit_losses) >= 5 else np.nan

    green_laps = set(laps[laps.green].LapNumber.unique()) - sc - vsc - red
    n_passes, passes_by = _overtakes(laps, pit_laps, green_laps)

    # ---- stints
    stint_rows = []
    for (drv, st), g in laps.groupby(['Driver', 'Stint']):
        comp = g.Compound.mode()
        stint_rows.append(dict(Driver=drv, stint=int(st), compound=comp.iloc[0] if len(comp) else None,
                               start=int(g.LapNumber.min()), end=int(g.LapNumber.max()),
                               laps=int(g.LapNumber.max() - g.LapNumber.min() + 1),
                               start_age=float(g.TyreLife.min()) if g.TyreLife.notna().any() else np.nan,
                               sc_stop=bool({int(g.LapNumber.min()) - 1, int(g.LapNumber.min())} & (sc | vsc | red))))
    stints = pd.DataFrame(stint_rows)
    if len(stints):
        stints['sc_stop'] = stints.sc_stop.astype(bool)

    # ---- per-driver table
    n_drivers = len(res)
    lap1 = laps[laps.LapNumber == 1].set_index('Driver').Position
    rows = []
    for _, r in res.iterrows():
        d = r.Abbreviation
        grid = r.GridPosition if pd.notna(r.GridPosition) and r.GridPosition > 0 else n_drivers
        cls = str(r.get('ClassifiedPosition', ''))
        finished = cls.isdigit()
        outcome = _outcome(cls, str(r.get('Status', '')))
        # dnf = the car stopped running (mechanical / incident) and was not classified. A disqualified
        # car ran the race and a non-starter never took the start: neither is a retirement.
        dnf = outcome == 'retired'
        n_run = int(laps[laps.Driver == d].LapNumber.max()) if (laps.Driver == d).any() else 0
        laps_done = int(r.Laps) if pd.notna(r.get('Laps')) and r.Laps > 0 else n_run
        pos = r.Position if pd.notna(r.Position) else n_drivers
        ds = stints[stints.Driver == d].sort_values('stint') if len(stints) else pd.DataFrame()
        seq = [c for c in ds.compound] if len(ds) else []
        dres = clean[(clean.Driver == d) & clean.used]
        # tyre management: how much faster than the field model this driver's laps fade with tyre age
        deg_rel = np.nan
        if len(dres) >= 12 and dres.TyreLife.std() > 2:
            xs = dres.TyreLife - dres.groupby('Stint').TyreLife.transform('mean')
            ys = dres.resid - dres.groupby('Stint').resid.transform('mean')
            if (xs ** 2).sum() > 0:
                deg_rel = float((xs * ys).sum() / (xs ** 2).sum())
        rows.append(dict(
            Driver=d, Team=r.TeamName, grid=int(grid), finish=int(pos), classified=finished, dnf=dnf,
            outcome=outcome, status=r.Status, laps_done=laps_done,
            points=float(r.Points) if pd.notna(r.get('Points')) else 0.0,
            pace_s=(fx[d] - ref_fx) if d in fx else np.nan,
            pace_pct=((fx[d] - ref_fx) / med_lap * 100) if d in fx else np.nan,
            pace_laps=int(len(dres)),
            consistency=float(dres.resid.std()) if len(dres) >= 5 else np.nan,
            deg_rel=deg_rel,
            pit_rel=(float(np.median(pit_by[d])) - pit_loss) if d in pit_by and np.isfinite(pit_loss) else np.nan,
            lap1_gain=float(grid - lap1[d]) if d in lap1.index and pd.notna(lap1[d]) else np.nan,
            passes=int(passes_by.get(d, 0)),
            n_stops=max(len(seq) - 1, 0),
            n_green_stops=int((~ds.sc_stop.iloc[1:]).sum()) if len(ds) > 1 else 0,
            strategy='-'.join(s[0] if s else '?' for s in seq),
            start_compound=seq[0] if seq else None,
            first_stop=int(ds.end.iloc[0]) if len(ds) > 1 else np.nan,
        ))
    drivers = pd.DataFrame(rows)

    summary = dict(
        year=year, round=rnd, code=code, event=ev.event, location=ev.location, n_laps=n_laps,
        n_drivers=n_drivers, wet=wet, fit_ok=fit_ok, base_lap=float(med_lap) if pd.notna(med_lap) else None,
        fuel=float(fuel) if fit_ok else None, ref_compound=ref_comp,
        offset={k: float(v) for k, v in offset.items()} if fit_ok else {},
        deg={k: float(v) for k, v in deg.items()} if fit_ok else {},
        lap_sd=lap_sd, pit_loss=pit_loss, n_pit_samples=len(pit_losses),
        sc=counts['sc'], vsc=counts['vsc'], red=counts['red'], lap1_sc=lap1_sc,
        sc_laps=len(sc), vsc_laps=len(vsc), passes=n_passes,
        green_laps=len(green_laps), dnfs=int(drivers.dnf.sum()),
    )
    blob = dict(version=ANALYSIS_VERSION, summary=summary,
                drivers=drivers.replace({np.nan: None}).to_dict('records'),
                stints=stints.replace({np.nan: None}).to_dict('records'))
    cache.write_text(json.dumps(blob, default=float))
    summary['location'] = venue(summary['location'])
    return dict(summary=summary, drivers=drivers, stints=stints)


def _outcome(classified_position, status):
    """finished | classified_retirement | retired | dsq | dns | not_classified (running, <90% distance)."""
    st = status.strip().lower()
    cp = classified_position.strip().upper()
    if cp == 'D' or 'disqualif' in st:
        return 'dsq'
    if cp == 'W' or 'did not start' in st or st in ('withdrawn', 'dns'):
        return 'dns'
    if cp.isdigit():
        return 'finished' if (st == 'finished' or st.startswith('+') or 'lap' in st) else 'classified_retirement'
    if cp == 'N':
        return 'not_classified'
    return 'retired'


def analyze_all(force=False, log=print):
    out = []
    for year, rnd in ingest.stored_events():
        for code in ('R', 'S'):
            if (ingest.session_dir(year, rnd, code) / 'laps.parquet').exists():
                try:
                    a = analyze(year, rnd, code, force=force)
                    if a:
                        out.append(a)
                except Exception as e:
                    log(f'  ! analyse {year} R{rnd} {code}: {e}')
    return out
