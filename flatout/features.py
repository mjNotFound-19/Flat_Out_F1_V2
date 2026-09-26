"""Pre-race features per (event, driver), strictly from information available before lights out.

Sources: qualifying, sprint qualifying, sprint race, FP long runs and short runs,
plus exponentially weighted form from earlier races. The training target is the
fuel/tyre-corrected race pace from race.analyze().
"""
import numpy as np
import pandas as pd

from . import ingest, race
from .config import DERIVED, DRY

TEAM_ALIAS = {
    'Kick Sauber': 'Audi', 'Stake F1 Team Kick Sauber': 'Audi', 'Sauber': 'Audi', 'Alfa Romeo': 'Audi',
    'RB': 'Racing Bulls', 'Visa Cash App RB': 'Racing Bulls', 'AlphaTauri': 'Racing Bulls',
    'Haas': 'Haas F1 Team', 'Red Bull': 'Red Bull Racing',
}

FEATURES = ['q_rel_pct', 'q_pos', 'q_tm_gap', 'sq_rel_pct', 'fp_lr_pct', 'fp_lr_laps', 'fp_best_pct',
            'sprint_pace_pct', 'sprint_finish', 'form_pace', 'form_n', 'form_q', 'form_finish',
            'team_form_pace', 'team_form_q', 'tm_form_gap', 'form_consistency', 'rookie']

FP_WEIGHT = {'FP1': 0.6, 'FP2': 1.0, 'FP3': 0.7}


def team_norm(t):
    return TEAM_ALIAS.get(t, t)


# ------------------------------------------------------------ weekend sessions

def quali(year, rnd, code='Q'):
    res = ingest.read(year, rnd, code, 'results')
    if res is None or len(res) == 0:
        return None
    qcols = [c for c in ('Q1', 'Q2', 'Q3') if c in res]
    res = res.copy()
    res['best'] = res[qcols].min(axis=1) if qcols else np.nan
    laps = ingest.read(year, rnd, code, 'laps')
    if res.best.notna().sum() < 10 and laps is not None and len(laps):
        # timing feed sometimes lacks Q1/Q2/Q3 (e.g. Miami 2025): use each driver's best valid lap
        ok = laps[laps.LapTime.notna()]
        if 'Deleted' in ok:
            ok = ok[~ok.Deleted.astype('boolean').fillna(False).astype(bool)]
        res['best'] = res.Abbreviation.map(ok.groupby('Driver').LapTime.min())
    med = res.best.median()
    out = pd.DataFrame(dict(Driver=res.Abbreviation, Team=res.TeamName.map(team_norm),
                            rel=(res.best / med - 1) * 100, pos=res.Position))
    out['rel'] = out.rel.clip(-4, 6)
    if laps is not None and len(laps) and laps.Compound.isin(['INTERMEDIATE', 'WET']).mean() > 0.3:
        # wet qualifying gaps say little about dry race pace: keep the grid, drop the pace signal
        out['rel'] = np.nan
    return out


def _long_run_pace(laps):
    """Driver pace on FP long runs: driver + compound + tyre-age regression on steady stints."""
    l = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
             & laps.Compound.isin(DRY) & laps.TyreLife.notna()].copy()
    if 'Deleted' in l:
        l = l[~l.Deleted.astype('boolean').fillna(False).astype(bool)]
    l = l[l.TrackStatus.astype(str).str.fullmatch('1+')]
    if len(l) < 30:
        return None
    best = l.groupby('Driver').LapTime.transform('min')
    runs = []
    for _, g in l.groupby(['Driver', 'Stint']):
        med = g.LapTime.median()
        g = g[(g.LapTime < med * 1.015) & (g.LapTime > med * 0.985)]
        if len(g) >= 4 and med > best.loc[g.index].iloc[0] * 1.012:  # not a quali-sim run
            runs.append(g)
    if not runs:
        return None
    r = pd.concat(runs)
    drivers = sorted(r.Driver.unique())
    comps = list(r.Compound.value_counts().index)
    X = np.zeros((len(r), len(drivers) + 2 * len(comps) - 1))
    X[np.arange(len(r)), r.Driver.map({d: i for i, d in enumerate(drivers)}).values] = 1
    col = len(drivers)
    for c in comps[1:]:
        X[:, col] = (r.Compound == c).values
        col += 1
    for c in comps:
        X[:, col] = np.where(r.Compound == c, r.TyreLife.values, 0)
        col += 1
    beta, *_ = np.linalg.lstsq(X, r.LapTime.values, rcond=None)
    fx = pd.Series(beta[:len(drivers)], index=drivers)
    n = r.Driver.value_counts()
    ref = fx[n.reindex(fx.index).fillna(0) >= 6].median() if (n >= 6).sum() >= 5 else fx.median()
    return pd.DataFrame(dict(lr=(fx / ref - 1) * 100, n=n.reindex(fx.index).fillna(0)))


MODES = {
    'pre_weekend': set(),
    'post_fp': {'FP1', 'FP2', 'FP3'},
    'post_sprint': {'FP1', 'FP2', 'FP3', 'SQ', 'S'},
    'post_quali': {'FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q'},
}


def practice(year, rnd, allowed=None):
    lr_parts, best_parts = [], []
    for code, w in FP_WEIGHT.items():
        if allowed is not None and code not in allowed:
            continue
        laps = ingest.read(year, rnd, code, 'laps')
        if laps is None or len(laps) == 0:
            continue
        lr = _long_run_pace(laps)
        if lr is not None:
            lr_parts.append((lr, w))
        dry = laps[laps.LapTime.notna() & laps.Compound.isin(DRY)]
        if len(dry) < 0.5 * laps.LapTime.notna().sum():
            continue   # mostly wet session: no dry pace signal
        b = dry.groupby('Driver').LapTime.min()
        if len(b) >= 10:
            best_parts.append((((b / b.median()) - 1) * 100, w))
    out = pd.DataFrame()
    if lr_parts:
        idx = sorted(set().union(*[p.index for p, _ in lr_parts]))
        num = sum((p.lr.clip(-4, 6) * w * np.sqrt(p.n)).reindex(idx).fillna(0) for p, w in lr_parts)
        den = sum((w * np.sqrt(p.n)).reindex(idx).fillna(0) for p, w in lr_parts)
        out['fp_lr_pct'] = (num / den.replace(0, np.nan))
        out['fp_lr_laps'] = sum(p.n.reindex(idx).fillna(0) for p, _ in lr_parts)
    if best_parts:
        idx = sorted(set().union(*[p.index for p, _ in best_parts]))
        out = out.reindex(sorted(set(out.index) | set(idx)))
        # best lap across sessions is the most representative of ultimate pace
        out['fp_best_pct'] = pd.concat([p.reindex(idx) for p, _ in best_parts], axis=1).min(axis=1).clip(-4, 8)
    return out


def sprint(year, rnd):
    a = race.analyze(year, rnd, 'S') if (ingest.session_dir(year, rnd, 'S') / 'laps.parquet').exists() else None
    if not a:
        return None
    d = a['drivers'].set_index('Driver')
    return pd.DataFrame(dict(sprint_pace_pct=d.pace_pct.clip(-4, 6), sprint_finish=d.finish))


# ------------------------------------------------------------ history table

def history_table(analyses):
    """One row per (race, driver) with outcomes, used for form features, ratings and training targets."""
    rows = []
    for a in analyses:
        s = a['summary']
        if s['code'] != 'R':
            continue
        q = quali(s['year'], s['round'])
        qd = q.set_index('Driver') if q is not None else pd.DataFrame()
        for d in a['drivers'].to_dict('records'):
            rows.append(dict(year=s['year'], round=s['round'], location=s['location'], wet=bool(s['wet']),
                             Driver=d['Driver'], Team=team_norm(d['Team']), grid=d['grid'], finish=d['finish'],
                             dnf=bool(d['dnf']), classified=bool(d['classified']), laps_done=d['laps_done'],
                             pace_pct=d['pace_pct'], pace_laps=d['pace_laps'], consistency=d['consistency'],
                             deg_rel=d.get('deg_rel'), pit_rel=d.get('pit_rel'),
                             lap1_gain=d['lap1_gain'], passes=d['passes'], n_stops=d['n_stops'],
                             n_green_stops=d.get('n_green_stops'), strategy=d['strategy'],
                             start_compound=d['start_compound'], first_stop=d['first_stop'],
                             q_rel_pct=qd.rel.get(d['Driver'], np.nan) if len(qd) else np.nan,
                             n_laps=s['n_laps']))
    h = pd.DataFrame(rows)
    return h.sort_values(['year', 'round']).reset_index(drop=True)


def _ewm(vals, seasons, cur_season, halflife, other_season=0.5):
    if len(vals) == 0:
        return np.nan
    k = np.arange(len(vals))[::-1]
    w = 0.5 ** (k / halflife) * np.where(np.array(seasons) == cur_season, 1.0, other_season)
    v = np.array(vals, float)
    ok = np.isfinite(v)
    return float(np.average(v[ok], weights=w[ok])) if ok.any() else np.nan


def form(hist, year, rnd, entry):
    """EWMA form for each (Driver, Team) in entry using only races before (year, rnd)."""
    past = hist[(hist.year < year) | ((hist.year == year) & (hist['round'] < rnd))]
    dry = past[~past.wet]
    rows = []
    for drv, team in entry[['Driver', 'Team']].itertuples(index=False):
        dp = dry[(dry.Driver == drv) & (dry.pace_laps >= 8)]
        da = past[past.Driver == drv]
        tp = dry[(dry.Team == team) & (dry.pace_laps >= 8)].groupby(['year', 'round'])
        tpace = tp.pace_pct.mean()
        tq = past[past.Team == team].groupby(['year', 'round']).q_rel_pct.mean()
        rows.append(dict(
            Driver=drv,
            form_pace=_ewm(dp.pace_pct.clip(-4, 6).values, dp.year.values, year, 5),
            form_n=len(da),
            form_q=_ewm(da.q_rel_pct.values, da.year.values, year, 4),
            form_finish=_ewm(da.finish.values, da.year.values, year, 5),
            form_consistency=_ewm(dp.consistency.values, dp.year.values, year, 6),
            team_form_pace=_ewm(tpace.clip(-4, 6).values, [y for y, _ in tpace.index], year, 3, 0.25),
            team_form_q=_ewm(tq.values, [y for y, _ in tq.index], year, 3, 0.25),
            rookie=float(len(da) < 8),
        ))
    f = pd.DataFrame(rows)
    f = f.merge(entry[['Driver', 'Team']], on='Driver')
    tm = f.groupby('Team').form_pace.transform(lambda s: s - s.mean() if s.notna().sum() == 2 else np.nan)
    f['tm_form_gap'] = tm
    return f.drop(columns='Team')


def entry_list(year, rnd, hist):
    """Who is racing: race results if stored, else the most recent weekend session, else the last race."""
    for code in ('R', 'Q', 'SQ', 'S', 'FP3', 'FP2', 'FP1'):
        res = ingest.read(year, rnd, code, 'results')
        if res is not None and len(res) >= 18:
            return pd.DataFrame(dict(Driver=res.Abbreviation, Team=res.TeamName.map(team_norm))).drop_duplicates('Driver')
    last = hist[hist.year == year]
    if len(last) == 0:
        last = hist
    last = last[(last.year == last.year.max())]
    last = last[last['round'] == last['round'].max()]
    return last[['Driver', 'Team']].reset_index(drop=True)


def event_features(year, rnd, hist, entry=None, mode=None):
    """mode: None/'auto' uses every stored session except the race; otherwise a key of MODES."""
    allowed = None if mode in (None, 'auto') else MODES[mode]
    ok = (lambda c: allowed is None or c in allowed)
    entry = entry if entry is not None else entry_list(year, rnd, hist)
    f = entry.copy()
    q = quali(year, rnd, 'Q') if ok('Q') else None
    if q is not None:
        f = f.merge(q.rename(columns={'rel': 'q_rel_pct', 'pos': 'q_pos'})[['Driver', 'q_rel_pct', 'q_pos']],
                    on='Driver', how='left')
    sq = quali(year, rnd, 'SQ') if ok('SQ') else None
    if sq is not None:
        f = f.merge(sq.rename(columns={'rel': 'sq_rel_pct'})[['Driver', 'sq_rel_pct']], on='Driver', how='left')
    fp = practice(year, rnd, allowed)
    if len(fp):
        f = f.merge(fp, left_on='Driver', right_index=True, how='left')
    sp = sprint(year, rnd) if ok('S') else None
    if sp is not None:
        f = f.merge(sp, left_on='Driver', right_index=True, how='left')
    f = f.merge(form(hist, year, rnd, entry), on='Driver', how='left')
    for c in FEATURES:
        if c not in f:
            f[c] = np.nan
    f['q_tm_gap'] = f.groupby('Team').q_rel_pct.transform(lambda s: s - s.mean() if s.notna().sum() == 2 else np.nan)
    f['year'], f['round'] = year, rnd
    return f


def build_dataset(analyses, log=print):
    """Training table: features + outcomes for every stored race."""
    hist = history_table(analyses)
    frames = []
    for (year, rnd), g in hist.groupby(['year', 'round']):
        entry = g[['Driver', 'Team']]
        f = event_features(year, rnd, hist, entry)
        f = f.merge(g[['Driver', 'location', 'wet', 'grid', 'finish', 'dnf', 'classified', 'pace_pct',
                       'pace_laps', 'n_laps']], on='Driver')
        frames.append(f)
    ds = pd.concat(frames, ignore_index=True)
    ds.to_parquet(DERIVED / 'dataset.parquet', index=False)
    hist.to_parquet(DERIVED / 'history.parquet', index=False)
    log(f'  dataset: {len(ds)} rows from {ds.groupby(["year", "round"]).ngroups} races')
    return ds, hist
