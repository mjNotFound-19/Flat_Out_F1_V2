"""Car-adjusted driver ratings.

A driver's raw results mostly measure the car. Each metric here is fitted as

    metric[race, driver] = car[race, team] + skill[driver] + noise

with one free car effect per (race, team) and ridge-shrunk driver skills, so a
driver is compared with their team-mate every weekend, and drivers who changed
teams tie the whole grid onto one scale. Recent races weigh more (half-life in
races), and every rating carries a standard error so small samples are visible.

Metrics (sign convention: positive = better in the output)
  race_pace    fuel/tyre-corrected race pace, s/lap vs an average driver
  quali_pace   qualifying pace, s/lap vs average (from % gap, on a 90 s lap)
  starts       lap-1 places gained beyond what the grid slot usually yields
  execution    finishing places gained vs where the driver's own pace ranked them
  consistency  clean-lap scatter, s (lower raw scatter = positive rating)
  tyre_mgmt    extra lap-time fade per lap of tyre age vs the field, ms/lap (less fade = positive)
"""
import numpy as np
import pandas as pd

LAP = 90.0   # s, to express % gaps as s/lap


def _ridge_effects(df, target, lam, halflife, now_idx, sign=1.0, scale=1.0):
    d = df[df[target].notna()].copy()
    if len(d) < 20:
        return pd.DataFrame(columns=['Driver', 'value', 'se', 'n'])
    d['car'] = d.year.astype(str) + '_' + d['round'].astype(str) + '_' + d.Team
    cars = {c: i for i, c in enumerate(d.car.unique())}
    drivers = {x: i for i, x in enumerate(sorted(d.Driver.unique()))}
    nc, nd = len(cars), len(drivers)
    X = np.zeros((len(d), nc + nd))
    X[np.arange(len(d)), d.car.map(cars).values] = 1
    X[np.arange(len(d)), nc + d.Driver.map(drivers).values] = 1
    w = 0.5 ** ((now_idx - d.event_idx.values) / halflife)
    y = d[target].values.astype(float)
    pen = np.r_[np.full(nc, 1e-4), np.full(nd, lam)]
    A = X.T @ (X * w[:, None]) + np.diag(pen)
    beta = np.linalg.solve(A, X.T @ (w * y))
    resid = y - X @ beta
    sigma2 = np.sum(w * resid ** 2) / max(np.sum(w) - nd, 1)
    cov = np.linalg.inv(A) * sigma2
    se = np.sqrt(np.diag(cov)[nc:])
    skill = beta[nc:]
    skill = skill - np.average(skill, weights=d.Driver.value_counts().reindex(drivers).values)
    n = d.Driver.value_counts()
    return pd.DataFrame(dict(Driver=list(drivers), value=sign * skill * scale, se=se * scale,
                             n=[int(n[x]) for x in drivers]))


def driver_ratings(hist, year=None, halflife=16, lam=2.0, min_races=4):
    """hist: features.history_table(). Ratings as of the latest race (or end of `year`)."""
    h = hist.copy()
    if year:
        h = h[h.year <= year]
    ev = h[['year', 'round']].drop_duplicates().sort_values(['year', 'round']).reset_index(drop=True)
    ev['event_idx'] = np.arange(len(ev))
    h = h.merge(ev, on=['year', 'round'])
    now = ev.event_idx.max()
    dry = h[~h.wet.astype(bool)]

    # derived per-race targets
    race = dry[(dry.pace_laps >= 10)].copy()
    race['pace'] = race.pace_pct.clip(-4, 6)
    q = h[h.q_rel_pct.notna()].copy()
    q['q'] = q.q_rel_pct.clip(-3, 5)
    st = h[h.lap1_gain.notna() & (h.laps_done.fillna(2) > 1)].copy()
    exp_gain = st.groupby('grid').lap1_gain.mean()
    st['start_gain'] = (st.lap1_gain - st.grid.map(exp_gain)).clip(-8, 8)
    ex = race[race.classified.astype(bool)].copy()
    ex['pace_rank'] = ex.groupby(['year', 'round']).pace.rank()
    ex['fin_rank'] = ex.groupby(['year', 'round']).finish.rank()
    ex['exec'] = (ex.pace_rank - ex.fin_rank).clip(-8, 8)
    co = race[race.consistency.notna()].copy()
    co['cons'] = co.consistency.clip(0, 3)
    ty = race[race.deg_rel.notna()].copy() if 'deg_rel' in race else race.iloc[:0].assign(deg_rel=[])
    ty['deg'] = ty.deg_rel.astype(float).clip(-0.1, 0.1)

    parts = {
        'race_pace': _ridge_effects(race, 'pace', lam, halflife, now, sign=-1, scale=LAP / 100),
        'quali_pace': _ridge_effects(q, 'q', lam, halflife, now, sign=-1, scale=LAP / 100),
        'starts': _ridge_effects(st, 'start_gain', lam * 2, halflife, now),
        'execution': _ridge_effects(ex, 'exec', lam * 2, halflife, now),
        'consistency': _ridge_effects(co, 'cons', lam, halflife, now, sign=-1),
        'tyre_mgmt': _ridge_effects(ty, 'deg', lam * 2, halflife, now, sign=-1, scale=1000),
    }
    out = None
    for k, p in parts.items():
        p = p.rename(columns={'value': k, 'se': f'{k}_se', 'n': f'{k}_n'})
        out = p if out is None else out.merge(p, on='Driver', how='outer')
    last = h.sort_values(['year', 'round']).groupby('Driver').tail(1).set_index('Driver')
    out['team'] = out.Driver.map(last.Team)
    out['last_race'] = out.Driver.map(last.year.astype(str) + ' R' + last['round'].astype(str))
    out['races'] = out.Driver.map(h.Driver.value_counts())
    dnf = h.groupby('Driver').dnf.mean()
    out['dnf_rate'] = out.Driver.map(dnf)
    out = out[out.races >= min_races]
    # composite: z-scores weighted by how much each skill moves results
    wts = dict(race_pace=0.4, quali_pace=0.25, execution=0.12, starts=0.08, tyre_mgmt=0.1, consistency=0.05)
    z = sum(w * (out[k] - out[k].mean()) / (out[k].std() or 1) for k, w in wts.items())
    out['overall'] = z
    return out.sort_values('overall', ascending=False).reset_index(drop=True)


def print_ratings(r, current_only=True):
    view = r.copy()
    if current_only:
        latest = view.last_race.value_counts().index[0]
        view = view[view.last_race == latest]
    cols = ['Driver', 'team', 'races', 'overall', 'race_pace', 'race_pace_se', 'quali_pace', 'quali_pace_se',
            'execution', 'starts', 'tyre_mgmt', 'consistency', 'dnf_rate']
    print('\n  Car-adjusted driver ratings (positive = better; pace in s/lap vs average driver)')
    print(view[cols].round(3).to_string(index=False))


# ---------------------------------------------------------------- cars / constructors

def _car_fit(d, target, lam=2.0):
    """Same car + driver model, returning per-race car effects centred on the median car."""
    d = d[d[target].notna()].copy()
    d['car'] = d.year.astype(str) + '_' + d['round'].astype(str) + '_' + d.Team
    cars = {c: i for i, c in enumerate(d.car.unique())}
    drivers = {x: i for i, x in enumerate(sorted(d.Driver.unique()))}
    nc, nd = len(cars), len(drivers)
    X = np.zeros((len(d), nc + nd))
    X[np.arange(len(d)), d.car.map(cars).values] = 1
    X[np.arange(len(d)), nc + d.Driver.map(drivers).values] = 1
    A = X.T @ X + np.diag(np.r_[np.full(nc, 1e-4), np.full(nd, lam)])
    beta = np.linalg.solve(A, X.T @ d[target].values.astype(float))
    key = d.drop_duplicates('car').set_index('car')[['year', 'round', 'Team']]
    out = key.assign(effect=[beta[cars[c]] for c in key.index]).reset_index(drop=True)
    out['effect'] = out.effect - out.groupby(['year', 'round']).effect.transform('median')
    return out


def car_performance(hist, year):
    """Per race and team: car race pace and quali pace (% vs median car, lower = faster), and the
    finishing slot the car should deliver if every driver were equal."""
    h = hist[hist.year >= year - 1]
    dry = h[~h.wet.astype(bool) & (h.pace_laps >= 8)].assign(p=lambda x: x.pace_pct.clip(-4, 6))
    race = _car_fit(dry, 'p').rename(columns={'effect': 'car_pace'})
    q = _car_fit(h.assign(q=h.q_rel_pct.clip(-3, 5)), 'q').rename(columns={'effect': 'car_quali'})
    cp = race.merge(q, on=['year', 'round', 'Team'], how='outer')
    cp = cp[cp.year == year].copy()
    # the k-th fastest car owns slots 2k-1 and 2k, so it is expected to finish around 2k - 0.5
    cp['car_rank'] = cp.groupby('round').car_pace.rank(method='min')
    cp['car_q_rank'] = cp.groupby('round').car_quali.rank(method='min')
    cp['exp_finish'] = 2 * cp.car_rank - 0.5
    cp['exp_grid'] = 2 * cp.car_q_rank - 0.5
    return cp


def _expected_slots(h, cp):
    """Fair, zero-sum expectations per race.

    Race: among classified finishers only, a driver 'should' finish behind every finisher in a faster
    car and share their own car's slots with a finishing team-mate. Retirements ahead therefore do
    not flatter backmarkers. Grid: the same with qualifying car pace over all starters.
    Positive delta = the driver beat what the car was worth."""
    h = h.copy()
    h['exp_finish'] = np.nan
    h['exp_grid'] = np.nan
    for rnd, g in h.groupby('round'):
        fin = g[g.classified.astype(bool)]
        for idx, r in fin.iterrows():
            faster = (fin.car_pace < r.car_pace).sum()
            mates = (fin.Team == r.Team).sum()
            h.loc[idx, 'exp_finish'] = faster + (mates + 1) / 2
        h.loc[fin.index, 'act_finish'] = fin.finish.rank(method='first')
        for idx, r in g.iterrows():
            faster = (g.car_quali < r.car_quali).sum()
            mates = (g.Team == r.Team).sum()
            h.loc[idx, 'exp_grid'] = faster + (mates + 1) / 2
        h.loc[g.index, 'act_grid'] = g.grid.rank(method='first')
    h['race_delta'] = h.exp_finish - h.act_finish
    h['quali_delta'] = h.exp_grid - h.act_grid
    return h


def driver_vs_car(hist, year):
    """Where each driver finished vs where their car was expected to finish, race by race."""
    cp = car_performance(hist, year)
    h = hist[hist.year == year].merge(cp[['round', 'Team', 'car_pace', 'car_quali', 'car_rank', 'exp_finish']],
                                      on=['round', 'Team'], how='left')
    h = _expected_slots(h[h.exp_finish.notna()].copy(), cp)
    rows = []
    for (drv, team), g in h.groupby(['Driver', 'Team']):
        fin = g[g.classified.astype(bool)]
        rows.append(dict(
            Driver=drv, Team=team, races=len(g), classified=len(fin), dnfs=int(g.dnf.astype(bool).sum()),
            car_expected=float(fin.exp_finish.mean()) if len(fin) else np.nan,
            actual=float(fin.finish.mean()) if len(fin) else np.nan,
            race_delta=float(fin.race_delta.mean()) if len(fin) else np.nan,
            grid_expected=float(g.exp_grid.mean()), grid_actual=float(g.act_grid.mean()),
            quali_delta=float(g.quali_delta.mean()),
            beat_car_pct=float((fin.race_delta > 0.5).mean()) if len(fin) else np.nan,
            per_race=[dict(round=int(r['round']), exp=float(r.exp_finish), actual=int(r.finish), dnf=bool(r.dnf),
                           grid=int(r.grid), exp_grid=float(r.exp_grid)) for _, r in g.sort_values('round').iterrows()]))
    return pd.DataFrame(rows).sort_values('race_delta', ascending=False, na_position='last').reset_index(drop=True)


def constructor_ratings(hist, year, halflife=6):
    """Car race pace, quali pace, reliability, pit work, race execution and development trend per team."""
    cp = car_performance(hist, year)
    h = hist[hist.year == year]
    dvc = _expected_slots(h.merge(cp[['round', 'Team', 'car_pace', 'car_quali', 'exp_finish']], on=['round', 'Team'])
                          .query('exp_finish == exp_finish'), cp)
    last = cp['round'].max()
    rows = []
    for team, g in cp.groupby('Team'):
        g = g.sort_values('round')
        hr = h[h.Team == team]
        gp, gq = g[g.car_pace.notna()], g[g.car_quali.notna()]
        wp = 0.5 ** ((last - gp['round']) / halflife)
        wq = 0.5 ** ((last - gq['round']) / halflife)
        fin = hr[hr.classified.astype(bool)]
        rows.append(dict(
            Team=team,
            race_pace=float(np.average(gp.car_pace, weights=wp)) if len(gp) else np.nan,   # % vs median car, recent-weighted
            quali_pace=float(np.average(gq.car_quali, weights=wq)) if len(gq) else np.nan,
            season_pace=float(gp.car_pace.mean()) if len(gp) else np.nan,
            development=float(np.polyfit(gp['round'], gp.car_pace, 1)[0]) if len(gp) >= 4 else np.nan,  # %/race, < 0 improving
            reliability=float(1 - hr.dnf.astype(bool).mean()) if len(hr) else np.nan,
            pit_ops=float(hr.pit_rel.mean()) if 'pit_rel' in hr and hr.pit_rel.notna().any() else np.nan,  # s vs race median
            execution=float(dvc.loc[dvc.Team == team, 'race_delta'].mean()),  # places vs car expectation (zero-sum)
            wins=int((hr.finish == 1).sum()), podiums=int((hr.finish <= 3).sum()),
            pace_trend=[dict(round=int(r['round']), pace=None if pd.isna(r.car_pace) else float(r.car_pace),
                             quali=None if pd.isna(r.car_quali) else float(r.car_quali)) for _, r in g.iterrows()],
        ))
    t = pd.DataFrame(rows)

    def score(col, better_low):
        """0-100 scale (50 = average team), higher is better on every axis."""
        v = t[col].astype(float)
        if v.notna().sum() < 2:
            return pd.Series(50.0, index=t.index)
        z = (v - v.mean()) / (v.std() or 1)
        return (50 + 18 * (-z if better_low else z)).clip(0, 100).fillna(50).round(1)
    t['s_race'] = score('race_pace', True)
    t['s_quali'] = score('quali_pace', True)
    t['s_reliability'] = score('reliability', False)
    t['s_pit'] = score('pit_ops', True)
    t['s_execution'] = score('execution', False)
    t['s_development'] = score('development', True)
    # car pace dominates; execution is capped by the same ceiling as driver-vs-car (the fastest car can
    # only lose places), so it and development are minor terms
    t['overall'] = (0.5 * t.s_race + 0.2 * t.s_quali + 0.15 * t.s_reliability + 0.07 * t.s_development
                    + 0.05 * t.s_execution + 0.03 * t.s_pit).round(1)
    return t.sort_values('overall', ascending=False).reset_index(drop=True)
