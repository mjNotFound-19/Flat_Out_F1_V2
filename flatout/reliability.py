"""Retirement hazard model (challenger to pipeline.reliability): gamma-Poisson empirical Bayes with exposure.

Retirements are counted per lap driven (exposure), not per race, and estimated in three shrunk layers:
    season rate   r_s = (Y_s + A_S) / (E_s + A_S / r_all)                  -> r_all while a season is young
    team factor   f_t = (Y_t + A_T) / (E_t * r_s + A_T)                     team within the current season
                        (+ previous season's team record at weight PREV_W)
    circuit factor f_c = (Y_c + A_C) / (sum_i E_ci * r_season(i) + A_C)     pooled over seasons; unseen -> 1
    hazard per lap  lambda = r_s * f_t * f_c ;  P(retire in n laps) = 1 - exp(-lambda * n)
Y = retirements, E = laps driven (a retired car's laps count up to its retirement). Rows from the current
season are down-weighted by recency (half-life HALF_LIFE races). Only races before the forecast race.

Why: the champion shrinks each team toward a multi-season per-race rate; 2026 cars retire about twice as
often per lap as 2024-25 cars, so the champion under-predicts retirements in 2026 (docs/EXPERIMENTS.md E4).

Limits: the 2026 results feed labels every stop 'Retired', so mechanical failures and incidents are not
separated; this is their combined rate. DSQ / DNS rows are excluded.
Prior strengths (pseudo-retirements) were fixed before the nested evaluation, not tuned:
"""
import numpy as np
import pandas as pd

A_S, A_T, A_C = 3.0, 3.0, 4.0     # prior strength in pseudo-retirements (season, team, circuit)
PREV_W = 0.3                       # weight of a team's previous-season record
HALF_LIFE = 12                     # races, recency within the current season


def _rows(hist, year, rnd):
    past = hist[(hist.year < year) | ((hist.year == year) & (hist['round'] < rnd))]
    if 'outcome' in past:
        past = past[~past.outcome.isin(['dsq', 'dns'])]
    laps = past.laps_done.where(past.laps_done > 0, past.n_laps).astype(float).clip(lower=1)
    return past.assign(E=laps, Y=past.dnf.astype(float))


def fit(hist, year, rnd):
    d = _rows(hist, year, rnd)
    if len(d) < 100:
        return None
    idx = d.groupby(['year', 'round']).ngroup()
    cur = d.year == year
    w = np.where(cur, 0.5 ** ((idx[cur].max() - idx) / HALF_LIFE) if cur.any() else 1.0, 1.0)
    d = d.assign(wY=d.Y * w, wE=d.E * w)
    r_all = d.Y.sum() / d.E.sum()
    by_season = d.groupby('year')[['wY', 'wE']].sum()
    r_season = {int(y): float((g.wY + A_S) / (g.wE + A_S / r_all)) for y, g in by_season.iterrows()}
    r_cur = r_season.get(year, r_all)
    # team factor: this season (weight 1) + previous season (PREV_W), relative to each season's rate
    team = {}
    for t in d.Team.unique():
        num, den = A_T, A_T
        for y, wt in ((year, 1.0), (year - 1, PREV_W)):
            g = d[(d.Team == t) & (d.year == y)]
            if len(g):
                num += wt * g.wY.sum()
                den += wt * g.wE.sum() * r_season.get(y, r_all)
        team[t] = num / den
    # circuit factor pooled over seasons, relative to each row's season rate
    exp_c = d.assign(ex=d.wE * d.year.map(lambda y: r_season.get(int(y), r_all))).groupby('location')[['wY', 'ex']].sum()
    circuit = {c: float((g.wY + A_C) / (g.ex + A_C)) for c, g in exp_c.iterrows()}
    return dict(year=year, r_all=float(r_all), r_season=r_season, r_cur=float(r_cur), team=team, circuit=circuit)


def team_probs(fitted, year, teams, circuit, n_laps):
    """P(retire before the flag) per team for a race of n_laps at `circuit` (unseen circuit -> factor 1)."""
    if fitted is None:
        return None
    fc = fitted['circuit'].get(circuit, 1.0)
    return {t: float(1 - np.exp(-fitted['r_cur'] * fitted['team'].get(t, 1.0) * fc * n_laps)) for t in teams}
