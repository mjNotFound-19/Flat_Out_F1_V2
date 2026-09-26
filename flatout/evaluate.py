"""Score a probabilistic race prediction against what happened.

Position metrics (lower is better unless noted)
  log_loss    -mean log P(actual finishing position)          proper scoring rule
  rps         ranked probability score over positions (0..1)  rewards 'close' misses
  brier_win / brier_podium / brier_points
  spearman    rank correlation of expected vs actual position  (higher better)
  mae_pos     |expected position - actual|
  p_winner    probability we gave the actual winner             (higher better)
Strategy metrics
  stops_acc   modal predicted stop count == actual green+SC stop count
  stops_ll    -log P(actual stop count)
  seq_acc     modal executed compound sequence == actual
  first_stop_mae  |median predicted first-stop lap - actual|
Every metric is also computed for two baselines so progress is measured, not assumed:
  grid     : finish = grid slot, spread with the historical grid->finish transition matrix
  pace_only: order by predicted pace alone (no simulation)
"""
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import OUTPUT

EPS = 1e-4


def position_metrics(P, actual, dnf=None):
    """P: (D, D) predicted position probabilities, rows=drivers; actual: 0-based positions."""
    D = P.shape[0]
    P = np.clip(P, EPS, 1)
    P = P / P.sum(1, keepdims=True)
    a = np.asarray(actual, int)
    onehot = np.zeros_like(P)
    onehot[np.arange(D), a] = 1
    cdf_p, cdf_a = np.cumsum(P, 1), np.cumsum(onehot, 1)
    exp_pos = P @ np.arange(D)
    win = P[:, 0]
    pod = P[:, :3].sum(1)
    pts = P[:, :10].sum(1)
    m = dict(
        log_loss=float(-np.mean(np.log(P[np.arange(D), a]))),
        rps=float(np.mean(np.sum((cdf_p - cdf_a) ** 2, 1) / (D - 1))),
        brier_win=float(np.mean((win - (a == 0)) ** 2)),
        brier_podium=float(np.mean((pod - (a < 3)) ** 2)),
        brier_points=float(np.mean((pts - (a < 10)) ** 2)),
        spearman=float(spearmanr(exp_pos, a).correlation),
        mae_pos=float(np.mean(np.abs(exp_pos - a))),
        p_winner=float(win[a == 0].sum()),
        top3_hit=float(len(set(np.argsort(exp_pos)[:3]) & set(np.where(a < 3)[0])) / 3),
        top10_hit=float(len(set(np.argsort(exp_pos)[:10]) & set(np.where(a < 10)[0])) / 10),
        winner_correct=bool(np.argmax(win) == np.argmin(a)),
    )
    if dnf is not None:
        m['brier_dnf'] = float(np.mean((P[:, :0].sum(1) * 0 + dnf[0] - dnf[1]) ** 2))
    return m


def grid_transition(hist, cutoff):
    """P(finish | grid) from earlier races, smoothed; used for the grid baseline."""
    past = hist[(hist.year < cutoff[0]) | ((hist.year == cutoff[0]) & (hist['round'] < cutoff[1]))]
    D = 22
    T = np.ones((D, D)) * 0.2
    for g, f in zip(past.grid.clip(1, D), past.finish.clip(1, D)):
        T[int(g) - 1, int(f) - 1] += 1
    return T / T.sum(1, keepdims=True)


def grid_baseline(grid, T):
    D = len(grid)
    P = np.array([T[min(int(g), T.shape[0]) - 1, :D] for g in grid])
    P = P / P.sum(1, keepdims=True)
    return _sinkhorn(P)


def pace_baseline(pace, sd=0.6, n=4000, seed=0):
    """Positions from pace alone: sample pace ~ N(mu, sd), rank."""
    rng = np.random.default_rng(seed)
    D = len(pace)
    draws = np.asarray(pace)[None] + sd * rng.standard_normal((n, D))
    ranks = np.argsort(np.argsort(draws, 1), 1)
    P = np.zeros((D, D))
    for d in range(D):
        P[d] = np.bincount(ranks[:, d], minlength=D) / n
    return P


def _sinkhorn(P, it=50):
    """Make a driver x position matrix doubly stochastic (each position filled once)."""
    for _ in range(it):
        P = P / P.sum(0, keepdims=True)
        P = P / P.sum(1, keepdims=True)
    return P


def strategy_metrics(summary, dist_extra, actual):
    """summary: prediction table; actual: history rows for the race (Driver, n_stops, strategy, first_stop)."""
    s = summary.set_index('Driver')
    a = actual.set_index('Driver')
    a = a[a.classified]
    common = [d for d in a.index if d in s.index]
    if not common:
        return {}
    probs = s.loc[common, ['stop1', 'stop2', 'stop3p']].values
    act_n = a.loc[common, 'n_stops'].clip(1, 3).values.astype(int)
    modal = probs.argmax(1) + 1
    pa = np.clip(probs[np.arange(len(common)), act_n - 1], EPS, 1)
    seq_ok = [str(s.loc[d, 'strategy']) == str(a.loc[d, 'strategy']) for d in common]
    fs = [(s.loc[d, 'first_stop_p50'], a.loc[d, 'first_stop']) for d in common
          if pd.notna(a.loc[d, 'first_stop']) and pd.notna(s.loc[d, 'first_stop_p50'])]
    in_iqr = [(s.loc[d, 'first_stop_p25'] <= a.loc[d, 'first_stop'] <= s.loc[d, 'first_stop_p75']) for d in common
              if pd.notna(a.loc[d, 'first_stop'])]
    return dict(
        stops_acc=float(np.mean(modal == act_n)),
        stops_ll=float(-np.mean(np.log(pa))),
        seq_acc=float(np.mean(seq_ok)),
        first_stop_mae=float(np.mean([abs(p - q) for p, q in fs])) if fs else np.nan,
        first_stop_iqr_cover=float(np.mean(in_iqr)) if in_iqr else np.nan,
    )


def evaluate_event(year, rnd, summary, dist, hist, pace=None, grid=None, save=True, tag='pre_race'):
    act = hist[(hist.year == year) & (hist['round'] == rnd)].set_index('Driver')
    drivers = [d for d in dist.Driver if d in act.index]
    if len(drivers) < 10:
        return None
    Pfull = dist.set_index('Driver').loc[drivers].values
    # renormalise onto the drivers that actually started
    D = len(drivers)
    P = Pfull[:, :D] + np.concatenate([np.zeros((D, D - 1)), Pfull[:, D:].sum(1, keepdims=True)], 1)
    P = _sinkhorn(P)
    actual = act.loc[drivers, 'finish'].rank(method='first').astype(int).values - 1
    res = dict(year=year, round=rnd, event=act.location.iloc[0], n=D)
    res.update({f'model_{k}': v for k, v in position_metrics(P, actual).items()})
    T = grid_transition(hist, (year, rnd))
    g = act.loc[drivers, 'grid'].values
    res.update({f'grid_{k}': v for k, v in position_metrics(grid_baseline(g, T), actual).items()})
    if pace is not None:
        pv = pace.set_index('Driver').reindex(drivers).race_mu.fillna(2).values
        res.update({f'pace_{k}': v for k, v in position_metrics(pace_baseline(pv), actual).items()})
    res.update({f'strat_{k}': v for k, v in strategy_metrics(summary, None, act.loc[drivers].reset_index()).items()})
    # per-driver residuals: where the model was most wrong (feeds the driver evaluation report)
    exp_pos = P @ np.arange(D)
    per = pd.DataFrame(dict(Driver=drivers, exp_pos=exp_pos + 1, actual=actual + 1,
                            p_actual=P[np.arange(D), actual], grid=g,
                            dnf=act.loc[drivers, 'dnf'].values))
    per['surprise'] = -np.log(np.clip(per.p_actual, EPS, 1))
    per['error'] = per.actual - per.exp_pos
    if save:
        d = OUTPUT / str(year) / f'R{rnd:02d}'
        d.mkdir(parents=True, exist_ok=True)
        (d / f'{tag}_evaluation.json').write_text(json.dumps(res, indent=1, default=float))
        per.to_csv(d / f'{tag}_driver_errors.csv', index=False)
    return res, per


def calibration_table(rows, kind='win', bins=(0, .02, .05, .1, .2, .35, .5, .7, 1.0001)):
    """rows: DataFrame with predicted prob column `p` and outcome `y`."""
    b = pd.cut(rows.p, bins, right=False)
    t = rows.groupby(b, observed=True).agg(n=('y', 'size'), predicted=('p', 'mean'), observed=('y', 'mean'))
    t.index = t.index.astype(str)
    return t.reset_index().rename(columns={'p': 'bin'})
