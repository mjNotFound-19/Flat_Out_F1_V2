"""Candidate race strategies and per-driver strategy probabilities.

For every legal dry compound sequence with 1-3 stops we find the stint lengths
that minimise race time under the circuit's degradation + pit-loss model. A
driver's strategy distribution then combines
  * how fast each plan is (softmax over time loss vs the best plan),
  * how many planned stops teams actually made at this venue (history),
  * which compound cars starting from that part of the grid tend to start on.
"""
from itertools import product

import numpy as np

from .config import DRY

COMP_ID = {c: i for i, c in enumerate(DRY)}
CLIFF_RATE = 0.04        # s/lap^2 beyond ~90% of a compound's observed max stint
MIN_STINT = 6


def cliff(comp_max, age):
    over = np.maximum(0.0, age - 0.9 * comp_max)
    return CLIFF_RATE * over ** 2


def stint_costs(c, n_laps):
    """cost[comp][L] = time of an L-lap stint on fresh `comp` relative to a zero-deg reference."""
    ages = np.arange(1, n_laps + 1, dtype=float)
    out = {}
    for comp in DRY:
        per_lap = c['offset'][comp] + c['deg'][comp] * ages + cliff(c['max_stint'][comp], ages)
        out[comp] = np.concatenate([[0.0], np.cumsum(per_lap)])
    return out


def candidates(c, max_stops=3):
    n = c['n_laps']
    cost = stint_costs(c, n)
    cands = []
    for stops in range(1, max_stops + 1):
        for seq in product(DRY, repeat=stops + 1):
            if len(set(seq)) < 2:
                continue
            frac = (c.get('stint_frac') or {}).get(stops)
            if frac is not None:
                # teams' real stint split for this stop count (learned), not the deg optimum
                lens = _split(frac, n)
                t = float(sum(cost[s][L] for s, L in zip(seq, lens)))
            else:
                best = _best_split(seq, cost, n)
                if best is None:
                    continue
                lens, t = best
            cands.append(dict(seq=seq, lens=lens, stops=stops, time=t + stops * c['pit_loss']))
    tmin = min(x['time'] for x in cands)
    for x in cands:
        x['delta'] = x['time'] - tmin
    # drop hopeless plans to keep sampling tables small
    cands = [x for x in cands if x['delta'] < 45]
    return sorted(cands, key=lambda x: x['time'])


def _split(frac, n):
    lens = np.maximum(np.round(np.array(frac) * n).astype(int), MIN_STINT)
    lens[-1] = n - lens[:-1].sum()
    return [int(x) for x in lens]


def _best_split(seq, cost, n):
    k = len(seq)
    lo = MIN_STINT
    if k == 2:
        L1 = np.arange(lo, n - lo + 1)
        t = cost[seq[0]][L1] + cost[seq[1]][n - L1]
        i = int(np.argmin(t))
        return [int(L1[i]), int(n - L1[i])], float(t[i])
    step = 1 if k == 3 else 2
    grid = np.arange(lo, n - lo * (k - 1) + 1, step)
    if k == 3:
        L1, L2 = np.meshgrid(grid, grid, indexing='ij')
        L3 = n - L1 - L2
        ok = L3 >= lo
        t = np.where(ok, cost[seq[0]][L1] + cost[seq[1]][L2] + cost[seq[2]][np.clip(L3, 0, n)], np.inf)
        i = np.unravel_index(np.argmin(t), t.shape)
        if not np.isfinite(t[i]):
            return None
        return [int(L1[i]), int(L2[i]), int(L3[i])], float(t[i])
    L1, L2, L3 = np.meshgrid(grid, grid, grid, indexing='ij')
    L4 = n - L1 - L2 - L3
    ok = L4 >= lo
    t = np.where(ok, cost[seq[0]][L1] + cost[seq[1]][L2] + cost[seq[2]][L3] + cost[seq[3]][np.clip(L4, 0, n)], np.inf)
    i = np.unravel_index(np.argmin(t), t.shape)
    if not np.isfinite(t[i]):
        return None
    return [int(L1[i]), int(L2[i]), int(L3[i]), int(L4[i])], float(t[i])


def driver_probs(cands, c, grids, tau=8.0, hist_weight=0.85, habit=1.5):
    """Probability of each candidate for each driver (rows) given expected grid slot."""
    delta = np.array([x['delta'] for x in cands])
    stops = np.array([x['stops'] for x in cands])
    start = [x['seq'][0] for x in cands]
    within = np.exp(-delta / tau)
    trans = c.get('transitions')
    if trans:  # teams' real compound habits (e.g. M->H) matter as much as the deg model
        within = within * np.array([np.prod([trans[a].get(b, 0.01) + 0.01 for a, b in zip(x['seq'], x['seq'][1:])])
                                    ** (habit / len(x['seq'][1:])) for x in cands])
    # model-implied stop-count distribution vs history
    model_n = {s: within[stops == s].sum() for s in (1, 2, 3)}
    tot = sum(model_n.values())
    model_n = {s: v / tot for s, v in model_n.items()}
    hist = c.get('stops_dist', {})
    hist = {int(k): v for k, v in hist.items()}
    hn = sum(hist.get(s, 0) for s in (1, 2, 3)) or 1
    target_n = {s: hist_weight * hist.get(s, 0) / hn + (1 - hist_weight) * model_n[s] for s in (1, 2, 3)}
    base = np.zeros(len(cands))
    for s in (1, 2, 3):
        m = stops == s
        if m.any() and within[m].sum() > 0:
            base[m] = target_n[s] * within[m] / within[m].sum()
    P = np.zeros((len(grids), len(cands)))
    for i, g in enumerate(grids):
        bucket = 'front' if g <= 6 else 'mid' if g <= 14 else 'back'
        sc = c['start_compound'].get(bucket) or {'MEDIUM': 0.6, 'SOFT': 0.25, 'HARD': 0.15}
        # tilt toward compounds this grid bucket usually starts on (relative to plan share)
        share = {comp: base[[s == comp for s in start]].sum() for comp in DRY}
        tilt = np.array([(sc.get(s, 0.02) + 0.02) / (share.get(s, 0) + 0.02) for s in start]) ** 0.5
        p = base * tilt
        P[i] = p / p.sum()
    return P


def table(cands):
    """Dense arrays for the simulator."""
    K = len(cands)
    lens = np.zeros((K, 4), np.int16)
    comps = np.full((K, 4), -1, np.int8)
    nst = np.zeros(K, np.int8)
    for k, x in enumerate(cands):
        lens[k, :len(x['lens'])] = x['lens']
        comps[k, :len(x['seq'])] = [COMP_ID[s] for s in x['seq']]
        nst[k] = x['stops']
    return lens, comps, nst


def label(seq):
    return '-'.join(s[0] for s in seq)
