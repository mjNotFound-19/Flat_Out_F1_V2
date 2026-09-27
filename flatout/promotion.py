"""Apply the pre-specified promotion criteria (docs/PROMOTION.md) to a champion/challenger pair of nested
experiments. Mechanical by design: the thresholds were written before any challenger was evaluated.

    python -m flatout promote --champion artifacts/experiments/<champ> --challenger artifacts/experiments/<chal>
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import nested

RULES = dict(primary_max=-0.0020, block_max=0.0040, logloss_max=0.010, brier_rel_max=0.02, ece_max=0.010,
             mc_se_bound=0.0003)


def _rows(exp):
    rows = []
    for f in sorted((Path(exp) / 'races').glob('*.json')):
        rows += json.loads(f.read_text())['rows']
    return pd.DataFrame(rows)


def _ece(df, key):
    drv = [d for ds in df.get('drivers', pd.Series(dtype=object)).dropna() for d in ds]
    return nested.calibration_table(drv, key)['ece'] if drv else None


def evaluate(champion, challenger):
    A, B = _rows(champion), _rows(challenger)
    keys = ['year', 'round', 'mode']
    m = A.merge(B, on=keys, suffixes=('_a', '_b'))
    out = dict(champion=str(champion), challenger=str(challenger), rules=RULES, n_pairs=int(len(m)), checks={})
    d = m.model_rps_b - m.model_rps_a
    prim = nested.paired(d)
    out['checks']['1_primary_rps'] = dict(value=prim, pass_=bool(prim['mean'] <= RULES['primary_max'] and prim['hi'] < 0))
    blocks = {f'{y}_{md}': float(g.model_rps_b.mean() - g.model_rps_a.mean()) for (y, md), g in m.groupby(['year', 'mode'])}
    out['checks']['2_no_regime_collapse'] = dict(value=blocks, pass_=all(v <= RULES['block_max'] for v in blocks.values()))
    ll = float((m.model_log_loss_b - m.model_log_loss_a).mean())
    briers = {k: float(m[f'model_{k}_b'].mean() / m[f'model_{k}_a'].mean() - 1)
              for k in ('brier_win', 'brier_podium', 'brier_points')}
    out['checks']['3_probabilities'] = dict(value=dict(log_loss_diff=ll, brier_rel=briers),
                                            pass_=bool(ll <= RULES['logloss_max'] and all(v <= RULES['brier_rel_max'] for v in briers.values())))
    ece = {k: (_ece(A, k), _ece(B, k)) for k in ('win', 'podium')}
    ok4 = all(a is not None and b is not None and b - a <= RULES['ece_max'] for a, b in ece.values())
    out['checks']['4_calibration'] = dict(value={k: dict(champion=a, challenger=b) for k, (a, b) in ece.items()},
                                          pass_=bool(ok4), note=None if ok4 else 'missing per-driver data or worse')
    # Monte Carlo: both arms share seeds per race (common random numbers) at 300k sims; per-race RPS agrees
    # with a 4M run to ~1e-4, so |mean paired diff| > 5 x 3e-4 / sqrt(n) is far outside simulation noise.
    mc = 5 * RULES['mc_se_bound'] / np.sqrt(max(1, len(m)))
    out['checks']['5_not_simulation_noise'] = dict(value=dict(abs_mean=abs(prim['mean']), threshold=mc),
                                                   pass_=bool(abs(prim['mean']) > mc))
    out['checks']['6_no_leakage'] = dict(pass_=None, note='run tests/test_leakage.py; challenger params must be pre-chosen or tuned inside the nested protocol')
    out['promote'] = all(c['pass_'] for k, c in out['checks'].items() if c['pass_'] is not None)
    return out
