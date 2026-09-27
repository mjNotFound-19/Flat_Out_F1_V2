"""Learn circuit behaviour from this weekend's practice (for new or changed circuits).

Tyre degradation per compound from practice long runs:
  * stints of >= MIN_LAPS consecutive green, non-pit, non-deleted laps within 107% of the stint median
  * LapTime = stint baseline + (deg_c + fuel) * tyre_age   ->  pooled slope per compound, stint intercepts
  * fuel burn is not observed in practice: the pooled race fuel coefficient (s/lap, negative) is removed,
    so deg_c = slope_c - fuel. Fuel loads in practice vary, so this is an estimate with extra noise.
  * shrunk toward the prior deg with PRIOR_STINTS pseudo-stints (evidence-weighted, not replaced)
Only sessions allowed by the forecast mode are used (post_fp / post_sprint / post_quali).
"""
import numpy as np
import pandas as pd

from . import ingest

MIN_LAPS = 6
PRIOR_STINTS = 6.0
COMPOUNDS = ('SOFT', 'MEDIUM', 'HARD')


def long_run_stints(year, rnd, sessions=('FP1', 'FP2', 'FP3')):
    out = []
    for code in sessions:
        laps = ingest.read(year, rnd, code, 'laps')
        if laps is None or not len(laps):
            continue
        L = laps[laps.LapTime.notna() & laps.PitInTime.isna() & laps.PitOutTime.isna()
                 & laps.Compound.isin(COMPOUNDS)].copy()
        if 'Deleted' in L:
            L = L[L.Deleted.fillna(False) != True]
        if 'TrackStatus' in L:
            L = L[L.TrackStatus.astype(str).isin(['1', '1.0'])]
        lt = L.LapTime
        L['t'] = lt.dt.total_seconds() if hasattr(lt, 'dt') and str(lt.dtype).startswith('timedelta') else pd.to_numeric(lt, errors='coerce')
        for (drv, st), g in L.groupby(['Driver', 'Stint']):
            g = g.sort_values('LapNumber')
            g = g[g.t < 1.07 * g.t.median()]
            if len(g) >= MIN_LAPS and g.Compound.nunique() == 1:
                out.append(dict(session=code, Driver=drv, stint=f'{code}_{drv}_{int(st)}', compound=g.Compound.iloc[0],
                                age=g.TyreLife.astype(float).values, t=g.t.values))
    return out


def degradation(stints, fuel, prior_deg):
    """Per-compound deg (s/lap per lap of tyre age), shrunk toward prior_deg. Returns (deg, n_stints)."""
    deg, n = dict(prior_deg), {}
    for c in COMPOUNDS:
        S = [s for s in stints if s['compound'] == c]
        n[c] = len(S)
        if not S:
            continue
        # pooled slope with stint fixed effects: demean age and time within each stint
        xs = np.concatenate([s['age'] - s['age'].mean() for s in S])
        ys = np.concatenate([s['t'] - s['t'].mean() for s in S])
        if (xs ** 2).sum() <= 0:
            continue
        slope = float((xs * ys).sum() / (xs ** 2).sum())
        est = slope - fuel                              # remove fuel burn-off (fuel < 0: laps get faster)
        p = prior_deg.get(c, est)
        deg[c] = float((len(S) * est + PRIOR_STINTS * p) / (len(S) + PRIOR_STINTS))
    return deg, n
