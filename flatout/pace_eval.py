"""Walk-forward evaluation of pace-model configurations (no race simulation, so it is cheap).

For each evaluated race: fit the pace model on races strictly before it with a given configuration,
predict every driver's race pace (% vs field median) with that forecast mode's information, and score
against the realised clean-lap pace (drivers with >= 8 clean laps, dry races only).

`nested_select` is the honest way to claim a *tuned* configuration helps: before each race it picks the
configuration with the lowest error on the `inner_n` preceding races (each itself walk-forward), and
only that choice is scored on the race.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .model import WEEKEND_FP, WEEKEND_Q, PaceModels

MASKS = {'post_quali': (), 'pre_weekend': WEEKEND_Q + WEEKEND_FP}

SEASON_WEIGHT_CANDIDATES = {
    'current_35_60_100': {2024: 0.35, 2025: 0.6, 2026: 1.0},
    'equal': {2024: 1.0, 2025: 1.0, 2026: 1.0},
    'mild_60_80_100': {2024: 0.6, 2025: 0.8, 2026: 1.0},
    'steep_10_30_100': {2024: 0.1, 2025: 0.3, 2026: 1.0},
    'newest_05_15_100': {2024: 0.05, 2025: 0.15, 2026: 1.0},
}


def events_of(ds):
    ev = ds[['year', 'round']].drop_duplicates().sort_values(['year', 'round'])
    return [tuple(int(v) for v in x) for x in ev.values]


def score_race(ds, event, config, mode):
    """Pace RMSE / MAE / Spearman for one race with a walk-forward fit. None if not scoreable."""
    y, r = event
    test = ds[(ds.year == y) & (ds['round'] == r)]
    test = test[test.pace_pct.notna() & (test.pace_laps >= 8)]
    if len(test) < 10 or bool(test.wet.any()):
        return None
    m = PaceModels(season_weight=config).fit(ds, cutoff=event, calibrate=False)
    X = test.assign(**{c: np.nan for c in MASKS[mode] if c in test})
    pred = m.predict(X).race_mu.values
    act = test.pace_pct.clip(-4, 6).values
    err = pred - act
    return dict(year=y, round=r, n=len(test), rmse=float(np.sqrt(np.mean(err ** 2))),
                mae=float(np.mean(np.abs(err))), spearman=float(spearmanr(pred, act).correlation))


def grid(ds, events, configs, mode):
    """Score every configuration on every event -> long DataFrame (config, year, round, metrics)."""
    rows = []
    for name, cfg in configs.items():
        for ev in events:
            s = score_race(ds, ev, cfg, mode)
            if s:
                rows.append(dict(config=name, mode=mode, **s))
    return pd.DataFrame(rows)


def nested_select(scores, outer_events, inner_n=8, metric='rmse', default='current_35_60_100'):
    """Per outer race, choose the config with the lowest mean `metric` over the `inner_n` preceding scored
    races (all configs are scored walk-forward already, so no future information enters the choice)."""
    piv = scores.pivot_table(index=['year', 'round'], columns='config', values=metric)
    order = list(piv.index)
    out = []
    for ev in outer_events:
        if ev not in piv.index:
            continue
        prior = [e for e in order if e < ev][-inner_n:]
        choice = piv.loc[prior].mean().idxmin() if len(prior) >= 3 else default
        out.append(dict(year=ev[0], round=ev[1], chosen=choice, selected=float(piv.loc[ev, choice]),
                        default=float(piv.loc[ev, default])))
    return pd.DataFrame(out)
