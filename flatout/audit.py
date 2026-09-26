"""Data and training audit: every stored GP, every check, one table of flags.

python -m flatout audit
"""
import json

import numpy as np
import pandas as pd

from . import features, ingest, race
from .config import OUTPUT


def race_checks(a):
    s, d = a['summary'], a['drivers']
    flags = []
    n = len(d)
    if n < 18:
        flags.append(f'only {n} drivers in results')
    if not s.get('fit_ok', True):
        flags.append('pace regression rejected' + (' (wet)' if s['wet'] else ''))
    paced = int((d.pace_laps >= 10).sum())
    if paced < max(10, n - 8):
        flags.append(f'pace for only {paced}/{n} drivers')
    if s['base_lap'] and not 60 < s['base_lap'] < 140:
        flags.append(f"base lap {s['base_lap']:.1f}s")
    for comp, v in (s.get('deg') or {}).items():
        if not -0.02 < v < 0.35:
            flags.append(f'deg {comp} {v:.3f}')
    if s.get('pit_loss') is None or not np.isfinite(s.get('pit_loss') or np.nan):
        flags.append(f"pit loss unmeasured ({s['n_pit_samples']} samples)")
    elif not 12 < s['pit_loss'] < 35:
        flags.append(f"pit loss {s['pit_loss']:.1f}s")
    g = d.grid.values
    if len(set(g)) < len(g) - 2:
        flags.append('duplicate grid slots')
    fin = d[d.classified.astype(bool)]
    if len(fin) and fin.n_stops.max() >= 5 and not s['red']:
        flags.append(f'max {int(fin.n_stops.max())} stops without red flag')
    if len(fin) and (fin.n_stops == 0).mean() > 0.3 and not s['wet']:
        flags.append(f'{(fin.n_stops == 0).mean():.0%} finishers with 0 stops')
    if s['dnfs'] > 8:
        flags.append(f"{s['dnfs']} DNFs")
    return flags


def run(log=print):
    analyses = race.analyze_all(log=log)
    hist = features.history_table(analyses)
    ds = features.build_dataset(analyses, log=lambda *a: None)[0]
    rows = []
    for a in analyses:
        s = a['summary']
        if s['code'] != 'R':
            continue
        y, r = s['year'], s['round']
        sess = ingest.available_sessions(y, r)
        flags = race_checks(a)
        if 'Q' not in sess:
            flags.append('no qualifying stored')
        if not any(c.startswith('FP') for c in sess):
            flags.append('no practice stored')
        f = ds[(ds.year == y) & (ds['round'] == r)]
        if len(f):
            if f.q_rel_pct.isna().mean() > 0.2:
                flags.append(f'quali feature missing {f.q_rel_pct.isna().mean():.0%}')
            if f.fp_best_pct.isna().mean() > 0.3 and any(c.startswith('FP') for c in sess):
                flags.append(f'FP feature missing {f.fp_best_pct.isna().mean():.0%}')
            # quali pace should track race pace; a big mismatch means a session mix-up or wet quali
            ok = f[['q_rel_pct', 'pace_pct']].dropna()
            if len(ok) >= 10:
                rho = ok.corr(method='spearman').iloc[0, 1]
                if rho < 0.5:
                    flags.append(f'quali vs race pace rho {rho:.2f}')
        rows.append(dict(year=y, round=r, event=s['event'], laps=s['n_laps'], wet=s['wet'], sc=s['sc'], vsc=s['vsc'],
                         red=s['red'], dnfs=s['dnfs'], sessions=' '.join(sess), flags='; '.join(flags)))
    out = pd.DataFrame(rows).sort_values(['year', 'round'])
    out.to_csv(OUTPUT / 'audit.csv', index=False)
    flagged = out[out['flags'] != '']
    log(f'  audited {len(out)} races, {len(flagged)} with flags -> {OUTPUT / "audit.csv"}')
    return out, hist, ds
