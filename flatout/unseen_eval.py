"""Unseen-venue evaluation: how well does the model forecast a race at a circuit it has never seen?

Every race of a nested champion experiment is re-forecast with its circuit's own history removed
(force_unseen), using that race's walk-forward simulator params from the experiment's checkpoint. Two ways of
handling an unseen circuit are compared on the same seeds, and the known-circuit forecast is the reference:
    sd_mult   : pooled circuit priors + driver pace sd x1.25   (champion's treatment of a new venue)
    param_unc : pooled circuit priors + track-level parameters drawn per batch from the leave-one-circuit-out
                error of the pooled prior (circuits.unseen_uncertainty)
Applies to any future new or returning venue, not one specific race.

    python -m flatout unseen --experiment artifacts/experiments/<champion nested run>
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import backtest, evaluate, nested, pipeline, sim

VARIANTS = {'sd_mult': dict(force_unseen=True, new_venue_mode='sd_mult'),
            'param_unc': dict(force_unseen=True, new_venue_mode='param_unc')}


def run(exp_dir, eval_sims=300_000, workers=None, log=print):
    exp_dir = Path(exp_dir)
    cfg = json.loads((exp_dir / 'config.json').read_text())
    analyses, hist = pipeline.load_state(log=lambda *a: None)
    ds = backtest.load_dataset(analyses, hist, lambda *a: None)
    out_dir = exp_dir / 'unseen'
    out_dir.mkdir(exist_ok=True)
    rows = []
    t0 = time.time()
    with sim.pool(sim.default_workers(workers)) as ex:
        for k, f in enumerate(sorted((exp_dir / 'races').glob('*.json'))):
            ck = json.loads(f.read_text())
            for base_row in ck['rows']:
                y, r, mode = int(base_row['year']), int(base_row['round']), base_row['mode']
                rows.append(dict(year=y, round=r, mode=mode, variant='known', rps=base_row['model_rps'],
                                 log_loss=base_row['model_log_loss'], drivers=base_row.get('drivers')))
                for name, over in VARIANTS.items():
                    params = dict(ck['params'], **over)
                    (_, _, spec, ctx), = backtest.prepare_specs([(y, r)], analyses, hist, ds, params, mode,
                                                                log=lambda *a: None)
                    agg = sim.simulate(spec, eval_sims, seed=70_000 + 101 * k, ex=ex)
                    summ, dist, _ = pipeline.summarise(agg, ctx)
                    res = evaluate.evaluate_event(y, r, summ, dist, hist, pace=ctx['pred'], save=False)
                    if res:
                        m, _ = res
                        rows.append(dict(year=y, round=r, mode=mode, variant=name, rps=m['model_rps'],
                                         log_loss=m['model_log_loss'], drivers=nested._driver_probs(y, r, summ, hist)))
            log(f'  {f.stem} done ({time.time() - t0:.0f}s)')
    df = pd.DataFrame(rows)
    report = {}
    for mode, g in df.groupby('mode'):
        piv = g.pivot_table(index=['year', 'round'], columns='variant', values='rps')
        rep = {v: dict(rps=float(piv[v].mean())) for v in piv}
        rep['param_unc_minus_sd_mult'] = nested.paired(piv['param_unc'] - piv['sd_mult'])
        rep['unseen_cost_sd_mult'] = nested.paired(piv['sd_mult'] - piv['known'])
        rep['unseen_cost_param_unc'] = nested.paired(piv['param_unc'] - piv['known'])
        for v in ('known', 'sd_mult', 'param_unc'):
            drv = [d for ds_ in g[g.variant == v].drivers.dropna() for d in ds_]
            if drv:
                rep[v].update({f'ece_{k}': nested.calibration_table(drv, k)['ece'] for k in ('win', 'podium', 'points')})
        report[mode] = rep
    df.drop(columns='drivers').to_csv(out_dir / 'per_race.csv', index=False)
    (out_dir / 'summary.json').write_text(json.dumps(report, indent=1, default=float))
    for mode, rep in report.items():
        log(f"\n  {mode}: RPS known {rep['known']['rps']:.4f} | unseen sd_mult {rep['sd_mult']['rps']:.4f} | "
            f"unseen param_unc {rep['param_unc']['rps']:.4f}")
        p = rep['param_unc_minus_sd_mult']
        log(f"    param_unc - sd_mult: {p['mean']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}] better in {p['share_better']:.0%}")
        for v in ('known', 'sd_mult', 'param_unc'):
            log(f"    {v:<9s} ECE win {rep[v].get('ece_win', np.nan):.3f} podium {rep[v].get('ece_podium', np.nan):.3f} "
                f"points {rep[v].get('ece_points', np.nan):.3f}")
    return report
