"""Nested walk-forward evaluation: the honest out-of-sample benchmark.

For every evaluated race k (in date order):
  inner  : simulator params are tuned (backtest.search_params) on the `calib_last_n` races strictly
           before k, each of those simulated with pace models fitted strictly before *that* race.
  outer  : race k is then forecast with pace models fitted strictly before k and the params from
           the inner step, and scored against the result and the baselines.
Nothing about race k (or later races) touches its forecast. The inner search warm-starts from the
previous outer race's params (which themselves only used earlier races) and the very first one
starts from sim.DEFAULTS.

Each outer race is checkpointed under a hash of the experiment configuration and data snapshot, so
an interrupted run resumes, and a changed configuration never resumes into an old experiment.

    python -m flatout nested --year 2026 --first 3
"""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest, evaluate, pipeline, provenance, sim
from .config import ROOT

EXPERIMENTS = ROOT / 'artifacts' / 'experiments'
LEDGER = EXPERIMENTS / 'ledger.jsonl'


def _all_events(hist):
    ev = hist[['year', 'round']].drop_duplicates().sort_values(['year', 'round'])
    return [tuple(int(v) for v in x) for x in ev.values]


def run(year, first=3, last=99, modes=('post_quali', 'pre_weekend'), eval_sims=300_000, calib_sims=6000,
        calib_last_n=13, rounds=2, workers=None, label='nested', log=print):
    t_start = time.time()
    analyses, hist = pipeline.load_state(log=lambda *a: None)
    ds = backtest.load_dataset(analyses, hist, lambda *a: None)
    outer = backtest._events(hist, year, first, last)
    every = _all_events(hist)
    config = dict(kind='nested_walk_forward', year=year, first=first, last=last, modes=list(modes),
                  eval_sims=eval_sims, calib_sims=calib_sims, calib_last_n=calib_last_n, rounds=rounds,
                  start_params=dict(sim.DEFAULTS), tunable=backtest.TUNABLE,
                  data=provenance.data_identity(), outer=[f'{y}_R{r:02d}' for y, r in outer])
    cfg_hash = provenance.sha256_json(config)[:12]
    exp_dir = EXPERIMENTS / f'{label}_{year}_{cfg_hash}'
    (exp_dir / 'races').mkdir(parents=True, exist_ok=True)
    (exp_dir / 'config.json').write_text(json.dumps(config, indent=1, default=str))
    log(f'  experiment {exp_dir.relative_to(ROOT)}  ({len(outer)} races, modes {", ".join(modes)})')

    params = dict(sim.DEFAULTS)
    rows, calib = [], []
    n_workers = sim.default_workers(workers)
    with sim.pool(n_workers) as ex:
        _loop(outer, every, exp_dir, analyses, hist, ds, params, rows, calib, modes, eval_sims, calib_sims,
              calib_last_n, rounds, n_workers, ex, log)
    _finish(exp_dir, rows, calib, cfg_hash, t_start, year, first, log)
    return pd.DataFrame(rows)


def _loop(outer, every, exp_dir, analyses, hist, ds, params, rows, calib, modes, eval_sims, calib_sims,
          calib_last_n, rounds, workers, ex, log):
    for k, (y, r) in enumerate(outer):
        ck = exp_dir / 'races' / f'{y}_R{r:02d}.json'
        if ck.exists():
            done = json.loads(ck.read_text())
            params = done['params']
            rows += done['rows']
            calib.append(done['calibration'])
            log(f'  {y} R{r:02d} resumed from checkpoint')
            continue
        t0 = time.time()
        inner = [e for e in every if e < (y, r)][-calib_last_n:]
        specs = backtest.prepare_specs(inner, analyses, hist, ds, params, 'post_quali', log=lambda *a: None)
        params, in_rps, in_ll, _ = backtest.search_params(specs, hist, params, calib_sims, rounds, workers,
                                                          log=lambda *a: None, ex=ex)
        cal = dict(race=f'{y}_R{r:02d}', inner=[f'{a}_R{b:02d}' for a, b in inner], inner_rps=in_rps,
                   inner_log_loss=in_ll, params={kk: params[kk] for kk in backtest.TUNABLE},
                   seconds=round(time.time() - t0, 1))
        race_rows = []
        for mode in modes:
            (_, _, spec, ctx), = backtest.prepare_specs([(y, r)], analyses, hist, ds, params, mode, log=lambda *a: None)
            agg = sim.simulate(spec, eval_sims, workers=workers, seed=50_000 + 101 * k, ex=ex)
            summ, dist, _ = pipeline.summarise(agg, ctx)
            res = evaluate.evaluate_event(y, r, summ, dist, hist, pace=ctx['pred'], save=False)
            if res is None:
                continue
            m, _ = res
            m.update(mode=mode, grid_eligible=(mode == 'post_quali'), new_venue=bool(ctx.get('new_venue')))
            race_rows.append({kk: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for kk, v in m.items()})
        ck.write_text(json.dumps(dict(params=params, rows=race_rows, calibration=cal), default=float))
        rows += race_rows
        calib.append(cal)
        post = [x for x in race_rows if x['mode'] == 'post_quali']
        log(f"  {y} R{r:02d} {race_rows[0]['event'] if race_rows else '':<18s} inner RPS {in_rps:.4f} | "
            + (f"post-quali model {post[0]['model_rps']:.4f} grid {post[0]['grid_rps']:.4f} | " if post else '')
            + ' '.join(f"{x['mode']} {x['model_rps']:.4f}" for x in race_rows if x['mode'] != 'post_quali')
            + f"   ({time.time() - t0:.0f}s)")


def _finish(exp_dir, rows, calib, cfg_hash, t_start, year, first, log):
    per_race = pd.DataFrame(rows)
    per_race.to_csv(exp_dir / 'per_race.csv', index=False)
    pd.DataFrame(calib).to_json(exp_dir / 'calibration.json', orient='records', indent=1)
    summary = summarise_experiment(per_race)
    (exp_dir / 'summary.json').write_text(json.dumps(summary, indent=1, default=float))
    entry = dict(id=exp_dir.name, created_utc=datetime.now(timezone.utc).isoformat(timespec='seconds'),
                 config_sha=cfg_hash, code=provenance.code_identity(), seconds=round(time.time() - t_start),
                 n_races=int(per_race[['year', 'round']].drop_duplicates().shape[0]) if len(per_race) else 0,
                 summary=summary, command=f'python -m flatout nested --year {year} --first {first}')
    EXPERIMENTS.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry, default=float) + '\n')
    log_summary(summary, log)


def paired(diff, n_boot=20_000, seed=0):
    """Mean paired per-race difference with a race-level bootstrap 95% interval."""
    d = np.asarray(diff, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return dict(n=int(len(d)), mean=float(d.mean()) if len(d) else None, lo=None, hi=None, share_better=None)
    rng = np.random.default_rng(seed)
    boots = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(1)
    return dict(n=int(len(d)), mean=float(d.mean()), lo=float(np.quantile(boots, 0.025)),
                hi=float(np.quantile(boots, 0.975)), share_better=float((d < 0).mean()))


def summarise_experiment(per_race):
    out = {}
    for mode, g in per_race.groupby('mode'):
        s = dict(n_races=int(len(g)))
        for who in ('model', 'grid', 'pace'):
            for met in ('rps', 'log_loss', 'brier_win', 'brier_podium', 'brier_points', 'top3_hit', 'winner_correct'):
                c = f'{who}_{met}'
                if c in g:
                    s[c] = float(pd.to_numeric(g[c], errors='coerce').mean())
        # paired: negative mean = model better (lower RPS / log loss)
        s['model_minus_pace_rps'] = paired(g.model_rps - g.pace_rps)
        s['model_minus_pace_log_loss'] = paired(g.model_log_loss - g.pace_log_loss)
        if mode == 'post_quali':
            s['model_minus_grid_rps'] = paired(g.model_rps - g.grid_rps)
            s['model_minus_grid_log_loss'] = paired(g.model_log_loss - g.grid_log_loss)
        else:
            s['grid_note'] = 'actual-grid baseline not eligible pre-weekend (grid unknown at forecast time)'
        out[mode] = s
    return out


def log_summary(summary, log=print):
    for mode, s in summary.items():
        log(f'\n  {mode}: {s["n_races"]} races   RPS model {s.get("model_rps", np.nan):.4f}   '
            f'pace-only {s.get("pace_rps", np.nan):.4f}' +
            (f'   grid {s.get("grid_rps", np.nan):.4f}' if mode == 'post_quali' else '   (grid not eligible)'))
        for key in ('model_minus_grid_rps', 'model_minus_pace_rps'):
            p = s.get(key)
            if p and p.get('lo') is not None:
                log(f'    {key:<24s} {p["mean"]:+.4f}  95% CI [{p["lo"]:+.4f}, {p["hi"]:+.4f}]  '
                    f'better in {p["share_better"]:.0%} of {p["n"]} races')
