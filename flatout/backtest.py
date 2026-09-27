"""Walk-forward backtesting and simulator calibration.

For every race in range: fit the pace models on races strictly before it, build
circuit params with the same cutoff, simulate, and score against the result and
against the grid / pace-only baselines. Nothing from the race weekend after the
chosen information cut-off (default: after qualifying) is used.
"""
import json
import multiprocessing
import queue
import time

import numpy as np
import pandas as pd

from . import circuits as circ_mod, evaluate, features, pipeline, sim
from .config import DERIVED, MODELS, OUTPUT
from .model import PaceModels

TUNABLE = {
    'pass_thr': (0.15, 2.5), 'lap1_thr_mult': (0.3, 1.5), 'pace_sd_mult': (0.4, 2.0), 'start_sd': (0.1, 1.2),
    'lap_sd_mult': (0.5, 2.0), 'opp_window': (0.2, 0.8), 'extra_stop_p': (0.05, 0.7),
    'grid_blend': (0.02, 0.5),
}


def _events(hist, year, first, last):
    ev = hist[(hist.year == year) & hist['round'].between(first, last)][['year', 'round']].drop_duplicates()
    return [tuple(x) for x in ev.sort_values('round').values]


def prepare_specs(events, analyses, hist, ds, params, mode='post_quali', log=print, season_weight=None):
    """Walk-forward spec per event (model + circuit fit with the event as cutoff)."""
    out = []
    # uncertainty calibration uses only races before the first backtested event (no leakage)
    ref = PaceModels(season_weight).fit(ds, cutoff=events[0], calibrate=True) if events else None
    for year, rnd in events:
        m = PaceModels(season_weight).fit(ds, cutoff=(year, rnd), calibrate=False)
        m.sd, m.blend = ref.sd, ref.blend
        m.race.blend, m.quali.blend = ref.race.blend, ref.quali.blend
        m.band_mult = getattr(ref, 'band_mult', {})
        circuits = circ_mod.build(analyses, cutoff=(year, rnd))
        spec, ctx = pipeline.build_spec(year, rnd, analyses, hist, m, circuits, params,
                                        use_grid=(mode == 'post_quali'), mode=mode)
        out.append((year, rnd, spec, ctx))
        log(f'    spec {year} R{rnd:02d} {ctx["info"]["event"]}')
    return out


def _sim_one(args):
    spec, n, seed, *ticks = args
    return sim._worker(sim._spec_arrays(spec), n, seed, 10000, ticks[0] if ticks else None)


def run(year, first=1, last=99, n_sims=20000, mode='post_quali', workers=None, log=print, save=True):
    analyses, hist = pipeline.load_state(log=log)
    ds = load_dataset(analyses, hist, log)
    params = pipeline.load_sim_params()
    events = _events(hist, year, first, last)
    specs = prepare_specs(events, analyses, hist, ds, params, mode, log)
    rows, errs = [], []
    workers = sim.default_workers(workers, jobs=len(specs))
    log(f'  simulating {len(specs)} races x {n_sims:,} on {workers} workers')
    with sim.pool(workers) as ex:
        if log is not print:
            aggs = list(ex.map(_sim_one, [(s, n_sims, 11 + i) for i, (_, _, s, _) in enumerate(specs)]))
        else:   # same jobs, plus a progress bar fed by each worker's finished 10k batches
            with multiprocessing.Manager() as mgr:
                ticks = mgr.Queue()
                futs = [ex.submit(_sim_one, (s, n_sims, 11 + i, ticks)) for i, (_, _, s, _) in enumerate(specs)]
                bar, done = sim.ProgressBar(len(specs) * n_sims), 0
                while True:
                    try:
                        done += ticks.get(timeout=1)
                    except queue.Empty:
                        pass
                    finished = sum(f.done() for f in futs)
                    if finished == len(futs):
                        break
                    bar.update(min(done, len(specs) * n_sims - 1), f'{finished}/{len(specs)} races done')
                aggs = [f.result() for f in futs]
                bar.update(len(specs) * n_sims, f'{len(specs)}/{len(specs)} races done')
    for (y, r, spec, ctx), agg in zip(specs, aggs):
        summ, dist, extra = pipeline.summarise(agg, ctx)
        res = evaluate.evaluate_event(y, r, summ, dist, hist, pace=ctx['pred'], save=save, tag=f'backtest_{mode}')
        if res is None:
            continue
        m, per = res
        m['mode'] = mode
        rows.append(m)
        errs.append(per.assign(year=y, round=r))
        if save:
            d = pipeline.out_dir(y, r)
            summ.to_csv(d / f'backtest_{mode}_summary.csv', index=False)
            dist.to_csv(d / f'backtest_{mode}_distribution.csv', index=False)
        log(f"  {y} R{r:02d} {m['event']:<18s} RPS model {m['model_rps']:.4f} | grid {m['grid_rps']:.4f} | "
            f"pace {m.get('pace_rps', np.nan):.4f}   rho {m['model_spearman']:.2f}   "
            f"P(winner) {m['model_p_winner']:.2f}   stops acc {m.get('strat_stops_acc', np.nan):.2f}")
    res = pd.DataFrame(rows)
    if save and len(res):
        res.to_csv(OUTPUT / f'backtest_{year}_{mode}.csv', index=False)
        pd.concat(errs).to_csv(OUTPUT / f'backtest_{year}_{mode}_driver_errors.csv', index=False)
    return res


def summary_table(res):
    keys = ['log_loss', 'rps', 'brier_win', 'brier_podium', 'brier_points', 'spearman', 'mae_pos', 'p_winner',
            'top3_hit', 'top10_hit', 'winner_correct']
    t = pd.DataFrame({src: [res[f'{src}_{k}'].mean() if f'{src}_{k}' in res else np.nan for k in keys]
                      for src in ('model', 'pace', 'grid')}, index=keys)
    return t


def load_dataset(analyses, hist, log=print, rebuild=False):
    p = DERIVED / 'dataset.parquet'
    if p.exists() and not rebuild:
        ds = pd.read_parquet(p)
        have = set(map(tuple, ds[['year', 'round']].drop_duplicates().values.tolist()))
        need = set(map(tuple, hist[['year', 'round']].drop_duplicates().values.tolist()))
        if need <= have:
            return ds
    ds, _ = features.build_dataset(analyses, log=log)
    return ds


# ------------------------------------------------------------------ calibration

def _score(specs_aggs, hist):
    rps, ll = [], []
    for (y, r, spec, ctx), agg in specs_aggs:
        summ, dist, _ = pipeline.summarise(agg, ctx)
        res = evaluate.evaluate_event(y, r, summ, dist, hist, save=False)
        if res:
            rps.append(res[0]['model_rps'])
            ll.append(res[0]['model_log_loss'])
    return float(np.mean(rps)), float(np.mean(ll))


def search_params(specs, hist, params, n_sims=6000, rounds=2, workers=None, log=print, seed0=1000, ex=None):
    """Coordinate search over TUNABLE simulator params on prepared (walk-forward) specs, minimising
    mean RPS + 0.01 * log loss. Common random numbers (seed per race, fixed across trials) keep the
    comparison between parameter values fair. Returns (params, best_rps, best_ll, history).
    ex: an open worker pool to reuse (starting 11 workers per trial dominated calibration time)."""
    if ex is None:
        with sim.pool(sim.default_workers(workers, jobs=len(specs))) as own:
            return search_params(specs, hist, params, n_sims, rounds, workers, log, seed0, own)
    params = dict(params)

    def objective(p):
        jobs = []
        for i, (_, _, s, _) in enumerate(specs):
            s = dict(s)
            s['params'] = p
            jobs.append((s, n_sims, seed0 + i))
        aggs = list(ex.map(_sim_one, jobs))
        for _, _, _, ctx in specs:
            ctx['grid_blend'] = p.get('grid_blend', 0.0)
        rps, ll = _score(list(zip(specs, aggs)), hist)
        return rps + 0.01 * ll, rps, ll

    n_trials = 1 + sum(len((0.6, 0.8, 1.25, 1.6) if rd == 0 else (0.85, 1.15)) * len(TUNABLE) for rd in range(rounds))
    bar = sim.ProgressBar(n_trials, label='calibrating', unit='trials') if log is print else None
    done = 1
    best, best_rps, best_ll = objective(params)
    if bar:
        bar.update(done, f'RPS {best_rps:.4f}')
    log(f'  start: RPS {best_rps:.4f}  logloss {best_ll:.3f}')
    history = [dict(step='start', rps=best_rps, ll=best_ll, **{k: params[k] for k in TUNABLE})]
    for rd in range(rounds):
        for k, (lo, hi) in TUNABLE.items():
            cur = params[k]
            mults = (0.6, 0.8, 1.25, 1.6) if rd == 0 else (0.85, 1.15)
            for mult in mults:
                v = float(np.clip(cur * mult, lo, hi))
                if abs(v - params[k]) < 1e-9:
                    continue
                trial = dict(params, **{k: v})
                obj, rps, ll = objective(trial)
                done += 1
                if bar:
                    bar.update(min(done, n_trials - 1), f'best RPS {min(best_rps, rps):.4f}')
                if obj < best - 1e-5:
                    best, best_rps, best_ll, params = obj, rps, ll, trial
                    log(f'    {k} -> {v:.3f}   RPS {rps:.4f}  logloss {ll:.3f}')
            history.append(dict(step=f'r{rd}_{k}', rps=best_rps, ll=best_ll, **{kk: params[kk] for kk in TUNABLE}))
    if bar:
        bar.update(n_trials, f'best RPS {best_rps:.4f}')
    return params, best_rps, best_ll, history


def calibrate(year=None, last_n=14, n_sims=6000, rounds=2, workers=None, log=print):
    """Tune simulator params on the most recent `last_n` races and save them for forecasting.
    The saved params are fitted on those races: scoring the same races with them is in-sample
    (use `python -m flatout nested` for out-of-sample evaluation)."""
    analyses, hist = pipeline.load_state(log=log)
    ds = load_dataset(analyses, hist, log)
    params = pipeline.load_sim_params()
    ev = hist[['year', 'round']].drop_duplicates().sort_values(['year', 'round'])
    if year:
        ev = ev[ev.year == year]
    ev = [tuple(x) for x in ev.values[-last_n:]]
    log(f'  calibrating on {len(ev)} races, {n_sims} sims each')
    specs = prepare_specs(ev, analyses, hist, ds, params, 'post_quali', log=lambda *_: None)
    params, best_rps, best_ll, history = search_params(specs, hist, params, n_sims, rounds, workers, log)
    (MODELS / 'sim_params.json').write_text(json.dumps(dict(
        params={k: params[k] for k in sim.DEFAULTS}, rps=best_rps, log_loss=best_ll,
        races=[f'{y}_R{r:02d}' for y, r in ev], n_sims=n_sims, calibrated=time.strftime('%Y-%m-%d %H:%M'),
        note='in-sample on the listed races'), indent=1))
    pd.DataFrame(history).to_csv(MODELS / 'calibration_history.csv', index=False)
    log(f'  final: RPS {best_rps:.4f}  logloss {best_ll:.3f}  -> models/sim_params.json')
    return params
