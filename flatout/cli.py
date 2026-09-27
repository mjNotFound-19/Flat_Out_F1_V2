"""Command line: python -m flatout <command>

  weekend    the whole loop: sync -> build -> score finished races -> retrain -> predict next race
  sync       pull finished FastF1 sessions into the parquet store
  build      analyse races + rebuild the training dataset
  train      fit pace models (walk-forward calibrated uncertainty)
  predict    simulate a race (default: next event, 4,000,000 sims)
  evaluate   score the stored pre-race prediction for a finished race
  backtest   walk-forward accuracy for a season vs grid / pace-only baselines
  calibrate  tune simulator behaviour params on recent races
  ratings    car-adjusted driver ratings (race pace, quali, tyres, starts, execution)
  status     what is stored, predicted and evaluated
  export     refresh the website data (web/data/v3/site.json)
"""
import argparse
import json
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import backtest, evaluate, features, ingest, pipeline, race
from .config import MODELS, OUTPUT

pd.set_option('display.width', 220)
pd.set_option('display.max_columns', 40)


def _next_event(year):
    sch = ingest.load_schedule(year)
    now = datetime.now(timezone.utc)
    for _, ev in sch.sort_values('round').iterrows():
        sess = json.loads(ev.sessions)
        race_start = datetime.fromisoformat(sess['R']) if 'R' in sess else None
        stored = (ingest.session_dir(year, ev['round'], 'R') / 'laps.parquet').exists()
        if not stored and (race_start is None or race_start > now - pd.Timedelta(hours=4)):
            return int(ev['round'])
    return None


def _auto_mode(year, rnd):
    have = set(ingest.available_sessions(year, rnd))
    if 'Q' in have:
        return 'post_quali'
    if 'S' in have or 'SQ' in have:
        return 'post_sprint'
    if have & {'FP1', 'FP2', 'FP3'}:
        return 'post_fp'
    return 'pre_weekend'


def cmd_sync(a):
    years = a.years or [datetime.now().year]
    ingest.sync(years, full_history=a.full)


def cmd_build(a):
    analyses = race.analyze_all(force=a.force)
    features.build_dataset(analyses)


def cmd_train(a):
    analyses, hist = pipeline.load_state()
    ds = backtest.load_dataset(analyses, hist, rebuild=True)
    from .model import PaceModels
    t = time.time()
    m = PaceModels().fit(ds, calibrate=not a.fast)
    m.save()
    print(f'  trained on {len(ds)} rows in {time.time() - t:.0f}s')
    print(f'  pace sd (% of lap) by info level: {json.dumps(m.sd)}')
    print(f'  blend (gbm share): {m.blend}   cv: {getattr(m, "cv", None)}')
    print(m.importance().head(10).to_string(index=False))


def _print_prediction(df, extra, ctx, n):
    info = ctx['info']
    print(f"\n  {info['event']} ({info['location']})  |  {n:,} races in {extra['sim_seconds']}s  |  "
          f"P(SC/red) {extra['p_sc']:.0%}  E[SC] {extra['exp_sc']:.2f}  |  laps {ctx['circuit']['n_laps']}")
    view = df.copy()
    for c in ('win', 'podium', 'points', 'dnf', 'stop1', 'stop2', 'stop3p', 'strategy_p'):
        view[c] = (view[c] * 100).round(1)
    view['range'] = view.p10.astype(str) + '-' + view.p90.astype(str)
    view['pit1'] = view.first_stop_p50.round(0).astype('Int64').astype(str) + ' (' + \
        view.first_stop_p25.round(0).astype('Int64').astype(str) + '-' + view.first_stop_p75.round(0).astype('Int64').astype(str) + ')'
    cols = ['rank', 'Driver', 'Team', 'grid', 'win', 'podium', 'points', 'dnf', 'exp_pos', 'range', 'exp_pts',
            'pace_pct', 'stop1', 'stop2', 'modal_stops', 'strategy', 'strategy_p', 'pit1', 'alt_strategies']
    print(view[cols].round(2).to_string(index=False))


def cmd_predict(a):
    year = a.year or datetime.now().year
    rnd = a.round or _next_event(year)
    if rnd is None:
        sys.exit('  no upcoming event found')
    mode = a.mode if a.mode != 'auto' else _auto_mode(year, rnd)
    analyses, hist = pipeline.load_state()
    grid_override = None
    if a.grid:
        codes = [d.strip().upper() for d in a.grid.split(',') if d.strip()]
        entry = set(features.entry_list(year, rnd, hist).Driver)
        unknown = [c for c in codes if c not in entry]
        dup = sorted({c for c in codes if codes.count(c) > 1})
        if unknown or dup:
            sys.exit(f'  --grid: unknown driver codes {unknown} / duplicates {dup}. Entry list: {", ".join(sorted(entry))}')
        missing = sorted(entry - set(codes))
        if missing:
            print(f'  ! --grid omits {", ".join(missing)}: they start from the back (pit lane / not yet classified)')
        grid_override = {c: i + 1 for i, c in enumerate(codes)}
    print(f'  predicting {year} R{rnd:02d} with {a.sims:,} sims, information: {mode}')
    df, dist, extra, ctx = pipeline.run_event(year, rnd, a.sims, analyses, hist, workers=a.workers,
                                              mode=mode, grid_override=grid_override)
    extra['mode'] = mode
    d = pipeline.save_prediction(year, rnd, df, dist, extra, ctx, a.sims)
    _print_prediction(df, extra, ctx, a.sims)
    print(f'\n  saved -> {d}')
    if a.recommend:
        _recommend(d, df, ctx, a)
    if a.scenarios:
        import json
        from . import scenarios
        print('  what-if scenarios (precomputed for the Lab)...')
        sc = scenarios.run(ctx['spec'], ctx, n_sims=a.scenario_sims, workers=a.workers)
        meta = json.loads((d / 'pre_race_meta.json').read_text())
        (d / 'scenarios.json').write_text(json.dumps(dict(sc, run=meta.get('run')), indent=1, default=float))
        cmd_export()
    # keep the dashboard in sync
    web = pipeline.OUTPUT.parent / 'web' / 'data'
    if web.exists():
        df.to_csv(web / 'v3_prediction.csv', index=False)
        dist.to_csv(web / 'v3_distribution.csv', index=False)
    cmd_export()


def _recommend(event_dir, df, ctx, a):
    """Strategy recommendation (forced plans under common random numbers) for the front of the field."""
    import json
    from . import strategy_eval
    drivers = df.sort_values('exp_pos').Driver.head(a.recommend_top).tolist()
    print(f'  strategy recommendation: {len(drivers)} drivers x 5 plans x {a.recommend_sims:,} sims')
    rec = strategy_eval.recommend(ctx['spec'], ctx, drivers=drivers, top_plans=5, n_sims=a.recommend_sims,
                                  workers=a.workers)
    meta = json.loads((event_dir / 'pre_race_meta.json').read_text())
    out = dict(run=meta.get('run'), created_utc=meta.get('created_utc'), mode=meta.get('mode'),
               scope=strategy_eval.SCOPE, status=meta.get('status'), drivers=rec)
    (event_dir / 'recommendations.json').write_text(json.dumps(out, indent=1, default=float))
    for d, r in rec.items():
        print(f"    {d}: best {r['best']:<8s} vs behaviour mix {r['gain_pos']:+.2f} places")
    cmd_export()


def cmd_evaluate(a):
    analyses, hist = pipeline.load_state()
    targets = [(a.year, a.round)] if a.round else _unevaluated(hist)
    if not targets:
        print('  nothing to evaluate')
    for y, r in targets:
        d = pipeline.out_dir(y, r)
        if not (d / 'pre_race_distribution.csv').exists():
            print(f'  {y} R{r:02d}: no stored pre-race prediction')
            continue
        summ = pd.read_csv(d / 'pre_race_summary.csv')
        dist = pd.read_csv(d / 'pre_race_distribution.csv')
        res = evaluate.evaluate_event(y, r, summ, dist, hist)
        if res is None:
            print(f'  {y} R{r:02d}: race not stored yet')
            continue
        m, per = res
        _log_performance(m)
        _print_eval(m, per)
    if targets:
        cmd_export()


def _unevaluated(hist):
    out = []
    for p in sorted(OUTPUT.glob('*/R*/pre_race_summary.csv')):
        y, r = int(p.parent.parent.name), int(p.parent.name[1:])
        if not (p.parent / 'pre_race_evaluation.json').exists() and \
                len(hist[(hist.year == y) & (hist['round'] == r)]):
            out.append((y, r))
    return out


def _log_performance(m):
    p = OUTPUT / 'performance_log.csv'
    row = pd.DataFrame([m])
    if p.exists():
        old = pd.read_csv(p)
        old = old[~((old.year == m['year']) & (old['round'] == m['round']))]
        row = pd.concat([old, row], ignore_index=True)
    row.to_csv(p, index=False)


def _print_eval(m, per):
    print(f"\n  {m['year']} R{m['round']:02d} {m['event']}")
    for k in ('rps', 'log_loss', 'spearman', 'mae_pos', 'brier_win', 'brier_podium', 'p_winner'):
        g = m.get(f'grid_{k}', np.nan)
        print(f"    {k:<13s} model {m[f'model_{k}']:.4f}   grid baseline {g:.4f}")
    for k in ('stops_acc', 'stops_ll', 'seq_acc', 'first_stop_mae', 'first_stop_iqr_cover'):
        if f'strat_{k}' in m:
            print(f"    {k:<20s} {m[f'strat_{k}']:.3f}")
    print('    biggest surprises:')
    print(per.sort_values('surprise', ascending=False).head(5).round(2).to_string(index=False))


def cmd_backtest(a):
    res = backtest.run(a.year, a.first, a.last, a.sims, a.mode, a.workers)
    if len(res):
        print('\n  season averages (lower is better except spearman / p_winner / hit rates):')
        print(backtest.summary_table(res).round(4).to_string())
        sc = [c for c in res.columns if c.startswith('strat_')]
        print('\n  strategy:', res[sc].mean().round(3).to_dict())
    cmd_export()


def cmd_calibrate(a):
    backtest.calibrate(year=a.year, last_n=a.last_n, n_sims=a.sims, workers=a.workers)


def cmd_nested(a):
    from . import nested
    modes = tuple(m.strip() for m in a.modes.split(','))
    import json
    cand = json.loads(a.candidate) if a.candidate else None
    if cand and 'season_weight' in cand:            # JSON keys are strings; seasons are ints
        cand['season_weight'] = {int(k): float(v) for k, v in cand['season_weight'].items()}
    nested.run(a.year, a.first, a.last, modes=modes, eval_sims=a.sims, calib_sims=a.calib_sims,
               calib_last_n=a.last_n, workers=a.workers, label=a.label, recal_every=a.recal_every, candidate=cand)


def cmd_promote(a):
    import json
    from . import promotion
    r = promotion.evaluate(a.champion, a.challenger)
    for k, c in r['checks'].items():
        print(f"  {k:<24s} {'PASS' if c['pass_'] else ('n/a ' if c['pass_'] is None else 'FAIL')}  {json.dumps(c.get('value'), default=float)[:150]}")
    print(f"\n  promote: {r['promote']}  ({r['n_pairs']} race-mode pairs)")
    out = __import__('pathlib').Path(a.challenger) / 'promotion.json'
    out.write_text(json.dumps(r, indent=1, default=float))


def cmd_unseen(a):
    from . import unseen_eval
    unseen_eval.run(a.experiment, eval_sims=a.sims, workers=a.workers)


def cmd_progress(a):
    from . import progress
    progress.watch() if a.watch else progress.show(progress.status())


def cmd_snapshot(a):
    from . import provenance
    provenance.snapshot(a.label)


def cmd_ratings(a):
    from . import ratings
    analyses, hist = pipeline.load_state()
    r = ratings.driver_ratings(hist, year=a.year)
    ratings.print_ratings(r)
    r.to_csv(OUTPUT / f'driver_ratings_{a.year or "all"}.csv', index=False)
    cmd_export()


def cmd_export(a=None):
    from . import export
    try:
        export.export()
    except Exception as e:  # the website must never break the modelling loop
        print(f'  ! website export failed: {e}')


def cmd_status(a):
    year = a.year or datetime.now().year
    for y in sorted({year - 2, year - 1, year}):
        ev = [r for yy, r in ingest.stored_events() if yy == y]
        print(f'  {y}: races stored {len(ev)} -> {ev[-5:] if ev else []}')
    print(f'  next event: {year} R{_next_event(year)}')
    p = OUTPUT / 'performance_log.csv'
    if p.exists():
        log = pd.read_csv(p)
        print(log[['year', 'round', 'event', 'model_rps', 'grid_rps', 'model_spearman', 'model_p_winner',
                   'strat_stops_acc']].tail(10).round(3).to_string(index=False))
    for f in ('pace_models_meta.json', 'sim_params.json'):
        if (MODELS / f).exists():
            print(f'  {f}: updated {datetime.fromtimestamp((MODELS / f).stat().st_mtime):%Y-%m-%d %H:%M}')


def cmd_weekend(a):
    """The full loop, safe to run after every session (e.g. from Task Scheduler)."""
    year = a.year or datetime.now().year
    print('== 1/5 sync'); ingest.sync([year])
    print('== 2/5 build'); analyses = race.analyze_all(); features.build_dataset(analyses)
    print('== 3/5 evaluate finished races'); cmd_evaluate(argparse.Namespace(year=None, round=None))
    print('== 4/5 retrain')
    cmd_train(argparse.Namespace(fast=a.fast))
    if a.calibrate:
        backtest.calibrate(year=year, n_sims=4000, workers=a.workers)
    print('== 5/5 predict next race')
    cmd_predict(argparse.Namespace(year=year, round=None, sims=a.sims, workers=a.workers, mode='auto', grid=None))


def main():
    ap = argparse.ArgumentParser(prog='python -m flatout', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sync'); s.add_argument('--years', type=int, nargs='*'); s.add_argument('--full', action='store_true')
    s.set_defaults(fn=cmd_sync)
    s = sub.add_parser('build'); s.add_argument('--force', action='store_true'); s.set_defaults(fn=cmd_build)
    s = sub.add_parser('train'); s.add_argument('--fast', action='store_true'); s.set_defaults(fn=cmd_train)
    s = sub.add_parser('predict')
    s.add_argument('--year', type=int); s.add_argument('--round', type=int)
    s.add_argument('--sims', type=int, default=4_000_000); s.add_argument('--workers', type=int)
    s.add_argument('--mode', default='auto', choices=['auto', *features.MODES])
    s.add_argument('--grid', help='comma separated driver codes in grid order (validated against the entry list)')
    s.add_argument('--recommend', action='store_true', help='also compare forced strategy plans (static, model-dependent)')
    s.add_argument('--recommend-top', type=int, default=12); s.add_argument('--recommend-sims', type=int, default=40_000)
    s.add_argument('--scenarios', action='store_true', help='precompute what-if scenarios for the site Lab')
    s.add_argument('--scenario-sims', type=int, default=200_000)
    s.set_defaults(fn=cmd_predict)
    s = sub.add_parser('evaluate'); s.add_argument('--year', type=int); s.add_argument('--round', type=int)
    s.set_defaults(fn=cmd_evaluate)
    s = sub.add_parser('backtest')
    s.add_argument('--year', type=int, default=datetime.now().year)
    s.add_argument('--first', type=int, default=1); s.add_argument('--last', type=int, default=99)
    s.add_argument('--sims', type=int, default=4_000_000); s.add_argument('--workers', type=int)
    s.add_argument('--mode', default='post_quali', choices=list(features.MODES))
    s.set_defaults(fn=cmd_backtest)
    s = sub.add_parser('calibrate'); s.add_argument('--year', type=int); s.add_argument('--last-n', type=int, default=14)
    s.add_argument('--sims', type=int, default=6000); s.add_argument('--workers', type=int)
    s.set_defaults(fn=cmd_calibrate)
    s = sub.add_parser('nested', help='out-of-sample benchmark: sim params re-tuned before each race')
    s.add_argument('--year', type=int, required=True); s.add_argument('--first', type=int, default=3)
    s.add_argument('--last', type=int, default=99); s.add_argument('--modes', default='post_quali,pre_weekend')
    s.add_argument('--sims', type=int, default=300_000, help='per race and mode (MC error << model differences)')
    s.add_argument('--calib-sims', type=int, default=6000); s.add_argument('--last-n', type=int, default=13)
    s.add_argument('--workers', type=int); s.add_argument('--label', default='nested')
    s.add_argument('--recal-every', type=int, default=1, help='re-tune sim params every k-th race (reuse is leak-free)')
    s.add_argument('--candidate', help='challenger as JSON, e.g. {"form_settings": {"team_carry": 0.5}}')
    s.set_defaults(fn=cmd_nested)
    s = sub.add_parser('promote', help='apply docs/PROMOTION.md criteria to champion vs challenger experiments')
    s.add_argument('--champion', required=True); s.add_argument('--challenger', required=True); s.set_defaults(fn=cmd_promote)
    s = sub.add_parser('unseen', help='re-forecast a nested run as if every circuit were new')
    s.add_argument('--experiment', required=True); s.add_argument('--sims', type=int, default=300_000)
    s.add_argument('--workers', type=int); s.set_defaults(fn=cmd_unseen)
    s = sub.add_parser('progress', help='progress bars for running experiments (from checkpoints)')
    s.add_argument('--watch', action='store_true', help='refresh every 20 s until everything finishes')
    s.set_defaults(fn=cmd_progress)
    s = sub.add_parser('snapshot', help='copy uncommitted generated outputs + sha256 manifest')
    s.add_argument('--label', required=True); s.set_defaults(fn=cmd_snapshot)
    s = sub.add_parser('ratings'); s.add_argument('--year', type=int); s.set_defaults(fn=cmd_ratings)
    s = sub.add_parser('status'); s.add_argument('--year', type=int); s.set_defaults(fn=cmd_status)
    s = sub.add_parser('export'); s.set_defaults(fn=cmd_export)
    s = sub.add_parser('weekend'); s.add_argument('--year', type=int)
    s.add_argument('--sims', type=int, default=4_000_000); s.add_argument('--workers', type=int)
    s.add_argument('--fast', action='store_true', help='skip walk-forward uncertainty calibration when training')
    s.add_argument('--calibrate', action='store_true', help='also re-tune simulator params')
    s.set_defaults(fn=cmd_weekend)
    a = ap.parse_args()
    a.fn(a)


if __name__ == '__main__':
    main()
