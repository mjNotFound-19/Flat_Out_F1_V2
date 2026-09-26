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
        grid_override = {d.strip().upper(): i + 1 for i, d in enumerate(a.grid.split(','))}
    print(f'  predicting {year} R{rnd:02d} with {a.sims:,} sims, information: {mode}')
    df, dist, extra, ctx = pipeline.run_event(year, rnd, a.sims, analyses, hist, workers=a.workers,
                                              mode=mode, grid_override=grid_override)
    extra['mode'] = mode
    d = pipeline.save_prediction(year, rnd, df, dist, extra, ctx, a.sims)
    _print_prediction(df, extra, ctx, a.sims)
    print(f'\n  saved -> {d}')
    # keep the dashboard in sync
    web = pipeline.OUTPUT.parent / 'web' / 'data'
    if web.exists():
        df.to_csv(web / 'v3_prediction.csv', index=False)
        dist.to_csv(web / 'v3_distribution.csv', index=False)
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
    s.add_argument('--grid', help='comma separated driver codes in grid order, e.g. RUS,VER,LEC')
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
