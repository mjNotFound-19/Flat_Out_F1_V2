"""
Quick comparison tool: score two prediction files against the same actual results.

Usage:
  python compare_predictions.py <original_pred.csv> <patched_pred.csv> <scored.csv>

Example:
  python compare_predictions.py v2_prediction_results_original.csv v2_prediction_results.csv predictions/round_03/scored.csv
"""
import sys
import pandas as pd
import numpy as np
from scipy import stats


def score_against(pred_df, actual_df):
    """Score a prediction file against actual results.

    pred_df needs columns: Driver, Pos
    actual_df needs columns: Driver, actual (from scored.csv)
    Returns dict of metrics.
    """
    merged = pred_df[['Driver', 'Pos']].merge(
        actual_df[['Driver', 'actual', 'status']], on='Driver'
    )
    merged['abs_err'] = (merged['Pos'] - merged['actual']).abs()
    merged['err'] = merged['Pos'] - merged['actual']

    classified = merged[merged.status == 'CLASSIFIED']

    tau, _ = stats.kendalltau(merged.Pos, merged.actual)
    rho, _ = stats.spearmanr(merged.Pos, merged.actual)

    return {
        'n': len(merged),
        'mae_all': merged.abs_err.mean(),
        'mae_classified': classified.abs_err.mean() if len(classified) > 0 else np.nan,
        'exact': int((merged.abs_err == 0).sum()),
        'within_1': int((merged.abs_err <= 1).sum()),
        'within_3': int((merged.abs_err <= 3).sum()),
        'tau': tau,
        'rho': rho,
        'merged': merged,
    }


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    orig_path, patched_path, scored_path = sys.argv[1:4]

    orig = pd.read_csv(orig_path)
    patched = pd.read_csv(patched_path)
    actual = pd.read_csv(scored_path)

    orig_m = score_against(orig, actual)
    patch_m = score_against(patched, actual)

    def fmt(x):
        return f"{x:.3f}" if isinstance(x, float) and not pd.isna(x) else str(x)

    print(f"\n{'='*62}")
    print(f"  HEAD-TO-HEAD: {orig_path} vs {patched_path}")
    print(f"{'='*62}\n")
    print(f"  {'METRIC':<22s} {'ORIGINAL':>12s} {'PATCHED':>12s} {'DELTA':>10s}")
    print(f"  {'-'*60}")
    for key, label in [('mae_all',        'MAE (all drivers)'),
                       ('mae_classified', 'MAE (classified)'),
                       ('tau',            'Kendall tau'),
                       ('rho',            'Spearman rho'),
                       ('exact',          'Exact matches'),
                       ('within_1',       'Within +/-1'),
                       ('within_3',       'Within +/-3')]:
        o = orig_m[key]; p = patch_m[key]
        if isinstance(o, (int, float)) and not pd.isna(o):
            delta = p - o
            sign = '+' if delta > 0 else ''
            # For MAE, lower is better — flip arrow
            better = ' better' if (key.startswith('mae') and delta < 0) or \
                                   (not key.startswith('mae') and delta > 0) else \
                     (' worse' if delta != 0 else '')
            print(f"  {label:<22s} {fmt(o):>12s} {fmt(p):>12s} {sign}{fmt(delta):>9s}{better}")

    print(f"\n  DRIVER-LEVEL DIFFERENCES:")
    print(f"  {'DRIVER':6s} {'ACTUAL':>6s} {'ORIG':>5s} {'PATCH':>5s} {'ORIG_ERR':>9s} {'PATCH_ERR':>10s}")
    print(f"  {'-'*50}")
    comparison = orig_m['merged'].merge(
        patch_m['merged'][['Driver', 'Pos', 'abs_err']].rename(
            columns={'Pos': 'patch_pos', 'abs_err': 'patch_err'}),
        on='Driver'
    ).sort_values('actual')
    for _, r in comparison.iterrows():
        improved = '✓' if r.patch_err < r.abs_err else (' ' if r.patch_err == r.abs_err else '✗')
        print(f"  {r.Driver:6s} P{int(r.actual):>3d}   P{int(r.Pos):>2d}  P{int(r.patch_pos):>2d}   {int(r.abs_err):>7d}   {int(r.patch_err):>7d}  {improved}")

    if patch_m['mae_all'] < orig_m['mae_all']:
        print(f"\n  RESULT: Patched model is BETTER by {orig_m['mae_all']-patch_m['mae_all']:.3f} MAE")
    elif patch_m['mae_all'] > orig_m['mae_all']:
        print(f"\n  RESULT: Patched model is WORSE by {patch_m['mae_all']-orig_m['mae_all']:.3f} MAE")
    else:
        print(f"\n  RESULT: No difference in overall MAE")


if __name__ == '__main__':
    main()
