"""
FLAT OUT F1 - ENSEMBLE PREDICTOR
==================================
Blends V2's prediction (v2_prediction_results.csv) with the historical MLP
model to produce a single ranked prediction.

Strategy:
  1. Read V2's final prediction (already has pace-derived scores)
  2. Run MLP on current-round features to get independent prediction
  3. Blend: weighted average of V2 and MLP predictions
  4. Re-rank by blended score

Usage:
  # After running f1_ml_predict_v2.py for the current round:
  python f1_ensemble_predict.py --round 3

Blend weight auto-scales with evidence:
  - Early season (R1-R3): heavy V2 weight (current-weekend data is dominant)
  - Mid season (R4+): MLP gets more weight as historical patterns stabilize
"""

import argparse
import json
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch")

SCRIPT_DIR = Path(__file__).resolve().parent
V2_PRED_PATH = SCRIPT_DIR / 'v2_prediction_results.csv'
SEASON_DATA_PATH = SCRIPT_DIR / 'season_race_data.csv'
MODEL_PATH = SCRIPT_DIR / 'historical_mlp.pt'
META_PATH = SCRIPT_DIR / 'historical_mlp_meta.json'


class F1MLP(nn.Module):
    """Same architecture as trainer — must match exactly."""
    def __init__(self, n_features, hidden1=64, hidden2=32, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


def encode_row(row, meta):
    """Encode a single driver's features using trained-model metadata."""
    # Continuous features
    feat_cols = meta['feature_cols']
    medians = meta['medians']
    num_vals = []
    for c in feat_cols:
        v = row.get(c, np.nan)
        if pd.isna(v):
            v = medians.get(c, 0)
        num_vals.append(float(v))

    # Apply scaler
    scaler_mean = np.array(meta['scaler_mean'])
    scaler_scale = np.array(meta['scaler_scale'])
    num_arr = (np.array(num_vals) - scaler_mean) / scaler_scale

    # One-hot categoricals
    cat_vals = []
    for col in meta['cat_cols']:
        val = str(row.get(col, '__MISSING__'))
        for c in meta['categories'].get(col, []):
            cat_vals.append(1.0 if val == c else 0.0)

    return np.concatenate([num_arr, np.array(cat_vals, dtype=np.float64)])


def build_current_features(round_num, season_data_path, v2_pred_df):
    """Build feature rows for current-round drivers using available data.

    Returns rows with the SAME schema as the trainer's FEATURE_COLS — specifically
    pre-race columns + `*_prior` rolling-history columns from prior races.
    """
    # Source columns that get rolled up into `_prior` features
    ROLLING_SOURCES = [
        'race_pace_mean', 'race_pace_std',
        'best_stint_pace', 'best_stint_deg',
        'gap_rate', 'gap_r2',
        'pace_delta_field',
        'start_pos', 'pos_std', 'lap1_gain', 'overtake_ratio',
        'n_stops', 'first_pit_lap', 'pit_timing_delta', 'n_compounds',
        'finish_pos',
    ]

    rows = []
    if season_data_path.exists():
        srd = pd.read_csv(season_data_path)
        srd_past = srd[srd.Round < round_num]
    else:
        srd_past = pd.DataFrame()

    for _, vr in v2_pred_df.iterrows():
        drv = vr.Driver
        row = {
            'Driver': drv, 'Team': vr.Team,
            'grid_pos': vr.get('GridPos', np.nan),
            'era': 3,  # 2026 regs
        }
        # Per-season row doesn't have track/air temp or q_delta_pct yet — they'll
        # be imputed to dataset medians by encode_row()
        row['track_temp'] = np.nan
        row['air_temp'] = np.nan
        row['q_delta_pct'] = np.nan

        # Rolling averages from prior-round season data
        if len(srd_past) > 0:
            dd = srd_past[srd_past.Driver == drv].sort_values('Round')
            if len(dd) > 0:
                # Exponential weighting matching trainer (half-life 4 races)
                ages = np.arange(len(dd))[::-1]
                w = 0.5 ** (ages / 4)
                w = w / w.sum()
                for src in ROLLING_SOURCES:
                    if src not in dd.columns: continue
                    vals = dd[src].values
                    mask = ~np.isnan(vals)
                    if mask.sum() == 0: continue
                    use_w = w[mask]; use_w = use_w / use_w.sum()
                    row[f"{src}_prior"] = np.average(vals[mask], weights=use_w)

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--round', type=int, required=True, help='Current round number')
    parser.add_argument('--mlp-weight', type=float, default=None,
                        help='Override auto-computed MLP weight (0.0-1.0)')
    parser.add_argument('--output', default=str(SCRIPT_DIR / 'ensemble_prediction.csv'))
    args = parser.parse_args()

    # Check prerequisites
    missing = []
    for p, label in [(V2_PRED_PATH, 'v2_prediction_results.csv'),
                     (MODEL_PATH, 'historical_mlp.pt'),
                     (META_PATH, 'historical_mlp_meta.json')]:
        if not p.exists():
            missing.append(label)
    if missing:
        sys.exit(f"Missing: {', '.join(missing)}")

    # Load everything
    v2 = pd.read_csv(V2_PRED_PATH)
    with open(META_PATH) as f:
        meta = json.load(f)
    n_features = meta['n_features']

    model = F1MLP(n_features)
    model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
    model.eval()

    print(f"  Loaded V2 predictions: {len(v2)} drivers")
    print(f"  Loaded MLP: {n_features} features, CV scores available: {len(meta.get('cv_scores', []))}")

    # Build feature rows for current round
    feat_df = build_current_features(args.round, SEASON_DATA_PATH, v2)

    # Encode + predict
    encoded = np.stack([encode_row(r, meta) for _, r in feat_df.iterrows()]).astype(np.float32)
    with torch.no_grad():
        mlp_preds = model(torch.tensor(encoded)).numpy()

    # V2 predictions are in the 'Pos' column (rank 1-22)
    v2_preds = v2.sort_values('avg_fin').reset_index(drop=True)
    v2_pos_by_drv = {r.Driver: r.Pos for _, r in v2_preds.iterrows()}

    # Build blended output
    out_rows = []
    for i, (_, fr) in enumerate(feat_df.iterrows()):
        drv = fr.Driver
        v2_pos = v2_pos_by_drv.get(drv, np.nan)
        mlp_pos = float(mlp_preds[i])
        out_rows.append({
            'Driver': drv, 'Team': fr.Team,
            'v2_pos': v2_pos, 'mlp_pos': mlp_pos,
        })

    out = pd.DataFrame(out_rows)

    # Auto-weighting: MLP gets more weight as we have more CV evidence and later in season
    if args.mlp_weight is not None:
        mlp_w = args.mlp_weight
    else:
        # Base weight 0.25, scales up to 0.45 by R15
        mlp_w = min(0.45, 0.25 + 0.015 * max(0, args.round - 1))

    v2_w = 1 - mlp_w
    out['blended_score'] = v2_w * out.v2_pos + mlp_w * out.mlp_pos
    out = out.sort_values('blended_score').reset_index(drop=True)
    # Name column `Pos` to match V2 output schema — lets compare_predictions.py work out of the box
    out['Pos'] = np.arange(1, len(out) + 1)
    out['ensemble_pos'] = out['Pos']

    print(f"\n  Blend weights: V2={v2_w:.2f}, MLP={mlp_w:.2f}")
    print(f"\n  {'POS':>3s} {'DRV':4s} {'V2':>3s} {'MLP':>5s} {'BLEND':>6s}")
    print(f"  {'-'*30}")
    for _, r in out.iterrows():
        print(f"  {int(r.Pos):>3d} {r.Driver:4s} {int(r.v2_pos):>3d} {r.mlp_pos:>5.1f} {r.blended_score:>6.2f}")

    out.to_csv(args.output, index=False)
    print(f"\n  Saved: {args.output}")


if __name__ == '__main__':
    main()
