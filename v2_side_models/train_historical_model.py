"""
FLAT OUT F1 - HISTORICAL MLP TRAINER
======================================
Trains a PyTorch MLP on historical race data to predict raw finish position.

Architecture choices for ~800-row regime:
  - Small network (3 layers, 64→32→1)
  - Heavy dropout (0.4) to combat overfitting
  - Weight decay (L2 regularization)
  - Early stopping based on validation loss
  - Time-series cross-validation (train on past seasons, validate on latest)

Outputs:
  - historical_mlp.pt        (trained model weights)
  - historical_mlp_meta.json (scaler params, feature list, era stats)
  - training_report.txt      (CV scores, feature importance via permutation)

Setup:
  pip install torch scikit-learn
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import TensorDataset, DataLoader
except ImportError:
    sys.exit("PyTorch not installed. Run: pip install torch")

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from scipy import stats as sp


SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / 'historical_training_data.csv'
MODEL_PATH = SCRIPT_DIR / 'historical_mlp.pt'
META_PATH = SCRIPT_DIR / 'historical_mlp_meta.json'
REPORT_PATH = SCRIPT_DIR / 'training_report.txt'


# FEATURE STRATEGY:
# At PREDICTION time, we only have pre-race info (grid, quali, weather) plus
# whatever we can compute from the driver's PRIOR races (rolling averages).
# To train consistently, we must construct these features the same way at
# training time — i.e. for each row, "pace/craft" features come from that
# driver's PREVIOUS races, NOT the current race.
#
# This is what `rollup_prior_races()` does below.

# True pre-race features (known before the race)
PRE_RACE_COLS = ['grid_pos', 'era', 'q_delta_pct', 'track_temp', 'air_temp']

# Rolling-history features — constructed from the driver's PRIOR races.
# Each gets a `_prior` suffix after rollup to make the distinction explicit.
ROLLING_SOURCES = [
    'race_pace_mean', 'race_pace_std',
    'best_stint_pace', 'best_stint_deg',
    'gap_rate', 'gap_r2',
    'pace_delta_field',
    'start_pos', 'pos_std', 'lap1_gain', 'overtake_ratio',
    'n_stops', 'first_pit_lap', 'pit_timing_delta', 'n_compounds',
    'finish_pos',  # prior-race finish avg is itself predictive
]

ROLLING_COLS = [f"{c}_prior" for c in ROLLING_SOURCES]

FEATURE_COLS = PRE_RACE_COLS + ROLLING_COLS

# Categorical features (one-hot encoded)
CAT_COLS = ['Team', 'Driver', 'Circuit']

TARGET_COL = 'finish_pos'


def rollup_prior_races(df, sources, half_life_races=4):
    """For each (Year, Round, Driver) row, construct rolling averages of
    `sources` from that driver's PRIOR races only.

    Uses exponentially-weighted averages with a `half_life_races` half-life
    so recent races get more weight. Returns a new DataFrame with `_prior`
    columns appended; rows with no prior-race history get NaN (imputed later).
    """
    df = df.sort_values(['Driver', 'Year', 'Round']).reset_index(drop=True)
    out = df.copy()

    new_cols = {f"{s}_prior": np.full(len(df), np.nan) for s in sources}

    for drv in df.Driver.unique():
        dd = df[df.Driver == drv]
        indices = dd.index.tolist()
        # Walk through races chronologically
        for i, idx in enumerate(indices):
            if i == 0:
                # No prior races — leave NaN, imputed at dataset-level median
                continue
            prior = dd.iloc[:i]
            # Exponential weights: most recent race has weight 1, earlier decay
            ages = np.arange(len(prior))[::-1]   # [n-1, n-2, ..., 0]
            weights = 0.5 ** (ages / half_life_races)
            weights = weights / weights.sum()
            for s in sources:
                if s not in prior.columns: continue
                vals = prior[s].values
                mask = ~np.isnan(vals)
                if mask.sum() == 0: continue
                w = weights[mask]
                w = w / w.sum()
                new_cols[f"{s}_prior"][idx] = np.average(vals[mask], weights=w)

    for col, arr in new_cols.items():
        out[col] = arr

    return out


class F1MLP(nn.Module):
    """Regularized MLP for small-N F1 data."""

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


def prepare_dataset(df, feature_cols, cat_cols, target_col, era_weight=True, fit_encoders=None):
    """Build X, y arrays + encoder metadata.

    Args:
        df: DataFrame with features + target
        feature_cols: continuous features
        cat_cols: categorical features to one-hot encode
        target_col: target column name
        era_weight: if True, return sample weights emphasizing recent era
        fit_encoders: if given, use these (for test set); else fit new

    Returns:
        X, y, sample_weights, meta_dict
    """
    df = df.dropna(subset=[target_col]).copy()
    # Drop DNFs from training — race outcome doesn't reflect pace for them
    # (but we keep low finishers who were lapped/classified)
    if 'dnf' in df.columns:
        df = df[df.dnf == 0]

    # Continuous features: fill NaN with column median
    X_num = df[feature_cols].copy()
    if fit_encoders is None:
        medians = X_num.median()
    else:
        medians = pd.Series(fit_encoders['medians'])
    X_num = X_num.fillna(medians).fillna(0)

    # Categorical features: one-hot. Use union of categories from fit if provided.
    X_cat_frames = []
    cat_categories = {}
    for col in cat_cols:
        if col not in df.columns: continue
        vals = df[col].fillna('__MISSING__').astype(str)
        if fit_encoders is not None and col in fit_encoders['categories']:
            cats = fit_encoders['categories'][col]
        else:
            cats = sorted(vals.unique())
        cat_categories[col] = cats
        # Dummy columns
        dummies = pd.DataFrame(
            {f'{col}_{c}': (vals == c).astype(float) for c in cats},
            index=df.index
        )
        X_cat_frames.append(dummies)

    if X_cat_frames:
        X_cat = pd.concat(X_cat_frames, axis=1)
        X_full = pd.concat([X_num.reset_index(drop=True), X_cat.reset_index(drop=True)], axis=1)
    else:
        X_full = X_num.reset_index(drop=True)

    feature_names = list(X_full.columns)

    # Scale continuous features only; leave one-hots as {0,1}
    if fit_encoders is None:
        scaler = StandardScaler()
        X_full[feature_cols] = scaler.fit_transform(X_full[feature_cols])
    else:
        scaler = fit_encoders['scaler']
        X_full[feature_cols] = scaler.transform(X_full[feature_cols])

    # Era weighting: emphasize most recent seasons
    if era_weight and 'Year' in df.columns:
        years = df.Year.values
        max_year = years.max()
        weights = np.exp(-(max_year - years) * 0.25)  # decay ~0.78 per year back
        weights = weights / weights.mean()
    else:
        weights = np.ones(len(df))

    y = df[target_col].values.astype(np.float32)
    X = X_full.values.astype(np.float32)

    meta = {
        'feature_names': feature_names,
        'feature_cols': feature_cols,
        'cat_cols': cat_cols,
        'medians': medians.to_dict(),
        'scaler': scaler,
        'categories': cat_categories,
    }
    return X, y, weights.astype(np.float32), meta


def train_mlp(X, y, weights, n_features,
              epochs=400, batch_size=32, lr=1e-3, weight_decay=1e-2,
              patience=40, val_split=0.15, verbose=True):
    """Train MLP with early stopping."""
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    val_n = int(n * val_split)
    val_idx = idx[:val_n]; tr_idx = idx[val_n:]

    X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], weights[tr_idx]
    X_vl, y_vl, w_vl = X[val_idx], y[val_idx], weights[val_idx]

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = F1MLP(n_features).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)

    X_tr_t = torch.tensor(X_tr, device=device)
    y_tr_t = torch.tensor(y_tr, device=device)
    w_tr_t = torch.tensor(w_tr, device=device)
    X_vl_t = torch.tensor(X_vl, device=device)
    y_vl_t = torch.tensor(y_vl, device=device)

    ds = TensorDataset(X_tr_t, y_tr_t, w_tr_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val = float('inf')
    best_state = None
    patience_count = 0

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0; n_batch = 0
        for xb, yb, wb in dl:
            opt.zero_grad()
            pred = model(xb)
            # Weighted L1 loss (robust to outlier DNFs)
            loss = (wb * (pred - yb).abs()).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss_sum += loss.item(); n_batch += 1
        train_loss = train_loss_sum / max(n_batch, 1)

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_vl_t)
            val_loss = (val_pred - y_vl_t).abs().mean().item()
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                if verbose:
                    print(f"    Early stop at epoch {epoch+1} (val MAE: {best_val:.3f})")
                break

        if verbose and (epoch + 1) % 50 == 0:
            print(f"    Epoch {epoch+1:3d}  train MAE: {train_loss:.3f}  val MAE: {val_loss:.3f}  (best: {best_val:.3f})")

    model.load_state_dict(best_state)
    return model, best_val, device


def time_series_cv(df, feature_cols, cat_cols, target_col, n_folds=4):
    """Train on past years, validate on most recent year. Folds = rolling forward."""
    years = sorted(df.Year.unique())
    if len(years) < n_folds + 1:
        print(f"  Not enough years ({len(years)}) for {n_folds}-fold CV")
        n_folds = max(1, len(years) - 1)

    fold_scores = []
    for fold, val_year in enumerate(years[-n_folds:], start=1):
        train = df[df.Year < val_year]
        val = df[df.Year == val_year]
        if len(train) < 50 or len(val) < 10:
            continue
        print(f"\n  Fold {fold}: train on {train.Year.min()}-{train.Year.max()}, val on {val_year} (n_train={len(train)}, n_val={len(val)})")

        X_tr, y_tr, w_tr, meta = prepare_dataset(train, feature_cols, cat_cols, target_col)
        X_vl, y_vl, _, _ = prepare_dataset(val, feature_cols, cat_cols, target_col, fit_encoders=meta)

        model, _, device = train_mlp(X_tr, y_tr, w_tr, X_tr.shape[1], verbose=False)
        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_vl, device=device)).cpu().numpy()
        mae = mean_absolute_error(y_vl, preds)
        tau, _ = sp.kendalltau(y_vl, preds)
        print(f"    Val MAE: {mae:.3f}  tau: {tau:.3f}")
        fold_scores.append({'year': int(val_year), 'mae': mae, 'tau': tau, 'n': len(val)})

    return fold_scores


def permutation_importance(model, X, y, feature_names, device, n_repeats=5):
    """Compute permutation importance to identify most useful features."""
    model.eval()
    with torch.no_grad():
        base_pred = model(torch.tensor(X, device=device)).cpu().numpy()
    base_mae = mean_absolute_error(y, base_pred)

    importances = {}
    rng = np.random.RandomState(42)
    for i, name in enumerate(feature_names):
        scores = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            rng.shuffle(X_perm[:, i])
            with torch.no_grad():
                pred = model(torch.tensor(X_perm, device=device)).cpu().numpy()
            scores.append(mean_absolute_error(y, pred) - base_mae)
        importances[name] = float(np.mean(scores))
    return importances


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=str(DATA_PATH))
    parser.add_argument('--skip-cv', action='store_true', help='Skip time-series CV (faster)')
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--min-era', type=int, default=2, help='Drop data from eras below this (2=ground effect, 3=2026)')
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}\nRun build_historical_dataset.py first.")

    df = pd.read_csv(data_path)
    if 'era' in df.columns:
        df = df[df.era >= args.min_era].copy()
    print(f"  Loaded {len(df)} rows, {df.Year.nunique()} seasons, {df.Driver.nunique()} drivers")

    # Build rolling prior-race features. This ensures training mirrors prediction:
    # at both times, "race pace" and "racecraft" come from the driver's PRIOR
    # races, never from the race we're predicting.
    print(f"  Building rolling prior-race features ({len(ROLLING_SOURCES)} sources)...")
    df = rollup_prior_races(df, ROLLING_SOURCES, half_life_races=4)
    prior_null_pct = df[ROLLING_COLS[0]].isna().mean() * 100
    print(f"  First-race rows (no prior history): {prior_null_pct:.1f}%")

    # Time-series CV first to estimate real generalization
    cv_scores = []
    if not args.skip_cv:
        print(f"\n{'='*60}\n  TIME-SERIES CV\n{'='*60}")
        cv_scores = time_series_cv(df, FEATURE_COLS, CAT_COLS, TARGET_COL)

    # Final training on all data
    print(f"\n{'='*60}\n  FINAL TRAINING (all seasons)\n{'='*60}")
    X, y, w, meta = prepare_dataset(df, FEATURE_COLS, CAT_COLS, TARGET_COL)
    n_features = X.shape[1]
    print(f"  n_samples: {len(X)}  n_features: {n_features}")
    print(f"  (continuous: {len(FEATURE_COLS)}, one-hot expanded: {n_features - len(FEATURE_COLS)})")

    model, best_val, device = train_mlp(X, y, w, n_features, epochs=args.epochs)
    print(f"  Best val MAE (internal split): {best_val:.3f}")

    # Permutation importance
    print(f"\n  Computing permutation importance...")
    imps = permutation_importance(model, X, y, meta['feature_names'], device, n_repeats=3)
    top = sorted(imps.items(), key=lambda x: -x[1])[:20]

    # Save model + metadata
    torch.save(model.state_dict(), MODEL_PATH)
    meta_to_save = {
        'feature_names': meta['feature_names'],
        'feature_cols': meta['feature_cols'],
        'cat_cols': meta['cat_cols'],
        'medians': meta['medians'],
        'categories': meta['categories'],
        'scaler_mean': meta['scaler'].mean_.tolist(),
        'scaler_scale': meta['scaler'].scale_.tolist(),
        'cv_scores': cv_scores,
        'best_val_mae': best_val,
        'min_era_used': args.min_era,
        'n_features': n_features,
    }
    with open(META_PATH, 'w') as f:
        json.dump(meta_to_save, f, indent=2, default=str)

    # Training report
    report_lines = [
        "FLAT OUT F1 — HISTORICAL MLP TRAINING REPORT",
        "=" * 60, "",
        f"Data: {len(df)} rows, {df.Year.nunique()} seasons",
        f"Era filter: >= {args.min_era} (2=ground effect, 3=2026)",
        f"Final n_features: {n_features}",
        "",
        "TIME-SERIES CV:",
    ]
    if cv_scores:
        for s in cv_scores:
            report_lines.append(f"  {s['year']}: MAE {s['mae']:.3f}  tau {s['tau']:.3f}  (n={s['n']})")
        cv_mae = np.mean([s['mae'] for s in cv_scores])
        cv_tau = np.mean([s['tau'] for s in cv_scores])
        report_lines.append(f"  AVERAGE: MAE {cv_mae:.3f}  tau {cv_tau:.3f}")
    else:
        report_lines.append("  (skipped)")
    report_lines += [
        "",
        f"Internal val MAE (final model, 85/15 split): {best_val:.3f}",
        "",
        "TOP 20 FEATURES BY PERMUTATION IMPORTANCE:",
    ]
    for name, imp in top:
        report_lines.append(f"  {name:40s} {imp:+.4f}")

    report = "\n".join(report_lines)
    REPORT_PATH.write_text(report)
    print(f"\n{report}")
    print(f"\n  Model saved: {MODEL_PATH}")
    print(f"  Meta saved: {META_PATH}")
    print(f"  Report saved: {REPORT_PATH}")


if __name__ == '__main__':
    main()
