"""
FLAT OUT F1 v2.0 - AUTONOMOUS SELF-TUNING ENGINE
==================================================
After each race, analyzes prediction errors and automatically adjusts:
  1. Tier weight percentages (FP/Season/Testing/History/Team)
  2. Grid penalty model (overtaking difficulty per circuit)
  3. Uncertainty parameters (per-driver variance calibration)
  4. Circuit profiles (safety car, deg, DRS effectiveness)
  5. Lap classification thresholds

Usage:
  Called automatically by f1_learn.py after ingesting a race.
  Can also run standalone: python f1_autotune.py <round>

Principle: gradient-free optimization via directional error analysis.
  "What would my MAE have been if I weighted X more/less?"
"""

import json, sys
import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats as sp

ROOT = Path(__file__).resolve().parent
PARAMS_PATH = ROOT / 'f1_hyperparams.json'
PRED_DIR = ROOT / 'predictions'

def load_params():
    if PARAMS_PATH.exists():
        with open(PARAMS_PATH) as f: return json.load(f)
    print("  No hyperparams file found. Run with default params.")
    return None

def save_params(params):
    with open(PARAMS_PATH, 'w') as f:
        json.dump(params, f, indent=2, default=str)


def tune_tier_weights(params, scored_df, feat_df):
    """
    Analyze: did drivers where FP data was strong get predicted well?
    Did historical-bias drivers get predicted poorly?
    Adjust tier weights accordingly.
    """
    lr = params['learning_rates']['tier_lr']
    tw = params['tier_weights']
    
    # Split errors by which tier had most influence on each driver
    fp_errors = []; hist_errors = []; team_errors = []
    
    for _, r in scored_df.iterrows():
        drv = r.Driver
        err = abs(r.error)
        fr = feat_df[feat_df.Driver == drv]
        if len(fr) == 0: continue
        
        fp_laps = fr.iloc[0].get('fp_laps', 0)
        has_history = pd.notna(fr.iloc[0].get('driver_score')) and fr.iloc[0].get('driver_score', 50) != 50
        
        if pd.notna(fp_laps) and fp_laps > 10:
            fp_errors.append(err)
        if has_history:
            hist_errors.append(err)
    
    # If FP-heavy drivers are more accurate, increase FP weight
    fp_mae = np.mean(fp_errors) if fp_errors else 5
    hist_mae = np.mean(hist_errors) if hist_errors else 5
    overall_mae = scored_df.abs_error.mean()
    
    adjustments = {}
    
    # FP adjustment: if FP drivers are MORE accurate than average, increase FP
    if fp_mae < overall_mae * 0.9:
        adj = lr * 0.5  # increase FP
        tw['fp_base'] = min(0.60, tw['fp_base'] + adj)
        tw['history_base'] = max(0.05, tw['history_base'] - adj * 0.5)
        adjustments['fp'] = f"+{adj:.3f} (FP MAE {fp_mae:.2f} < avg {overall_mae:.2f})"
    elif fp_mae > overall_mae * 1.1:
        adj = lr * 0.3
        tw['fp_base'] = max(0.20, tw['fp_base'] - adj)
        adjustments['fp'] = f"-{adj:.3f} (FP MAE {fp_mae:.2f} > avg {overall_mae:.2f})"
    
    # History adjustment: if history-heavy drivers are LESS accurate, decrease
    if hist_mae > overall_mae * 1.1:
        adj = lr * 0.3
        tw['history_base'] = max(0.05, tw['history_base'] - adj)
        adjustments['history'] = f"-{adj:.3f} (hist MAE {hist_mae:.2f} > avg {overall_mae:.2f})"
    
    params['tier_weights'] = tw
    return params, adjustments


def tune_grid_model(params, scored_df, grid):
    """
    Analyze: did drivers gain/lose positions as our grid model predicted?
    Compare predicted overtaking vs actual to calibrate penalties.
    """
    lr = params['learning_rates']['grid_lr']
    gm = params['grid_model']
    
    # For each driver, compare predicted vs actual position change from grid
    over_penalized = 0  # drivers we predicted worse than they finished (grid too harsh)
    under_penalized = 0  # drivers we predicted better than they finished (grid too lenient)
    
    for _, r in scored_df.iterrows():
        grd = grid.get(r.Driver, 11)
        pred_gain = grd - r.pred  # how many positions we predicted they'd gain
        actual_gain = grd - r.actual  # how many they actually gained
        
        if actual_gain > pred_gain + 2:
            over_penalized += 1  # reality: easier to overtake than model thought
        elif actual_gain < pred_gain - 2:
            under_penalized += 1  # reality: harder to overtake
    
    n = len(scored_df)
    if n == 0: return params, {}
    
    adjustments = {}
    
    if over_penalized > under_penalized + 2:
        # Overtaking was easier than model predicted -> reduce penalties
        for key in ['p4_p10_penalty_per_slot', 'p11_p15_penalty_per_slot', 'p16_plus_penalty_per_slot']:
            gm[key] = max(0.1, gm[key] - lr)
        gm['clean_air_bonus_per_slot'] = max(0.3, gm['clean_air_bonus_per_slot'] - lr * 0.5)
        adjustments['grid'] = f"SOFTENED (overtaking easier: {over_penalized} over-penalized vs {under_penalized} under)"
    
    elif under_penalized > over_penalized + 2:
        # Overtaking was harder -> increase penalties
        for key in ['p4_p10_penalty_per_slot', 'p11_p15_penalty_per_slot', 'p16_plus_penalty_per_slot']:
            gm[key] = min(2.0, gm[key] + lr)
        gm['clean_air_bonus_per_slot'] = min(2.5, gm['clean_air_bonus_per_slot'] + lr * 0.5)
        adjustments['grid'] = f"HARDENED (overtaking harder: {under_penalized} under-penalized vs {over_penalized} over)"
    else:
        adjustments['grid'] = f"OK (balanced: {over_penalized} over vs {under_penalized} under)"
    
    # Grid blend: if grid-position-correlated errors, adjust blend
    if len(scored_df) >= 10:
        grid_vals = [grid.get(r.Driver, 11) for _, r in scored_df.iterrows()]
        corr, _ = sp.spearmanr(grid_vals, scored_df.actual.values)
        if corr > 0.8:
            # Grid is very predictive -> increase blend
            gm['grid_blend_pct'] = min(0.25, gm['grid_blend_pct'] + lr * 0.5)
            adjustments['blend'] = f"UP to {gm['grid_blend_pct']:.0%} (grid corr={corr:.2f})"
        elif corr < 0.5:
            gm['grid_blend_pct'] = max(0.05, gm['grid_blend_pct'] - lr * 0.5)
            adjustments['blend'] = f"DOWN to {gm['grid_blend_pct']:.0%} (grid corr={corr:.2f})"
    
    params['grid_model'] = gm
    return params, adjustments


def learn_circuit_profile(params, scored_df, race_laps, grid, circuit_name, round_num):
    """
    Build/update circuit-specific profile from actual race data:
    - Overtaking factor: how many positions changed vs grid
    - Safety car probability: from race control data
    - Tyre deg: from stint degradation slopes
    - DRS effectiveness: from overtaking in DRS zones
    """
    from pathlib import Path
    
    def to_sec(td):
        try:
            if pd.isna(td) or td=='': return np.nan
            p=str(td).split(' days ')
            if len(p)==2: h,m,s=p[1].split(':'); return float(h)*3600+float(m)*60+float(s)
        except: pass
        return np.nan
    
    cp = params.get('circuit_profiles', {})
    defaults = cp.get('_default', {})
    
    # Compute overtaking factor: average abs(grid - finish) / 11 (expected random shuffle)
    changes = []
    for _, r in scored_df.iterrows():
        grd = grid.get(r.Driver, 0)
        if grd > 0 and r.actual > 0:
            changes.append(abs(grd - r.actual))
    
    overtaking = np.mean(changes) / 5.0 if changes else 1.0  # normalized: 1.0 = average
    
    # Tyre degradation from race stints
    race_laps_c = race_laps.copy()
    race_laps_c['lt'] = race_laps_c['LapTime'].apply(to_sec)
    clean = race_laps_c[(race_laps_c['lt']>70)&(race_laps_c['lt']<200)&
                         race_laps_c.PitInTime.isna()&race_laps_c.PitOutTime.isna()]
    
    degs = []
    for (drv, stint), sg in clean.groupby(['Driver', 'Stint']):
        if len(sg) >= 10:
            sl = sp.linregress(sg.TyreLife.values.astype(float), sg['lt'].values).slope
            if 0 < sl < 0.5:  # reasonable deg range
                degs.append(sl)
    
    tyre_deg = np.median(degs) / 0.05 if degs else 1.0  # normalized: 1.0 = 0.05s/lap avg
    
    # Safety car: check race control data
    rc_path = None
    import re
    for item in (ROOT / 'fastf1_data').iterdir():
        if item.is_dir() and re.match(rf'\d+_R{round_num}_.*_Race', item.name, re.I):
            rc_candidate = item / 'race_control.csv'
            if rc_candidate.exists(): rc_path = rc_candidate; break
    
    sc_count = 0
    if rc_path:
        rc = pd.read_csv(rc_path)
        sc_count = len(rc[rc.Message.str.contains('SAFETY CAR', case=False, na=False)]) if 'Message' in rc.columns else 0
    
    sc_prob = sc_count / 58 if sc_count > 0 else 0.025  # per-lap probability
    
    circuit_data = {
        'overtaking_factor': round(float(overtaking), 3),
        'safety_car_prob': round(float(sc_prob), 4),
        'tyre_deg_factor': round(float(tyre_deg), 3),
        'drs_effectiveness': 1.0,  # TODO: analyze DRS zone overtakes
        'n_races': 1,
        'last_round': round_num,
    }
    
    # Update with exponential average if circuit already exists
    if circuit_name in cp and circuit_name != '_default':
        old = cp[circuit_name]
        n = old.get('n_races', 0)
        alpha = 1 / (n + 1) if n < 5 else 0.2
        for key in ['overtaking_factor', 'safety_car_prob', 'tyre_deg_factor']:
            if key in old:
                circuit_data[key] = round(old[key] * (1 - alpha) + circuit_data[key] * alpha, 3)
        circuit_data['n_races'] = n + 1
    
    cp[circuit_name] = circuit_data
    params['circuit_profiles'] = cp
    return params, circuit_data


def tune_uncertainty(params, scored_df):
    """
    Calibrate uncertainty: are our confidence intervals correct?
    If actual results fall outside our P10-P90 range too often,
    uncertainty is too low. If they're always inside, too high.
    """
    lr = params['learning_rates']['uncertainty_lr']
    unc = params['uncertainty']
    
    # Load full predictions to check ranges
    pred_path = ROOT / 'v2_full_predictions.csv'
    if not pred_path.exists(): return params, {}
    
    full = pd.read_csv(pred_path)
    
    outside_range = 0
    inside_tight = 0
    n = 0
    
    for _, r in scored_df.iterrows():
        fp = full[full.Driver == r.Driver]
        if len(fp) == 0: continue
        p10 = fp.iloc[0].get('p10', 1)
        p90 = fp.iloc[0].get('p90', 22)
        actual = r.actual
        n += 1
        
        if actual < p10 or actual > p90:
            outside_range += 1  # actual fell outside our range
        elif abs(actual - fp.iloc[0].get('Pos', 11)) <= 1:
            inside_tight += 1  # actual was very close to prediction
    
    adjustments = {}
    if n >= 10:
        outside_pct = outside_range / n
        if outside_pct > 0.25:
            # Too many surprises -> increase uncertainty
            unc['base_unc'] = min(4.0, unc['base_unc'] + lr)
            adjustments['unc'] = f"INCREASED (outside range: {outside_pct:.0%})"
        elif outside_pct < 0.10:
            # Too conservative -> decrease uncertainty
            unc['base_unc'] = max(1.0, unc['base_unc'] - lr)
            adjustments['unc'] = f"DECREASED (outside range: {outside_pct:.0%})"
        else:
            adjustments['unc'] = f"OK (outside range: {outside_pct:.0%})"
    
    params['uncertainty'] = unc
    return params, adjustments


def log_performance(params, round_num, mae, tau):
    """Track performance over time for trend analysis."""
    log = params.get('performance_log', [])
    log.append({
        'round': round_num,
        'mae': round(mae, 3),
        'tau': round(tau, 3),
        'tier_weights': dict(params['tier_weights']),
        'grid_blend': params['grid_model']['grid_blend_pct'],
    })
    params['performance_log'] = log
    
    # Trend analysis
    if len(log) >= 3:
        recent_mae = np.mean([l['mae'] for l in log[-3:]])
        early_mae = np.mean([l['mae'] for l in log[:3]])
        if recent_mae < early_mae:
            print(f"  TREND: Improving (early MAE {early_mae:.2f} -> recent {recent_mae:.2f})")
        else:
            print(f"  TREND: Degrading (early MAE {early_mae:.2f} -> recent {recent_mae:.2f})")
    
    return params


def autotune(round_num, circuit_name='Unknown'):
    """Main entry point: run all tuning after a race."""
    print(f"\n{'='*70}")
    print(f"  AUTONOMOUS SELF-TUNING - Round {round_num}")
    print(f"{'='*70}")
    
    params = load_params()
    if params is None: return
    
    # Load scored predictions
    scored_path = PRED_DIR / f'round_{round_num:02d}' / 'scored.csv'
    if not scored_path.exists():
        print(f"  No scored data for R{round_num}. Run f1_learn.py ingest first.")
        return
    
    scored = pd.read_csv(scored_path)
    mae = scored.abs_error.mean()
    tau, _ = sp.kendalltau(scored.pred, scored.actual)
    
    # Load features
    feat_path = ROOT / 'v2_driver_features.csv'
    feat_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
    
    # Load race laps
    import re
    race_laps = None; grid = {}
    for item in (ROOT / 'fastf1_data').iterdir():
        if item.is_dir() and re.match(rf'\d+_R{round_num}_.*_Race', item.name, re.I):
            lp = item / 'laps.csv'
            if lp.exists(): race_laps = pd.read_csv(lp)
    
    # Get grid from GRID_OVERRIDES or scored data
    grid = {r.Driver: int(r.grid) for _, r in scored.iterrows() if pd.notna(r.get('grid'))}
    
    print(f"\n  Current performance: MAE={mae:.2f}, tau={tau:.3f}")
    print(f"  Current params: FP={params['tier_weights']['fp_base']:.0%} "
          f"Hist={params['tier_weights']['history_base']:.0%} "
          f"Grid={params['grid_model']['grid_blend_pct']:.0%}")
    
    # 1. Tune tier weights
    print(f"\n  1. TIER WEIGHTS:")
    params, adj = tune_tier_weights(params, scored, feat_df)
    for k, v in adj.items(): print(f"     {k}: {v}")
    
    # 2. Tune grid model
    print(f"\n  2. GRID MODEL:")
    params, adj = tune_grid_model(params, scored, grid)
    for k, v in adj.items(): print(f"     {k}: {v}")
    
    # 3. Learn circuit profile
    print(f"\n  3. CIRCUIT PROFILE ({circuit_name}):")
    if race_laps is not None:
        params, cp = learn_circuit_profile(params, scored, race_laps, grid, circuit_name, round_num)
        print(f"     overtaking={cp['overtaking_factor']:.2f} sc_prob={cp['safety_car_prob']:.3f} "
              f"deg={cp['tyre_deg_factor']:.2f}")
    else:
        print(f"     No race laps available")
    
    # 4. Tune uncertainty
    print(f"\n  4. UNCERTAINTY:")
    params, adj = tune_uncertainty(params, scored)
    for k, v in adj.items(): print(f"     {k}: {v}")
    
    # 5. Log performance
    print(f"\n  5. PERFORMANCE LOG:")
    params = log_performance(params, round_num, mae, tau)
    
    # Update metadata
    from datetime import datetime
    params['_last_updated'] = datetime.now().isoformat()
    params['_last_round'] = round_num
    params['_version'] = params.get('_version', 0) + 1
    
    save_params(params)
    
    print(f"\n  Updated params: FP={params['tier_weights']['fp_base']:.0%} "
          f"Hist={params['tier_weights']['history_base']:.0%} "
          f"Grid={params['grid_model']['grid_blend_pct']:.0%}")
    print(f"  Saved to {PARAMS_PATH}")
    print(f"  Next run of f1_ml_predict_v2.py will use these params.\n")


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        rnd = int(sys.argv[1])
        circ = sys.argv[2] if len(sys.argv) >= 3 else 'Unknown'
        autotune(rnd, circ)
    else:
        print("  Usage: python f1_autotune.py <round> [circuit_name]")
