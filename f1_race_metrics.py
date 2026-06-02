"""
F1 Race Metrics Extractor
==========================
Extracts position trace, gap evolution, and racecraft features
from race data. Called by f1_learn.py after each race.
Saves to season_race_data.csv for the prediction pipeline to read.
"""

import pandas as pd, numpy as np
from pathlib import Path
from scipy import stats as sp

def to_sec(td):
    try:
        if pd.isna(td) or td=='': return np.nan
        p=str(td).split(' days ')
        if len(p)==2: h,m,s=p[1].split(':'); return float(h)*3600+float(m)*60+float(s)
    except: pass
    return np.nan


def extract_race_metrics(laps_df, round_num, circuit_name):
    """Extract per-driver race metrics from lap data.
    
    Returns DataFrame with one row per driver containing:
      - Position trace metrics (start, end, best, worst)
      - Gap evolution (rate of gap change to leader = TRUE pace delta)
      - Racecraft (lap 1 gains, overtaking, position volatility)
      - Stint pace (best stint pace, degradation)
    """
    laps = laps_df.copy()
    laps['lt'] = laps['LapTime'].apply(to_sec)
    laps['cumtime'] = laps['Time'].apply(to_sec)
    
    total_laps = int(laps.LapNumber.max())
    
    # Build leader cumulative time reference
    leader_times = {}
    for lap in range(1, total_laps + 1):
        lap_data = laps[laps.LapNumber == lap]
        valid = lap_data[lap_data.cumtime.notna()]
        if len(valid) > 0:
            leader_times[lap] = valid.cumtime.min()
    
    metrics = []
    
    for drv in laps.Driver.unique():
        dl = laps[laps.Driver == drv].sort_values('LapNumber')
        if len(dl) < 5: continue
        
        clean = dl[(dl['lt'] > 60) & (dl['lt'] < 200) & dl.PitInTime.isna() & dl.PitOutTime.isna()]
        positions = dl[dl.Position.notna()].copy()
        positions['pos_int'] = positions.Position.astype(int)
        
        max_lap = int(dl.LapNumber.max())
        team = dl.Team.iloc[0] if 'Team' in dl.columns else 'Unknown'
        
        m = {
            'Driver': drv, 'Team': team, 'Round': round_num, 'Circuit': circuit_name,
            'total_laps': max_lap, 'race_laps': total_laps,
        }
        
        # ── POSITION TRACE ──
        if len(positions) >= 3:
            m['start_pos'] = int(positions.pos_int.iloc[0])
            m['end_pos'] = int(positions.pos_int.iloc[-1])
            m['best_pos'] = int(positions.pos_int.min())
            m['worst_pos'] = int(positions.pos_int.max())
            m['pos_range'] = m['worst_pos'] - m['best_pos']
            
            # Lap 1 craft (positions gained in first 3 laps)
            lap3 = positions[positions.LapNumber <= 3]
            if len(lap3) > 0:
                m['lap1_gain'] = m['start_pos'] - int(lap3.pos_int.iloc[-1])
            else:
                m['lap1_gain'] = 0
            
            # Position volatility (how much they move around)
            m['pos_changes'] = int(abs(positions.pos_int.diff().dropna()).sum())
            m['pos_std'] = positions.pos_int.std()
            
            # Net positions gained
            m['net_gained'] = m['start_pos'] - m['end_pos']
        
        # ── GAP TO LEADER EVOLUTION ──
        # This is the CLEANEST pace signal — removes traffic effects
        driver_gaps = []
        for _, r in dl.iterrows():
            lap = int(r.LapNumber)
            if pd.notna(r.cumtime) and lap in leader_times:
                driver_gaps.append((lap, r.cumtime - leader_times[lap]))
        
        if len(driver_gaps) >= 10:
            xs, ys = zip(*driver_gaps)
            reg = sp.linregress(xs, ys)
            m['gap_rate'] = reg.slope           # seconds per lap lost to leader
            m['gap_r2'] = reg.rvalue ** 2       # how consistent is the gap growth
            m['gap_at_10'] = next((g for l,g in driver_gaps if l >= 10), np.nan)
            m['gap_at_half'] = next((g for l,g in driver_gaps if l >= total_laps//2), np.nan)
            m['gap_final'] = driver_gaps[-1][1]
            
            # Phase analysis: was driver faster in first or second half?
            half = total_laps // 2
            first_half = [(l,g) for l,g in driver_gaps if l <= half]
            second_half = [(l,g) for l,g in driver_gaps if l > half]
            if len(first_half) >= 5 and len(second_half) >= 5:
                fh_rate = sp.linregress(*zip(*first_half)).slope
                sh_rate = sp.linregress(*zip(*second_half)).slope
                m['pace_trend'] = fh_rate - sh_rate  # positive = faster in 2nd half (good deg management)
            
        else:
            m['gap_rate'] = np.nan
        
        # ── CLEAN AIR PACE (best stint only) ──
        best_stint_pace = None
        best_stint_deg = None
        for stint_id, sg in clean.groupby(dl[dl.Driver==drv].Stint):
            if len(sg) < 5: continue
            # Skip first 2 laps (outlap, cold tyres)
            stint_clean = sg.iloc[2:] if len(sg) > 4 else sg
            pace = stint_clean['lt'].mean()
            if best_stint_pace is None or pace < best_stint_pace:
                best_stint_pace = pace
                if len(stint_clean) >= 5:
                    best_stint_deg = sp.linregress(
                        stint_clean.TyreLife.values.astype(float),
                        stint_clean['lt'].values
                    ).slope
        
        m['best_stint_pace'] = best_stint_pace
        m['best_stint_deg'] = best_stint_deg
        
        # ── OVERTAKING ──
        if len(positions) >= 5:
            pos_diff = positions.pos_int.diff().dropna()
            m['overtakes_made'] = int((pos_diff < 0).sum())   # gained positions
            m['positions_lost'] = int((pos_diff > 0).sum())   # lost positions
            m['overtake_ratio'] = m['overtakes_made'] / max(m['positions_lost'], 1)

        # ── STRATEGY FEATURES ──
        # Pit timing and undercut effectiveness — separates racecraft from pace
        pit_in_laps = dl[dl.PitInTime.notna()].LapNumber.values
        m['n_stops'] = len(pit_in_laps)

        if len(pit_in_laps) > 0:
            # First pit stop lap
            first_pit = int(pit_in_laps[0])
            m['first_pit_lap'] = first_pit

            # Undercut/overcut effectiveness at first stop:
            # Position 1 lap before pit vs 2 laps after pit (time to rejoin cleanly)
            pos_before = dl[dl.LapNumber == first_pit - 1]
            pos_after = dl[dl.LapNumber == first_pit + 2]
            if len(pos_before) > 0 and len(pos_after) > 0:
                pb = pos_before.Position.iloc[0]
                pa = pos_after.Position.iloc[0]
                if pd.notna(pb) and pd.notna(pa):
                    m['undercut_gain'] = int(pb) - int(pa)  # +ve = gained positions via pit strategy
        else:
            m['first_pit_lap'] = np.nan
            m['undercut_gain'] = 0

        # Compound strategy: did driver use a non-default compound mix?
        if 'Compound' in dl.columns:
            compounds = dl.Compound.dropna().unique()
            m['n_compounds'] = len(compounds)
        else:
            m['n_compounds'] = np.nan

        # Strategy deviation from field:
        # How different was this driver's first-pit timing from field median?
        # Computed at field level below, filled in post-loop

        metrics.append(m)

    metrics_df = pd.DataFrame(metrics)

    # Fill in field-relative strategy feature (needs all drivers computed first)
    if 'first_pit_lap' in metrics_df.columns:
        field_median_pit = metrics_df.first_pit_lap.median()
        metrics_df['pit_timing_delta'] = metrics_df.first_pit_lap - field_median_pit
        # Convention: negative = pitted earlier (aggressive), positive = later (extending stint)

    return metrics_df


def save_season_data(metrics_df, root_path):
    """Append race metrics to season_race_data.csv"""
    season_path = root_path / 'season_race_data.csv'
    
    if season_path.exists():
        existing = pd.read_csv(season_path)
        # Remove old data for this round (in case of re-ingestion)
        rnd = metrics_df.Round.iloc[0]
        existing = existing[existing.Round != rnd]
        combined = pd.concat([existing, metrics_df], ignore_index=True)
    else:
        combined = metrics_df
    
    combined.to_csv(season_path, index=False)
    return combined


def compute_season_features(season_data):
    """Compute rolling season features from accumulated race data.
    Returns per-driver features for the prediction pipeline."""
    
    features = {}
    for drv in season_data.Driver.unique():
        dd = season_data[season_data.Driver == drv].sort_values('Round')
        if len(dd) == 0: continue
        
        f = {}
        # Gap rate: weighted average (recent races weighted more)
        gaps = dd[dd.gap_rate.notna()]
        if len(gaps) > 0:
            weights = np.array([0.5 ** (len(gaps) - 1 - i) for i in range(len(gaps))])
            weights /= weights.sum()
            f['season_gap_rate'] = np.average(gaps.gap_rate.values, weights=weights)
        
        # Best stint pace delta (normalized across races via gap_rate)
        f['season_best_stint_deg'] = dd.best_stint_deg.dropna().mean()
        
        # Racecraft
        f['season_lap1_gain'] = dd.lap1_gain.mean() if 'lap1_gain' in dd.columns else 0
        f['season_net_gained'] = dd.net_gained.mean() if 'net_gained' in dd.columns else 0
        f['season_overtake_ratio'] = dd.overtake_ratio.mean() if 'overtake_ratio' in dd.columns else 1.0
        f['season_pos_volatility'] = dd.pos_std.mean() if 'pos_std' in dd.columns else 3.0
        
        # Pace trend (positive = better in 2nd half = good tyre management)
        if 'pace_trend' in dd.columns:
            f['season_pace_trend'] = dd.pace_trend.dropna().mean()

        # ── STRATEGY FEATURES (aggregated across races) ──
        # Undercut gain: how often does the driver's team nail the first pit stop?
        if 'undercut_gain' in dd.columns:
            ug = dd.undercut_gain.dropna()
            if len(ug) > 0:
                f['season_undercut_gain'] = ug.mean()
                # Consistency: low std = team executes strategy reliably
                f['season_undercut_std'] = ug.std() if len(ug) >= 2 else 0

        # Pit timing pattern: does the team pit early, late, or with the field?
        if 'pit_timing_delta' in dd.columns:
            ptd = dd.pit_timing_delta.dropna()
            if len(ptd) > 0:
                f['season_pit_timing'] = ptd.mean()  # negative = aggressive undercut approach

        # Number of stops (strategic flexibility)
        if 'n_stops' in dd.columns:
            ns = dd.n_stops.dropna()
            if len(ns) > 0:
                f['season_avg_stops'] = ns.mean()
        
        features[drv] = f
    
    return pd.DataFrame.from_dict(features, orient='index').reset_index().rename(columns={'index': 'Driver'})


if __name__ == '__main__':
    import sys
    if len(sys.argv) >= 3:
        laps_path = sys.argv[1]
        round_num = int(sys.argv[2])
        circuit = sys.argv[3] if len(sys.argv) >= 4 else 'Unknown'
        
        laps = pd.read_csv(laps_path)
        metrics = extract_race_metrics(laps, round_num, circuit)
        
        print(f"\nRace metrics for {len(metrics)} drivers:")
        print(f"\n{'DRV':4s} {'START':>5s} {'END':>3s} {'L1':>3s} {'GAP_RATE':>9s} {'DEG':>8s} {'OT':>3s}")
        print(f"{'-'*40}")
        for _, r in metrics.sort_values('end_pos').iterrows():
            gr = f"{r.gap_rate:+.3f}" if pd.notna(r.get('gap_rate')) else '  N/A'
            dg = f"{r.best_stint_deg:+.3f}" if pd.notna(r.get('best_stint_deg')) else '  N/A'
            ot = int(r.get('overtakes_made', 0))
            l1 = int(r.get('lap1_gain', 0))
            print(f"  {r.Driver:4s}  P{int(r.start_pos):>2d}  P{int(r.end_pos):>2d}  {l1:+2d}  {gr:>9s}  {dg:>8s}  {ot:>3d}")
    else:
        print("Usage: python f1_race_metrics.py <laps.csv> <round> [circuit]")
