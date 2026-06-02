"""
FLAT OUT F1 - HISTORICAL DATASET BUILDER
==========================================
Pulls 2022-2025 race + qualifying data via FastF1 and extracts per-driver
per-race features compatible with V2's feature schema.

Outputs: historical_training_data.csv

Usage:
  python build_historical_dataset.py                    # pull 2022-2025
  python build_historical_dataset.py --years 2024 2025  # subset
  python build_historical_dataset.py --resume           # skip races we already have

Setup:
  pip install fastf1
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats as sp
import warnings
warnings.filterwarnings('ignore')

try:
    import fastf1
except ImportError:
    sys.exit("fastf1 not installed. Run: pip install fastf1")


SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR = SCRIPT_DIR / 'fastf1_cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_PATH = SCRIPT_DIR / 'historical_training_data.csv'


def to_sec(td):
    """Convert pandas Timedelta to seconds."""
    if pd.isna(td): return np.nan
    try:
        return td.total_seconds()
    except AttributeError:
        return np.nan


def extract_features(year, rnd, session_race, session_quali):
    """Extract per-driver features for a single race.

    Schema kept deliberately close to V2's feat_df columns so historical features
    can be merged with current-weekend features during prediction.
    """
    try:
        session_race.load(telemetry=False, laps=True, weather=True)
        session_quali.load(telemetry=False, laps=True, weather=False)
    except Exception as e:
        print(f"    Load failed: {e}")
        return pd.DataFrame()

    race_laps = session_race.laps
    quali_laps = session_quali.laps
    results = session_race.results

    if race_laps is None or race_laps.empty or results is None:
        return pd.DataFrame()

    total_laps = int(race_laps.LapNumber.max()) if len(race_laps) > 0 else 0
    if total_laps == 0:
        return pd.DataFrame()

    # Leader cumulative-time reference (same logic as f1_race_metrics.py)
    race_laps = race_laps.copy()
    race_laps['lt_s'] = race_laps.LapTime.apply(to_sec)
    race_laps['cumtime_s'] = race_laps.Time.apply(to_sec)

    leader_times = {}
    for lap in range(1, total_laps + 1):
        lap_data = race_laps[race_laps.LapNumber == lap]
        valid = lap_data[lap_data.cumtime_s.notna()]
        if len(valid) > 0:
            leader_times[lap] = valid.cumtime_s.min()

    # Quali best time per driver
    quali_laps = quali_laps.copy()
    quali_laps['lt_s'] = quali_laps.LapTime.apply(to_sec)
    quali_best = quali_laps.groupby('Driver').lt_s.min().to_dict()
    pole_time = min(quali_best.values()) if quali_best else np.nan

    # Event metadata
    event = session_race.event
    circuit = event.get('EventName', 'Unknown')
    if hasattr(circuit, 'replace'):
        circuit = circuit.replace('Grand Prix', '').strip()

    # Weather
    weather = session_race.weather_data
    avg_track_temp = weather.TrackTemp.mean() if weather is not None and len(weather) > 0 else np.nan
    avg_air_temp = weather.AirTemp.mean() if weather is not None and len(weather) > 0 else np.nan

    rows = []
    for _, res in results.iterrows():
        drv = res.get('Abbreviation', '')
        if not drv: continue

        dl = race_laps[race_laps.Driver == drv].sort_values('LapNumber')
        if len(dl) < 5:
            continue

        team = res.get('TeamName', 'Unknown')
        finish_pos = res.get('Position', np.nan)
        if pd.isna(finish_pos):
            continue
        finish_pos = int(finish_pos)

        grid_pos = res.get('GridPosition', np.nan)
        grid_pos = int(grid_pos) if pd.notna(grid_pos) and grid_pos > 0 else 20

        status = str(res.get('Status', ''))
        dnf = 'Finished' not in status and '+' not in status
        max_lap = int(dl.LapNumber.max())

        # Clean laps for pace
        clean = dl[(dl['lt_s'] > 60) & (dl['lt_s'] < 200) &
                   dl.PitInTime.isna() & dl.PitOutTime.isna()]

        row = {
            'Year': year, 'Round': rnd, 'Circuit': circuit,
            'Driver': drv, 'Team': team,
            'grid_pos': grid_pos, 'finish_pos': finish_pos,
            'dnf': int(dnf), 'laps_completed': max_lap, 'race_laps': total_laps,
            'track_temp': avg_track_temp, 'air_temp': avg_air_temp,
        }

        # Era indicator
        if year >= 2026: row['era'] = 3  # new 2026 regs
        elif year >= 2022: row['era'] = 2  # ground effect era
        elif year >= 2017: row['era'] = 1  # wide-car era
        else: row['era'] = 0              # narrow-car hybrid

        # --- QUALIFYING FEATURES ---
        q_best = quali_best.get(drv, np.nan)
        row['q_best_time'] = q_best
        row['q_delta_pct'] = (q_best / pole_time * 100) if pd.notna(q_best) and pd.notna(pole_time) and pole_time > 0 else np.nan

        # --- RACE PACE ---
        if len(clean) >= 5:
            row['race_pace_mean'] = clean['lt_s'].mean()
            row['race_pace_median'] = clean['lt_s'].median()
            row['race_pace_std'] = clean['lt_s'].std()
        else:
            row['race_pace_mean'] = row['race_pace_median'] = row['race_pace_std'] = np.nan

        # Best-stint pace (same logic as V2)
        best_stint_pace = np.nan
        best_stint_deg = np.nan
        if 'Stint' in dl.columns:
            for _, sg in clean.groupby(dl.Stint):
                if len(sg) < 5: continue
                stint_clean = sg.iloc[2:] if len(sg) > 4 else sg
                pace = stint_clean['lt_s'].mean()
                if pd.isna(best_stint_pace) or pace < best_stint_pace:
                    best_stint_pace = pace
                    if len(stint_clean) >= 5 and 'TyreLife' in stint_clean.columns:
                        try:
                            best_stint_deg = sp.linregress(
                                stint_clean.TyreLife.values.astype(float),
                                stint_clean['lt_s'].values
                            ).slope
                        except Exception:
                            pass
        row['best_stint_pace'] = best_stint_pace
        row['best_stint_deg'] = best_stint_deg

        # --- GAP-TO-LEADER RATE (V2's strongest feature) ---
        driver_gaps = []
        for _, r in dl.iterrows():
            lap = int(r.LapNumber)
            if pd.notna(r.cumtime_s) and lap in leader_times:
                driver_gaps.append((lap, r.cumtime_s - leader_times[lap]))
        if len(driver_gaps) >= 10:
            xs, ys = zip(*driver_gaps)
            try:
                reg = sp.linregress(xs, ys)
                row['gap_rate'] = reg.slope
                row['gap_r2'] = reg.rvalue ** 2
                row['gap_final'] = driver_gaps[-1][1]
            except Exception:
                row['gap_rate'] = row['gap_r2'] = row['gap_final'] = np.nan
        else:
            row['gap_rate'] = row['gap_r2'] = row['gap_final'] = np.nan

        # --- RACECRAFT ---
        positions = dl[dl.Position.notna()].copy()
        if len(positions) >= 3:
            positions['pos_int'] = positions.Position.astype(int)
            row['start_pos'] = int(positions.pos_int.iloc[0])
            row['end_pos'] = int(positions.pos_int.iloc[-1])
            row['pos_std'] = positions.pos_int.std()
            row['net_gained'] = row['start_pos'] - row['end_pos']
            # Lap 1 craft
            lap3 = positions[positions.LapNumber <= 3]
            row['lap1_gain'] = (row['start_pos'] - int(lap3.pos_int.iloc[-1])) if len(lap3) > 0 else 0
            # Overtake ratio
            pos_diff = positions.pos_int.diff().dropna()
            overtakes = int((pos_diff < 0).sum())
            losses = int((pos_diff > 0).sum())
            row['overtake_ratio'] = overtakes / max(losses, 1)
        else:
            for k in ['start_pos', 'end_pos', 'pos_std', 'net_gained', 'lap1_gain', 'overtake_ratio']:
                row[k] = np.nan

        # --- STRATEGY ---
        pit_laps = dl[dl.PitInTime.notna()].LapNumber.values
        row['n_stops'] = len(pit_laps)
        row['first_pit_lap'] = int(pit_laps[0]) if len(pit_laps) > 0 else np.nan

        if 'Compound' in dl.columns:
            row['n_compounds'] = dl.Compound.dropna().nunique()
        else:
            row['n_compounds'] = np.nan

        rows.append(row)

    race_df = pd.DataFrame(rows)

    # Field-relative features (must be computed after all drivers)
    if 'first_pit_lap' in race_df.columns and race_df.first_pit_lap.notna().any():
        race_df['pit_timing_delta'] = race_df.first_pit_lap - race_df.first_pit_lap.median()

    if 'race_pace_mean' in race_df.columns and race_df.race_pace_mean.notna().any():
        field_median = race_df.race_pace_mean.median()
        race_df['pace_delta_field'] = race_df.race_pace_mean - field_median

    return race_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--years', nargs='+', type=int, default=[2022, 2023, 2024, 2025])
    parser.add_argument('--resume', action='store_true',
                        help='Skip races already present in output CSV')
    parser.add_argument('--output', default=str(OUTPUT_PATH))
    args = parser.parse_args()

    output_path = Path(args.output)

    existing_keys = set()
    existing_df = None
    if args.resume and output_path.exists():
        existing_df = pd.read_csv(output_path)
        existing_keys = set(zip(existing_df.Year, existing_df.Round))
        print(f"  Resuming: {len(existing_keys)} races already extracted")

    all_rows = [existing_df] if existing_df is not None and len(existing_df) > 0 else []
    total_races = 0
    skipped = 0

    for year in args.years:
        print(f"\n{'='*60}")
        print(f"  YEAR {year}")
        print(f"{'='*60}")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"  Schedule fetch failed: {e}")
            continue

        for _, event in schedule.iterrows():
            rnd = event.get('RoundNumber', 0)
            if rnd == 0: continue
            if (year, rnd) in existing_keys:
                skipped += 1
                continue

            gp_name = event.get('EventName', f'Round {rnd}')
            print(f"\n  R{rnd} {gp_name}")

            try:
                race_sess = fastf1.get_session(year, rnd, 'R')
                quali_sess = fastf1.get_session(year, rnd, 'Q')
            except Exception as e:
                print(f"    Session fetch failed: {e}")
                continue

            race_df = extract_features(year, rnd, race_sess, quali_sess)
            if len(race_df) == 0:
                print(f"    No data extracted")
                continue

            all_rows.append(race_df)
            total_races += 1
            print(f"    {len(race_df)} drivers")

            # Save incrementally in case of crash
            if total_races % 5 == 0:
                combined = pd.concat(all_rows, ignore_index=True)
                combined.to_csv(output_path, index=False)

    if not all_rows:
        print("\n  No data extracted!")
        return

    combined = pd.concat(all_rows, ignore_index=True).drop_duplicates(subset=['Year', 'Round', 'Driver'])
    combined.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")
    print(f"  New races: {total_races}")
    print(f"  Skipped (already cached): {skipped}")
    print(f"  Total rows: {len(combined)}")
    print(f"  Unique driver-races: {combined.groupby(['Year','Round','Driver']).ngroups}")
    print(f"  Output: {output_path}")
    print(f"\n  Feature nullity:")
    for col in combined.columns:
        pct = combined[col].isna().mean() * 100
        if pct > 0:
            print(f"    {col:25s} {pct:5.1f}% null")


if __name__ == '__main__':
    main()
