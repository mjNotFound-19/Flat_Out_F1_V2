"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  FLAT OUT F1 — HISTORICAL DATA PULL                                         ║
║                                                                              ║
║  Pulls DRIVER PERFORMANCE data (not lap times) from 2024-2025               ║
║  because 2026 reg change makes absolute pace non-transferable.              ║
║                                                                              ║
║  What transfers across regulations:                                          ║
║    ✓ Finishing position (relative performance)                               ║
║    ✓ Teammate delta (positions, not seconds)                                 ║
║    ✓ Grid → Race position delta (racecraft / overtaking)                    ║
║    ✓ Points scored                                                           ║
║    ✓ DNF rate                                                                ║
║    ✓ Consistency (std dev of finishing positions)                            ║
║    ✓ Wet weather results                                                     ║
║    ✓ First-lap position changes                                              ║
║                                                                              ║
║  Also pulls 2026 Test 2 telemetry data.                                     ║
║                                                                              ║
║  Usage:                                                                      ║
║    python fastf1_pull_history.py                                            ║
║                                                                              ║
║  Requirements:                                                               ║
║    pip install fastf1                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import json
import time

# ═════════════════════════════════════════════════════════════════════════════
#  CONFIG
# ═════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR  = SCRIPT_DIR / 'fastf1_cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

DATA_DIR = SCRIPT_DIR / 'fastf1_data'
DATA_DIR.mkdir(exist_ok=True)

# Seasons to pull driver performance from
SEASONS = [2024, 2025]

# 2026 Test 2 session info (update if needed)
TEST2_YEAR = 2026
TEST2_EVENT = 'Pre-Season Testing'  # May need adjustment based on fastf1 event name
TEST2_DAYS = [1, 2, 3]  # Update as days complete


# ═════════════════════════════════════════════════════════════════════════════
#  PART 1: PULL HISTORICAL DRIVER PERFORMANCE (2024-2025)
# ═════════════════════════════════════════════════════════════════════════════

def pull_season_results(year):
    """Pull race results for an entire season — driver performance only."""
    print(f"\n{'═'*60}")
    print(f"  PULLING {year} SEASON RESULTS")
    print(f"{'═'*60}")

    season_dir = DATA_DIR / 'historical' / str(year)
    season_dir.mkdir(parents=True, exist_ok=True)

    schedule = fastf1.get_event_schedule(year)
    # Filter to actual races (not testing/pre-season)
    races = schedule[schedule['EventFormat'].isin(['conventional', 'sprint_shootout',
                                                     'sprint_qualifying', 'sprint'])]
    # Also try standard race filter
    if len(races) == 0:
        races = schedule[schedule['EventFormat'] != 'testing']

    print(f"  Found {len(races)} events")

    all_results = []

    for _, event in races.iterrows():
        event_name = event['EventName']
        round_num = event['RoundNumber']

        if round_num == 0:  # Skip testing
            continue

        print(f"\n  Round {round_num}: {event_name}")

        try:
            session = fastf1.get_session(year, round_num, 'R')
            session.load(telemetry=False, weather=True, messages=True)

            results = session.results
            if results is None or len(results) == 0:
                print(f"    ⚠ No results available, skipping")
                continue

            # Extract driver performance data
            for _, drv in results.iterrows():
                row = {
                    'year': year,
                    'round': round_num,
                    'event': event_name,
                    'driver': drv.get('Abbreviation', ''),
                    'team': drv.get('TeamName', ''),
                    'grid_position': drv.get('GridPosition', np.nan),
                    'finish_position': drv.get('Position', np.nan),
                    'classified_position': drv.get('ClassifiedPosition', np.nan),
                    'points': drv.get('Points', 0),
                    'status': drv.get('Status', ''),
                    'time': str(drv.get('Time', '')),
                    'fastest_lap_rank': drv.get('FastestLapRank', np.nan),
                }

                # Compute grid-to-race delta
                gp = row['grid_position']
                fp = row['finish_position']
                if pd.notna(gp) and pd.notna(fp) and gp > 0 and fp > 0:
                    row['position_delta'] = gp - fp  # positive = gained positions
                else:
                    row['position_delta'] = np.nan

                # DNF flag
                status = str(row['status']).lower()
                row['dnf'] = 1 if any(x in status for x in
                    ['retired', 'accident', 'collision', 'mechanical',
                     'engine', 'gearbox', 'hydraulic', 'electrical',
                     'spin', 'damage', 'puncture', 'brake', 'disqualified']) else 0

                all_results.append(row)

            print(f"    ✓ {len(results)} drivers")

            # Also try to get qualifying results
            try:
                quali = fastf1.get_session(year, round_num, 'Q')
                quali.load(telemetry=False, weather=False, messages=False)
                q_results = quali.results
                if q_results is not None and len(q_results) > 0:
                    for _, qd in q_results.iterrows():
                        # Find matching race result and add quali position
                        for res in all_results:
                            if (res['year'] == year and res['round'] == round_num
                                and res['driver'] == qd.get('Abbreviation', '')):
                                res['quali_position'] = qd.get('Position', np.nan)
                                break
            except Exception as e:
                print(f"    ⚠ Qualifying data unavailable: {e}")

            # Check if it was a wet race (from weather data)
            try:
                weather_data = session.weather_data
                if weather_data is not None and 'Rainfall' in weather_data.columns:
                    any_rain = weather_data['Rainfall'].any()
                    for res in all_results:
                        if res['year'] == year and res['round'] == round_num:
                            res['wet_race'] = 1 if any_rain else 0
            except:
                pass

        except Exception as e:
            print(f"    ✗ Error: {e}")
            continue

        # Be nice to the API
        time.sleep(1)

    # Save
    df = pd.DataFrame(all_results)
    df.to_csv(season_dir / f'{year}_race_results.csv', index=False)
    print(f"\n  ✓ Saved {len(df)} results to {season_dir}/{year}_race_results.csv")

    return df


def compute_driver_profiles(all_results_df):
    """
    Compute transferable driver performance metrics from historical results.
    These are the features that survive a regulation change.
    """
    print(f"\n{'═'*60}")
    print(f"  COMPUTING DRIVER PERFORMANCE PROFILES")
    print(f"{'═'*60}")

    profiles = []

    for drv in all_results_df['driver'].unique():
        dd = all_results_df[all_results_df.driver == drv]

        # Basics
        n_races = len(dd)
        if n_races < 3:
            continue

        p = {'driver': drv}
        p['team_last'] = dd.iloc[-1]['team']
        p['n_races'] = n_races

        # Finishing position stats
        fps = dd['finish_position'].dropna()
        if len(fps) > 0:
            p['avg_finish'] = fps.mean()
            p['median_finish'] = fps.median()
            p['std_finish'] = fps.std()
            p['best_finish'] = fps.min()
            p['worst_finish'] = fps.max()
            p['top3_pct'] = (fps <= 3).sum() / len(fps) * 100
            p['top5_pct'] = (fps <= 5).sum() / len(fps) * 100
            p['top10_pct'] = (fps <= 10).sum() / len(fps) * 100
            p['points_per_race'] = dd['points'].sum() / n_races

        # DNF rate
        p['dnf_rate'] = dd['dnf'].mean() * 100

        # Racecraft: grid → finish position delta
        deltas = dd['position_delta'].dropna()
        if len(deltas) > 0:
            p['avg_pos_gained'] = deltas.mean()
            p['median_pos_gained'] = deltas.median()
            p['max_pos_gained'] = deltas.max()
            p['overtake_consistency'] = deltas.std()

        # Qualifying performance
        if 'quali_position' in dd.columns:
            qps = dd['quali_position'].dropna()
            if len(qps) > 0:
                p['avg_quali'] = qps.mean()
                p['quali_std'] = qps.std()
                p['poles'] = (qps == 1).sum()

        # Wet weather performance (if data available)
        if 'wet_race' in dd.columns:
            wet = dd[dd.wet_race == 1]
            dry = dd[dd.wet_race == 0]
            if len(wet) >= 2 and len(dry) >= 2:
                p['wet_avg_finish'] = wet['finish_position'].dropna().mean()
                p['dry_avg_finish'] = dry['finish_position'].dropna().mean()
                p['wet_advantage'] = p['dry_avg_finish'] - p['wet_avg_finish']
            else:
                p['wet_avg_finish'] = p['dry_avg_finish'] = p['wet_advantage'] = np.nan

        # Teammate comparison (same team races)
        teammate_wins = 0
        teammate_total = 0
        for _, race in dd.iterrows():
            tm = all_results_df[
                (all_results_df.year == race['year']) &
                (all_results_df['round'] == race['round']) &
                (all_results_df.team == race['team']) &
                (all_results_df.driver != drv)
            ]
            if len(tm) == 1:
                tm_fp = tm.iloc[0]['finish_position']
                drv_fp = race['finish_position']
                if pd.notna(tm_fp) and pd.notna(drv_fp):
                    teammate_total += 1
                    if drv_fp < tm_fp:
                        teammate_wins += 1

        if teammate_total > 0:
            p['tm_beat_rate'] = teammate_wins / teammate_total * 100
            p['tm_comparisons'] = teammate_total
        else:
            p['tm_beat_rate'] = p['tm_comparisons'] = np.nan

        # Recent form: last 5 races weighted more
        recent = dd.tail(5)
        rfp = recent['finish_position'].dropna()
        if len(rfp) > 0:
            p['recent_avg_finish'] = rfp.mean()
            p['recent_trend'] = p.get('avg_finish', 10) - rfp.mean()  # positive = improving

        # Consistency: rolling 3-race average std
        if len(fps) >= 6:
            rolling_std = fps.rolling(3).std().dropna()
            p['consistency_score'] = rolling_std.mean()
        else:
            p['consistency_score'] = fps.std() if len(fps) > 1 else np.nan

        profiles.append(p)

    profiles_df = pd.DataFrame(profiles)

    # Save
    out_path = DATA_DIR / 'historical' / 'driver_profiles.csv'
    profiles_df.to_csv(out_path, index=False)
    print(f"\n  ✓ {len(profiles_df)} driver profiles saved to {out_path}")

    # Print summary
    print(f"\n  Driver profiles (sorted by avg finish):")
    for _, r in profiles_df.sort_values('avg_finish').iterrows():
        print(f"    {r.driver:4s}  avg_fin: {r.avg_finish:5.1f}  "
              f"top3: {r.top3_pct:5.1f}%  "
              f"tm_beat: {r.tm_beat_rate:5.1f}%  "
              f"pos_gained: {r.avg_pos_gained:+.1f}  "
              f"dnf: {r.dnf_rate:4.1f}%  "
              f"({int(r.n_races)} races)")

    return profiles_df


# ═════════════════════════════════════════════════════════════════════════════
#  PART 2: PULL 2026 TEST 2 TELEMETRY
# ═════════════════════════════════════════════════════════════════════════════

def pull_test2_data():
    """Pull 2026 Pre-Season Test 2 telemetry data."""
    print(f"\n{'═'*60}")
    print(f"  PULLING 2026 TEST 2 DATA")
    print(f"{'═'*60}")

    test2_dir = DATA_DIR / 'testing2'
    test2_dir.mkdir(parents=True, exist_ok=True)

    # First, list available 2026 events
    try:
        schedule = fastf1.get_event_schedule(2026)
        print(f"\n  Available 2026 events:")
        for _, ev in schedule.iterrows():
            print(f"    Round {ev['RoundNumber']}: {ev['EventName']} ({ev['EventFormat']})")
    except Exception as e:
        print(f"  ⚠ Could not fetch 2026 schedule: {e}")
        print(f"  Trying to load testing sessions directly...")

    # Try loading Test 2 sessions
    # fastf1 names may vary — try common patterns
    test_names = [
        'Pre-Season Test 2',
        'Pre-Season Testing 2',
        'Pre-Season Test',
        'Bahrain Pre-Season Test 2',
        2,  # Sometimes test events are indexed
    ]

    for test_name in test_names:
        for day in TEST2_DAYS:
            session_names = [f'Day {day}', f'Session {day}', day]
            for sname in session_names:
                try:
                    print(f"\n  Trying: event='{test_name}', session='{sname}'")
                    session = fastf1.get_session(2026, test_name, sname)
                    session.load(telemetry=False, weather=True, messages=True)

                    # Save laps
                    if session.laps is not None and len(session.laps) > 0:
                        session.laps.to_csv(test2_dir / f'day{day}_laps.csv', index=False)
                        print(f"    ✓ Day {day} laps: {len(session.laps)} rows")

                    # Save weather
                    if session.weather_data is not None and len(session.weather_data) > 0:
                        session.weather_data.to_csv(test2_dir / f'day{day}_weather.csv', index=False)
                        print(f"    ✓ Day {day} weather: {len(session.weather_data)} rows")

                    # Save results
                    if session.results is not None and len(session.results) > 0:
                        session.results.to_csv(test2_dir / f'day{day}_results.csv', index=False)
                        print(f"    ✓ Day {day} results: {len(session.results)} rows")

                    # Save race control
                    if session.race_control_messages is not None:
                        session.race_control_messages.to_csv(
                            test2_dir / f'day{day}_race_control.csv', index=False)

                    break  # Session found, move to next day
                except Exception as e:
                    continue
            else:
                continue
            break  # Found the right session name pattern

    print(f"\n  Test 2 data saved to: {test2_dir}")
    return test2_dir


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  FLAT OUT F1 — HISTORICAL + TEST 2 DATA PULL               ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Step 1: Pull historical race results
    all_results = []
    for year in SEASONS:
        try:
            df = pull_season_results(year)
            all_results.append(df)
        except Exception as e:
            print(f"\n  ✗ Failed to pull {year}: {e}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(DATA_DIR / 'historical' / 'all_race_results.csv', index=False)
        print(f"\n  Combined results: {len(combined)} rows saved")

        # Step 2: Compute driver performance profiles
        profiles = compute_driver_profiles(combined)
    else:
        print("\n  ⚠ No historical data pulled, skipping profiles")

    # Step 3: Pull Test 2 data
    try:
        pull_test2_data()
    except Exception as e:
        print(f"\n  ✗ Test 2 pull failed: {e}")
        print(f"  This might mean the session hasn't been uploaded to the F1 API yet.")
        print(f"  Try again in a few hours after the session ends.")

    print(f"\n{'═'*60}")
    print(f"  ALL DONE!")
    print(f"{'═'*60}")
    print(f"\n  Files saved to: {DATA_DIR}/")
    print(f"  Structure:")
    print(f"    fastf1_data/")
    print(f"    ├── historical/")
    print(f"    │   ├── 2024/2024_race_results.csv")
    print(f"    │   ├── 2025/2025_race_results.csv")
    print(f"    │   ├── all_race_results.csv")
    print(f"    │   └── driver_profiles.csv     ← USE THIS IN v2 PIPELINE")
    print(f"    ├── testing/                     ← Test 1 (already have)")
    print(f"    └── testing2/                    ← Test 2 (just pulled)")
    print(f"        ├── day1_laps.csv")
    print(f"        ├── day1_weather.csv")
    print(f"        └── ...")
