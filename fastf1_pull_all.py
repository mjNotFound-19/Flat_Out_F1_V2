r"""
+==============================================================================+
|  FLAT OUT F1 - COMPLETE DATA PULL                                            |
|                                                                              |
|  Pulls everything needed for the v2 prediction pipeline:                    |
|                                                                              |
|  1. 2024 race results (all rounds) - driver performance metrics             |
|  2. 2025 race results (all rounds) - driver performance metrics             |
|  3. 2026 Pre-Season Test 1 (Bahrain) - telemetry                           |
|  4. 2026 Pre-Season Test 2 (Bahrain) - telemetry                           |
|                                                                              |
|  Usage:                                                                      |
|    cd C:\Users\manas\projects\flat_out_f1_v2                                |
|    python fastf1_pull_all.py                                                |
|                                                                              |
|  Output structure:                                                           |
|    fastf1_data/                                                              |
|    C-- historical/                                                           |
|    3   C-- 2024_race_results.csv                                            |
|    3   C-- 2025_race_results.csv                                            |
|    3   C-- all_race_results.csv                                             |
|    3   @-- driver_profiles.transferable driver skill metrics    |
|    C-- Test 1 (already have, re-pulls)     |
|    3   C-- day1_laps.csv ... day3_laps.csv                                  |
|    3   C-- day1_weather.csv ... day3_weather.csv                            |
|    3   @-- day1_results.csv ... day3_results.csv                            |
|    @-- Test 2 (new)                        |
|        C-- day1_laps.csv ... day3_laps.csv                                  |
|        C-- day1_weather.csv ... day3_weather.csv                            |
|        @-- day1_results.csv ... day3_results.csv                            |
+==============================================================================+
"""

import fastf1
import pandas as pd
import numpy as np
from pathlib import Path
import time
import sys

# =============================================================================
#  CONFIG
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CACHE_DIR  = SCRIPT_DIR / 'fastf1_cache'
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

DATA_DIR = SCRIPT_DIR / 'fastf1_data'
DATA_DIR.mkdir(exist_ok=True)

# What to pull (set False to skip if you already have it)
PULL_2024        = True
PULL_2025        = True
PULL_TEST1       = False   # Set True if you want to re-pull Test 1
PULL_TEST2       = True
BUILD_PROFILES   = True


# =============================================================================
#  HELPERS
# =============================================================================

def banner(text):
    print(f"\n{'='*70}")
    print(f"  {text}")
    print(f"{'='*70}")


def safe_get(row, key, default=np.nan):
    """Safely extract a value from a fastf1 results row."""
    try:
        v = row.get(key, default)
        if v is None: return default
        return v
    except:
        return default


# =============================================================================
#  PART 1: HISTORICAL RACE RESULTS
# =============================================================================

def pull_race_results(year):
    """Pull all race results for a season. Returns DataFrame."""
    banner(f"PULLING {year} RACE RESULTS")

    out_dir = DATA_DIR / 'historical'
    out_dir.mkdir(parents=True, exist_ok=True)

    schedule = fastf1.get_event_schedule(year)
    print(f"  Schedule loaded: {len(schedule)} events")

    all_rows = []
    event_count = 0

    for _, event in schedule.iterrows():
        rnd = event['RoundNumber']
        name = event['EventName']
        fmt = event.get('EventFormat', '')

        # Skip pre-season testing
        if rnd == 0 or 'test' in str(name).lower() or fmt == 'testing':
            continue

        event_count += 1
        print(f"\n  R{rnd:02d}: {name} ", end='', flush=True)

        # -- RACE --------------------------------------------------------
        try:
            race = fastf1.get_session(year, rnd, 'R')
            race.load(telemetry=False, weather=True, messages=False)
            res = race.results

            if res is None or len(res) == 0:
                print("!! no results")
                continue

            # Check for rain
            wet = 0
            try:
                wd = race.weather_data
                if wd is not None and 'Rainfall' in wd.columns:
                    wet = 1 if wd['Rainfall'].any() else 0
            except:
                pass

            # Get first-lap positions from lap data if available
            first_lap_pos = {}
            try:
                laps = race.laps
                if laps is not None:
                    lap1 = laps[laps.LapNumber == 1]
                    for _, l in lap1.iterrows():
                        first_lap_pos[l.Driver] = l.get('Position', np.nan)
            except:
                pass

            for _, drv in res.iterrows():
                abbr = safe_get(drv, 'Abbreviation', '')
                gp = safe_get(drv, 'GridPosition')
                fp = safe_get(drv, 'Position')
                status = str(safe_get(drv, 'Status', ''))

                # DNF detection
                dnf_keywords = ['retired', 'accident', 'collision', 'mechanical',
                                'engine', 'gearbox', 'hydraulic', 'electrical',
                                'spin', 'damage', 'puncture', 'brake',
                                'disqualified', 'power', 'suspension', 'wheel',
                                'fire', 'oil', 'water', 'overheating', 'exhaust',
                                'did not finish', 'dnf', 'withdrew', 'dns']
                is_dnf = 1 if any(k in status.lower() for k in dnf_keywords) else 0
                # Also mark as DNF if not classified
                cp = safe_get(drv, 'ClassifiedPosition')
                if pd.isna(cp) or str(cp) == '' or str(cp) == 'R':
                    is_dnf = 1

                # Position delta
                pos_delta = np.nan
                if pd.notna(gp) and pd.notna(fp) and gp > 0 and fp > 0:
                    pos_delta = float(gp) - float(fp)

                # First-lap delta
                fl_delta = np.nan
                if abbr in first_lap_pos and pd.notna(gp) and pd.notna(first_lap_pos[abbr]):
                    fl_delta = float(gp) - float(first_lap_pos[abbr])

                all_rows.append({
                    'year': year,
                    'round': rnd,
                    'event': name,
                    'driver': abbr,
                    'team': safe_get(drv, 'TeamName', ''),
                    'grid': gp,
                    'finish': fp,
                    'points': safe_get(drv, 'Points', 0),
                    'status': status,
                    'dnf': is_dnf,
                    'pos_delta': pos_delta,
                    'first_lap_delta': fl_delta,
                    'fastest_lap_rank': safe_get(drv, 'FastestLapRank'),
                    'wet': wet,
                })

            print(f"OK {len(res)} drivers", end='')
        except Exception as e:
            print(f"FAIL Race error: {e}", end='')

        # -- QUALIFYING --------------------------------------------------
        try:
            quali = fastf1.get_session(year, rnd, 'Q')
            quali.load(telemetry=False, weather=False, messages=False)
            qres = quali.results
            if qres is not None:
                for _, qd in qres.iterrows():
                    abbr = safe_get(qd, 'Abbreviation', '')
                    qpos = safe_get(qd, 'Position')
                    for r in all_rows:
                        if r['year'] == year and r['round'] == rnd and r['driver'] == abbr:
                            r['quali'] = qpos
                            break
                print(f" +Q", end='')
        except:
            pass

        # -- SPRINT (if applicable) --------------------------------------
        try:
            sprint = fastf1.get_session(year, rnd, 'S')
            sprint.load(telemetry=False, weather=False, messages=False)
            sres = sprint.results
            if sres is not None and len(sres) > 0:
                for _, sd in sres.iterrows():
                    abbr = safe_get(sd, 'Abbreviation', '')
                    for r in all_rows:
                        if r['year'] == year and r['round'] == rnd and r['driver'] == abbr:
                            r['sprint_finish'] = safe_get(sd, 'Position')
                            r['sprint_points'] = safe_get(sd, 'Points', 0)
                            break
                print(f" +Sprint", end='')
        except:
            pass

        print()
        time.sleep(0.5)  # Rate limit

    df = pd.DataFrame(all_rows)
    df.to_csv(out_dir / f'{year}_race_results.csv', index=False)
    print(f"\n  OK {year}: {len(df)} rows from {event_count} events  {out_dir}/{year}_race_results.csv")
    return df


# =============================================================================
#  PART 2: DRIVER PROFILES (transferable metrics)
# =============================================================================

def build_driver_profiles(results_df):
    """Build transferable driver performance profiles from race results."""
    banner("BUILDING DRIVER PROFILES")

    profiles = []

    for drv in results_df['driver'].unique():
        dd = results_df[results_df.driver == drv].copy()
        n = len(dd)
        if n < 3:
            continue

        p = {'driver': drv, 'n_races': n}
        p['teams'] = ','.join(dd.team.unique())
        p['last_team'] = dd.iloc[-1]['team']

        # -- Finishing position stats ------------------------------------
        fps = pd.to_numeric(dd['finish'], errors='coerce').dropna()
        if len(fps) > 0:
            p['avg_finish']     = round(fps.mean(), 2)
            p['median_finish']  = round(fps.median(), 1)
            p['std_finish']     = round(fps.std(), 2)
            p['best_finish']    = int(fps.min())
            p['wins']           = int((fps == 1).sum())
            p['podiums']        = int((fps <= 3).sum())
            p['top3_pct']       = round((fps <= 3).mean() * 100, 1)
            p['top5_pct']       = round((fps <= 5).mean() * 100, 1)
            p['top10_pct']      = round((fps <= 10).mean() * 100, 1)
            p['points_total']   = dd['points'].sum()
            p['points_per_race'] = round(dd['points'].sum() / n, 2)

        # -- DNF ---------------------------------------------------------
        p['dnf_count']  = int(dd['dnf'].sum())
        p['dnf_rate']   = round(dd['dnf'].mean() * 100, 1)

        # -- Racecraft: position deltas ----------------------------------
        deltas = pd.to_numeric(dd['pos_delta'], errors='coerce').dropna()
        if len(deltas) > 0:
            p['avg_pos_gained']     = round(deltas.mean(), 2)
            p['median_pos_gained']  = round(deltas.median(), 1)
            p['max_pos_gained']     = int(deltas.max())
            p['pos_gained_std']     = round(deltas.std(), 2)

        # -- First-lap aggression ----------------------------------------
        fl = pd.to_numeric(dd['first_lap_delta'], errors='coerce').dropna()
        if len(fl) > 0:
            p['avg_first_lap_gain'] = round(fl.mean(), 2)
            p['first_lap_std']      = round(fl.std(), 2)

        # -- Qualifying --------------------------------------------------
        if 'quali' in dd.columns:
            qps = pd.to_numeric(dd['quali'], errors='coerce').dropna()
            if len(qps) > 0:
                p['avg_quali']  = round(qps.mean(), 2)
                p['quali_std']  = round(qps.std(), 2)
                p['poles']      = int((qps == 1).sum())
                p['front_row_pct'] = round((qps <= 2).mean() * 100, 1)

                # Quali vs race: positive = does better in race than grid
                gps = pd.to_numeric(dd['grid'], errors='coerce')
                valid_both = dd[gps.notna() & fps.index.isin(dd.index)]
                if len(valid_both) > 3:
                    # Use grid instead of quali since grid = actual starting pos
                    pass

        # -- Wet weather -------------------------------------------------
        wet_races = dd[dd.wet == 1]
        dry_races = dd[dd.wet == 0]
        if len(wet_races) >= 2 and len(dry_races) >= 5:
            wet_fp = pd.to_numeric(wet_races['finish'], errors='coerce').dropna()
            dry_fp = pd.to_numeric(dry_races['finish'], errors='coerce').dropna()
            if len(wet_fp) > 0 and len(dry_fp) > 0:
                p['wet_avg']       = round(wet_fp.mean(), 2)
                p['dry_avg']       = round(dry_fp.mean(), 2)
                p['wet_advantage'] = round(dry_fp.mean() - wet_fp.mean(), 2)

        # -- Teammate head-to-head ---------------------------------------
        tm_wins, tm_total = 0, 0
        tm_quali_wins, tm_quali_total = 0, 0

        for _, race in dd.iterrows():
            teammates = results_df[
                (results_df.year == race['year']) &
                (results_df['round'] == race['round']) &
                (results_df.team == race['team']) &
                (results_df.driver != drv)
            ]
            if len(teammates) == 1:
                tm = teammates.iloc[0]
                # Race head-to-head
                drv_fp = pd.to_numeric(race.get('finish'), errors='coerce')
                tm_fp = pd.to_numeric(tm.get('finish'), errors='coerce')
                if pd.notna(drv_fp) and pd.notna(tm_fp):
                    tm_total += 1
                    if drv_fp < tm_fp:
                        tm_wins += 1

                # Quali head-to-head
                drv_q = pd.to_numeric(race.get('quali'), errors='coerce')
                tm_q = pd.to_numeric(tm.get('quali'), errors='coerce')
                if pd.notna(drv_q) and pd.notna(tm_q):
                    tm_quali_total += 1
                    if drv_q < tm_q:
                        tm_quali_wins += 1

        if tm_total > 0:
            p['tm_race_beat_pct']  = round(tm_wins / tm_total * 100, 1)
            p['tm_race_total']     = tm_total
        if tm_quali_total > 0:
            p['tm_quali_beat_pct'] = round(tm_quali_wins / tm_quali_total * 100, 1)
            p['tm_quali_total']    = tm_quali_total

        # -- Recent form (last 8 races) ----------------------------------
        recent = dd.tail(8)
        rfp = pd.to_numeric(recent['finish'], errors='coerce').dropna()
        if len(rfp) > 0:
            p['recent_avg_finish'] = round(rfp.mean(), 2)
            if 'avg_finish' in p:
                p['form_trend'] = round(p['avg_finish'] - rfp.mean(), 2)  # positive = improving

        # -- Season-level splits -----------------------------------------
        for yr in results_df.year.unique():
            yr_data = dd[dd.year == yr]
            yr_fp = pd.to_numeric(yr_data['finish'], errors='coerce').dropna()
            if len(yr_fp) > 0:
                p[f'avg_finish_{yr}'] = round(yr_fp.mean(), 2)
                p[f'points_{yr}'] = yr_data['points'].sum()

        profiles.append(p)

    profiles_df = pd.DataFrame(profiles).sort_values('avg_finish')

    out_path = DATA_DIR / 'historical' / 'driver_profiles.csv'
    profiles_df.to_csv(out_path, index=False)

    print(f"\n  {len(profiles_df)} driver profiles built")
    print(f"\n  {'DRV':4s} {'AVG':>5s} {'W':>3s} {'POD':>3s} {'T3%':>5s} "
          f"{'TM_R%':>5s} {'TM_Q%':>5s} {'+POS':>5s} {'DNF%':>5s} {'PTS/R':>5s} {'FORM':>5s} {'N':>3s}")
    print(f"  {'-'*65}")

    for _, r in profiles_df.head(30).iterrows():
        print(f"  {r.driver:4s} {r.get('avg_finish',99):5.1f} "
              f"{int(r.get('wins',0)):3d} {int(r.get('podiums',0)):3d} "
              f"{r.get('top3_pct',0):5.1f} "
              f"{r.get('tm_race_beat_pct',50):5.1f} "
              f"{r.get('tm_quali_beat_pct',50):5.1f} "
              f"{r.get('avg_pos_gained',0):+5.1f} "
              f"{r.get('dnf_rate',0):5.1f} "
              f"{r.get('points_per_race',0):5.1f} "
              f"{r.get('form_trend',0):+5.1f} "
              f"{int(r.n_races):3d}")

    return profiles_df


# =============================================================================
#  PART 3: TESTING DATA (2026)
# =============================================================================

def pull_testing(year, test_num, out_folder):
    """Pull pre-season testing telemetry."""
    banner(f"PULLING {year} TEST {test_num}")

    out_dir = DATA_DIR / out_folder
    out_dir.mkdir(parents=True, exist_ok=True)

    # Get schedule to find correct event name
    try:
        schedule = fastf1.get_event_schedule(year)
        testing_events = schedule[
            (schedule['EventFormat'] == 'testing') |
            (schedule['EventName'].str.contains('test', case=False, na=False))
        ]
        print(f"  Testing events found:")
        for _, ev in testing_events.iterrows():
            print(f"    Round {ev['RoundNumber']}: {ev['EventName']}")
    except Exception as e:
        print(f"  !! Schedule error: {e}")
        testing_events = pd.DataFrame()

    # Try to find the right event
    event_identifiers = []

    # From schedule
    if len(testing_events) >= test_num:
        ev = testing_events.iloc[test_num - 1]
        event_identifiers.append(ev['EventName'])
        event_identifiers.append(int(ev['RoundNumber']))

    # Common name patterns
    event_identifiers.extend([
        f'Pre-Season Test {test_num}',
        f'Pre-Season Testing {test_num}',
        f'Pre-Season Test' if test_num == 1 else f'Pre-Season Test {test_num}',
        f'Bahrain Pre-Season Test {test_num}',
        f'Pre-Season Testing',
    ])

    # Session name patterns
    session_patterns = [
        lambda d: f'Day {d}',
        lambda d: f'Session {d}',
        lambda d: d,
    ]

    success = False
    for event_id in event_identifiers:
        if success:
            break
        for day in [1, 2, 3]:
            day_success = False
            for sp in session_patterns:
                sname = sp(day)
                try:
                    print(f"  Trying event='{event_id}', session='{sname}' ... ", end='', flush=True)
                    session = fastf1.get_session(year, event_id, sname)
                    session.load(telemetry=False, weather=True, messages=True)

                    if session.laps is not None and len(session.laps) > 0:
                        session.laps.to_csv(out_dir / f'day{day}_laps.csv', index=False)
                        n_laps = len(session.laps)
                    else:
                        print("no laps")
                        continue

                    if session.weather_data is not None and len(session.weather_data) > 0:
                        session.weather_data.to_csv(out_dir / f'day{day}_weather.csv', index=False)

                    if session.results is not None and len(session.results) > 0:
                        session.results.to_csv(out_dir / f'day{day}_results.csv', index=False)

                    try:
                        if session.race_control_messages is not None:
                            session.race_control_messages.to_csv(
                                out_dir / f'day{day}_race_control.csv', index=False)
                    except:
                        pass

                    print(f"OK {n_laps} laps")
                    day_success = True
                    success = True
                    break

                except Exception as e:
                    print(f"FAIL")
                    continue

            if not day_success and success:
                print(f"  Day {day}: not available yet (session may not have completed)")

    if success:
        print(f"\n  OK Test {test_num} data saved to {out_dir}/")
    else:
        print(f"\n  FAIL Could not find Test {test_num} data.")
        print(f"    The data may not be on the API yet. Try again later.")
        print(f"    You can also check available events with:")
        print(f"      import fastf1")
        print(f"      print(fastf1.get_event_schedule({year}))")

    return out_dir


# =============================================================================
#  MAIN
# =============================================================================

if __name__ == '__main__':
    print("+==================================================================+")
    print("|        FLAT OUT F1 - COMPLETE DATA PULL                        |")
    print("|                                                                  |")
    print("|  This will take 10-20 minutes depending on API speed.           |")
    print("|  Progress is printed as it goes.                                 |")
    print("+==================================================================+")

    all_results = []

    # -- 2024 Season -----------------------------------------------------
    if PULL_2024:
        try:
            df = pull_race_results(2024)
            all_results.append(df)
        except Exception as e:
            print(f"\n  FAIL 2024 pull failed: {e}")

    # -- 2025 Season -----------------------------------------------------
    if PULL_2025:
        try:
            df = pull_race_results(2025)
            all_results.append(df)
        except Exception as e:
            print(f"\n  FAIL 2025 pull failed: {e}")

    # -- Combine + Build Profiles ----------------------------------------
    if all_results and BUILD_PROFILES:
        combined = pd.concat(all_results, ignore_index=True)
        hist_dir = DATA_DIR / 'historical'
        hist_dir.mkdir(parents=True, exist_ok=True)
        combined.to_csv(hist_dir / 'all_race_results.csv', index=False)
        print(f"\n  Combined: {len(combined)} race entries saved")

        profiles = build_driver_profiles(combined)

    # -- 2026 Test 1 ----------------------------------------------------
    if PULL_TEST1:
        try:
            pull_testing(2026, 1, 'testing')
        except Exception as e:
            print(f"\n  FAIL Test 1 pull failed: {e}")

    # -- 2026 Test 2 ----------------------------------------------------
    if PULL_TEST2:
        try:
            pull_testing(2026, 2, 'testing2')
        except Exception as e:
            print(f"\n  FAIL Test 2 pull failed: {e}")

    # -- Summary --------------------------------------------------------
    banner("COMPLETE")

    print(f"\n  All data saved to: {DATA_DIR}/\n")
    print(f"  Files to upload to Claude for prediction pipeline:")
    print(f"  -------------------------------------------------")

    files_to_check = [
        ('historical/driver_profiles.csv',    'Driver performance profiles'),
        ('historical/all_race_results.csv',   'All race results'),
        ('testing/day1_laps.csv',             'Test 1 Day 1 laps'),
        ('testing/day2_laps.csv',             'Test 1 Day 2 laps'),
        ('testing/day3_laps.csv',             'Test 1 Day 3 laps'),
        ('testing/day1_weather.csv',          'Test 1 Day 1 weather'),
        ('testing/day2_weather.csv',          'Test 1 Day 2 weather'),
        ('testing/day3_weather.csv',          'Test 1 Day 3 weather'),
        ('testing2/day1_laps.csv',            'Test 2 Day 1 laps'),
        ('testing2/day2_laps.csv',            'Test 2 Day 2 laps'),
        ('testing2/day3_laps.csv',            'Test 2 Day 3 laps'),
        ('testing2/day1_weather.csv',         'Test 2 Day 1 weather'),
        ('testing2/day2_weather.csv',         'Test 2 Day 2 weather'),
        ('testing2/day3_weather.csv',         'Test 2 Day 3 weather'),
    ]

    for fpath, desc in files_to_check:
        full = DATA_DIR / fpath
        if full.exists():
            size = full.stat().st_size
            print(f"    OK {fpath:40s}  ({size/1024:.0f} KB)  {desc}")
        else:
            print(f"    FAIL {fpath:40s}  MISSING       {desc}")

    print(f"\n  Upload all OK files to Claude to run the prediction pipeline.")
    print(f"  Done! DONE\n")
