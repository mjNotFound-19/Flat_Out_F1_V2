"""
FastF1 Data Pull — dump raw session data to CSV files.

SETUP:
    pip install fastf1

USAGE:
    # Pull preseason testing data (test event 1, session 1)
    python fastf1_pull.py --testing 1 1

    # Pull preseason testing data (test event 1, all 3 sessions)
    python fastf1_pull.py --testing 1

    # Pull race weekend data (round 1, Race session)
    python fastf1_pull.py --round 1 --session R

    # Pull qualifying for round 3
    python fastf1_pull.py --round 3 --session Q

    # Pull all sessions for a race weekend
    python fastf1_pull.py --round 1 --session all

    # Specify a different year
    python fastf1_pull.py --year 2025 --round 5 --session R

    # Include telemetry (large files)
    python fastf1_pull.py --round 1 --session R --telemetry

    # Show the full season schedule
    python fastf1_pull.py --schedule

SESSION IDENTIFIERS:
    FP1, FP2, FP3  — Free Practice 1/2/3
    Q               — Qualifying
    SQ              — Sprint Qualifying / Sprint Shootout
    S               — Sprint
    R               — Race
    all             — every session in the weekend
"""

import argparse
import sys
import os
from pathlib import Path

try:
    import fastf1
except ImportError:
    sys.exit("fastf1 not installed. Run: pip install fastf1")

import pandas as pd


# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_YEAR = 2026
CACHE_DIR = Path("fastf1_cache")
OUTPUT_DIR = Path("fastf1_data")

SESSION_MAP = {
    "FP1": "Practice 1",
    "FP2": "Practice 2",
    "FP3": "Practice 3",
    "Q":   "Qualifying",
    "SQ":  "Sprint Qualifying",
    "S":   "Sprint",
    "R":   "Race",
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def sanitize(name: str) -> str:
    return "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in str(name)).strip()


def save_df(df, path: Path, label: str):
    """Save a DataFrame to CSV if it has data."""
    if df is None or (hasattr(df, "empty") and df.empty) or len(df) == 0:
        print(f"    {label:20s} — no data")
        return
    df.to_csv(path, index=False)
    print(f"    {label:20s} — {len(df):>6,} rows  →  {path.name}")


def export_session(session, out_dir: Path, include_telemetry: bool = False, file_prefix: str = ""):
    """
    Load a session and dump all available data to CSV files.

    Args:
        session:           FastF1 Session object
        out_dir:           Directory to write CSVs into
        include_telemetry: If True, also dump per-driver car/position data
        file_prefix:       Optional prefix for filenames (e.g. "day1_")
    """
    event_name = session.event["EventName"]
    session_name = session.name
    print(f"\n  Loading: {event_name} — {session_name} …")

    try:
        session.load(
            telemetry=include_telemetry,
            laps=True,
            weather=True,
        )
    except Exception as e:
        print(f"  ✗ Failed to load session: {e}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    p = file_prefix  # shorthand

    # ── Laps ──────────────────────────────────────────
    try:
        laps = session.laps
        if laps is not None and not laps.empty:
            save_df(laps, out_dir / f"{p}laps.csv", "Laps")
        else:
            print(f"    {'Laps':20s} — no data")
    except Exception as e:
        print(f"    {'Laps':20s} — error: {e}")

    # ── Results ───────────────────────────────────────
    try:
        results = session.results
        if results is not None and not results.empty:
            save_df(results, out_dir / f"{p}results.csv", "Results")
        else:
            print(f"    {'Results':20s} — no data")
    except Exception as e:
        print(f"    {'Results':20s} — error: {e}")

    # ── Weather ───────────────────────────────────────
    try:
        weather = session.weather_data
        if weather is not None and not weather.empty:
            save_df(weather, out_dir / f"{p}weather.csv", "Weather")
        else:
            print(f"    {'Weather':20s} — no data")
    except Exception as e:
        print(f"    {'Weather':20s} — error: {e}")

    # ── Race Control Messages ─────────────────────────
    try:
        rcm = session.race_control_messages
        if rcm is not None and not rcm.empty:
            save_df(rcm, out_dir / f"{p}race_control.csv", "Race Control")
        else:
            print(f"    {'Race Control':20s} — no data")
    except Exception as e:
        print(f"    {'Race Control':20s} — error: {e}")

    # ── Car Data / Telemetry (per driver) ─────────────
    if include_telemetry:
        try:
            laps = session.laps
            if laps is not None and not laps.empty:
                drivers = laps["Driver"].unique()
                for drv in drivers:
                    try:
                        drv_laps = laps.pick_driver(drv)
                        car_data = drv_laps.get_car_data()
                        if car_data is not None and not car_data.empty:
                            save_df(
                                car_data,
                                out_dir / f"{p}telemetry_{drv}.csv",
                                f"Telemetry [{drv}]",
                            )
                    except Exception:
                        pass

                    try:
                        pos_data = drv_laps.get_pos_data()
                        if pos_data is not None and not pos_data.empty:
                            save_df(
                                pos_data,
                                out_dir / f"{p}position_{drv}.csv",
                                f"Position  [{drv}]",
                            )
                    except Exception:
                        pass
        except Exception as e:
            print(f"    {'Telemetry':20s} — error: {e}")

    print(f"  ✓ Saved to: {out_dir}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Pull F1 data via FastF1 and export to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python fastf1_pull.py --testing 1 1           # Test event 1, session 1
  python fastf1_pull.py --testing 1             # Test event 1, all sessions
  python fastf1_pull.py --round 1 --session R   # Round 1 Race
  python fastf1_pull.py --round 1 --session all # Round 1 all sessions
  python fastf1_pull.py --schedule              # Print season calendar
  python fastf1_pull.py --year 2025 --round 5 --session Q
        """,
    )

    p.add_argument("--year", type=int, default=DEFAULT_YEAR,
                   help=f"Season year (default: {DEFAULT_YEAR})")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--testing", nargs="+", type=int, metavar=("EVENT", "SESSION"),
                      help="Pre-season testing: event number [session number]. "
                           "Omit session number to pull all sessions for that test event.")
    mode.add_argument("--round", type=int, metavar="N",
                      help="Race weekend round number")
    mode.add_argument("--schedule", action="store_true",
                      help="Print the season schedule and exit")

    p.add_argument("--session", default="R",
                   help="Session: FP1, FP2, FP3, Q, SQ, S, R, or 'all' (default: R)")
    p.add_argument("--telemetry", action="store_true",
                   help="Include per-driver telemetry/car data (large files)")
    p.add_argument("--outdir", default="fastf1_data",
                   help="Output directory (default: fastf1_data)")
    p.add_argument("--cache", default="fastf1_cache",
                   help="Cache directory (default: fastf1_cache)")

    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    year = args.year
    OUTPUT_DIR = Path(args.outdir)
    CACHE_DIR = Path(args.cache)

    # Enable cache
    CACHE_DIR.mkdir(exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))

    print(f"\n  FastF1 Data Pull")
    print(f"  Year: {year}")
    print(f"  Cache: {CACHE_DIR.resolve()}\n")

    # ── Schedule mode ─────────────────────────────────
    if args.schedule:
        print(f"  {year} Season Schedule:\n")
        try:
            schedule = fastf1.get_event_schedule(year, include_testing=True)
            cols = ["RoundNumber", "EventName", "Location", "EventDate", "EventFormat"]
            available = [c for c in cols if c in schedule.columns]
            print(schedule[available].to_string(index=False))
        except Exception as e:
            print(f"  ✗ Could not load schedule: {e}")
        return

    # ── Testing mode ──────────────────────────────────
    if args.testing is not None:
        test_event = args.testing[0]
        test_dir = OUTPUT_DIR / "testing"

        if len(args.testing) >= 2:
            # Specific session/day
            test_session = args.testing[1]
            print(f"  Mode: Pre-Season Testing — Event {test_event}, Day {test_session}")
            try:
                session = fastf1.get_testing_session(year, test_event, test_session)
                prefix = f"day{test_session}_"
                export_session(session, test_dir, args.telemetry, file_prefix=prefix)
            except Exception as e:
                print(f"  ✗ Error: {e}")
        else:
            # All sessions for the test event (typically 3 days)
            print(f"  Mode: Pre-Season Testing — Event {test_event}, all days")
            for sess_num in range(1, 4):
                try:
                    session = fastf1.get_testing_session(year, test_event, sess_num)
                    prefix = f"day{sess_num}_"
                    export_session(session, test_dir, args.telemetry, file_prefix=prefix)
                except Exception as e:
                    print(f"\n  ⚠ Day {sess_num}: {e}")
        return

    # ── Race weekend mode ─────────────────────────────
    if args.round is not None:
        rnd = args.round
        session_arg = args.session.upper().strip()

        if session_arg == "ALL":
            identifiers = list(SESSION_MAP.keys())
        else:
            if session_arg not in SESSION_MAP:
                sys.exit(f"  Unknown session '{session_arg}'. Use: {', '.join(SESSION_MAP.keys())} or 'all'")
            identifiers = [session_arg]

        for sid in identifiers:
            full_name = SESSION_MAP[sid]
            print(f"  Mode: Round {rnd} — {full_name}")
            try:
                session = fastf1.get_session(year, rnd, full_name)
                event_name = sanitize(session.event["EventName"])
                label = sanitize(f"{year}_R{rnd}_{event_name}_{sid}")
                out_dir = OUTPUT_DIR / label
                export_session(session, out_dir, args.telemetry)
            except Exception as e:
                print(f"  ⚠ {full_name}: {e}")
        return

    # No mode selected
    print("  No mode selected. Use --testing, --round, or --schedule.")
    print("  Run with --help for usage examples.")


if __name__ == "__main__":
    main()
