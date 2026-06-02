#!/usr/bin/env python3
"""
+==================================================================+
|              FLAT OUT F1 v2 - TELEMETRY COLLECTOR               |
|         Background Service for Live & Historical F1 Data        |
+==================================================================+

Collects telemetry data from F1 sessions via:
  1. OpenF1 API   - car data, laps, stints, weather, positions (free historical)
  2. FastF1       - SignalR live recording + post-session enrichment

Runs as a background daemon whenever your laptop is on.
Only collects data when an F1 session is active or recently finished.

Data is stored in a format compatible with the Flat Out F1 v2 prediction pipeline.

Usage:
    python f1_collector.py              # Run in foreground
    python f1_collector.py --daemon     # Run as background daemon
    python f1_collector.py --status     # Check collector status
    python f1_collector.py --backfill   # Backfill missed sessions from OpenF1

Author: Flat Out F1 v2 Pipeline
"""

import os
import sys
import json
import time
import signal
import logging
import hashlib
import argparse
import threading
import subprocess
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class CollectorConfig:
    """All tuneable knobs in one place."""

    # -- Directories --
    base_dir: str = os.path.expanduser("~/f1_telemetry_data")
    raw_dir: str = ""          # set in __post_init__
    processed_dir: str = ""    # set in __post_init__
    cache_dir: str = ""        # set in __post_init__
    log_dir: str = ""          # set in __post_init__
    live_dir: str = ""         # set in __post_init__

    # -- OpenF1 API --
    openf1_base: str = "https://api.openf1.org/v1"
    openf1_rate_limit: float = 0.35       # seconds between requests (~3 req/s)
    openf1_batch_size: int = 500           # max rows per paginated fetch

    # -- Schedule polling --
    schedule_check_interval: int = 300     # seconds between schedule checks (5 min)
    pre_session_lead: int = 180            # start collecting 3 min before session
    post_session_linger: int = 1800        # keep collecting 30 min after session ends
    idle_poll_interval: int = 600          # poll every 10 min when no session near

    # -- FastF1 live recording --
    signalr_timeout: int = 7200            # 2 hours (server disconnects around here)
    signalr_overlap: int = 300             # start 2nd recording 5 min before timeout

    # -- Data channels to collect from OpenF1 --
    openf1_endpoints: list = field(default_factory=lambda: [
        "car_data",
        "laps",
        "stints",
        "weather",
        "position",
        "intervals",
        "pit",
        "race_control",
        "drivers",
        "location",
    ])

    # -- Daemon --
    pid_file: str = ""         # set in __post_init__

    def __post_init__(self):
        self.raw_dir = os.path.join(self.base_dir, "raw")
        self.processed_dir = os.path.join(self.base_dir, "processed")
        self.cache_dir = os.path.join(self.base_dir, "cache")
        self.log_dir = os.path.join(self.base_dir, "logs")
        self.live_dir = os.path.join(self.base_dir, "live_recordings")
        self.pid_file = os.path.join(self.base_dir, ".collector.pid")

        for d in [self.base_dir, self.raw_dir, self.processed_dir,
                  self.cache_dir, self.log_dir, self.live_dir]:
            os.makedirs(d, exist_ok=True)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(config: CollectorConfig) -> logging.Logger:
    logger = logging.getLogger("f1_collector")
    logger.setLevel(logging.DEBUG)

    # File handler - rotate daily via naming
    log_file = os.path.join(
        config.log_dir,
        f"collector_{datetime.now().strftime('%Y%m%d')}.log"
    )
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)

    # Console handler - force utf-8 on Windows to avoid cp1252 emoji crashes
    import io
    if sys.platform == "win32":
        ch = logging.StreamHandler(
            stream=io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        )
    else:
        ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


# ---------------------------------------------------------------------------
# HTTP helpers (stdlib only - no requests dependency required)
# ---------------------------------------------------------------------------

import urllib.request
import urllib.error
import urllib.parse

def api_get(url: str, timeout: int = 30) -> Optional[Any]:
    """GET JSON from a URL. Returns parsed JSON or None on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "FlatOutF1v2/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, TimeoutError) as exc:
        logging.getLogger("f1_collector").warning(f"API GET failed: {url} - {exc}")
        return None


# ---------------------------------------------------------------------------
# Session state tracking
# ---------------------------------------------------------------------------

class SessionState(Enum):
    IDLE = "idle"                     # No session near
    PRE_SESSION = "pre_session"       # Session starting soon
    LIVE = "live"                     # Session in progress
    POST_SESSION = "post_session"     # Session recently ended
    COLLECTING = "collecting"         # Actively pulling data


@dataclass
class SessionInfo:
    """Represents a detected F1 session."""
    session_key: int
    meeting_key: int
    session_name: str          # e.g. "Race", "Qualifying", "Practice 1"
    session_type: str          # e.g. "Race", "Qualifying", "Practice"
    circuit_short_name: str
    country_name: str
    date_start: datetime
    date_end: Optional[datetime] = None
    year: int = 0
    gmt_offset: str = ""
    collected: bool = False

    @property
    def safe_name(self) -> str:
        """Filesystem-safe session identifier."""
        return (
            f"{self.year}_{self.meeting_key}_{self.circuit_short_name}_"
            f"{self.session_name.replace(' ', '_')}"
        ).lower()


# ---------------------------------------------------------------------------
# Schedule Manager - detects upcoming / active sessions
# ---------------------------------------------------------------------------

class ScheduleManager:
    """Polls OpenF1 for session schedule and determines what to collect."""

    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.log = logger
        self._known_sessions: Dict[int, SessionInfo] = {}
        self._last_check = datetime.min.replace(tzinfo=timezone.utc)

    def refresh(self) -> None:
        """Fetch latest session schedule from OpenF1."""
        now = datetime.now(timezone.utc)
        if (now - self._last_check).total_seconds() < self.config.schedule_check_interval:
            return

        self.log.debug("Refreshing session schedule from OpenF1...")
        year = now.year

        # Get meetings for current year
        meetings = api_get(f"{self.config.openf1_base}/meetings?year={year}")
        if not meetings:
            self.log.warning("Could not fetch meetings schedule")
            return

        for meeting in meetings:
            mk = meeting.get("meeting_key")
            # Get sessions for this meeting
            sessions = api_get(
                f"{self.config.openf1_base}/sessions?meeting_key={mk}"
            )
            if not sessions:
                continue

            for s in sessions:
                sk = s.get("session_key")
                if sk in self._known_sessions:
                    continue

                date_start_str = s.get("date_start", "")
                if not date_start_str:
                    continue

                try:
                    # Parse ISO format
                    ds = datetime.fromisoformat(
                        date_start_str.replace("Z", "+00:00")
                    )
                except ValueError:
                    continue

                date_end_str = s.get("date_end", "")
                de = None
                if date_end_str:
                    try:
                        de = datetime.fromisoformat(
                            date_end_str.replace("Z", "+00:00")
                        )
                    except ValueError:
                        pass

                info = SessionInfo(
                    session_key=sk,
                    meeting_key=mk,
                    session_name=s.get("session_name", "Unknown"),
                    session_type=s.get("session_type", "Unknown"),
                    circuit_short_name=s.get("circuit_short_name", "unknown"),
                    country_name=s.get("country_name", "Unknown"),
                    date_start=ds,
                    date_end=de,
                    year=year,
                    gmt_offset=s.get("gmt_offset", ""),
                )
                self._known_sessions[sk] = info
                self.log.info(
                    f"Discovered session: {info.session_name} @ "
                    f"{info.circuit_short_name} ({info.date_start})"
                )

        self._last_check = now

    def get_active_session(self) -> Optional[SessionInfo]:
        """Return the session that is currently live or about to start."""
        self.refresh()
        now = datetime.now(timezone.utc)
        lead = timedelta(seconds=self.config.pre_session_lead)
        linger = timedelta(seconds=self.config.post_session_linger)

        for info in sorted(self._known_sessions.values(),
                           key=lambda s: s.date_start):
            start = info.date_start - lead
            # Estimate end if not known (3h for race, 1.5h for others)
            if info.date_end:
                end = info.date_end + linger
            elif "race" in info.session_type.lower():
                end = info.date_start + timedelta(hours=3) + linger
            else:
                end = info.date_start + timedelta(hours=1, minutes=30) + linger

            if start <= now <= end:
                return info

        return None

    def get_uncollected_past_sessions(self, lookback_days: int = 7) -> List[SessionInfo]:
        """Sessions that ended recently but haven't been collected yet."""
        self.refresh()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)
        result = []
        for info in self._known_sessions.values():
            end = info.date_end or (info.date_start + timedelta(hours=3))
            if cutoff <= end <= now and not info.collected:
                result.append(info)
        return result


# ---------------------------------------------------------------------------
# OpenF1 Data Collector - pulls historical / post-session data
# ---------------------------------------------------------------------------

class OpenF1Collector:
    """Fetches data from the OpenF1 REST API for a given session."""

    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.log = logger

    def collect_session(self, session: SessionInfo) -> Dict[str, Path]:
        """
        Pull all configured endpoints for a session.
        Returns dict of endpoint -> saved file path.
        """
        self.log.info(
            f"[START] Collecting OpenF1 data for {session.session_name} "
            f"@ {session.circuit_short_name} (key={session.session_key})"
        )

        session_dir = os.path.join(
            self.config.raw_dir, "openf1", session.safe_name
        )
        os.makedirs(session_dir, exist_ok=True)

        saved = {}
        for endpoint in self.config.openf1_endpoints:
            filepath = self._fetch_endpoint(endpoint, session, session_dir)
            if filepath:
                saved[endpoint] = filepath
            # Rate limiting
            time.sleep(self.config.openf1_rate_limit)

        # Save metadata
        meta_path = os.path.join(session_dir, "_metadata.json")
        with open(meta_path, "w") as f:
            json.dump({
                "session_key": session.session_key,
                "meeting_key": session.meeting_key,
                "session_name": session.session_name,
                "session_type": session.session_type,
                "circuit": session.circuit_short_name,
                "country": session.country_name,
                "date_start": session.date_start.isoformat(),
                "date_end": session.date_end.isoformat() if session.date_end else None,
                "collected_at": datetime.now(timezone.utc).isoformat(),
                "endpoints_collected": list(saved.keys()),
            }, f, indent=2)

        self.log.info(
            f"[OK] Collected {len(saved)} endpoints for {session.session_name}"
        )
        return saved

    def _fetch_endpoint(
        self, endpoint: str, session: SessionInfo, out_dir: str
    ) -> Optional[Path]:
        """Fetch a single endpoint, handling pagination for large datasets."""
        url = (
            f"{self.config.openf1_base}/{endpoint}"
            f"?session_key={session.session_key}"
        )

        self.log.debug(f"  Fetching {endpoint}...")
        all_data = []

        # For high-frequency endpoints (car_data, location), fetch per-driver
        if endpoint in ("car_data", "location"):
            drivers = api_get(
                f"{self.config.openf1_base}/drivers"
                f"?session_key={session.session_key}"
            )
            if not drivers:
                self.log.warning(f"  No drivers found for {endpoint}")
                return None

            for drv in drivers:
                dn = drv.get("driver_number")
                drv_url = f"{url}&driver_number={dn}"
                data = api_get(drv_url)
                if data:
                    all_data.extend(data)
                    self.log.debug(
                        f"    {endpoint} driver {dn}: {len(data)} records"
                    )
                time.sleep(self.config.openf1_rate_limit)
        else:
            data = api_get(url)
            if data:
                all_data = data

        if not all_data:
            self.log.debug(f"  No data for {endpoint}")
            return None

        filepath = os.path.join(out_dir, f"{endpoint}.json")
        with open(filepath, "w") as f:
            json.dump(all_data, f)

        self.log.info(
            f"  [OK] {endpoint}: {len(all_data)} records -> {filepath}"
        )
        return Path(filepath)


# ---------------------------------------------------------------------------
# FastF1 Live Recorder - records SignalR stream during live sessions
# ---------------------------------------------------------------------------

class LiveRecorder:
    """
    Manages FastF1 SignalR live recording processes.
    Handles the 2-hour server disconnect by spawning overlapping recordings.
    """

    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.log = logger
        self._process: Optional[subprocess.Popen] = None
        self._recording_files: List[str] = []
        self._start_time: Optional[datetime] = None
        self._stop_event = threading.Event()

    @property
    def is_recording(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, session: SessionInfo) -> None:
        """Begin live recording for a session."""
        if self.is_recording:
            self.log.warning("Already recording - skipping start")
            return

        self._stop_event.clear()
        self._recording_files = []

        # Spawn the recording in a thread so the main loop isn't blocked
        thread = threading.Thread(
            target=self._recording_loop,
            args=(session,),
            daemon=True,
            name="live-recorder",
        )
        thread.start()
        self.log.info(f"[REC] Live recording started for {session.session_name}")

    def stop(self) -> List[str]:
        """Stop recording and return list of recorded files."""
        self._stop_event.set()
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self.log.info("[STOP] Live recording stopped")
        return self._recording_files

    def _recording_loop(self, session: SessionInfo) -> None:
        """
        Continuously record, spawning new processes to handle the
        ~2 hour server disconnect limit with overlap.
        """
        segment = 0
        while not self._stop_event.is_set():
            segment += 1
            filename = os.path.join(
                self.config.live_dir,
                f"{session.safe_name}_live_seg{segment:02d}.txt"
            )
            self._recording_files.append(filename)

            self.log.info(f"  Recording segment {segment} -> {filename}")
            try:
                self._process = subprocess.Popen(
                    [
                        sys.executable, "-m", "fastf1.livetiming",
                        "save", filename,
                        "--timeout", str(self.config.signalr_timeout),
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                self._start_time = datetime.now(timezone.utc)

                # Wait for either timeout-overlap or stop event
                overlap_wait = (
                    self.config.signalr_timeout - self.config.signalr_overlap
                )
                if self._stop_event.wait(timeout=overlap_wait):
                    # Stop requested
                    break

                # Timeout approaching - let this keep running but start next
                self.log.info(
                    f"  Segment {segment} approaching timeout, "
                    f"starting overlap segment..."
                )

            except FileNotFoundError:
                self.log.error(
                    "fastf1 not installed - cannot do live recording. "
                    "Install with: pip install fastf1"
                )
                break
            except Exception as exc:
                self.log.error(f"Recording error: {exc}")
                time.sleep(10)


# ---------------------------------------------------------------------------
# Post-Session Processor - enriches raw data for the prediction pipeline
# ---------------------------------------------------------------------------

class PostSessionProcessor:
    """
    Transforms raw OpenF1 + live data into the format expected by
    the Flat Out F1 v2 prediction pipeline.
    """

    def __init__(self, config: CollectorConfig, logger: logging.Logger):
        self.config = config
        self.log = logger

    def process(self, session: SessionInfo, raw_files: Dict[str, Path],
                live_files: Optional[List[str]] = None) -> Optional[Path]:
        """
        Process raw data into pipeline-ready format.
        Returns path to processed session directory.
        """
        self.log.info(f"Processing data for {session.session_name}...")

        out_dir = os.path.join(
            self.config.processed_dir, session.safe_name
        )
        os.makedirs(out_dir, exist_ok=True)

        try:
            # 1. Build consolidated laps DataFrame
            laps_data = self._process_laps(raw_files, out_dir)

            # 2. Build telemetry summary per driver per lap
            telemetry_summary = self._process_car_data(raw_files, out_dir)

            # 3. Process stints (tire strategy)
            stints_data = self._process_stints(raw_files, out_dir)

            # 4. Weather data
            weather_data = self._process_weather(raw_files, out_dir)

            # 5. Position / intervals for race sessions
            position_data = self._process_positions(raw_files, out_dir)

            # 6. Driver info
            driver_data = self._process_drivers(raw_files, out_dir)

            # 7. Try FastF1 enrichment for full telemetry
            self._fastf1_enrich(session, out_dir)

            # 8. Save processing manifest
            manifest = {
                "session": asdict(session) if hasattr(session, '__dataclass_fields__') else str(session),
                "processed_at": datetime.now(timezone.utc).isoformat(),
                "files": {
                    "laps": laps_data,
                    "telemetry_summary": telemetry_summary,
                    "stints": stints_data,
                    "weather": weather_data,
                    "positions": position_data,
                    "drivers": driver_data,
                },
                "live_recordings": live_files or [],
            }
            # Sanitize for JSON
            clean_manifest = json.loads(json.dumps(manifest, default=str))
            with open(os.path.join(out_dir, "_manifest.json"), "w") as f:
                json.dump(clean_manifest, f, indent=2)

            self.log.info(f"[OK] Processed data saved to {out_dir}")
            return Path(out_dir)

        except Exception as exc:
            self.log.error(f"Processing failed: {exc}", exc_info=True)
            return None

    def _load_raw(self, raw_files: Dict[str, Path], key: str) -> Optional[List]:
        """Load a raw JSON file by endpoint key."""
        path = raw_files.get(key)
        if not path or not os.path.exists(path):
            return None
        with open(path) as f:
            return json.load(f)

    def _save_json(self, data: Any, out_dir: str, name: str) -> Optional[str]:
        """Save data as JSON, return relative filename or None."""
        if not data:
            return None
        filepath = os.path.join(out_dir, f"{name}.json")
        with open(filepath, "w") as f:
            json.dump(data, f)
        return f"{name}.json"

    def _process_laps(self, raw_files, out_dir) -> Optional[str]:
        """Process lap timing data into pipeline format."""
        laps = self._load_raw(raw_files, "laps")
        if not laps:
            return None

        # Enrich with computed fields matching pipeline expectations
        for lap in laps:
            # Convert duration strings to seconds where needed
            for field in ["lap_duration", "duration_sector_1",
                          "duration_sector_2", "duration_sector_3"]:
                val = lap.get(field)
                if isinstance(val, str):
                    lap[field] = self._parse_duration(val)

            # Add is_pit_out_lap flag
            lap["is_pit_out_lap"] = lap.get("is_pit_out_lap", False)

        return self._save_json(laps, out_dir, "laps")

    def _process_car_data(self, raw_files, out_dir) -> Optional[str]:
        """
        Aggregate high-frequency car data into per-driver summary stats.
        Full car_data is too large - we compute features the pipeline needs:
          - avg/max/min speed per driver
          - throttle %, brake % distributions
          - DRS usage
          - RPM stats
        """
        car_data = self._load_raw(raw_files, "car_data")
        if not car_data:
            return None

        from collections import defaultdict
        driver_stats = defaultdict(lambda: {
            "speeds": [], "throttles": [], "brakes": [],
            "rpms": [], "gears": [], "drs_on_count": 0, "total_count": 0,
        })

        for rec in car_data:
            dn = rec.get("driver_number")
            if dn is None:
                continue
            stats = driver_stats[dn]
            stats["speeds"].append(rec.get("speed", 0))
            stats["throttles"].append(rec.get("throttle", 0))
            stats["brakes"].append(rec.get("brake", 0))
            stats["rpms"].append(rec.get("rpm", 0))
            stats["gears"].append(rec.get("n_gear", 0))
            if rec.get("drs", 0) in (10, 12, 14):
                stats["drs_on_count"] += 1
            stats["total_count"] += 1

        # Compute summary
        summaries = []
        for dn, stats in driver_stats.items():
            if not stats["speeds"]:
                continue
            speeds = stats["speeds"]
            summaries.append({
                "driver_number": dn,
                "speed_avg": sum(speeds) / len(speeds),
                "speed_max": max(speeds),
                "speed_p90": sorted(speeds)[int(len(speeds) * 0.9)],
                "throttle_avg": sum(stats["throttles"]) / len(stats["throttles"]),
                "brake_pct": sum(1 for b in stats["brakes"] if b > 0) / len(stats["brakes"]),
                "rpm_avg": sum(stats["rpms"]) / len(stats["rpms"]),
                "rpm_max": max(stats["rpms"]),
                "drs_usage_pct": stats["drs_on_count"] / max(stats["total_count"], 1),
                "gear_avg": sum(stats["gears"]) / len(stats["gears"]),
                "sample_count": stats["total_count"],
            })

        return self._save_json(summaries, out_dir, "telemetry_summary")

    def _process_stints(self, raw_files, out_dir) -> Optional[str]:
        return self._save_json(
            self._load_raw(raw_files, "stints"), out_dir, "stints"
        )

    def _process_weather(self, raw_files, out_dir) -> Optional[str]:
        return self._save_json(
            self._load_raw(raw_files, "weather"), out_dir, "weather"
        )

    def _process_positions(self, raw_files, out_dir) -> Optional[str]:
        return self._save_json(
            self._load_raw(raw_files, "position"), out_dir, "positions"
        )

    def _process_drivers(self, raw_files, out_dir) -> Optional[str]:
        return self._save_json(
            self._load_raw(raw_files, "drivers"), out_dir, "drivers"
        )

    def _fastf1_enrich(self, session: SessionInfo, out_dir: str) -> None:
        """
        Attempt to load session via FastF1 for full telemetry enrichment.
        This provides the detailed per-lap telemetry traces that OpenF1
        aggregates don't capture.
        """
        try:
            import fastf1
            fastf1.Cache.enable_cache(
                os.path.join(self.config.cache_dir, "fastf1_cache")
            )

            self.log.info("  Attempting FastF1 enrichment...")

            # Map session names to FastF1 identifiers
            session_map = {
                "Practice 1": "FP1", "Practice 2": "FP2",
                "Practice 3": "FP3", "Sprint Shootout": "SS",
                "Sprint Qualifying": "SQ", "Sprint": "S",
                "Qualifying": "Q", "Race": "R",
            }
            ff1_id = session_map.get(session.session_name)
            if not ff1_id:
                self.log.debug(f"  No FastF1 mapping for {session.session_name}")
                return

            ff1_session = fastf1.get_session(
                session.year, session.circuit_short_name, ff1_id
            )
            ff1_session.load(telemetry=True, weather=True, messages=True)

            # Export laps with telemetry to parquet for pipeline consumption
            laps_df = ff1_session.laps
            if laps_df is not None and len(laps_df) > 0:
                laps_path = os.path.join(out_dir, "fastf1_laps.parquet")
                laps_df.to_parquet(laps_path, index=False)
                self.log.info(f"  [OK] FastF1 laps: {len(laps_df)} -> {laps_path}")

            # Export results
            results = ff1_session.results
            if results is not None and len(results) > 0:
                results_path = os.path.join(out_dir, "fastf1_results.parquet")
                results.to_parquet(results_path, index=False)
                self.log.info(f"  [OK] FastF1 results -> {results_path}")

        except ImportError:
            self.log.debug("  FastF1 not installed - skipping enrichment")
        except Exception as exc:
            self.log.warning(f"  FastF1 enrichment failed: {exc}")

    @staticmethod
    def _parse_duration(val: str) -> Optional[float]:
        """Parse a duration string like '1:23.456' to seconds."""
        if not val:
            return None
        try:
            if ":" in val:
                parts = val.split(":")
                return float(parts[0]) * 60 + float(parts[1])
            return float(val)
        except (ValueError, IndexError):
            return None


# ---------------------------------------------------------------------------
# Main Collector Daemon
# ---------------------------------------------------------------------------

class F1Collector:
    """
    The main orchestrator. Runs continuously:
      1. Checks schedule for active / upcoming sessions
      2. When a session is live -> starts SignalR recording + polls OpenF1
      3. When a session ends -> pulls full historical data + processes
      4. Sleeps when nothing is happening
    """

    def __init__(self, config: Optional[CollectorConfig] = None):
        self.config = config or CollectorConfig()
        self.log = setup_logging(self.config)
        self.schedule = ScheduleManager(self.config, self.log)
        self.openf1 = OpenF1Collector(self.config, self.log)
        self.recorder = LiveRecorder(self.config, self.log)
        self.processor = PostSessionProcessor(self.config, self.log)
        self._running = False
        self._current_session: Optional[SessionInfo] = None
        self._state = SessionState.IDLE

    def run(self) -> None:
        """Main event loop."""
        self._running = True
        self.log.info("=" * 60)
        self.log.info("  FLAT OUT F1 v2 - Telemetry Collector Started")
        self.log.info(f"  Data directory: {self.config.base_dir}")
        self.log.info("=" * 60)

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._handle_signal)
        signal.signal(signal.SIGINT, self._handle_signal)

        # Write PID file
        with open(self.config.pid_file, "w") as f:
            f.write(str(os.getpid()))

        try:
            while self._running:
                self._tick()
        except KeyboardInterrupt:
            self.log.info("Interrupted by user")
        finally:
            self._shutdown()

    def _tick(self) -> None:
        """Single iteration of the main loop."""
        try:
            session = self.schedule.get_active_session()

            if session:
                self._handle_active_session(session)
            else:
                self._handle_idle()

        except Exception as exc:
            self.log.error(f"Tick error: {exc}", exc_info=True)
            time.sleep(30)

    def _handle_active_session(self, session: SessionInfo) -> None:
        """A session is active or about to start."""
        now = datetime.now(timezone.utc)
        is_new = (
            self._current_session is None
            or self._current_session.session_key != session.session_key
        )

        if is_new:
            self.log.info(
                f"[SESSION] Session detected: {session.session_name} "
                f"@ {session.circuit_short_name}"
            )
            self._current_session = session
            self._state = SessionState.PRE_SESSION

        # Determine if session has actually started
        if now >= session.date_start:
            if self._state == SessionState.PRE_SESSION:
                self._state = SessionState.LIVE
                self.log.info("[LIVE] Session is LIVE - starting data collection")
                # Start live recording
                self.recorder.start(session)

            # Check if session has ended
            est_end = session.date_end or (
                session.date_start + timedelta(hours=3)
            )
            if now > est_end:
                if self._state == SessionState.LIVE:
                    self._state = SessionState.POST_SESSION
                    self.log.info("[SESSION] Session ended - collecting full data")

                    # Stop live recording
                    live_files = self.recorder.stop()

                    # Collect full historical data from OpenF1
                    # (wait a bit for data to be fully published)
                    time.sleep(60)
                    raw_files = self.openf1.collect_session(session)

                    # Process everything
                    self.processor.process(session, raw_files, live_files)

                    session.collected = True
                    self._current_session = None
                    self._state = SessionState.IDLE

        # Poll interval while active
        time.sleep(30)

    def _handle_idle(self) -> None:
        """No active session - check for missed sessions and sleep."""
        if self._state != SessionState.IDLE:
            self._state = SessionState.IDLE

        # Check for uncollected past sessions (backfill)
        missed = self.schedule.get_uncollected_past_sessions()
        for session in missed:
            self.log.info(f"[BACKFILL] Backfilling missed session: {session.session_name}")
            raw_files = self.openf1.collect_session(session)
            self.processor.process(session, raw_files)
            session.collected = True

        # Find next session for informative logging
        self._log_next_session()

        time.sleep(self.config.idle_poll_interval)

    def _log_next_session(self) -> None:
        """Log when the next session is."""
        now = datetime.now(timezone.utc)
        future = [
            s for s in self.schedule._known_sessions.values()
            if s.date_start > now
        ]
        if future:
            nxt = min(future, key=lambda s: s.date_start)
            delta = nxt.date_start - now
            hours = delta.total_seconds() / 3600
            if hours < 24:
                self.log.info(
                    f"[IDLE] Idle - next session: {nxt.session_name} "
                    f"@ {nxt.circuit_short_name} in {hours:.1f}h"
                )
            else:
                self.log.info(
                    f"[IDLE] Idle - next session: {nxt.session_name} "
                    f"@ {nxt.circuit_short_name} in {delta.days}d {hours%24:.0f}h"
                )

    def _handle_signal(self, signum, frame):
        self.log.info(f"Signal {signum} received - shutting down...")
        self._running = False

    def _shutdown(self):
        """Clean shutdown."""
        if self.recorder.is_recording:
            self.recorder.stop()
        if os.path.exists(self.config.pid_file):
            os.remove(self.config.pid_file)
        self.log.info("Collector shut down cleanly.")

    def backfill(self, year: int = None, lookback_days: int = 30) -> None:
        """
        One-shot: collect all sessions from the past N days
        that we don't already have.
        """
        year = year or datetime.now().year
        self.log.info(f"[SYNC] Backfilling sessions (lookback={lookback_days} days)...")

        self.schedule.refresh()
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=lookback_days)

        count = 0
        for session in sorted(
            self.schedule._known_sessions.values(),
            key=lambda s: s.date_start
        ):
            end = session.date_end or (session.date_start + timedelta(hours=3))
            if cutoff <= end <= now:
                # Check if already collected
                proc_dir = os.path.join(
                    self.config.processed_dir, session.safe_name
                )
                if os.path.exists(os.path.join(proc_dir, "_manifest.json")):
                    self.log.debug(f"  Skipping {session.safe_name} (already exists)")
                    continue

                raw_files = self.openf1.collect_session(session)
                self.processor.process(session, raw_files)
                count += 1

        self.log.info(f"[OK] Backfill complete: {count} sessions collected")

    def status(self) -> Dict[str, Any]:
        """Return current status information."""
        pid_exists = os.path.exists(self.config.pid_file)
        running_pid = None
        if pid_exists:
            with open(self.config.pid_file) as f:
                running_pid = f.read().strip()

        # Count collected sessions
        collected = 0
        if os.path.exists(self.config.processed_dir):
            collected = sum(
                1 for d in os.listdir(self.config.processed_dir)
                if os.path.isdir(os.path.join(self.config.processed_dir, d))
            )

        # Disk usage
        total_size = 0
        for dirpath, _, filenames in os.walk(self.config.base_dir):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)

        return {
            "daemon_running": pid_exists,
            "pid": running_pid,
            "data_directory": self.config.base_dir,
            "sessions_collected": collected,
            "disk_usage_mb": round(total_size / (1024 * 1024), 2),
            "state": self._state.value,
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Flat Out F1 v2 - Telemetry Data Collector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python f1_collector.py                     # Run collector in foreground
  python f1_collector.py --daemon            # Run as background daemon
  python f1_collector.py --status            # Check if collector is running
  python f1_collector.py --backfill          # Collect missed recent sessions
  python f1_collector.py --backfill --days 60  # Backfill last 60 days
  python f1_collector.py --data-dir /path    # Custom data directory
        """
    )
    parser.add_argument(
        "--daemon", action="store_true",
        help="Run as a background daemon process"
    )
    parser.add_argument(
        "--status", action="store_true",
        help="Show collector status and exit"
    )
    parser.add_argument(
        "--backfill", action="store_true",
        help="One-shot: collect all missed recent sessions"
    )
    parser.add_argument(
        "--days", type=int, default=30,
        help="Lookback days for --backfill (default: 30)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Custom data directory (default: ~/f1_telemetry_data)"
    )
    parser.add_argument(
        "--stop", action="store_true",
        help="Stop a running daemon"
    )

    args = parser.parse_args()

    # Build config
    config = CollectorConfig()
    if args.data_dir:
        config = CollectorConfig(base_dir=args.data_dir)

    collector = F1Collector(config)

    if args.status:
        status = collector.status()
        print("\n+======================================+")
        print("|   F1 Telemetry Collector - Status    |")
        print("+======================================+")
        for k, v in status.items():
            print(f"|  {k:<24} {str(v):>10} |")
        print("+======================================+\n")
        return

    if args.stop:
        if os.path.exists(config.pid_file):
            with open(config.pid_file) as f:
                pid = int(f.read().strip())
            try:
                os.kill(pid, signal.SIGTERM)
                print(f"Sent SIGTERM to PID {pid}")
            except ProcessLookupError:
                print("Process not found - cleaning up PID file")
                os.remove(config.pid_file)
        else:
            print("No running collector found.")
        return

    if args.backfill:
        collector.backfill(lookback_days=args.days)
        return

    if args.daemon:
        # Fork into background (Unix only)
        if sys.platform == "win32":
            print("Daemon mode not supported on Windows.")
            print("Use Task Scheduler instead (see setup instructions).")
            print("Running in foreground...")
            collector.run()
        else:
            pid = os.fork()
            if pid > 0:
                print(f"Collector daemon started (PID: {pid})")
                print(f"Data dir: {config.base_dir}")
                print(f"Logs: {config.log_dir}")
                sys.exit(0)
            else:
                # Child process - detach
                os.setsid()
                collector.run()
    else:
        collector.run()


if __name__ == "__main__":
    main()
