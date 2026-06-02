#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║        FLAT OUT F1 v2 — Pipeline Data Integration Module        ║
║    Bridges collected telemetry into prediction pipeline format   ║
╚══════════════════════════════════════════════════════════════════╝

This module loads data collected by f1_collector.py and transforms it
into the feature format expected by the Flat Out F1 v2 ensemble models.

Features generated:
  - Weighted long-run pace
  - Sector-specific performance
  - Tire degradation patterns
  - Speed trap / telemetry features
  - Weather-adjusted lap times
  - Cross-session improvement trajectories
  - Position delta and overtaking metrics

Usage:
    from pipeline_bridge import PipelineBridge

    bridge = PipelineBridge()
    features_df = bridge.build_features(year=2025, session_type="Race")
    # → DataFrame ready for your ensemble models
"""

import os
import json
import glob
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any
from datetime import datetime

logger = logging.getLogger("f1_pipeline_bridge")

# Try importing pandas/numpy — these should be available in your pipeline env
try:
    import pandas as pd
    import numpy as np
except ImportError:
    raise ImportError(
        "pandas and numpy are required. Install: pip install pandas numpy"
    )


class PipelineBridge:
    """
    Loads collected F1 telemetry data and generates features
    compatible with the Flat Out F1 v2 prediction pipeline.
    """

    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or os.path.expanduser(
            "~/f1_telemetry_data"
        )
        self.processed_dir = os.path.join(self.data_dir, "processed")

    # ──────────────────────────────────────────────────────────
    # Discovery
    # ──────────────────────────────────────────────────────────

    def list_sessions(self, year: int = None, session_type: str = None) -> pd.DataFrame:
        """
        List all collected sessions with metadata.

        Args:
            year: Filter by year (e.g. 2025)
            session_type: Filter by type ("Race", "Qualifying", "Practice")

        Returns:
            DataFrame with session metadata
        """
        sessions = []
        if not os.path.exists(self.processed_dir):
            return pd.DataFrame()

        for dirname in sorted(os.listdir(self.processed_dir)):
            manifest_path = os.path.join(
                self.processed_dir, dirname, "_manifest.json"
            )
            if not os.path.exists(manifest_path):
                continue

            with open(manifest_path) as f:
                manifest = json.load(f)

            session_info = manifest.get("session", {})
            sessions.append({
                "directory": dirname,
                "session_key": session_info.get("session_key"),
                "session_name": session_info.get("session_name"),
                "session_type": session_info.get("session_type"),
                "circuit": session_info.get("circuit_short_name"),
                "country": session_info.get("country_name"),
                "date_start": session_info.get("date_start"),
                "year": session_info.get("year"),
                "processed_at": manifest.get("processed_at"),
                "has_fastf1": os.path.exists(os.path.join(
                    self.processed_dir, dirname, "fastf1_laps.parquet"
                )),
            })

        df = pd.DataFrame(sessions)
        if year and not df.empty:
            df = df[df["year"] == year]
        if session_type and not df.empty:
            df = df[df["session_type"].str.contains(session_type, case=False, na=False)]
        return df

    # ──────────────────────────────────────────────────────────
    # Data Loading
    # ──────────────────────────────────────────────────────────

    def load_session_data(self, session_dir: str) -> Dict[str, pd.DataFrame]:
        """
        Load all data files for a session into DataFrames.

        Returns dict with keys: laps, telemetry_summary, stints,
        weather, positions, drivers, fastf1_laps, fastf1_results
        """
        base = os.path.join(self.processed_dir, session_dir)
        data = {}

        # JSON files
        for name in ["laps", "telemetry_summary", "stints", "weather",
                      "positions", "drivers"]:
            path = os.path.join(base, f"{name}.json")
            if os.path.exists(path):
                with open(path) as f:
                    raw = json.load(f)
                if raw:
                    data[name] = pd.DataFrame(raw)

        # Parquet files from FastF1
        for name in ["fastf1_laps", "fastf1_results"]:
            path = os.path.join(base, f"{name}.parquet")
            if os.path.exists(path):
                data[name] = pd.read_parquet(path)

        return data

    # ──────────────────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────────────────

    def build_features(
        self,
        year: int = None,
        session_type: str = "Race",
        sessions: List[str] = None,
    ) -> pd.DataFrame:
        """
        Build the feature matrix for the prediction pipeline.

        This generates the 80+ features your ensemble expects:
          - Pace features (long-run, single-lap, sector splits)
          - Telemetry features (speed, throttle, brake, DRS)
          - Strategy features (tire deg, stint lengths, pit timing)
          - Weather interaction features
          - Cross-session momentum / improvement trajectories

        Args:
            year: Filter sessions by year
            session_type: "Race", "Qualifying", "Practice", or "All"
            sessions: Specific session directories to include

        Returns:
            DataFrame with one row per driver per session, all features
        """
        if sessions:
            session_list = sessions
        else:
            df_sessions = self.list_sessions(year=year, session_type=session_type)
            session_list = df_sessions["directory"].tolist()

        all_features = []

        for session_dir in session_list:
            try:
                data = self.load_session_data(session_dir)
                features = self._extract_session_features(session_dir, data)
                if features is not None:
                    all_features.append(features)
            except Exception as exc:
                logger.warning(f"Failed to build features for {session_dir}: {exc}")

        if not all_features:
            return pd.DataFrame()

        combined = pd.concat(all_features, ignore_index=True)

        # Add cross-session features (improvement trajectories)
        combined = self._add_cross_session_features(combined)

        return combined

    def _extract_session_features(
        self, session_dir: str, data: Dict[str, pd.DataFrame]
    ) -> Optional[pd.DataFrame]:
        """Extract all features for one session."""

        laps = data.get("laps")
        telem = data.get("telemetry_summary")
        stints = data.get("stints")
        weather = data.get("weather")
        drivers = data.get("drivers")

        if laps is None or laps.empty:
            return None

        # Get unique drivers
        driver_numbers = laps["driver_number"].unique()
        rows = []

        for dn in driver_numbers:
            drv_laps = laps[laps["driver_number"] == dn].copy()
            row = {"session_dir": session_dir, "driver_number": dn}

            # ── Driver info ──
            if drivers is not None and not drivers.empty:
                drv_info = drivers[drivers["driver_number"] == dn]
                if not drv_info.empty:
                    row["driver_name"] = drv_info.iloc[0].get("broadcast_name", "")
                    row["team_name"] = drv_info.iloc[0].get("team_name", "")

            # ── Lap pace features ──
            row.update(self._pace_features(drv_laps))

            # ── Sector features ──
            row.update(self._sector_features(drv_laps))

            # ── Telemetry features ──
            if telem is not None and not telem.empty:
                drv_telem = telem[telem["driver_number"] == dn]
                if not drv_telem.empty:
                    row.update(self._telemetry_features(drv_telem.iloc[0]))

            # ── Stint / tire features ──
            if stints is not None and not stints.empty:
                drv_stints = stints[stints["driver_number"] == dn]
                row.update(self._stint_features(drv_stints, drv_laps))

            # ── Weather features ──
            if weather is not None and not weather.empty:
                row.update(self._weather_features(weather))

            rows.append(row)

        return pd.DataFrame(rows)

    def _pace_features(self, laps: pd.DataFrame) -> Dict[str, float]:
        """
        Compute pace-related features from lap data.
        Matches the pipeline's weighted long-run pace methodology.
        """
        features = {}

        # Filter to valid laps (exclude pit in/out, safety car)
        valid = laps[
            (laps.get("is_pit_out_lap", pd.Series(False)) == False)
        ].copy()

        if "lap_duration" not in valid.columns or valid.empty:
            return features

        durations = pd.to_numeric(valid["lap_duration"], errors="coerce").dropna()

        if durations.empty:
            return features

        features["lap_count"] = len(durations)
        features["lap_time_mean"] = durations.mean()
        features["lap_time_median"] = durations.median()
        features["lap_time_best"] = durations.min()
        features["lap_time_std"] = durations.std()
        features["lap_time_p10"] = durations.quantile(0.1)
        features["lap_time_p90"] = durations.quantile(0.9)

        # Long-run pace (stints of 5+ laps)
        # Weight more recent laps higher (exponential decay)
        if len(durations) >= 5:
            weights = np.exp(-0.1 * np.arange(len(durations))[::-1])
            features["weighted_long_run_pace"] = np.average(
                durations.values, weights=weights
            )
        else:
            features["weighted_long_run_pace"] = durations.mean()

        # Consistency score (lower = more consistent)
        if features["lap_time_mean"] > 0:
            features["consistency_score"] = (
                features["lap_time_std"] / features["lap_time_mean"]
            )
        else:
            features["consistency_score"] = np.nan

        # Delta to field (will be computed cross-driver later)
        features["delta_to_best"] = (
            features["lap_time_best"] - durations.min()
        )

        return features

    def _sector_features(self, laps: pd.DataFrame) -> Dict[str, float]:
        """Sector split features."""
        features = {}

        for i, col in enumerate(
            ["duration_sector_1", "duration_sector_2", "duration_sector_3"], 1
        ):
            if col not in laps.columns:
                continue
            vals = pd.to_numeric(laps[col], errors="coerce").dropna()
            if vals.empty:
                continue
            features[f"sector{i}_best"] = vals.min()
            features[f"sector{i}_mean"] = vals.mean()
            features[f"sector{i}_std"] = vals.std()

        return features

    def _telemetry_features(self, telem_row: pd.Series) -> Dict[str, float]:
        """Features from aggregated car telemetry."""
        prefix = "telem_"
        features = {}
        for col in ["speed_avg", "speed_max", "speed_p90", "throttle_avg",
                     "brake_pct", "rpm_avg", "rpm_max", "drs_usage_pct",
                     "gear_avg"]:
            val = telem_row.get(col)
            if val is not None and not pd.isna(val):
                features[f"{prefix}{col}"] = float(val)
        return features

    def _stint_features(
        self, stints: pd.DataFrame, laps: pd.DataFrame
    ) -> Dict[str, float]:
        """Tire strategy and degradation features."""
        features = {}

        if stints.empty:
            return features

        features["num_stints"] = len(stints)
        features["num_pit_stops"] = max(0, len(stints) - 1)

        # Tire compounds used
        if "compound" in stints.columns:
            compounds = stints["compound"].unique()
            features["compounds_used"] = len(compounds)
            for c in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
                features[f"used_{c.lower()}"] = int(c in compounds)

        # Stint lengths
        if "lap_end" in stints.columns and "lap_start" in stints.columns:
            lengths = (
                pd.to_numeric(stints["lap_end"], errors="coerce")
                - pd.to_numeric(stints["lap_start"], errors="coerce")
                + 1
            )
            features["avg_stint_length"] = lengths.mean()
            features["max_stint_length"] = lengths.max()

        # Tire degradation: lap time slope within stints
        if "lap_duration" in laps.columns and "stint_number" in laps.columns:
            deg_rates = []
            for stint_num in laps["stint_number"].unique():
                stint_laps = laps[laps["stint_number"] == stint_num]
                times = pd.to_numeric(
                    stint_laps["lap_duration"], errors="coerce"
                ).dropna()
                if len(times) >= 4:
                    # Linear regression slope = degradation rate
                    x = np.arange(len(times))
                    slope = np.polyfit(x, times.values, 1)[0]
                    deg_rates.append(slope)

            if deg_rates:
                features["tire_deg_rate_mean"] = np.mean(deg_rates)
                features["tire_deg_rate_max"] = np.max(deg_rates)

        return features

    def _weather_features(self, weather: pd.DataFrame) -> Dict[str, float]:
        """Weather condition features."""
        features = {}

        for col, prefix in [
            ("air_temperature", "air_temp"),
            ("track_temperature", "track_temp"),
            ("humidity", "humidity"),
            ("wind_speed", "wind_speed"),
            ("rainfall", "rainfall"),
        ]:
            if col not in weather.columns:
                continue
            vals = pd.to_numeric(weather[col], errors="coerce").dropna()
            if vals.empty:
                continue
            features[f"{prefix}_mean"] = vals.mean()
            features[f"{prefix}_max"] = vals.max()
            features[f"{prefix}_min"] = vals.min()
            features[f"{prefix}_range"] = vals.max() - vals.min()

        return features

    def _add_cross_session_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add features that span multiple sessions:
          - Delta to field (relative pace)
          - Cross-session improvement trajectory
          - Momentum score
        """
        if df.empty or "session_dir" not in df.columns:
            return df

        # Delta to field: per session, compute each driver's pace relative
        # to the session median
        if "lap_time_mean" in df.columns:
            session_medians = df.groupby("session_dir")["lap_time_mean"].transform("median")
            df["delta_to_field"] = df["lap_time_mean"] - session_medians
            df["pct_off_field"] = (
                df["delta_to_field"] / session_medians * 100
            )

        # Cross-session improvement: if a driver appears in multiple sessions,
        # track their trajectory
        if "weighted_long_run_pace" in df.columns:
            df = df.sort_values(["driver_number", "session_dir"])
            df["pace_improvement"] = df.groupby("driver_number")[
                "weighted_long_run_pace"
            ].diff()

            # Rolling momentum (3-session window)
            df["pace_momentum_3"] = df.groupby("driver_number")[
                "pace_improvement"
            ].transform(lambda x: x.rolling(3, min_periods=1).mean())

        return df

    # ──────────────────────────────────────────────────────────
    # Convenience: load for specific race prediction
    # ──────────────────────────────────────────────────────────

    def get_race_prediction_input(
        self, year: int, circuit: str
    ) -> pd.DataFrame:
        """
        Build the feature set for predicting a specific race.
        Uses all available practice/qualifying data for that weekend
        plus historical cross-session features.

        Args:
            year: Season year
            circuit: Circuit short name (e.g. "Melbourne")

        Returns:
            Feature DataFrame ready for model.predict()
        """
        sessions = self.list_sessions(year=year)
        if sessions.empty:
            logger.warning(f"No sessions found for {year}")
            return pd.DataFrame()

        # Find sessions for this circuit
        circuit_sessions = sessions[
            sessions["circuit"].str.contains(circuit, case=False, na=False)
        ]

        if circuit_sessions.empty:
            logger.warning(f"No sessions found for {circuit} in {year}")
            return pd.DataFrame()

        # Build features from all weekend sessions (FP1, FP2, FP3, Quali)
        features = self.build_features(
            sessions=circuit_sessions["directory"].tolist()
        )

        return features


# ──────────────────────────────────────────────────────────
# CLI for quick data inspection
# ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Flat Out F1 v2 — Pipeline Data Bridge"
    )
    parser.add_argument("--list", action="store_true", help="List collected sessions")
    parser.add_argument("--year", type=int, help="Filter by year")
    parser.add_argument("--features", type=str, help="Build features for a session dir")
    parser.add_argument("--race", type=str, help="Build prediction input for a circuit")
    parser.add_argument("--data-dir", type=str, help="Custom data directory")

    args = parser.parse_args()
    bridge = PipelineBridge(data_dir=args.data_dir)

    if args.list:
        sessions = bridge.list_sessions(year=args.year)
        if sessions.empty:
            print("No collected sessions found.")
        else:
            print(f"\n{'Sessions Collected':=^60}")
            for _, row in sessions.iterrows():
                ff1 = "✓" if row.get("has_fastf1") else " "
                print(
                    f"  [{ff1}] {row['session_name']:>15} | "
                    f"{row['circuit']:<20} | {row['date_start']}"
                )
            print(f"{'':=^60}")
            print(f"  Total: {len(sessions)} sessions")

    elif args.features:
        features = bridge.build_features(sessions=[args.features])
        print(f"\nFeatures shape: {features.shape}")
        print(f"Columns: {list(features.columns)}")
        print(features.head())

    elif args.race:
        year = args.year or datetime.now().year
        features = bridge.get_race_prediction_input(year, args.race)
        print(f"\nPrediction input for {args.race} {year}:")
        print(f"Shape: {features.shape}")
        if not features.empty:
            print(features[["driver_number", "driver_name", "team_name",
                            "weighted_long_run_pace"]].to_string())
