# Flat Out F1 · Version 2.3.0

Flat Out F1 is a Formula 1 race forecaster. It simulates each Grand Prix lap by lap, 4,000,000 times, from a pace model trained on real race laps. It predicts finishing positions, win/podium/points/retirement chances, tyre strategies and pit windows. Every forecast is scored against the result, and accuracy is only claimed from out-of-sample tests.

The results are published at **f1.h**: <https://manasjha.online/f1/>.

| | |
|---|---|
| **Version** | 2.3.0 (27 September 2026). See [`CHANGELOG.md`](CHANGELOG.md). |
| **Data** | FastF1: every race of 2024 and 2025, and 2026 rounds 1–15 (63 races, 1,288 driver-races) |
| **Accuracy (2026, after qualifying)** | RPS **0.1087** against the starting grid's 0.1204. That is about **10% lower error**, with a 95% range of the difference of −0.019 to −0.005, better in 12 of 13 races. |
| **Accuracy (2026, before the weekend)** | RPS **0.1145** against a pace-only baseline's 0.1234. The grid is unknown at that point, so it is not compared. |
| **Evidence status** | Nested walk-forward development evidence. The prospective record starts at 2026 round 16 ([`docs/PROSPECTIVE.md`](docs/PROSPECTIVE.md)). |
| **Next race** | 2026 round 16, **Bahrain Grand Prix in Malaysia at Sepang** (56 laps, 4 October, 07:00 UTC). The forecast is provisional: a new circuit for the data. |

## How it works

1. **Data** (`ingest.py`, `race.py`)
   - FastF1 sessions are stored as parquet and synced incrementally.
   - Each race is analysed with a regression on clean green-flag laps: driver + fuel + compound + tyre age. That gives fuel- and tyre-corrected pace, degradation, pit loss, safety cars, passes, stints and outcomes (finished, retired, disqualified, did not start).
2. **Event identity** (`events.py`, `registry/`)
   - Every event maps to a physical circuit through a sourced registry.
   - The commercial title and the host can differ (the "Bahrain" GP is at Sepang). Unknown or conflicting venues raise errors instead of being guessed.
3. **Features and pace model** (`features.py`, `model.py`)
   - Only information from before the race is used: qualifying, practice, sprint, and recency-weighted driver and team form.
   - The model is a LightGBM + ridge blend with walk-forward calibrated uncertainty.
   - The 2026 rule change is handled by season weights and a low carry-over of pre-2026 team form. Both were tested walk-forward and retained.
4. **Circuit settings** (`circuits.py`)
   - Per circuit: tyre wear, pit loss, safety-car rate, overtaking and stop count. Each is shrunk towards the all-circuit average.
   - An unseen circuit uses the pooled averages. Optionally, it draws those settings per batch from their measured uncertainty (`new_venue_mode='param_unc'`).
5. **Simulator** (`sim.py`)
   - A vectorised lap-by-lap Monte Carlo covering tyres and cliff, fuel, planned and safety-car stops, VSC and red flags, pace-dependent passing with traffic, start chaos and reliability.
   - It uses `SeedSequence` streams over fixed chunks, so results are identical on any machine.
6. **Outputs** (`pipeline.py`, `contract.py`)
   - Each forecast is validated against a written contract (verified venue, coherent probabilities, stated assumptions).
   - It is written to an immutable run folder, then becomes the event's current forecast.
7. **Evaluation** (`evaluate.py`, `nested.py`, `promotion.py`)
   - Scoring rules and baselines, and the nested walk-forward benchmark (simulator settings re-tuned before each race on earlier races only).
   - Challengers are promoted only if they pass the rules written beforehand in [`docs/PROMOTION.md`](docs/PROMOTION.md).
8. **Extras**
   - `strategy_eval.py`: a strategy recommendation, kept separate from the behaviour forecast.
   - `scenarios.py`: what-if scenarios.
   - `ratings.py`: driver, constructor and Driver vs Car ratings, with intervals.
   - `unseen_eval.py`: forecasting as if the circuit were new.

More detail:
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md): data flow and forecast contract
- [`docs/METRICS.md`](docs/METRICS.md): scoring definitions
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md): every experiment and its decision
- [`docs/AUDIT_2026-09-27.md`](docs/AUDIT_2026-09-27.md): audit of earlier claims
- [`docs/HANDOFF.md`](docs/HANDOFF.md): limitations and next experiments

## Setup (PowerShell)

```powershell
pip install pandas numpy scipy scikit-learn lightgbm pyarrow fastf1
python -m unittest discover -s tests          # identity, contract, leakage, reproducibility and CLI checks
```

## Race-weekend runbook (PowerShell, from the project folder)

| Step | Command | Notes |
|---|---|---|
| Pull new sessions | `python -m flatout sync` | Incremental. FastF1's API limit is handled by waiting. |
| Rebuild analyses and features | `python -m flatout build` | Add `--force` after changing race-analysis code. Analyses are versioned (`race.ANALYSIS_VERSION = 8`). |
| Retrain pace models | `python -m flatout train` | |
| Forecast before the weekend | `python -m flatout predict --mode pre_weekend` | 4,000,000 simulations, about 3.5–5 min on 12 cores. The grid is simulated. |
| Forecast after qualifying | `python -m flatout predict --mode post_quali` | Uses the stored grid. `--grid VER,NOR,...` sets it manually (for penalties) and is checked against the entry list. |
| Add strategy and what-ifs | `python -m flatout predict --recommend --scenarios` | Model-dependent forced-plan comparison, plus precomputed Lab scenarios. |
| Score finished races | `python -m flatout evaluate` | |
| Everything | `python -m flatout weekend` (or `run_weekend.bat`) | sync → build → evaluate → train → predict |
| Tune simulator settings for forecasting | `python -m flatout calibrate --year 2026 --last-n 13` | In-sample on those races: never quote it as accuracy. |
| Status | `python -m flatout status` | |

## Evaluation and experiments

| Task | Command | Notes |
|---|---|---|
| Out-of-sample benchmark | `python -m flatout nested --year 2026 --first 3` | 1–2.5 h. `--recal-every 2` halves the cost. It checkpoints and resumes, and appends to `artifacts/experiments/ledger.jsonl`. |
| Test a challenger | `python -m flatout nested --year 2026 --candidate '{"params": {"rel_model": "hazard"}}' --label chal` | Also accepts `form_settings` and `season_weight`. |
| Promotion decision | `python -m flatout promote --champion <exp> --challenger <exp>` | Mechanical application of `docs/PROMOTION.md`. |
| New-circuit test | `python -m flatout unseen --experiment <exp>` | Re-forecasts every race as if its circuit had never been raced. |
| Snapshot before a big change | `python -m flatout snapshot --label before_x` | Copies uncommitted outputs with a sha256 manifest. |

## Website (f1.h)

| Task | Command |
|---|---|
| Preview locally | `node web/server.mjs` (serves http://localhost:4173) |
| Check every page (both modes, desktop and phone) | `bash web/scripts/audit.sh` (needs the preview running and `playwright-cli`) |
| Publish to manasjha.online/f1 | `bash web/scripts/publish-portfolio.sh` (or the `publish` PowerShell function); `--no-deploy` stages only |

Pages: **Race**, **Strategy**, **Drivers**, **Driver vs Car**, **Teams**, **Season**, **Accuracy**, and **Lab** (Nerd mode). The site reads only `web/data/v3/site.json`.

- **Race:** verified venue, forecast mode and time, and a provisional status with stated assumptions.
- **Circuit views:** a 3D circuit from real telemetry. Where no telemetry exists (Sepang), the official layout is traced from the formula1.com map and clearly labelled.
- **Strategy:** likely plans against the plan the model rates best.
- **Accuracy:** built on the nested benchmark.
- **Driver vs Car:** 90% intervals and an identifiability note.
- **Lab:** what-if scenarios and reproducibility metadata.

## Outputs

| Path | What |
|---|---|
| `predictions_v3/<y>/R<rr>/runs/<utc>_<mode>/` | Immutable forecast runs: `summary.csv`, `distribution.csv` and `meta.json` (schema v2: identity, assumptions, definitions, provenance). |
| `predictions_v3/<y>/R<rr>/pre_race_*` | The event's current forecast, switched atomically after validation. It is accompanied by `recommendations.json` and `scenarios.json` when computed. |
| `predictions_v3/<y>/R<rr>/superseded/` | Forecasts found to be invalid, kept for audit (for example the Sakhir-based round 16 runs). |
| `artifacts/experiments/` | Experiment folders (config, per-race results, calibration, promotion decisions) and `ledger.jsonl`. |
| `flatout/registry/circuits.json`, `flatout/registry/layouts/` | The sourced circuit registry and traced official layouts. |

## Known limitations (2.3.0)

- **Retirements are under-predicted in 2026:** about 13% per car predicted against 18% observed. Challengers are under test.
- **Strategy recommendations are static pre-race plans.** There is no in-race rolling-horizon policy.
- **2026 energy, active aero and Overtake Mode are not modelled explicitly**; they enter only through lap data and calibrated passing. Weather is not modelled.
- **The version name and folder names differ.** Folder names (`predictions_v3/`, `web/data/v3/`) are historical and unchanged, so existing paths keep working.

---

# Flat Out F1 v2 (legacy scripts, still working)

Flat Out F1 v2 is a local Formula 1 race prediction pipeline. It pulls and stores FastF1 session data, engineers driver/team/session features, generates pre-race finishing-order predictions, scores those predictions after each race, updates driver and constructor profiles, self-tunes model hyperparameters, and exposes the latest prediction output through a small static web dashboard.

The project is intentionally file-based. Most state lives in CSV and JSON files in the repository so predictions, profiles, feature importance, scoring reports, and dashboard data can be inspected directly without a database.

## Table Of Contents

- [What This Project Does](#what-this-project-does)
- [Repository Layout](#repository-layout)
- [Pipeline Overview](#pipeline-overview)
- [Requirements](#requirements)
- [First-Time Setup](#first-time-setup)
- [Quick Start](#quick-start)
- [Main Workflow](#main-workflow)
- [Data Collection](#data-collection)
- [Prediction Pipeline](#prediction-pipeline)
- [Post-Race Learning](#post-race-learning)
- [Web Dashboard](#web-dashboard)
- [Optional Historical MLP Side Model](#optional-historical-mlp-side-model)
- [Background Telemetry Collector](#background-telemetry-collector)
- [Important Files And Artifacts](#important-files-and-artifacts)
- [Modeling Notes](#modeling-notes)
- [Changing Seasons, Rounds, Grids, And Rosters](#changing-seasons-rounds-grids-and-rosters)
- [Common Commands](#common-commands)
- [Troubleshooting](#troubleshooting)
- [Development Notes](#development-notes)

## What This Project Does

This repository contains a full local workflow for F1 prediction experiments:

1. Pull raw FastF1 data for testing, practice, qualifying, sprint, and race sessions.
2. Store session data under `fastf1_data/` as CSV files.
3. Auto-discover available sessions and determine the current round from local data.
4. Build per-driver features from:
   - Current-weekend free practice and qualifying data
   - Sprint and sprint qualifying data when available
   - Preseason test data
   - Historical driver performance profiles
   - Constructor/team profiles
   - Rolling season race metrics from prior races
   - Hard-coded grid overrides when available
5. Train/blend a collection of classic ML models for race-position scoring.
6. Run Monte Carlo race simulations to produce:
   - Predicted finishing order
   - Win probability
   - Podium probability
   - Expected points
   - Average finish
   - P10/P50/P90 finish ranges
   - Full per-position probability distributions
7. Score predictions after race results are available.
8. Update driver profiles, constructor profiles, season race metrics, and self-tuned hyperparameters.
9. Copy output CSVs into `web/data/` and display them in a local browser dashboard.

This is not a packaged Python library. It is a research/workbench-style project where scripts are run directly from the repository root.

## Repository Layout

```text
.
|-- f1_ml_predict_v2.py             # Main v2 prediction pipeline
|-- f1_learn.py                     # Post-race scoring, profile updates, strategy analysis
|-- f1_autotune.py                  # Self-tuning hyperparameter updater
|-- f1_race_metrics.py              # Race metrics extractor used by f1_learn.py
|-- fastf1_pull.py                  # Targeted FastF1 data pull utility
|-- fastf1_pull_all.py              # Larger bootstrap data pull for historical/test data
|-- fastf1_pull_history.py          # Historical driver-performance pull utility
|-- compare_predictions.py          # Compare two prediction CSVs against scored actuals
|-- f1_hyperparams.json             # Self-tuned model parameters
|-- driver_profiles.csv             # Learned driver profile table
|-- constructor_profiles.csv        # Learned constructor/team profile table
|-- season_race_data.csv            # Rolling per-race metrics used as future features
|-- v2_prediction_results.csv       # Main prediction summary output
|-- v2_full_predictions.csv         # Full prediction output with position distributions
|-- v2_driver_features.csv          # Feature matrix generated by the predictor
|-- v2_feature_importance.csv       # Feature importance from tree models
|-- predictions/                    # Per-round prediction, scoring, and metric archives
|-- fastf1_data/                    # Raw and derived FastF1 CSV data
|-- fastf1_cache/                   # FastF1 cache, ignored by git
|-- web/                            # Static browser dashboard
|-- model/                          # Older/alternate model artifacts
|-- v2_side_models/                 # Optional PyTorch historical MLP and ensemble workflow
|-- datacollectors/                 # Optional background telemetry collector tooling
|-- *_backup.csv / *_backup.json    # Manual backup snapshots
|-- *_ORIGINAL.*                    # Original copies kept for comparison/regression work
`-- .gitignore
```

Some directories in the current checkout are empty placeholders or local runtime folders:

- `season/` exists but the active season-level CSV currently lives at `season_race_data.csv`.
- `Flat_Out_F1_V2/` is present but empty in this checkout.
- `__pycache__/` is Python runtime output and should not be edited manually.

## Pipeline Overview

High-level data flow:

```text
FastF1/OpenF1 APIs
        |
        v
fastf1_pull.py / fastf1_pull_all.py / fastf1_pull_history.py
        |
        v
fastf1_data/
        |
        v
f1_ml_predict_v2.py
        |
        +--> v2_prediction_results.csv
        +--> v2_full_predictions.csv
        +--> v2_driver_features.csv
        +--> v2_feature_importance.csv
        |
        v
web/scripts/sync-data.mjs
        |
        v
web/data/*.csv --> web/index.html dashboard

After the race:

fastf1_data/<round race session>/
        |
        v
f1_learn.py ingest <round>
        |
        +--> predictions/round_NN/prediction.csv
        +--> predictions/round_NN/scored.csv
        +--> predictions/round_NN/metrics.csv
        +--> driver_profiles.csv
        +--> constructor_profiles.csv
        +--> season_race_data.csv
        +--> f1_hyperparams.json via f1_autotune.py
```

The main predictor is `f1_ml_predict_v2.py`. It is designed to run without command-line arguments. It inspects local data, discovers the current round, builds features, trains/combines models, simulates the race, and exports the root-level `v2_*` CSV files.

## Requirements

### Python

Use a modern Python 3 version. Python 3.10+ is a reasonable baseline for the libraries used here.

Core Python packages:

```powershell
pip install pandas numpy scipy scikit-learn fastf1
```

Optional Python packages:

```powershell
pip install torch pyarrow
```

Optional packages are used for:

- `torch`: `v2_side_models/` historical MLP training and ensemble prediction.
- `pyarrow`: parquet support in the background collector bridge.

### Node.js

The dashboard under `web/` uses Node only for a tiny local static server and a data-copy script. It has no runtime npm dependencies in `web/package.json`.

Recommended:

- Node.js 18+
- npm

### Network Access

FastF1 data pulls require internet access. Once data is already saved under `fastf1_data/`, the prediction scripts can run locally from those files.

### Operating System

The repository is currently set up on Windows and includes Windows-specific helper scripts in `datacollectors/`. Most core Python scripts should also work on macOS/Linux if dependencies and paths are adjusted.

## First-Time Setup

From the repository root:

```powershell
cd C:\Users\manas\projects\flat_out_f1_v2
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install Python dependencies:

```powershell
pip install pandas numpy scipy scikit-learn fastf1
```

Install optional side-model/collector dependencies if needed:

```powershell
pip install torch pyarrow
```

Install dashboard metadata/dependencies:

```powershell
cd web
npm install
cd ..
```

`web/package.json` currently has no external dependencies, but running `npm install` keeps the `web/package-lock.json` flow consistent.

## Quick Start

If the repository already contains usable `fastf1_data/` files, generate a fresh prediction:

```powershell
python f1_ml_predict_v2.py
```

Then open the local dashboard:

```powershell
cd web
npm run dev
```

Open:

```text
http://localhost:4173
```

`npm run dev` runs `npm run sync-data` first, which copies these root-level output files into `web/data/`:

- `v2_prediction_results.csv`
- `v2_full_predictions.csv`
- `v2_feature_importance.csv`

## Main Workflow

### 1. Pull Or Update Session Data

Use `fastf1_pull.py` for targeted pulls.

Show the schedule:

```powershell
python fastf1_pull.py --year 2026 --schedule
```

Pull every session for a round:

```powershell
python fastf1_pull.py --year 2026 --round 5 --session all
```

Pull one session:

```powershell
python fastf1_pull.py --year 2026 --round 5 --session FP1
python fastf1_pull.py --year 2026 --round 5 --session Q
python fastf1_pull.py --year 2026 --round 5 --session R
```

Pull preseason testing:

```powershell
python fastf1_pull.py --year 2026 --testing 1
python fastf1_pull.py --year 2026 --testing 2
```

### 2. Generate A Prediction

Run:

```powershell
python f1_ml_predict_v2.py
```

This writes:

- `v2_prediction_results.csv`
- `v2_full_predictions.csv`
- `v2_driver_features.csv`
- `v2_feature_importance.csv`

### 3. View The Prediction

Run:

```powershell
cd web
npm run dev
```

Open `http://localhost:4173`.

### 4. Ingest Race Results After The Race

Make sure the race session exists, for example:

```text
fastf1_data/2026_R5_Canadian Grand Prix_R/
```

or:

```text
fastf1_data/2026_R5_Canadian Grand Prix_Race/
```

Then run:

```powershell
python f1_learn.py ingest 5
```

This scores the current root prediction against race results, updates profile files, extracts race metrics, and triggers self-tuning.

### 5. Review Season Accuracy And Profiles

```powershell
python f1_learn.py report
python f1_learn.py profiles
```

## Data Collection

### Targeted FastF1 Pulls

`fastf1_pull.py` is the most useful day-to-day data collection script.

Supported session identifiers:

| Identifier | Meaning |
| --- | --- |
| `FP1` | Free Practice 1 |
| `FP2` | Free Practice 2 |
| `FP3` | Free Practice 3 |
| `Q` | Qualifying |
| `SQ` | Sprint Qualifying / Sprint Shootout |
| `S` | Sprint |
| `R` | Race |
| `all` | All mapped sessions for the weekend |

Typical output path:

```text
fastf1_data/2026_R5_Canadian Grand Prix_FP1/
```

Typical files inside a session directory:

```text
laps.csv
results.csv
weather.csv
race_control.csv
```

If telemetry is requested, the script can also export larger per-driver telemetry/car/position files depending on what FastF1 returns.

Example with telemetry:

```powershell
python fastf1_pull.py --year 2026 --round 5 --session R --telemetry
```

Telemetry exports can be large. Use them only when the extra detail is needed.

### Bootstrap Pulls

`fastf1_pull_all.py` is a broader bootstrap script. It is configured in code with flags such as:

- `PULL_2024`
- `PULL_2025`
- `PULL_TEST1`
- `PULL_TEST2`
- `BUILD_PROFILES`

It pulls historical race results, combines them, builds historical driver profiles, and can pull preseason testing data.

Run:

```powershell
python fastf1_pull_all.py
```

The script itself notes that a full run can take 10-20 minutes depending on API speed.

### Historical Pull

`fastf1_pull_history.py` is another historical bootstrap utility focused on transferable driver-performance data from 2024-2025 plus 2026 Test 2 data.

Run:

```powershell
python fastf1_pull_history.py
```

It writes under:

```text
fastf1_data/historical/
fastf1_data/testing2/
```

## Prediction Pipeline

### Main Entrypoint

```powershell
python f1_ml_predict_v2.py
```

The script has staged output printed to the console. It does not currently expose command-line flags; configuration is mostly embedded in the script and in `f1_hyperparams.json`.

### How Current Round Is Chosen

The predictor scans `fastf1_data/` and detects session directories matching patterns like:

```text
2026_R3_Japanese Grand Prix_FP1
2026_R3_Japanese Grand Prix_Q
2026_R3_Japanese Grand Prix_R
```

The current round is the highest round number found in local session data.

This means if `fastf1_data/` contains a future or partial round, the predictor may treat that as current. Remove/move incomplete future-round folders or adjust the script if you want to predict an earlier round.

### Main Input Sources

The predictor combines multiple evidence tiers:

- Current-weekend FP/quali/sprint data
- Earlier current-season sessions
- Preseason testing data
- Historical driver profiles
- Constructor profiles
- Season race metrics
- Grid position when available
- Self-tuned hyperparameters

Important root input files:

| File | Purpose |
| --- | --- |
| `fastf1_data/` | Session data discovered by the predictor |
| `driver_profiles.csv` | Learned driver priors from post-race ingests |
| `constructor_profiles.csv` | Learned team priors from post-race ingests |
| `season_race_data.csv` | Rolling racecraft/pace/strategy metrics |
| `f1_hyperparams.json` | Tuned tier weights, grid model, circuit profiles, uncertainty settings |

### Main Output Files

| File | Purpose |
| --- | --- |
| `v2_prediction_results.csv` | Compact prediction table for ranking and headline metrics |
| `v2_full_predictions.csv` | Prediction table plus per-position probabilities |
| `v2_driver_features.csv` | Generated feature matrix used by the model |
| `v2_feature_importance.csv` | Feature importance from the tree models |
| `v2_model_comparison.csv` | Model comparison table when generated |

### `v2_prediction_results.csv` Schema

Current columns include:

| Column | Meaning |
| --- | --- |
| `Driver` | Driver abbreviation |
| `Team` | Constructor/team name |
| `score` | Internal model score |
| `win_pct` | Monte Carlo win probability |
| `pod_pct` | Monte Carlo podium probability |
| `avg_pts` | Expected points |
| `p10` | 10th percentile finish position |
| `p50` | Median finish position |
| `p90` | 90th percentile finish position |
| `avg_fin` | Average simulated finish position |
| `Pos` | Final predicted rank |
| `Name` | Full driver name |
| `fm` | Form marker used by the printed/dashboard output |

When grid data is available, `GridPos` may be present in related outputs.

### `v2_full_predictions.csv` Schema

This file includes the compact columns plus per-position probability columns:

```text
P1_pct, P2_pct, P3_pct, ...
```

The dashboard uses these columns for the driver-level probability bar chart.

### `v2_driver_features.csv` Schema

This is the widest and most diagnostic output. It includes engineered columns such as:

- FP race pace
- Qualifying pace
- Race/quali gap
- Degradation slope
- Tyre pressure proxy
- Lap counts
- Historical driver profile features
- Constructor profile features
- Rolling season metrics
- Grid score
- Target/model feature columns

Use this file when debugging why a prediction moved.

### `v2_feature_importance.csv` Schema

Columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature name |
| `importance` | Average tree-model feature importance |

The dashboard renders the top feature importances from this file.

## Post-Race Learning

### Main Entrypoint

```powershell
python f1_learn.py ingest <round>
```

Example:

```powershell
python f1_learn.py ingest 4
```

`f1_learn.py` expects race data for the round under `fastf1_data/`, with a folder name ending in `Race` or `R`.

Example:

```text
fastf1_data/2026_R4_Miami Grand Prix_R/
```

### What Ingest Does

For the selected round, ingest:

1. Loads `laps.csv` and `results.csv` from the race directory.
2. Classifies each driver's finish status:
   - `DNS`
   - `DNF_LAP1`
   - `DNF_EARLY`
   - `DNF_MID`
   - `DNF_LATE`
   - `CLASSIFIED`
3. Copies the current `v2_prediction_results.csv` into `predictions/round_NN/prediction.csv`.
4. Scores predicted vs actual finishing positions.
5. Writes `predictions/round_NN/scored.csv`.
6. Writes `predictions/round_NN/metrics.csv`.
7. Updates `driver_profiles.csv`.
8. Updates `constructor_profiles.csv`.
9. Runs `f1_race_metrics.py` logic to extract race metrics.
10. Updates `season_race_data.csv`.
11. Runs `f1_autotune.py` to tune future prediction parameters.

### Post-Race Reports

Season accuracy dashboard:

```powershell
python f1_learn.py report
```

Current profiles:

```powershell
python f1_learn.py profiles
```

### Per-Round Archive

Each ingested round has:

```text
predictions/round_04/
|-- prediction.csv
|-- scored.csv
`-- metrics.csv
```

`prediction.csv` is the snapshot that was scored. `scored.csv` contains driver-level errors. `metrics.csv` contains summary accuracy metrics such as MAE, Spearman rho, Kendall tau, exact matches, within-one, within-three, and winner correctness.

## Web Dashboard

The dashboard lives in `web/` and is served by `web/server.mjs`.

Run:

```powershell
cd web
npm run dev
```

Open:

```text
http://localhost:4173
```

### Dashboard Scripts

`web/package.json` defines:

| Script | Command | Purpose |
| --- | --- | --- |
| `sync-data` | `node scripts/sync-data.mjs` | Copy root `v2_*` CSVs into `web/data/` |
| `dev` | `npm run sync-data && node server.mjs` | Sync data and run the local server |
| `start` | `npm run dev` | Alias for dev |
| `build` | `npm run sync-data` | Copy data only |

### Dashboard Data Files

The browser reads:

```text
web/data/v2_prediction_results.csv
web/data/v2_full_predictions.csv
web/data/v2_feature_importance.csv
```

These are copied from the repository root by:

```powershell
npm run sync-data
```

### Dashboard Views

Simple mode shows:

- Predicted finishing order
- Driver name
- Team
- Win chance
- Podium chance
- Median finish

Complex mode shows:

- Searchable driver cards
- Sort controls
- Win/podium/median/score metrics
- Driver detail distribution chart
- Top 3 and top 10 probability aggregates
- P10-P90 range
- Expected points
- Feature importance bars

### Port Configuration

Default port:

```text
4173
```

Override it with:

```powershell
$env:PORT=5000
npm run dev
```

## Optional Historical MLP Side Model

The `v2_side_models/` directory contains an optional PyTorch workflow that builds a historical dataset, trains an MLP, and blends MLP predictions with the main v2 output.

This path is optional. The main project works through `f1_ml_predict_v2.py` without it.

### Side Model Files

| File | Purpose |
| --- | --- |
| `build_historical_dataset.py` | Pull/extract historical race features into a training CSV |
| `historical_training_data.csv` | Training dataset |
| `train_historical_model.py` | Train the PyTorch MLP |
| `historical_mlp.pt` | Saved PyTorch model weights |
| `historical_mlp_meta.json` | Feature metadata, scalers, medians, categories |
| `training_report.txt` | Training/CV report |
| `f1_ensemble_predict.py` | Blend v2 prediction with MLP output |
| `ensemble_prediction.csv` | Blended prediction output |

### Build Historical Dataset

From `v2_side_models/`:

```powershell
python build_historical_dataset.py --years 2022 2023 2024 2025 --resume
```

The `--resume` flag skips year/round pairs that are already present in the output CSV.

### Train MLP

```powershell
python train_historical_model.py
```

Useful options:

```powershell
python train_historical_model.py --skip-cv
python train_historical_model.py --epochs 800
python train_historical_model.py --min-era 2
```

`--min-era 2` keeps ground-effect era and later data. `--min-era 3` would restrict more aggressively to the 2026-style era coding used by the script.

### Run Ensemble Blend

After the main v2 predictor has produced `v2_prediction_results.csv`, copy or generate the required files in `v2_side_models/` as expected by the side-model script, then run:

```powershell
python f1_ensemble_predict.py --round 3
```

Override the MLP blend weight:

```powershell
python f1_ensemble_predict.py --round 3 --mlp-weight 0.35
```

By default, the script increases MLP weight later in the season.

## Background Telemetry Collector

The `datacollectors/` directory contains optional tooling for continuous telemetry collection.

Main files:

| File | Purpose |
| --- | --- |
| `f1_collector.py` | Background collector using OpenF1 and FastF1-style data sources |
| `pipeline_bridge.py` | Converts collected telemetry into feature matrices |
| `setup.py` | Installs dependencies and configures auto-start services |
| `start_collector.bat` | Windows helper to start the collector |
| `start_collector.vbs` | Hidden-window Windows launcher for the batch file |

The collector defaults to storing data under:

```text
~/f1_telemetry_data
```

Common commands:

```powershell
cd datacollectors
python f1_collector.py
python f1_collector.py --daemon
python f1_collector.py --status
python f1_collector.py --backfill
```

Installer:

```powershell
python setup.py
```

Be careful with `datacollectors/setup.py`: it can create OS-level auto-start entries using launchd, systemd, or Windows Task Scheduler depending on platform.

Also note that `start_collector.bat` currently contains absolute local paths, including a specific Python path. Update it if your Python install or repository path changes.

## Important Files And Artifacts

### Root Prediction Outputs

| File | Generated By | Description |
| --- | --- | --- |
| `v2_prediction_results.csv` | `f1_ml_predict_v2.py` | Compact prediction output |
| `v2_full_predictions.csv` | `f1_ml_predict_v2.py` | Full prediction plus per-position distribution |
| `v2_driver_features.csv` | `f1_ml_predict_v2.py` | Feature matrix used for prediction |
| `v2_feature_importance.csv` | `f1_ml_predict_v2.py` | Feature importances |
| `v2_model_comparison.csv` | Predictor/model tooling | Model comparison metrics |

### Learned State

| File | Updated By | Description |
| --- | --- | --- |
| `driver_profiles.csv` | `f1_learn.py ingest` | Per-driver learned performance profile |
| `constructor_profiles.csv` | `f1_learn.py ingest` | Per-team learned performance profile |
| `season_race_data.csv` | `f1_race_metrics.py` via `f1_learn.py` | Rolling season race features |
| `f1_hyperparams.json` | `f1_autotune.py` via `f1_learn.py` | Self-tuned model parameters |

### Backups And Originals

The repository includes several manually preserved copies:

- `driver_profiles_backup.csv`
- `constructor_profiles_backup.csv`
- `f1_hyperparams_backup.json`
- `season_race_data_backup.csv`
- `f1_ml_predict_v2_ORIGINAL.py`
- `v2_prediction_results_ORIGINAL.csv`

These are useful for regression comparison or recovery, but the active scripts read the non-backup filenames by default.

### Model Directory

`model/` contains older or alternate artifacts:

| File | Description |
| --- | --- |
| `ensemble.pkl` | Pickled ensemble artifact |
| `driver_profiles.csv` | Model-local profile copy |
| `circuit_weights.json` | Circuit weighting data |
| `latest_predictions.csv` | Older latest-prediction output |

The active v2 workflow mostly uses the root-level `v2_*` files and profile CSVs.

### FastF1 Data Directory

Common structure:

```text
fastf1_data/
|-- historical/
|   |-- all_race_results.csv
|   |-- driver_profiles.csv
|   |-- 2024_race_results.csv
|   |-- 2025_race_results.csv
|   |-- 2024/
|   `-- 2025/
|-- testing/
|   |-- day1_laps.csv
|   |-- day1_weather.csv
|   `-- ...
|-- testing2/
|   |-- day1_laps.csv
|   |-- day1_weather.csv
|   `-- ...
|-- 2026_R1_Australian Grand Prix_FP1/
|-- 2026_R1_Australian Grand Prix_Q/
|-- 2026_R1_Australian Grand Prix_Race/
`-- ...
```

The predictor recognizes race weekend folders whose names include:

```text
<year>_R<round>_<grand prix name>_<session>
```

where session can be:

```text
FP1, FP2, FP3, SQ, S, Sprint, Q, Qualifying, R, Race
```

## Modeling Notes

### Main v2 Predictor

`f1_ml_predict_v2.py` uses a staged pipeline. The top-level constants include:

| Constant | Current Meaning |
| --- | --- |
| `N_SIM = 10000` | Number of Monte Carlo race simulations |
| `RACE_LAPS = 58` | Default race-lap assumption where needed |
| `FUEL_PER_KG` | Fuel correction coefficient |
| `FUEL_PER_LAP` | Fuel burn assumption |
| `TEMP_COEFF` | Temperature correction coefficient |
| `REF_TEMP` | Reference track/air temperature baseline |

The model set includes:

- `GradientBoostingRegressor`
- `RandomForestRegressor`
- `ExtraTreesRegressor`
- `Ridge`
- `BayesianRidge`
- `SVR`

It also uses:

- `StandardScaler`
- `LeaveOneOut`
- `mean_absolute_error`
- `scipy.stats` correlation utilities

### Feature Tiers

The model weighs evidence from several tiers. The current hyperparameter file stores base tier multipliers under:

```json
"tier_weights": {
  "fp_base": 0.4,
  "season_base": 0.0,
  "testing_base": 0.25,
  "history_base": 0.2,
  "team_base": 0.1
}
```

These are adjusted by the predictor as the season progresses and by `f1_autotune.py` after races are ingested.

### Grid Model

Grid effects are modeled through a tunable grid model:

- Clean-air bonus for front starters
- P4-P10 penalty
- P11-P15 penalty
- P16+ penalty
- Circuit-specific overtaking factor
- Grid blend percentage

The grid model is stored in `f1_hyperparams.json`.

### Monte Carlo Simulation

The simulation accounts for:

- Driver score
- Driver uncertainty
- Team-level shocks
- Driver-level shocks
- DNF probability
- Grid effects
- Safety car style compression
- Close-battle stochastic swaps
- Points allocation

The final output probabilities are derived from simulation counts.

### Self-Tuning

`f1_autotune.py` is called automatically by `f1_learn.py ingest`.

It updates:

- Tier weights
- Grid penalty/bonus model
- Circuit profiles
- Safety car probability proxy
- Tyre degradation factor
- DRS/overtaking effectiveness proxy
- Uncertainty parameters
- Performance log

Manual run:

```powershell
python f1_autotune.py 4 Miami
```

If the scored data for the round does not exist, autotune exits without changing parameters.

## Changing Seasons, Rounds, Grids, And Rosters

### Change Data Pull Year

For `fastf1_pull.py`, pass `--year`:

```powershell
python fastf1_pull.py --year 2027 --round 1 --session all
```

The script default is currently:

```python
DEFAULT_YEAR = 2026
```

### Predict A Specific Round

The main predictor does not currently accept `--round`. It auto-detects the current round from the highest available local round in `fastf1_data/`.

To predict a specific earlier round, use one of these approaches:

1. Temporarily move later-round folders out of `fastf1_data/`.
2. Work from a copy of the data directory containing only data up through the target round.
3. Modify `discover_data()` / current-round selection logic in `f1_ml_predict_v2.py`.

### Add Or Correct A Starting Grid

Grid overrides are hard-coded.

In `f1_ml_predict_v2.py`:

```python
GRID_OVERRIDES = {
    1: {'RUS': 1, 'ANT': 2, ...}
}
```

In `f1_learn.py`:

```python
GRIDS = {
    1: {'RUS': 1, 'ANT': 2, ...}
}
```

If grid order matters for a round, add the round's starting grid to both files so prediction and post-race scoring use the same assumptions.

### Add Or Correct Driver Names

Driver abbreviation mappings live in:

- `f1_ml_predict_v2.py` as `NAMES`
- `f1_learn.py` as `NAMES`

If a new driver appears in FastF1 data but is missing from `NAMES`, outputs will fall back to the abbreviation in some places. Add the driver there for cleaner reports and dashboard output.

### Add Or Correct Teams

Team names mostly flow through FastF1 data and learned constructor profiles. The dashboard also has a `TEAM_COLORS` map in `web/app.js`.

If a new team appears and the dashboard color is missing, add it to:

```javascript
const TEAM_COLORS = {
  "Team Name": "#hex"
};
```

## Common Commands

### Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install pandas numpy scipy scikit-learn fastf1
pip install torch pyarrow
cd web
npm install
cd ..
```

### Pull Data

```powershell
python fastf1_pull.py --year 2026 --schedule
python fastf1_pull.py --year 2026 --round 5 --session FP1
python fastf1_pull.py --year 2026 --round 5 --session Q
python fastf1_pull.py --year 2026 --round 5 --session R
python fastf1_pull.py --year 2026 --round 5 --session all
python fastf1_pull.py --year 2026 --testing 2
```

### Predict

```powershell
python f1_ml_predict_v2.py
```

### Learn After Race

```powershell
python f1_learn.py ingest 5
python f1_learn.py report
python f1_learn.py profiles
```

### Compare Predictions

```powershell
python compare_predictions.py v2_prediction_results_ORIGINAL.csv v2_prediction_results.csv predictions/round_03/scored.csv
```

### Run Dashboard

```powershell
cd web
npm run dev
```

### Sync Dashboard Data Only

```powershell
cd web
npm run sync-data
```

### Optional Side Model

```powershell
cd v2_side_models
python build_historical_dataset.py --years 2022 2023 2024 2025 --resume
python train_historical_model.py
python f1_ensemble_predict.py --round 5
```

## Troubleshooting

### `fastf1 not installed`

Install FastF1 into the active Python environment:

```powershell
pip install fastf1
```

### `No mode selected. Use --testing, --round, or --schedule.`

`fastf1_pull.py` needs one mode:

```powershell
python fastf1_pull.py --schedule
python fastf1_pull.py --round 5 --session R
python fastf1_pull.py --testing 2
```

### FastF1 Session Pull Fails

Common causes:

- The session has not happened yet.
- FastF1 has not published the data yet.
- The session identifier does not match FastF1's available session names.
- Network access is unavailable.
- FastF1 cache is stale or corrupted.

Try:

```powershell
python fastf1_pull.py --year 2026 --schedule
```

Then confirm the round and event format.

### Prediction Uses The Wrong Round

The predictor chooses the highest round present under `fastf1_data/`.

If it predicts a later partial round, move that later round's directories elsewhere or modify current-round selection in `f1_ml_predict_v2.py`.

### `f1_learn.py ingest` Cannot Find Race Data

The race directory must match a folder ending in `Race` or `R`, such as:

```text
fastf1_data/2026_R4_Miami Grand Prix_R/
fastf1_data/2026_R4_Miami Grand Prix_Race/
```

Pull race data first:

```powershell
python fastf1_pull.py --year 2026 --round 4 --session R
```

Then run:

```powershell
python f1_learn.py ingest 4
```

### Dashboard Shows Old Data

Run the sync script:

```powershell
cd web
npm run sync-data
```

or rerun:

```powershell
npm run dev
```

The dashboard reads files from `web/data/`, not directly from the root directory.

### Dashboard Port Is Busy

Use another port:

```powershell
$env:PORT=5000
npm run dev
```

### Browser CSV Load Fails

Open the dashboard through the local server, not by double-clicking `index.html`.

Use:

```powershell
cd web
npm run dev
```

The browser fetches local CSV files through HTTP.

### CSV Parsing Limitations In The Dashboard

`web/app.js` uses a simple CSV parser based on splitting lines and commas. It works for the current generated CSVs because their values are simple. If future fields contain quoted commas, replace the parser with a real CSV parser or ensure generated CSV fields stay comma-free.

### Garbled Symbols In Script Comments

Some existing script comments/docstrings contain garbled box-drawing or symbol characters from an encoding mismatch. The executable logic is still plain Python, but console comments may look odd on some terminals.

### `torch` Missing In Side Model

Install PyTorch before running `v2_side_models/train_historical_model.py` or `v2_side_models/f1_ensemble_predict.py`:

```powershell
pip install torch
```

### Collector Writes Outside The Repository

The background collector defaults to:

```text
~/f1_telemetry_data
```

That is separate from `fastf1_data/`. Use `pipeline_bridge.py` or custom copying/conversion if you want collector output to feed the main prediction pipeline.

## Development Notes

### Git Ignore Behavior

`.gitignore` currently ignores:

- Python runtime artifacts
- virtual environments
- Node dependency/build folders
- FastF1 cache directories
- SQLite files
- OS/editor files

It does not ignore root-level prediction CSVs, profile CSVs, or `fastf1_data/` CSVs. Those files may be intentional project artifacts in this repo.

### No Requirements File

There is no root `requirements.txt` in the current checkout. Dependency installation is documented in this README instead.

Suggested core requirements if one is added later:

```text
pandas
numpy
scipy
scikit-learn
fastf1
```

Suggested optional requirements:

```text
torch
pyarrow
```

### Reproducibility

For reproducible experiments:

1. Keep a copy of the exact `fastf1_data/` folders used for a prediction.
2. Keep the matching `f1_hyperparams.json`.
3. Keep the matching `driver_profiles.csv`.
4. Keep the matching `constructor_profiles.csv`.
5. Keep the matching `season_race_data.csv`.
6. Archive the generated `v2_*` output files.
7. After race ingest, use `predictions/round_NN/prediction.csv` as the scored snapshot.

### Current Checked-Out State

At the time this README was written, the repository contains:

- Scored prediction archives for rounds 1 through 4 under `predictions/`.
- `f1_hyperparams.json` with `_last_round` set to `4`.
- Root-level active prediction outputs in `v2_prediction_results.csv`, `v2_full_predictions.csv`, `v2_driver_features.csv`, and `v2_feature_importance.csv`.
- Dashboard data copies under `web/data/`.
- Historical MLP artifacts under `v2_side_models/`.
- FastF1 session folders under `fastf1_data/`, including historical data, testing data, and several 2026 race-weekend sessions.

### Editing Guidance

When changing model behavior:

- Prefer keeping a backup of the current output files before rerunning.
- Compare old vs new predictions with `compare_predictions.py`.
- Inspect `v2_driver_features.csv` when a driver's ranking changes unexpectedly.
- Inspect `v2_feature_importance.csv` when feature weighting seems counterintuitive.
- Run `f1_learn.py report` after ingesting races to monitor season-level error.

When changing dashboard behavior:

- Remember that `web/data/` files are copied from root outputs.
- Keep `web/scripts/sync-data.mjs` in sync with any renamed output files.
- If new dashboard fields are added, update both the predictor export and `web/app.js` parsing.

## Minimal End-To-End Example

This example pulls a race weekend, predicts it, serves the dashboard, then ingests the race after results are available:

```powershell
# From repo root
python fastf1_pull.py --year 2026 --round 5 --session all

# Generate prediction outputs
python f1_ml_predict_v2.py

# View dashboard
cd web
npm run dev
```

After the race:

```powershell
# From repo root
python fastf1_pull.py --year 2026 --round 5 --session R
python f1_learn.py ingest 5
python f1_learn.py report

# Refresh dashboard data
cd web
npm run sync-data
```

