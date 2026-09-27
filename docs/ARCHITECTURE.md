# Architecture and data flow

```
FastF1 ──sync──▶ data/store/<year>/<rr>_<SESSION>/*.parquet      (raw, incremental, schedule.parquet per season)
                        │
                        ▼
   events.py + registry/circuits.json ── resolve(year, round) ──▶ EventIdentity
        (title ≠ host: e.g. 2026 R16 "Bahrain GP" at Sepang, Malaysia; unknown/conflicting → error)
                        │
race.py (analysis v8) ──▶ data/derived/races/*.json    per race: clean-lap regression (driver, fuel, compound,
                        │                               tyre age), pit loss, SC/VSC/red, passes, stints, outcomes
                        ▼
features.py ──▶ history table + dataset.parquet        as-of features (form EWMAs, quali/FP/sprint), targets
                        │
        ┌───────────────┼──────────────────────────┐
        ▼               ▼                          ▼
   model.py         circuits.py                 ratings.py
   pace models      per-circuit priors          car-adjusted ratings, constructors,
   (LGBM+ridge,     (shrunk to global;          Driver vs Car (+90% intervals)
   walk-forward     cutoff-aware)
   uncertainty)
        └───────┬───────┘
                ▼
   pipeline.build_spec ──▶ spec (arrays)  + ctx (identity, assumptions, candidates, grid prior)
                ▼
   sim.simulate  — vectorised lap-by-lap Monte Carlo; SeedSequence streams over fixed 200k chunks;
                   one reusable worker pool; single-threaded BLAS workers; memory-capped worker count
                ▼
   pipeline.summarise ──▶ contract.publish ──▶ predictions_v3/<y>/R<rr>/runs/<utc>_<mode>/  (immutable)
                                          └──▶ pre_race_* (current; atomic switch after validation)
                ▼
   strategy_eval.recommend (optional) ──▶ recommendations.json (tied to the forecast run)
                ▼
   export.py ──▶ web/data/v3/site.json ──▶ web/ (f1.h: app.js, motion.js, hud.js, circuit3d.js)

Evaluation
   evaluate.py      metrics + grid / pace-only baselines          (docs/METRICS.md)
   backtest.py      walk-forward specs; search_params (calibration search)
   nested.py        nested walk-forward benchmark: params re-tuned on earlier races before each scored race;
                    checkpoints keyed by config+data hash; artifacts/experiments/ledger.jsonl
   pace_eval.py     cheap walk-forward pace-level comparison of configurations + nested selection
   provenance.py    code identity (git rev + dirty-diff hash), data-store fingerprint, snapshots
```

## Forecast contract (schema v2, `flatout/contract.py`)

| Field | Meaning |
|---|---|
| `identity` | Event title, circuit key and name, host country, the location and country the feed listed, length, official laps and their source, race start (UTC), resolution status (`verified` or `override`), sources, registry version. |
| `mode`, `sessions_used`, `inputs_missing`, `information_cutoff_utc` | What information the forecast could see. |
| `status`, `assumptions` | `provisional` when the circuit has no race in the data. Every borrowed value is listed. |
| `definitions` | Exact meaning of win, podium, points, dnf, positions, stop counts, pit windows, and the post-qualifying blend. |
| `provenance` | Code identity, data fingerprint, model-file hash, simulator parameters and their hash, seed scheme. |
| `n_sims`, `sim_seconds`, `warnings` | Compute used, and non-fatal validation notes. |

Validation refuses a forecast when:
- the simulated circuit differs from the verified circuit
- a driver's position probabilities don't sum to 1
- a position is not filled exactly once
- the win, podium or points totals are wrong
- the forecast is provisional but lists no assumptions

## What changed in this upgrade (Stages 1–3; see the git log for details)

- **Event identity.** Replaced the substitution of a historical venue by event name with a sourced registry and explicit resolution.
- **Old round 16 forecasts.** The Sakhir-based forecasts were moved to `superseded/`.
- **Race outcomes.** Disqualifications and non-starts are no longer counted as retirements (analysis v8).
- **Honest evaluation.** Added the nested walk-forward benchmark, leakage tests (perturbing future data), paired bootstrap comparisons, and promotion rules written before any results were seen.
- **Reproducibility.** Reproducible Monte Carlo, independent of worker count, and a shared worker pool (calibration is about 6× faster).
- **Forecast contract.** v2, with immutable run folders and atomic publishing.
- **Pace model settings.** Season weights and form settings are now parameters, and were tested walk-forward. The hand-set values were retained because no alternative beat them beyond noise.
- **Site.** Verified venue, provisional status, honest no-map fallbacks, an Accuracy page built on the nested results, Driver vs Car intervals, and a strategy behaviour-vs-recommendation panel.
