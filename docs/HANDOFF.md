# Handoff: Flat Out F1 upgrade, 27 September 2026

- **Branch:** `v3-race-intelligence`
- **Commits:** 31 on top of the two earlier unpushed commits (`6434679`, `6dfa479`), which are preserved.
- **Nothing was pushed or published.**

## Status in one paragraph

The forecast for the next race was simulating the wrong circuit. That is fixed.

The accuracy claim is now measured properly. Every fitted component, including the simulator settings, is re-fitted on earlier races only before each scored race. The claim holds: after qualifying, the model's RPS is 0.1087 against the grid's 0.1204 over 13 races of 2026. The paired difference is −0.0117, with a 95% interval of [−0.0191, −0.0052], and the model is better in 12 of 13 races. These races were inspected during development, so they are development evidence. The prospective record starts at Sepang.

The current model remains champion. Two families of pace-model changes and one simulator challenger were tested under rules written beforehand, and none earned promotion.

## What changed and why

| Area | Change | Evidence |
|---|---|---|
| **Event identity** | The 2026 "Bahrain Grand Prix" is at **Sepang, Malaysia** (56 laps, race 4 October 07:00 UTC). Every earlier round 16 forecast used Sakhir. Added a sourced circuit registry (26 circuits) and explicit resolution; conflicting or unknown metadata now raises an error. | formula1.com event page; `tests/test_events.py` (all 71 schedule rows resolve) |
| **Old forecasts** | The Sakhir runs were moved byte for byte to `predictions_v3/2026/R16/superseded/` (`invalid_for_event`). The pre-contract Sepang run is kept as `runs/*_legacy_v1`. | sha256 verified |
| **Leakage audit** | The simulator settings had been tuned on the same 2026 races that the backtest reported. The 2025 backtest used settings tuned on later races. | `docs/AUDIT_2026-09-27.md` |
| **Honest benchmark** | `python -m flatout nested`: settings are re-tuned before each race on earlier races, with checkpoints, a ledger, paired bootstrap intervals, and each mode compared only with baselines available at that time. | E3, E4 in `docs/EXPERIMENTS.md` |
| **Leakage tests** | Perturbing future data leaves the as-of features, circuit priors, pace predictions and grid prior unchanged. Positive controls confirm the test can fail. | `tests/test_leakage.py` |
| **Data correctness** | Disqualifications (8) and non-starts (13) are no longer counted as retirements. Disqualified cars' laps now come from lap data (analysis v8). | `race._outcome` |
| **Reproducibility** | Results are identical for any worker count (SeedSequence streams over fixed 200k chunks). One shared worker pool makes calibration about 6× faster. Forecast provenance records code and data hashes. | `tests/test_reproducibility.py` |
| **Forecast contract v2** | Identity, assumptions, probability definitions, missing inputs and provenance. Validated, written to immutable run folders, and switched over atomically. The export refuses a forecast whose circuit doesn't match the verified one. | `tests/test_contract.py` |
| **Pace model** | Season weights and form carry-over/half-lives are now parameters, tested walk-forward. The hand-set values were retained. | E1, E2 |
| **Strategy** | A behaviour forecast (what teams will do) is kept separate from a recommendation (forced plans under common random numbers), and the recommendation's scope is stated. | `strategy_eval.py` |
| **Lab** | Precomputed what-if scenarios (no live recomputation on the static site; weather marked as not modelled) and a reproducibility panel. | `scenarios.py` |
| **Site** | Verified venue, provisional status with assumptions, honest no-map and no-3D fallbacks for Sepang. The Accuracy page is built on the nested results, with a visible note on the old in-sample figure. Driver vs Car shows 90% intervals (only 5 of 22 drivers are clearly nonzero) and an identifiability note. | Audit: 30 of 30 page/mode/size combinations clean |
| **CLI** | `--grid` is validated against the entry list. New commands: `nested`, `promote`, `snapshot`, `predict --recommend --scenarios`. | `tests/test_cli.py` |

## Models evaluated

| Candidate | Result | Decision |
|---|---|---|
| Season weights (5 variants and nested selection) | No improvement beyond noise. Down-weighting old seasons is worse. | Champion kept |
| Form carry-over and half-lives (11 variants) | Higher team carry-over is clearly worse in 2026. Nothing is clearly better. | Champion kept |
| R1: season-level reliability prior | RPS +0.0001 [−0.0002, +0.0005]. Fails promotion criteria 1 and 5. | Research artifact |

**Champion.** LightGBM + ridge pace, the hand-set recency settings, and simulator settings tuned on the 13 most recent races (for forecasting).

## Current forecast: 2026 round 16, Sepang (provisional, pre-weekend, 4M simulations)

- **Win chances:** Antonelli 24.2%, Russell 18.2%, Leclerc 13.0%, Norris 11.5%, Verstappen 10.4%, Hamilton 10.1%.
- **Why it's provisional:** Sepang has no race in the 2024–26 data. The base lap comes from a pooled rate (17.65 s/km × 5.543 km). Pit loss, tyre wear, safety cars and overtaking are pooled priors, and driver pace uncertainty is ×1.25.
- **Strategy:** the model rates a one-stop soft→hard about 1 place better than the likely-plan mix. That rating is driven by the pooled priors.

## Known limitations

1. **Retirements are under-predicted in 2026** (13–15% predicted against 18% observed per car). The simple fix (R1) did not improve race forecasts.
2. **Strategy is static pre-race plans.** No rolling-horizon in-race policy was built (the brief's Stage 4 decision layer). It is documented as not done, rather than faked.
3. **No 2026 energy, active-aero or Overtake Mode modelling.** Their effects enter only through lap data and the calibrated passing settings. No reduced-order energy state was attempted, because nothing public identifies it.
4. **Weather is not modelled.**
5. **Sepang geometry and telemetry are unavailable.** FastF1 position data starts in 2018, and Sepang's last race was in 2017.
6. **The 2025 nested benchmark was not run**, for compute reasons. Only 2026 is measured out-of-sample.
7. **`exp_pts` can credit retired cars** ordered inside the top 10. This is documented in the contract definitions and is negligible in practice.

## Remaining expensive experiments (in priority order)

1. **2025 nested champion benchmark** (`nested --year 2025 --first 3`, about 2 h). Doubles the evidence base.
2. **Per-lap retirement hazard with exposure and a season effect.** It targets limitation 1 and needs a nested run of about 40 min.
3. **Overtaking model normalised by pace spread** (a partially pooled logistic on labelled passes). Aimed at the low-passing circuits where the model lost to the grid in 2025.
4. **Rolling-horizon pit decisions** with bounded look-ahead. Large engineering work, and it needs the simulator changes validated first.

## Release

- **Build.** The site is staged locally (`publish-portfolio.sh --no-deploy`) with the Sepang forecast and the E4 benchmark. The portfolio's tracked files were not touched.
- **Live site.** manasjha.online/f1 **still shows the invalid Sakhir forecast**. `publish` deploys the corrected build; it was not run, pending your instruction.
- **Push.** 33 local commits on `v3-race-intelligence` are not pushed.

## Runbook

See `README.md` (PowerShell table). The key commands:

```powershell
python -m flatout predict --mode pre_weekend --recommend --scenarios
python -m flatout predict --mode post_quali --grid <codes>    # after qualifying; remember COL's 5-place drop
python -m flatout nested --year 2026 --first 3               # honest benchmark
python -m flatout promote --champion <exp> --challenger <exp>
python -m unittest discover -s tests
```
