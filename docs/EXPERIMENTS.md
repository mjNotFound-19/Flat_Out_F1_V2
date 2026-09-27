# Experiment ledger (human-readable)

The machine-readable record is `artifacts/experiments/ledger.jsonl`, with one folder per experiment under `artifacts/experiments/`.
Every experiment is walk-forward: each scored race uses only information from before it.

| # | Date (UTC) | Experiment | Data / code | Budget, runtime | Result | Decision |
|---|---|---|---|---|---|---|
| E1 | 2026-09-27 | **Season weights** (pace level). 5 fixed weightings, plus nested selection over the 8 preceding races. `artifacts/experiments/pace_season_weights/` | analysis v8, commit `8a7098b` | 350 walk-forward fits, 606 s | No weighting beats the hand-set 0.35/0.6/1.0 beyond noise. Down-weighting old seasons (0.1/0.3 or 0.05/0.15) is clearly worse for post-quali 2026 (+0.015 to +0.020 pace RMSE, 95% interval above 0). Nested selection does not help, and is +0.0024 worse for pre-weekend 2025 (interval above 0). | Keep the current weights. |
| E2 | 2026-09-27 | **Form carry-over and half-lives** (pace level). 11 settings. `artifacts/experiments/pace_form_settings/` | analysis v8, commit `3899938`. The default reproduces the saved dataset exactly. | 11 dataset builds plus 770 fits, 1,648 s | Carrying more of a team's pre-2026 form is clearly worse in 2026: team_carry 0.5 is +0.016 to +0.022 and 1.0 is +0.022 to +0.036, intervals above 0. A team half-life of 5 races (+0.013 to +0.022) and a pace half-life of 3 (+0.009, post-quali 2026) are clearly worse. No setting is clearly better than the default: the best is team_carry 0.1 pre-weekend 2026, −0.005 with interval [−0.015, +0.007]. | Keep the defaults. No challenger clears the pre-registered bar, so no simulation challenger was run. |
| E3 | 2026-09-27 | **Champion nested benchmark, 2026 rounds 3–15.** Both modes, 300k simulations per race and mode, re-tuned on the 13 preceding races before every race. | **analysis v7** (before the DSQ/DNS fix), commit `9ba4d7e` | about 6 min per race | *see below once complete* | Reference for the correction of the reported figures. |

**Units.** Pace RMSE is in % of lap time, per race, over drivers with 8 or more clean laps in dry races. Intervals are 95% race-level bootstraps of the paired difference against the reference.
