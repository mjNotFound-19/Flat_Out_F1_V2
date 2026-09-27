# Changelog

## 2.3.0: 27 September 2026

This version replaces the v2 score-plus-noise pipeline as the main engine. The v2 scripts are kept and still run.
Internal folder names (`predictions_v3/`, `web/data/v3/`) are unchanged, so existing paths and the live site keep working.

### Race engine (`flatout/`)
- **Simulator:** lap-by-lap Monte Carlo covering tyres, fuel, pit stops, safety cars, VSC, red flags, overtaking and reliability. 4,000,000 races per forecast take about 3.5–5 min on 12 cores, and results are identical on any worker count.
- **Pace model:** LightGBM + ridge, trained on fuel- and tyre-corrected race pace, with walk-forward calibrated uncertainty.
- **Event identity:** a sourced circuit registry. The 2026 "Bahrain Grand Prix" is correctly simulated at **Sepang** (it had been Sakhir before). Unknown or conflicting venues raise errors.
- **Forecast contract v2:** each forecast records the verified venue, its assumptions, what every probability means, which inputs were missing, and the code and data versions. It is validated and then written to immutable run folders.
- **Race outcomes:** disqualifications and non-starts are no longer counted as retirements.
- **Strategy:** the behaviour forecast (what teams will do) is kept separate from a model recommendation (forced plans compared under common random numbers).
- **Lab:** precomputed what-if scenarios.
- **Unseen circuits:** optional circuit-parameter uncertainty, plus a permanent unseen-venue evaluation (`python -m flatout unseen`).

### Honest evaluation
- **Nested walk-forward benchmark** (`python -m flatout nested`). Before each scored race, the simulator settings are re-tuned on earlier races only.
- **2026 result, after qualifying:** RPS 0.1087 against the grid's 0.1204. The paired difference is −0.0117 [−0.0191, −0.0052], and the model is better in 12 of 13 races. This is development evidence; the prospective record starts at Sepang.
- **Promotion rules written before testing** (`docs/PROMOTION.md`); `python -m flatout promote` applies them mechanically.
- **Recorded outcomes** (`docs/EXPERIMENTS.md`): season-weight and form-setting sweeps, the reliability-prior challenger (not promoted), and new-venue studies.
- **Tests:** leakage (perturbing future data), reproducibility, the forecast contract, event identity and the CLI.

### Website (f1.h)
- **Race page:** verified venue ("Bahrain Grand Prix in Malaysia · Sepang International Circuit"), provisional status with stated assumptions, and the official Sepang layout traced from the formula1.com map (no telemetry exists for it).
- **Accuracy:** built on the nested benchmark, comparing each mode only with the baselines available at that time.
- **Driver vs Car:** 90% intervals, and what the decomposition can and cannot show.
- **Strategy and Lab:** the behaviour vs recommendation panel, the what-if scenarios and the reproducibility panel.
- **Audit:** every page passes in Fan and Nerd modes at desktop and phone widths (`web/scripts/audit.sh`).

See `docs/HANDOFF.md` for limitations and the remaining experiments.
