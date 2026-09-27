# Evaluation: targets, metrics and baselines

This describes the evaluation as implemented in `flatout/evaluate.py` and `flatout/nested.py`.
Everything is scored per race, and the race is the unit of independence.

## Scoring target

- **Drivers scored.** Only the drivers who appear in the official race results (the starters) are scored. The field size `D` is 19–22 depending on the race. A predicted driver who does not start is dropped. The predicted position matrix is then collapsed onto the `D` starters, and Sinkhorn-rebalanced so that every position is filled exactly once.
- **Actual position.** The actual position is the official `Position` order, converted to ranks 0 … D−1. Retired cars are ordered behind classified cars, as officially recorded. **Disqualified cars are placed at the back**, following the official result order (for example China 2025: LEC, HAM, GAS at 18–20).
  - The simulator does not model disqualification, so a DSQ counts as a large miss. That is the honest outcome for a forecast of the official result.
- **Outcome field** (race analysis v8). Each row carries one of:
  - `finished`
  - `classified_retirement` (stopped, but covered 90% of the distance)
  - `retired`
  - `dsq`
  - `dns`
  - `not_classified` (still running, but under 90% distance)
- **Retirement definition.** For reliability modelling, `dnf` means `outcome == 'retired'`. Before v8, DSQ and DNS rows were counted as retirements, which inflated the breakdown rates of the affected teams.

## Metrics (per race, then averaged over races)

`P` is the D×D matrix of predicted position probabilities, with rows as drivers. `a` is each driver's actual position.

| Metric | Definition | Better |
|---|---|---|
| RPS | mean over drivers of Σ_k (CDF_pred(k) − CDF_actual(k))² / (D−1). Normalised by D−1, so it is comparable across field sizes. | lower |
| Log loss | −mean over drivers of log P[d, a_d], with probabilities clipped at 1e-4 | lower |
| Brier win / podium / points | mean over drivers of (P(≤k) − 1[a ≤ k])², for k = 1, 3, 10 | lower |
| Spearman | rank correlation of expected position with actual position | higher |
| MAE position | mean \|expected position − actual\| | lower |
| P(winner) | probability given to the actual winner | higher |
| Top-3 hit | share of the three drivers with the best expected positions who actually finished in the top three | higher |
| Winner correct | the driver with the highest win probability won | higher |

## Baselines

- **Grid (post-qualifying only).**
  - The actual starting slot is turned into probabilities with a grid→finish transition matrix `T`. `T` is estimated only from races strictly before the scored race, with add-0.2 smoothing, and the result is Sinkhorn-rebalanced.
  - This baseline is not eligible pre-weekend, because the grid is not known at that forecast time. An actual-grid reference may be shown for context, but it is not a competitor.
- **Pace-only.**
  - Positions come from ranking predicted race pace plus N(0, 0.6²) noise (0.6% of a lap, a fixed value chosen by hand), estimated from 4,000 draws.
  - It uses the same pace predictions as the model, but no race simulation.

## Uncertainty on comparisons

- `nested.paired()` reports the mean paired per-race difference (model − baseline), with a 95% interval from a race-level bootstrap (20,000 resamples) and the share of races where the model is better.
- The number of races is the independent sample count. Driver rows within a race are not independent.
- Monte Carlo error at 300k simulations per race is negligible next to these differences. A 200k run and a 4M run of the same races agree to 3–4 decimal places in RPS.

## Out-of-sample status of reported numbers

- **In-sample development numbers:** results from `backtest` and `calibrate` are in-sample for the simulator parameters (see `docs/AUDIT_2026-09-27.md`).
- **Nested walk-forward numbers:** only `nested` results are walk-forward for every fitted component.
- **Still development, not an untouched holdout:** the 2025 and 2026 seasons have been inspected repeatedly during development. The frozen prospective protocol in `docs/PROSPECTIVE.md` covers races from 2026 round 16 onward.
