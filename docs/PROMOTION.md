# Model promotion criteria (pre-specified)

Written on 27 September 2026 at about 01:15 UTC. At that point only the baseline nested run (the current champion) had started, and no challenger had been evaluated.

## Champion

- **Champion A** is the current system: the LightGBM + ridge pace blend, hand-set season weights and form half-lives, the current simulator, and simulator parameters re-tuned walk-forward before each race.
- **Its benchmark** is `python -m flatout nested --year 2026 --first 3` (plus 2025 where run).
- **It stays published** unless a challenger meets every criterion below.

## Evaluation set

- **Primary:**
  - 2026 rounds 3–15 (13 races), in both `post_quali` and `pre_weekend` modes
  - 2025 rounds 3–24 (22 races) in `post_quali`, if compute allows
- **Protocol:** nested walk-forward (see `docs/METRICS.md`).
- **Matched simulations:**
  - Champion and challenger use the same simulation budget per race (300k).
  - They use the same random seeds per race, so the comparison has common random numbers.
- **Development status:**
  - These seasons have been inspected during development, so they count as development evidence.
  - Races from round 16 onward are scored under `docs/PROSPECTIVE.md`. They are reported separately and never used for tuning.

## A challenger is promoted only if ALL hold

1. **Primary metric.** The mean paired per-race RPS difference (challenger − champion), pooled over all evaluated race-mode pairs:
   - is **≤ −0.0020**
   - and the upper end of its race-level bootstrap 95% interval is **< 0**.
2. **No regime collapse.** Within each (season, mode) block, the challenger's mean paired RPS difference is no worse than **+0.0040**.
3. **Probabilities are not worse.**
   - Log loss is not worse by more than 0.010 (pooled mean).
   - Brier win, podium and points are each not worse by more than 2% relative.
4. **Calibration is not worse.** The calibration error for win and podium probabilities (equal-mass bins, pooled) is not worse by more than 0.010.
5. **Not simulation noise.**
   - The primary-metric result holds on a second run with independent seeds.
   - Alternatively, the paired difference is larger than 5× its Monte Carlo standard error.
6. **No leakage.**
   - The challenger passes `tests/test_leakage.py`.
   - Every tuned value in it is chosen inside the nested protocol.

## Not grounds for promotion on its own

- Better winner-pick rate, P(winner) or top-3 hit rate. These are small-sample and noisy; they are reported but not gated.
- Better results on a hand-picked subset of circuits.
- Added sophistication.

## Reporting

- **Every evaluated challenger is reported**, whether it passes or fails: per-race differences, the bootstrap interval, the block results and runtime.
- **A failed challenger stays a research artifact.** It lives under `artifacts/experiments/` and is recorded in the ledger.
