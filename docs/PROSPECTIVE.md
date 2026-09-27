# Prospective evaluation protocol (frozen)

Frozen on 27 September 2026. It applies to every race from **2026 round 16 (Bahrain Grand Prix at Sepang)** onward.

## Rule

A forecast counts as prospective only if all three hold:

1. It was saved through the forecast contract, meaning an immutable `predictions_v3/<y>/R<rr>/runs/<created_utc>_<mode>/` folder with a validated `meta.json`.
2. Its `created_utc` is **before** the start of the first session its mode must not see:
   - `pre_weekend`: before FP1 (or before sprint qualifying at a sprint weekend)
   - `post_quali`: before the race start
3. Its `provenance.code.git_rev` is a committed revision, recorded along with the dirty-diff hash if the tree was dirty.

## Scoring

- **Which run is scored.** After the race, the last eligible run per mode is scored against the official result (`docs/METRICS.md`). Runs created after the cutoff are ignored for prospective scoring.
- **Where results go.** Prospective scores are reported separately from retrospective backtests. They are never used to tune anything until the frozen protocol is reopened, which must be recorded in this file with a date and a reason.
- **Honesty about gaps.**
  - A race with no eligible archived forecast is reported as *missed*, not back-filled.
  - Superseded forecasts (such as the Sakhir-based round 16 runs from 26 September) are invalid for their event and are not scored.

## Timeline

| Event | Eligible modes | Notes |
|---|---|---|
| 2026 round 16, Sepang (FP1 2 October 04:30 UTC, race 4 October 07:00 UTC) | pre_weekend, post_quali | New venue: the forecast is `provisional`. |
