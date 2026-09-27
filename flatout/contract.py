"""Forecast contract: what a saved forecast contains, what each number means, and validation.

Every forecast is written to an immutable run folder
    predictions_v3/<year>/R<rr>/runs/<created_utc>_<mode>/{summary.csv, distribution.csv, meta.json}
and only after validation is the event's *current* forecast (pre_race_*.csv/json) switched to it by
atomic file replacement. Earlier runs are never modified; superseded ones move to superseded/.

SCHEMA_VERSION history
  1  implicit (before 2026-09-27): meta without identity/provenance
  2  identity, assumptions, status, provenance, probability definitions, information cutoff
"""
import json
import os
import tempfile
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 2

# What the published numbers mean. Stored inside every forecast's meta.
DEFINITIONS = {
    'position': 'Order at the chequered flag over all starters; cars not running at the end are ordered '
                'behind every running car (by laps completed). The 90%-distance classification rule is '
                'not modelled.',
    'win': 'P(position 1). Equals P(finish first): a retired car is always behind running cars.',
    'podium': 'P(position <= 3).',
    'points': 'P(position <= 10).',
    'dnf': 'P(not running at the chequered flag), any cause (mechanical, incident). DNS and DSQ are not '
           'modelled separately.',
    'exp_pos': 'Mean position.',
    'exp_pts': 'Expected points from the position distribution (known issue: counts points for retired '
               'cars ordered inside the top 10; negligible except in mass-retirement races).',
    'post_quali_blend': 'Post-qualifying position probabilities are a stacked blend: (1-w) simulation + '
                        'w empirical grid->finish prior (w = grid_blend, fitted on earlier races).',
    'stop_count': 'stop1/stop2/stop3p: P(exactly 1 / 2 / 3+ pit stops) over all simulated races incl. '
                  'retirements (stops made before retiring count).',
    'first_stop_window': 'first_stop_p25/p50/p75: laps of the first stop, conditional on the car stopping.',
    'strategy': 'Most likely executed compound sequence within the modal stop count, among finishers.',
    'grid': 'post_quali: actual starting slot (penalties applied if the stored grid includes them). '
            'pre_weekend: expected slot from simulated qualifying (a model output, not a known grid).',
}

MODES_INFO = {
    'pre_weekend': 'No session of this weekend is used. Grid is simulated.',
    'post_fp': 'Practice sessions of this weekend are used; grid is simulated.',
    'post_sprint': 'Practice, sprint qualifying and sprint are used; grid is simulated.',
    'post_quali': 'Qualifying and the starting grid are used.',
}


class ContractError(ValueError):
    pass


def validate(summary, dist, meta, tol=1e-6):
    """Raise ContractError if a forecast is malformed. Returns a list of warnings (non-fatal)."""
    warn = []
    for k in ('year', 'round', 'location', 'identity', 'mode', 'n_sims', 'created_utc', 'provenance'):
        if k not in meta:
            raise ContractError(f'meta missing {k!r}')
    ident = meta['identity']
    if meta['location'] != ident.get('circuit'):
        raise ContractError(f"simulated circuit {meta['location']} != verified circuit {ident.get('circuit')}")
    D = len(summary)
    if len(dist) != D or set(dist.Driver) != set(summary.Driver):
        raise ContractError('summary and distribution cover different drivers')
    P = dist.drop(columns='Driver').to_numpy(float)
    if P.shape[1] != D:
        raise ContractError(f'distribution has {P.shape[1]} positions for {D} drivers')
    if (P < -tol).any() or (P > 1 + tol).any():
        raise ContractError('probability outside [0, 1]')
    if not np.allclose(P.sum(1), 1, atol=1e-3):
        raise ContractError('a driver\'s position probabilities do not sum to 1')
    if not np.allclose(P.sum(0), 1, atol=1e-2):
        raise ContractError('a position is not filled exactly once across drivers')
    for col, total in (('win', 1), ('podium', 3), ('points', min(10, D))):
        if abs(summary[col].sum() - total) > 1e-2 * total:
            raise ContractError(f'{col} probabilities sum to {summary[col].sum():.4f}, expected {total}')
    for col in ('dnf', 'stop1', 'stop2', 'stop3p'):
        if ((summary[col] < -tol) | (summary[col] > 1 + tol)).any():
            raise ContractError(f'{col} outside [0, 1]')
    if meta['n_sims'] < 100_000:
        warn.append(f"only {meta['n_sims']:,} simulations: Monte Carlo error on small probabilities is material")
    if meta.get('status') == 'provisional' and not meta.get('assumptions'):
        raise ContractError('provisional forecast without stated assumptions')
    return warn


def _atomic_write(path, write):
    path = Path(path)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp')
    os.close(fd)
    try:
        write(tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def publish(event_dir, summary, dist, meta, tag='pre_race'):
    """Write an immutable run folder, validate, then atomically switch the event's current forecast."""
    event_dir = Path(event_dir)
    meta = dict(meta, schema_version=SCHEMA_VERSION, definitions=DEFINITIONS,
                information=MODES_INFO.get(meta.get('mode'), ''))
    warnings = validate(summary, dist, meta)
    run = event_dir / 'runs' / f"{meta['created_utc'].replace(':', '').replace('+0000', 'Z')}_{meta.get('mode', 'auto')}"
    run.mkdir(parents=True, exist_ok=False)
    summary.to_csv(run / 'summary.csv', index=False)
    dist.to_csv(run / 'distribution.csv', index=False)
    (run / 'meta.json').write_text(json.dumps(dict(meta, warnings=warnings), indent=1, default=float))
    _atomic_write(event_dir / f'{tag}_summary.csv', lambda p: summary.to_csv(p, index=False))
    _atomic_write(event_dir / f'{tag}_distribution.csv', lambda p: dist.to_csv(p, index=False))
    _atomic_write(event_dir / f'{tag}_meta.json',
                  lambda p: Path(p).write_text(json.dumps(dict(meta, warnings=warnings, run=run.name), indent=1, default=float)))
    return run, warnings
