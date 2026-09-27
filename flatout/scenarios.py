"""Precomputed what-if scenarios for the next event (static site: no live recomputation).

Each scenario perturbs one input of the forecast spec and re-simulates with the same seed (common random
numbers), so differences reflect the input, not Monte Carlo noise. The site shows the resulting win /
podium / expected-position changes; it never fabricates values for inputs outside this bundle.
Weather is not modelled by the simulator, so no weather scenario is offered.
"""
import numpy as np

from . import sim

SCENARIOS = [
    ('base', 'As forecast', {}),
    ('sc_half', 'Safety cars half as likely', dict(sc_mult=0.5)),
    ('sc_double', 'Safety cars twice as likely', dict(sc_mult=2.0)),
    ('deg_low', 'Tyres wear 30% less', dict(deg_mult=0.7)),
    ('deg_high', 'Tyres wear 50% more', dict(deg_mult=1.5)),
    ('pit_plus3', 'Pit stops cost 3 s more', dict(pit_add=3.0)),
    ('pace_wide', 'Pace less certain (sd x1.5)', dict(pace_sd_mult=1.5)),
    ('fav_slower', 'Favourite 0.2% slower', dict(fav_slower_pct=0.2)),
]
NOT_AVAILABLE = ['Rain / changing weather: not modelled by the simulator.',
                 'Grid penalties: use predict --grid after qualifying instead.']


def _apply(spec, ctx, change, fav):
    s = dict(spec)
    if 'sc_mult' in change:
        s['sc_lap'] = spec['sc_lap'] * change['sc_mult']
        s['vsc_lap'] = spec['vsc_lap'] * change['sc_mult']
    if 'deg_mult' in change:
        s['deg_mult'] = np.asarray(spec['deg_mult'], float) * change['deg_mult']
    if 'pit_add' in change:
        s['pit_loss'] = spec['pit_loss'] + change['pit_add']
    if 'pace_sd_mult' in change:
        s['pace_sd'] = np.asarray(spec['pace_sd'], float) * change['pace_sd_mult']
    if 'fav_slower_pct' in change:
        mu = np.asarray(spec['pace_mu'], float).copy()
        mu[fav] += change['fav_slower_pct'] / 100 * spec['base_lap']
        s['pace_mu'] = mu
    return s


def run(spec, ctx, n_sims=200_000, seed=31, workers=None, top=8):
    names = ctx['drivers']
    base = None
    out = []
    with sim.pool(sim.default_workers(workers)) as ex:
        for key, label, change in SCENARIOS:
            fav = int(np.argmax(base['pos'][:, 0])) if base is not None else 0
            agg = sim.simulate(_apply(spec, ctx, change, fav), n_sims, seed=seed, ex=ex)
            if base is None:
                base = agg
            P = agg['pos'] / agg['n']
            D = len(names)
            rows = [dict(Driver=names[d], win=float(P[d, 0]), podium=float(P[d, :3].sum()),
                         exp_pos=float(np.dot(P[d], np.arange(1, D + 1))), dnf=float(agg['dnf'][d] / agg['n']),
                         stops=float(np.dot(agg['stops'][d] / agg['stops'][d].sum(), np.arange(6))))
                    for d in range(D)]
            rows.sort(key=lambda r: -r['win'])
            out.append(dict(key=key, label=label, change=change, drivers=rows[:top],
                            p_sc=float(1 - agg['sc'][0] / agg['n'])))
    return dict(n_sims=n_sims, seed=seed, scenarios=out, not_available=NOT_AVAILABLE,
                note='Precomputed with common random numbers; each row changes one input. Not a live model.')
