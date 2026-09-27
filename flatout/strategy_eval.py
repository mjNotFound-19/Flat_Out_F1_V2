"""Strategy recommendation (distinct from the behaviour forecast).

Behaviour forecast (sim.py / pipeline.summarise): what each team is likely to do - a probability over
candidate plans learned from teams' habits, plus the simulator's safety-car opportunism.

Recommendation (this module): for one driver, force each candidate plan in turn while the rest of the
field keeps its behaviour model, and compare outcomes under common random numbers (same seed, so every
plan faces the same safety cars, pace draws and incidents). The best plan is the one with the lowest
mean finishing position under the model.

Scope and limits - reported with every result:
  * static pre-race plan choice: stop count, compounds and planned stint lengths; the in-race reaction
    to neutralisations uses the simulator's existing rules for every plan. It is NOT a rolling-horizon
    policy (no decision re-optimisation during the race, no opponent reaction to this driver's plan).
  * the value of a plan is model-dependent: it inherits every assumption of the tyre, pit-loss, passing
    and safety-car models (pooled priors at a new circuit). It is not evidence of real race-time gain.
"""
import numpy as np

from . import sim, strategy

SCOPE = ('Static pre-race plan comparison under the model, common random numbers. Not a rolling-horizon '
         'policy; opponents do not react; values inherit all model assumptions.')


def recommend(spec, ctx, drivers=None, top_plans=6, n_sims=40_000, seed=77, workers=None, ex=None):
    """For each driver: behaviour-mix outcome and forced-plan outcomes for the most likely `top_plans`
    candidates. Returns {driver: dict(behaviour=..., plans=[...], best=..., scope=SCOPE)}."""
    D = spec['D']
    names = ctx['drivers']
    idx = [names.index(d) for d in (drivers or names)]
    cands = ctx['cands']
    base = sim.simulate(spec, n_sims, workers=workers, seed=seed, ex=ex)
    out = {}
    for d in idx:
        order = np.argsort(-spec['strat_p'][d])[:top_plans]
        plans = []
        for k in order:
            s = dict(spec)
            p = np.array(spec['strat_p'], float, copy=True)
            p[d] = 0.0
            p[d, k] = 1.0
            s['strat_p'] = p
            agg = sim.simulate(s, n_sims, workers=workers, seed=seed, ex=ex)
            plans.append(dict(plan=strategy.label(cands[k]['seq']), lens=[int(x) for x in cands[k]['lens']],
                              p_behaviour=float(spec['strat_p'][d, k]), **_outcome(agg, d, D)))
        best = min(plans, key=lambda x: x['exp_pos'])
        out[names[d]] = dict(behaviour=_outcome(base, d, D), plans=plans, best=best['plan'],
                             gain_pos=float(_outcome(base, d, D)['exp_pos'] - best['exp_pos']),
                             n_sims=n_sims, scope=SCOPE)
    return out


def _outcome(agg, d, D):
    P = agg['pos'][d] / agg['n']
    pos = np.arange(1, D + 1)
    # Monte Carlo standard error of the mean position (for "is this difference noise?")
    var = float(np.dot(P, (pos - np.dot(P, pos)) ** 2))
    return dict(exp_pos=float(np.dot(P, pos)), se_pos=float(np.sqrt(var / agg['n'])),
                podium=float(P[:3].sum()), points=float(P[:10].sum()), exp_pts=float(agg['pts'][d] / agg['n']),
                dnf=float(agg['dnf'][d] / agg['n']))
