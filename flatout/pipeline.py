"""Glue: turn an event into a simulator spec, run it, summarise, and persist."""
import json
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import circuits as circ_mod, evaluate, features, ingest, race, sim, strategy
from . import events
from .config import DERIVED, DRY, MODELS, OUTPUT, POINTS, venue
from .model import PaceModels


def load_sim_params():
    p = MODELS / 'sim_params.json'
    params = dict(sim.DEFAULTS)
    if p.exists():
        params.update(json.loads(p.read_text()).get('params', {}))
    return params


def event_info(year, rnd, log=print):
    """Schedule row + resolved event identity. The circuit comes only from the sourced registry
    (flatout/events.py); conflicting or unknown metadata raises EventResolutionError."""
    sch_p = ingest.STORE / str(year) / 'schedule.parquet'
    sch = pd.read_parquet(sch_p) if sch_p.exists() else ingest.load_schedule(year)
    info = sch[sch['round'] == rnd].iloc[0].to_dict()
    try:
        start = json.loads(info.get('sessions') or '{}').get('R')
    except (TypeError, ValueError):
        start = None
    ident = events.resolve(year, rnd, info['event'], info['location'], info['country'], start)
    info['location_listed'], info['location'] = info['location'], ident.circuit
    info['identity'] = ident.as_dict()
    if ident.status == 'override':
        log(f"  venue: {ident.title} at {ident.circuit_name}, {ident.host_country} "
            f"(schedule lists '{ident.listed_location}', {ident.listed_country}; sourced override)")
    return info


def reliability(hist, year, rnd, teams, g_rate, season_k=0.0):
    """Per-team DNF probability per race: this season's record shrunk hard toward a prior rate.

    season_k = 0 (champion): the prior is the multi-season rate g_rate.
    season_k > 0 (challenger R1): the prior is this season's field-wide rate so far, itself shrunk toward
    g_rate with season_k car-races of prior weight - so a regulation era with a different failure rate
    moves the prior within a few races. Uses only races before (year, rnd)."""
    past = hist[(hist.year == year) & (hist['round'] < rnd)]
    prev = hist[(hist.year == year - 1)]
    if season_k > 0:
        g_rate = float((past.dnf.sum() + season_k * g_rate) / (len(past) + season_k))
    out = {}
    for t in teams:
        cur = past[past.Team == t]
        pv = prev[prev.Team == t]
        k = 12.0
        num = cur.dnf.sum() + 0.3 * pv.dnf.sum() + k * g_rate
        den = len(cur) + 0.3 * len(pv) + k
        out[t] = float(num / den)
    return out


def actual_grid(year, rnd, drivers):
    """Grid after penalties if the race is stored, else qualifying order."""
    res = ingest.read(year, rnd, 'R', 'results')
    if res is not None and res.GridPosition.notna().any():
        g = res.set_index('Abbreviation').GridPosition
        g = g.where(g > 0, len(drivers))
    else:
        q = ingest.read(year, rnd, 'Q', 'results')
        if q is None:
            return None
        g = q.set_index('Abbreviation').Position
    g = g.reindex(drivers)
    if g.isna().all():
        return None
    # unknown drivers go to the back, then re-rank to 1..D
    g = g.fillna(len(drivers) + 1).rank(method='first')
    return g.astype(int).values


def _fallback_base_lap(year, rnd):
    """Race lap estimate for a venue with no race history: ~6% off the best weekend lap."""
    best = []
    for code in ('Q', 'SQ', 'FP3', 'FP2', 'FP1'):
        laps = ingest.read(year, rnd, code, 'laps')
        if laps is not None and laps.LapTime.notna().any():
            best.append(laps.LapTime.min())
    return min(best) * 1.06 if best else None


# A circuit with no race in the dataset has unknown car-circuit fit: widen driver pace uncertainty.
# Assumption (not fitted): x1.25. Tested only on one new venue so far (Madrid 2026); revisit in Stage 3.
NEW_VENUE_SD_MULT = 1.25


def pooled_base_lap(circuits, ident):
    """Base race lap for an unseen circuit: median seconds-per-km over circuits with a fitted base lap,
    times the official length. Returns (seconds, explanation)."""
    rates = []
    for key, c in circuits.items():
        if key.startswith('_') or not c.get('base_lap'):
            continue
        try:
            km = events.circuit(key).get('length_km')
        except KeyError:
            km = None
        if km:
            rates.append(c['base_lap'] / km)
    if not rates or not ident.get('length_km'):
        raise events.EventResolutionError(f"no base-lap prior for {ident['circuit']}: needs circuit length and history")
    r = float(np.median(rates))
    return r * ident['length_km'], (f"base lap {r * ident['length_km']:.1f}s = pooled median {r:.2f} s/km over "
                                    f"{len(rates)} circuits x {ident['length_km']} km (not Sepang data)")


def build_spec(year, rnd, analyses, hist, models, circuits, params, use_grid=True, grid_override=None,
               mode=None):
    info = event_info(year, rnd)
    entry = features.entry_list(year, rnd, hist)
    f = features.event_features(year, rnd, hist, entry, mode=mode)
    pred = models.predict(f)
    known = hist[(hist.year == year) & (hist['round'] == rnd)]
    ident = info['identity']
    n_laps = int(known.n_laps.iloc[0]) if len(known) else ident.get('laps')
    if not n_laps:
        raise events.EventResolutionError(
            f"{year} R{rnd}: no lap count (not raced yet, and the registry has no sourced laps for "
            f"{ident['circuit']} in {year}). Add the official lap count to flatout/registry/circuits.json.")
    have_base = (circuits.get(info['location']) or {}).get('base_lap')
    assumptions = []
    base = None
    if not have_base:
        base = _fallback_base_lap(year, rnd)
        if base:
            assumptions.append(f"base lap {base:.1f}s from this weekend's practice (x1.06)")
        else:
            base, how = pooled_base_lap(circuits, ident)
            assumptions.append(how)
    c = circ_mod.for_event(circuits, info['location'], n_laps=n_laps, base_lap=base)
    new_venue = not c.get('n_races')
    if new_venue:
        assumptions.append(f"no {info['location']} race in the dataset: pit loss, tyre wear, stops, safety-car rate "
                           f"and overtaking use pooled all-circuit priors; driver pace sd x{NEW_VENUE_SD_MULT}")
    D = len(entry)
    drivers = entry.Driver.tolist()
    grid = None
    if grid_override:
        grid = np.array([grid_override.get(d, D) for d in drivers])
        grid = pd.Series(grid).rank(method='first').astype(int).values
    elif use_grid and mode in (None, 'auto', 'post_quali'):
        grid = actual_grid(year, rnd, drivers)
    rel = reliability(hist, year, rnd, entry.Team.unique(), c['dnf_rate'], params.get('rel_season_k', 0.0))
    if params.get('rel_model') == 'hazard':      # challenger R2: exposure-based gamma-Poisson hazard
        from . import reliability as rel_mod
        haz = rel_mod.team_probs(rel_mod.fit(hist, year, rnd), year, entry.Team.unique(), info['location'], c['n_laps'])
        rel = haz or rel
    p_dnf = np.array([rel[t] for t in entry.Team])
    lap1_share = c['lap1_dnf_share']
    n = c['n_laps']
    # per-lap hazard such that lap1 carries its historical share of retirements
    h = -np.log(1 - p_dnf) / (n - 1 + params['lap1_dnf_mult'])
    exp_grid = grid if grid is not None else pd.Series(pred.quali_mu.values).rank(method='first').values
    cands = strategy.candidates(c)
    probs = strategy.driver_probs(cands, c, exp_grid)
    lens, comps, nst = strategy.table(cands)
    exp_dnfs = p_dnf.sum()
    sc_indep = max(0.05, c['sc_per_race'] - params['sc_from_dnf'] * 0.6 * exp_dnfs)
    vsc_indep = max(0.05, c['vsc_per_race'] - params['sc_from_dnf'] * 0.4 * exp_dnfs)
    spec = dict(
        D=D, n_laps=n, base_lap=c['base_lap'], fuel=c['fuel'], offset=c['offset'], deg=c['deg'],
        max_stint=c['max_stint'], pit_loss=c['pit_loss'], lap_sd=c['lap_sd'],
        overtake_factor=c['overtake_factor'], sc_lap=sc_indep / n, vsc_lap=vsc_indep / n,
        red_share=c.get('red_share', 0.08),
        pace_mu=pred.race_mu.values / 100 * c['base_lap'],
        pace_sd=pred.race_sd.values / 100 * c['base_lap'] * (NEW_VENUE_SD_MULT if new_venue else 1.0),
        quali_mu=pred.quali_mu.values, quali_sd=np.maximum(pred.quali_sd.values, 0.05),
        grid=grid, dnf_lap=h, deg_mult=np.ones(D),
        strat_p=probs, strat_lens=lens, strat_comps=comps, strat_n=nst, params=params,
    )
    ctx = dict(info=info, drivers=drivers, teams=entry.Team.tolist(), pred=pred, features=f,
               assumptions=assumptions, new_venue=new_venue, params_used=dict(params),
               circuit=c, cands=cands, strat_p=probs, grid=grid, p_dnf=p_dnf,
               grid_T=evaluate.grid_transition(hist, (year, rnd)), grid_blend=params.get('grid_blend', 0.0))
    return spec, ctx


def summarise(agg, ctx):
    D = len(ctx['drivers'])
    n = agg['n']
    P = agg['pos'] / n
    w = ctx.get('grid_blend', 0.0)
    if w > 0 and ctx.get('grid') is not None:
        # stacking: blend in the empirical grid->finish prior for the value of track position
        P = evaluate._sinkhorn((1 - w) * P + w * evaluate.grid_baseline(ctx['grid'], ctx['grid_T']))
    pos = np.arange(1, D + 1)
    rows = []
    for d, drv in enumerate(ctx['drivers']):
        cdf = np.cumsum(P[d])
        stops = agg['stops'][d] / agg['stops'][d].sum()
        first = agg['first'][d].copy()
        first[0] = 0
        fl = np.repeat(np.arange(len(first)), first.astype(np.int64) // max(1, int(first.sum() // 20000) or 1))
        seqs = sorted(agg['seq'].get(d, {}).items(), key=lambda kv: -kv[1])
        tot = sum(c for _, c in seqs) or 1
        # headline = most likely sequence within the most likely stop count
        modal_n = int(np.argmax(stops[1:4]) + 1) if stops[1:4].sum() > 0 else 1
        by_n = [(sim.decode_seq(k), c / tot) for k, c in seqs if sim.decode_seq(k).count('-') == modal_n]
        rest = [(sim.decode_seq(k), c / tot) for k, c in seqs if sim.decode_seq(k).count('-') != modal_n]
        top = (by_n[:1] + sorted(by_n[1:] + rest, key=lambda x: -x[1]))[:3]
        rows.append(dict(
            Driver=drv, Team=ctx['teams'][d],
            grid=int(ctx['grid'][d]) if ctx['grid'] is not None else float(np.dot(agg['grid'][d] / n, pos)),
            win=P[d, 0], podium=P[d, :3].sum(), points=P[d, :10].sum(), dnf=agg['dnf'][d] / n,
            exp_pos=float(np.dot(P[d], pos)), exp_pts=float(np.dot(P[d, :10], POINTS)),
            p10=int(np.searchsorted(cdf, 0.1) + 1), p50=int(np.searchsorted(cdf, 0.5) + 1),
            p90=int(np.searchsorted(cdf, 0.9) + 1),
            pace_pct=float(ctx['pred'].race_mu.iloc[d]), pace_sd=float(ctx['pred'].race_sd.iloc[d]),
            stop1=stops[1], stop2=stops[2], stop3p=stops[3:].sum(),
            exp_stops=float(np.dot(stops, np.arange(6))),
            modal_stops=modal_n,
            strategy=top[0][0] if top else '', strategy_p=top[0][1] if top else np.nan,
            alt_strategies='; '.join(f'{s} {p:.0%}' for s, p in top[1:]),
            first_stop_p50=float(np.median(fl)) if len(fl) else np.nan,
            first_stop_p25=float(np.percentile(fl, 25)) if len(fl) else np.nan,
            first_stop_p75=float(np.percentile(fl, 75)) if len(fl) else np.nan,
        ))
    df = pd.DataFrame(rows).sort_values('exp_pos').reset_index(drop=True)
    df.insert(0, 'rank', np.arange(1, D + 1))
    dist = pd.DataFrame(P, columns=[f'P{i}' for i in pos])
    dist.insert(0, 'Driver', ctx['drivers'])
    sc = agg['sc'] / n
    return df, dist, dict(p_sc=float(1 - sc[0]), exp_sc=float(np.dot(sc, np.arange(len(sc)))))


def out_dir(year, rnd):
    p = OUTPUT / str(year) / f'R{rnd:02d}'
    p.mkdir(parents=True, exist_ok=True)
    return p


MODE_INPUTS = {   # weekend sessions a forecast mode may use; everything else of the weekend is "missing"
    'pre_weekend': [], 'post_fp': ['FP1', 'FP2', 'FP3'], 'post_sprint': ['FP1', 'SQ', 'S'],
    'post_quali': ['FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q'],
}


def save_prediction(year, rnd, df, dist, extra, ctx, n_sims, tag='pre_race'):
    """Validate and publish a forecast (flatout/contract.py): immutable runs/<utc>_<mode>/ folder, then an
    atomic switch of the event's current pre_race_* files."""
    from . import contract, provenance
    d = out_dir(year, rnd)
    now = datetime.now(timezone.utc)
    sessions = [s for s in ingest.available_sessions(year, rnd) if s != 'R']
    mode = extra.get('mode') or 'auto'
    allowed = MODE_INPUTS.get(mode)
    used = sessions if allowed is None else [s for s in sessions if s in allowed]
    ident = ctx['info'].get('identity', {})
    models_meta = MODELS / 'pace_models_meta.json'
    meta = dict(year=year, round=rnd, event=ident.get('title', ctx['info']['event']), location=ctx['info']['location'],
                created=now.strftime('%Y%m%d_%H%M%S'), created_utc=now.isoformat(timespec='seconds'),
                n_sims=n_sims, sessions_used=used,
                inputs_missing=[s for s in ('FP1', 'FP2', 'FP3', 'SQ', 'S', 'Q') if s not in used]
                               + ([] if ctx['grid'] is not None else ['starting grid']),
                information_cutoff_utc=now.isoformat(timespec='seconds'),
                grid_known=ctx['grid'] is not None, **extra,
                identity=ident, assumptions=ctx.get('assumptions', []),
                status='provisional' if ctx.get('new_venue') else 'standard',
                provenance=dict(code=provenance.code_identity(), data=provenance.data_identity(),
                                model_meta_sha256=provenance.sha256_file(models_meta) if models_meta.exists() else None,
                                sim_params=ctx.get('params_used', {}),
                                sim_params_sha256=provenance.sha256_json(ctx.get('params_used', {})),
                                seed=dict(root=2026, scheme=f"SeedSequence(root).spawn(ceil(n/{sim.CHUNK})); fixed {sim.CHUNK:,}-sim chunks")),
                circuit={k: v for k, v in ctx['circuit'].items() if k not in ('start_compound',)},
                strategies=[dict(seq=strategy.label(x['seq']), lens=x['lens'], delta=round(x['delta'], 2))
                            for x in ctx['cands'][:12]])
    run, warnings = contract.publish(d, df, dist, meta, tag=tag)
    for w in warnings:
        print(f'  ! {w}')
    return d


def load_state(log=print):
    analyses = race.analyze_all(log=log)
    hp = DERIVED / 'history.parquet'
    hist = features.history_table(analyses)
    return analyses, hist


def run_event(year, rnd, n_sims, analyses, hist, models=None, params=None, workers=None, use_grid=True,
              grid_override=None, cutoff=None, mode=None, log=print):
    params = params or load_sim_params()
    cutoff = cutoff or (year, rnd)
    circuits = circ_mod.build(analyses, cutoff=cutoff)
    models = models or PaceModels.load()
    spec, ctx = build_spec(year, rnd, analyses, hist, models, circuits, params, use_grid, grid_override, mode)
    t0 = time.time()
    bar = sim.ProgressBar(n_sims) if log is print else None   # so a 5-10 minute run doesn't look frozen
    agg = sim.simulate(spec, n_sims, workers=workers, progress=(lambda d, t: bar.update(d)) if bar else None)
    df, dist, extra = summarise(agg, ctx)
    extra['sim_seconds'] = round(time.time() - t0, 1)
    ctx['spec'] = spec          # kept in memory for follow-up analyses (strategy recommendation); not saved
    return df, dist, extra, ctx
