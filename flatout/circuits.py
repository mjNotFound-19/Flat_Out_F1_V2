"""Per-circuit simulation parameters learned from past races, walk-forward safe.

Every estimate is a recency/season-weighted mean shrunk toward the global prior
by how many races back it, so a new venue (e.g. Madrid 2026) still gets sane
numbers and a venue with many races mostly speaks for itself.

Two things are season-relative because regulations/tyres move them a lot:
  * degradation: older races are rescaled by the season-to-season deg ratio
    measured at venues visited in both seasons;
  * planned stops: this season's average stop count + the venue's historical
    offset from its own season's average.
"""
from collections import Counter, defaultdict

import numpy as np

from .config import DRY, SEASON_WEIGHT

# Global priors used when nothing better exists.
PRIOR = dict(pit_loss=21.5, fuel=-0.055, lap_sd=0.45, sc_per_race=0.55, vsc_per_race=0.35,
             offset={'SOFT': -0.45, 'MEDIUM': 0.0, 'HARD': 0.35},
             deg={'SOFT': 0.09, 'MEDIUM': 0.055, 'HARD': 0.035})
SHRINK = dict(pit_loss=1.0, fuel=3.0, sc=4.0, deg=2.0, passes=2.0, stops=0.5)
STOPS_SD = 0.6


def _w(summary, cutoff_year):
    return SEASON_WEIGHT.get(summary['year'], 0.3) * (1.0 if summary['year'] == cutoff_year else 0.8)


def _ws(summary, cutoff_year):
    """Strategy habits change fast between seasons: weight the current one heavily."""
    return 1.0 if summary['year'] == cutoff_year else 0.2 if summary['year'] == cutoff_year - 1 else 0.1


def before(a, cutoff):
    s = a['summary']
    return (s['year'], s['round']) < cutoff


def _wmean(vals):
    vals = [(v, w) for v, w in vals if v is not None and np.isfinite(v)]
    if not vals:
        return None
    v, w = np.array(vals, float).T
    return float(np.average(v, weights=w))


def _shrunk(vals, prior, k):
    vals = [(v, w) for v, w in vals if v is not None and np.isfinite(v)]
    if not vals:
        return float(prior)
    v, w = np.array(vals, float).T
    return float((np.sum(v * w) + k * prior) / (w.sum() + k))


def _mean_deg(s):
    d = [s['deg'][c] for c in ('MEDIUM', 'HARD', 'SOFT') if c in s['deg']]
    return float(np.mean(d)) if d else None


def _deg_factor(dry, cutoff_year):
    """deg multiplier that maps each season onto the cutoff season, from shared venues."""
    by = defaultdict(lambda: defaultdict(list))
    for a in dry:
        s = a['summary']
        md = _mean_deg(s)
        if md and md > 0.005:
            by[s['location']][s['year']].append(md)
    years = sorted({y for v in by.values() for y in v})
    out = {cutoff_year: 1.0}
    for y in years:
        if y == cutoff_year:
            continue
        logs = [np.log(np.mean(v[cutoff_year]) / np.mean(v[y])) for v in by.values() if y in v and cutoff_year in v]
        n = len(logs)
        out[y] = float(np.exp(np.clip(np.mean(logs), -1.2, 1.2) * n / (n + 3))) if n else 1.0
    return out


def _race_stops(a):
    """Mean executed stops of classified dry-tyre finishers (SC stops included)."""
    d = a['drivers']
    d = d[d.classified.astype(bool) & d.start_compound.isin(DRY)]
    return float(d.n_stops.clip(0, 4).mean()) if len(d) >= 8 else None


def _neutralised(a):
    s = a['summary']
    return (s['sc'] + s['vsc'] + s['red']) > 0


def _stops_dist(mu):
    n = np.array([1, 2, 3])
    p = np.exp(-(n - mu) ** 2 / (2 * STOPS_SD ** 2))
    p /= p.sum()
    return {0: 0.0, 1: float(p[0]), 2: float(p[1]), 3: float(p[2])}


def build(analyses, cutoff=(9999, 99)):
    """analyses: list from race.analyze_all(). Returns {location: params, '_global': params}."""
    races = [a for a in analyses if a['summary']['code'] == 'R' and before(a, cutoff)]
    cutoff_year = cutoff[0] if cutoff[0] < 9999 else max((a['summary']['year'] for a in races), default=2026)
    dry = [a for a in races if not a['summary']['wet'] and a['summary'].get('fit_ok', True)]
    W = lambda a: _w(a['summary'], cutoff_year)
    dfac = _deg_factor(dry, cutoff_year)

    # ---------- global values
    g = dict(deg_year_factor=dfac)
    g['pit_loss'] = _wmean([(a['summary']['pit_loss'], W(a)) for a in dry]) or PRIOR['pit_loss']
    g['fuel'] = _wmean([(a['summary']['fuel'], W(a)) for a in dry]) or PRIOR['fuel']
    g['lap_sd'] = _wmean([(a['summary']['lap_sd'], W(a)) for a in dry]) or PRIOR['lap_sd']
    g['sc_per_race'] = _wmean([(a['summary']['sc'] + a['summary']['red'], W(a)) for a in races]) or PRIOR['sc_per_race']
    g['vsc_per_race'] = _wmean([(a['summary']['vsc'], W(a)) for a in races]) or PRIOR['vsc_per_race']
    reds = _wmean([(a['summary']['red'], W(a)) for a in races])
    g['red_share'] = float(np.clip((reds or 0.05) / max(g['sc_per_race'], 0.1), 0.03, 0.4))
    g['pass_rate'] = _wmean([(a['summary']['passes'] / max(a['summary']['green_laps'], 1), W(a)) for a in dry]) or 0.4
    g['offset'], g['deg'] = {}, {}
    for comp in DRY:
        off = [(a['summary']['offset'][comp] - a['summary']['offset'].get('MEDIUM', 0), W(a))
               for a in dry if comp in a['summary']['offset'] and 'MEDIUM' in a['summary']['offset']]
        dg = [(a['summary']['deg'][comp] * dfac.get(a['summary']['year'], 1.0), W(a))
              for a in dry if comp in a['summary']['deg']]
        g['offset'][comp] = _wmean(off) if off else PRIOR['offset'][comp]
        g['deg'][comp] = max(0.01, _wmean(dg)) if dg else PRIOR['deg'][comp]

    # DNF behaviour (per car per race) and lap-1 share
    cars = [(d, a['summary']) for a in races for d in a['drivers'].to_dict('records')]
    g['dnf_rate'] = _wmean([(1.0 if d['dnf'] else 0.0, _w(s, cutoff_year)) for d, s in cars]) or 0.08
    lap1 = [1.0 if d['dnf'] and (d.get('laps_done') or 99) <= 1 else 0.0 for d, s in cars if d['dnf']]
    g['lap1_dnf_share'] = float(np.mean(lap1)) if lap1 else 0.25
    gains = [d['lap1_gain'] for d, s in cars if d.get('lap1_gain') is not None and np.isfinite(d['lap1_gain'])]
    g['lap1_gain_sd'] = float(np.std(gains)) if gains else 1.8

    # Planned stops. Executed stops include SC 'free' stops, so convert each race to a
    # planned-equivalent count by removing the average SC effect, then model
    # planned = this season's mean + the venue's offset from its own season's mean.
    clean = [a for a in dry if not a['summary']['red']]
    per_race = [(a, _race_stops(a)) for a in clean]
    per_race = [(a, m) for a, m in per_race if m is not None]
    neu = [m for a, m in per_race if _neutralised(a)]
    calm = [m for a, m in per_race if not _neutralised(a)]
    sc_effect = float(np.clip(np.mean(neu) - np.mean(calm), 0, 1.0)) if len(neu) >= 3 and len(calm) >= 3 else 0.35
    g['sc_stop_effect'] = sc_effect
    per_race = [(a, m - sc_effect * _neutralised(a)) for a, m in per_race]
    season_mean = defaultdict(list)
    for a, m in per_race:
        season_mean[a['summary']['year']].append(m)
    season_mean = {y: float(np.mean(v)) for y, v in season_mean.items()}
    n_cur = len([1 for a, _ in per_race if a['summary']['year'] == cutoff_year])
    prev = [season_mean[y] for y in sorted(season_mean) if y < cutoff_year][-1:] or [1.3]
    mu_global = float(np.clip((season_mean.get(cutoff_year, 0) * n_cur + prev[0] * 2) / (n_cur + 2), 1.0, 3.0))
    g['stops_mu'] = mu_global
    g['stops_dist'] = _stops_dist(mu_global)
    start_by_bucket = defaultdict(Counter)
    trans = defaultdict(Counter)
    for a in clean:
        w = _ws(a['summary'], cutoff_year)
        for d in a['drivers'].to_dict('records'):
            if d['classified'] and d.get('start_compound') in DRY:
                start_by_bucket[_bucket(d['grid'])][d['start_compound']] += w
        if len(a['stints']):
            for _, st in a['stints'].sort_values(['Driver', 'stint']).groupby('Driver'):
                st = st.to_dict('records')
                for p, nx in zip(st, st[1:]):
                    if not nx['sc_stop'] and p['compound'] in DRY and nx['compound'] in DRY:
                        trans[p['compound']][nx['compound']] += w
    g['start_compound'] = {b: _norm(c, {'MEDIUM': 0.6, 'SOFT': 0.25, 'HARD': 0.15}) for b, c in start_by_bucket.items()}
    frac_rows = _stint_fractions(clean, cutoff_year)
    g['stint_frac'] = _mean_fracs(frac_rows, None)
    g['transitions'] = {p: _norm(trans[p], {c: 1 / 3 for c in DRY}) for p in DRY}
    g['max_stint'] = _max_stints(dry)
    g['year_scale'] = _year_scale(dry)

    # ---------- per circuit
    by_loc = defaultdict(list)
    for a in races:
        by_loc[a['summary']['location']].append(a)
    out = {'_global': g}
    for loc, rs in by_loc.items():
        rs = sorted(rs, key=lambda a: (a['summary']['year'], a['summary']['round']))
        drs = [a for a in rs if not a['summary']['wet'] and a['summary'].get('fit_ok', True)]
        latest = rs[-1]['summary']
        c = dict(location=loc, event=latest['event'], n_races=len(rs), n_laps=latest['n_laps'])
        valid = [a for a in drs if a['summary'].get('fit_ok', True) and a['summary']['base_lap']]
        if valid:
            ref = valid[-1]['summary']
            scale = g['year_scale'].get(cutoff_year, 1.0) / g['year_scale'].get(ref['year'], 1.0)
            c['base_lap'] = ref['base_lap'] * scale
        else:
            c['base_lap'] = None   # filled from the weekend's own laps in for_event
        c['pit_loss'] = _shrunk([(a['summary']['pit_loss'], W(a)) for a in drs], g['pit_loss'], SHRINK['pit_loss'])
        c['fuel'] = _shrunk([(a['summary']['fuel'], W(a)) for a in drs], g['fuel'], SHRINK['fuel'])
        c['lap_sd'] = _shrunk([(a['summary']['lap_sd'], W(a)) for a in drs], g['lap_sd'], 2.0)
        c['sc_per_race'] = _shrunk([(a['summary']['sc'] + a['summary']['red'], W(a)) for a in rs],
                                   g['sc_per_race'], SHRINK['sc'])
        c['vsc_per_race'] = _shrunk([(a['summary']['vsc'], W(a)) for a in rs], g['vsc_per_race'], SHRINK['sc'])
        prate = _shrunk([(a['summary']['passes'] / max(a['summary']['green_laps'], 1), W(a)) for a in drs],
                        g['pass_rate'], SHRINK['passes'])
        c['overtake_factor'] = float(np.clip(prate / g['pass_rate'], 0.15, 3.0))
        c['offset'], c['deg'] = {}, {}
        for comp in DRY:
            off = [(a['summary']['offset'][comp] - a['summary']['offset'].get('MEDIUM', 0), W(a))
                   for a in drs if comp in a['summary']['offset'] and 'MEDIUM' in a['summary']['offset']]
            dg = [(a['summary']['deg'][comp] * dfac.get(a['summary']['year'], 1.0), W(a))
                  for a in drs if comp in a['summary']['deg']]
            c['offset'][comp] = _shrunk(off, g['offset'][comp], SHRINK['deg'])
            c['deg'][comp] = max(0.008, _shrunk(dg, g['deg'][comp], SHRINK['deg']))
        # a venue's stop count relative to its season is stable year to year, so every year counts
        offs = [(m - season_mean.get(a['summary']['year'], m), 1.0)
                for a, m in per_race if a['summary']['location'] == loc]
        c['stops_offset'] = _shrunk(offs, 0.0, SHRINK['stops'])
        c['stops_mu'] = float(np.clip(mu_global + c['stops_offset'], 0.9, 3.0))
        c['stops_dist'] = _stops_dist(c['stops_mu'])
        c['stint_frac'] = _mean_fracs([r for r in frac_rows if r[0] == loc], g['stint_frac'])
        out[loc] = c
    return out


def for_event(circuits, location, n_laps=None, base_lap=None):
    """Params for a location, falling back to the global prior for unseen venues."""
    g = circuits['_global']
    c = dict(circuits.get(location) or dict(
        location=location, n_races=0, n_laps=n_laps or 57, base_lap=base_lap or 92.0,
        pit_loss=g['pit_loss'], fuel=g['fuel'], lap_sd=g['lap_sd'], sc_per_race=g['sc_per_race'],
        vsc_per_race=g['vsc_per_race'], overtake_factor=1.0, offset=dict(g['offset']), deg=dict(g['deg']),
        stops_mu=g['stops_mu'], stops_dist=dict(g['stops_dist']), stint_frac=g['stint_frac']))
    if n_laps:
        c['n_laps'] = n_laps
    if base_lap and not c.get('base_lap'):
        c['base_lap'] = base_lap
    if not c.get('base_lap'):
        c['base_lap'] = 92.0
    for k in ('dnf_rate', 'lap1_dnf_share', 'lap1_gain_sd', 'start_compound', 'max_stint', 'transitions',
              'red_share'):
        c[k] = g[k]
    return c


def _stint_fractions(races, cutoff_year):
    """(location, n_stops, fractions, weight) for drivers whose stops were all planned (green)."""
    rows = []
    for a in races:
        s = a['summary']
        if not len(a['stints']):
            continue
        w = _ws(s, cutoff_year)
        ok = {d['Driver'] for d in a['drivers'].to_dict('records') if d['classified']}
        for drv, st in a['stints'].groupby('Driver'):
            st = st.sort_values('stint')
            if drv not in ok or st.sc_stop.iloc[1:].any() or not 2 <= len(st) <= 4:
                continue
            lens = st.laps.values.astype(float)
            if lens.min() < 3:
                continue
            rows.append((s['location'], len(st) - 1, lens / lens.sum(), w))
    return rows


def _mean_fracs(rows, prior, k=6.0):
    out = {}
    for n in (1, 2, 3):
        r = [(f, w) for _, nn, f, w in rows if nn == n]
        base = np.array(prior[n]) if prior and n in prior else np.full(n + 1, 1.0 / (n + 1))
        if r:
            F = np.array([f for f, _ in r])
            W = np.array([w for _, w in r])
            m = (np.sum(F * W[:, None], 0) + k * base) / (W.sum() + k)
        else:
            m = base
        out[n] = list(m / m.sum())
    return out


def _bucket(grid):
    return 'front' if grid <= 6 else 'mid' if grid <= 14 else 'back'


def _norm(counter, default):
    tot = sum(counter.values())
    if tot <= 0:
        return dict(default)
    return {k: v / tot for k, v in counter.items()}


def _max_stints(dry):
    out = {}
    for comp in DRY:
        lens = [s['laps'] for a in dry for s in a['stints'].to_dict('records') if s.get('compound') == comp]
        out[comp] = int(np.percentile(lens, 97)) if len(lens) > 20 else {'SOFT': 25, 'MEDIUM': 38, 'HARD': 50}[comp]
    return out


def _year_scale(dry):
    """Relative lap-time level per year using venues raced in consecutive years."""
    by = defaultdict(dict)
    for a in dry:
        s = a['summary']
        if s['base_lap']:
            by[s['location']][s['year']] = s['base_lap']
    years = sorted({y for v in by.values() for y in v})
    scale = {years[0]: 1.0} if years else {}
    for y0, y1 in zip(years, years[1:]):
        r = [v[y1] / v[y0] for v in by.values() if y0 in v and y1 in v]
        scale[y1] = scale[y0] * (float(np.median(r)) if r else 1.0)
    return scale
