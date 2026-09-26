"""Circuit outlines for the website: the fastest lap's X/Y trace from the most recent stored race at
a venue, rotated like the official map and normalised to a 1000-unit box. Cached per venue."""
import json
import logging

import numpy as np

from . import ingest
from .config import CACHE, DERIVED, venue

TRACKS = DERIVED / 'tracks'
TRACKS.mkdir(parents=True, exist_ok=True)


def _rotate(xy, deg):
    a = np.deg2rad(deg)
    return xy @ np.array([[np.cos(a), np.sin(a)], [-np.sin(a), np.cos(a)]])


def outline(location, n_points=260):
    loc = venue(location)
    path = TRACKS / f"{loc.replace(' ', '_')}.json"
    if path.exists():
        return json.loads(path.read_text())
    sched = ingest.all_schedules()
    races = [(int(y), int(r)) for y, r, l in sched[['year', 'round', 'location']].itertuples(index=False)
             if venue(l) == loc and (ingest.session_dir(int(y), int(r), 'R') / 'meta.json').exists()]
    if not races:
        return None
    year, rnd = max(races)
    import fastf1
    logging.getLogger('fastf1').setLevel(logging.ERROR)
    fastf1.Cache.enable_cache(str(CACHE))
    s = fastf1.get_session(year, rnd, 'R')
    s.load(laps=True, telemetry=True, weather=False, messages=False)
    pos = s.laps.pick_fastest().get_pos_data()
    ci = s.get_circuit_info()
    xy = _rotate(pos[['X', 'Y']].to_numpy(float), ci.rotation)
    corners = _rotate(ci.corners[['X', 'Y']].to_numpy(float), ci.rotation)
    lo, span = xy.min(0), (xy.max(0) - xy.min(0)).max()
    norm = lambda p: (p - lo) / span * 1000
    xy, corners = norm(xy), norm(corners)
    top = xy[:, 1].max()
    xy[:, 1], corners[:, 1] = top - xy[:, 1], top - corners[:, 1]   # SVG y grows downward
    idx = np.linspace(0, len(xy) - 1, n_points).astype(int)
    out = dict(location=loc, source=f'{year} R{rnd}', w=float(xy[:, 0].max()), h=float(top),
               points=[[round(float(x), 1), round(float(y), 1)] for x, y in xy[idx]],
               corners=[dict(n=int(c.Number), x=round(float(p[0]), 1), y=round(float(p[1]), 1))
                        for c, p in zip(ci.corners.itertuples(), corners)])
    path.write_text(json.dumps(out))
    return out
