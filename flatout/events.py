"""Event identity: which physical circuit a season/round is raced on.

An event has several separate identities that must not be conflated:
  * commercial title (e.g. 'Bahrain Grand Prix')      * season / round
  * host country and circuit (e.g. Sepang, Malaysia)  * session times (UTC)
Resolution uses the sourced registry in flatout/registry/circuits.json:

  1. A sourced per-event override (registry "events") wins. It must acknowledge exactly what
     FastF1 lists, so a changed feed is noticed rather than silently re-mapped.
  2. Otherwise the FastF1 location string must appear in some circuit's `fastf1_locations`,
     and that circuit's country must equal the listed country.
  3. Anything else raises EventResolutionError with the fix to apply. There is no fallback to
     "the venue this event name used last year".
"""
import json
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path

REGISTRY = Path(__file__).resolve().parent / 'registry' / 'circuits.json'


class EventResolutionError(ValueError):
    pass


@dataclass(frozen=True)
class EventIdentity:
    year: int
    round: int
    title: str                    # commercial event name
    circuit: str                  # canonical circuit key (used by history / priors / site)
    circuit_name: str
    host_country: str
    listed_location: str          # what the schedule feed said
    listed_country: str
    length_km: float | None
    laps: int | None              # official scheduled laps when known for this season
    laps_source: str | None
    race_start_utc: str | None
    status: str                   # 'verified' (registry agrees) | 'override' (sourced per-event mapping)
    sources: tuple = field(default_factory=tuple)
    registry_version: str = ''
    note: str = ''

    def as_dict(self):
        d = asdict(self)
        d['sources'] = list(self.sources)
        return d


@lru_cache(maxsize=1)
def registry():
    reg = json.loads(REGISTRY.read_text(encoding='utf-8'))
    alias = {}
    for key, c in reg['circuits'].items():
        for loc in c['fastf1_locations']:
            if loc in alias and alias[loc] != key:
                raise EventResolutionError(f'registry: location {loc!r} maps to both {alias[loc]} and {key}')
            alias[loc] = key
    reg['_alias'] = alias
    return reg


def circuit_key(location):
    """Canonical circuit key for a FastF1 location string (historical data path). Raises if unknown."""
    key = registry()['_alias'].get(location)
    if key is None:
        raise EventResolutionError(
            f'unknown schedule location {location!r}: add it to fastf1_locations of the right circuit in '
            f'{REGISTRY.name} (with a source) - locations are never guessed')
    return key


def circuit(key):
    return registry()['circuits'][key]


def resolve(year, rnd, event, location, country, race_start_utc=None):
    """EventIdentity for one schedule row. Raises EventResolutionError on unknown or conflicting data."""
    reg = registry()
    ov = reg.get('events', {}).get(f'{int(year)}-{int(rnd)}')
    if ov:
        listed = ov.get('fastf1_listed', {})
        if listed.get('location') != location or listed.get('country') != country:
            raise EventResolutionError(
                f'{year} R{rnd}: override expects FastF1 to list {listed}, feed now says '
                f'{{location: {location!r}, country: {country!r}}}. Re-verify the venue and update the override.')
        key = ov['circuit']
        c = reg['circuits'][key]
        if c['country'] != ov['host_country']:
            raise EventResolutionError(f'{year} R{rnd}: override host {ov["host_country"]} != circuit country {c["country"]}')
        if race_start_utc and ov.get('race_start_utc') and race_start_utc[:16] != ov['race_start_utc'][:16]:
            raise EventResolutionError(
                f'{year} R{rnd}: race start {race_start_utc} differs from sourced {ov["race_start_utc"]}; re-verify')
        return EventIdentity(
            year=int(year), round=int(rnd), title=ov.get('title', event), circuit=key, circuit_name=c['name'],
            host_country=c['country'], listed_location=location, listed_country=country,
            length_km=c.get('length_km'), laps=ov.get('laps') or _season_laps(c, year),
            laps_source=(ov['sources'][0] if ov.get('laps') else _laps_source(c, year)),
            race_start_utc=ov.get('race_start_utc') or race_start_utc, status='override',
            sources=tuple(ov.get('sources', ())), registry_version=reg['_version'], note=ov.get('note', ''))
    key = circuit_key(location)
    c = reg['circuits'][key]
    if c['country'] != country:
        raise EventResolutionError(
            f'{year} R{rnd} {event!r}: FastF1 lists {location!r} with country {country!r}, but {c["name"]} is in '
            f'{c["country"]}. If the title and host differ, add a sourced entry under "events" in {REGISTRY.name}.')
    return EventIdentity(
        year=int(year), round=int(rnd), title=event, circuit=key, circuit_name=c['name'], host_country=c['country'],
        listed_location=location, listed_country=country, length_km=c.get('length_km'),
        laps=_season_laps(c, year), laps_source=_laps_source(c, year), race_start_utc=race_start_utc,
        status='verified', sources=(c['source'],), registry_version=reg['_version'])


def _season_laps(c, year):
    """Official lap count only for the season it was sourced for (layouts and distances change)."""
    return c.get('laps') if c.get('source_season') == int(year) else None


def _laps_source(c, year):
    return c.get('source') if c.get('source_season') == int(year) else None


def validate_schedule(sched):
    """Resolve every row of a schedule frame; returns (identities, errors) without raising."""
    ok, errors = [], []
    for r in sched.itertuples(index=False):
        start = None
        try:
            start = json.loads(r.sessions).get('R') if isinstance(getattr(r, 'sessions', None), str) else None
        except (TypeError, ValueError):
            pass
        try:
            ok.append(resolve(r.year, r.round, r.event, r.location, r.country, start))
        except EventResolutionError as e:
            errors.append(str(e))
    return ok, errors
