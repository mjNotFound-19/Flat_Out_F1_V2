"""Event identity: the circuit an event is raced on must come from sourced data, never a guess.

    python -m unittest discover -s tests -v
"""
import copy
import json
import unittest

from flatout import events


class RegistryTests(unittest.TestCase):
    def test_every_circuit_is_sourced_and_complete(self):
        for key, c in events.registry()['circuits'].items():
            with self.subTest(circuit=key):
                self.assertTrue(c['source'].startswith('https://www.formula1.com/'))
                self.assertIn(key, c['fastf1_locations'], 'canonical key must resolve to itself')
                self.assertGreater(c['length_km'], 3.0)
                self.assertLess(c['length_km'], 7.5)

    def test_locations_map_to_exactly_one_circuit(self):
        seen = {}
        for key, c in events.registry()['circuits'].items():
            for loc in c['fastf1_locations']:
                self.assertNotIn(loc, seen, f'{loc} claimed by {seen.get(loc)} and {key}')
                seen[loc] = key


class ResolutionTests(unittest.TestCase):
    def test_bahrain_title_resolves_to_sepang(self):
        ident = events.resolve(2026, 16, 'Bahrain Grand Prix', 'Kuala Lumpur', 'Bahrain', '2026-10-04T07:00:00+00:00')
        self.assertEqual(ident.circuit, 'Sepang')
        self.assertEqual(ident.host_country, 'Malaysia')
        self.assertEqual(ident.laps, 56)
        self.assertEqual(ident.status, 'override')
        self.assertNotEqual(ident.circuit, 'Sakhir')

    def test_title_country_mismatch_without_override_fails(self):
        reg = events.registry()
        saved = copy.deepcopy(reg['events'])
        try:
            reg['events'].pop('2026-16')
            with self.assertRaises(events.EventResolutionError):
                events.resolve(2026, 16, 'Bahrain Grand Prix', 'Kuala Lumpur', 'Bahrain')
        finally:
            reg['events'].clear()
            reg['events'].update(saved)

    def test_override_rejects_a_changed_feed(self):
        with self.assertRaises(events.EventResolutionError):
            events.resolve(2026, 16, 'Bahrain Grand Prix', 'Sakhir', 'Bahrain')

    def test_override_rejects_a_moved_race_time(self):
        with self.assertRaises(events.EventResolutionError):
            events.resolve(2026, 16, 'Bahrain Grand Prix', 'Kuala Lumpur', 'Bahrain', '2026-10-04T11:00:00+00:00')

    def test_unknown_location_fails(self):
        with self.assertRaises(events.EventResolutionError):
            events.circuit_key('Kuala Lumpur City')

    def test_historical_events_resolve_to_their_real_venue(self):
        self.assertEqual(events.resolve(2025, 4, 'Bahrain Grand Prix', 'Sakhir', 'Bahrain').circuit, 'Sakhir')
        self.assertEqual(events.resolve(2026, 14, 'Spanish Grand Prix', 'Madrid', 'Spain').circuit, 'Madrid')
        self.assertEqual(events.resolve(2025, 9, 'Spanish Grand Prix', 'Barcelona', 'Spain').circuit, 'Barcelona')

    def test_lap_count_only_for_its_sourced_season(self):
        self.assertEqual(events.resolve(2026, 3, 'Japanese Grand Prix', 'Suzuka', 'Japan').laps, 53)
        self.assertIsNone(events.resolve(2024, 4, 'Japanese Grand Prix', 'Suzuka', 'Japan').laps)

    def test_whole_stored_schedule_resolves(self):
        try:
            from flatout import ingest
            sched = ingest.all_schedules()
        except Exception as e:                      # no local store (fresh clone): nothing to check
            self.skipTest(f'no schedule store: {e}')
        ok, errors = events.validate_schedule(sched)
        self.assertEqual(errors, [])
        self.assertEqual(len(ok), len(sched))


class SupersededForecastTests(unittest.TestCase):
    def test_sakhir_forecast_is_not_active_for_round_16(self):
        from flatout.config import OUTPUT
        d = OUTPUT / '2026' / 'R16'
        meta = d / 'pre_race_meta.json'
        if meta.exists():
            m = json.loads(meta.read_text())
            self.assertNotEqual(m.get('location'), 'Sakhir', 'a Sakhir forecast is active for the Sepang race')
        marker = d / 'superseded' / '20260927_sakhir_venue_error' / 'SUPERSEDED.json'
        if marker.exists():
            self.assertEqual(json.loads(marker.read_text())['status'], 'invalid_for_event')


class LayoutTests(unittest.TestCase):
    def test_sepang_traced_layout(self):
        import numpy as np
        from pathlib import Path
        d = json.loads((Path(events.__file__).parent / 'registry' / 'layouts' / 'Sepang.json').read_text())
        self.assertEqual(d['kind'], 'official_map_trace')
        self.assertFalse(d['telemetry'])
        self.assertNotIn('speed', d)                     # never pretend a traced map is telemetry
        self.assertEqual(sorted(c['n'] for c in d['corners']), list(range(1, 16)))
        P = np.array(d['points'])
        step = np.sqrt((np.diff(np.vstack([P, P[:1]]), axis=0) ** 2).sum(1))
        self.assertLess(step.max(), 4 * np.median(step), 'gap in the traced loop')


if __name__ == '__main__':
    unittest.main()
