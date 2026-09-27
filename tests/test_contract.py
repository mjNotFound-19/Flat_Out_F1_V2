"""Forecast contract: malformed or mislabelled forecasts must never be published."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from flatout import contract


def _forecast(D=20, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.random((D, D))
    for _ in range(200):                      # Sinkhorn: doubly stochastic like a real position matrix
        P /= P.sum(1, keepdims=True)
        P /= P.sum(0, keepdims=True)
    drivers = [f'D{i:02d}' for i in range(D)]
    dist = pd.DataFrame(P, columns=[f'P{k + 1}' for k in range(D)]).assign(Driver=drivers)
    dist = dist[['Driver'] + [f'P{k + 1}' for k in range(D)]]
    summ = pd.DataFrame(dict(Driver=drivers, win=P[:, 0], podium=P[:, :3].sum(1), points=P[:, :10].sum(1),
                             dnf=0.1, stop1=0.4, stop2=0.5, stop3p=0.1))
    meta = dict(year=2026, round=16, location='Sepang', identity=dict(circuit='Sepang'), mode='pre_weekend',
                n_sims=4_000_000, created_utc='2026-09-27T05:00:00+00:00', provenance={},
                status='provisional', assumptions=['pooled priors'])
    return summ, dist, meta


class ContractTests(unittest.TestCase):
    def test_valid_forecast_passes(self):
        self.assertEqual(contract.validate(*_forecast()), [])

    def test_wrong_circuit_rejected(self):
        s, d, m = _forecast()
        m['location'] = 'Sakhir'
        with self.assertRaises(contract.ContractError):
            contract.validate(s, d, m)

    def test_probabilities_must_be_coherent(self):
        s, d, m = _forecast()
        s.loc[0, 'win'] += 0.2
        with self.assertRaises(contract.ContractError):
            contract.validate(s, d, m)
        s, d, m = _forecast()
        d.iloc[0, 1:] = d.iloc[0, 1:] * 1.5
        with self.assertRaises(contract.ContractError):
            contract.validate(s, d, m)

    def test_provisional_needs_assumptions(self):
        s, d, m = _forecast()
        m['assumptions'] = []
        with self.assertRaises(contract.ContractError):
            contract.validate(s, d, m)

    def test_publish_is_immutable_and_switches_current(self):
        s, d, m = _forecast()
        with tempfile.TemporaryDirectory() as tmp:
            run, _ = contract.publish(tmp, s, d, m)
            cur = json.loads((Path(tmp) / 'pre_race_meta.json').read_text())
            self.assertEqual(cur['run'], run.name)
            self.assertEqual(cur['schema_version'], contract.SCHEMA_VERSION)
            self.assertIn('win', cur['definitions'])
            with self.assertRaises(FileExistsError):      # same run id is never overwritten
                contract.publish(tmp, s, d, m)
            bad = dict(m, location='Sakhir')
            with self.assertRaises(contract.ContractError):
                contract.publish(tmp, s, d, dict(bad, created_utc='2026-09-27T06:00:00+00:00'))
            # the failed publish left the current forecast untouched
            self.assertEqual(json.loads((Path(tmp) / 'pre_race_meta.json').read_text())['run'], run.name)


if __name__ == '__main__':
    unittest.main()
