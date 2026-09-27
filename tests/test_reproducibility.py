"""Monte Carlo reproducibility: results depend on (spec, n_sims, seed) only - never on worker count,
pool reuse or scheduling. Needs the local data store to build a real race spec."""
import unittest

import numpy as np


class SimulationReproducibility(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from flatout import backtest, circuits, pipeline
            from flatout.model import PaceModels
            a, h = pipeline.load_state(log=lambda *x: None)
            ds = backtest.load_dataset(a, h, lambda *x: None)
            m = PaceModels().fit(ds, cutoff=(2026, 10), calibrate=False)
            cls.spec, cls.ctx = pipeline.build_spec(2026, 10, a, h, m, circuits.build(a, cutoff=(2026, 10)),
                                                    pipeline.load_sim_params(), mode='post_quali')
        except Exception as e:
            raise unittest.SkipTest(f'no local data store: {e}')
        from flatout import sim
        cls.sim = sim
        cls._chunk = sim.CHUNK
        sim.CHUNK = 4000                     # 3 chunks of 4,000 keeps the test fast

    @classmethod
    def tearDownClass(cls):
        cls.sim.CHUNK = cls._chunk

    def _run(self, **kw):
        return self.sim.simulate(self.spec, 12_000, batch=2000, seed=123, **kw)

    def assertSameAgg(self, a, b):
        for k in ('pos', 'dnf', 'stops', 'first', 'grid', 'sc', 'pts'):
            np.testing.assert_array_equal(a[k], b[k], err_msg=k)
        self.assertEqual(a['n'], b['n'])
        self.assertEqual(a['seq'], b['seq'])

    def test_worker_count_does_not_change_results(self):
        self.assertSameAgg(self._run(workers=1), self._run(workers=3))

    def test_shared_pool_matches_fresh_pool(self):
        with self.sim.pool(2) as ex:
            shared = self._run(workers=2, ex=ex)
        self.assertSameAgg(shared, self._run(workers=2))

    def test_different_seed_differs(self):
        a = self._run(workers=1)
        b = self.sim.simulate(self.spec, 12_000, batch=2000, seed=124, workers=1)
        self.assertFalse(np.array_equal(a['pos'], b['pos']))

    def test_positions_are_a_permutation_in_every_race(self):
        """Every simulated race assigns each position exactly once (no duplicates, no gaps)."""
        rng = np.random.default_rng(5)
        out = self.sim.run_batch(self.sim._spec_arrays(self.spec), 500, rng)
        pos = np.sort(out['pos'], 1)
        np.testing.assert_array_equal(pos, np.broadcast_to(np.arange(self.spec['D']), pos.shape))


if __name__ == '__main__':
    unittest.main()
