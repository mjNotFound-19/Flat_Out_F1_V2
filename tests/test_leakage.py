"""Temporal leakage: perturbing data from races after a cutoff must not change anything computed
for the cutoff race (features, circuit priors, fitted pace-model predictions, grid baseline).

Needs the local data store; skipped on a fresh clone.
"""
import copy
import unittest

import numpy as np
import pandas as pd

CUTOFF = (2026, 10)


def _after(df, cutoff=CUTOFF):
    return (df.year > cutoff[0]) | ((df.year == cutoff[0]) & (df['round'] > cutoff[1]))


class FutureDataPerturbation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from flatout import backtest, pipeline
            cls.analyses, cls.hist = pipeline.load_state(log=lambda *a: None)
            cls.ds = backtest.load_dataset(cls.analyses, cls.hist, lambda *a: None)
        except Exception as e:
            raise unittest.SkipTest(f'no local data store: {e}')
        rng = np.random.default_rng(7)
        # history: scramble finishing order, grid and pace of every later race
        h = cls.hist.copy()
        m = _after(h)
        h.loc[m, 'finish'] = rng.permutation(h.loc[m, 'finish'].values)
        h.loc[m, 'grid'] = rng.permutation(h.loc[m, 'grid'].values)
        h.loc[m, 'pace_pct'] = h.loc[m, 'pace_pct'] * -3 + 1
        h.loc[m, 'q_rel_pct'] = rng.permutation(h.loc[m, 'q_rel_pct'].values)
        cls.hist_p = h
        d = cls.ds.copy()
        md = _after(d)
        for c in ('pace_pct', 'finish', 'grid', 'q_rel_pct', 'form_pace', 'team_form_pace'):
            if c in d:
                d.loc[md, c] = rng.permutation(d.loc[md, c].values)
        cls.ds_p = d
        # race analyses: distort degradation, pit loss and base lap of later races
        a = copy.deepcopy(cls.analyses)
        for x in a:
            s = x['summary']
            if (s['year'], s['round']) > CUTOFF:
                s['base_lap'] = (s.get('base_lap') or 90) * 1.3
                s['pit_loss'] = (s.get('pit_loss') or 20) + 15
                s['deg'] = {k: v * 5 for k, v in (s.get('deg') or {}).items()}
        cls.analyses_p = a

    def test_form_features(self):
        from flatout import features
        entry = features.entry_list(*CUTOFF, self.hist)
        f0 = features.form(self.hist, *CUTOFF, entry)
        f1 = features.form(self.hist_p, *CUTOFF, entry)
        pd.testing.assert_frame_equal(f0, f1)

    def test_circuit_priors(self):
        from flatout import circuits
        c0 = circuits.build(self.analyses, cutoff=CUTOFF)
        c1 = circuits.build(self.analyses_p, cutoff=CUTOFF)
        self.assertEqual(sorted(c0), sorted(c1))
        for k in c0:
            for field in ('base_lap', 'pit_loss', 'sc_per_race', 'overtake_factor', 'n_races'):
                self.assertEqual(c0[k].get(field), c1[k].get(field), f'{k}.{field} depends on later races')

    def test_pace_model_predictions(self):
        from flatout import features
        from flatout.model import PaceModels
        f = features.event_features(*CUTOFF, self.hist, mode='post_quali')
        p0 = PaceModels().fit(self.ds, cutoff=CUTOFF, calibrate=False).predict(f)
        p1 = PaceModels().fit(self.ds_p, cutoff=CUTOFF, calibrate=False).predict(f)
        pd.testing.assert_frame_equal(p0.reset_index(drop=True), p1.reset_index(drop=True))

    def test_positive_controls(self):
        """The perturbation is strong enough to be detected once the cutoff moves past it."""
        from flatout import circuits, evaluate, features
        from flatout.model import PaceModels
        late = (2026, 16)
        c0 = circuits.build(self.analyses, cutoff=late)
        c1 = circuits.build(self.analyses_p, cutoff=late)
        self.assertTrue(any(c0[k].get('pit_loss') != c1[k].get('pit_loss') for k in c0 if not k.startswith('_')))
        entry = features.entry_list(*late, self.hist)
        self.assertFalse(features.form(self.hist, *late, entry).equals(features.form(self.hist_p, *late, entry)))
        self.assertFalse(np.array_equal(evaluate.grid_transition(self.hist, late),
                                        evaluate.grid_transition(self.hist_p, late)))
        f = features.event_features(*late, self.hist, mode='pre_weekend')
        p0 = PaceModels().fit(self.ds, cutoff=late, calibrate=False).predict(f)
        p1 = PaceModels().fit(self.ds_p, cutoff=late, calibrate=False).predict(f)
        self.assertFalse(np.allclose(p0.race_mu.values, p1.race_mu.values))

    def test_grid_baseline(self):
        from flatout import evaluate
        np.testing.assert_array_equal(evaluate.grid_transition(self.hist, CUTOFF),
                                      evaluate.grid_transition(self.hist_p, CUTOFF))


if __name__ == '__main__':
    unittest.main()
