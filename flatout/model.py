"""Pace models trained on real outcomes.

RacePace : predicts fuel/tyre-corrected race pace (% vs field median) per driver.
QualiPace: predicts qualifying pace (% vs field median) so the grid can be simulated
           before qualifying has happened.

Each is a LightGBM + ridge blend. Uncertainty is not guessed: it is the RMSE of
walk-forward residuals, split by how much weekend information was available
(no weekend data / practice only / qualifying known).
"""
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from .config import MODELS, SEASON_WEIGHT

RACE_FEATURES = ['q_rel_pct', 'q_tm_gap', 'sq_rel_pct', 'fp_lr_pct', 'fp_lr_laps', 'fp_best_pct',
                 'sprint_pace_pct', 'form_pace', 'form_n', 'form_q', 'team_form_pace', 'team_form_q',
                 'tm_form_gap', 'form_consistency', 'rookie']
QUALI_FEATURES = ['sq_rel_pct', 'fp_best_pct', 'fp_lr_pct', 'sprint_pace_pct', 'form_q', 'form_pace',
                  'team_form_q', 'team_form_pace', 'form_n', 'rookie']


PACE_BANDS = [-np.inf, -1.0, -0.3, 0.4, np.inf]
BAND_NAMES = ['front', 'upper', 'mid', 'back']


def _robust_sd(e):
    e = np.asarray(e, float)
    return float(1.4826 * np.median(np.abs(e - np.median(e)))) if len(e) else np.nan


def info_level(df):
    has_q = df.q_rel_pct.notna() if 'q_rel_pct' in df else pd.Series(False, index=df.index)
    has_fp = df.fp_best_pct.notna() | df.get('sq_rel_pct', pd.Series(np.nan, index=df.index)).notna()
    return np.where(has_q, 'quali', np.where(has_fp, 'practice', 'none'))


class _Blend:
    def __init__(self, features, target, clip):
        self.features, self.target, self.clip = features, target, clip

    def _ridge_X(self, df, fit=False):
        X = df[self.features].astype(float)
        miss = X.isna().astype(float).add_suffix('_na')
        if fit:
            self.med = X.median()
        X = X.fillna(self.med).fillna(0)
        Z = pd.concat([X, miss], axis=1).values
        if fit:
            self.sc = StandardScaler().fit(Z)
        return self.sc.transform(Z)

    def fit(self, df, w):
        import lightgbm as lgb
        y = df[self.target].clip(*self.clip).values
        self.gbm = lgb.LGBMRegressor(n_estimators=350, learning_rate=0.03, num_leaves=7, min_child_samples=25,
                                     subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=2.0,
                                     verbose=-1, random_state=7)
        self.gbm.fit(df[self.features].astype(float), y, sample_weight=w)
        self.ridge = Ridge(alpha=3.0).fit(self._ridge_X(df, fit=True), y, sample_weight=w)
        self.blend = 0.5
        return self

    def predict(self, df):
        g = self.gbm.predict(df[self.features].astype(float))
        r = self.ridge.predict(self._ridge_X(df))
        return self.blend * g + (1 - self.blend) * r, g, r


def _weights(df, season_weight=None):
    sw = SEASON_WEIGHT if season_weight is None else season_weight
    w = df.year.map(sw).fillna(min(sw.values()) * 0.85).values.astype(float)
    if 'pace_laps' in df:
        w = w * np.clip(df.pace_laps.fillna(0).values / 20, 0.2, 1.0)
    if 'wet' in df:
        w = w * np.where(df.wet.fillna(False).astype(bool), 0.3, 1.0)
    return w


WEEKEND_Q = ('q_rel_pct', 'q_tm_gap', 'q_pos')
WEEKEND_FP = ('fp_lr_pct', 'fp_lr_laps', 'fp_best_pct', 'sq_rel_pct', 'sprint_pace_pct')


def _augment(df, masks, season_weight=None):
    """Stack copies of df with the given feature groups blanked; each copy gets an equal weight share."""
    w = _weights(df, season_weight)
    parts = [df.assign(**{c: np.nan for c in cols if c in df}) for cols in masks]
    return pd.concat(parts, ignore_index=True), np.concatenate([w / len(masks)] * len(masks))


def _train_rows(ds, target):
    return ds[ds[target].notna()].copy()


class PaceModels:
    """Race + quali pace models with walk-forward calibrated uncertainty.

    season_weight: {year: training-row weight}. None = config.SEASON_WEIGHT (hand-set 2026=1, 2025=0.6,
    2024=0.35). Relative to the newest season, so it is a recency/regulation-era prior."""

    season_weight = None   # class default: models pickled before this attribute existed

    def __init__(self, season_weight=None):
        self.season_weight = season_weight

    def fit(self, ds, cutoff=(9999, 99), calibrate=True):
        before = (ds.year < cutoff[0]) | ((ds.year == cutoff[0]) & (ds['round'] < cutoff[1]))
        tr = ds[before]
        rr = _train_rows(tr, 'pace_pct')
        rr = rr[rr.pace_laps >= 5]
        # Predictions are made at several points in a weekend (before FP, after FP, after quali),
        # so train on masked copies too - otherwise missing quali lands in barely-trained tree branches.
        ra, rw = _augment(rr, [(), WEEKEND_Q, WEEKEND_Q + WEEKEND_FP], self.season_weight)
        self.race = _Blend(RACE_FEATURES, 'pace_pct', (-4, 6)).fit(ra, rw)
        qr = _train_rows(tr, 'q_rel_pct')
        qa, qw = _augment(qr, [(), WEEKEND_FP], self.season_weight)
        self.quali = _Blend(QUALI_FEATURES, 'q_rel_pct', (-4, 6)).fit(qa, qw)
        self.sd = dict(race={'quali': 0.45, 'practice': 0.6, 'none': 0.75}, quali={'practice': 0.45, 'none': 0.6})
        self.blend = {'race': 0.5, 'quali': 0.5}
        if calibrate:
            self._calibrate(tr)
        return self

    def _calibrate(self, tr):
        """Walk-forward over the most recent races: blend weight + robust residual sd per info level
        (and, for race pace, per predicted pace band)."""
        events = tr[['year', 'round']].drop_duplicates().sort_values(['year', 'round']).values.tolist()
        test_events = events[-24:] if len(events) > 30 else events[len(events) // 2:]
        res = {'race': [], 'quali': []}
        wipe_fp = {k: np.nan for k in ('fp_lr_pct', 'fp_lr_laps', 'fp_best_pct', 'sq_rel_pct', 'sprint_pace_pct')}
        wipe_q = {k: np.nan for k in ('q_rel_pct', 'q_tm_gap', 'q_pos')}
        for y, r in test_events[::2]:  # every other event keeps it quick
            prior = tr[(tr.year < y) | ((tr.year == y) & (tr['round'] < r))]
            cur = tr[(tr.year == y) & (tr['round'] == r)]
            if prior[['year', 'round']].drop_duplicates().shape[0] < 8:
                continue
            m = PaceModels(self.season_weight).fit(prior, calibrate=False)
            for kind, target in (('race', 'pace_pct'), ('quali', 'q_rel_pct')):
                c = cur[cur[target].notna()]
                if kind == 'race':
                    c = c[c.pace_laps >= 10]
                if len(c) == 0:
                    continue
                if kind == 'race':   # score with qualifying known, before it, and before the weekend
                    variants = [(c, None), (c.assign(**wipe_q), None), (c.assign(**wipe_q, **wipe_fp), None)]
                else:
                    variants = [(c, 'practice'), (c.assign(**wipe_fp), 'none')]
                for v, lvl_fixed in variants:
                    _, g, rdg = getattr(m, kind).predict(v)
                    y_true = v[target].clip(-4, 6).values
                    # pace is relative, so remove the event-level bias
                    g = g - (np.mean(g) - np.mean(y_true))
                    rdg = rdg - (np.mean(rdg) - np.mean(y_true))
                    if lvl_fixed is None:
                        lvl = info_level(v)
                    else:
                        has_fp = v.fp_best_pct.notna() | v.sq_rel_pct.notna()
                        lvl = np.where(has_fp, 'practice', 'none')
                    res[kind] += list(zip(lvl, g, rdg, y_true))
        self.cv = {}
        for kind in ('race', 'quali'):
            if len(res[kind]) < 50:
                continue
            arr = pd.DataFrame(res[kind], columns=['lvl', 'g', 'r', 'y'])
            best = min(np.linspace(0, 1, 11), key=lambda b: np.mean(np.abs(b * arr.g + (1 - b) * arr.r - arr.y)))
            self.blend[kind] = float(best)
            getattr(self, kind).blend = float(best)
            pred = best * arr.g + (1 - best) * arr.r
            e = (pred - arr.y).values
            for lvl, idx in arr.groupby('lvl').groups.items():
                if len(idx) >= 20:
                    self.sd[kind][str(lvl)] = _robust_sd(e[arr.index.get_indexer(idx)])
            if kind == 'race':
                # front-runners are more predictable than the packed midfield: scale sd by predicted pace band
                band = pd.cut(pred, PACE_BANDS, labels=BAND_NAMES)
                tot = _robust_sd(e)
                self.band_mult = {b: float(np.clip(_robust_sd(e[(band == b).values]) / tot, 0.5, 1.8))
                                  for b in BAND_NAMES if (band == b).sum() >= 25}
            self.cv[kind] = dict(n=len(arr), mae=float(np.mean(np.abs(e))), robust_sd=_robust_sd(e),
                                 rmse=float(np.sqrt(np.mean(e ** 2))))

    def predict(self, df):
        """Returns df with race_mu, race_sd, quali_mu, quali_sd (all in % vs field median)."""
        out = df[['Driver', 'Team']].copy()
        mu, *_ = self.race.predict(df)
        out['race_mu'] = mu - np.median(mu)
        out['race_sd'] = [self.sd['race'].get(l, 0.7) for l in info_level(df)]
        bm = getattr(self, 'band_mult', {})
        if bm:
            band = pd.cut(out.race_mu, PACE_BANDS, labels=BAND_NAMES).astype(str)
            out['race_sd'] = out.race_sd * band.map(bm).fillna(1.0).values
        if 'q_rel_pct' in df and df.q_rel_pct.notna().all():
            out['quali_mu'], out['quali_sd'] = df.q_rel_pct.values, 0.0
        else:
            q, *_ = self.quali.predict(df)
            out['quali_mu'] = q - np.median(q)
            lvl = np.where(df.fp_best_pct.notna() | df.get('sq_rel_pct', pd.Series(np.nan, index=df.index)).notna(),
                           'practice', 'none')
            out['quali_sd'] = [self.sd['quali'].get(l, 0.6) for l in lvl]
        return out

    def save(self, name='pace_models'):
        with open(MODELS / f'{name}.pkl', 'wb') as f:
            pickle.dump(self, f)
        (MODELS / f'{name}_meta.json').write_text(json.dumps(dict(sd=self.sd, blend=self.blend,
                                                                  cv=getattr(self, 'cv', None)), indent=1))

    @staticmethod
    def load(name='pace_models'):
        with open(MODELS / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)

    def importance(self):
        return pd.DataFrame(dict(feature=RACE_FEATURES, gain=self.race.gbm.booster_.feature_importance('gain')))\
            .sort_values('gain', ascending=False)
