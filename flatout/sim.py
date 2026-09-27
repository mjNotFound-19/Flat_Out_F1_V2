"""Vectorised lap-by-lap Monte Carlo race simulator.

A batch of B races runs in lock-step: every array is (B sims x D drivers) and each
lap is a handful of numpy ops, so 4M races finish in about a minute across the
CPU cores. Per lap, per car:

  lap time = base + race-day pace + fuel(lap) + compound offset + deg * tyre age
             + cliff + noise  [+ pit loss]
  SC  : slow laps, field bunched behind the leader, pit stops ~45% cheaper
  VSC : slow laps with gaps frozen, pit stops ~35% cheaper
  pass: a car that would catch the one ahead gets past with probability
        sigmoid((pace advantage - threshold) / scale); otherwise it is held behind.

Outputs are streamed into counters (finishing-position matrix, DNF, points,
stop counts, first-stop lap, executed compound sequences, SC count), so memory
stays constant no matter how many races run.
"""
import contextlib
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .config import DRY, POINTS
from .strategy import CLIFF_RATE

# Global behaviour parameters (calibrated by `python -m flatout calibrate`).
DEFAULTS = dict(
    pass_thr=0.55,        # s of lap-time advantage for a 50% pass chance at an average circuit
    pass_scale=0.25,      # softness of the pass curve
    min_gap=0.35,         # s a held-up car sits behind the car in front
    start_gap=0.22,       # s between grid slots at the line
    start_sd=0.45,        # s of launch / turn-1 randomness
    lap1_thr_mult=0.45,   # passing is much easier on lap 1
    pace_sd_mult=1.0,     # scales model pace uncertainty
    lap_sd_mult=1.0,      # scales per-lap noise
    sc_gap=0.9,           # s between cars in the SC train
    sc_lap_factor=1.35, vsc_lap_factor=1.28,
    sc_pit_factor=0.55, vsc_pit_factor=0.65,
    sc_from_dnf=0.35,     # chance a retirement brings out an SC/VSC
    opp_window=0.45,      # pit under SC if planned stop is within this share of the stint
    extra_stop_p=0.35,    # chance of a 'free' extra stop under SC on old tyres
    slow_stop_p=0.04, slow_stop_mean=4.0,
    lap1_dnf_mult=6.0,
    pace_df=4.0,          # tail heaviness of race-day pace (Student-t degrees of freedom)
    new_venue_mode='sd_mult',  # unseen circuit: 'sd_mult' (champion: driver pace sd x1.25) or 'param_unc' (R3)
    force_unseen=False,   # evaluation only: forecast as if the circuit had never been raced
    rel_model='hazard',   # retirement model: 'hazard' (promoted 2026-09-27, E11) or 'shrink' (previous champion)
    rel_season_k=0.0,     # reliability prior: 0 = multi-season rate; >0 = season rate shrunk with this weight (not tuned)
    grid_blend=0.15,      # stacking weight of the empirical grid->finish prior (when the grid is known)
)

# robust sd (1.4826 * MAD) of a unit Student-t, so t draws can be scaled to a robust spread
_T_MAD = {3.0: 1.134, 4.0: 1.098, 5.0: 1.076, 6.0: 1.062, 8.0: 1.046}


def _spec_arrays(spec):
    return {k: (np.asarray(v) if isinstance(v, (list, tuple)) else v) for k, v in spec.items()}


def run_batch(spec, B, rng):
    s = spec
    P = s['params']
    D, N = s['D'], s['n_laps']
    base = s['base_lap']
    comp_off = np.array([s['offset'][c] for c in DRY])
    comp_deg = np.array([s['deg'][c] for c in DRY])
    comp_max = np.array([s['max_stint'][c] for c in DRY], float)

    # --- race-day pace and grid
    # heavy-tailed race-day pace: Student-t scaled so its robust (MAD) spread equals pace_sd
    df_t = P.get('pace_df', 4.0)
    z = rng.standard_t(df_t, (B, D)) / _T_MAD.get(df_t, 1.0) if df_t < 30 else rng.standard_normal((B, D))
    pace = s['pace_mu'] + s['pace_sd'] * P['pace_sd_mult'] * z
    if s['grid'] is not None:
        grid = np.broadcast_to(np.asarray(s['grid']), (B, D)).copy()
    else:
        q = s['quali_mu'] + s['quali_sd'] * rng.standard_normal((B, D))
        grid = np.argsort(np.argsort(q, axis=1), axis=1) + 1
    # --- strategies: sample a plan per car, jitter its pit laps
    cum_p = np.cumsum(s['strat_p'], axis=1)
    k = (rng.random((B, D, 1)) > cum_p[None]).sum(axis=2)
    k = np.minimum(k, cum_p.shape[1] - 1)
    lens = s['strat_lens'][k].astype(float)          # B,D,4
    comps = s['strat_comps'][k].astype(np.int16)      # B,D,4
    nplan = s['strat_n'][k].astype(np.int16)          # B,D
    stop_laps = np.cumsum(lens, axis=2)[..., :3]
    stop_laps = np.clip(np.round(stop_laps + rng.normal(0, 2.5, stop_laps.shape)), 3, N - 2)
    stop_laps = np.maximum.accumulate(stop_laps + np.arange(3) * 1e-3, axis=2)
    stop_laps = np.where(np.arange(3) < nplan[..., None], stop_laps, np.inf)
    stop_laps = np.concatenate([stop_laps, np.full((B, D, 1), np.inf)], axis=2)

    # --- per-car current-stint state (updated only when a car pits)
    stop_i = np.zeros((B, D), np.int16)
    comp = comps[..., 0].copy()
    next_stop = stop_laps[..., 0].copy()
    cur_len = lens[..., 0].copy()
    deg_mult = np.broadcast_to(np.asarray(s['deg_mult'], float), (B, D))
    off_c, deg_c, max_c = comp_off[comp], comp_deg[comp] * deg_mult, comp_max[comp]
    age = np.zeros((B, D))
    stops_done = np.zeros((B, D), np.int16)
    first_stop = np.zeros((B, D), np.int16)
    exec_seq = comp.astype(np.int32) + 1              # base-4 digits of executed compounds

    # --- retirement lap pre-sampled from the per-lap hazard (lap 1 is riskier)
    h = np.broadcast_to(np.asarray(s['dnf_lap'], float), (B, D))
    E = rng.exponential(1.0, (B, D))
    h1 = h * P['lap1_dnf_mult']
    retire_at = np.where(E < h1, 1, 1 + np.ceil((E - h1) / np.maximum(h, 1e-9))).astype(np.int32)
    retire_at = np.minimum(retire_at, N + 1)
    alive = np.ones((B, D), bool)
    retire = np.full((B, D), N + 1, np.int32)

    cum = (grid - 1) * P['start_gap'] + P['start_sd'] * rng.standard_normal((B, D))
    sc_left = np.zeros(B, np.int16)
    vsc_left = np.zeros(B, np.int16)
    sc_age = np.zeros(B, np.int16)
    n_sc = np.zeros(B, np.int16)
    h_sc, h_vsc = s['sc_lap'], s['vsc_lap']
    thr = P['pass_thr'] / s['overtake_factor']
    lap_sd = s['lap_sd'] * P['lap_sd_mult']
    fuel = s['fuel']
    soft_max = comp_max[0]

    for lap in range(1, N + 1):
        # ---- retirements
        die = alive & (retire_at == lap)
        alive &= ~die
        retire[die] = lap
        # ---- neutralisations (independent + triggered by retirements)
        free = (sc_left == 0) & (vsc_left == 0)
        trig = die.any(axis=1) & (rng.random(B) < P['sc_from_dnf'])
        r = rng.random(B)
        new_sc = free & ((r < h_sc) | (trig & (rng.random(B) < 0.6)))
        new_vsc = free & ~new_sc & ((r > 1 - h_vsc) | trig)
        new_red = new_sc & (rng.random(B) < s.get('red_share', 0.0))   # red flag: free tyre change
        if new_sc.any():
            sc_left[new_sc] = rng.integers(3, 7, int(new_sc.sum()))
        if new_vsc.any():
            vsc_left[new_vsc] = rng.integers(2, 4, int(new_vsc.sum()))
        n_sc += new_sc
        under_sc, under_vsc = sc_left > 0, vsc_left > 0
        neutral = under_sc | under_vsc
        sc_age = np.where(new_sc | new_vsc, 0, sc_age + 1)

        # ---- pit decisions
        pit = alive & (lap >= next_stop)
        fresh = neutral & (sc_age <= 1)
        if fresh.any():
            rows = np.nonzero(fresh)[0]
            ns, si, npl = next_stop[rows], stop_i[rows], nplan[rows]
            opp = (si < npl) & (ns - lap <= P['opp_window'] * cur_len[rows])
            extra = (si >= npl) & (age[rows] > 14) & (N - lap > 8) & (rng.random((len(rows), D)) < P['extra_stop_p'])
            pit[rows] |= alive[rows] & (opp | extra)
        if new_red.any():
            pit[new_red] |= alive[new_red]
        if lap == 1 or lap >= N:
            pit[:] = False
        tpit = None
        if pit.any():
            bi, di = np.nonzero(pit)
            planned = stop_i[bi, di] < nplan[bi, di]
            stop_i[bi, di] += planned
            si = stop_i[bi, di]
            extra_comp = 0 if (N - lap) <= soft_max * 0.9 else 1
            newc = np.where(planned, comps[bi, di, np.minimum(si, 3)], extra_comp).astype(np.int16)
            comp[bi, di] = newc
            off_c[bi, di] = comp_off[newc]
            deg_c[bi, di] = comp_deg[newc] * deg_mult[bi, di]
            max_c[bi, di] = comp_max[newc]
            next_stop[bi, di] = stop_laps[bi, di, np.minimum(si, 3)]
            cur_len[bi, di] = np.where(planned, lens[bi, di, np.minimum(si, 3)], N - lap)
            age[bi, di] = 0
            stops_done[bi, di] += 1
            sd = stops_done[bi, di]
            first_stop[bi, di] = np.where(sd == 1, lap, first_stop[bi, di])
            exec_seq[bi, di] = np.where(sd <= 4, exec_seq[bi, di] + (newc + 1) * 4 ** sd.astype(np.int32),
                                        exec_seq[bi, di])
            factor = np.where(new_red[bi], 0.0, np.where(under_sc[bi], P['sc_pit_factor'],
                                                         np.where(under_vsc[bi], P['vsc_pit_factor'], 1.0)))
            cost = factor * s['pit_loss']
            slow = rng.random(len(bi)) < P['slow_stop_p']
            cost = cost + slow * rng.exponential(P['slow_stop_mean'], len(bi))
            tpit = np.zeros((B, D))
            tpit[bi, di] = cost

        # ---- lap times
        age += 1
        over = np.maximum(age - 0.9 * max_c, 0.0)
        t = (base + pace + fuel * lap + off_c + deg_c * age + CLIFF_RATE * over * over
             + lap_sd * rng.standard_normal((B, D)))
        if lap == 1:
            t += 0.5 * lap_sd * rng.standard_normal((B, D))
        if neutral.any():
            t[under_sc] = base * P['sc_lap_factor']
            vr = under_vsc & ~under_sc
            t[vr] = base * P['vsc_lap_factor'] + 0.1 * (t[vr] - base)
        if tpit is not None:
            t += tpit
        dead_key = 1e7 + (N - retire) * 1e3
        tent = np.where(alive, cum + t, dead_key)
        new_cum = tent.copy()                          # VSC rows keep gaps as they are

        # ---- green rows: passing
        G = np.nonzero(~neutral)[0]
        if len(G):
            start = np.where(alive[G], cum[G], dead_key[G])
            order = np.argsort(start, axis=1)
            final = np.ascontiguousarray(np.take_along_axis(tent[G], order, 1).T)      # D x nG
            pitting = np.ascontiguousarray(np.take_along_axis(pit[G], order, 1).T)
            U = rng.random(final.shape)
            lthr = thr * (P['lap1_thr_mult'] if lap == 1 else 1.0)
            run_max = final[0].copy()
            mg, ps = P['min_gap'], P['pass_scale']
            for p in range(1, D):
                me = final[p]
                adv = run_max - me
                close = adv > -mg
                if not close.any():
                    np.maximum(run_max, me, out=run_max)
                    continue
                z = np.clip((adv + mg - lthr) / ps, -30, 30)
                ok = close & (pitting[p - 1] | (run_max > 1e6) | (U[p] * (1.0 + np.exp(-z)) < 1.0))
                held = close & ~ok
                me = np.where(held, run_max + mg, np.where(ok, np.minimum(me, run_max - 0.05), me))
                final[p] = me
                np.maximum(run_max, me, out=run_max)
            out = np.empty((len(G), D))
            np.put_along_axis(out, order, final.T, 1)
            new_cum[G] = out

        # ---- SC rows: bunch the field behind the leader, keep order
        S = np.nonzero(under_sc)[0]
        if len(S):
            ts = tent[S]
            o2 = np.argsort(ts, axis=1)
            ts2 = np.take_along_axis(ts, o2, 1)
            queue = ts2[:, :1] + np.arange(D) * P['sc_gap']
            b = np.where(ts2 < 1e6, np.minimum(ts2, queue), ts2)
            out = np.empty_like(b)
            np.put_along_axis(out, o2, b, 1)
            new_cum[S] = out
        cum = new_cum
        sc_left = np.maximum(sc_left - 1, 0)
        vsc_left = np.maximum(vsc_left - 1, 0)

    finish_order = np.argsort(cum, axis=1)
    pos = np.empty((B, D), np.int16)
    np.put_along_axis(pos, finish_order, np.broadcast_to(np.arange(D, dtype=np.int16), (B, D)), 1)
    return dict(pos=pos, alive=alive, grid=grid, stops=stops_done, first_stop=first_stop,
                seq=exec_seq, n_sc=n_sc, plan=k)


def _aggregate(out, D, N, agg=None):
    if agg is None:
        agg = dict(pos=np.zeros((D, D)), dnf=np.zeros(D), pts=np.zeros(D), stops=np.zeros((D, 6)),
                   first=np.zeros((D, N + 1)), grid=np.zeros((D, D)), sc=np.zeros(8), seq={}, n=0,
                   gain=np.zeros(D))
    pos, B = out['pos'], out['pos'].shape[0]
    for d in range(D):
        agg['pos'][d] += np.bincount(pos[:, d], minlength=D)
        agg['grid'][d] += np.bincount(out['grid'][:, d] - 1, minlength=D)
        agg['stops'][d] += np.bincount(np.minimum(out['stops'][:, d], 5), minlength=6)
        agg['first'][d] += np.bincount(out['first_stop'][:, d], minlength=N + 1)[:N + 1]
        fin = out['alive'][:, d]
        vals, cnt = np.unique(out['seq'][fin, d], return_counts=True)
        sd = agg['seq'].setdefault(d, {})
        for v, c in zip(vals.tolist(), cnt.tolist()):
            sd[v] = sd.get(v, 0) + c
    agg['dnf'] += (~out['alive']).sum(0)
    pts = np.zeros(D + 1)
    pts[:10] = POINTS
    agg['pts'] += np.where(out['alive'], pts[pos], 0).sum(0)
    agg['gain'] += (out['grid'] - 1 - pos).sum(0)
    agg['sc'] += np.bincount(np.minimum(out['n_sc'], 7), minlength=8)
    agg['n'] += B
    return agg


def _merge(a, b):
    for k in ('pos', 'dnf', 'pts', 'stops', 'first', 'grid', 'sc', 'gain'):
        a[k] += b[k]
    a['n'] += b['n']
    for d, sd in b['seq'].items():
        tgt = a['seq'].setdefault(d, {})
        for v, c in sd.items():
            tgt[v] = tgt.get(v, 0) + c
    return a


def _draw_circuit(spec, unc, rng):
    """One draw of the unknown track-level parameters (a new venue): pit loss additive, the rest log-normal.
    Shared by every car in the batch - the uncertainty is about the circuit, not about individual drivers."""
    s = dict(spec)
    z = rng.standard_normal(4)
    s['pit_loss'] = max(5.0, spec['pit_loss'] + unc['pit_loss_sd'] * z[0])
    s['deg_mult'] = np.asarray(spec['deg_mult'], float) * np.exp(unc['deg_log_sd'] * z[1])
    f = np.exp(unc['sc_log_sd'] * z[2])
    s['sc_lap'], s['vsc_lap'] = spec['sc_lap'] * f, spec['vsc_lap'] * f
    s['overtake_factor'] = spec['overtake_factor'] * np.exp(unc['overtake_log_sd'] * z[3])
    return s


def _worker(spec, n, seed, batch, ticks=None):
    """ticks: optional queue that receives the size of each finished batch (progress reporting only)."""
    rng = np.random.default_rng(seed)
    agg = None
    done = 0
    unc = spec.get('param_unc')
    while done < n:
        b = min(batch, n - done)
        sb = _draw_circuit(spec, unc, rng) if unc else spec
        agg = _aggregate(run_batch(sb, b, rng), spec['D'], spec['n_laps'], agg)
        done += b
        if ticks is not None:
            ticks.put(b)
    return agg


# ---- process pools
# Each simulation worker is single-threaded numpy. Left alone, every worker's BLAS reserves buffers for one
# thread per core: measured ~1.0 GB committed per worker vs ~0.27 GB with one thread (and faster, since 12
# workers x 12 threads just contend). 12 default workers used to exhaust the page file on a 16 GB machine.
_THREAD_VARS = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
PER_WORKER_GB = 0.45          # measured peak ~0.27 GB at batch 10-20k, plus headroom


def _free_commit_gb():
    """Memory that new processes can still commit (RAM + page file), in GB; None if unknown."""
    try:
        if sys.platform == 'win32':
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [('len', ctypes.c_ulong), ('load', ctypes.c_ulong)] +                            [(k, ctypes.c_ulonglong) for k in ('tot_phys', 'avail_phys', 'tot_pf', 'avail_pf',
                                                                'tot_virt', 'avail_virt', 'avail_ext')]
            m = MS(); m.len = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.avail_pf / 2 ** 30
        return os.sysconf('SC_AVPHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 2 ** 30
    except (AttributeError, OSError, ValueError):
        return None


def default_workers(requested=None, jobs=None):
    """CPU-based worker count, capped by free memory (keeping ~1.5 GB for the main process and the OS)."""
    n = requested or max(1, (os.cpu_count() or 2) - 1)
    free = _free_commit_gb()
    if free is not None and not requested:
        n = min(n, max(1, int((free - 1.5) / PER_WORKER_GB)))
    return max(1, min(n, jobs or n))


@contextlib.contextmanager
def pool(workers):
    """ProcessPoolExecutor whose workers start with single-threaded BLAS/OpenMP. Workers are spawned lazily
    while the block runs, so the thread-count env vars stay set until it exits, then the parent's are restored."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    os.environ.update({k: '1' for k in _THREAD_VARS})
    try:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            yield ex
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class ProgressBar:
    """One self-updating terminal line: [#####-----]  42%  1,680,000 / 4,000,000 races  3.1 / 7.4 min."""

    def __init__(self, total, label='simulating', unit='races', width=30, stream=None):
        self.total, self.label, self.unit, self.width = total, label, unit, width
        self.stream = stream or sys.stdout
        self.t0 = time.time()
        enc = getattr(self.stream, 'encoding', None) or 'ascii'
        try:
            '█░'.encode(enc)
            self.full, self.empty = '█', '░'
        except (UnicodeEncodeError, LookupError):
            self.full, self.empty = '#', '-'
        self.update(0)

    def update(self, done, note=''):
        k = done / self.total if self.total else 1.0
        fill = int(round(k * self.width))
        el = time.time() - self.t0
        tail = f'{el / 60:.1f} min' if done >= self.total or not done else f'{el / 60:.1f} / ~{el / k / 60:.1f} min'
        line = (f'\r  {self.label} {self.full * fill}{self.empty * (self.width - fill)} {k:4.0%}  '
                f'{done:,} / {self.total:,} {self.unit}  {tail}  {note}')
        self.stream.write(line.ljust(getattr(self, '_last', 0)))   # pad over a longer previous line
        self._last = len(line)
        if done >= self.total:
            self.stream.write('\n')
        self.stream.flush()


CHUNK = 200_000   # sims per chunk: a chunk's identity (index -> SeedSequence child) never depends on workers


def chunk_plan(n_sims, seed, chunk=None):
    """[(size, SeedSequence)] - fixed-size chunks with independent child streams of one root seed.
    The same (n_sims, seed) gives the same chunks and streams on any machine and any worker count."""
    chunk = chunk or CHUNK                     # read at call time (tests shrink it)
    n = max(1, -(-n_sims // chunk))
    sizes = [n_sims // n + (1 if i < n_sims % n else 0) for i in range(n)]
    return list(zip(sizes, np.random.SeedSequence(seed).spawn(n)))


def simulate(spec, n_sims, workers=None, batch=20000, seed=2026, progress=None, ex=None):
    """Aggregate n_sims races. Reproducible: results depend only on (spec, n_sims, seed), not on workers."""
    spec = _spec_arrays(spec)
    plan = chunk_plan(n_sims, seed)
    workers = default_workers(workers, jobs=len(plan))
    agg = None
    if len(plan) == 1 or (workers == 1 and ex is None):
        for sz, ss in plan:
            r = _worker(spec, sz, ss, batch)
            agg = r if agg is None else _merge(agg, r)
            if progress:
                progress(agg['n'], n_sims)
        return agg
    with (contextlib.nullcontext(ex) if ex is not None else pool(workers)) as ex:
        futs = [ex.submit(_worker, spec, sz, ss, batch) for sz, ss in plan]
        for f in futs:                               # merge in chunk order: deterministic
            r = f.result()
            agg = r if agg is None else _merge(agg, r)
            if progress:
                progress(agg['n'], n_sims)
    return agg


def decode_seq(code):
    out = []
    while code:
        d = code % 4
        if d:
            out.append(DRY[d - 1][0])
        code //= 4
    return '-'.join(out)
