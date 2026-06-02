"""
FLAT OUT F1 v2.0 - POST-RACE LEARNING SYSTEM
==============================================
Scores predictions, updates driver + constructor profiles,
analyzes strategy effectiveness with proper DNF/DNS handling.

Usage:
  python f1_learn.py ingest <round>     Score + learn from race
  python f1_learn.py report             Season accuracy dashboard
  python f1_learn.py profiles           Current profiles
"""

import sys, json
import pandas as pd, numpy as np, warnings
from pathlib import Path
from scipy import stats as sp
import re
warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
PRED_DIR = ROOT / 'predictions'
SEASON_DIR = ROOT / 'season'
DATA_ROOT = ROOT / 'fastf1_data'
for d in (PRED_DIR, SEASON_DIR): d.mkdir(parents=True, exist_ok=True)

NAMES = {
    'VER':'Max Verstappen','NOR':'Lando Norris','LEC':'Charles Leclerc',
    'PIA':'Oscar Piastri','HAM':'Lewis Hamilton','RUS':'George Russell',
    'ALO':'Fernando Alonso','GAS':'Pierre Gasly','OCO':'Esteban Ocon',
    'ALB':'Alexander Albon','BOT':'Valtteri Bottas','HUL':'Nico Hulkenberg',
    'SAI':'Carlos Sainz','STR':'Lance Stroll','PER':'Sergio Perez',
    'LAW':'Liam Lawson','ANT':'Kimi Antonelli','COL':'Franco Colapinto',
    'BEA':'Oliver Bearman','BOR':'Gabriel Bortoleto','HAD':'Isack Hadjar',
    'LIN':'Arvid Lindblad',
}

GRIDS = {
    1: {'RUS':1,'ANT':2,'HAD':3,'LEC':4,'PIA':5,'NOR':6,'HAM':7,'LAW':8,
        'LIN':9,'BOR':10,'HUL':11,'BEA':12,'OCO':13,'GAS':14,'ALB':15,
        'COL':16,'ALO':17,'PER':18,'BOT':19,'VER':20,'SAI':21,'STR':22},
}

def to_sec(td):
    try:
        if pd.isna(td) or td=='': return np.nan
        p=str(td).split(' days ')
        if len(p)==2: h,m,s=p[1].split(':'); return float(h)*3600+float(m)*60+float(s)
    except: pass
    return np.nan


def find_race_dir(rnd):
    if not DATA_ROOT.exists(): return None
    for item in sorted(DATA_ROOT.iterdir()):
        if not item.is_dir(): continue
        m = re.match(r'(\d{4})_R(\d+)_(.+?)_(Race|R)$', item.name, re.I)
        if m and int(m.group(2)) == rnd: return item
    return None


def classify_finish(laps_df, results_df, total_race_laps=None):
    """Classify each driver's race outcome properly.
    Auto-detects race length from actual data instead of hardcoding."""
    
    # Auto-detect actual race length from the data
    if total_race_laps is None:
        total_race_laps = int(laps_df.LapNumber.max()) if len(laps_df) > 0 else 58
    
    classified = []
    for _, r in results_df.iterrows():
        drv = r.Abbreviation
        pos = int(r.Position) if pd.notna(r.Position) and r.Position > 0 else 0
        team = r.TeamName
        dl = laps_df[laps_df.Driver == drv]
        max_lap = int(dl.LapNumber.max()) if len(dl) > 0 else 0
        n_stints = dl.Stint.nunique() if len(dl) > 0 else 0

        if max_lap == 0:
            status = 'DNS'
        elif max_lap <= 3:
            status = 'DNF_LAP1'
        elif max_lap < total_race_laps * 0.40:
            status = 'DNF_EARLY'
        elif max_lap < total_race_laps * 0.75:
            status = 'DNF_MID'
        elif max_lap < total_race_laps - 5:
            # More than 5 laps behind leader = likely retired
            status = 'DNF_LATE'
        else:
            # Within 5 laps of leader = finished (classified or lapped)
            status = 'CLASSIFIED'

        classified.append(dict(
            Driver=drv, Position=pos, Team=team, MaxLap=max_lap,
            Stints=n_stints, Status=status,
            Finished=status == 'CLASSIFIED',
            HasPaceData=max_lap >= 10,
            HasStratData=max_lap >= total_race_laps * 0.40,
        ))
    
    print(f"  Race length: {total_race_laps} laps (auto-detected)")
    return pd.DataFrame(classified)


def score_predictions(rnd, race_class, grid):
    """Score predictions, only counting classified finishers + sensible DNFs."""
    pred_path = ROOT / 'v2_prediction_results.csv'
    if not pred_path.exists():
        print("  No predictions found."); return None

    pred = pd.read_csv(pred_path)
    rdir = PRED_DIR / f'round_{rnd:02d}'; rdir.mkdir(parents=True, exist_ok=True)
    pred.to_csv(rdir / 'prediction.csv', index=False)

    # Only score drivers who actually raced (exclude DNS)
    scoreable = race_class[race_class.Status != 'DNS']
    scores = []
    for _, r in scoreable.iterrows():
        drv = r.Driver; act = r.Position
        pr = pred[pred.Driver == drv]
        if len(pr) == 0 or act == 0: continue
        pred_pos = int(pr.iloc[0].Pos)
        scores.append(dict(Driver=drv, pred=pred_pos, actual=act,
                          error=pred_pos-act, abs_error=abs(pred_pos-act),
                          grid=grid.get(drv, 0), status=r.Status))

    sc = pd.DataFrame(scores)
    n = len(sc); mae = sc.abs_error.mean()
    exact = (sc.abs_error==0).sum(); w1 = (sc.abs_error<=1).sum()
    w3 = (sc.abs_error<=3).sum(); w5 = (sc.abs_error<=5).sum()
    rho, _ = sp.spearmanr(sc.pred, sc.actual)
    tau, _ = sp.kendalltau(sc.pred, sc.actual)
    pw = sc.loc[sc.pred.idxmin(),'Driver']; aw = sc.loc[sc.actual.idxmin(),'Driver']

    print(f"\n  {'='*60}")
    print(f"  SCORECARD - Round {rnd}")
    print(f"  {'='*60}")
    print(f"  MAE: {mae:.2f} | rho: {rho:.3f} | tau: {tau:.3f}")
    print(f"  Exact: {exact}/{n} | +/-1: {w1}/{n} | +/-3: {w3}/{n} | +/-5: {w5}/{n}")
    print(f"  Winner: {'CORRECT' if pw==aw else 'WRONG'} (pred:{pw} actual:{aw})")
    print(f"\n  {'DRV':4s} {'GRD':>3s} {'PRD':>3s} {'ACT':>3s} {'ERR':>4s}  {'STAT':10s}")
    print(f"  {'-'*45}")
    for _, r in sc.sort_values('actual').iterrows():
        sym = 'OK' if r.abs_error<=1 else ('~' if r.abs_error<=3 else f'MISS({int(r.abs_error)})')
        print(f"  {r.Driver:4s} P{int(r.grid):>2d} P{int(r.pred):>2d} P{int(r.actual):>2d} {int(r.error):+3d}   {r.status:10s} {sym}")

    metrics = dict(round=rnd, mae=mae, rho=rho, tau=tau,
                   exact=exact, within_1=w1, within_3=w3, within_5=w5,
                   n=n, winner_correct=pw==aw)
    sc.to_csv(rdir / 'scored.csv', index=False)
    pd.DataFrame([metrics]).to_csv(rdir / 'metrics.csv', index=False)
    return metrics


def update_driver_profiles(rnd, race_class, race_laps, grid):
    """Update profiles - DNS excluded, DNFs handled carefully.

    If REG_CHANGE_ROUND is set and this race is within 2 rounds of it,
    alpha is boosted so profiles adapt faster to the new regime.
    """
    prof_path = ROOT / 'driver_profiles.csv'
    profiles = pd.read_csv(prof_path) if prof_path.exists() else pd.DataFrame(columns=['driver'])

    # Regulation-change acceleration (set REG_CHANGE_ROUND when rules change mid-season)
    # e.g. REG_CHANGE_ROUND = 5 for Miami 2026 if rule changes ship there
    REG_CHANGE_ROUND = None  # set to int when known; None disables the boost
    reg_boost_active = (REG_CHANGE_ROUND is not None
                        and REG_CHANGE_ROUND <= rnd < REG_CHANGE_ROUND + 3)
    if reg_boost_active:
        print(f"  [REG CHANGE ACTIVE] alpha boosted for rounds {REG_CHANGE_ROUND}-{REG_CHANGE_ROUND+2}")

    for _, r in race_class.iterrows():
        if r.Status == 'DNS': continue  # skip DNS entirely
        drv = r.Driver; pos = r.Position; grd = grid.get(drv, 0)
        dnf = not r.Finished

        if drv not in profiles.driver.values:
            profiles = pd.concat([profiles, pd.DataFrame([{'driver':drv,'n_races':0}])], ignore_index=True)
        idx = profiles[profiles.driver==drv].index[0]
        n = profiles.loc[idx,'n_races'] if 'n_races' in profiles.columns and pd.notna(profiles.loc[idx,'n_races']) else 0
        alpha = 1/(n+1) if n<10 else 0.1
        # Reg-change boost: force faster updating for the first 3 races post-change
        if reg_boost_active:
            alpha = max(alpha, 0.25)

        # Position: for DNFs, use a penalty position instead of their classified pos
        # Early DNFs get worse penalty than late DNFs
        if dnf:
            if r.Status == 'DNF_LAP1': eff_pos = 20
            elif r.Status == 'DNF_EARLY': eff_pos = 19
            elif r.Status == 'DNF_MID': eff_pos = max(pos, 18)
            else: eff_pos = pos  # late DNF, classified position is roughly right
        else:
            eff_pos = pos

        for col, val in [('avg_finish', eff_pos), ('dnf_rate', 100 if dnf else 0)]:
            if col not in profiles.columns: profiles[col] = np.nan
            old = profiles.loc[idx, col]
            profiles.loc[idx, col] = old*(1-alpha)+val*alpha if pd.notna(old) else val

        # Points and pos gained only for finishers
        if r.Finished:
            pts = 0  # TODO: from results
            gained = grd - pos if grd > 0 and pos > 0 else 0
            for col, val in [('avg_pos_gained', gained)]:
                if col not in profiles.columns: profiles[col] = np.nan
                old = profiles.loc[idx, col]
                profiles.loc[idx, col] = old*(1-alpha)+val*alpha if pd.notna(old) else val

        # Recent form (higher alpha = more weight on recent)
        if 'recent_avg_finish' not in profiles.columns: profiles['recent_avg_finish'] = np.nan
        old = profiles.loc[idx, 'recent_avg_finish']
        ra = min(0.3, 1/max(n-5, 1))
        profiles.loc[idx, 'recent_avg_finish'] = old*(1-ra)+eff_pos*ra if pd.notna(old) else eff_pos

        if 'form_trend' not in profiles.columns: profiles['form_trend'] = 0
        la = profiles.loc[idx, 'avg_finish']; rc = profiles.loc[idx, 'recent_avg_finish']
        if pd.notna(la) and pd.notna(rc): profiles.loc[idx, 'form_trend'] = la - rc

        profiles.loc[idx, 'n_races'] = n + 1

    # Recompute score
    profiles['driver_score'] = (
        (22 - profiles['avg_finish'].clip(1,22)) / 21 * 30 +
        profiles.get('top3_pct', pd.Series(0,index=profiles.index)).fillna(0)/100*20 +
        profiles.get('tm_race_beat_pct', pd.Series(50,index=profiles.index)).fillna(50)/100*15 +
        profiles.get('points_per_race', pd.Series(0,index=profiles.index)).fillna(0)/20*15 +
        profiles.get('avg_pos_gained', pd.Series(0,index=profiles.index)).fillna(0).clip(-5,5)/5*10 +
        (100 - profiles.get('dnf_rate', pd.Series(10,index=profiles.index)).fillna(10))/100*5 +
        profiles.get('form_trend', pd.Series(0,index=profiles.index)).fillna(0).clip(-5,5)/5*5
    ).clip(0, 100)

    profiles.to_csv(prof_path, index=False)
    print(f"\n  Driver profiles updated ({len(race_class[race_class.Status!='DNS'])} drivers)")
    return profiles


def build_constructor_profiles(rnd, race_class, race_laps, grid):
    """Constructor profiles based on car pace, deg, reliability, strategy."""
    cpath = ROOT / 'constructor_profiles.csv'
    cprof = pd.read_csv(cpath) if cpath.exists() else pd.DataFrame(columns=['team'])

    clean = race_laps.copy()
    clean['lt'] = clean['LapTime'].apply(to_sec)
    clean = clean[(clean['lt']>70)&(clean['lt']<200)&clean.PitInTime.isna()&clean.PitOutTime.isna()]

    # Pace rankings (only drivers with enough data)
    paces = {}
    for drv in clean.Driver.unique():
        dl = clean[clean.Driver==drv]
        if len(dl) >= 10:  # need meaningful sample
            paces[drv] = dl['lt'].mean()
    pace_rank = {d:i+1 for i,(d,_) in enumerate(sorted(paces.items(), key=lambda x:x[1]))}

    teams = {}
    for _, r in race_class.iterrows():
        if r.Status == 'DNS': continue  # DNS doesn't count
        t = r.Team; drv = r.Driver
        if t not in teams: teams[t] = dict(drivers=[])

        dl = clean[clean.Driver==drv]
        drv_pace = dl['lt'].mean() if len(dl)>=10 else np.nan

        # Tyre degradation
        deg = np.nan
        if len(dl) >= 15:
            # Use longest clean stint
            best_stint = None; best_n = 0
            for sid, sg in dl.groupby(race_laps[race_laps.Driver==drv].Stint):
                if len(sg) > best_n: best_stint = sg; best_n = len(sg)
            if best_stint is not None and len(best_stint) >= 8:
                deg = sp.linregress(best_stint.TyreLife.values.astype(float),
                                    best_stint['lt'].values).slope

        # Strategy effectiveness (only if they had a real strategy)
        strat_gain = 0
        if r.HasStratData and drv in pace_rank:
            strat_gain = pace_rank[drv] - r.Position

        teams[t]['drivers'].append(dict(
            driver=drv, pos=r.Position, grid=grid.get(drv,0),
            pace=drv_pace, deg=deg, strat_gain=strat_gain,
            finished=r.Finished, has_pace=r.HasPaceData,
            status=r.Status, gained=grid.get(drv,0)-r.Position if r.Position>0 else 0
        ))

    # Same reg-change logic as driver profiles — read constant from driver fn scope
    REG_CHANGE_ROUND = None  # keep in sync with update_driver_profiles
    reg_boost_active = (REG_CHANGE_ROUND is not None
                        and REG_CHANGE_ROUND <= rnd < REG_CHANGE_ROUND + 3)

    for team, data in teams.items():
        if team not in cprof.team.values:
            cprof = pd.concat([cprof, pd.DataFrame([{'team':team,'n_races':0}])], ignore_index=True)
        idx = cprof[cprof.team==team].index[0]
        n = cprof.loc[idx,'n_races'] if 'n_races' in cprof.columns and pd.notna(cprof.loc[idx,'n_races']) else 0
        alpha = 1/(n+1) if n<10 else 0.1
        if reg_boost_active: alpha = max(alpha, 0.25)

        # Only use drivers with pace data for car performance metrics
        pace_drivers = [d for d in data['drivers'] if d['has_pace'] and pd.notna(d['pace'])]
        strat_drivers = [d for d in data['drivers'] if d['finished']]
        all_drivers = data['drivers']

        avg_pace = np.mean([d['pace'] for d in pace_drivers]) if pace_drivers else np.nan
        avg_deg = np.nanmean([d['deg'] for d in pace_drivers if pd.notna(d['deg'])]) if pace_drivers else np.nan
        avg_fin = np.mean([d['pos'] for d in all_drivers if d['pos']>0])
        avg_strat = np.mean([d['strat_gain'] for d in strat_drivers]) if strat_drivers else 0

        # Reliability: DNS=excluded, DNF_LAP1=incident, others count
        real_starters = [d for d in all_drivers]  # DNS already excluded
        finished = sum(1 for d in real_starters if d['finished'])
        rel = finished / len(real_starters) * 100 if real_starters else 100

        for col, val in [('avg_finish',avg_fin), ('avg_race_pace',avg_pace),
                         ('avg_deg',avg_deg), ('strategy_score',avg_strat),
                         ('reliability',rel)]:
            if col not in cprof.columns: cprof[col] = np.nan
            old = cprof.loc[idx, col]
            cprof.loc[idx, col] = old*(1-alpha)+val*alpha if pd.notna(old) and pd.notna(val) else (val if pd.notna(val) else old)

        cprof.loc[idx, 'n_races'] = n + 1

    # Team score: PACE is king (50%), then consistency/reliability
    # Pace score: normalize avg_race_pace (lower = better)
    pace_vals = cprof['avg_race_pace'].dropna()
    if len(pace_vals) >= 2:
        p_min, p_max = pace_vals.min(), pace_vals.max()
        p_rng = p_max - p_min or 1
        cprof['pace_score'] = ((p_max - cprof['avg_race_pace']) / p_rng * 100).fillna(50)
    else:
        cprof['pace_score'] = 50

    cprof['team_score'] = (
        cprof['pace_score'] * 0.50 +
        (22 - cprof['avg_finish'].clip(1,22)) / 21 * 100 * 0.20 +
        cprof.get('strategy_score', pd.Series(0,index=cprof.index)).fillna(0).clip(-5,5) / 5 * 50 * 0.10 +
        cprof.get('reliability', pd.Series(100,index=cprof.index)).fillna(100) / 100 * 100 * 0.10 +
        cprof.get('avg_deg', pd.Series(0,index=cprof.index)).fillna(0).clip(-0.5,0.5) * -100 * 0.10
    ).clip(0, 100)

    cprof.to_csv(cpath, index=False)

    print(f"\n  {'='*70}")
    print(f"  CONSTRUCTOR PROFILES (after R{rnd})")
    print(f"  {'='*70}")
    print(f"  {'TEAM':20s} {'SCORE':>5s} {'PACE':>8s} {'AVG':>5s} {'DEG':>8s} {'STRAT':>5s} {'REL':>4s}")
    print(f"  {'-'*60}")
    for _, r in cprof.sort_values('team_score', ascending=False).iterrows():
        pace_s = f"{r.avg_race_pace:.2f}s" if pd.notna(r.avg_race_pace) else 'N/A'
        deg_s = f"{r.avg_deg:+.3f}" if pd.notna(r.get('avg_deg')) else 'N/A'
        ss = f"{r.strategy_score:+.1f}" if pd.notna(r.get('strategy_score')) else 'N/A'
        print(f"  {r.team:20s} {r.team_score:5.1f} {pace_s:>8s} P{r.avg_finish:4.1f} {deg_s:>8s} {ss:>5s} {r.reliability:4.0f}%")

    return cprof


def analyze_strategy(race_laps, race_class, grid):
    """Strategy analysis with proper DNF/DNS handling."""
    clean = race_laps.copy()
    clean['lt'] = clean['LapTime'].apply(to_sec)
    valid = clean[(clean['lt']>70)&(clean['lt']<200)&clean.PitInTime.isna()&clean.PitOutTime.isna()]

    # Pace rankings (only drivers with 10+ clean laps)
    paces = {}
    for drv in valid.Driver.unique():
        dl = valid[valid.Driver==drv]
        if len(dl) >= 10: paces[drv] = dl['lt'].mean()
    pace_rank = {d:i+1 for i,(d,_) in enumerate(sorted(paces.items(), key=lambda x:x[1]))}

    print(f"\n  {'='*70}")
    print(f"  STRATEGY ANALYSIS")
    print(f"  {'='*70}")

    teams = {}
    for _, r in race_class.iterrows():
        t = r.Team
        if t not in teams: teams[t] = []
        drv = r.Driver
        dl = race_laps[race_laps.Driver==drv].sort_values('LapNumber')
        dl_clean = valid[valid.Driver==drv]

        pit_laps = dl[dl.PitInTime.notna()].LapNumber.tolist()
        stints = []
        for sid, sg in dl_clean.groupby(dl[dl.Driver==drv].Stint):
            if len(sg)<2: continue
            comp = sg.Compound.iloc[0]; n = len(sg); pace = sg['lt'].mean()
            deg = sp.linregress(sg.TyreLife.values.astype(float), sg['lt'].values).slope if n>=5 else np.nan
            stints.append(dict(comp=comp, n=n, pace=pace, deg=deg))

        pp = pace_rank.get(drv, 22)
        strat_gain = pp - r.Position if r.HasStratData and drv in pace_rank and r.Position>0 else 0

        # Verdict
        if r.Status == 'DNS':
            verdict = 'DNS'
        elif r.Status in ('DNF_LAP1','DNF_EARLY'):
            verdict = f'RETIRED L{r.MaxLap}'
        elif not r.HasStratData:
            verdict = f'RETIRED L{r.MaxLap}'
        elif strat_gain >= 3:
            verdict = 'STRAT_WIN'
        elif strat_gain <= -3:
            verdict = 'STRAT_LOSS'
        elif (grid.get(drv,0) - r.Position) >= 5:
            verdict = 'DRIVER_EXCELLENCE'
        elif (grid.get(drv,0) - r.Position) <= -5:
            verdict = 'DROPPED'
        else:
            verdict = 'AS_EXPECTED'

        teams[t].append(dict(
            driver=drv, pos=r.Position, grid=grid.get(drv,0),
            pace_rank=pp, strat_gain=strat_gain, n_stops=len(pit_laps),
            pit_laps=pit_laps, stints=stints, verdict=verdict,
            status=r.Status, max_lap=r.MaxLap
        ))

    for team in sorted(teams.keys(), key=lambda t: np.mean([d['pos'] for d in teams[t] if d['pos']>0])):
        drivers = teams[team]
        real = [d for d in drivers if d['status'] != 'DNS' and d['status'] not in ('DNF_LAP1','DNF_EARLY')]
        avg_s = np.mean([d['strat_gain'] for d in real]) if real else 0
        print(f"\n  {team} (strat: {avg_s:+.1f})")
        for d in sorted(drivers, key=lambda x: x['pos'] if x['pos']>0 else 99):
            strat_str = ' -> '.join(f"{s['comp'][0]}({s['n']}L)" for s in d['stints'])
            pit_str = ','.join(str(int(p)) for p in d['pit_laps']) if d['pit_laps'] else '-'

            if d['status'] == 'DNS':
                print(f"    {d['driver']:4s}  ** DNS **")
            elif d['max_lap'] < 20:
                print(f"    {d['driver']:4s} P{d['grid']:>2d}->P{d['pos']:>2d}  ** {d['status']} (L{d['max_lap']}) **")
            else:
                print(f"    {d['driver']:4s} P{d['grid']:>2d}->P{d['pos']:>2d}  "
                      f"pace=P{d['pace_rank']:>2d}  strat={d['strat_gain']:+2d}  "
                      f"{d['n_stops']}-stop [{strat_str}]  pits:L{pit_str}  -> {d['verdict']}")


def season_report():
    mfiles = sorted(PRED_DIR.glob('round_*/metrics.csv'))
    if not mfiles: print("  No data yet."); return
    am = pd.concat([pd.read_csv(f) for f in mfiles], ignore_index=True).sort_values('round')
    print(f"\n  SEASON DASHBOARD ({len(am)} races)")
    print(f"  {'RND':>3s} {'MAE':>5s} {'rho':>5s} {'tau':>5s} {'WIN':>3s} {'+/-1':>4s} {'+/-3':>4s}")
    print(f"  {'-'*35}")
    for _, r in am.iterrows():
        w='Y' if r.get('winner_correct',False) else '-'
        print(f"  R{int(r['round']):>2d} {r.mae:5.2f} {r.rho:5.3f} {r.tau:5.3f}  {w} {int(r.within_1):>4d} {int(r.within_3):>4d}")
    print(f"  AVG {am.mae.mean():5.2f} {am.rho.mean():5.3f} {am.tau.mean():5.3f}")


def show_profiles():
    dp = ROOT / 'driver_profiles.csv'
    if dp.exists():
        p = pd.read_csv(dp)
        print(f"\n  DRIVER PROFILES:")
        print(f"  {'DRV':4s} {'SCORE':>5s} {'AVG':>5s} {'RECENT':>6s} {'FORM':>5s}")
        print(f"  {'-'*30}")
        for _, r in p.sort_values('driver_score', ascending=False).iterrows():
            rc = r.get('recent_avg_finish', r.get('avg_finish',0))
            ft = r.get('form_trend', 0)
            print(f"  {r.driver:4s} {r.driver_score:5.1f} P{r.avg_finish:4.1f} P{rc:5.1f} {ft:+5.1f}")
    cp = ROOT / 'constructor_profiles.csv'
    if cp.exists():
        c = pd.read_csv(cp)
        print(f"\n  CONSTRUCTOR PROFILES:")
        for _, r in c.sort_values('team_score', ascending=False).iterrows():
            pace = f"{r.avg_race_pace:.2f}s" if pd.notna(r.get('avg_race_pace')) else 'N/A'
            print(f"  {r.team:20s} score={r.team_score:5.1f} pace={pace} avg=P{r.avg_finish:.1f}")


def main():
    if len(sys.argv) < 2:
        print("  f1_learn.py ingest <round> | report | profiles"); return

    cmd = sys.argv[1].lower()
    if cmd == 'ingest':
        if len(sys.argv)<3: print("  Usage: ingest <round>"); return
        rnd = int(sys.argv[2])
        grid = GRIDS.get(rnd, {})

        race_dir = find_race_dir(rnd)
        if race_dir is None:
            print(f"  No race data. Place in fastf1_data/2026_R{rnd}_*_Race/"); return

        laps = pd.read_csv(race_dir / 'laps.csv')
        results = pd.read_csv(race_dir / 'results.csv')
        results = results.sort_values('Position')
        print(f"  Loaded: {len(laps)} laps from {race_dir.name}")

        # Classify finishes
        race_class = classify_finish(laps, results)
        print(f"\n  Race classification:")
        for _, r in race_class.sort_values('Position').iterrows():
            print(f"    P{r.Position:>2d} {r.Driver:4s} L{r.MaxLap:>2d} {r.Status:12s} "
                  f"{'pace:Y' if r.HasPaceData else 'pace:N'} {'strat:Y' if r.HasStratData else 'strat:N'}")

        score_predictions(rnd, race_class, grid)
        update_driver_profiles(rnd, race_class, laps, grid)
        build_constructor_profiles(rnd, race_class, laps, grid)
        analyze_strategy(laps, race_class, grid)

        # Extract race metrics (position traces, gap evolution, racecraft)
        try:
            from f1_race_metrics import extract_race_metrics, save_season_data
            circuit_name = race_dir.name.split('_')[2] if '_' in race_dir.name else 'Unknown'
            # Clean up circuit name
            import re as _re
            _cm = _re.match(r'\d+_R\d+_(.+?)_(Race|R)$', race_dir.name, _re.I)
            if _cm: circuit_name = _cm.group(1).replace('_',' ').replace('Grand Prix','').strip()
            
            race_metrics = extract_race_metrics(laps, rnd, circuit_name)
            season_data = save_season_data(race_metrics, ROOT)
            
            print(f"\n  Race metrics extracted: {len(race_metrics)} drivers")
            print(f"  Season data: {len(season_data)} total entries across {season_data.Round.nunique()} races")
            
            # Show key metrics
            print(f"\n  {'DRV':4s} {'P_START':>7s} {'P_END':>5s} {'L1':>3s} {'GAP/LAP':>8s} {'DEG':>7s} {'OT':>3s}")
            print(f"  {'-'*45}")
            for _, r in race_metrics.sort_values('end_pos').head(10).iterrows():
                gr = f"{r.gap_rate:+.3f}s" if pd.notna(r.get('gap_rate')) else '   N/A'
                dg = f"{r.best_stint_deg:+.3f}" if pd.notna(r.get('best_stint_deg')) else '   N/A'
                l1 = int(r.get('lap1_gain', 0))
                ot = int(r.get('overtakes_made', 0))
                print(f"  {r.Driver:4s}    P{int(r.start_pos):>2d}   P{int(r.end_pos):>2d}  {l1:+2d}  {gr:>8s} {dg:>7s}  {ot:>3d}")
        except ImportError:
            print("\n  f1_race_metrics.py not found — skipping race metrics")
        except Exception as e:
            print(f"\n  Race metrics error: {e}")
        print(f"\n  Round {rnd} complete!")
        
        # Auto-tune hyperparameters
        try:
            from f1_autotune import autotune
            circuit = 'Unknown'
            import re as _re
            for item in DATA_ROOT.iterdir():
                _m = _re.match(rf'\d+_R{rnd}_(.+?)_(Race|R)$', item.name, _re.I)
                if _m: circuit = _m.group(1).replace('_',' ').replace('Grand Prix','').strip(); break
            autotune(rnd, circuit)
        except ImportError:
            print("  f1_autotune.py not found - skipping self-tune")
        except Exception as e:
            print(f"  Autotune error: {e}")

    elif cmd == 'report': season_report()
    elif cmd == 'profiles': show_profiles()
    else: print(f"  Unknown: {cmd}")

if __name__ == '__main__':
    main()
