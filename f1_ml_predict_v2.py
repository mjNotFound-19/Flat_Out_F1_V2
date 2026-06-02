r"""
FLAT OUT F1 v2.0 - IMPROVED ADAPTIVE RACE PREDICTION
======================================================
Improvements from R1 analysis:
  - Grid position properly weighted (P1-3 clean air, P16+ traffic)
  - Overtaking model uses 2026 DRS data (easier passing than old eras)
  - Rookies trust current-weekend data more when no history exists
  - Historical profiles capped at 20% max (2026 new regs)
  - Constructor pace profiles integrated as features
  - DNF probability from constructor reliability data
"""

import pandas as pd, numpy as np, warnings, re
from pathlib import Path
from scipy import stats as sp
from collections import defaultdict
warnings.filterwarnings('ignore')
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / 'fastf1_data'
if not DATA_ROOT.exists(): DATA_ROOT = Path.cwd() / 'fastf1_data'

N_SIM = 10000; RACE_LAPS = 58
FUEL_PER_KG = 0.030; FUEL_PER_LAP = 1.7; TEMP_COEFF = 0.06; REF_TEMP = 30.0

NAMES = {
    'VER':'Max Verstappen','NOR':'Lando Norris','LEC':'Charles Leclerc',
    'PIA':'Oscar Piastri','HAM':'Lewis Hamilton','RUS':'George Russell',
    'ALO':'Fernando Alonso','GAS':'Pierre Gasly','OCO':'Esteban Ocon',
    'ALB':'Alexander Albon','BOT':'Valtteri Bottas','HUL':'Nico Hulkenberg',
    'SAI':'Carlos Sainz','STR':'Lance Stroll','PER':'Sergio Perez',
    'LAW':'Liam Lawson','ANT':'Kimi Antonelli','COL':'Franco Colapinto',
    'BEA':'Oliver Bearman','BOR':'Gabriel Bortoleto','HAD':'Isack Hadjar',
    'LIN':'Arvid Lindblad','CRA':'Jack Crawford','DOO':'Jack Doohan',
    'TSU':'Yuki Tsunoda','DEV':'Nyck de Vries','RIC':'Daniel Ricciardo',
    'ZHO':'Guanyu Zhou','MAG':'Kevin Magnussen','SER':'Jamie Chadwick',
}

GRID_OVERRIDES = {
    1: {'RUS':1,'ANT':2,'HAD':3,'LEC':4,'PIA':5,'NOR':6,'HAM':7,'LAW':8,
        'LIN':9,'BOR':10,'HUL':11,'BEA':12,'OCO':13,'GAS':14,'ALB':15,
        'COL':16,'ALO':17,'PER':18,'BOT':19,'VER':20,'SAI':21,'STR':22},
}

# Load self-tuned hyperparameters (updated by f1_autotune.py after each race)
HP_PATH = SCRIPT_DIR / 'f1_hyperparams.json'
HP = None
if HP_PATH.exists():
    import json
    with open(HP_PATH) as _f: HP = json.load(_f)
    print(f"  Hyperparams: v{HP.get('_version',0)} (R{HP.get('_last_round',0)})")
else:
    print(f"  Hyperparams: using defaults (no f1_hyperparams.json)")

def to_sec(td):
    try:
        if pd.isna(td) or td=='': return np.nan
        p=str(td).split(' days ')
        if len(p)==2: h,m,s=p[1].split(':'); return float(h)*3600+float(m)*60+float(s)
    except: pass
    return np.nan

def banner(n, t): print(f"\n{'='*78}\n  STAGE {n}: {t}\n{'='*78}")

# =========================================================================
#  STAGE 1: AUTO-DISCOVERY
# =========================================================================
banner(1, "AUTO-DISCOVERY")

def discover_data(root):
    sources = []
    if not root.exists(): return sources, 0, 'Unknown', 0
    for item in sorted(root.iterdir()):
        if not item.is_dir(): continue
        name = item.name
        if 'testing' in name.lower():
            tnum = 2 if '2' in name else 1
            for day in (1,2,3):
                lp=item/f'day{day}_laps.csv'; wp=item/f'day{day}_weather.csv'
                if lp.exists():
                    sources.append(dict(tier=3,type='test',laps_path=lp,
                        weather_path=wp if wp.exists() else None,
                        circuit='Bahrain',round=0,test_num=tnum,day=day,
                        session=f'T{tnum}D{day}',label=f'T{tnum}D{day}'))
        m = re.match(r'(\d{4})_R(\d+)_(.+?)_(FP\d+|SQ|S|Sprint|Q|Qualifying|R|Race)', name, re.I)
        if m:
            yr,rnd,gp,sess = m.groups(); rnd=int(rnd)
            circuit=gp.replace('_',' ').replace('Grand Prix','').strip()
            su=sess.upper()
            if su in ('Q','QUALIFYING'): stype='quali'
            elif su=='SQ': stype='sprint_quali'
            elif su in ('S','SPRINT'): stype='sprint_race'
            elif su in ('R','RACE'): stype='race'  # race data (for f1_learn.py, not used in prediction)
            else: stype='fp'
            lp=item/'laps.csv'; wp=item/'weather.csv'; rp=item/'results.csv'
            if lp.exists() or (stype in ('quali','sprint_quali') and rp.exists()):
                sources.append(dict(tier=0,type=stype,
                    laps_path=lp if lp.exists() else None,
                    weather_path=wp if wp.exists() else None,
                    results_path=rp if rp.exists() else None,  # Sprint/SQ/Q all have results
                    circuit=circuit,round=rnd,test_num=0,day=0,
                    session=sess.upper(),label=f'R{rnd} {circuit} {sess.upper()}'))
    rounds=sorted(set(s['round'] for s in sources if s['round']>0))
    cur_rnd=max(rounds) if rounds else 0
    cur_circ=next((s['circuit'] for s in sources if s['round']==cur_rnd),'Unknown')
    past=len([r for r in rounds if r<cur_rnd])
    for s in sources:
        if s['type'] in('fp','quali','sprint_quali','sprint_race') and s['round']==cur_rnd: s['tier']=1
        elif s['type'] in('fp','quali','sprint_quali','sprint_race') and s['round']<cur_rnd: s['tier']=2
        elif s['type']=='race': s['tier']=99  # race results -> profiles, not raw laps
    return sources, cur_rnd, cur_circ, past

def compute_weights(sources, cur_rnd, past):
    spf=min(1.0,(cur_rnd-1)/12)
    T={1:3.0, 2:0.5+2.0*spf, 3:max(0.1,0.7-0.6*spf)}
    for s in sources:
        base=T.get(s['tier'],0.3)
        if s['tier']==1:
            mults={'Q':1.3,'FP3':1.2,'FP2':1.15,'FP1':0.7,
                       'SQ':1.1,'S':1.4,'SPRINT':1.4}  # Sprint race = actual race data
            s['weight']=base*mults.get(s['session'],0.5)
        elif s['tier']==2:
            recency=max(0.3,1.0-(cur_rnd-s['round'])*0.08)
            s['weight']=base*recency*{'S':1.3,'SPRINT':1.3,'FP2':1.0,'FP3':0.9,'Q':0.8,'FP1':0.6,'SQ':0.7}.get(s['session'],0.5)
        elif s['tier']==3:
            s['weight']=base*(1.2 if s.get('test_num')==2 else 0.8)
    return sources

sources, cur_rnd, cur_circ, past_races = discover_data(DATA_ROOT)
sources = compute_weights(sources, cur_rnd, past_races)
spf = min(1.0,(cur_rnd-1)/12)

print(f"  R{cur_rnd} {cur_circ} | Past: {past_races} | Season: {spf:.0%}")
for tier,lbl in [(1,'Current Weekend'),(2,'Past Races'),(3,'Testing')]:
    ts=[s for s in sources if s['tier']==tier]
    if ts:
        print(f"\n  TIER {tier}: {lbl}")
        for s in sorted(ts, key=lambda x:-x['weight']):
            print(f"    {s['label']:<30s} wt={s['weight']:.2f}")

# Load laps
frames=[]
for s in sources:
    # Skip ALL race laps — race data is noisy (traffic, SC, incidents, dead tyres)
    # Past race RESULTS feed through driver_profiles.csv and constructor_profiles.csv
    # which are updated by f1_learn.py after each race. That's the clean learning path.
    if s['type']=='race': continue
    if s.get('laps_path') is None or not s['laps_path'].exists(): continue
    df=pd.read_csv(s['laps_path'])
    df['source']=s['label']; df['source_weight']=s['weight']
    df['source_circuit']=s['circuit']; df['source_round']=s['round']
    df['source_session']=s['session']; df['source_tier']=s['tier']
    df['Test']=s.get('test_num',0); df['Day']=s.get('day',1)
    if s.get('weather_path') and s['weather_path'].exists():
        w=pd.read_csv(s['weather_path'])
        df['TrackTemp']=w.TrackTemp.median(); df['AirTemp']=w.AirTemp.median()
    frames.append(df)

laps=pd.concat(frames,ignore_index=True)
for c in ('LapTime','Sector1Time','Sector2Time','Sector3Time','PitInTime','PitOutTime','LapStartTime'):
    if c in laps.columns: laps[c+'_s']=laps[c].apply(to_sec)
laps['is_clean']=laps.PitInTime_s.isna()&laps.PitOutTime_s.isna()
med_t=laps.LapTime_s.dropna().median()
V=laps[laps.LapTime_s.notna()&(laps.LapTime_s>med_t*0.80)&(laps.LapTime_s<med_t*1.70)&laps.is_clean].copy()
V['time_bin']=(V.LapStartTime_s//900).astype('Int64')
# Detect weekend format
sprint_sessions = [s for s in sources if s['type'] in ('sprint_quali','sprint_race') and s['round']==cur_rnd]
is_sprint_weekend = len(sprint_sessions) > 0
weekend_type = 'SPRINT' if is_sprint_weekend else 'NORMAL'
print(f"\n  {len(V):,} valid laps, {V.Driver.nunique()} drivers ({weekend_type} weekend)")

HAS_FP=any(s['tier']==1 and s['type'] in ('fp','sprint_race') for s in sources)
HAS_PAST=any(s['tier']==2 for s in sources)
HAS_TEST=any(s['tier']==3 for s in sources)

# =========================================================================
#  STAGE 2: CORRECTIONS
# =========================================================================
banner(2,"CORRECTIONS")
def fuel_corr(g):
    g=g.copy();n=len(g)
    if n<3:g['fc']=0;return g
    tl=g.TyreLife.values;mn,mx=tl.min(),tl.max();rng=mx-mn or 1
    g['fc']=(0.5-(tl-mn)/rng)*n*FUEL_PER_LAP*FUEL_PER_KG;return g
V=pd.concat([fuel_corr(g) for _,g in V.groupby(['Driver','source','Day','Stint'])],ignore_index=True)
V['lt_corr']=V.LapTime_s+V.fc-(V.TrackTemp.fillna(REF_TEMP)-REF_TEMP)*TEMP_COEFF
print(f"  Done")

# =========================================================================
#  STAGE 3: LAP CLASSIFICATION
# =========================================================================
banner(3,"LAP CLASSIFICATION")
V['lap_type']='OTHER'
V=V.sort_values(['Driver','source','LapStartTime_s']).reset_index(drop=True)
_lc=HP['lap_classification'] if HP else {}
_qp=_lc.get('quali_push_pct',1.5); _rp=_lc.get('race_pace_pct',4.0)
_cp=_lc.get('cooldown_pct',8.0); _sp=_lc.get('slow_pct',10.0)

for (drv,src),grp in V.groupby(['Driver','source']):
    if len(grp)<3:continue
    best=grp.lt_corr.min(); idxs=grp.index.tolist(); lts=grp.lt_corr.values
    src_session=grp.source_session.iloc[0]

    # Sprint: all laps are race pace (short race, mostly clean)
    if src_session in ('S','SPRINT'):
        for i,idx in enumerate(idxs):
            dpct=(lts[i]-best)/best*100
            if dpct>15: V.loc[idx,'lap_type']='SLOW'
            elif dpct>8: V.loc[idx,'lap_type']='TRANSITION'
            else: V.loc[idx,'lap_type']='RACE_PACE'
        continue

    # Full race: only use BEST STINT per driver (clean air, representative pace)
    # This avoids counting traffic laps, safety car laps, tyre-dead laps
    if src_session in ('R','RACE'):
        # Find the stint with the best (lowest) median pace per driver
        stint_col = grp.Stint if 'Stint' in grp.columns else pd.Series(1, index=grp.index)
        best_stint_med = float('inf')
        best_stint_idxs = []
        for stint_id, stint_grp in grp.groupby(stint_col):
            if len(stint_grp) < 5: continue  # skip short stints
            stint_lts = stint_grp.lt_corr.values
            # Remove first 2 laps of stint (outlap, cold tyres)
            clean = stint_lts[2:] if len(stint_lts) > 4 else stint_lts
            med = np.median(clean)
            if med < best_stint_med:
                best_stint_med = med
                best_stint_idxs = stint_grp.index[2:].tolist() if len(stint_grp) > 4 else stint_grp.index.tolist()
        # Only classify best stint as RACE_PACE, rest as TRANSITION
        for i, idx in enumerate(idxs):
            if idx in best_stint_idxs:
                dpct = (lts[i] - best) / best * 100
                if dpct > 10: V.loc[idx, 'lap_type'] = 'SLOW'
                else: V.loc[idx, 'lap_type'] = 'RACE_PACE'
            else:
                V.loc[idx, 'lap_type'] = 'TRANSITION'  # traffic/SC/dead tyres
        continue

    # Sprint quali (SQ): treat like regular qualifying
    if src_session=='SQ':
        for i,idx in enumerate(idxs):
            dpct=(lts[i]-best)/best*100
            ns=(lts[i+1]-best)/best*100>8 if i<len(lts)-1 else False
            pf=(lts[i-1]-best)/best*100<2.5 if i>0 else False
            if dpct>_sp: V.loc[idx,'lap_type']='COOLDOWN' if pf else 'SLOW'
            elif dpct<_qp and ns: V.loc[idx,'lap_type']='QUALI_PUSH'
            elif dpct<_qp: V.loc[idx,'lap_type']='HOT_LAP'
            elif dpct<_rp: V.loc[idx,'lap_type']='RACE_PACE'
            else: V.loc[idx,'lap_type']='TRANSITION'
        continue

    # Normal FP sessions
    for i,idx in enumerate(idxs):
        dpct=(lts[i]-best)/best*100
        ns=(lts[i+1]-best)/best*100>8 if i<len(lts)-1 else False
        pf=(lts[i-1]-best)/best*100<2.5 if i>0 else False
        if dpct>_sp: V.loc[idx,'lap_type']='COOLDOWN' if pf else 'SLOW'
        elif dpct<_qp and ns: V.loc[idx,'lap_type']='QUALI_PUSH'
        elif dpct<_qp: V.loc[idx,'lap_type']='HOT_LAP'
        elif dpct<_rp: V.loc[idx,'lap_type']='RACE_PACE'
        else: V.loc[idx,'lap_type']='TRANSITION'

for lt in ['QUALI_PUSH','HOT_LAP','RACE_PACE','COOLDOWN','SLOW','TRANSITION','OTHER']:
    n=len(V[V.lap_type==lt])
    if n>0: print(f"    {lt:12s}: {n:5d}")

# =========================================================================
#  STAGE 4: DELTA-TO-FIELD
# =========================================================================
banner(4,"DELTA-TO-FIELD")
V['df']=np.nan; V['dc']=np.nan
for src,sg in V.groupby('source'):
    for tb,g in sg.groupby('time_bin'):
        if len(g)>=3: V.loc[g.index,'df']=g.lt_corr-g.lt_corr.median()
        for comp in g.Compound.dropna().unique():
            cm=g[g.Compound==comp]
            if len(cm)>=2: V.loc[cm.index,'dc']=cm.lt_corr-cm.lt_corr.median()
V.df=V.df.fillna(0); V.dc=V.dc.fillna(0)

V['race_df']=np.nan
for src,sg in V.groupby('source'):
    rl=sg[sg.lap_type=='RACE_PACE']
    if len(rl)>=5:
        med=rl.lt_corr.median(); V.loc[rl.index,'race_df']=rl.lt_corr-med
V.race_df=V.race_df.fillna(V.df)

a=2/(8+1)
for drv in V.Driver.unique():
    m=V.Driver==drv;dv=V.loc[m,'df'].values
    ew=np.empty_like(dv);ew[0]=dv[0]
    for i in range(1,len(dv)):ew[i]=a*dv[i]+(1-a)*ew[i-1]
    V.loc[m,'mom']=ew;V.loc[m,'vol']=pd.Series(dv).rolling(10,min_periods=3).std().values
print(f"  Race pace: {len(V[V.lap_type=='RACE_PACE']):,} laps | Quali: {len(V[V.lap_type=='QUALI_PUSH']):,}")

# =========================================================================
#  STAGE 5: FEATURES
# =========================================================================
banner(5,"FEATURES")
drv_df=V.groupby(['Driver','Team']).size().reset_index()[['Driver','Team']]
feats={}
for _,row in drv_df.iterrows():
    drv,team=row.Driver,row.Team
    d=V[(V.Driver==drv)&(V.Team==team)]; f={}
    d_t1=d[d.source_tier==1]; d_t3=d[d.source_tier==3]
    def wmean(data,col):
        v=data[[col,'source_weight']].dropna(subset=[col])
        if len(v)==0:return np.nan
        return np.average(v[col],weights=v['source_weight'])

    # Race pace (ALL sources via delta)
    rl=d[d.lap_type=='RACE_PACE']; rl1=d_t1[d_t1.lap_type=='RACE_PACE']; rl3=d_t3[d_t3.lap_type=='RACE_PACE']
    if len(rl1)>=3:
        rp=rl1.lt_corr.sort_values().iloc[:max(3,len(rl1)*2//3)]
        f['fp_race_pace']=rp.mean(); f['fp_race_n']=len(rl1)
        f['fp_race_df']=rl1.race_df.mean()
    elif len(rl1)>=1:
        # Few race pace laps — use what we have but mark as low confidence
        f['fp_race_pace']=rl1.lt_corr.mean(); f['fp_race_n']=len(rl1)
        f['fp_race_df']=rl1.race_df.mean() if rl1.race_df.notna().any() else np.nan
    else:
        hot=d_t1[d_t1.lap_type.isin(['HOT_LAP','QUALI_PUSH'])]
        f['fp_race_pace']=hot.lt_corr.min()*1.03 if len(hot)>=1 else np.nan
        f['fp_race_n']=0; f['fp_race_df']=np.nan
    if len(rl)>=3:
        f['race_df_all']=wmean(rl,'race_df'); f['race_df_q10']=rl.race_df.quantile(.10); f['race_n_all']=len(rl)
    else: f['race_df_all']=f['race_df_q10']=np.nan; f['race_n_all']=0
    if len(rl3)>=5: f['test_race_df']=wmean(rl3,'race_df')
    else: f['test_race_df']=np.nan

    # Quali pace
    ql1=d_t1[d_t1.lap_type.isin(['QUALI_PUSH','HOT_LAP'])]
    f['quali_pace']=ql1.lt_corr.min() if len(ql1)>=1 else np.nan
    f['quali_n']=len(ql1)

    # Race-to-quali gap
    ql=d[d.lap_type.isin(['QUALI_PUSH','HOT_LAP'])]; rpd=d[d.lap_type=='RACE_PACE']
    if len(ql)>=1 and len(rpd)>=3:
        gaps=[]
        for src in d.source.unique():
            sq=ql[ql.source==src];sr=rpd[rpd.source==src]
            if len(sq)>=1 and len(sr)>=2: gaps.append(sr.lt_corr.mean()-sq.lt_corr.min())
        f['race_quali_gap']=np.mean(gaps) if gaps else np.nan
    else: f['race_quali_gap']=np.nan

    # FP improvement
    if len(d_t1)>=3:
        fps=d_t1.source_session.unique()
        if 'FP1' in fps and 'FP2' in fps:
            f1d=d_t1[d_t1.source_session=='FP1'];f2d=d_t1[d_t1.source_session=='FP2']
            f['fp_improve']=f1d.df.mean()-f2d.df.mean() if len(f1d)>=2 and len(f2d)>=2 else np.nan
        else: f['fp_improve']=np.nan
    else: f['fp_improve']=np.nan

    # Overall delta
    f['df_mean']=wmean(d,'df'); f['df_q10']=d.df.quantile(.10); f['df_std']=d.df.std()

    # Sectors
    sd=d_t1 if len(d_t1)>=5 else d
    for si,sc in enumerate(('Sector1Time_s','Sector2Time_s','Sector3Time_s'),1):
        sv=sd[sc].dropna()
        if len(sv)>=3: f[f'bs{si}']=sv.min(); f[f'ms{si}']=sv.mean()
        else: f[f'bs{si}']=f[f'ms{si}']=np.nan

    # Degradation
    stints=[]
    for(src,day,sid) in d[['source','Day','Stint']].drop_duplicates().values:
        s=d[(d.source==src)&(d.Day==day)&(d.Stint==sid)&(d.TyreLife>=2)].sort_values('TyreLife')
        if len(s)>=5:
            med=s.lt_corr.median();s=s[abs(s.lt_corr-med)<4]
        if len(s)>=5: stints.append(s)
    if stints:
        A=pd.concat(stints);tl=A.TyreLife.values.astype(float);lt=A.lt_corr.values
        if len(A)>=8: f['deg_slope']=sp.linregress(tl,lt).slope
        else: f['deg_slope']=np.nan
        green=[l.lt_corr for s in stints for _,l in s.iterrows() if 3<=l.TyreLife<=7]
        degp=[l.lt_corr for s in stints for _,l in s.iterrows() if l.TyreLife>15]
        f['tyre_pres']=(np.mean(degp)/np.mean(green)-1) if(green and degp) else np.nan
    else: f['deg_slope']=f['tyre_pres']=np.nan

    # Consistency + meta
    f['iqr']=d.lt_corr.quantile(.75)-d.lt_corr.quantile(.25)
    f['push_pct']=(d.lt_corr<d.lt_corr.min()*1.03).sum()/max(len(d),1)
    f['total_laps']=len(d); f['fp_laps']=len(d_t1)
    dvals=d.sort_values('LapStartTime_s').df.dropna().values
    f['con_ac']=np.corrcoef(dvals[:-1],dvals[1:])[0,1] if len(dvals)>=10 else np.nan

    # Momentum
    dd=d.sort_values('LapStartTime_s')
    if len(dd)>=5:
        f['mom_fin']=dd.mom.iloc[-1];f['vol_mean']=dd.vol.mean()
        f['sharpe']=-f['df_mean']/f['vol_mean'] if pd.notna(f['vol_mean']) and f['vol_mean']>0 else np.nan
    else: f['mom_fin']=f['vol_mean']=f['sharpe']=np.nan

    # Speed
    sd=d_t1 if len(d_t1)>=3 else d
    f['spd_st']=sd.SpeedST.max();f['spd_fl']=sd.SpeedFL.max()

    feats[drv]=f

# ── PACE DELTA (% of fastest) ──
# Fastest driver = 100%, everyone else > 100%
# Combines FP race pace + actual race gap_rate from past races

# Load season race gap rates if available
_season_gap_rates = {}
_season_path = SCRIPT_DIR / 'season_race_data.csv'
if _season_path.exists():
    _srd = pd.read_csv(_season_path)
    _srd_past = _srd[_srd.Round < cur_rnd]
    if len(_srd_past) > 0:
        # Per-driver weighted average gap_rate (recent races weighted more)
        for drv in _srd_past.Driver.unique():
            dd = _srd_past[_srd_past.Driver == drv].sort_values('Round')
            gr = dd[dd.gap_rate.notna()]
            if len(gr) > 0:
                wts = np.array([0.5 ** (len(gr)-1-i) for i in range(len(gr))])
                wts /= wts.sum()
                _season_gap_rates[drv] = np.average(gr.gap_rate.values, weights=wts)

# Combined race pace delta: blend FP pace + race gap_rate
# FP pace = raw speed on this circuit
# Gap rate = how fast they actually race (includes racecraft, strategy, consistency)
race_paces = {}
for d in feats:
    fp_pace = feats[d].get('fp_race_pace')
    gap_rate = _season_gap_rates.get(d)
    
    if pd.notna(fp_pace) and gap_rate is not None:
        # Blend: 60% current FP pace, 40% season race gap rate
        # Convert gap_rate to a pace-like metric (lower = better)
        # gap_rate ~0 = leader pace, gap_rate ~1.0 = ~1s/lap slower
        # Scale gap_rate to match FP pace units
        race_pace_from_gap = fp_pace * (1 + gap_rate / 100)
        race_paces[d] = fp_pace * 0.60 + race_pace_from_gap * 0.40
    elif pd.notna(fp_pace):
        race_paces[d] = fp_pace
    elif gap_rate is not None:
        # No FP data but have race history — use median FP pace + gap adjustment
        med_fp = np.median([feats[x].get('fp_race_pace') for x in feats 
                           if pd.notna(feats[x].get('fp_race_pace'))])
        race_paces[d] = med_fp * (1 + gap_rate / 100)

if race_paces:
    best_race = min(race_paces.values())
    for d in feats:
        rp = race_paces.get(d)
        feats[d]['race_delta_pct'] = (rp / best_race) * 100 if rp is not None else np.nan

# Quali pace delta %
quali_paces = {d: feats[d].get('quali_pace') for d in feats if pd.notna(feats[d].get('quali_pace'))}
if quali_paces:
    best_quali = min(quali_paces.values())
    for d in feats:
        qp = feats[d].get('quali_pace')
        feats[d]['quali_delta_pct'] = (qp / best_quali) * 100 if pd.notna(qp) else np.nan

# Overall best lap delta %
best_laps = {d: feats[d].get('fp_race_pace', feats[d].get('quali_pace')) for d in feats 
             if pd.notna(feats[d].get('fp_race_pace', feats[d].get('quali_pace')))}
if best_laps:
    field_best = min(best_laps.values())
    for d in feats:
        bl = best_laps.get(d)
        feats[d]['overall_delta_pct'] = (bl / field_best) * 100 if bl and pd.notna(bl) else np.nan

# Teammate delta % (how far off your teammate)
for team in set(drv_df.Team):
    team_drvs = drv_df[drv_df.Team == team].Driver.tolist()
    if len(team_drvs) < 2: continue
    team_paces = {d: feats[d].get('fp_race_pace') for d in team_drvs if pd.notna(feats[d].get('fp_race_pace'))}
    if team_paces:
        team_best = min(team_paces.values())
        for d in team_drvs:
            tp = team_paces.get(d)
            feats[d]['tm_race_delta_pct'] = (tp / team_best) * 100 if tp and pd.notna(tp) else np.nan

# Cross-source race delta % (testing + FP combined via race_df_all)
race_dfs = {d: feats[d].get('race_df_all') for d in feats if pd.notna(feats[d].get('race_df_all'))}
if race_dfs:
    best_df = min(race_dfs.values())
    worst_df = max(race_dfs.values())
    rng = worst_df - best_df if worst_df != best_df else 1
    for d in feats:
        df_val = feats[d].get('race_df_all')
        # Convert delta-to-field into 100% scale (best=100, worst=100+spread)
        feats[d]['race_df_delta_pct'] = 100 + ((df_val - best_df) / abs(best_df) * 100) if pd.notna(df_val) and best_df != 0 else np.nan

feat_df=pd.DataFrame.from_dict(feats,orient='index')
feat_df.index.name='Driver';feat_df=feat_df.reset_index().merge(drv_df,on='Driver')
# Remove drivers with insufficient data (reserve drivers doing 1 FP session)
MIN_LAPS = 15  # need at least 15 laps to make a meaningful prediction
low_data = feat_df[feat_df.fp_laps < MIN_LAPS].Driver.tolist()
if low_data:
    print(f"  Excluded (< {MIN_LAPS} laps on current weekend): {', '.join(low_data)}")
    feat_df = feat_df[feat_df.fp_laps >= MIN_LAPS].reset_index(drop=True)
print(f"  {len(feat_df.columns)-2} features x {len(feat_df)} drivers")

# =========================================================================
#  STAGE 6: PROFILES (History capped, constructors integrated)
# =========================================================================
banner(6,"PROFILES")

# Driver profiles - CAP history influence for new regs
hp=None
for p in [SCRIPT_DIR/'driver_profiles.csv',SCRIPT_DIR/'model'/'driver_profiles.csv',
          DATA_ROOT/'historical'/'driver_profiles.csv']:
    if p.exists(): hp=pd.read_csv(p); print(f"  Drivers: {len(hp)} from {p.name}"); break
if hp is None: hp=pd.DataFrame()

hcols=[c for c in hp.columns if c not in('driver','driver_score','n_races') and hp[c].dtype in('float64','int64')]
for col in hcols: feat_df[f'h_{col}']=np.nan
if len(hp)>0:
    for _,row in hp.iterrows():
        drv=row.get('driver','')
        if drv in feat_df.Driver.values:
            for col in hcols:
                if col in row.index and pd.notna(row[col]):
                    feat_df.loc[feat_df.Driver==drv,f'h_{col}']=row[col]
            if 'driver_score' in row.index and pd.notna(row['driver_score']):
                feat_df.loc[feat_df.Driver==drv,'driver_score']=row['driver_score']
for col in hcols: feat_df[f'h_{col}']=feat_df[f'h_{col}'].fillna(feat_df[f'h_{col}'].median())
if 'driver_score' not in feat_df.columns: feat_df['driver_score']=50.0
feat_df['driver_score']=feat_df['driver_score'].fillna(feat_df['driver_score'].median())

# Constructor profiles - integrate if available
cp=None
for p in [SCRIPT_DIR/'constructor_profiles.csv']:
    if p.exists(): cp=pd.read_csv(p); print(f"  Constructors: {len(cp)} teams"); break
if cp is not None:
    for _,row in cp.iterrows():
        team=row['team']
        mask=feat_df.Team==team
        if mask.any():
            if pd.notna(row.get('team_score')): feat_df.loc[mask,'team_score']=row['team_score']
            if pd.notna(row.get('avg_race_pace')): feat_df.loc[mask,'team_pace']=row['avg_race_pace']
            if pd.notna(row.get('reliability')): feat_df.loc[mask,'team_rel']=row['reliability']
            if pd.notna(row.get('strategy_score')): feat_df.loc[mask,'team_strat']=row['strategy_score']

for c in ['team_score','team_pace','team_rel','team_strat']:
    if c not in feat_df.columns: feat_df[c]=np.nan
    feat_df[c]=feat_df[c].fillna(feat_df[c].median())

# Load season race metrics (gap rates, racecraft, deg from past races)
season_data_path = SCRIPT_DIR / 'season_race_data.csv'
HAS_SEASON_DATA = False
if season_data_path.exists():
    try:
        from f1_race_metrics import compute_season_features
        srd = pd.read_csv(season_data_path)
        # Only use data from PAST rounds (not current)
        srd_past = srd[srd.Round < cur_rnd]
        if len(srd_past) > 0:
            sf = compute_season_features(srd_past)
            for _, row in sf.iterrows():
                drv = row.Driver
                if drv in feat_df.Driver.values:
                    for col in sf.columns:
                        if col == 'Driver': continue
                        feat_df.loc[feat_df.Driver==drv, col] = row[col]
            for col in sf.columns:
                if col != 'Driver' and col not in fcols and col in feat_df.columns:
                    if feat_df[col].notna().any(): fcols.append(col)
                    feat_df[col] = feat_df[col].fillna(feat_df[col].median())
            HAS_SEASON_DATA = True
            print(f"  Season race data: {len(srd_past)} entries from {srd_past.Round.nunique()} past races")
    except ImportError:
        pass
    except Exception as e:
        print(f"  Season data error: {e}")

# Teammate delta
for team,grp in V.groupby('Team'):
    drvs=grp.Driver.unique()
    if len(drvs)<2:continue
    bests={d:V[(V.Driver==d)&(V.Team==team)].lt_corr.min() for d in drvs}
    sb=sorted(bests.items(),key=lambda x:x[1])
    for d in drvs: feat_df.loc[feat_df.Driver==d,'tm_delta']=bests[d]-sb[0][1]
feat_df['tm_delta']=feat_df['tm_delta'].fillna(feat_df['tm_delta'].median())

# =========================================================================
#  STAGE 7: FEATURE SELECTION
# =========================================================================
banner(7,"FEATURE SELECTION")
meta={'Driver','Team'}
fcols=[c for c in feat_df.columns if c not in meta and feat_df[c].notna().any()]
Xt=feat_df[fcols].fillna(feat_df[fcols].median()).fillna(0)
cm=Xt.corr().abs();up=cm.where(np.triu(np.ones(cm.shape),k=1).astype(bool))
drop=set()
for col in up.columns:
    for hc in up.index[up[col]>0.95]:
        for pref in ['race_','fp_','team_','df','h_']:
            if pref in hc and pref not in col: drop.add(col);break
            elif pref in col and pref not in hc: drop.add(hc);break
        else:
            if Xt[col].std()>=Xt[hc].std():drop.add(hc)
            else:drop.add(col)
fcols=[c for c in fcols if c not in drop]
print(f"  {len(fcols)} features")

# =========================================================================
#  STAGE 8: ADAPTIVE TARGET
# =========================================================================
banner(8,f"TARGET (R{cur_rnd} {cur_circ})")

def ni(col,df):
    if col not in df.columns: return pd.Series(50,index=df.index)
    v=df[col];vv=v.dropna()
    if len(vv)==0:return pd.Series(50,index=df.index)
    mn,mx=vv.min(),vv.max();rng=mx-mn or 1
    return((mx-v)/rng*100).fillna(50)
def np_(col,df):
    if col not in df.columns: return pd.Series(50,index=df.index)
    v=df[col];vv=v.dropna()
    if len(vv)==0:return pd.Series(50,index=df.index)
    mn,mx=vv.min(),vv.max();rng=mx-mn or 1
    return((v-mn)/rng*100).fillna(50)

# IMPROVED: History capped at 20% even at season start
# Current data always trusted more than history
# Tier weights from self-tuned hyperparams (or defaults)
_tw = HP['tier_weights'] if HP else {}
_fp_b = _tw.get('fp_base', 0.40)
_hist_b = _tw.get('history_base', 0.20)
_test_b = _tw.get('testing_base', 0.25)
_team_b = _tw.get('team_base', 0.10)

# History: cap at 12% and decays fast (new regs = old data is misleading)
hw = min(0.12, max(0.03, 0.12 - 0.09*spf))

# Testing: decays aggressively (different circuit, different conditions)
# By R2 (spf=0.08), testing should be ~12% not 24%
tw3 = max(0.02, 0.15 - 0.13*spf) if HAS_TEST else 0

# Past races: grows as season progresses
tw2 = min(0.40, 0.00 + 0.40*spf) if HAS_PAST else 0

# Current weekend FP/Sprint/Q: should be the dominant signal
# Minimum 50% even at R1, growing to 70%+
tw1 = max(0.50, _fp_b + 0.20*spf) if HAS_FP else 0

# Team profiles: useful but small
tw_team = min(0.08, _team_b) if cp is not None else 0
tot=hw+tw3+tw2+tw1+tw_team
if tot>0: hw/=tot;tw3/=tot;tw2/=tot;tw1/=tot;tw_team/=tot
print(f"  FP={tw1:.0%} Season={tw2:.0%} Testing={tw3:.0%} History={hw:.0%} Team={tw_team:.0%}")

target=pd.Series(0.0,index=feat_df.index)
if tw1>0:
    # FP tier is PURELY about race performance
    target+=(ni('fp_race_pace',feat_df)*0.30+       # race pace = #1 signal
             ni('race_delta_pct',feat_df)*0.20+      # pace as % of fastest
             ni('fp_race_df',feat_df)*0.15+           # race pace delta to field
             ni('deg_slope',feat_df)*0.10+             # tyre degradation = race differentiator
             ni('race_to_q_gap',feat_df)*0.10+         # race vs Q gap = untapped race pace
             np_('fp_improve',feat_df)*0.10+           # setup progression
             np_('race_n_all',feat_df)*0.05            # sample confidence
             )*tw1
if tw2>0:
    if HAS_SEASON_DATA:
        target+=(ni('season_gap_rate',feat_df)*0.30+        # gap to leader growth = TRUE pace
                 ni('season_best_stint_deg',feat_df)*0.20+  # tyre degradation in races
                 np_('season_net_gained',feat_df)*0.15+     # positions gained = racecraft
                 np_('season_lap1_gain',feat_df)*0.10+      # start craft
                 np_('season_overtake_ratio',feat_df)*0.10+ # overtaking ability
                 ni('race_df_all',feat_df)*0.15             # FP race pace delta
                 )*tw2
    else:
        target+=(ni('race_df_all',feat_df)*0.50+ni('race_df_q10',feat_df)*0.30+
                 np_('race_n_all',feat_df)*0.20)*tw2
if tw3>0:
    target+=(ni('test_race_df',feat_df)*0.35+       # testing race pace delta
             ni('deg_slope',feat_df)*0.25+              # tyre deg from testing long runs
             ni('tyre_pres',feat_df)*0.15+              # tyre preservation ability
             ni('race_df_all',feat_df)*0.15+            # overall race delta
             np_('total_laps',feat_df)*0.10             # mileage confidence
             )*tw3
if hw>0:
    target+=(np_('driver_score',feat_df)*0.30+       # overall driver rating
             np_('h_tm_race_beat_pct',feat_df)*0.25+   # beats teammate in RACES
             np_('h_avg_pos_gained',feat_df)*0.20+     # gains positions (racecraft)
             ni('tm_delta',feat_df)*0.15+               # teammate gap (current data)
             np_('sharpe',feat_df)*0.10                 # consistency
             )*hw
if tw_team>0:
    target+=(ni('team_pace',feat_df)*0.50+            # team race pace = primary signal
             np_('team_score',feat_df)*0.20+            # composite score
             np_('team_strat',feat_df)*0.15+            # strategy effectiveness
             np_('team_rel',feat_df)*0.15               # reliability
             )*tw_team

feat_df['target']=target
for i,(_,r) in enumerate(feat_df[['Driver','Team','target']].sort_values('target',ascending=False).iterrows(),1):
    print(f"  P{i:2d}  {r.Driver:4s}  {r.Team:20s}  {r.target:.2f}")

# =========================================================================
#  STAGE 8b: GRID
# =========================================================================
# For grid: prefer Q results, fall back to SQ if sprint weekend hasn't had Q yet
quali_src=[s for s in sources if s['type']=='quali' and s['round']==cur_rnd]
if not quali_src:
    # Sprint weekend before qualifying - use SQ for grid order
    quali_src=[s for s in sources if s['type']=='sprint_quali' and s['round']==cur_rnd]
HAS_GRID=False; grid_order={}
if cur_rnd in GRID_OVERRIDES:
    banner('8b','GRID (override)')
    grid_order=GRID_OVERRIDES[cur_rnd]; HAS_GRID=True
elif quali_src and quali_src[0].get('results_path') and quali_src[0]['results_path'].exists():
    banner('8b','GRID (qualifying)')
    qr=pd.read_csv(quali_src[0]['results_path'])
    cl=qr[qr.Position.notna()].sort_values('Position')
    ucl=qr[qr.Position.isna()]; mx=int(cl.Position.max()) if len(cl)>0 else 0
    for _,row in cl.iterrows(): grid_order[row.Abbreviation]=int(row.Position)
    for i,(_,row) in enumerate(ucl.iterrows()): grid_order[row.Abbreviation]=mx+1+i
    HAS_GRID=len(grid_order)>0

# Load sprint results if available (finish position = strongest mid-weekend signal)
sprint_src=[s for s in sources if s['type']=='sprint_race' and s['round']==cur_rnd and s.get('results_path')]
HAS_SPRINT=False; sprint_results={}
if sprint_src and sprint_src[0].get('results_path') and sprint_src[0]['results_path'].exists():
    spr=pd.read_csv(sprint_src[0]['results_path'])
    for _,row in spr.iterrows():
        if pd.notna(row.get('Position')) and row.Position>0:
            sprint_results[row.Abbreviation]=int(row.Position)
    HAS_SPRINT=len(sprint_results)>0
    if HAS_SPRINT:
        nd_s=max(sprint_results.values())
        feat_df['sprint_pos']=feat_df.Driver.map(sprint_results).fillna(nd_s+1)
        feat_df['sprint_score']=(nd_s+1-feat_df['sprint_pos'])/nd_s*100
        # Sprint pos/score used ONLY in target blend, NOT as ML features
        # (prevents overfitting: model shouldn't learn "sprint P1 = race P1")
        # The pace data from sprint laps already feeds through race_pace features
        print(f"  Sprint results loaded: {len(sprint_results)} drivers")
        for d in sorted(sprint_results.keys(), key=lambda x: sprint_results[x])[:5]:
            print(f"    P{sprint_results[d]:>2d}  {d}")

if HAS_GRID:
    nd=len(feat_df)
    feat_df['grid_pos']=feat_df.Driver.map(grid_order).fillna(nd)
    feat_df['grid_score']=(nd-feat_df['grid_pos'])/(nd-1)*100

# Extract qualifying metrics from results.csv (Q1/Q2/Q3 times, segments reached)
quali_results_src=[s for s in sources if s['type']=='quali' and s['round']==cur_rnd and s.get('results_path')]
if not quali_results_src and cur_rnd in GRID_OVERRIDES:
    # Try loading from directory directly
    import re as _re
    for item in DATA_ROOT.iterdir():
        _m = _re.match(rf'\d+_R{cur_rnd}_(.+?)_(Q|Qualifying)$', item.name, _re.I)
        if _m and (item/'results.csv').exists():
            quali_results_src = [{'results_path': item/'results.csv'}]
            break

if quali_results_src and quali_results_src[0].get('results_path') and quali_results_src[0]['results_path'].exists():
    qr = pd.read_csv(quali_results_src[0]['results_path'])
    
    def _qtime(td):
        try:
            if pd.isna(td) or td=='': return np.nan
            s=str(td)
            if 'days' in s:
                p=s.split(' days '); h,m,sec=p[1].split(':')
                return float(h)*3600+float(m)*60+float(sec)
            return float(s)
        except: return np.nan
    
    # Parse Q1/Q2/Q3 times
    for qcol in ['Q1','Q2','Q3']:
        if qcol in qr.columns:
            qr[f'{qcol}_s'] = qr[qcol].apply(_qtime)
    
    # Best qualifying time per driver
    q_cols_s = [c for c in ['Q3_s','Q2_s','Q1_s'] if c in qr.columns]
    if q_cols_s:
        qr['q_best'] = qr[q_cols_s].min(axis=1)
        pole_time = qr['q_best'].min()
        
        print(f"\n  Qualifying metrics extracted:")
        print(f"  Pole time: {pole_time:.3f}s")
        
        for _, row in qr.iterrows():
            drv = row.get('Abbreviation', '')
            if drv not in feat_df.Driver.values: continue
            idx = feat_df[feat_df.Driver==drv].index[0]
            
            # 1. Qualifying delta to pole (%)
            if pd.notna(row.get('q_best')) and pd.notna(pole_time):
                feat_df.loc[idx, 'q_delta_pct'] = (row['q_best'] / pole_time) * 100
            
            # 2. Best qualifying time
            feat_df.loc[idx, 'q_best_time'] = row.get('q_best', np.nan)
            
            # 3. Segment reached (Q3=3, Q2=2, Q1=1, none=0)
            if 'Q3_s' in qr.columns and pd.notna(row.get('Q3_s')): feat_df.loc[idx, 'q_segment'] = 3
            elif 'Q2_s' in qr.columns and pd.notna(row.get('Q2_s')): feat_df.loc[idx, 'q_segment'] = 2
            elif 'Q1_s' in qr.columns and pd.notna(row.get('Q1_s')): feat_df.loc[idx, 'q_segment'] = 1
            else: feat_df.loc[idx, 'q_segment'] = 0
            
            # 4. Q2->Q3 improvement (setup progression, only for Q3 drivers)
            if ('Q2_s' in qr.columns and 'Q3_s' in qr.columns and 
                pd.notna(row.get('Q2_s')) and pd.notna(row.get('Q3_s'))):
                feat_df.loc[idx, 'q_improve'] = row['Q2_s'] - row['Q3_s']
            
            # 5. Qualifying position (redundant with grid but useful as feature)
            if pd.notna(row.get('Position')):
                feat_df.loc[idx, 'q_pos'] = int(row['Position'])
        
        # 6. Teammate qualifying gap
        for team in feat_df.Team.unique():
            tm = feat_df[feat_df.Team==team]
            qt = tm['q_best_time'].dropna()
            if len(qt) >= 2:
                best_tm = qt.min()
                for idx in tm.index:
                    if pd.notna(feat_df.loc[idx, 'q_best_time']):
                        feat_df.loc[idx, 'q_tm_gap'] = feat_df.loc[idx, 'q_best_time'] - best_tm
        
        # 7. Race pace vs quali pace gap (from FP race runs vs Q time)
        # Bigger gap = more fuel-adjusted pace advantage in race
        for idx, row in feat_df.iterrows():
            rp = row.get('fp_race_pace')
            qp = row.get('q_best_time')
            if pd.notna(rp) and pd.notna(qp) and qp > 0:
                feat_df.loc[idx, 'race_to_q_gap'] = (rp / qp - 1) * 100  # % slower in race trim
        
        # Add new features to fcols
        for qf in ['q_delta_pct','q_segment','q_improve','q_tm_gap','race_to_q_gap','q_pos']:
            if qf in feat_df.columns and feat_df[qf].notna().any():
                if qf not in fcols: fcols.append(qf)
        
        # Fill NaN with medians
        for qf in ['q_delta_pct','q_segment','q_improve','q_tm_gap','race_to_q_gap']:
            if qf in feat_df.columns:
                feat_df[qf] = feat_df[qf].fillna(feat_df[qf].median())
        
        # Print qualifying summary
        print(f"\n  {'DRV':4s} {'GRD':>3s} {'SEG':>3s} {'Q-DELTA':>8s} {'TM GAP':>7s} {'R/Q GAP':>7s}")
        print(f"  {'-'*38}")
        for _, r in feat_df.sort_values('grid_pos').head(15).iterrows():
            gp = int(r.get('grid_pos', 22))
            seg = f"Q{int(r.get('q_segment', 0))}" if r.get('q_segment', 0) > 0 else ' --'
            qd = f"{r.get('q_delta_pct', 0):.2f}%" if pd.notna(r.get('q_delta_pct')) else '   N/A'
            tg = f"{r.get('q_tm_gap', 0):+.3f}s" if pd.notna(r.get('q_tm_gap')) else '   N/A'
            rqg = f"{r.get('race_to_q_gap', 0):+.1f}%" if pd.notna(r.get('race_to_q_gap')) else '   N/A'
            print(f"  {r.Driver:4s} P{gp:>2d}  {seg} {qd:>8s} {tg:>7s} {rqg:>7s}")
    # IMPROVED: 15% grid blend (up from 10%)
    # Grid blend: 20% base, higher when we have full Q data
    _gb_base = HP['grid_model']['grid_blend_pct'] if HP else 0.20
    has_full_q = 'q_delta_pct' in feat_df.columns and feat_df.q_delta_pct.notna().sum() >= 10
    GRID_BLEND = min(0.25, _gb_base + 0.05) if has_full_q else _gb_base
    # Single blend operation — sprint replaces grid weight, not additive
    grid_t=(nd-feat_df['grid_pos'])/max(nd-1,1)*100
    if HAS_SPRINT:
        sprint_t=feat_df['sprint_score'].fillna(50)
        # Sprint subsumes grid: total position-based weight stays at GRID_BLEND
        # but split between sprint (stronger signal) and grid
        SP_SHARE=0.70  # 70% of position weight goes to sprint, 30% to grid
        feat_df['target'] = (feat_df['target']*(1-GRID_BLEND) +
                            sprint_t*GRID_BLEND*SP_SHARE +
                            grid_t*GRID_BLEND*(1-SP_SHARE))
        print(f"  {1-GRID_BLEND:.0%} model + {GRID_BLEND*SP_SHARE:.0%} sprint + {GRID_BLEND*(1-SP_SHARE):.0%} grid")
    else:
        feat_df['target']=feat_df['target']*(1-GRID_BLEND)+grid_t*GRID_BLEND
        print(f"  {1-GRID_BLEND:.0%} model + {GRID_BLEND:.0%} grid")
    for gc in ['grid_pos','grid_score']:
        if gc not in fcols: fcols.append(gc)
    print(f"  {1-GRID_BLEND:.0%} model + {GRID_BLEND:.0%} grid")
else:
    print(f"\n  No qualifying data")

# =========================================================================
#  STAGE 9: ENSEMBLE
# =========================================================================
banner(9,"ENSEMBLE")
X=feat_df[fcols].fillna(feat_df[fcols].median()).fillna(0);y=feat_df.target.values;nd=len(feat_df)
print(f"  {X.shape[1]} features, {nd} drivers")
sc=StandardScaler();Xs=sc.fit_transform(X)
models={
    'GBR':GradientBoostingRegressor(n_estimators=300,max_depth=3,learning_rate=.03,subsample=.8,min_samples_leaf=2,random_state=42),
    'RF':RandomForestRegressor(n_estimators=500,max_depth=4,min_samples_leaf=2,max_features='sqrt',random_state=42),
    'ET':ExtraTreesRegressor(n_estimators=500,max_depth=4,min_samples_leaf=2,random_state=42),
    'Ridge':Ridge(alpha=1.0),'BRdg':BayesianRidge(),'SVR':SVR(kernel='rbf',C=10,epsilon=.5),
}
ns={'Ridge','BRdg','SVR'};loo=LeaveOneOut();msc={}
print(f"\n  LOO CV:")
for nm,mdl in models.items():
    preds=[]
    for tr,te in loo.split(X):
        if nm in ns:mdl.fit(Xs[tr],y[tr]);preds.append(mdl.predict(Xs[te])[0])
        else:mdl.fit(X.values[tr],y[tr]);preds.append(mdl.predict(X.values[te])[0])
    preds=np.array(preds);mae=mean_absolute_error(y,preds)
    rho,_=sp.spearmanr(y,preds);tau,_=sp.kendalltau(y,preds)
    msc[nm]=dict(mae=mae,rho=rho,tau=tau)
    print(f"    {nm:6s}  MAE:{mae:6.3f}  rho:{rho:.4f}  tau:{tau:.4f}")
wts={n:max(0,s['tau']) for n,s in msc.items()};tw=sum(wts.values()) or 1
wts={k:v/tw for k,v in wts.items()}
ipreds={}
for nm,mdl in models.items():
    if nm in ns:mdl.fit(Xs,y);ipreds[nm]=mdl.predict(Xs)
    else:mdl.fit(X.values,y);ipreds[nm]=mdl.predict(X.values)
fscore=sum(ipreds[n]*w for n,w in wts.items())

# =========================================================================
#  STAGE 10: IMPROVED MONTE CARLO
# =========================================================================
banner(10,f"MONTE CARLO ({N_SIM:,})")
R=feat_df[['Driver','Team']].copy();R['score']=fscore
dp_mc={};tidx=defaultdict(list)
for i,row in R.iterrows():
    drv=row.Driver;mstd=np.std([ipreds[n][i] for n in models])
    hp_row=hp[hp.driver==drv] if len(hp)>0 and 'driver' in hp.columns else pd.DataFrame()
    # DNF rate from constructor profile if available, else historical
    dnf_rate=0.05  # base DNF rate ~5%
    if cp is not None:
        cp_row=cp[cp.team==row.Team]
        if len(cp_row)>0 and pd.notna(cp_row.iloc[0].get('reliability')):
            team_dnf=(100-cp_row.iloc[0].reliability)/100
            # Blend with base rate — don't let 1 bad race set 50% DNF
            dnf_rate=dnf_rate*0.5+team_dnf*0.5
    elif len(hp_row)>0 and 'dnf_rate' in hp_row.columns:
        dnf_rate=min(0.15, hp_row.iloc[0].dnf_rate/100)  # cap at 15%
    dnf_rate=min(dnf_rate, 0.15)  # absolute cap — nobody has >15% DNF rate
    
    iqr_v=feat_df.loc[feat_df.Driver==drv,'iqr'].values[0]
    hist_std=hp_row.iloc[0].std_finish if len(hp_row)>0 and 'std_finish' in hp_row.columns and pd.notna(hp_row.iloc[0].std_finish) else 5
    unc=np.sqrt(mstd**2+(hist_std*.8)**2+(iqr_v if pd.notna(iqr_v) else 3)**2+2**2)
    
    # IMPROVED: Rookies with no history get LESS uncertainty reduction from FP
    # (their FP data is real but race execution is unknown)
    fp_n=feat_df.loc[feat_df.Driver==drv,'fp_laps'].values[0]
    has_history=len(hp_row)>0 and pd.notna(hp_row.iloc[0].get('avg_finish'))
    _up=HP['uncertainty'] if HP else {}
    if pd.notna(fp_n) and fp_n>10:
        unc*=_up.get('fp_reduction',0.85) if has_history else _up.get('fp_reduction_rookie',0.92)
    if past_races>=3: unc*=_up.get('season_data_reduction',0.90)
    if HAS_GRID: unc*=_up.get('grid_reduction',0.85)
    if HAS_SPRINT: unc*=0.88  # sprint data helps but doesn't eliminate uncertainty
    
    grd=grid_order.get(drv,11) if HAS_GRID else 11
    # Extra upside variance for strong drivers starting out of position
    if has_history:
        exp=hp_row.iloc[0].get('avg_finish',11)
        if pd.notna(exp) and grd>exp+5: unc+=1.5  # reduced from 2.0

    dp_mc[i]=dict(score=row.score,unc=unc,dnf=dnf_rate,team=row.Team,grid=grd)
    tidx[row.Team].append(i)

bt={i:np.exp(dp_mc[i]['score']/20) for i in range(nd)}
np.random.seed(42)
fc=np.zeros((nd,nd));wc=np.zeros(nd);pc=np.zeros(nd);pt=np.zeros(nd)
pm={0:25,1:18,2:15,3:12,4:10,5:8,6:6,7:4,8:2,9:1}

for _ in range(N_SIM):
    scores=np.zeros(nd)
    tshock={t:np.random.standard_t(5)*1.5 for t in tidx}
    fln=np.random.standard_t(4,size=nd)*1.5
    nsc=np.random.poisson(0.025*RACE_LAPS);scc=1-.12*min(nsc,3)
    for i in range(nd):
        p=dp_mc[i];tb=.03 if tshock[p['team']]<-2 else 0
        if np.random.random()<p['dnf']+tb:scores[i]=-1e4;continue
        gpos=p['grid']
        # IMPROVED GRID PENALTY: 2026 regs allow more overtaking
        # P1-3: clean air bonus, P4-10: mild traffic, P11+: moderate
        # Less harsh than before (overtaking is easier in 2026)
        gp=0
        if HAS_GRID:
            _gm=HP['grid_model'] if HP else {}
            _ca=_gm.get('clean_air_bonus_per_slot',1.8)  # stronger clean air
            _p1=_gm.get('p4_p10_penalty_per_slot',0.5)
            _p2=_gm.get('p11_p15_penalty_per_slot',0.7)
            _p3=_gm.get('p16_plus_penalty_per_slot',1.0)
            # Apply circuit-specific overtaking factor if available
            _of=1.0
            if HP and cur_circ in HP.get('circuit_profiles',{}):
                _of=HP['circuit_profiles'][cur_circ].get('overtaking_factor',1.0)
                _of=max(0.5,min(2.0,_of))  # clamp
            if gpos==1: gp=_ca*2.5/_of  # POLE BONUS: P1 has massive first-corner + strategy advantage
            elif gpos<=3: gp=(3-gpos)*_ca/_of
            elif gpos<=10: gp=-(gpos-3)*_p1*_of
            elif gpos<=15: gp=-(7*_p1+(gpos-10)*_p2)*_of
            else: gp=-(7*_p1+5*_p2+(gpos-15)*_p3)*_of
        scores[i]=(p['score']+gp+np.random.standard_t(5)*p['unc']*.5+
                   tshock[p['team']]+fln[i]+np.random.normal(0,2))*scc
    ranking=np.argsort(-scores)
    for pos in range(len(ranking)-1):
        a2,b2=ranking[pos],ranking[pos+1]
        if scores[a2]>-5000 and scores[b2]>-5000 and abs(scores[a2]-scores[b2])<3:
            pa=bt[a2]/(bt[a2]+bt[b2])
            if np.random.random()>pa:ranking[pos],ranking[pos+1]=ranking[pos+1],ranking[pos]
    for pos,idx in enumerate(ranking):
        fc[idx,pos]+=1
        if pos==0:wc[idx]+=1
        if pos<3:pc[idx]+=1
        if pos in pm:pt[idx]+=pm[pos]

R['win_pct']=wc/N_SIM*100;R['pod_pct']=pc/N_SIM*100;R['avg_pts']=pt/N_SIM
# Sanity cap: max win probability 40% (even dominant drivers face DNF/SC risk)
WIN_CAP = 40
if R['win_pct'].max() > WIN_CAP:
    excess = R['win_pct'].max() - WIN_CAP
    top_idx = R['win_pct'].idxmax()
    R.loc[top_idx, 'win_pct'] = WIN_CAP
    others = R.index[R.index != top_idx]
    top5 = R.loc[others].nlargest(5, 'win_pct').index
    R.loc[top5, 'win_pct'] += excess / len(top5)
for tag,pct in(('p10',10),('p50',50),('p90',90)):
    R[tag]=[np.percentile(np.repeat(np.arange(1,nd+1),fc[i].astype(int)),pct) for i in range(nd)]
R['avg_fin']=[np.mean(np.repeat(np.arange(1,nd+1),fc[i].astype(int))) for i in range(nd)]
R['Name']=R.Driver.map(NAMES).fillna(R.Driver)  # fallback to abbreviation
print(f"  Done")

# Print pace delta table
if 'race_delta_pct' in feat_df.columns and feat_df.race_delta_pct.notna().any():
    has_race_data = len(_season_gap_rates) > 0
    src_tag = "FP + Race history" if has_race_data else "FP only"
    print(f"\n  PACE DELTA ({src_tag}, fastest = 100%)")
    print(f"  {'DRV':4s} {'COMBINED':>8s} {'FP%':>7s} {'RACE':>8s} {'TM%':>7s}  BAR")
    print(f"  {'-'*60}")
    for _, r in feat_df.sort_values('race_delta_pct', na_position='last').iterrows():
        rp = r.get('race_delta_pct', np.nan)
        qp = r.get('quali_delta_pct', np.nan)
        tp = r.get('tm_race_delta_pct', np.nan)
        # Show season gap rate as indicator
        sgr = _season_gap_rates.get(r.Driver)
        rp_s = f"{rp:.2f}%" if pd.notna(rp) else '   N/A  '
        qp_s = f"{qp:.2f}%" if pd.notna(qp) else '  N/A  '
        gr_s = f"{sgr:+.2f}s/L" if sgr is not None else '   N/A '
        tp_s = f"{tp:.2f}%" if pd.notna(tp) else '  N/A  '
        bar_len = int((rp - 100) * 10) if pd.notna(rp) else 0
        bar = '|' + '#' * min(bar_len, 40)
        print(f"  {r.Driver:4s} {rp_s:>8s} {qp_s:>7s} {gr_s:>8s} {tp_s:>7s}  {bar}")

# =========================================================================
#  STAGE 11: OUTPUT
# =========================================================================
banner(11,f"R{cur_rnd} {cur_circ} GP")
CAREER={'VER':[1.5,1.4,2.8,2.2],'HAM':[6.0,5.5,7.2,6.8],'LEC':[5.0,4.8,4.5,4.2],
'NOR':[8.5,6.5,3.2,3.0],'RUS':[5.5,6.0,5.8,5.0],'PIA':[None,7.5,4.8,4.0],
'ALO':[5.0,5.5,9.0,10.5],'SAI':[5.5,5.0,4.5,5.0],'GAS':[9.0,10.0,11.0,11.5],
'OCO':[10.0,10.5,11.5,12.0],'ALB':[14.0,13.0,11.0,10.0],'BOT':[8.5,15.0,16.0,14.0],
'HUL':[None,12.0,10.5,11.0],'STR':[12.0,10.0,14.0,15.0],'PER':[4.5,4.0,8.5,12.0],
'LAW':[None,None,15.0,11.0],'ANT':[None,None,None,14.0],'COL':[None,None,14.0,None],
'BEA':[None,None,12.0,11.0],'BOR':[None]*4,'HAD':[None]*4,'LIN':[None]*4}
def _form(drv,pp):
    h=CAREER.get(drv,[None]*4);vh=[(i,p) for i,p in enumerate(h) if p is not None]
    if not vh:d=15-pp;return '^^' if d>6 else '^>' if d>2 else '>>' if d>-2 else 'v>'
    pts=vh+[(4,pp)];xs,ys=zip(*pts);sl=sp.linregress(xs,ys).slope;rd=vh[-1][1]-pp
    if sl<-1.5 and rd>3:return '^^'
    if sl<-.5 or rd>1.5:return '^>'
    if abs(sl)<=.5 and abs(rd)<=1.5:return '>>'
    return 'v>'
R=R.sort_values('avg_fin',ascending=True).reset_index(drop=True);R['Pos']=range(1,nd+1)
R['fm']=[_form(r.Driver,r.avg_fin) for _,r in R.iterrows()]

print(f"\n  FP={tw1:.0%} Season={tw2:.0%} Test={tw3:.0%} Hist={hw:.0%} Team={tw_team:.0%}")
if HAS_GRID:
    R['GridPos']=R.Driver.map(grid_order).fillna(nd).astype(int)
    mx_g=max(grid_order.values()) if grid_order else 22
    print(f"\n  {'POS':>3s} {'FM':>2s} {'GRD':>3s}  {'DRIVER':<22s}  {'TEAM':<20s}  {'WIN%':>5s} {'POD%':>5s} {'E[PT]':>6s} {'AVG':>4s}  {'RANGE':>9s}")
    print(f"  {'-'*100}")
    for _,r in R.iterrows():
        grd=f"P{r.GridPos:>2d}" if r.GridPos<=mx_g else "PIT"
        print(f"  {int(r.Pos):>3d}  {r.fm:>2s} {grd:>3s}  {r.Name:<22s}  {r.Team:<20s}  {r.win_pct:5.1f} {r.pod_pct:5.1f} {r.avg_pts:6.2f} {r.avg_fin:4.1f}  P{int(r.p10):>2d}-P{int(r.p90):<2d}")
else:
    print(f"\n  {'POS':>3s} {'FM':>2s}  {'DRIVER':<22s}  {'TEAM':<20s}  {'WIN%':>5s} {'POD%':>5s} {'E[PT]':>6s} {'AVG':>4s}  {'RANGE':>9s}")
    print(f"  {'-'*96}")
    for _,r in R.iterrows():
        print(f"  {int(r.Pos):>3d}  {r.fm:>2s}  {r.Name:<22s}  {r.Team:<20s}  {r.win_pct:5.1f} {r.pod_pct:5.1f} {r.avg_pts:6.2f} {r.avg_fin:4.1f}  P{int(r.p10):>2d}-P{int(r.p90):<2d}")

dist=fc/N_SIM*100
print(f"\n  POSITION PROBABILITY (%)")
hdr=f"  {'':7s}{'DRV':<5s}";
for p in range(1,11):hdr+=f"{'P'+str(p):>6s}"
hdr+=f"  {'11-15':>6s} {'16+':>5s}";print(hdr);print(f"  {'-'*82}")
for _,r in R.iterrows():
    i=feat_df[feat_df.Driver==r.Driver].index[0];dd=dist[i]
    gp=grid_order.get(r.Driver,0) if HAS_GRID else 0
    gs=f"{gp:>2d}" if gp>0 else "  "
    line=f"  {r.fm:>2s} {gs} {r.Driver:<5s}"
    for p in range(10):line+=f"{dd[p]:6.1f}" if dd[p]>=0.1 else f"{'--':>6s}"
    line+=f"  {sum(dd[10:15]):6.1f} {sum(dd[15:]):5.1f}";print(line)

# =========================================================================
#  STAGE 12: EXPORT
# =========================================================================
banner(12,"EXPORT")
avg_imp=(models['GBR'].feature_importances_+models['RF'].feature_importances_+models['ET'].feature_importances_)/3
imp_df=pd.DataFrame({'feature':fcols,'importance':avg_imp}).sort_values('importance',ascending=False)
print(f"\n  Top 15:")
for i,(_,r) in enumerate(imp_df.head(15).iterrows(),1):
    print(f"    {i:2d}. {r.feature:28s} {r.importance:.4f}")
out=str(SCRIPT_DIR)
R.to_csv(f'{out}/v2_prediction_results.csv',index=False)
feat_df.to_csv(f'{out}/v2_driver_features.csv',index=False)
imp_df.to_csv(f'{out}/v2_feature_importance.csv',index=False)
fp_out=R[['Pos','fm','Name','Driver','Team','win_pct','pod_pct','avg_pts','avg_fin','p10','p50','p90','score']].copy()
if HAS_GRID: fp_out['GridPos']=fp_out.Driver.map(grid_order)
for p in range(1,nd+1):fp_out[f'P{p}_pct']=[dist[feat_df[feat_df.Driver==d].index[0],p-1] for d in fp_out.Driver]
fp_out.to_csv(f'{out}/v2_full_predictions.csv',index=False)
print(f"\n  Sources: {', '.join(s['label'] for s in sorted(sources,key=lambda x:-x.get('weight',0)))}")
print(f"  DONE!\n")
