"""Paths and constants shared by every stage."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
STORE = DATA / 'store'            # raw FastF1 sessions as parquet: store/<year>/<rr>_<SESSION>/
DERIVED = DATA / 'derived'        # built tables: features, circuit params, strategies
MODELS = ROOT / 'models'          # trained pace model + calibrated sim params
OUTPUT = ROOT / 'predictions_v3'  # per-race predictions, evaluations, backtests
CACHE = ROOT / 'fastf1_cache'

for _p in (STORE, DERIVED, MODELS, OUTPUT, CACHE):
    _p.mkdir(parents=True, exist_ok=True)

SESSION_CODES = {
    'Practice 1': 'FP1', 'Practice 2': 'FP2', 'Practice 3': 'FP3',
    'Qualifying': 'Q', 'Sprint Qualifying': 'SQ', 'Sprint Shootout': 'SQ',
    'Sprint': 'S', 'Race': 'R',
}

POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
DRY = ('SOFT', 'MEDIUM', 'HARD')

# Season weighting for training rows: 2026 is a new rules era, so older seasons count less.
SEASON_WEIGHT = {2024: 0.35, 2025: 0.6, 2026: 1.0}


def event_key(year: int, rnd: int) -> str:
    return f'{year}_R{rnd:02d}'


# Same circuit, renamed between seasons in the FastF1 schedule.
VENUE_ALIAS = {'Yas Marina': 'Yas Island', 'Miami Gardens': 'Miami', 'Monte Carlo': 'Monaco'}


def venue(location):
    return VENUE_ALIAS.get(location, location)
