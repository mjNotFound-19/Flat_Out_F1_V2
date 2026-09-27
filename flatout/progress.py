"""Progress of running experiments (nested benchmarks, unseen-venue tests), from their checkpoint files.

    python -m flatout progress            one snapshot of every unfinished experiment
    python -m flatout progress --watch    refresh every 20 s until all are finished

Works for jobs started in any terminal or in the background: it only reads artifacts/experiments/.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import ROOT

EXPERIMENTS = ROOT / 'artifacts' / 'experiments'


def bar(done, total, width=30, stream=sys.stdout):
    k = done / total if total else 1.0
    enc = getattr(stream, 'encoding', None) or 'ascii'
    try:
        '█░'.encode(enc)
        full, empty = '█', '░'
    except (UnicodeEncodeError, LookupError):
        full, empty = '#', '-'
    n = int(round(k * width))
    return f'{full * n}{empty * (width - n)} {k:4.0%}'


def _eta(stamps, total):
    """Seconds left from the average gap between finished checkpoints (None until two exist)."""
    if len(stamps) < 2:
        return None
    stamps = sorted(stamps)
    per = (stamps[-1] - stamps[0]) / (len(stamps) - 1)
    return per * (total - len(stamps))


def status():
    rows = []
    for d in sorted(EXPERIMENTS.glob('*')):
        cfg_p = d / 'config.json'
        if not d.is_dir() or not cfg_p.exists():
            continue
        cfg = json.loads(cfg_p.read_text())
        total = len(cfg.get('outer', []))
        ck = sorted((d / 'races').glob('*.json'))
        if not (d / 'summary.json').exists() and total:
            stamps = [f.stat().st_mtime for f in ck]
            rows.append(dict(name=d.name, kind='nested', done=len(ck), total=total, eta=_eta(stamps, total)))
        prog = d / 'unseen' / 'progress.json'
        if prog.exists() and not (d / 'unseen' / 'summary.json').exists():
            p = json.loads(prog.read_text())
            rows.append(dict(name=d.name + ' (unseen-venue test)', kind='unseen', done=p['done'], total=p['total'],
                             eta=_eta(p.get('stamps', []), p['total'])))
    return rows


def show(rows, stream=sys.stdout):
    now = datetime.now().strftime('%H:%M:%S')
    if not rows:
        print(f'  {now}  no experiment running', file=stream)
        return
    for r in rows:
        eta = f"~{r['eta'] / 60:.0f} min left" if r['eta'] is not None else 'estimating...'
        print(f"  {now}  {r['name']:<44s} {bar(r['done'], r['total'], stream=stream)}  "
              f"{r['done']}/{r['total']} races  {eta}", file=stream)


def watch(every=20):
    while True:
        rows = status()
        show(rows)
        if not rows:
            return
        time.sleep(every)
