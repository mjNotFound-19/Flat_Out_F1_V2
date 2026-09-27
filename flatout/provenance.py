"""Provenance: content hashes, code identity and experiment snapshots.

Every forecast and experiment records *what* produced it: the git revision (plus a hash of any
uncommitted source diff), the data snapshot hash, and the configuration hash. Snapshots copy
generated artifacts into an immutable folder with a manifest so earlier results can be recovered
exactly before anything regenerates them.

    python -m flatout snapshot --label pre_upgrade      copy dirty/untracked outputs + manifest
"""
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .config import ROOT

SNAPSHOTS = ROOT / 'artifacts' / 'snapshots'
SOURCE_DIRS = ('flatout', 'web')               # code whose uncommitted diff changes behaviour
OUTPUT_PREFIXES = ('data/derived/', 'models/', 'predictions_v3/', 'web/data/', 'data/fix_run_log')


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_json(obj):
    """Stable hash of a JSON-serialisable object (sorted keys, no whitespace)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def _git(*args):
    try:
        return subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return ''


def code_identity():
    """Git revision plus a hash of the uncommitted diff in source directories (None when clean)."""
    rev = _git('rev-parse', 'HEAD').strip() or None
    diff = _git('diff', 'HEAD', '--', *SOURCE_DIRS, ':(exclude)web/data')
    untracked = [p for p in _git('ls-files', '--others', '--exclude-standard', '--', 'flatout').split('\n') if p.endswith('.py')]
    for p in sorted(untracked):                # new source files count as part of the dirty state
        diff += f'\n#untracked {p}\n' + (ROOT / p).read_text(encoding='utf-8', errors='replace')
    return dict(git_rev=rev, branch=_git('branch', '--show-current').strip() or None,
                dirty=bool(diff.strip()), dirty_diff_sha256=hashlib.sha256(diff.encode()).hexdigest() if diff.strip() else None)


def data_identity(store=None):
    """Hash of the parquet store listing (path + size + mtime-independent content hash of meta files).
    Hashing every parquet byte is slow; path+size is a fast, sufficient fingerprint for sync changes."""
    store = Path(store or ROOT / 'data' / 'store')
    items = sorted((str(p.relative_to(store)).replace('\\', '/'), p.stat().st_size)
                   for p in store.rglob('*') if p.is_file())
    return dict(store_files=len(items), store_sha256=sha256_json(items))


def dirty_outputs():
    """Generated artifacts that differ from HEAD or are untracked (the user's uncommitted run outputs)."""
    out = []
    for line in _git('status', '--porcelain', '--untracked-files=all').splitlines():
        path = line[3:].strip().strip('"')
        if ' -> ' in path:
            path = path.split(' -> ', 1)[1]
        if path.startswith(OUTPUT_PREFIXES):
            out.append((line[:2].strip(), path))
    return out


def snapshot(label, paths=None, log=print):
    """Copy generated artifacts into artifacts/snapshots/<utc>_<label>/ with a sha256 manifest."""
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest = SNAPSHOTS / f'{stamp}_{label}'
    dest.mkdir(parents=True, exist_ok=False)
    entries = paths if paths is not None else dirty_outputs()
    files = []
    for status, rel in entries:
        src = ROOT / rel
        if not src.is_file():
            continue
        tgt = dest / 'files' / rel
        tgt.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, tgt)
        files.append(dict(path=rel, git_status=status, bytes=src.stat().st_size, sha256=sha256_file(src)))
    manifest = dict(label=label, created_utc=stamp, code=code_identity(), data=data_identity(),
                    n_files=len(files), files=files)
    (dest / 'MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    log(f'  snapshot {dest.relative_to(ROOT)}: {len(files)} files, '
        f'{sum(f["bytes"] for f in files) / 2 ** 20:.1f} MB')
    return dest
