#!/usr/bin/env bash
# Publish the f1.h site to manasjha.online/f1/ (served from the portfolio's GitHub Pages deploy).
#   bash web/scripts/publish-portfolio.sh           copy files + build + deploy
#   bash web/scripts/publish-portfolio.sh --no-deploy   copy files only (to preview with the portfolio dev server)
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"                     # .../flat_out_f1_v2/web
PORTFOLIO="${PORTFOLIO:-$HERE/../../manas-portfolio-final}"
DEST="$PORTFOLIO/public/f1"

[ -f "$PORTFOLIO/package.json" ] || { echo "portfolio not found at $PORTFOLIO (set PORTFOLIO=...)"; exit 1; }
[ -f "$HERE/data/v3/site.json" ] || { echo "no site data: run python -m flatout export first"; exit 1; }

rm -rf "${DEST:?}"
mkdir -p "$DEST/data/v3"
cp "$HERE/index.html" "$HERE"/*.js "$HERE/styles.css" "$DEST/"
cp -r "$HERE/assets" "$DEST/"
cp "$HERE/data/v3/site.json" "$DEST/data/v3/"
cp -r "$HERE/legacy" "$DEST/"
cp "$HERE"/data/v2_*.csv "$DEST/data/" 2>/dev/null || true
echo "copied site -> $DEST ($(du -sh "$DEST" | cut -f1))"

if [ "${1:-}" != "--no-deploy" ]; then
  cd "$PORTFOLIO"
  npm run deploy
  echo "deployed: https://manasjha.online/f1/"
fi
