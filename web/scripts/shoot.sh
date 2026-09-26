#!/usr/bin/env bash
# Visual regression shots with Playwright CLI: every tab, fan + nerd, desktop + phone.
# usage: bash web/scripts/shoot.sh <label> [tabs...]   (site must be running on :4173)
set -e
LABEL=${1:-shot}; shift || true
TABS=${@:-race strategy drivers dvc teams season accuracy lab}
OUT="data/derived/_shots/$LABEL"
mkdir -p "$OUT"
playwright-cli goto "http://localhost:4173/" >/dev/null
for size in "1440 900 desk" "390 844 phone"; do
  set -- $size
  playwright-cli resize "$1" "$2" >/dev/null
  for mode in fan nerd; do
    playwright-cli localstorage-set flatout-mode "$mode" >/dev/null
    for tab in $TABS; do
      [ "$mode" = fan ] && [ "$tab" = lab ] && continue
      playwright-cli goto "http://localhost:4173/?t=$RANDOM#$tab" >/dev/null
      playwright-cli eval "() => new Promise(r => setTimeout(r, 2200))" >/dev/null
      playwright-cli eval "async () => { for (let y = 0; y < document.body.scrollHeight; y += 500) { scrollTo(0, y); await new Promise(r => setTimeout(r, 120)); } await new Promise(r => setTimeout(r, 1800)); scrollTo(0, 0); await new Promise(r => setTimeout(r, 400)); }" >/dev/null
      playwright-cli eval "() => new Promise(r => setTimeout(r, 900))" >/dev/null
      playwright-cli screenshot --full-page --filename "$OUT/$3-$mode-$tab.png" >/dev/null
      echo "  $OUT/$3-$mode-$tab.png"
    done
  done
done
