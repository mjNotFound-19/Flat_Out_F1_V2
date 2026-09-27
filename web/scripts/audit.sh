#!/usr/bin/env bash
# Automated UI audit with Playwright CLI: every tab, fan + nerd, desktop + phone.
# Scrolls each page like a user, then reports layout/asset problems as JSON lines.
# usage: bash web/scripts/audit.sh [base_url]   (default http://localhost:4173/)
BASE=${1:-http://localhost:4173/}
CHECK='async () => {
  for (let y = 0; y < document.body.scrollHeight; y += 450) { scrollTo(0, y); await new Promise(r => setTimeout(r, 140)); }
  await new Promise(r => setTimeout(r, 1800)); scrollTo(0, 0); await new Promise(r => setTimeout(r, 300));
  const vis = (e) => { const r = e.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const hiddenBy = (e) => { for (let p = e; p; p = p.parentElement) { const s = getComputedStyle(p); if (s.display === "none" || s.visibility === "hidden") return "hidden"; if (+s.opacity < 0.05) return p; } return null; };
  const ignore = ".fl-leaf, .c3-turn, .c3-fallback, .lights-intro, .sr-only, .skip, .drawer, .scrim, .tip, .cam-noise, [aria-hidden=true] .depth, .depth";
  const stuck = [...document.querySelectorAll("#app *")].filter(e => vis(e) && e.childElementCount === 0 && e.textContent.trim().length > 1 && !e.closest(ignore))
    .filter(e => { const h = hiddenBy(e); return h && h !== "hidden"; }).slice(0, 5).map(e => e.tagName + "." + e.className + ": " + e.textContent.trim().slice(0, 30));
  const broken = [...document.images].filter(i => i.complete && i.naturalWidth === 0 && vis(i)).map(i => i.src.slice(-60));
  const clipped = [...document.querySelectorAll("#app *")].filter(e => { const s = getComputedStyle(e); return vis(e) && !e.closest(".sr-only") && e.childElementCount === 0 && e.scrollWidth > e.clientWidth + 2 && s.overflow !== "visible" && s.textOverflow !== "ellipsis" && e.textContent.trim(); })
    .slice(0, 5).map(e => e.tagName + "." + e.className + ": " + e.textContent.trim().slice(0, 30));
  const small = innerWidth < 500 ? [...document.querySelectorAll("#app button, #app a, #app [role=button], header button, header a")].filter(e => { const r = e.getBoundingClientRect(); return r.width > 0 && (r.height < 32 || r.width < 32) && !e.closest(".tabs"); })
    .slice(0, 5).map(e => e.tagName + "." + e.className + " " + Math.round(e.getBoundingClientRect().width) + "x" + Math.round(e.getBoundingClientRect().height)) : [];
  return JSON.stringify({ overflowX: document.documentElement.scrollWidth - innerWidth, stuck, broken, clipped, small });
}'
playwright-cli goto "$BASE" >/dev/null
for size in "1440 900 desk" "390 844 phone"; do
  set -- $size
  playwright-cli resize "$1" "$2" >/dev/null
  for mode in fan nerd; do
    playwright-cli localstorage-set flatout-mode "$mode" >/dev/null
    for tab in race strategy drivers dvc teams season accuracy lab; do
      [ "$mode" = fan ] && [ "$tab" = lab ] && continue
      playwright-cli goto "$BASE?t=$RANDOM#$tab" >/dev/null
      playwright-cli eval "() => new Promise(r => setTimeout(r, 2500))" >/dev/null
      playwright-cli eval "() => document.querySelector('.lights-intro')?.click()" >/dev/null
      out=$(playwright-cli eval "$CHECK" | sed -n 2p)
      errs=$(playwright-cli console error 2>/dev/null | grep -o "Errors: [0-9]*")
      echo "$3 $mode $tab $errs $out"
    done
  done
done
