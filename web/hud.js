// f1.h HUD layer (igloo.inc-inspired): decoding text, out-of-focus depth type,
// and a viewport frame. Pure DOM + Web Animations; no dependencies. Hooks into app.js through `view:render`.

const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const GLYPHS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_/#%+-.";

/* ---------------------------------------------------------------- text decode */
export function scramble(el, { duration = 700, delay = 0 } = {}) {
  const target = el.dataset.text ?? el.textContent;
  el.dataset.text = target;
  if (REDUCED) { el.textContent = target; return; }
  clearInterval(el._scr);
  const t0 = performance.now() + delay;
  el._scr = setInterval(() => {
    const k = (performance.now() - t0) / duration;
    if (k < 0) { el.textContent = target.replace(/\S/g, " "); return; }
    if (k >= 1) { el.textContent = target; clearInterval(el._scr); return; }
    const done = Math.floor(k * target.length);
    el.textContent = [...target].map((c, i) => (i < done || c === " " ? c : i < done + 6 ? GLYPHS[(Math.random() * GLYPHS.length) | 0] : " ")).join("");
  }, 33);
}

/* ---------------------------------------------------------------- depth type behind section heads */
function addDepth(root) {
  root.querySelectorAll(".section-head").forEach((sh) => {
    if (sh.querySelector(".depth")) return;
    const t = sh.querySelector("h1")?.textContent?.trim();
    if (!t) return;
    const d = document.createElement("div");
    d.className = "depth"; d.setAttribute("aria-hidden", "true");
    d.innerHTML = `<span>${t.toUpperCase().replace(/\s+/g, "_").replace(/[<>&]/g, "")}</span>`;
    sh.prepend(d);
  });
}

/* ---------------------------------------------------------------- boot */
function boot() {
  const frame = document.createElement("div");
  frame.className = "hud-frame"; frame.setAttribute("aria-hidden", "true"); frame.innerHTML = "<i></i><i></i>";
  document.body.append(frame);
  // tabs decode on hover
  document.querySelectorAll(".tabs a").forEach((a) => a.addEventListener("pointerenter", () => { a.style.minWidth = `${a.offsetWidth}px`; scramble(a, { duration: 380 }); }));
}
document.addEventListener("view:render", (e) => {
  const root = e.detail?.root || document;
  addDepth(root);
});
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
// the first render can fire before this module loads
const first = document.querySelector(".view.active") || document;
addDepth(first);
