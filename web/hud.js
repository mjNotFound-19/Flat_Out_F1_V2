// f1.h HUD layer (igloo.inc-inspired): decoding text, drawn leader lines, out-of-focus depth type,
// a viewport frame, and optional UI sound. Pure DOM + Web Animations + Web Audio; no dependencies.
// Hooks into app.js through the `view:render` event and the window.f1Sound(kind) callback.

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

/* ---------------------------------------------------------------- hero annotations */
function drawHud(hud) {
  if (hud.dataset.drawn) return;
  hud.dataset.drawn = "1";
  const bits = [...hud.querySelectorAll("[data-scramble]")];
  if (REDUCED) return;
  // leader lines wipe in (dash-draw is unreliable with non-scaling strokes on a stretched viewBox)
  hud.querySelector(".hud-svg")?.animate([{ clipPath: "inset(0 100% 0 0)" }, { clipPath: "inset(0 0 0 0)" }], { duration: 1100, delay: 600, easing: "cubic-bezier(.2,.7,.2,1)", fill: "both" });
  hud.querySelectorAll(".hud-pt").forEach((pt, i) => pt.animate([{ opacity: 0, transform: "scale(.4)" }, { opacity: 1, transform: "none" }], { duration: 400, delay: 1100 + i * 120, fill: "both" }));
  bits.forEach((b, i) => scramble(b, { duration: 650, delay: 900 + i * 90 }));
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

/* ---------------------------------------------------------------- sound (off by default) */
const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
let soundOn = store.get("f1h-sound") === "on";
let ac = null, last = {};
function ctx() { if (!ac) ac = new (window.AudioContext || window.webkitAudioContext)(); if (ac.state === "suspended") ac.resume(); return ac; }
function noise(a, dur, gain, freq) {
  const buf = a.createBuffer(1, Math.ceil(a.sampleRate * dur), a.sampleRate), ch = buf.getChannelData(0);
  for (let i = 0; i < ch.length; i++) ch[i] = (Math.random() * 2 - 1) * (1 - i / ch.length) ** 2;
  const src = a.createBufferSource(), f = a.createBiquadFilter(), g = a.createGain();
  src.buffer = buf; f.type = "bandpass"; f.frequency.value = freq; f.Q.value = 0.9; g.gain.value = gain;
  src.connect(f).connect(g).connect(a.destination); src.start();
}
function tone(a, freq, dur, gain) {
  const o = a.createOscillator(), g = a.createGain(), t = a.currentTime;
  o.type = "triangle"; o.frequency.setValueAtTime(freq, t); o.frequency.exponentialRampToValueAtTime(freq * 0.6, t + dur);
  g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.0001, t + dur);
  o.connect(g).connect(a.destination); o.start(t); o.stop(t + dur);
}
window.f1Sound = (kind) => {
  if (!soundOn || document.hidden) return;
  const now = performance.now();
  if (now - (last[kind] || 0) < 70) return;   // the clock flips several digits at once: one click is enough
  last[kind] = now;
  const a = ctx();
  if (kind === "flap") noise(a, 0.045, 0.35, 2400);
  else if (kind === "static") noise(a, 0.22, 0.18, 1600);
  else tone(a, 1100, 0.06, 0.05);
};
function soundButton() {
  const b = document.createElement("button");
  b.type = "button"; b.className = "sound-btn";
  const paint = () => { b.setAttribute("aria-pressed", String(soundOn)); b.innerHTML = `<span class="bars" aria-hidden="true"><i></i><i></i><i></i></span>Sound: ${soundOn ? "On" : "Off"}`; };
  b.addEventListener("click", () => { soundOn = !soundOn; store.set("f1h-sound", soundOn ? "on" : "off"); paint(); if (soundOn) window.f1Sound("tick"); });
  paint();
  document.body.append(b);
}

/* ---------------------------------------------------------------- boot */
function boot() {
  const frame = document.createElement("div");
  frame.className = "hud-frame"; frame.setAttribute("aria-hidden", "true"); frame.innerHTML = "<i></i><i></i>";
  document.body.append(frame);
  soundButton();
  // tabs decode on hover, UI ticks on navigation
  document.querySelectorAll(".tabs a").forEach((a) => a.addEventListener("pointerenter", () => { a.style.minWidth = `${a.offsetWidth}px`; scramble(a, { duration: 380 }); }));
  document.addEventListener("click", (e) => { if (e.target.closest(".tabs a, .seg button, .mode button")) window.f1Sound("tick"); });
}
document.addEventListener("view:render", (e) => {
  const root = e.detail?.root || document;
  addDepth(root);
  root.querySelectorAll(".hud").forEach(drawHud);
});
if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", boot); else boot();
// the first render can fire before this module loads
const first = document.querySelector(".view.active") || document;
addDepth(first);
first.querySelectorAll(".hud").forEach(drawHud);
