// f1.h · race intelligence. Reads web/data/v3/site.json (written by `python -m flatout export`).

/* ------------------------------------------------------------------ reference data */
let TEAM = {   // replaced by official colours from the data bundle
  "Mercedes": "#00D7B6", "Red Bull Racing": "#4781D7", "Ferrari": "#ED1131", "McLaren": "#F47600",
  "Aston Martin": "#229971", "Alpine": "#00A1E8", "Williams": "#1868DB", "Racing Bulls": "#6C98FF",
  "Haas F1 Team": "#9C9FA2", "Audi": "#F50537", "Cadillac": "#909090"
};
const SHORT = { "Red Bull Racing": "Red Bull", "Haas F1 Team": "Haas", "Racing Bulls": "Racing Bulls", "Aston Martin": "Aston Martin" };
const ISO = {
  "Australia": "au", "China": "cn", "Japan": "jp", "United States": "us", "USA": "us", "Canada": "ca", "Monaco": "mc",
  "Spain": "es", "Austria": "at", "United Kingdom": "gb", "Great Britain": "gb", "UK": "gb", "Belgium": "be", "Hungary": "hu",
  "Netherlands": "nl", "Italy": "it", "Azerbaijan": "az", "Bahrain": "bh", "Singapore": "sg", "Mexico": "mx", "Brazil": "br",
  "Qatar": "qa", "United Arab Emirates": "ae", "UAE": "ae", "Abu Dhabi": "ae", "Saudi Arabia": "sa", "Malaysia": "my",
  "Portugal": "pt", "France": "fr", "Germany": "de", "Russia": "ru", "Turkey": "tr", "Vietnam": "vn", "Argentina": "ar", "South Africa": "za"
};
const MODE_LABEL = { pre_weekend: "Pre-weekend forecast", post_fp: "After practice", post_sprint: "After the sprint", post_quali: "After qualifying" };
const COMP = { S: "SOFT", M: "MEDIUM", H: "HARD" };
const COMP_COLOR = { S: "#ef4444", M: "#facc15", H: "#f1f5f9", SOFT: "#ef4444", MEDIUM: "#facc15", HARD: "#f1f5f9" };
const ICON = {
  check: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 12l5 5 9-10" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  cross: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 6l12 12M18 6L6 18" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round"/></svg>',
  sims: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 18h16M6 15l4-5 3 3 5-7" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/></svg>',
  laps: '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="8" fill="none" stroke="currentColor" stroke-width="2"/><path d="M12 7v5l3 2" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"/></svg>',
  sc: '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3 15l2-5h14l2 5v3H3z M7 18v2M17 18v2" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round"/></svg>',
  cpu: '<svg viewBox="0 0 24 24" aria-hidden="true"><rect x="6" y="6" width="12" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" stroke="currentColor" stroke-width="2"/></svg>',
};

const state = { data: null, mode: "fan", tab: "race", sort: {}, accMode: "post_quali", raceIdx: null, dvcMode: "race", dvcDriver: null, teamSel: null };

/* ------------------------------------------------------------------ helpers */
const $ = (s, r = document) => r.querySelector(s);
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v);
  }
  for (const c of kids.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}
const teamColor = (t) => TEAM[t] || "#8b98a6";
const person = (d) => state.data?.people?.[d] || {};
const name = (d) => person(d).name || d;
const pct = (p, dp) => {
  if (p == null || isNaN(p)) return "-";
  if (p > 0 && p < 0.001) return "<0.1%";
  const v = p * 100;
  return v.toFixed(dp ?? (v < 10 ? 1 : 0)) + "%";
};
const fx = (v, d = 2) => (v == null || isNaN(v) ? "-" : Number(v).toFixed(d));
const sgn = (v, d = 2) => (v == null || isNaN(v) ? "-" : (v > 0 ? "+" : "") + Number(v).toFixed(d));
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const nerd = () => state.mode === "nerd";
const clickable = (fn) => ({ onclick: fn, tabindex: "0", role: "button", onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); fn(); } } });

/* image with a fallback chain: 2026 F1 photo -> older F1 headshot -> onFail() */
function chainImg(urls, attrs, onFail) {
  const list = urls.filter(Boolean);
  if (!list.length) { onFail && onFail(); return null; }
  const img = h("img", { ...attrs, src: list[0] });
  let i = 0;
  img.addEventListener("error", () => { i += 1; if (i < list.length) img.src = list[i]; else { img.remove(); onFail && onFail(); } });
  return img;
}
function avatar(code, size = 36, team) {
  const p = person(code), t = team || p.team;
  const box = h("span", { class: "avatar", style: `--team:${teamColor(t)};width:${size}px;height:${size}px` });
  const ini = h("span", { class: "ini", style: `font-size:${Math.round(size * 0.36)}px` }, code.slice(0, 3));
  const img = chainImg([p.photo, p.headshot], { alt: "", loading: "lazy", decoding: "async", width: size, height: size }, () => box.append(ini));
  if (img) box.append(img);
  return box;
}
function cutout(code, attrs = {}) {
  const p = person(code);
  return chainImg([p.cutout, p.headshot], { alt: attrs.alt ?? "", decoding: "async", ...attrs });
}
function teamLogo(team, size = 28) {
  const url = state.data?.team_logos?.[team];
  if (!url) return null;
  const img = h("img", { class: "tlogo", src: url, alt: `${team} logo`, loading: "lazy", style: `height:${size}px`, height: size });
  img.addEventListener("error", () => img.remove());
  return img;
}
function flag(country, cls = "") {
  const iso = ISO[country];
  if (!iso) return null;
  return h("img", { class: `flag ${cls}`, src: `https://flagcdn.com/w80/${iso}.png`, alt: `${country} flag`, loading: "lazy", width: 28, height: 19 });
}
function drvCell(code, team, sub, size = 34) {   // spans only: valid inside <button>
  return h("span", { class: "drv" }, avatar(code, size, team),
    h("span", { style: "min-width:0;display:block" }, h("span", { class: "code", style: "display:block" }, code), h("span", { class: "nm", style: "display:block" }, sub ?? name(code))));
}
function pbar(p, color, label) {
  const w = Math.max(0, Math.min(1, p || 0)) * 100;
  return h("span", { class: "pbar", style: `display:block;--c:${color || "var(--primary)"}` }, h("i", { style: `width:${w}%` }), h("span", {}, label ?? pct(p)));
}
const btn = (cls, onclick, style, ...kids) => h("button", { type: "button", class: cls, onclick, style }, ...kids);
function stagger(el) { el.classList.add("stagger"); [...el.children].forEach((c, i) => c.style.setProperty("--i", i)); return el; }
const em = (t) => h("em", {}, t);
function tyres(seq) {
  if (!seq) return h("span", { class: "muted" }, "-");
  const parts = seq.split("-");
  const out = h("span", { class: "tyres", "data-tip": parts.map((c) => COMP[c] || c).join(" → "), "aria-label": parts.map((c) => COMP[c] || c).join(" then ") });
  parts.forEach((c, i) => { if (i) out.append(h("span", { class: "arrow", "aria-hidden": "true" }, "›")); out.append(h("span", { class: `tyre ${c}`, "aria-hidden": "true" }, c)); });
  return out;
}
function panel(title, sub, ...kids) {
  return h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("h2", {}, title, sub ? h("small", {}, sub) : null)), ...kids);
}
function stat(v, label, desc, cls = "") {
  return h("div", { class: `stat ${cls}` }, h("div", { class: "v", html: v }), h("div", { class: "l" }, label), desc ? h("div", { class: "d" }, desc) : null);
}
function sectionHead(title, text, right) {   // no eyebrow: the hero owns the only one (taste: eyebrow restraint)
  return h("div", { class: "section-head" }, h("div", {}, h("h1", {}, title), text ? h("p", {}, text) : null), right || null);
}
function countUp(el, to, fmt, ms = 900) {
  el.textContent = fmt(to);
  if (matchMedia("(prefers-reduced-motion: reduce)").matches || document.visibilityState !== "visible") return;
  const t0 = performance.now();
  const step = (t) => { const k = Math.min(1, (t - t0) / ms), e = 1 - Math.pow(1 - k, 3); el.textContent = fmt(to * e); if (k < 1) requestAnimationFrame(step); };
  requestAnimationFrame(step);
}

function table(rows, cols, { key, onRow, initial } = {}) {
  const sk = (key && state.sort[key]) || initial || null;
  let data = rows.slice();
  if (sk) {
    const c = cols.find((c) => c.id === sk.id);
    if (c) data.sort((a, b) => {
      const va = (c.sort || c.val)(a), vb = (c.sort || c.val)(b);
      const na = va == null || Number.isNaN(va), nb = vb == null || Number.isNaN(vb);
      if (na && nb) return 0; if (na) return 1; if (nb) return -1;
      return (va > vb ? 1 : va < vb ? -1 : 0) * sk.dir;
    });
  }
  const vis = cols.filter((c) => !c.nerd || nerd());
  const thead = h("tr", {}, vis.map((c) => h("th", {
    class: [c.num ? "num" : "", key && c.val ? "sortable" : "", sk && sk.id === c.id ? "sorted" : ""].join(" "), scope: "col",
    "data-tip": c.tip || null, tabindex: key && c.val ? "0" : null,
    "aria-sort": sk && sk.id === c.id ? (sk.dir > 0 ? "ascending" : "descending") : null,
    onclick: key && c.val ? () => { const cur = state.sort[key]; state.sort[key] = { id: c.id, dir: cur && cur.id === c.id ? -cur.dir : (c.desc ? -1 : 1) }; rerender(); } : null,
  }, c.label, sk && sk.id === c.id ? (sk.dir > 0 ? " ▲" : " ▼") : "")));
  const tbody = h("tbody", {}, data.map((r) => h("tr", onRow ? { class: "click", ...clickable(() => onRow(r)) } : {},
    vis.map((c) => h("td", { class: c.num ? "num mono" : "" }, c.render ? c.render(r) : c.val(r))))));
  return h("div", { class: "table-wrap" }, h("table", {}, h("thead", {}, thead), tbody));
}

/* ------------------------------------------------------------------ svg charts */
const svgWrap = (w, ht, inner, label = "chart") => `<svg class="chart" viewBox="0 0 ${w} ${ht}" preserveAspectRatio="xMidYMid meet" role="img" aria-label="${esc(label)}">${inner}</svg>`;
function lineChart({ series, labels, height = 240, width = 720, yFmt = (v) => fx(v, 3), yLabel = "", lowerBetter = false, invert = false, zero = false }) {
  const L = 64, R = series.some((x) => x.endLabel) ? 110 : 14, T = 14, B = 34, W = width - L - R, H = height - T - B;
  const all = series.flatMap((s) => s.values.filter((v) => v != null));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
  const pad = (hi - lo) * 0.12 || 0.01; lo -= pad; hi += pad;
  const x = (i) => L + (labels.length === 1 ? W / 2 : (i * W) / (labels.length - 1));
  const y = (v) => (invert ? T + ((v - lo) / (hi - lo)) * H : T + H - ((v - lo) / (hi - lo)) * H);
  let g = "";
  for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4; g += `<line class="gridl" x1="${L}" x2="${L + W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${yFmt(v)}</text>`; }
  if (zero) g += `<line class="axis" x1="${L}" x2="${L + W}" y1="${y(0)}" y2="${y(0)}"/>`;
  labels.forEach((l, i) => { if (labels.length <= 16 || i % 2 === 0) g += `<text x="${x(i)}" y="${height - 10}" text-anchor="middle">${esc(l)}</text>`; });
  if (yLabel) g += `<text x="12" y="${T + H / 2}" transform="rotate(-90 12 ${T + H / 2})" text-anchor="middle">${esc(yLabel)}${lowerBetter ? " (lower = better)" : ""}</text>`;
  const ends = [];
  for (const s of series) {
    const pts = s.values.map((v, i) => (v == null ? null : [x(i), y(v), v, i])).filter(Boolean);
    const d = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    g += `<path d="${d}" fill="none" stroke="${s.color}" stroke-width="${s.width || 2.2}" stroke-linejoin="round" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""} opacity="${s.opacity || 1}"/>`;
    for (const p of pts) g += `<circle cx="${p[0]}" cy="${p[1]}" r="${s.r || 3.6}" fill="${s.hollow ? "var(--bg)" : s.color}" stroke="${s.color}" stroke-width="2" data-tip="${esc(s.name)} · ${esc(labels[p[3]])}: ${yFmt(p[2])}"/>`;
    if (s.endLabel && pts.length) { const p = pts[pts.length - 1]; ends.push({ x: p[0] + 8, y: p[1] + 4, t: s.endLabel, c: s.color }); }
  }
  ends.sort((a, b) => a.y - b.y).forEach((e, i, arr) => { if (i && e.y - arr[i - 1].y < 15) e.y = arr[i - 1].y + 15; });   // keep end labels from colliding
  for (const e of ends) g += `<text x="${e.x}" y="${e.y}" style="fill:${e.c};font-weight:600">${esc(e.t)}</text>`;
  return svgWrap(width, height, g, yLabel);
}
function scatter({ points, max, height = 380, width = 420, xLabel, yLabel }) {
  const L = 44, R = 12, T = 12, B = 40, W = width - L - R, H = height - T - B;
  const x = (v) => L + ((v - 1) / (max - 1)) * W, y = (v) => T + ((v - 1) / (max - 1)) * H;
  let g = `<line class="axis" x1="${x(1)}" y1="${y(1)}" x2="${x(max)}" y2="${y(max)}" stroke-dasharray="4 4"/>`;
  for (let v = 1; v <= max; v += Math.ceil(max / 6)) g += `<text x="${x(v)}" y="${height - 22}" text-anchor="middle">P${v}</text><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">P${v}</text><line class="gridl" x1="${L}" x2="${L + W}" y1="${y(v)}" y2="${y(v)}"/>`;
  g += `<text x="${L + W / 2}" y="${height - 4}" text-anchor="middle">${esc(xLabel)}</text><text x="12" y="${T + H / 2}" transform="rotate(-90 12 ${T + H / 2})" text-anchor="middle">${esc(yLabel)}</text>`;
  for (const p of points) {
    g += p.ring ? `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="7" fill="none" stroke="${p.color}" stroke-width="2" data-tip="${esc(p.tip)}"/>`
      : `<circle cx="${x(p.x)}" cy="${y(p.y)}" r="7" fill="${p.color}" stroke="#000" stroke-width=".8" data-tip="${esc(p.tip)}"/>`;
    g += `<text x="${x(p.x) + 10}" y="${y(p.y) + 4}" style="fill:var(--text-2);font-size:11px">${esc(p.label)}</text>`;
  }
  return svgWrap(width, height, g, "predicted vs actual finish");
}
function reliability(bins, color, title, width = 420, height = 340) {
  const L = 44, R = 10, T = 28, B = 38, W = width - L - R, H = height - T - B;
  const x = (v) => L + v * W, y = (v) => T + H - v * H;
  let g = `<text x="${L}" y="16" style="fill:var(--text);font-weight:600">${esc(title)}</text>`;
  for (let k = 0; k <= 4; k++) { const v = k / 4; g += `<line class="gridl" x1="${L}" x2="${L + W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 6}" y="${y(v) + 4}" text-anchor="end">${v * 100}%</text><text x="${x(v)}" y="${height - 20}" text-anchor="middle">${v * 100}%</text>`; }
  g += `<line class="axis" x1="${x(0)}" y1="${y(0)}" x2="${x(1)}" y2="${y(1)}" stroke-dasharray="4 4"/><text x="${L + W / 2}" y="${height - 3}" text-anchor="middle">predicted chance</text>`;
  const pts = (bins || []).filter((b) => b.n > 0), nmax = Math.max(...pts.map((b) => b.n), 1);
  g += `<path d="${pts.map((b, i) => `${i ? "L" : "M"}${x(b.predicted)},${y(b.observed)}`).join("")}" fill="none" stroke="${color}" stroke-width="2"/>`;
  for (const b of pts) g += `<circle cx="${x(b.predicted)}" cy="${y(b.observed)}" r="${3 + 9 * Math.sqrt(b.n / nmax)}" fill="${color}" fill-opacity=".35" stroke="${color}" data-tip="predicted ${pct(b.predicted)} → happened ${pct(b.observed)} (n=${b.n})"/>`;
  return svgWrap(width, height, g, `${title} calibration`);
}
function histogram(vals, { height = 150, width = 500, color = "var(--primary)" } = {}) {
  const L = 8, R = 8, T = 10, B = 24, W = width - L - R, H = height - T - B, n = vals.length, bw = W / n, mx = Math.max(...vals, 1e-9);
  let g = `<rect x="${L}" y="${T}" width="${bw * 3}" height="${H}" fill="rgba(251,191,36,.07)"/><rect x="${L + bw * 3}" y="${T}" width="${bw * 7}" height="${H}" fill="rgba(34,197,94,.05)"/>`;
  vals.forEach((v, i) => {
    const hh = (v / mx) * H;
    g += `<rect x="${L + i * bw + 1.5}" y="${T + H - hh}" width="${bw - 3}" height="${hh}" rx="2" fill="${color}" fill-opacity="${0.4 + 0.6 * (v / mx)}" data-tip="P${i + 1}: ${pct(v, 1)}"/>`;
    if (i === 0 || (i + 1) % 2 === 0) g += `<text x="${L + i * bw + bw / 2}" y="${height - 8}" text-anchor="middle">${i + 1}</text>`;
  });
  return svgWrap(width, height, g, "finishing position distribution");
}
function forest(rows, { key, se, unit = "", width = 640, rowH = 24 }) {
  const L = 70, R = 60, T = 18, B = 26, height = T + B + rows.length * rowH, W = width - L - R;
  const m = Math.max(...rows.flatMap((r) => [Math.abs(r[key] - 1.96 * (r[se] || 0)), Math.abs(r[key] + 1.96 * (r[se] || 0))])) * 1.05 || 1;
  const x = (v) => L + ((v + m) / (2 * m)) * W;
  let g = `<line class="axis" x1="${x(0)}" x2="${x(0)}" y1="${T - 6}" y2="${height - B + 4}"/>`;
  [-m, -m / 2, 0, m / 2, m].forEach((v) => { g += `<text x="${x(v)}" y="${height - 6}" text-anchor="middle">${sgn(v, 2)}</text>`; });
  rows.forEach((r, i) => {
    const yy = T + i * rowH + rowH / 2, c = teamColor(r.team);
    g += `<text x="${L - 10}" y="${yy + 4}" text-anchor="end" style="fill:var(--text);font-weight:600">${esc(r.Driver)}</text>`;
    g += `<line x1="${x(r[key] - 1.96 * (r[se] || 0))}" x2="${x(r[key] + 1.96 * (r[se] || 0))}" y1="${yy}" y2="${yy}" stroke="${c}" stroke-width="2" opacity=".6"/>`;
    g += `<circle cx="${x(r[key])}" cy="${yy}" r="5.5" fill="${c}" data-tip="${esc(r.Driver)}: ${sgn(r[key], 3)}${unit} ± ${fx(1.96 * (r[se] || 0), 3)} (95%)"/><text x="${width - R + 6}" y="${yy + 4}">${sgn(r[key], 2)}</text>`;
  });
  return svgWrap(width, height, g, "ratings with 95% intervals");
}
function degChart(c, width = 640, height = 260) {
  const L = 52, R = 14, T = 14, B = 34, W = width - L - R, H = height - T - B;
  const maxAge = Math.max(...Object.values(c.max_stint || { H: 45 })) + 5;
  const curve = (k, a) => { const over = Math.max(0, a - 0.9 * (c.max_stint?.[k] ?? 40)); return (c.offset?.[k] ?? 0) + (c.deg?.[k] ?? 0.05) * a + 0.04 * over * over; };
  const ks = ["SOFT", "MEDIUM", "HARD"], vals = ks.flatMap((k) => Array.from({ length: maxAge }, (_, a) => curve(k, a + 1)));
  const lo = Math.min(...vals), hi = Math.min(Math.max(...vals), lo + 5);
  const x = (a) => L + (a / maxAge) * W, y = (v) => T + H - ((Math.min(v, hi) - lo) / (hi - lo)) * H;
  let g = "";
  for (let k = 0; k <= 4; k++) { const v = lo + ((hi - lo) * k) / 4; g += `<line class="gridl" x1="${L}" x2="${L + W}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">+${fx(v - lo, 1)}s</text>`; }
  for (let a = 0; a <= maxAge; a += 10) g += `<text x="${x(a)}" y="${height - 14}" text-anchor="middle">${a}</text>`;
  g += `<text x="${L + W / 2}" y="${height - 1}" text-anchor="middle">tyre age (laps)</text>`;
  for (const k of ks) {
    g += `<path d="${Array.from({ length: maxAge }, (_, a) => `${a ? "L" : "M"}${x(a + 1).toFixed(1)},${y(curve(k, a + 1)).toFixed(1)}`).join("")}" fill="none" stroke="${COMP_COLOR[k]}" stroke-width="2.5"/>`;
    if (c.max_stint?.[k]) g += `<line x1="${x(c.max_stint[k])}" x2="${x(c.max_stint[k])}" y1="${T}" y2="${T + H}" stroke="${COMP_COLOR[k]}" stroke-dasharray="3 5" opacity=".5"/>`;
  }
  return svgWrap(width, height, g, "tyre degradation curves");
}
function radar(teams, axes, size = 420) {
  const cx = size / 2, cy = size / 2 + 6, r = size / 2 - 70, n = axes.length;
  const pt = (i, v) => { const a = -Math.PI / 2 + (2 * Math.PI * i) / n; return [cx + Math.cos(a) * r * v / 100, cy + Math.sin(a) * r * v / 100]; };
  let g = "";
  [25, 50, 75, 100].forEach((lv) => { g += `<polygon points="${axes.map((_, i) => pt(i, lv).join(",")).join(" ")}" fill="none" stroke="var(--border)" ${lv === 50 ? 'stroke-dasharray="4 4"' : ""}/>`; });
  axes.forEach(([, label], i) => {
    const [ex, ey] = pt(i, 100), [lx, ly] = pt(i, 122);
    g += `<line x1="${cx}" y1="${cy}" x2="${ex}" y2="${ey}" stroke="var(--border)"/><text x="${lx}" y="${ly + 4}" text-anchor="middle" style="fill:var(--text-2);font-size:12px">${esc(label)}</text>`;
  });
  const dashes = ["", "6 4", "2 3"];
  teams.forEach((t, j) => {
    const c = teamColor(t.Team);
    g += `<polygon points="${axes.map(([k], i) => pt(i, t[k] ?? 50).join(",")).join(" ")}" fill="${c}" fill-opacity=".14" stroke="${c}" stroke-width="2.4" stroke-dasharray="${dashes[j] || ""}"/>`;
    axes.forEach(([k, label], i) => { const [px, py] = pt(i, t[k] ?? 50); g += `<circle cx="${px}" cy="${py}" r="4" fill="${c}" data-tip="${esc(t.Team)} · ${esc(label)}: ${fx(t[k], 0)}/100"/>`; });
  });
  return svgWrap(size, size + 12, g, "constructor comparison radar");
}

/* ------------------------------------------------------------------ RACE */
function flagWave(country) {
  const iso = ISO[country];
  if (!iso) return null;
  const N = 28, url = `url(https://flagcdn.com/w1280/${iso}.png)`;
  const wrap = h("div", { class: "flag-wave", "aria-hidden": "true" });
  for (let k = 0; k < N; k++) wrap.append(h("i", { style: `--k:${k};--n:${N};--flag:${url}` }));
  return wrap;
}
function renderRace(root) {
  const nx = state.data.next;
  if (!nx) { root.append(h("div", { class: "empty" }, "No prediction yet. Run python -m flatout predict to create one.")); return; }
  const m = nx.meta, c = nx.circuit, D = nx.drivers;
  const byWin = D.slice().sort((a, b) => b.win - a.win);
  const date = m.date ? new Date(m.date + "T12:00:00") : null;

  const fav = byWin[0];
  const dateStr = date ? new Intl.DateTimeFormat(undefined, { day: "numeric", month: "long", year: "numeric" }).format(date) : String(m.year);
  const hero = h("div", { class: "hero" }, flagWave(m.country),
    h("div", { class: "hero-copy" },
      h("div", { class: "eyebrow" }, flag(m.country, "sm"), `Round ${m.round}, ${dateStr}`),
      titleEl(m.event),
      venueEl(m),
      h("div", { class: "chips" },
        h("span", { class: "chip red" }, MODE_LABEL[m.mode] || m.mode),
        h("span", { class: "chip", html: `${ICON.sims}${Number(m.n_sims).toLocaleString()} races simulated` }),
        h("span", { class: "chip", html: `${ICON.sc}Safety car ${pct(m.p_sc)}` }),
        nerd() ? h("span", { class: "chip", html: `${ICON.cpu}${fx(m.sim_seconds, 0)} s compute` }) : null,
        nerd() ? h("span", { class: "chip" }, `grid ${m.grid_known ? "known" : "simulated"}`) : null,
        m.status === "provisional" ? h("span", { class: "chip warn", "data-tip": "First race at this circuit in the model's data: circuit behaviour comes from pooled priors" }, "Provisional") : null),
      provisionalEl(m),
      h("div", { class: "hero-meta" }, countdownEl(m), trackCard(nx)),
      h("p", { class: "hero-note" }, nerd()
        ? `Lap-by-lap Monte Carlo: Student-t race-day pace, tyre deg with cliff, planned and safety-car stops, red-flag tyre changes, pace-dependent passing (overtake factor ${fx(c.overtake_factor, 2)}) and reliability hazards.`
        : `${Number(m.n_sims).toLocaleString()} simulated races, lap by lap, with tyres, pit stops, safety cars and breakdowns. This is what happened most often.`)),
    h("div", { class: "depth", "aria-hidden": "true" },
      h("span", {}, `${Number(m.n_sims).toLocaleString()} RACES`), h("span", {}, `RND_${String(m.round).padStart(2, "0")} // ${m.event.toUpperCase()}`), h("span", {}, `P(SC) ${pct(m.p_sc)}`)),
    h("div", { class: "hero-portrait" },
      cutout(fav.Driver, { loading: "eager", fetchpriority: "high", alt: `${name(fav.Driver)}, race favourite`, width: 480, height: 480 }),
      h("button", { type: "button", class: "tag", onclick: () => openDriver(fav.Driver), "aria-label": `Open ${name(fav.Driver)}` }, h("b", {}, pct(fav.win)), h("span", {}, `${name(fav.Driver)} to win`))));
  const marquee = marqueeEl(byWin.slice(0, 8));
  const podium = h("div", { class: "podium" }, [byWin[1], byWin[0], byWin[2]].map((d, i) => {
    const place = [2, 1, 3][i], p = person(d.Driver), big = h("span", {}, "0");
    const shot = h("div", { class: "shot" }, h("div", { class: "num", "aria-hidden": "true" }, place),
      cutout(d.Driver, { loading: "eager", width: 480, height: 480 }) || h("div", { class: "ini" }, d.Driver),
      h("div", { class: "who" }, h("div", { class: "code" }, d.Driver), h("div", { class: "name" }, `${name(d.Driver)} · ${d.Team}`)));
    countUp(big, d.win * 100, (v) => v.toFixed(1));
    return h("button", { class: `pod p${place}`, style: `--team:${teamColor(d.Team)}`, onclick: () => openDriver(d.Driver), "aria-label": `${name(d.Driver)}, ${pct(d.win)} to win` },
      shot, h("div", { class: "body" }, h("div", { class: "big" }, big, h("small", {}, "%")), h("div", { class: "lbl" }, "chance to win"),
        h("div", { class: "mini" }, h("span", {}, "Podium ", h("b", {}, pct(d.podium))), h("span", {}, "Avg finish ", h("b", {}, "P" + fx(d.exp_pos, 1))), nerd() ? h("span", {}, "E[pts] ", h("b", {}, fx(d.exp_pts, 1))) : null)));
  }));

  const sc = 1 / Math.max(...byWin.map((x) => x.podium));
  const winList = stagger(h("div", { class: "stack", style: "gap:2px" }, (nerd() ? byWin : byWin.slice(0, 10)).map((d) => {
    const row = btn("rowbtn", () => openDriver(d.Driver), "grid-template-columns:150px 1fr;gap:12px;align-items:center;min-height:44px", drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team, 32));
    if (nerd()) {
      const dd = nx.dist[d.Driver] || [];
      row.style.gridTemplateColumns = "150px 1fr 112px";
      row.append(h("div", { class: "pbar", "data-tip": `P1 ${pct(dd[0])} · P2 ${pct(dd[1])} · P3 ${pct(dd[2])}` },
        h("i", { style: `width:${dd[0] * sc * 100}%;background:var(--gold);border-radius:6px 0 0 6px` }),
        h("i", { style: `left:${dd[0] * sc * 100}%;width:${dd[1] * sc * 100}%;background:var(--silver);border-radius:0` }),
        h("i", { style: `left:${(dd[0] + dd[1]) * sc * 100}%;width:${dd[2] * sc * 100}%;background:var(--bronze);border-radius:0 6px 6px 0` })),
        h("span", { class: "mono", style: "text-align:right" }, pct(d.win), h("span", { class: "muted" }, ` / ${pct(d.podium)}`)));
    } else row.append(pbar(d.win / byWin[0].win, teamColor(d.Team), pct(d.win)));
    return row;
  })));
  const winPanel = panel(nerd() ? "Top-3 split" : "Who wins?", nerd() ? "P1 / P2 / P3 share · win / podium" : "chance of winning", winList,
    nerd() ? h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--gold)" }), "P1"), h("span", {}, h("i", { style: "background:var(--silver)" }), "P2"), h("span", {}, h("i", { style: "background:var(--bronze)" }), "P3")) : null);

  const cols = [
    { id: "rank", label: "#", val: (d) => d.rank, render: (d) => h("span", { class: "pos" }, d.rank) },
    { id: "drv", label: "Driver", val: (d) => d.Driver, render: (d) => drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team) },
    { id: "grid", label: "Grid", num: true, nerd: true, val: (d) => d.grid, render: (d) => (m.grid_known ? "P" + Math.round(d.grid) : "~" + fx(d.grid, 1)), tip: m.grid_known ? "starting slot" : "expected grid slot (qualifying simulated)" },
    { id: "exp", label: nerd() ? "E[pos]" : "Avg finish", num: true, val: (d) => d.exp_pos, render: (d) => "P" + fx(d.exp_pos, nerd() ? 2 : 1) },
    { id: "range", label: "P10-P90", nerd: true, val: (d) => d.p90 - d.p10, render: (d) => rangeBar(d, D.length), tip: "80% of simulated finishes fall in the bar; tick = median, ring = mean" },
    { id: "win", label: "Win", desc: true, val: (d) => d.win, render: (d) => pbar(d.win, teamColor(d.Team)) },
    { id: "pod", label: "Podium", desc: true, val: (d) => d.podium, render: (d) => pbar(d.podium, "var(--gold)") },
    { id: "pts", label: "Points", desc: true, val: (d) => d.points, render: (d) => pbar(d.points, "var(--good)"), tip: "chance of a top-10 finish" },
    { id: "dnf", label: "DNF", num: true, desc: true, nerd: true, val: (d) => d.dnf, render: (d) => pct(d.dnf) },
    { id: "ept", label: "E[pts]", num: true, desc: true, nerd: true, val: (d) => d.exp_pts, render: (d) => fx(d.exp_pts, 2) },
    { id: "pace", label: "Pace Δ%", num: true, nerd: true, val: (d) => d.pace_pct, render: (d) => `${sgn(d.pace_pct, 2)} ±${fx(d.pace_sd, 2)}`, tip: "predicted race pace vs field median (% of lap) ± model σ" },
    { id: "stops", label: "Stops 1/2/3", nerd: true, val: (d) => d.exp_stops, render: (d) => stopBar(d) },
    { id: "strat", label: "Likely strategy", val: (d) => d.strategy, render: (d) => h("span", {}, tyres(d.strategy), h("span", { class: "strat-txt" }, nerd() ? `L${fx(d.first_stop_p50, 0)}` : `${d.modal_stops ?? "?"}-stop · pit ~L${fx(d.first_stop_p50, 0)}`)) },
  ];
  root.append(hero, marquee, consoleEl(nx),
    h("div", { class: "grid g-main" }, h("div", { class: "stack" }, podium, quickFacts(nx)), winPanel),
    circuit3dEl(nx, fav),
    h("div", { class: "mt" }, panel(nerd() ? "Simulated classification" : "The grid, predicted", nerd() ? "sortable · click a row for the driver file" : "tap a driver for details", table(D, cols, { key: "race", onRow: (d) => openDriver(d.Driver) }))));
  if (nerd()) root.append(h("div", { class: "mt" }, heatmapPanel(nx)));
}
function trackCard(nx) {
  const t = nx.track, c = nx.circuit, id = nx.meta.identity || {};
  if (!t) return h("div", { class: "trackcard nomap" },
    h("div", { class: "map nomap-box", "aria-hidden": "true" }, h("span", {}, "no map")),
    h("div", {}, h("div", { class: "k" }, "Circuit"), h("div", { class: "n" }, id.circuit_name || nx.meta.location),
      h("div", { class: "meta" }, h("span", {}, `${c.n_laps} laps`), id.length_km ? h("span", {}, `${fx(id.length_km, 3)} km`) : null,
        h("span", {}, "no recorded lap in 2024–26 data"))));
  const d = "M" + t.points.map((p) => p.join(",")).join(" L") + " Z";
  const pad = 60, vb = `${-pad} ${-pad} ${t.w + 2 * pad} ${t.h + 2 * pad}`;
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const svg = `<svg viewBox="${vb}" role="img" aria-label="${esc(nx.meta.location)} circuit map">
    <path class="line-bg" d="${d}"/><path class="line" pathLength="1" d="${d}" id="trk"/>
    ${nerd() ? t.corners.map((k) => `<text class="corner" x="${k.x + 18}" y="${k.y - 18}">${k.n}</text>`).join("") : ""}
    <circle r="16" fill="var(--primary)" stroke="#fff" stroke-width="5">${still ? "" : `<animateMotion dur="7s" repeatCount="indefinite" begin="2.6s"><mpath href="#trk"/></animateMotion>`}</circle></svg>`;
  return h("div", { class: "trackcard" }, h("div", { class: "map", html: svg }),
    h("div", {}, h("div", { class: "k" }, "Circuit"), h("div", { class: "n" }, nx.meta.location),
      h("div", { class: "meta" }, h("span", {}, `${c.n_laps} laps`), h("span", {}, `${t.corners.length} turns`), h("span", {}, `pit loss ${fx(c.pit_loss, 1)}s`))));
}
/* 3D circuit section: the reference lap as a speed-coloured ribbon (three.js, loaded when scrolled near) */
function circuit3dEl(nx, fav) {
  const t = nx.track, c = nx.circuit, m = nx.meta;
  if (!t || !t.speed) {
    const id = m.identity || {};
    return h("section", { class: "c3 c3-none", "aria-labelledby": "c3-title" },
      h("div", { class: "c3-hud" },
        h("div", { class: "c3-tl" }, h("span", {}, `CIRCUIT_${String(m.round).padStart(2, "0")}`), h("h2", { id: "c3-title" }, id.circuit_name || m.location)),
        h("p", { class: "c3-none-msg" }, `There is no recorded ${id.circuit_name || m.location} lap in the 2024–26 telemetry this site uses, `
          + "so no 3D circuit is drawn. The last Formula 1 race here predates FastF1 position data (2018 onwards). "
          + "A borrowed layout from another circuit would be misleading, so none is shown.")));
  }
  if (state.c3dispose) { state.c3dispose(); state.c3dispose = null; }
  const vmin = Math.min(...t.speed), vmax = Math.max(...t.speed);
  const elev = (Math.max(...t.z) / 1000) * t.scale_m;
  const lapStr = `${Math.floor(t.lap_time / 60)}:${(t.lap_time % 60).toFixed(3).padStart(6, "0")}`;
  const speedV = h("b", {}, "---"), speedBar = h("i");
  const stats = [["Length", `${fx(t.length_m / 1000, 3)} km`], ["Turns", t.corners.length], ["Elevation Δ", `${fx(elev, 1)} m`], ["Top speed", `${vmax} km/h`], ["Laps", c.n_laps]];
  if (nerd()) stats.push(["Min speed", `${vmin} km/h`], ["Pit loss", `${fx(c.pit_loss, 1)} s`]);
  const fallback = h("div", { class: "c3-fallback" }, trackCard(nx)?.querySelector(".map") || null);
  const el = h("section", { class: "c3", "aria-labelledby": "c3-title", style: `--team:${teamColor(fav.Team)}` },
    fallback,
    h("div", { class: "c3-labels", "aria-hidden": "true" }),
    h("div", { class: "c3-hud" },
      h("div", { class: "c3-tl" }, h("span", { "data-scramble": "" }, `CIRCUIT_${String(m.round).padStart(2, "0")}`), h("h2", { id: "c3-title" }, m.location)),
      h("dl", { class: "c3-tr" }, stats.map(([k, v]) => h("div", {}, h("dt", {}, k), h("dd", {}, String(v))))),
      h("div", { class: "c3-bl" }, h("div", { class: "c3-speed", "aria-hidden": "true" }, h("span", {}, "Speed"), speedV, h("small", {}, "km/h"), h("div", { class: "c3-sbar" }, speedBar)),
        h("p", {}, nerd() ? `Reference: ${name(t.driver)}, fastest lap ${lapStr} (${t.source}). Ribbon = telemetry speed, height = elevation ×7.`
          : `${name(t.driver)}'s fastest lap from last year, coloured by speed. Hills are exaggerated 7× so you can see them.`)),
      h("div", { class: "c3-br", "aria-hidden": "true" }, h("div", { class: "c3-legend" }, h("span", {}, `${vmin}`), h("i"), h("span", {}, `${vmax} km/h`)), h("span", { class: "c3-hint" }, "Drag to rotate"))));
  const io = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    io.disconnect();
    import("./circuit3d.js").then((mod) => mod.mount(el, t, {
      accent: teamColor(fav.Team),
      onSpeed: (v, k) => { speedV.textContent = v; speedBar.style.transform = `scaleX(${Math.max(0.04, k)})`; const [r, g, b] = mod.speedColor(k); speedBar.style.background = `rgb(${r * 255 | 0},${g * 255 | 0},${b * 255 | 0})`; },
    })).then((dispose) => { el.classList.add("live"); state.c3dispose = dispose; })
      .catch((err) => { el.classList.add("flat"); console.info("3D circuit unavailable, showing the flat map:", err.message); });
  }, { rootMargin: "400px 0px" });
  io.observe(el);
  return el;
}
const reduced = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
/* LCD that types its lines; screen readers get the final text once via a polite live region */
function lcdPrint(con, lines) {
  const lcd = con.querySelector(".c-lcd"), live = con.querySelector(".c-live");
  clearInterval(con._typer);
  lcd.innerHTML = "";
  const rows = lines.map((l) => { const d = h("div", { class: `ln ${l.dim ? "dim" : ""}` }); lcd.append(d); return [d, l.t]; });
  const cursor = h("span", { class: "cursor" });
  live.textContent = lines.map((l) => l.t).join(". ");
  if (reduced()) { rows.forEach(([d, t]) => { d.textContent = t; }); rows[rows.length - 1][0].append(cursor); return; }
  let r = 0, c = 0;
  con._typer = setInterval(() => {
    if (r >= rows.length) { clearInterval(con._typer); return; }
    const [d, t] = rows[r];
    d.textContent = t.slice(0, ++c);
    d.append(cursor);
    if (c >= t.length) { r += 1; c = 0; }
  }, 14);
}
function consoleEl(nx) {
  const m = nx.meta, D = nx.drivers;
  const byWin = D.slice().sort((a, b) => b.win - a.win);
  const avg = (k) => D.reduce((a, d) => a + d[k], 0) / D.length;
  const angle = (v) => `${-135 + 270 * Math.max(0, Math.min(1, v))}deg`;
  const intro = [
    { t: "F1.H SIM ENGINE v3" , dim: true },
    { t: `${m.event.toUpperCase()}  ${Number(m.n_sims).toLocaleString()} RACES` },
    { t: `STATUS: ${(MODE_LABEL[m.mode] || m.mode).toUpperCase()}` },
    { t: "> PRESS A DRIVER KEY", dim: true },
  ];
  const readout = (d) => [
    { t: `> ${d.Driver}  ${name(d.Driver).toUpperCase()}`, dim: true },
    { t: `WIN ${pct(d.win)}   PODIUM ${pct(d.podium)}   POINTS ${pct(d.points)}` },
    { t: `AVG P${fx(d.exp_pos, 1)}   RANGE P${d.p10}-P${d.p90}   DNF ${pct(d.dnf)}` },
    { t: `PLAN ${d.strategy}   ${d.modal_stops}-STOP   PIT ~L${fx(d.first_stop_p50, 0)}` },
  ];
  const keys = h("div", { class: "c-cell c-keys", role: "group", "aria-label": "Driver keys" });
  const con = h("section", { class: "console", "aria-label": "Race engineer console" });
  const cam = camEl();
  let showDials = () => {};
  const select = (d) => {
    keys.querySelectorAll(".key").forEach((k) => k.setAttribute("aria-pressed", String(k.dataset.d === d.Driver)));
    cam.show(d);
    showDials(d);
    lcdPrint(con, readout(d));
  };
  const carNo = (d) => parseInt(person(d.Driver).number, 10) || 999;
  D.slice().sort((a, b) => carNo(a) - carNo(b)).forEach((d) => keys.append(h("button", { type: "button", class: "key", "data-d": d.Driver, style: `--team:${teamColor(d.Team)}`, "aria-pressed": "false",
    "aria-label": `${name(d.Driver)}, ${pct(d.win)} to win`, onclick: () => { stopAuto(); select(d); } }, h("i", { "aria-hidden": "true" }), h("em", { class: "kn", "aria-hidden": "true" }, person(d.Driver).number || ""), d.Driver, h("small", {}, pct(d.win)))));
  const auto = h("button", { type: "button", class: "c-btn", "aria-pressed": "false" }, h("span", { class: "led", "aria-hidden": "true" }), "Autoplay");
  let idx = 0;
  function stopAuto() { clearInterval(con._auto); auto.setAttribute("aria-pressed", "false"); }
  auto.addEventListener("click", () => {
    if (auto.getAttribute("aria-pressed") === "true") return stopAuto();
    auto.setAttribute("aria-pressed", "true");
    const step = () => { select(byWin[idx % 10]); idx += 1; };
    step(); con._auto = setInterval(() => { if (!document.contains(con)) return clearInterval(con._auto); step(); }, 4200);
  });
  const dial = () => {
    const knob = h("div", { class: "knob", "aria-hidden": "true" }), b = h("b"), span = h("span");
    const el = h("div", { class: "dial" }, knob, b, span);
    const set = ([v, value, label, tip]) => {
      const a = angle(v);
      if (window.gsap && knob.style.getPropertyValue("--a")) gsap.to(knob, { "--a": a, duration: 1.1, ease: "elastic.out(1, 0.6)", overwrite: true });
      else knob.style.setProperty("--a", a);
      b.textContent = value; span.textContent = label; el.dataset.tip = tip;
    };
    return { el, set };
  };
  const margin = byWin[0].win - byWin[1].win;
  const raceDials = [
    [m.p_sc, pct(m.p_sc), "Safety car", "chance of at least one safety car or red flag"],
    [avg("stop2") + avg("stop3p"), pct(avg("stop2") + avg("stop3p")), "Two stops+", "share of cars making two or more stops"],
    [Math.min(1, margin / 0.2), `+${fx(margin * 100, 1)}`, "Fav. margin", "favourite's win chance minus the next driver's, in points"],
  ];
  const driverDials = (d) => [
    [d.podium, pct(d.podium), "Podium", `${name(d.Driver)}: chance of a top-3 finish`],
    [d.points, pct(d.points), "Points", `${name(d.Driver)}: chance of finishing in the top 10`],
    [(D.length - d.exp_pos) / (D.length - 1), `P${fx(d.exp_pos, 1)}`, "Avg finish", `${name(d.Driver)}: average finishing position over all simulations`],
  ];
  const dials = [dial(), dial(), dial()];
  dials.forEach((x, i) => x.set(raceDials[i]));
  const dialBox = h("div", { class: "c-cell c-dials" }, dials.map((x) => x.el));
  showDials = (d) => { dialBox.style.setProperty("--knob", teamColor(d.Team)); dials.forEach((x, i) => x.set(driverDials(d)[i])); };
  con.append(h("div", { class: "console-grid" },
    cam.el,
    h("div", { class: "c-cell c-lcd", "aria-hidden": "true" }),
    dialBox,
    keys,
    h("div", { class: "c-foot" }, auto, h("span", { class: "c-brand" }, h("b", {}, "f1"), ".h race engineer"), h("span", {}, "keys = drivers by car number"))),
    h("p", { class: "sr-only c-live", "aria-live": "polite" }));
  lcdPrint(con, intro);
  // autoplay is the default: it starts once the console is on screen (after the intro has been read)
  const seen = new IntersectionObserver(([e]) => {
    if (!e.isIntersecting) return;
    seen.disconnect();
    setTimeout(() => { if (document.contains(con) && auto.getAttribute("aria-pressed") === "false" && !keys.querySelector('[aria-pressed="true"]')) auto.click(); }, 1800);
  }, { threshold: 0.4 });
  seen.observe(con);
  return con;
}
/* driver cam: a small CRT that cuts to whichever driver key is live (static burst between channels) */
function camEl() {
  const layer = h("div", { class: "cam-feed" });
  const tag = h("div", { class: "cam-tag" }, h("b", {}, "--"), h("span", {}, "no signal"));
  const num = h("div", { class: "cam-num", "aria-hidden": "true" });
  const el = h("div", { class: "c-cell c-cam idle", "aria-hidden": "true" },
    layer, num, h("div", { class: "cam-noise" }), h("div", { class: "cam-scan" }),
    h("div", { class: "cam-rec" }, h("i"), "CAM"), tag);
  let token = 0;
  const show = (d) => {
    const p = person(d.Driver), my = ++token;
    el.style.setProperty("--team", teamColor(d.Team));
    el.classList.remove("idle", "cut"); void el.offsetWidth; el.classList.add("cut");
    const img = cutout(d.Driver, { width: 240, height: 240 });
    const swap = () => {
      if (my !== token) return;
      layer.replaceChildren(img || h("div", { class: "ini" }, d.Driver));
      num.textContent = p.number || "";
      tag.replaceChildren(h("b", {}, d.Driver), h("span", {}, `${pct(d.win)} win`));
    };
    setTimeout(swap, reduced() ? 0 : 140);
  };
  return { el, show };
}
function marqueeEl(list) {
  const item = (d) => h("span", {}, d.Driver, h("i", {}, (name(d.Driver).split(" ").slice(-1)[0] || "")), h("b", {}, pct(d.win)));
  const track = h("div", { class: "marquee-track" }, list.map(item), list.map((d) => { const e = item(d); e.setAttribute("aria-hidden", "true"); return e; }));
  return h("div", { class: "marquee", role: "img", "aria-label": `Win chances: ${list.map((d) => `${d.Driver} ${pct(d.win)}`).join(", ")}` }, track);
}
/* split-flap (Solari) clock: each digit is a tile whose top half falls away to reveal the next digit */
function flapDigit(ch) {
  const half = (cls) => h("span", { class: cls, "aria-hidden": "true" }, h("span", {}, ch));
  const el = h("span", { class: "flap" }, half("fl-top"), half("fl-bottom"), half("fl-leaf fl-leaf-top"), half("fl-leaf fl-leaf-bottom"));
  el.dataset.v = ch;
  return el;
}
function flapTo(el, ch) {
  const old = el.dataset.v;
  if (old === ch) return;
  el.dataset.v = ch;
  const [top, bottom, leafTop, leafBottom] = el.children;
  const set = (node, v) => { node.firstChild.textContent = v; };
  if (reduced()) { [top, bottom, leafTop, leafBottom].forEach((n) => set(n, ch)); return; }
  set(top, ch);          // behind the falling leaf: next digit, upper half
  set(leafTop, old);     // falling leaf: current digit, upper half
  set(leafBottom, ch);   // rising leaf: next digit, lower half
  set(bottom, old);      // stays until the new lower half lands
  el.classList.remove("flipping"); void el.offsetWidth; el.classList.add("flipping");
  clearTimeout(el._t);
  el._t = setTimeout(() => { set(bottom, ch); set(leafTop, ch); el.classList.remove("flipping"); }, 620);
}
function countdownEl(m) {
  const iso = m.sessions && m.sessions.R;
  if (!iso) return null;
  const t = new Date(iso).getTime();
  const units = [["days", 86400], ["hours", 3600], ["minutes", 60], ["seconds", 1]];
  const box = h("div", { class: "flipclock", role: "timer", "aria-label": "time to lights out" });
  const live = h("span", { class: "sr-only" });
  const groups = units.map(([label], i) => {
    const digits = h("span", { class: "fc-digits" }, flapDigit("0"), flapDigit("0"));
    const g = h("span", { class: "fc-group" }, digits, h("span", { class: "fc-label" }, label));
    if (i) box.append(h("span", { class: "fc-colon", "aria-hidden": "true" }, h("i"), h("i")));
    box.append(g);
    return digits;
  });
  box.append(live);
  const tick = () => {
    let s = Math.max(0, Math.floor((t - Date.now()) / 1000));
    if (s === 0) { box.innerHTML = '<span class="fc-go">Lights out</span>'; clearInterval(state.cd); return; }
    const vals = units.map(([, sec]) => { const v = Math.floor(s / sec); s %= sec; return v; });
    vals.forEach((v, i) => {
      const str = String(v).padStart(2, "0");
      const digits = groups[i];
      while (digits.children.length < str.length) digits.prepend(flapDigit("0"));   // 100+ days
      [...str].forEach((c, k) => flapTo(digits.children[k], c));
    });
    live.textContent = `${vals[0]} days ${vals[1]} hours ${vals[2]} minutes to lights out`;
  };
  tick(); clearInterval(state.cd); state.cd = setInterval(tick, 1000);
  return box;
}
function rangeBar(d, n) {
  const x = (p) => ((p - 1) / (n - 1)) * 100;
  return h("div", { class: "range", style: `--team:${teamColor(d.Team)}`, "data-tip": `P10 ${d.p10} · median ${d.p50} · P90 ${d.p90} · mean ${fx(d.exp_pos, 2)}` },
    h("div", { class: "track" }), h("div", { class: "span", style: `left:${x(d.p10)}%;width:${x(d.p90) - x(d.p10)}%` }), h("div", { class: "med", style: `left:${x(d.p50)}%` }), h("div", { class: "mean", style: `left:${x(d.exp_pos)}%` }));
}
function stopBar(d) {
  return h("div", { class: "stackbar", "data-tip": `1 stop ${pct(d.stop1)} · 2 stops ${pct(d.stop2)} · 3+ ${pct(d.stop3p)}` },
    h("i", { style: `width:${d.stop1 * 100}%;background:#60a5fa` }), h("i", { style: `width:${d.stop2 * 100}%;background:var(--data)` }), h("i", { style: `width:${d.stop3p * 100}%;background:#a78bfa` }));
}
function quickFacts(nx) {
  const D = nx.drivers, sorted = D.slice().sort((a, b) => b.win - a.win), fav = sorted[0];
  const upset = 1 - sorted.slice(0, 3).reduce((s, d) => s + d.win, 0), risky = D.slice().sort((a, b) => b.dnf - a.dnf)[0];
  const multi = D.reduce((s, d) => s + d.stop2 + d.stop3p, 0) / D.length;
  return h("div", { class: "grid g-2" },
    stat(pct(fav.win), "Favourite", `${name(fav.Driver)} still loses ${pct(1 - fav.win)} of the time`, "accent"),
    stat(pct(upset), "Surprise winner", "someone outside the top-3 favourites wins", "cyan"),
    stat(pct(multi), "Two stops or more", `${fx(D.reduce((s, d) => s + d.exp_stops, 0) / D.length, 2)} stops per car on average`, ""),
    stat(pct(risky.dnf), "Biggest DNF risk", `${name(risky.Driver)} (${risky.Team})`, ""));
}
function heatmapPanel(nx) {
  const D = nx.drivers, n = D.length, g = h("div", { class: "heat", style: `grid-template-columns:120px repeat(${n}, minmax(24px,1fr))` });
  g.append(h("div"));
  for (let p = 1; p <= n; p++) g.append(h("div", { class: "hh" }, p));
  for (const d of D) {
    g.append(h("div", { class: "hl" }, avatar(d.Driver, 22, d.Team), d.Driver));
    const row = nx.dist[d.Driver] || [];
    for (let p = 0; p < n; p++) {
      const v = row[p] || 0, a = Math.min(1, Math.sqrt(v / 0.5));
      g.append(h("div", { class: "hc", style: `background:rgba(${p < 3 ? "251,191,36" : p < 10 ? "56,189,248" : "220,38,38"},${a.toFixed(3)})`, "data-tip": `${d.Driver} finishes P${p + 1}: ${pct(v, 2)}` }, v >= 0.1 ? Math.round(v * 100) : ""));
    }
  }
  return panel("Finishing-position probability matrix", "every driver × every position · √-scaled colour · numbers ≥ 10%", h("div", { class: "table-wrap" }, g));
}

/* ------------------------------------------------------------------ DRIVER DRAWER */
function openDriver(code) {
  const nx = state.data.next, d = nx?.drivers.find((x) => x.Driver === code);
  const r = state.data.ratings.find((x) => x.Driver === code);
  const st = state.data.standings.drivers.find((x) => x.Driver === code);
  const dv = state.data.driver_vs_car.filter((x) => x.Driver === code).sort((a, b) => b.races - a.races)[0];
  const p = person(code), team = d?.Team || r?.team || st?.Team || p.team;
  const body = $("#drawer-body"); body.innerHTML = "";
  body.append(h("div", { class: "dhead", style: `--team:${teamColor(team)}` },
    h("div", { class: "num", "aria-hidden": "true" }, p.number || ""),
    cutout(code, { alt: `${name(code)}`, width: 480, height: 480 }),
    h("div", { class: "info" }, teamLogo(team, 26), h("div", { class: "code" }, code), h("div", { class: "name" }, name(code)), h("div", { class: "team" }, team))));
  if (d) {
    body.append(h("div", { class: "dstats" },
      h("div", {}, h("b", {}, pct(d.win)), h("span", {}, "win")), h("div", {}, h("b", {}, pct(d.podium)), h("span", {}, "podium")),
      h("div", {}, h("b", {}, "P" + fx(d.exp_pos, 1)), h("span", {}, "avg finish")), h("div", {}, h("b", {}, pct(d.dnf)), h("span", {}, "DNF"))));
    body.append(h("div", { class: "dsec" }, h("h4", {}, `${nx.meta.event}: where they finish`), h("div", { html: histogram(nx.dist[code] || [], { color: teamColor(team) }) }), h("p", { class: "sub" }, "gold zone = podium, green = points")));
    const alts = (d.alt_strategies || "").split(";").map((s) => s.trim()).filter(Boolean);
    body.append(h("div", { class: "dsec" }, h("h4", {}, "Strategy"), h("div", { class: "stack", style: "gap:8px" },
      h("div", {}, tyres(d.strategy), h("span", { class: "strat-txt" }, `most likely · ${pct(d.strategy_p)} of races`)),
      alts.map((a) => { const [s, q] = a.split(" "); return h("div", {}, tyres(s), h("span", { class: "strat-txt" }, q)); }),
      h("p", { class: "sub" }, `First stop around lap ${fx(d.first_stop_p50, 0)} (window L${fx(d.first_stop_p25, 0)}-L${fx(d.first_stop_p75, 0)}). 1-stop ${pct(d.stop1)}, 2-stop ${pct(d.stop2)}, 3+ ${pct(d.stop3p)}.`))));
    if (nerd()) body.append(h("div", { class: "dsec" }, h("h4", {}, "Model internals"), h("dl", { class: "kv" },
      h("dt", {}, "race pace Δ"), h("dd", {}, `${sgn(d.pace_pct, 3)} % ± ${fx(d.pace_sd, 3)}`), h("dt", {}, "expected grid"), h("dd", {}, fx(d.grid, 2)),
      h("dt", {}, "P10 / P50 / P90"), h("dd", {}, `${d.p10} / ${d.p50} / ${d.p90}`), h("dt", {}, "expected points"), h("dd", {}, fx(d.exp_pts, 3)), h("dt", {}, "expected stops"), h("dd", {}, fx(d.exp_stops, 3)))));
  }
  if (dv) body.append(h("div", { class: "dsec" }, h("h4", {}, "Driver vs car"),
    h("p", { class: "explain" }, `The ${dv.Team} was worth about `, h("b", {}, `P${fx(dv.car_expected, 1)}`), " on average; ", name(code), " finished ", h("b", {}, `P${fx(dv.actual, 1)}`), `: `,
      h("b", { style: `color:${dv.race_delta >= 0 ? "var(--good)" : "var(--bad)"}` }, `${sgn(dv.race_delta, 1)} places per race`), ` (beat the car in ${pct(dv.beat_car_pct)} of finishes).`)));
  if (r) {
    const metrics = [["race_pace", "Race pace", "s/lap"], ["quali_pace", "Quali pace", "s/lap"], ["starts", "Starts", "places"], ["execution", "Race craft", "places"], ["tyre_mgmt", "Tyre care", "ms/lap²"], ["consistency", "Consistency", "s"]];
    body.append(h("div", { class: "dsec" }, h("h4", {}, "Car-adjusted ratings vs the field"), metrics.map(([k, lbl, unit]) => {
      const m = Math.max(...state.data.ratings.map((x) => Math.abs(x[k] ?? 0))) || 1, v = r[k] ?? 0, w = (Math.abs(v) / m) * 50;
      return h("div", { class: "rrow", "data-tip": `${lbl}: ${sgn(v, 3)} ${unit}${nerd() ? ` ± ${fx(r[k + "_se"], 3)} (SE)` : ""}` },
        h("span", { class: "muted" }, lbl), h("div", { class: "rbar" }, h("span", { class: "mid" }), h("i", { style: `${v >= 0 ? "left:50%" : `left:${50 - w}%`};width:${w}%;background:${v >= 0 ? "var(--good)" : "var(--bad)"}` })),
        h("span", { class: "mono", style: "text-align:right" }, sgn(v, 2)));
    })));
  }
  if (st) body.append(h("div", { class: "dsec" }, h("h4", {}, `${state.data.season} season · ${st.points} pts`), resultStrip(st.results)));
  $("#drawer").classList.add("open"); $("#scrim").classList.add("open"); $("#drawer").setAttribute("aria-hidden", "false");
  document.dispatchEvent(new CustomEvent("drawer:open", { detail: { body } }));
  state.lastFocus = document.activeElement; $("#drawer-close").focus();
}
function closeDrawer() {
  $("#drawer").classList.remove("open"); $("#scrim").classList.remove("open"); $("#drawer").setAttribute("aria-hidden", "true");
  if (state.lastFocus && document.contains(state.lastFocus)) state.lastFocus.focus();
}
function resultStrip(results) {
  return h("div", { class: "strip" }, results.map((r) => {
    const cls = r.dnf ? "dnf" : r.finish === 1 ? "p1" : r.finish === 2 ? "p2" : r.finish === 3 ? "p3" : r.finish <= 10 ? "pts" : "";
    return h("i", { class: cls, "data-tip": `R${r.round}: ${r.dnf ? "DNF" : "P" + r.finish} (from P${r.grid})` }, r.dnf ? "×" : r.finish);
  }));
}

/* ------------------------------------------------------------------ STRATEGY */
function renderStrategy(root) {
  const nx = state.data.next;
  if (!nx) return root.append(h("div", { class: "empty" }, "No prediction yet."));
  const c = nx.circuit, D = nx.drivers, N = c.n_laps, avg = (k) => D.reduce((s, d) => s + d[k], 0) / D.length;
  const expStops = avg("exp_stops"), pass = c.overtake_factor < 0.7 ? "Hard" : c.overtake_factor > 1.3 ? "Easy" : "Average";
  const pooled = nx.meta.status === "provisional" ? " (pooled estimate: no race here in our data)" : "";
  root.append(sectionHead(["Strategy, ", em(nx.meta.identity?.circuit || nx.meta.event.replace(/\s*Grand Prix$/i, ""))], "How the race is likely to be run: stops, tyres and pit windows, from the same simulations."),
    h("div", { class: "grid g-4" },
      stat(fx(expStops, 1), "Stops per car", `1-stop ${pct(avg("stop1"))} · 2-stop ${pct(avg("stop2"))} · 3+ ${pct(avg("stop3p"))}`, "cyan"),
      stat(`${fx(c.pit_loss, 1)}<small>s</small>`, "Pit stop cost", "time lost driving through the pit lane" + pooled, ""),
      stat(pct(nx.meta.p_sc), "Safety car chance", `${fx(c.sc_per_race, 2)} SC/red + ${fx(c.vsc_per_race, 2)} VSC per race here`, "accent"),
      stat(pass, "Overtaking", `${fx(c.overtake_factor, 2)}× an average track's passing rate` + pooled, "")));
  const reco = strategyRecoPanel(nx);
  const why = h("p", { class: "explain" }, h("b", {}, `Why ${expStops > 1.6 ? "two stops" : "one stop"}? `),
    `Tyres here lose about ${fx((c.deg?.MEDIUM ?? 0.05) * 1000, 0)} thousandths of a second per lap on the medium, and a stop costs ~${fx(c.pit_loss, 0)} s. `,
    expStops > 1.6 ? "Wear adds up fast enough that fresh tyres pay for the extra stop." : "Stopping again costs more than the wear it saves, so most teams run long stints.",
    ` A safety car (${pct(nx.meta.p_sc)} likely) makes stops cheaper and often rewrites the plan.`);
  const lanes = h("div", { class: "lanes" }, h("div", { class: "lane-axis" }, h("div"), h("div", {}, ...[1, Math.round(N / 4), Math.round(N / 2), Math.round((3 * N) / 4), N].map((l) => h("span", {}, "L" + l))), h("div")));
  for (const d of D) {
    const seq = (d.strategy || "M-H").split("-"), k = seq.length - 1, frac = c.stint_frac?.[k] || Array(k + 1).fill(1 / (k + 1));
    const track = h("div", { class: "lane-track" });
    let acc = 0;
    seq.forEach((cmp, i) => { const w = frac[i] * 100; track.append(h("div", { class: `lane-stint ${cmp}`, style: `left:calc(${acc}% + 1px);width:calc(${w}% - 2px)`, "data-tip": `${COMP[cmp]} · ~${Math.round(frac[i] * N)} laps` })); acc += w; });
    if (d.first_stop_p25 != null) track.append(h("div", { class: "lane-win", style: `left:${(d.first_stop_p25 / N) * 100}%;width:${((d.first_stop_p75 - d.first_stop_p25) / N) * 100}%`, "data-tip": `first stop window L${fx(d.first_stop_p25, 0)}-L${fx(d.first_stop_p75, 0)}` }));
    lanes.append(btn("lane rowbtn", () => openDriver(d.Driver), null, drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team, 30), track,
      h("span", { class: "strat-txt", style: "margin:0", "data-tip": `this exact compound order: ${pct(d.strategy_p)}` }, `${k}-stop · ${pct([null, d.stop1, d.stop2, d.stop3p][k])}`)));
  }
  const right = [panel("The strategy call", null, why)];
  if (nerd()) {
    right.push(panel("Degradation model", "lap-time loss vs tyre age · dashed = 97th-pct stint length", h("div", { html: degChart(c) }),
      h("div", { class: "legend" }, ["SOFT", "MEDIUM", "HARD"].map((k) => h("span", {}, h("i", { style: `background:${COMP_COLOR[k]}` }), `${k} ${fx(c.deg?.[k] * 1000, 0)} ms/lap · max ~${c.max_stint?.[k] ?? "?"}`)))));
    right.push(panel("Candidate plans", "time vs best plan under the deg model", table(nx.meta.strategies || [], [
      { id: "seq", label: "Plan", render: (s) => tyres(s.seq) }, { id: "st", label: "Stops", num: true, render: (s) => s.seq.split("-").length - 1 },
      { id: "lens", label: "Stint laps", render: (s) => h("span", { class: "mono muted" }, s.lens.join(" / ")) },
      { id: "d", label: "Δ time", num: true, render: (s) => (s.delta === 0 ? h("span", { class: "badge good" }, "best") : "+" + fx(s.delta, 1) + "s") }])));
    right.push(panel("Stop-count prior", "planned = season norm + venue offset", h("dl", { class: "kv" }, h("dt", {}, "planned stops μ"), h("dd", {}, fx(c.stops_mu, 2)),
      ...[1, 2, 3].flatMap((k) => [h("dt", {}, `P(${k} planned)`), h("dd", {}, pct(c.stops_dist?.[k]))]),
      ...[1, 2, 3].flatMap((k) => [h("dt", {}, `${k}-stop split`), h("dd", {}, (c.stint_frac?.[k] || []).map((f) => Math.round(f * 100) + "%").join(" / "))]))));
  }
  root.append(h("div", { class: "grid g-main mt" }, panel("Tyre plan by driver", "most likely plan · shaded = first-stop window", lanes,
    h("div", { class: "legend" }, ["S", "M", "H"].map((k) => h("span", {}, h("i", { style: `background:${COMP_COLOR[k]}` }), COMP[k])), h("span", {}, h("i", { style: "background:var(--data)" }), "pit window"))), h("div", { class: "stack" }, right)));
  if (reco) root.append(h("div", { class: "mt" }, reco));
}

/* ------------------------------------------------------------------ DRIVERS */
const AWARDS = [
  ["race_pace", "Fastest racer", "s/lap vs average driver", (v) => sgn(v, 2) + " s/lap", "M13 2 3 14h9l-1 8 10-12h-9z"],
  ["quali_pace", "Qualifying king", "one-lap pace, car removed", (v) => sgn(v, 2) + " s/lap", "M12 2l3 6 7 1-5 5 1 7-6-3-6 3 1-7-5-5 7-1z"],
  ["execution", "Race craft", "places gained vs own pace", (v) => sgn(v, 2) + " places", "M4 20l6-6 4 4 6-10M14 8h6v6"],
  ["starts", "Rocket starts", "lap-1 places vs grid slot", (v) => sgn(v, 2) + " places", "M5 19c3-1 5-3 6-6l7-7-4 1-7 7c-1 3-1 4-2 5z"],
  ["tyre_mgmt", "Tyre whisperer", "slowest tyre fade", (v) => sgn(v, 1) + " ms/lap²", "M12 3a9 9 0 1 0 0 18 9 9 0 0 0 0-18zm0 5a4 4 0 1 1 0 8 4 4 0 0 1 0-8z"],
  ["consistency", "Metronome", "tightest lap-time scatter", (v) => sgn(v * 1000, 0) + " ms", "M12 6v6l4 2M12 2a10 10 0 1 0 0 20 10 10 0 0 0 0-20z"],
];
function renderDrivers(root) {
  const R = state.data.ratings.slice();
  if (!R.length) return root.append(h("div", { class: "empty" }, "No ratings yet."));
  root.append(sectionHead(["Drivers, ", em("car removed")], "Every driver compared with their team-mate in the same car, race after race. Drivers who switched teams tie the whole grid onto one scale."));
  const [fk, ft, fsub, ff] = AWARDS[0];
  const top = R.filter((r) => r[fk] != null).sort((a, b) => b[fk] - a[fk])[0];
  const feature = h("button", { type: "button", class: "panel clickable award-feature", style: `--team:${teamColor(top.team)}`, onclick: () => openDriver(top.Driver) },
    cutout(top.Driver, { loading: "lazy", width: 480, height: 480 }),
    h("span", { class: "copy", style: "display:block" }, h("span", { class: "kicker", style: "display:block" }, ft),
      h("span", { class: "who", style: "display:block" }, name(top.Driver)), h("span", { class: "mono hl", style: "display:block" }, ff(top[fk])),
      h("span", { class: "sub", style: "display:block;margin-top:6px" }, fsub)));
  const list = stagger(h("div", { class: "panel award-list" }, AWARDS.slice(1).map(([k, t, sub, f]) => {
    const best = R.filter((r) => r[k] != null).sort((a, b) => b[k] - a[k])[0];
    return btn("rowbtn", () => openDriver(best.Driver), null, avatar(best.Driver, 44, best.team),
      h("span", {}, h("span", { class: "t", style: "display:block" }, t), h("span", { class: "n", style: "display:block" }, name(best.Driver))),
      h("span", { class: "v", "data-tip": sub }, f(best[k])));
  })));
  root.append(h("div", { class: "awards" }, feature, list));
  const expl = nerd()
    ? h("p", { class: "explain", html: "Each metric is fitted as <b>metric[race, driver] = car[race, team] + skill[driver] + ε</b>, with one free car effect per team per race, ridge-shrunk (λ=2) driver skills, recency-weighted (half-life 16 races). Whiskers are 95% intervals; overlapping whiskers mean the data can't separate the drivers." })
    : h("p", { class: "explain", html: "Results mostly measure the <b>car</b>. These ratings compare drivers in the same car every weekend, so what's left is the <b>driver</b>." });
  if (!nerd()) {
    const max = Math.max(...R.map((r) => Math.abs(r.overall)));
    const view = state.driverView || "cards";
    const toggle = h("div", { class: "seg", role: "group", "aria-label": "Ranking view" }, [["cards", "Cards"], ["list", "List"]].map(([k, l]) =>
      h("button", { type: "button", class: view === k ? "on" : "", "aria-pressed": String(view === k), onclick: () => { state.driverView = k; rerender(); } }, l)));
    if (view === "cards") {
      root.append(h("div", { class: "mt" }, h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("h2", {}, "Driver power ranking", h("small", {}, "car removed")), toggle),
        stagger(h("div", { class: "gallery" }, R.map((r, i) => h("button", { type: "button", class: "gcard", style: `--team:${teamColor(r.team)}`, onclick: () => openDriver(r.Driver), "aria-label": `${i + 1}. ${name(r.Driver)}, rating ${sgn(r.overall, 2)}` },
          cutout(r.Driver, { loading: "lazy", width: 240, height: 240 }), h("span", { class: "rk" }, i + 1),
          h("span", { class: "nm2" }, name(r.Driver).split(" ").slice(-1)[0]), h("span", { class: "sc" }, `${sgn(r.overall, 2)}  ${SHORT[r.team] || r.team}`))))))),
        h("div", { class: "mt" }, panel("How to read this", null, expl)));
      return;
    }
    root.append(h("div", { class: "grid g-main mt" },
      h("div", { class: "panel" }, h("div", { class: "panel-head" }, h("h2", {}, "Driver power ranking", h("small", {}, "car removed")), toggle), stagger(h("div", { class: "stack", style: "gap:2px" }, R.map((r, i) => btn("rowbtn", () => openDriver(r.Driver), "grid-template-columns:30px 190px 1fr;gap:12px;align-items:center;min-height:44px",
        h("span", { class: "pos" }, i + 1), drvCell(r.Driver, r.team, SHORT[r.team] || r.team), pbar(0.5 + r.overall / (2 * max), teamColor(r.team), sgn(r.overall, 2))))))),
      h("div", { class: "sticky-side" }, panel("How to read this", null, expl))));
    return;
  }
  const cols = [
    { id: "d", label: "Driver", val: (r) => r.Driver, render: (r) => drvCell(r.Driver, r.team, SHORT[r.team] || r.team) },
    { id: "n", label: "Races", num: true, desc: true, val: (r) => r.races }, { id: "o", label: "Overall z", num: true, desc: true, val: (r) => r.overall, render: (r) => fx(r.overall, 2) },
    ...[["race_pace", "Race s/lap", 3], ["quali_pace", "Quali s/lap", 3], ["execution", "Exec", 2], ["starts", "Starts", 2], ["tyre_mgmt", "Tyre ms", 1], ["consistency", "Consist.", 3]].map(([k, l, dp]) => ({
      id: k, label: l, num: true, desc: true, val: (r) => r[k], render: (r) => h("span", { "data-tip": `SE ${fx(r[k + "_se"], 3)} · n=${r[k + "_n"]}` }, sgn(r[k], dp), h("span", { class: "muted" }, ` ±${fx(r[k + "_se"], dp)}`)) })),
    { id: "dnf", label: "DNF rate", num: true, val: (r) => r.dnf_rate, render: (r) => pct(r.dnf_rate) },
  ];
  root.append(h("div", { class: "grid g-2 mt" },
    panel("Race pace", "s/lap vs average driver · 95% CI", h("div", { html: forest(R.slice().sort((a, b) => b.race_pace - a.race_pace), { key: "race_pace", se: "race_pace_se", unit: " s/lap" }) })),
    panel("Qualifying pace", "s/lap vs average driver · 95% CI", h("div", { html: forest(R.slice().sort((a, b) => b.quali_pace - a.quali_pace), { key: "quali_pace", se: "quali_pace_se", unit: " s/lap" }) }))),
    h("div", { class: "mt" }, panel("All ratings", "± standard error · sortable", table(R, cols, { key: "ratings", onRow: (r) => openDriver(r.Driver) }))),
    h("div", { class: "mt" }, panel("Method", null, expl)));
}

/* ------------------------------------------------------------------ DRIVER VS CAR */
function renderDvC(root) {
  const all = (state.data.driver_vs_car || []).filter((d) => d.races >= 3);
  if (!all.length) return root.append(h("div", { class: "empty" }, "No data yet."));
  const q = state.dvcMode === "quali";
  const ek = q ? "grid_expected" : "car_expected", ak = q ? "grid_actual" : "actual", dk = q ? "quali_delta" : "race_delta";
  const rows = all.filter((d) => d[dk] != null).sort((a, b) => b[dk] - a[dk]);
  const seg = h("div", { class: "seg", role: "group", "aria-label": "Race or qualifying" }, [["race", "Race finish"], ["quali", "Qualifying"]].map(([k, l]) =>
    h("button", { class: state.dvcMode === k ? "on" : "", "aria-pressed": String(state.dvcMode === k), onclick: () => { state.dvcMode = k; rerender(); } }, l)));
  root.append(sectionHead(q ? ["Who out-qualifies ", em("their car")] : ["Who beats ", em("their car")],
    q ? "Where each car should start from its one-lap pace, against where its drivers actually qualified."
      : "Where each car should finish on pace alone, against where its drivers actually finished. Positive means the driver found places the car didn't have.", seg));
  const best = rows[0], worst = rows[rows.length - 1];
  const bestTeamPair = Object.values(rows.reduce((m, r) => { (m[r.Team] = m[r.Team] || []).push(r); return m; }, {})).filter((p) => p.length === 2)
    .map((p) => ({ team: p[0].Team, gap: Math.abs(p[0][dk] - p[1][dk]), a: p[0], b: p[1] })).sort((a, b) => b.gap - a.gap)[0];
  const hlFeature = h("button", { type: "button", class: "panel clickable award-feature", style: `--team:${teamColor(best.Team)};min-height:240px`, onclick: () => openDriver(best.Driver) },
    cutout(best.Driver, { loading: "lazy", width: 480, height: 480 }),
    h("span", { class: "copy", style: "display:block" }, h("span", { class: "kicker", style: "display:block" }, "Biggest over-performer"),
      h("span", { class: "who", style: "display:block" }, name(best.Driver)), h("span", { class: "mono", style: "display:block;color:var(--good)" }, `${sgn(best[dk], 1)} places per ${q ? "session" : "race"}`),
      !q && best.race_delta_clear === false ? h("span", { class: "unclear-note", style: "display:block" }, `not yet clear of race-to-race noise (90% range ${sgn(best.race_delta_lo, 1)} to ${sgn(best.race_delta_hi, 1)})`) : null));
  const hlList = h("div", { class: "panel award-list" },
    btn("rowbtn", () => openDriver(worst.Driver), null, avatar(worst.Driver, 44, worst.Team),
      h("span", {}, h("span", { class: "t", style: "display:block" }, "Leaves most on the table"), h("span", { class: "n", style: "display:block" }, name(worst.Driver))),
      h("span", { class: "v", style: "color:var(--bad)" }, sgn(worst[dk], 1))),
    bestTeamPair ? btn("rowbtn", () => openDriver(bestTeamPair.a.Driver), null, teamLogo(bestTeamPair.team, 30) || avatar(bestTeamPair.a.Driver, 44),
      h("span", {}, h("span", { class: "t", style: "display:block" }, "Biggest team-mate gap"), h("span", { class: "n", style: "display:block" }, `${bestTeamPair.a.Driver} vs ${bestTeamPair.b.Driver}`)),
      h("span", { class: "v" }, `${fx(bestTeamPair.gap, 1)} places`)) : null);
  root.append(h("div", { class: "awards" }, hlFeature, hlList));
  const noteSlot = h("div");
  root.append(noteSlot);

  // dumbbell: hollow = car, filled = driver
  const maxP = 22, x = (p) => ((p - 1) / (maxP - 1)) * 100;
  const chart = h("div", { class: "stack", style: "gap:2px" },
    h("div", { class: "dvc-row", style: "cursor:default;min-height:24px" }, h("span"), h("div", { style: "display:flex;justify-content:space-between;font:500 12px var(--mono);color:var(--muted)" }, ...[1, 5, 10, 15, 20].map((p) => h("span", {}, "P" + p))), h("span", { class: "mono muted", style: "text-align:right;font-size:12px" }, "Δ")),
    ...rows.map((d, i) => {
      const e = d[ek], a = d[ak], up = d[dk] >= 0, c = teamColor(d.Team);
      const lo = Math.min(x(e), x(a)), wdt = Math.abs(x(e) - x(a));
      return h("button", { type: "button", class: "dvc-row rowbtn", onclick: () => { state.dvcDriver = d.Driver; rerender(); setTimeout(() => $("#dvc-detail")?.scrollIntoView({ behavior: "smooth", block: "start" }), 50); },
        "aria-label": `${name(d.Driver)}: car worth P${fx(e, 1)}, finished P${fx(a, 1)}` },
        drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team, 32),
        h("span", { style: "position:relative;height:28px;display:block" },
          h("span", { style: "position:absolute;left:0;right:0;top:13px;height:2px;background:var(--border)" }),
          h("span", { style: `position:absolute;top:11px;height:6px;border-radius:3px;left:${lo}%;width:${wdt}%;background:${up ? "var(--good)" : "var(--bad)"};opacity:.55` }),
          h("span", { "data-tip": `car worth P${fx(e, 1)}`, style: `position:absolute;top:6px;left:calc(${x(e)}% - 8px);width:16px;height:16px;border-radius:50%;border:2.5px solid ${c};background:var(--surface)` }),
          h("span", { "data-tip": `${d.Driver} averaged P${fx(a, 1)}`, style: `position:absolute;top:4px;left:calc(${x(a)}% - 10px)` }, avatar(d.Driver, 20, d.Team))),
        h("span", { class: `dvc-val ${up ? "up" : "down"} ${!q && d.race_delta_clear === false ? "unclear" : ""}`,
          "data-tip": !q && d.race_delta_lo != null ? `90% range ${sgn(d.race_delta_lo, 1)} to ${sgn(d.race_delta_hi, 1)} places${d.race_delta_clear ? "" : " (includes zero: within noise)"}` : null },
          sgn(d[dk], 1), nerd() && !q && d.race_delta_lo != null ? h("small", { class: "dvc-ci" }, `${sgn(d.race_delta_lo, 1)}\u2026${sgn(d.race_delta_hi, 1)}`) : null));
    }));
  const clearN = rows.filter((d) => d.race_delta_clear).length;
  queueMicrotask(() => dvcNote && noteSlot.append(dvcNote));
  const dvcNote = !q ? h("p", { class: "note dvc-honest" }, h("b", {}, "What this can and can't show: "),
    nerd() ? (state.data.driver_vs_car_note || "") : "Each car's pace is worked out from both of its drivers, so this mostly compares team-mates. ",
    ` Only ${clearN} of ${rows.length} drivers are clearly above or below their car so far; the rest are within normal race-to-race luck (dimmed numbers).`) : null;
  const legend = h("div", { class: "legend" }, h("span", {}, h("i", { class: "hollow" }), "where the car should be"), h("span", {}, h("i", { class: "filled" }), "where the driver was"),
    h("span", {}, h("i", { style: "background:var(--good)" }), "beat the car"), h("span", {}, h("i", { style: "background:var(--bad)" }), "below the car"));
  const note = h("p", { class: "note mt" }, h("b", {}, "How it works: "), q
    ? "Car pace comes from the car-adjusted model (team effect with each driver's skill removed). Cars are ranked by one-lap pace each weekend; a driver 'should' start behind everyone in a faster car and share their own car's slots with their team-mate."
    : "Among the cars that finished each race, a driver 'should' finish behind everyone in a faster car and share their own car's slots with their team-mate. Retirements ahead don't flatter backmarkers and every race sums to zero. Caveat: the fastest car can only lose places (there's no P0), so front-runners read slightly negative.");
  root.append(h("div", { class: "grid g-main mt" }, panel(q ? "Qualifying vs car" : "Race finish vs car", `average per ${q ? "session" : "race"} · ${state.data.season} · tap a driver`, chart, legend, note), dvcDetail(all, q)));
  if (nerd()) {
    const cols = [
      { id: "d", label: "Driver", val: (r) => r.Driver, render: (r) => drvCell(r.Driver, r.Team, SHORT[r.Team] || r.Team) },
      { id: "n", label: "Races", num: true, val: (r) => r.races }, { id: "c", label: "Finished", num: true, val: (r) => r.classified },
      { id: "ce", label: "Car worth", num: true, val: (r) => r.car_expected, render: (r) => "P" + fx(r.car_expected, 2) }, { id: "a", label: "Finished avg", num: true, val: (r) => r.actual, render: (r) => "P" + fx(r.actual, 2) },
      { id: "rd", label: "Race Δ", num: true, desc: true, val: (r) => r.race_delta, render: (r) => sgn(r.race_delta, 2) },
      { id: "ge", label: "Grid worth", num: true, val: (r) => r.grid_expected, render: (r) => "P" + fx(r.grid_expected, 2) }, { id: "ga", label: "Grid avg", num: true, val: (r) => r.grid_actual, render: (r) => "P" + fx(r.grid_actual, 2) },
      { id: "qd", label: "Quali Δ", num: true, desc: true, val: (r) => r.quali_delta, render: (r) => sgn(r.quali_delta, 2) },
      { id: "b", label: "Beat car", num: true, desc: true, val: (r) => r.beat_car_pct, render: (r) => pct(r.beat_car_pct) },
    ];
    root.append(h("div", { class: "mt" }, panel("Full table", "data behind the chart · sortable", table(all, cols, { key: "dvc", initial: { id: "rd", dir: -1 } }))));
  }
}
function dvcDetail(all, q) {
  const code = state.dvcDriver && all.find((d) => d.Driver === state.dvcDriver) ? state.dvcDriver : all[0].Driver;
  const d = all.filter((x) => x.Driver === code).sort((a, b) => b.races - a.races)[0];
  const pr = d.per_race || [];
  const labels = pr.map((r) => "R" + r.round);
  const exp = pr.map((r) => (q ? r.exp_grid : r.exp)), act = pr.map((r) => (q ? r.grid : (r.dnf ? null : r.actual)));
  const sel = h("select", { "aria-label": "Choose driver", onchange: (e) => { state.dvcDriver = e.target.value; rerender(); } },
    all.slice().sort((a, b) => a.Driver.localeCompare(b.Driver)).map((x) => h("option", { value: x.Driver, selected: x.Driver === code ? "selected" : null }, `${x.Driver} · ${SHORT[x.Team] || x.Team}`)));
  return h("div", { id: "dvc-detail", class: "sticky-side" }, panel("Race by race", null,
    h("div", { style: "display:flex;gap:12px;align-items:center;margin-bottom:12px;flex-wrap:wrap" }, avatar(code, 48, d.Team), h("div", {}, h("div", { style: "font:700 22px var(--display)" }, name(code)), h("div", { class: "sub" }, d.Team)), h("div", { style: "margin-left:auto" }, sel)),
    h("div", { html: lineChart({ labels, invert: true, yFmt: (v) => "P" + Math.min(22, Math.max(1, Math.round(v))), yLabel: q ? "grid slot" : "finish", series: [
      { name: "Car worth", color: "var(--muted)", values: exp, dash: "5 5", hollow: true },
      { name: name(code), color: teamColor(d.Team), values: act, width: 3 }] }) }),
    h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--muted)" }), "car worth (dashed)"), h("span", {}, h("i", { style: `background:${teamColor(d.Team)}` }), name(code)), q ? null : h("span", {}, "gaps = retirements")),
    h("p", { class: "sub mt" }, `${d.races} races · ${d.dnfs} DNF · beat the car in ${pct(d.beat_car_pct)} of finishes`)));
}

/* ------------------------------------------------------------------ TEAMS */
const TEAM_AXES = [["s_race", "Race pace"], ["s_quali", "Quali pace"], ["s_reliability", "Reliability"], ["s_execution", "Execution"], ["s_development", "Development"], ["s_pit", "Pit work"]];
function renderTeams(root) {
  const T = state.data.constructors || [];
  if (!T.length) return root.append(h("div", { class: "empty" }, "No constructor data yet."));
  const drivers = (team) => (state.data.standings.drivers || []).filter((d) => d.Team === team).slice(0, 2);
  root.append(sectionHead(["Rating ", em("the cars")], "Each car's pace with the drivers taken out, plus reliability, conversion of pace into results, pit work and in-season development. 50 is an average team."));
  const cards = stagger(h("div", { class: "stack" }, T.map((t, i) => h("div", { class: "teamcard", style: `--team:${teamColor(t.Team)}` },
    h("div", { class: "rank" }, i + 1),
    h("div", {}, h("div", { class: "tname" }, teamLogo(t.Team, 26), t.Team), h("div", { class: "tdrivers" }, drivers(t.Team).map((d) => h("button", { type: "button", class: "chipbtn", onclick: () => openDriver(d.Driver) }, avatar(d.Driver, 26, t.Team), d.Driver)),
      h("span", {}, `· ${t.wins} wins · ${t.podiums} podiums`))),
    h("div", { class: "score" }, h("b", {}, fx(t.overall, 0)), h("span", {}, "rating")),
    h("div", { class: "axisbars" }, TEAM_AXES.map(([k, l]) => h("div", { "data-tip": `${l}: ${fx(t[k], 0)}/100` }, l, h("div", { class: "b" }, h("i", { style: `width:${t[k]}%` })))))))));

  const sel = state.teamSel || T.slice(0, 3).map((t) => t.Team);
  const toggles = h("div", { style: "display:flex;flex-wrap:wrap;gap:8px;margin-bottom:12px" }, T.map((t) => h("button", {
    class: "pill-toggle", style: `--team:${teamColor(t.Team)}`, "aria-pressed": String(sel.includes(t.Team)),
    onclick: () => { let s = sel.slice(); s = s.includes(t.Team) ? s.filter((x) => x !== t.Team) : [...s, t.Team].slice(-3); state.teamSel = s.length ? s : [t.Team]; rerender(); },
  }, teamLogo(t.Team, 18) || h("i"), SHORT[t.Team] || t.Team)));
  const radarPanel = panel("Compare cars", "pick up to 3", toggles, h("div", { html: radar(T.filter((t) => sel.includes(t.Team)), TEAM_AXES) }),
    h("div", { class: "legend" }, T.filter((t) => sel.includes(t.Team)).map((t, j) => h("span", {}, h("i", { style: `background:${teamColor(t.Team)};height:${[3, 2, 2][j]}px` }), t.Team + ["", " (dashed)", " (dotted)"][j]))));
  root.append(h("div", { class: "grid g-main" }, cards, radarPanel));

  // pace trend
  const rounds = [...new Set(T.flatMap((t) => (t.pace_trend || []).map((p) => p.round)))].sort((a, b) => a - b);
  const show = nerd() ? T : T.filter((t) => sel.includes(t.Team));
  const trend = lineChart({ labels: rounds.map((r) => "R" + r), width: 1240, height: 340, yLabel: "pace vs median car, % (up = faster)", invert: true, zero: true, yFmt: (v) => sgn(v, 1) + "%",
    series: show.map((t) => ({ name: t.Team, color: teamColor(t.Team), values: rounds.map((r) => (t.pace_trend || []).find((p) => p.round === r)?.pace ?? null), width: sel.includes(t.Team) ? 3 : 1.4, opacity: sel.includes(t.Team) ? 1 : 0.35, endLabel: sel.includes(t.Team) ? (SHORT[t.Team] || t.Team) : null, r: sel.includes(t.Team) ? 3.6 : 2.4 })) });
  root.append(h("div", { class: "mt" }, panel("Car pace through the season", nerd() ? "all teams · selected highlighted · higher on chart = faster" : "selected teams · higher on chart = faster", h("div", { html: trend }))));
  if (nerd()) {
    const cols = [
      { id: "t", label: "Team", val: (t) => t.Team, render: (t) => h("span", {}, h("span", { style: `display:inline-block;width:10px;height:10px;border-radius:3px;background:${teamColor(t.Team)};margin-right:8px` }), t.Team) },
      { id: "o", label: "Rating", num: true, desc: true, val: (t) => t.overall, render: (t) => fx(t.overall, 1) },
      { id: "rp", label: "Race pace %", num: true, val: (t) => t.race_pace, render: (t) => sgn(t.race_pace, 2), tip: "recency-weighted car effect vs median car" },
      { id: "qp", label: "Quali pace %", num: true, val: (t) => t.quali_pace, render: (t) => sgn(t.quali_pace, 2) },
      { id: "sp", label: "Season avg %", num: true, val: (t) => t.season_pace, render: (t) => sgn(t.season_pace, 2) },
      { id: "dv", label: "Dev %/race", num: true, val: (t) => t.development, render: (t) => sgn(t.development, 3), tip: "negative = getting faster" },
      { id: "rl", label: "Finish rate", num: true, desc: true, val: (t) => t.reliability, render: (t) => pct(t.reliability) },
      { id: "ex", label: "Execution", num: true, desc: true, val: (t) => t.execution, render: (t) => sgn(t.execution, 2), tip: "places per car vs car pace (zero-sum)" },
      { id: "pt", label: "Pit Δ s", num: true, val: (t) => t.pit_ops, render: (t) => sgn(t.pit_ops, 2), tip: "median pit loss vs race median" },
      { id: "w", label: "Wins", num: true, desc: true, val: (t) => t.wins },
    ];
    root.append(h("div", { class: "mt" }, panel("Constructor data", "raw values behind the scores · sortable", table(T, cols, { key: "teams" }))));
  }
}

/* ------------------------------------------------------------------ SEASON */
function renderSeason(root) {
  const S = state.data.standings, maxP = S.drivers[0]?.points || 1;
  const rounds = Math.max(0, ...S.drivers.flatMap((d) => d.results.map((r) => r.round)));
  const cols = [
    { id: "p", label: "Pos", render: (d) => h("span", { class: "pos" }, S.drivers.indexOf(d) + 1) },
    { id: "d", label: "Driver", render: (d) => drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team) },
    { id: "pts", label: "Points", render: (d) => pbar(d.points / maxP, teamColor(d.Team), String(d.points)) },
    { id: "gap", label: "Gap", num: true, render: (d) => (d === S.drivers[0] ? "-" : "−" + (maxP - d.points)) },
    { id: "w", label: "Wins", num: true, nerd: true, render: (d) => d.results.filter((r) => r.finish === 1 && !r.dnf).length },
    { id: "pd", label: "Podiums", num: true, nerd: true, render: (d) => d.results.filter((r) => r.finish <= 3 && !r.dnf).length },
    { id: "af", label: "Avg finish", num: true, nerd: true, render: (d) => fx(d.results.filter((r) => !r.dnf).reduce((s, r, _, a) => s + r.finish / a.length, 0), 1) },
    { id: "r", label: `Results R1-R${rounds}`, render: (d) => resultStrip(d.results) },
  ];
  const tmax = S.teams[0]?.points || 1;
  root.append(sectionHead([`${state.data.season} `, em("championship")], "Race and sprint points after every round."),
    h("div", { class: "stack" },
      panel("Drivers", null, table(S.drivers, cols, { onRow: (d) => openDriver(d.Driver) })),
      h("div", { class: "grid g-2" },
        panel("Constructors", null, h("div", { class: "stack", style: "gap:8px" }, S.teams.map((t, i) => h("div", { style: "display:grid;grid-template-columns:26px 170px 1fr;gap:10px;align-items:center;min-height:36px" },
          h("span", { class: "pos" }, i + 1), h("span", { style: `display:flex;align-items:center;gap:8px;font:700 16px var(--display);text-transform:uppercase;color:${teamColor(t.Team)}` }, teamLogo(t.Team, 20), SHORT[t.Team] || t.Team), pbar(t.points / tmax, teamColor(t.Team), String(t.points)))))),
        panel("Legend", null, h("div", { class: "strip", style: "flex-wrap:wrap;gap:8px;align-items:center;font-size:14px;color:var(--text-2)" },
          h("i", { class: "p1" }, "1"), " win ", h("i", { class: "p2" }, "2"), h("i", { class: "p3" }, "3"), " podium ", h("i", { class: "pts" }, "7"), " points ", h("i", {}, "14"), " no points ", h("i", { class: "dnf" }, "×"), " DNF")))));
}

/* ------------------------------------------------------------------ ACCURACY */
function renderAccuracy(root) {
  const B = state.data.benchmarks?.[String(state.data.season)];
  if (B) benchmarkSection(root, B);
  const ev = state.data.season_eval?.[String(state.data.season)] || {};
  const mode = ev[state.accMode] ? state.accMode : Object.keys(ev)[0], E = ev[mode];
  if (!E) return B ? null : root.append(h("div", { class: "empty" }, "No evaluation yet. Run python -m flatout nested --year 2026 to score past races."));
  if (B) {   // the retrospective backtest below is kept for the per-race explorer, clearly labelled
    root.append(h("h2", { class: "subhead mt" }, "Race-by-race detail",
      h("small", {}, nerd() ? "retrospective backtest \u00b7 simulator settings tuned on these same races (in-sample) \u00b7 use the numbers above for accuracy" : "what we tipped at each race")));
  }
  const tag = "backtest_" + mode, races = state.data.races.filter((r) => r.predictions[tag]), a = E.avg;
  const wins = E.per_race.filter((r) => r.model_winner_correct).length, better = 1 - a.model.rps / a.grid.rps;
  if (!B) root.append(sectionHead(["How good are ", em("the predictions")], "Walk-forward test: the model is retrained before every race using only earlier races, then scored against the result.",
    h("div", { class: "seg", role: "group", "aria-label": "Information available" }, Object.keys(ev).map((k) => h("button", { class: k === mode ? "on" : "", "aria-pressed": String(k === mode), onclick: () => { state.accMode = k; rerender(); } }, MODE_LABEL[k] || k)))));
  if (!B) root.append(h("div", { class: "grid g-4" },
    stat(`${wins}<small>/${E.n}</small>`, "Favourite won", `trusting the grid: ${pct(a.grid.winner_correct)} of races`, "good"),
    stat(pct(better), "More accurate than the grid", "ranked probability score vs 'finish where you start'", "cyan"),
    stat(pct(a.model.p_winner), "Chance we gave the winner", `pace-only baseline ${pct(a.pace.p_winner)}`, "accent"),
    stat(pct(E.strategy.stops_acc), "Stop count right", `first stop off by ~${fx(E.strategy.first_stop_mae, 1)} laps`, "")));
  const calls = stagger(h("div", { class: "calls" }, races.slice().reverse().map((r) => {
    const P = r.predictions[tag].drivers, fav = P.slice().sort((x, y) => y.win - x.win)[0], won = P.find((x) => x.actual === 1), ok = fav && won && fav.Driver === won.Driver;
    return h("button", { type: "button", class: "call", onclick: (() => { state.raceIdx = state.data.races.indexOf(r); if (!nerd()) setMode("nerd"); else rerender(); setTimeout(() => $("#race-explorer")?.scrollIntoView({ behavior: "smooth" }), 60); }) },
      h("span", { class: `ok ${ok ? "y" : "n"}`, html: ok ? ICON.check : ICON.cross, "aria-label": ok ? "called it" : "missed" }), flag(r.country, "sm") || h("span"),
      h("span", {}, h("span", { class: "t", style: "display:block" }, `R${r.round} ${r.location}`), h("span", { class: "s", style: "display:block" }, `Tipped ${fav?.Driver} (${pct(fav?.win)}), won by ${won?.Driver ?? "?"} (${pct(won?.win)} from P${won?.actual_grid ?? "?"})`)),
      won ? avatar(won.Driver, 32, won.Team) : h("span"));
  })));
  const labels = E.per_race.map((r) => "R" + r.round);
  root.append(h("div", { class: "grid g-main mt" },
    panel(nerd() ? "Ranked probability score per race" : "Error per race", nerd() ? "mean over drivers of Σ(CDF_pred − CDF_actual)² / (N−1)" : "lower is better · our simulator vs trusting the starting grid",
      h("div", { html: lineChart({ labels, yLabel: "RPS", lowerBetter: true, series: [
        { name: "Simulator", color: "var(--primary-hi)", values: E.per_race.map((r) => r.model_rps), width: 3 },
        { name: "Grid baseline", color: "#9aa3b0", values: E.per_race.map((r) => r.grid_rps), dash: "6 5" },
        { name: "Pace-only", color: "var(--data)", values: E.per_race.map((r) => r.pace_rps), dash: "2 4", opacity: .85 }] }) }),
      h("div", { class: "legend" }, h("span", {}, h("i", { style: "background:var(--primary-hi)" }), "simulator (solid)"), h("span", {}, h("i", { style: "background:#9aa3b0" }), "grid baseline (dashed)"), h("span", {}, h("i", { style: "background:var(--data)" }), "pace-only (dotted)"))),
    panel("Did we call the winner?", "newest first", calls)));
  if (!nerd()) return;
  const keys = [["log_loss", "Log loss", true], ["rps", "RPS", true], ["brier_win", "Brier · win", true], ["brier_podium", "Brier · podium", true], ["brier_points", "Brier · points", true], ["mae_pos", "MAE position", true], ["spearman", "Spearman ρ", false], ["p_winner", "P(actual winner)", false], ["top3_hit", "Top-3 hit", false], ["top10_hit", "Top-10 hit", false], ["winner_correct", "Favourite won", false]];
  const mt = h("table", {}, h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Metric"), ["Simulator", "Pace-only", "Grid"].map((x) => h("th", { class: "num", scope: "col" }, x)))),
    h("tbody", {}, keys.map(([k, l, low]) => { const vals = ["model", "pace", "grid"].map((s) => a[s]?.[k]), best = low ? Math.min(...vals) : Math.max(...vals);
      return h("tr", {}, h("td", {}, l, h("span", { class: "muted" }, low ? " ↓" : " ↑")), vals.map((v) => h("td", { class: `num mono ${v === best ? "best" : ""}` }, fx(v, 4)))); })));
  const st = E.strategy, rel = state.data.reliability || {};
  root.append(h("div", { class: "grid g-main mt" },
    panel("Season scorecard", `${E.n} races · ${MODE_LABEL[mode]} · best per row in green`, h("div", { class: "table-wrap" }, mt)),
    panel("Strategy scorecard", "predicted vs executed", h("div", { class: "grid g-2" },
      stat(pct(st.stops_acc), "Stop count", "modal predicted = actual", "cyan"), stat(fx(st.stops_ll, 2), "Stop log loss", "uniform guess = 1.10", ""),
      stat(pct(st.seq_acc), "Exact sequence", "e.g. M-H-H predicted and run", ""), stat(`${fx(st.first_stop_mae, 1)}<small>laps</small>`, "First-stop error", `inside predicted window ${pct(st.first_stop_iqr_cover)}`, "")))));
  root.append(h("div", { class: "mt" }, panel("Calibration", "when we say X%, does it happen X% of the time? · dot size = number of predictions",
    h("div", { class: "grid g-3" }, [["win", "WIN", "var(--gold)"], ["podium", "PODIUM", "var(--primary-hi)"], ["points", "POINTS", "var(--data)"]].map(([k, t, col]) => h("div", { html: reliability(rel[k], col, t) }))))));
  root.append(h("div", { class: "mt", id: "race-explorer" }, raceExplorer(races, tag)));
}
function raceExplorer(races, tag) {
  if (!races.length) return h("div");
  const r = state.raceIdx != null && state.data.races[state.raceIdx]?.predictions[tag] ? state.data.races[state.raceIdx] : races[races.length - 1];
  const P = r.predictions[tag], D = P.drivers.filter((d) => d.actual != null), m = P.metrics;
  const sel = h("select", { "aria-label": "Choose race", onchange: (e) => { state.raceIdx = state.data.races.indexOf(races[+e.target.value]); rerender(); } },
    races.map((x, i) => h("option", { value: i, selected: x === r ? "selected" : null }, `R${x.round} ${x.location}`)));
  const cols = [
    { id: "d", label: "Driver", render: (d) => drvCell(d.Driver, d.Team, SHORT[d.Team] || d.Team, 30) }, { id: "g", label: "Grid", num: true, render: (d) => d.actual_grid },
    { id: "e", label: "Pred", num: true, render: (d) => "P" + fx(d.exp_pos, 1) }, { id: "a", label: "Actual", num: true, render: (d) => (d.actual_dnf ? h("span", { class: "badge bad" }, "DNF") : "P" + d.actual) },
    { id: "err", label: "Error", num: true, render: (d) => { const e = d.actual - d.exp_pos; return h("span", { class: `badge ${Math.abs(e) <= 2 ? "good" : Math.abs(e) >= 6 ? "bad" : "neutral"}` }, sgn(e, 1)); } },
    { id: "w", label: "Win %", num: true, render: (d) => pct(d.win) }, { id: "pp", label: "Pace pred / act", num: true, render: (d) => `${sgn(d.pace_pct, 2)} / ${sgn(d.actual_pace, 2)}` },
    { id: "sp", label: "Strategy pred", render: (d) => tyres(d.strategy) }, { id: "sa", label: "Strategy actual", render: (d) => tyres(d.actual_strategy) },
    { id: "fs", label: "1st stop pred/act", num: true, render: (d) => `L${fx(d.first_stop_p50, 0)} / ${d.actual_first_stop != null ? "L" + d.actual_first_stop : "-"}` },
  ];
  return panel("Race explorer", "any past race: prediction vs what happened",
    h("div", { style: "display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-bottom:12px" }, flag(r.country), sel,
      h("span", { class: "chip" }, `RPS ${fx(m.model_rps, 4)} vs grid ${fx(m.grid_rps, 4)}`), h("span", { class: "chip" }, `log loss ${fx(m.model_log_loss, 3)}`),
      h("span", { class: "chip" }, `ρ ${fx(m.model_spearman, 2)}`), h("span", { class: "chip" }, `stop count ${pct(m.strat_stops_acc)}`)),
    h("div", { class: "grid rx" },
      h("div", { html: scatter({ points: D.map((d) => ({ x: d.exp_pos, y: d.actual, color: teamColor(d.Team), label: d.Driver, ring: d.actual_dnf, tip: `${d.Driver}: predicted P${fx(d.exp_pos, 1)} → P${d.actual}${d.actual_dnf ? " (DNF)" : ""} · win ${pct(d.win)}` })), max: D.length, xLabel: "predicted average finish", yLabel: "actual finish" }) }),
      h("p", { class: "explain" }, h("b", {}, "Reading the chart: "), "each dot is a driver. On the dashed diagonal = predicted exactly. Above it = finished worse than expected, below = better. Hollow rings are retirements, usually the biggest misses.")),
    h("div", { class: "mt" }, table(D.slice().sort((a, b) => a.actual - b.actual), cols)));
}

/* ------------------------------------------------------------------ LAB */
const PARAM_DOC = {
  pass_thr: "s/lap advantage for a 50% pass at an average track", pass_scale: "softness of the pass curve", min_gap: "s a held-up car sits behind",
  start_gap: "s between grid slots", start_sd: "launch randomness (s)", lap1_thr_mult: "lap-1 pass threshold multiplier", pace_sd_mult: "scales model pace σ",
  lap_sd_mult: "scales per-lap noise", sc_gap: "s between cars behind the SC", sc_lap_factor: "SC lap-time multiplier", vsc_lap_factor: "VSC lap-time multiplier",
  sc_pit_factor: "pit loss under SC (× green)", vsc_pit_factor: "pit loss under VSC (× green)", sc_from_dnf: "P(retirement triggers SC/VSC)",
  opp_window: "pit under SC if planned stop is within this share of stint", extra_stop_p: "P(free extra stop under SC)", slow_stop_p: "P(slow pit stop)",
  slow_stop_mean: "mean extra time of a slow stop (s)", lap1_dnf_mult: "lap-1 retirement hazard multiplier", pace_df: "Student-t tail weight of race-day pace", grid_blend: "stacking weight of the grid→finish prior",
};
function renderLab(root) {
  const M = state.data.model || {}, pm = M.pace_models_meta || {}, sp = M.sim_params || {};
  root.append(sectionHead(["Under ", em("the hood")], "Model internals, simulator parameters and the circuit file for the next race."),
    panel("Pipeline", "python -m flatout weekend runs all of it", h("div", { class: "flow" },
      [["SYNC", "FastF1 → parquet"], ["AUDIT", "every GP checked"], ["ANALYSE", "per-race pace regression"], ["CIRCUITS", "deg · pit loss · SC · passing"], ["FEATURES", "pre-race only"], ["TRAIN", "LightGBM + ridge"], ["SIMULATE", "4M races, lap by lap"], ["EVALUATE", "proper scores vs baselines"], ["CALIBRATE", "tune sim behaviour"]]
        .flatMap(([a, b], i) => [i ? h("span", { class: "arr", "aria-hidden": "true" }, "→") : null, h("div", { class: "step" }, h("b", {}, a), b)]))));
  const scp = scenarioPanel(state.data.next), rep = reproPanel(state.data.next);
  if (scp || rep) root.append(h("div", { class: "grid g-main mt" }, scp || h("div"), rep || h("div")));
  const imp = (M.importance || []).slice(0, 12), imax = Math.max(...imp.map((x) => x.gain), 1);
  root.append(h("div", { class: "grid g-3 mt" },
    panel("Pace uncertainty", "walk-forward robust σ, % of lap", h("div", { class: "stack", style: "gap:10px" },
      Object.entries(pm.sd?.race || {}).map(([k, v]) => h("div", {}, h("div", { class: "sub" }, { quali: "after qualifying", practice: "after practice", none: "before the weekend" }[k] || k), pbar(v, "var(--data)", fx(v, 3) + " %")))),
      h("dl", { class: "kv mt" }, h("dt", {}, "race CV MAE"), h("dd", {}, fx(pm.cv?.race?.mae, 3)), h("dt", {}, "race CV RMSE"), h("dd", {}, fx(pm.cv?.race?.rmse, 3)),
        h("dt", {}, "quali CV MAE"), h("dd", {}, fx(pm.cv?.quali?.mae, 3)), h("dt", {}, "GBM share (race / quali)"), h("dd", {}, `${fx(pm.blend?.race, 1)} / ${fx(pm.blend?.quali, 1)}`))),
    panel("What drives the pace model", "LightGBM split gain", h("div", { class: "stack", style: "gap:6px" }, imp.map((x) => h("div", { style: "display:grid;grid-template-columns:150px 1fr;gap:10px;align-items:center" },
      h("span", { class: "mono muted" }, x.feature), pbar(x.gain / imax, "var(--primary)", fx(x.gain, 0)))))),
    panel("Simulator parameters", `calibrated ${sp.calibrated || "-"} · RPS ${fx(sp.rps, 4)}`, h("div", { class: "table-wrap" }, h("table", {}, h("tbody", {},
      Object.entries(sp.params || {}).map(([k, v]) => h("tr", {}, h("td", { class: "mono" }, k), h("td", { class: "num mono" }, fx(v, 3)), h("td", { class: "muted", style: "white-space:normal;font-size:13px" }, PARAM_DOC[k] || "")))))))));
  const ch = M.calibration_history || [], c = state.data.next?.circuit || {};
  root.append(h("div", { class: "grid g-2 mt" },
    panel("Calibration search", "lower RPS is better", ch.length ? h("div", { html: lineChart({ labels: ch.map((x, i) => (i % 3 === 0 ? String(i) : "")), yLabel: "RPS", lowerBetter: true, yFmt: (v) => fx(v, 4), series: [{ name: "RPS", color: "var(--data)", values: ch.map((x) => x.rps) }] }) }) : h("p", { class: "sub" }, "run python -m flatout calibrate")),
    panel(`Circuit file · ${c.location || ""}`, `${c.n_races ?? 0} past races here`, h("dl", { class: "kv" },
      ...[["laps", c.n_laps], ["base lap (s)", fx(c.base_lap, 2)], ["pit loss (s)", fx(c.pit_loss, 2)], ["fuel (s/lap)", fx(c.fuel, 4)], ["lap noise σ (s)", fx(c.lap_sd, 3)],
        ["SC+red / race", fx(c.sc_per_race, 2)], ["VSC / race", fx(c.vsc_per_race, 2)], ["red share of SC", pct(c.red_share)], ["overtake factor", fx(c.overtake_factor, 3)],
        ["DNF rate / car", pct(c.dnf_rate)], ["max stint S/M/H", c.max_stint ? `${c.max_stint.SOFT}/${c.max_stint.MEDIUM}/${c.max_stint.HARD}` : "-"]].flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v ?? "-")])))));
  root.append(h("div", { class: "mt" }, panel("Glossary", null, h("div", { class: "glossary" }, [
    ["Log loss", "−mean log P(actual position). Punishes confident misses hard. Lower is better."],
    ["RPS", "Ranked probability score: compares predicted and actual cumulative finishing distributions, so near-misses earn credit. 0 = perfect."],
    ["Brier", "Mean squared error of a yes/no probability (win, podium, points)."], ["Spearman ρ", "Rank correlation between expected and actual finishing order."],
    ["Walk-forward", "Each race is predicted by a model retrained only on earlier races."], ["Grid baseline", "Finish distribution given starting slot, learned from history."],
    ["Pace-only baseline", "Order drivers by predicted pace plus noise; no strategy, traffic, SC or reliability."], ["Car effect", "Team pace each race with both drivers' skill removed."],
    ["Driver vs car", "Finish vs the slot a car's pace is worth among that race's finishers; zero-sum per race."],
  ].map(([k, v]) => h("div", {}, h("b", {}, k), h("span", {}, v)))))));
}

/* ------------------------------------------------------------------ shell */
const RENDER = { race: renderRace, strategy: renderStrategy, drivers: renderDrivers, dvc: renderDvC, teams: renderTeams, season: renderSeason, accuracy: renderAccuracy, lab: renderLab };
const revealer = "IntersectionObserver" in window
  ? new IntersectionObserver((es) => es.forEach((e) => { if (e.isIntersecting) { e.target.classList.add("in"); revealer.unobserve(e.target); } }), { rootMargin: "0px 0px -8% 0px" })
  : null;
function armReveals(root) {
  if (!revealer || reduced()) return;
  root.querySelectorAll(".panel, .teamcard, .stat, .console, .podium, .awards").forEach((el) => { el.classList.add("reveal"); revealer.observe(el); });
}
function rerender() {
  const y = window.scrollY;
  for (const [k, fn] of Object.entries(RENDER)) {
    const root = $(`#view-${k}`); root.innerHTML = "";
    if (k === state.tab) { try { fn(root); if (!window.gsap) armReveals(root); document.dispatchEvent(new CustomEvent("view:render", { detail: { tab: k, root, mode: state.mode } })); } catch (e) { console.error(e); root.append(h("div", { class: "empty" }, "Could not render this view. Reload the page; if it persists, re-run python -m flatout export. (" + e.message + ")")); } }
  }
  window.scrollTo({ top: y, behavior: "instant" });
}
function route() {
  let tab = (location.hash || "#race").slice(1);
  if (!RENDER[tab] || (tab === "lab" && !nerd())) tab = "race";
  const changed = tab !== state.tab;
  const apply = () => {
    state.tab = tab;
    document.querySelectorAll(".view").forEach((v) => v.classList.toggle("active", v.dataset.view === tab));
    document.querySelectorAll("#tabs a").forEach((a) => { const on = a.dataset.tab === tab; a.classList.toggle("active", on); if (on) a.setAttribute("aria-current", "page"); else a.removeAttribute("aria-current"); });
    rerender();
    if (changed) window.scrollTo({ top: 0, behavior: "instant" });
  };
  if (changed && state.tab && document.startViewTransition && !reduced()) document.startViewTransition(apply); else apply();
}
function setMode(m) {
  state.mode = m; document.body.dataset.mode = m;
  document.querySelectorAll(".mode-switch button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === m)));
  try { localStorage.setItem("flatout-mode", m); } catch { /* storage unavailable */ }
  const url = new URL(location.href); url.searchParams.set("mode", m); history.replaceState(null, "", url);
  if (state.tab === "lab" && m === "fan") location.hash = "#race"; else rerender();
}
function bindTips() {
  const tip = $("#tip");
  const show = (t, x, y) => { tip.textContent = t.getAttribute("data-tip"); tip.classList.add("on"); place(x, y); };
  const place = (x, y) => { const w = tip.offsetWidth, ht = tip.offsetHeight; tip.style.left = Math.min(window.innerWidth - w - 8, x + 14) + "px"; tip.style.top = (y + ht + 20 > window.innerHeight ? y - ht - 12 : y + 16) + "px"; };
  document.addEventListener("mouseover", (e) => { const t = e.target.closest("[data-tip]"); if (!t) tip.classList.remove("on"); else show(t, e.clientX, e.clientY); });
  document.addEventListener("mousemove", (e) => { if (tip.classList.contains("on")) place(e.clientX, e.clientY); });
  document.addEventListener("focusin", (e) => { const t = e.target.closest?.("[data-tip]"); if (t) { const r = t.getBoundingClientRect(); show(t, r.left, r.bottom); } else tip.classList.remove("on"); });
}
async function init() {
  let saved = null;
  try { saved = localStorage.getItem("flatout-mode"); } catch { /* storage unavailable */ }
  document.querySelectorAll(".mode-switch button").forEach((b) => b.addEventListener("click", () => setMode(b.dataset.mode)));
  $(".skip")?.addEventListener("click", (e) => { e.preventDefault(); $("#app").focus(); });   // don't let #app hit the hash router
  $("#drawer-close").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
  bindTips();
  if (!window.gsap) document.body.classList.add("js-reveal");   // CSS reveals are only the fallback; GSAP owns motion when loaded
  let spot = 0;   // pointer spotlight: one rAF per frame, only for the panel under the cursor
  document.addEventListener("pointermove", (e) => {
    if (spot || e.pointerType !== "mouse") return;
    spot = requestAnimationFrame(() => {
      spot = 0;
      const p = e.target.closest?.(".panel");
      if (!p) return;
      const r = p.getBoundingClientRect();
      p.style.setProperty("--mx", `${e.clientX - r.left}px`); p.style.setProperty("--my", `${e.clientY - r.top}px`);
    });
  }, { passive: true });
  try {
    const res = await fetch("data/v3/site.json", { cache: "no-store" });
    if (!res.ok) throw new Error(res.status + " " + res.statusText);
    state.data = await res.json();
  } catch (e) {
    $("#loading").innerHTML = `<p>No data yet (${esc(e.message)}).<br>Run <code>python -m flatout export</code> from the project root, then reload.</p>`;
    return;
  }
  if (state.data.team_colors) TEAM = { ...TEAM, ...state.data.team_colors };
  $("#loading").remove();
  const urlMode = new URLSearchParams(location.search).get("mode");
  state.mode = (urlMode || saved) === "nerd" ? "nerd" : "fan";
  document.body.dataset.mode = state.mode;
  document.querySelectorAll(".mode-switch button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === state.mode)));
  $("#footer").append(h("span", {}, `Data refreshed ${new Date(state.data.generated).toLocaleString()} · season ${state.data.season} · photos & logos © Formula 1 / teams, flags flagcdn.com`),
    h("span", {}, "Refresh with ", h("code", {}, "python -m flatout weekend"), " · ", h("a", { href: "legacy/" }, "v2 dashboard")));
  window.addEventListener("hashchange", route);
  state.tab = null;
  route();
}
init();
/* verified venue under the title, and the provisional-forecast explanation */
function titleEl(event) {
  const m = /^(.*?)\s*Grand Prix\s*(.*)$/i.exec(event || "");
  if (!m) return h("h1", {}, event);
  return h("h1", {}, m[1], em("Grand Prix"), m[2] ? h("span", { class: "title-tail" }, m[2]) : null);
}
function venueEl(m) {
  const id = m.identity;
  if (!id || !id.circuit_name) return null;
  return h("div", { class: "venue" }, h("b", {}, id.circuit_name), ` \u00b7 ${id.host_country}`,
    id.status === "override" && nerd() ? h("span", { class: "venue-note", "data-tip": id.note }, " \u00b7 venue verified against formula1.com") : null);
}
function provisionalEl(m) {
  if (m.status !== "provisional") return null;
  const a = m.assumptions || [];
  if (!nerd()) return h("p", { class: "provisional" }, "First race at this circuit in our data, so tyre wear, pit-stop time, safety cars and overtaking use averages from other tracks. Treat these odds as rougher than usual.");
  return h("details", { class: "provisional" }, h("summary", {}, `Provisional forecast \u00b7 ${a.length} stated assumption${a.length === 1 ? "" : "s"}`),
    h("ul", {}, a.map((x) => h("li", {}, x))));
}
const avgOf = (rows, k) => { const v = rows.map((r) => r[k]).filter((x) => x != null && isFinite(x)); return v.length ? v.reduce((a, x) => a + x, 0) / v.length : null; };
/* Accuracy headline from the nested walk-forward benchmark (the only fully out-of-sample numbers) */
function benchmarkSection(root, B) {
  const modes = Object.keys(B.summary || {});
  const mode = modes.includes(state.benchMode) ? state.benchMode : (modes.includes("post_quali") ? "post_quali" : modes[0]);
  const S = B.summary[mode], rows = B.per_race.filter((r) => r.mode === mode);
  const vsGrid = S.model_minus_grid_rps, vsPace = S.model_minus_pace_rps;
  const ref = mode === "post_quali" ? vsGrid : vsPace, refName = mode === "post_quali" ? "the starting grid" : "a pace-only ranking";
  const rel = (p, base) => (p && base ? -p.mean / base : null);
  const ci = (p) => (p && p.lo != null ? `${sgn(p.lo, 4)} to ${sgn(p.hi, 4)}` : "n/a");
  root.append(sectionHead(["How good are ", em("the predictions")],
    "Every race below was forecast using only what was known before it: the pace model, circuit settings and simulator settings were all re-fitted on earlier races. These 2026 races were also studied while building the model, so this is development evidence; the live record starts at Sepang.",
    h("div", { class: "seg", role: "group", "aria-label": "Information available" }, modes.map((k) => h("button", { class: k === mode ? "on" : "", "aria-pressed": String(k === mode), onclick: () => { state.benchMode = k; rerender(); } }, MODE_LABEL[k] || k)))));
  const better = rows.filter((r) => r.model_rps < (mode === "post_quali" ? r.grid_rps : r.pace_rps)).length;
  root.append(h("div", { class: "grid g-4" },
    stat(`${better}<small>/${rows.length}</small>`, `Races better than ${mode === "post_quali" ? "the grid" : "pace-only"}`, `lower error than ${refName}`, "good"),
    stat(ref && ref.mean != null ? pct(rel(ref, mode === "post_quali" ? S.grid_rps : S.pace_rps)) : "n/a", "Less error on average", `95% range of the difference ${ci(ref)} (RPS)`, "cyan"),
    stat(pct(avgOf(rows, "model_p_winner")), "Chance we gave the winner", mode === "post_quali" ? `grid baseline ${pct(avgOf(rows, "grid_p_winner"))}` : "grid unknown at this point", "accent"),
    stat(`0<small>races</small>`, "Live record", "first archived forecast: Sepang, 4 Oct", "")));
  const labels = rows.map((r) => "R" + r.round);
  const series = [{ name: "Simulator", color: "var(--primary-hi)", values: rows.map((r) => r.model_rps), width: 3 },
    { name: "Pace-only", color: "var(--data)", values: rows.map((r) => r.pace_rps), dash: "2 4", opacity: .85 }];
  if (mode === "post_quali") series.splice(1, 0, { name: "Grid baseline", color: "#9aa3b0", values: rows.map((r) => r.grid_rps), dash: "6 5" });
  root.append(h("div", { class: "mt" }, panel(nerd() ? "Ranked probability score per race" : "Error per race",
    mode === "post_quali" ? "lower is better \u00b7 simulator vs the starting grid and a pace-only ranking" : "lower is better \u00b7 before the weekend the grid is unknown, so it is not a fair comparison",
    h("div", { html: lineChart({ labels, yLabel: "RPS", lowerBetter: true, series }) }),
    h("div", { class: "legend" }, series.map((s) => h("span", {}, h("i", { style: `background:${s.color}` }), s.name))))));
  if (!nerd()) return;
  const line = (name, p) => h("tr", {}, h("td", {}, name), h("td", { class: "num mono" }, p?.mean != null ? sgn(p.mean, 4) : "n/a"),
    h("td", { class: "num mono" }, ci(p)), h("td", { class: "num mono" }, p?.share_better != null ? pct(p.share_better) : "n/a"), h("td", { class: "num mono" }, p?.n ?? "n/a"));
  const t = h("table", {}, h("thead", {}, h("tr", {}, ["Comparison (model \u2212 baseline)", "Mean", "95% bootstrap range", "Races better", "Races"].map((x, i) => h("th", { class: i ? "num" : "", scope: "col" }, x)))),
    h("tbody", {}, mode === "post_quali" ? [line("RPS vs grid", S.model_minus_grid_rps), line("Log loss vs grid", S.model_minus_grid_log_loss)] : [],
      line("RPS vs pace-only", S.model_minus_pace_rps), line("Log loss vs pace-only", S.model_minus_pace_log_loss)));
  root.append(h("div", { class: "grid g-main mt" },
    panel("Paired comparison", `${rows.length} races \u00b7 negative = model better \u00b7 races are the independent unit`, h("div", { class: "table-wrap" }, t)),
    panel("How this was measured", B.id, h("ul", { class: "method" },
      h("li", {}, `Simulator settings re-tuned before each race on the ${B.config.calib_last_n} races before it (${B.config.calib_sims.toLocaleString()} simulations per race per trial).`),
      h("li", {}, `Each scored race simulated ${B.config.eval_sims.toLocaleString()} times per mode; Monte Carlo error is far below the differences shown.`),
      h("li", {}, "Grid baseline: actual grid spread with a grid-to-finish table learned from earlier races. Only used after qualifying."),
      h("li", {}, "Note: before 27 Sep 2026 this page quoted a backtest whose simulator settings had been tuned on the same races (in-sample). Re-measured properly (above), the result is similar, but these numbers replace it.")))));
}
/* behaviour forecast vs model recommendation (static plans, common random numbers) */
function strategyRecoPanel(nx) {
  const R = nx.recommendations;
  if (!R || !R.drivers) return null;
  const rows = Object.entries(R.drivers).map(([code, r]) => {
    const d = nx.drivers.find((x) => x.Driver === code) || {};
    const best = r.plans.find((p) => p.plan === r.best) || {};
    return { code, team: d.Team, likely: d.strategy, best: r.best, gain: r.gain_pos, bestPod: best.podium, se: best.se_pos };
  });
  const t = h("table", {}, h("thead", {}, h("tr", {}, ["Driver", "Likely plan (behaviour)", "Best plan under the model", "Gain vs likely mix"].map((x, i) => h("th", { class: i === 3 ? "num" : "", scope: "col" }, x)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, h("td", {}, drvCell(r.code, r.team, SHORT[r.team] || r.team, 28)), h("td", {}, tyres(r.likely)), h("td", {}, tyres(r.best)),
      h("td", { class: "num mono", "data-tip": `Monte Carlo s.e. ~${fx(r.se, 3)} places` }, `${sgn(r.gain, 2)} places`)))));
  return panel(nerd() ? "Behaviour forecast vs recommendation" : "What teams will likely do vs what the model rates best",
    nerd() ? R.scope : "Each plan was simulated against the same race conditions. This compares plans inside the model, not in the real world.",
    h("div", { class: "table-wrap" }, t),
    h("p", { class: "note" }, nx.meta.status === "provisional"
      ? "Provisional: tyre wear and pit-lane loss here come from other circuits, so these plan rankings could change once practice data exists."
      : "Plans are fixed before the start; reactions to safety cars use the same rules for every plan."));
}
/* Lab: precomputed what-if scenarios (static site, no live recomputation) */
function scenarioPanel(nx) {
  const S = nx?.scenarios;
  if (!S || !S.scenarios?.length) return null;
  const base = S.scenarios[0], fav = base.drivers.slice(0, 5).map((d) => d.Driver);
  const val = (sc, code) => (sc.drivers.find((d) => d.Driver === code) || {}).win;
  const t = h("table", {}, h("thead", {}, h("tr", {}, h("th", { scope: "col" }, "Scenario"), fav.map((c) => h("th", { class: "num", scope: "col" }, c)), h("th", { class: "num", scope: "col" }, "P(SC)"))),
    h("tbody", {}, S.scenarios.map((sc) => h("tr", {}, h("td", {}, sc.label),
      fav.map((c) => { const v = val(sc, c), b = val(base, c), dlt = v != null && b != null ? v - b : null;
        return h("td", { class: "num mono" }, v == null ? "\u2013" : pct(v), sc === base || dlt == null ? null : h("small", { class: dlt > 0 ? "up" : "down" }, ` ${dlt > 0 ? "+" : ""}${(dlt * 100).toFixed(1)}`)); }),
      h("td", { class: "num mono" }, pct(sc.p_sc))))));
  return panel("What if?", `win chance of today's top five \u00b7 ${S.n_sims.toLocaleString()} races per scenario, same random numbers`,
    h("div", { class: "table-wrap" }, t),
    h("p", { class: "note" }, S.note + " " + (S.not_available || []).join(" ")));
}
function reproPanel(nx) {
  const P = nx?.provenance, m = nx?.meta;
  if (!P) return null;
  const short = (x) => (x ? String(x).slice(0, 10) : "–");
  const rows = [["Forecast run", m.run || m.created_utc], ["Mode", MODE_LABEL[m.mode] || m.mode], ["Issued (UTC)", m.created_utc],
    ["Code", `${short(P.code?.git_rev)}${P.code?.dirty ? " + uncommitted changes " + short(P.code?.dirty_diff_sha256) : ""}`],
    ["Data store", `${P.data?.store_files ?? "?"} files · ${short(P.data?.store_sha256)}`], ["Pace model", short(P.model_meta_sha256)],
    ["Sim settings", short(P.sim_params_sha256)], ["Seed", P.seed ? `${P.seed.root ?? P.seed.base} · ${P.seed.scheme}` : "–"],
    ["Simulations", Number(m.n_sims).toLocaleString()], ["Inputs missing", (m.inputs_missing || []).join(", ") || "none"]];
  return panel("Reproducibility", "everything needed to re-run this exact forecast",
    h("table", { class: "kv" }, h("tbody", {}, rows.map(([k, v]) => h("tr", {}, h("th", { scope: "row" }, k), h("td", { class: "mono" }, v))))));
}
