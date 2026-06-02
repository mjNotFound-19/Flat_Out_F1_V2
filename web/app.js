const DATA_PATHS = {
  simple: "./data/v2_prediction_results.csv",
  full: "./data/v2_full_predictions.csv",
  features: "./data/v2_feature_importance.csv"
};

const TEAM_COLORS = {
  "Red Bull Racing": "#1e41ff",
  "Ferrari": "#e10600",
  "Mercedes": "#00d2be",
  "McLaren": "#ff8700",
  "Haas F1 Team": "#b6babd",
  "Racing Bulls": "#00293f",
  "Alpine": "#0090d0",
  "Williams": "#00a3e0",
  "Aston Martin": "#2d826d",
  "Audi": "#9b9b9b",
  "Cadillac": "#2b2e3b"
};

const state = {
  mode: "simple",
  predictions: [],
  full: [],
  features: [],
  fullMap: {},
  search: "",
  sortKey: "score",
  selectedDriver: null
};

const elements = {
  simpleView: document.getElementById("simple-view"),
  complexView: document.getElementById("complex-view"),
  simpleList: document.getElementById("simple-list"),
  driverGrid: document.getElementById("driver-grid"),
  driverDetail: document.getElementById("driver-detail"),
  featureBars: document.getElementById("feature-bars"),
  heroStats: document.getElementById("hero-stats"),
  modeToggle: document.getElementById("mode-toggle"),
  search: document.getElementById("driver-search"),
  sortSelect: document.getElementById("sort-select"),
  toast: document.getElementById("toast")
};

init().catch((err) => showToast(`Load failed: ${err.message}`));

async function init() {
  bindModeToggle();
  bindControls();

  const [simpleRows, fullRows, featureRows] = await Promise.all([
    loadCSV(DATA_PATHS.simple),
    loadCSV(DATA_PATHS.full),
    loadCSV(DATA_PATHS.features)
  ]);

  state.predictions = parsePredictions(simpleRows).sort((a, b) => a.pos - b.pos);
  state.full = parseFull(fullRows);
  state.fullMap = Object.fromEntries(state.full.map((r) => [r.driver, r]));
  state.features = parseFeatures(featureRows);
  state.selectedDriver = state.predictions[0]?.driver || null;

  renderHero();
  renderSimple();
  renderComplex();
}

// Data loading and parsing
async function loadCSV(path) {
  const res = await fetch(path);
  if (!res.ok) throw new Error(`${path} (${res.status})`);
  const text = await res.text();
  return parseCSV(text);
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  if (!lines.length) return [];
  const headers = lines[0]
    .split(",")
    .map((h, i) => (h.trim() === "" && i === 0 ? "Driver" : h.trim()));
  return lines
    .slice(1)
    .filter((l) => l.trim().length)
    .map((line) => {
      const parts = line.split(",");
      const row = {};
      headers.forEach((h, i) => (row[h] = parts[i]?.trim() ?? ""));
      return row;
    });
}

function parsePredictions(rows) {
  return rows.map((r) => ({
    driver: r.Driver,
    name: r.Name,
    team: r.Team,
    pos: num(r.Pos),
    win: num(r.win_pct),
    podium: num(r.pod_pct),
    avgPoints: num(r.avg_pts),
    avgFinish: num(r.avg_fin),
    p10: num(r.p10),
    p50: num(r.p50),
    p90: num(r.p90),
    score: num(r.score),
    flag: r.fm
  }));
}

function parseFull(rows) {
  return rows.map((r) => {
    const distribution = [];
    for (let i = 1; i <= 22; i++) {
      const key = `P${i}_pct`;
      if (r[key] !== undefined) {
        distribution.push({ pos: i, pct: num(r[key]) });
      }
    }
    return {
      driver: r.Driver,
      name: r.Name,
      team: r.Team,
      pos: num(r.Pos),
      win: num(r.win_pct),
      podium: num(r.pod_pct),
      avgPoints: num(r.avg_pts),
      avgFinish: num(r.avg_fin),
      p10: num(r.p10),
      p50: num(r.p50),
      p90: num(r.p90),
      score: num(r.score),
      distribution
    };
  });
}

function parseFeatures(rows) {
  return rows
    .map((r) => ({ feature: r.feature, importance: num(r.importance) }))
    .sort((a, b) => b.importance - a.importance);
}

// Rendering
function renderHero() {
  if (!state.predictions.length) return;
  const leader = state.predictions[0];
  const avgWin = avg(state.predictions.map((d) => d.win));
  const avgPod = avg(state.predictions.map((d) => d.podium));
  const medianFinish = avg(state.predictions.map((d) => d.p50));

  elements.heroStats.innerHTML = `
    ${statCard("Leader", `${leader.name}`, `${leader.team}`)}
    ${statCard("Win chance avg", formatPct(avgWin))}
    ${statCard("Podium chance avg", formatPct(avgPod))}
    ${statCard("Median finish (grid-wide)", `P${medianFinish.toFixed(1)}`)}
  `;
}

function renderSimple() {
  const list = state.predictions
    .map(
      (d) => `
      <div class="simple-row">
        <div class="rank">${d.pos}</div>
        <div>
          <div class="driver-name">${d.name}</div>
          <div class="team">${d.team}</div>
        </div>
        <div class="metric-group">
          <span class="chip">Win ${formatPct(d.win)}</span>
          <span class="chip soft">Podium ${formatPct(d.podium)}</span>
          <span class="chip ghost">Median P${d.p50}</span>
        </div>
      </div>`
    )
    .join("");
  elements.simpleList.innerHTML = list;
}

function renderComplex() {
  renderDriverGrid();
  renderDriverDetail();
  renderFeatureBars();
}

function renderDriverGrid() {
  const q = state.search.toLowerCase();
  const filtered = state.predictions.filter(
    (d) =>
      d.name.toLowerCase().includes(q) ||
      d.team.toLowerCase().includes(q) ||
      d.driver.toLowerCase().includes(q)
  );

  const sorted = filtered.sort((a, b) => compareDrivers(a, b, state.sortKey));

  elements.driverGrid.innerHTML = sorted
    .map((d) => driverCard(d, state.fullMap[d.driver]))
    .join("");

  elements.driverGrid.querySelectorAll(".driver-card").forEach((card) => {
    card.addEventListener("click", () => {
      state.selectedDriver = card.dataset.driver;
      renderDriverDetail();
      highlightSelected();
    });
  });

  highlightSelected();
}

function renderDriverDetail() {
  const code = state.selectedDriver;
  if (!code || !state.fullMap[code]) {
    elements.driverDetail.innerHTML = `
      <div class="panel-title">
        <p class="eyebrow">Driver detail</p>
        <h3>Select a driver to see their distribution</h3>
      </div>`;
    return;
  }

  const d = state.fullMap[code];
  const top3 = d.distribution.slice(0, 3).reduce((s, p) => s + p.pct, 0);
  const top10 = d.distribution.slice(0, 10).reduce((s, p) => s + p.pct, 0);
  const bestBar = Math.max(...d.distribution.map((p) => p.pct), 1);
  const barHtml = d.distribution
    .map(
      (p) =>
        `<div class="bar" style="height:${(p.pct / bestBar) * 100}%" title="P${p.pos}: ${formatPct(p.pct)}"></div>`
    )
    .join("");

  elements.driverDetail.innerHTML = `
    <div class="detail-header">
      <div>
        <p class="eyebrow">Driver detail</p>
        <h3>${d.name} · ${d.team}</h3>
      </div>
      <div class="detail-meta">
        <span class="chip">Win ${formatPct(d.win)}</span>
        <span class="chip">Podium ${formatPct(d.podium)}</span>
        <span class="chip ghost">Median P${d.p50}</span>
      </div>
    </div>
    <div class="bar-chart">${barHtml}</div>
    <div class="bar-labels"><span>P1</span><span>P22</span></div>
    <div class="metric-group" style="margin-top:8px;">
      <span class="chip soft">Top 3: ${formatPct(top3)}</span>
      <span class="chip soft">Top 10: ${formatPct(top10)}</span>
      <span class="chip ghost">Range P${d.p10} — P${d.p90}</span>
      <span class="chip ghost">Expected pts: ${d.avgPoints.toFixed(1)}</span>
    </div>
  `;
}

function renderFeatureBars() {
  if (!state.features.length) return;
  const top = state.features.slice(0, 10);
  const maxImp = Math.max(...top.map((f) => f.importance), 1);
  elements.featureBars.innerHTML = top
    .map(
      (f) => `
      <div class="feature-row">
        <div class="feature-name">${f.feature}</div>
        <div class="feature-track">
          <div class="feature-fill" style="width:${(f.importance / maxImp) * 100}%"></div>
        </div>
        <div class="feature-value">${f.importance.toFixed(3)}</div>
      </div>`
    )
    .join("");
}

// UI helpers
function bindModeToggle() {
  elements.modeToggle.querySelectorAll(".mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      const mode = btn.dataset.mode;
      if (mode === state.mode) return;
      state.mode = mode;
      elements.modeToggle.querySelectorAll(".mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      toggleViews();
    });
  });
}

function bindControls() {
  elements.search.addEventListener("input", (e) => {
    state.search = e.target.value;
    renderDriverGrid();
  });
  elements.sortSelect.addEventListener("change", (e) => {
    state.sortKey = e.target.value;
    renderDriverGrid();
  });
}

function toggleViews() {
  const simpleActive = state.mode === "simple";
  elements.simpleView.classList.toggle("active", simpleActive);
  elements.complexView.classList.toggle("active", !simpleActive);
}

function driverCard(d, full) {
  if (!full) return "";
  const range = finishRange(full.p10, full.p50, full.p90);
  const color = TEAM_COLORS[d.team] || "#ffffff";
  return `
    <div class="driver-card ${state.selectedDriver === d.driver ? "active" : ""}" data-driver="${d.driver}">
      <div class="card-top">
        <div>
          <div class="badge" style="background:${color}22;border:1px solid ${color}55;">#${d.pos} · ${d.name}</div>
          <div class="team" style="color:${color}">${d.team}</div>
        </div>
        <button class="ghost-btn">Details</button>
      </div>
      <div class="card-metrics">
        ${metric("Win", formatPct(d.win))}
        ${metric("Podium", formatPct(d.podium))}
        ${metric("Median", `P${d.p50}`)}
        ${metric("Expected pts", d.avgPoints.toFixed(1))}
      </div>
      <div class="range">
        <div class="range-fill" style="left:${range.start}%;width:${range.width}%;"></div>
        <div class="range-marker" style="left:${range.marker}%;"></div>
      </div>
      <div class="range-labels">
        <span>P${d.p10}</span>
        <span>Median</span>
        <span>P${d.p90}</span>
      </div>
    </div>`;
}

function finishRange(p10, p50, p90) {
  const maxPos = 22;
  const base = 100 / (maxPos - 1);
  return {
    start: clamp((p10 - 1) * base, 0, 100),
    width: clamp((p90 - p10) * base, 6, 100),
    marker: clamp((p50 - 1) * base, 0, 100)
  };
}

function metric(label, value) {
  return `<div class="metric"><span class="label">${label}</span><span class="value">${value}</span></div>`;
}

function statCard(label, value, footer = "") {
  return `<div class="stat-card"><div class="label">${label}</div><div class="value">${value}</div><div class="team">${footer}</div></div>`;
}

function compareDrivers(a, b, key) {
  if (key === "win") return b.win - a.win;
  if (key === "podium") return b.podium - a.podium;
  if (key === "median") return a.p50 - b.p50;
  return b.score - a.score;
}

function highlightSelected() {
  elements.driverGrid.querySelectorAll(".driver-card").forEach((card) => {
    card.classList.toggle("active", card.dataset.driver === state.selectedDriver);
  });
}

function showToast(msg, timeout = 3200) {
  elements.toast.textContent = msg;
  elements.toast.classList.remove("hidden");
  setTimeout(() => elements.toast.classList.add("hidden"), timeout);
}

// utils
function num(v) {
  const n = parseFloat(v);
  return Number.isFinite(n) ? n : 0;
}

function formatPct(v) {
  return `${num(v).toFixed(1)}%`;
}

function avg(arr) {
  if (!arr.length) return 0;
  return arr.reduce((s, n) => s + n, 0) / arr.length;
}

function clamp(v, min = 0, max = 100) {
  return Math.min(Math.max(v, min), max);
}
