// f1.h motion layer (GSAP 3.13 + ScrollTrigger + SplitText).
// app.js announces every render with a `view:render` event; this file choreographs it.
// Entrances only play when the tab or mode changes (not on sorts / selections), and nothing runs under
// prefers-reduced-motion. If GSAP fails to load, the CSS fallbacks in styles.css still apply.

const g = window.gsap;
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;

if (g && !REDUCED) {
  const { ScrollTrigger, SplitText } = window;
  g.registerPlugin(ScrollTrigger, SplitText);
  document.documentElement.classList.add("gsap-on");
  g.defaults({ ease: "expo.out", duration: 0.8 });

  let ctx = null;
  let last = "";
  let introDone = false;
  // request the display font explicitly; fonts.ready alone can resolve before Tektur is even requested
  const settled = () => new Promise((res) => {   // other faces (Fira Code) can re-enter "loading"; wait until the set is idle, max 2s
    const t0 = performance.now();
    (function poll() { if (document.fonts.status === "loaded" || performance.now() - t0 > 2000) res(); else setTimeout(poll, 50); })();
  });
  const fontsReady = document.fonts
    ? Promise.all([document.fonts.load("800 40px Tektur"), document.fonts.load("400 40px Tektur"), document.fonts.load("500 14px 'Fira Code'")]).catch(() => {}).then(settled)
    : Promise.resolve();
  const pending = [];

  /* ---------------------------------------------------------------- helpers */
  const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];
  const onEnter = (el, fn, start = "top 88%") => el && ScrollTrigger.create({ trigger: el, start, once: true, onEnter: () => fn(el) });

  function splitHeadline(h1, delay = 0) {
    if (!h1 || h1.dataset.split) return;
    h1.dataset.split = "1";
    // autoSplit re-splits (and re-runs the tween) if fonts finish loading after the first split
    SplitText.create(h1, { type: "words,chars", mask: "words", autoSplit: true,
      onSplit: (self) => g.from(self.chars, { yPercent: 110, rotateX: -50, opacity: 0, duration: 0.9, stagger: 0.022, delay }) });
  }
  function countUp(el) {
    const node = [...el.childNodes].find((n) => n.nodeType === 3 && /[\d.]+/.test(n.textContent));
    if (!node) return;
    const m = node.textContent.match(/^([^\d-+]*)([-+]?\d+(?:\.\d+)?)(.*)$/s);
    if (!m) return;
    const [, pre, num, post] = m, target = parseFloat(num), dp = (num.split(".")[1] || "").length;
    const o = { v: 0 };
    g.to(o, { v: target, duration: 1.3, ease: "power3.out", onUpdate: () => { node.textContent = `${pre}${o.v.toFixed(dp)}${post}`; } });
  }
  function drawPaths(root) {
    $$("svg.chart path[fill='none']", root).forEach((p) => {
      const len = p.getTotalLength?.();
      if (!len) return;
      g.set(p, { strokeDasharray: len, strokeDashoffset: len });
      onEnter(p.closest("svg"), () => g.to(p, { strokeDashoffset: 0, duration: 1.8, ease: "power2.inOut", onComplete: () => g.set(p, { clearProps: "strokeDasharray,strokeDashoffset" }) }));
    });
    $$("svg.chart circle", root).forEach((c) => g.set(c, { transformBox: "fill-box", transformOrigin: "center" }));
    $$("svg.chart", root).forEach((svg) => onEnter(svg, () => g.from(svg.querySelectorAll("circle"), { scale: 0, duration: 0.5, stagger: { each: 0.008, from: "start" }, delay: 0.6, ease: "back.out(2)" })));
  }
  function growBars(root, sel = ".pbar > i") {
    ScrollTrigger.batch($$(sel, root), { start: "top 92%", once: true,
      onEnter: (els) => g.fromTo(els, { scaleX: 0 }, { scaleX: 1, duration: 1.1, stagger: 0.03, transformOrigin: "left center", ease: "expo.out" }) });
  }
  function batchRise(root, sel, y = 16) {
    const els = $$(sel, root);
    g.set(els, { opacity: 0, y });
    ScrollTrigger.batch(els, { start: "top 94%", once: true, onEnter: (b) => g.to(b, { opacity: 1, y: 0, duration: 0.7, stagger: 0.035 }) });
  }
  function tilt(el, max = 7) {
    const rx = g.quickTo(el, "rotateX", { duration: 0.5, ease: "power3" }), ry = g.quickTo(el, "rotateY", { duration: 0.5, ease: "power3" });
    g.set(el, { transformPerspective: 900 });
    el.addEventListener("pointermove", (e) => { const r = el.getBoundingClientRect(); ry(((e.clientX - r.left) / r.width - 0.5) * max * 2); rx(-((e.clientY - r.top) / r.height - 0.5) * max * 2); });
    el.addEventListener("pointerleave", () => { rx(0); ry(0); });
  }
  function magnetic(el, strength = 0.25) {
    if (el.dataset.mag) return;
    el.dataset.mag = "1";
    const x = g.quickTo(el, "x", { duration: 0.4, ease: "power3" }), y = g.quickTo(el, "y", { duration: 0.4, ease: "power3" });
    el.addEventListener("pointermove", (e) => { const r = el.getBoundingClientRect(); x((e.clientX - r.left - r.width / 2) * strength); y((e.clientY - r.top - r.height / 2) * strength); });
    el.addEventListener("pointerleave", () => { x(0); y(0); });
  }

  /* ---------------------------------------------------------------- global, once */
  // scroll progress
  const bar = document.createElement("div");
  bar.className = "scroll-progress";
  bar.setAttribute("aria-hidden", "true");
  document.body.append(bar);
  g.to(bar, { scaleX: 1, ease: "none", scrollTrigger: { start: 0, end: "max", scrub: 0.25 } });
  $$(".mode-switch button, .brand").forEach((b) => magnetic(b, 0.18));

  // F1 start-lights intro: once per session, click/key to skip
  function lightsIntro() {
    let seen = false;
    try { seen = sessionStorage.getItem("f1h-lights") === "1"; sessionStorage.setItem("f1h-lights", "1"); } catch { /* storage unavailable */ }
    if (seen) { introDone = true; return; }
    const ov = document.createElement("div");
    ov.className = "lights-intro";
    ov.setAttribute("aria-hidden", "true");
    ov.innerHTML = `<div class="gantry">${"<span><i></i><i></i></span>".repeat(5)}</div><p>Lights out and away we go</p>`;
    document.body.append(ov);
    const lamps = ov.querySelectorAll("span");
    const tl = g.timeline({ onComplete: finish });
    tl.from(ov.querySelector(".gantry"), { y: -60, opacity: 0, duration: 0.5 });
    lamps.forEach((l) => tl.to(l, { "--on": 1, duration: 0.05, onStart: () => l.classList.add("on") }, "+=0.28"));
    tl.to({}, { duration: 0.55 });
    tl.add(() => lamps.forEach((l) => l.classList.remove("on")));
    tl.from(ov.querySelector("p"), { opacity: 0, y: 12, duration: 0.3 });
    tl.to(ov, { yPercent: -100, duration: 0.8, ease: "expo.inOut" }, "+=0.2");
    function finish() { ov.remove(); introDone = true; pending.splice(0).forEach((f) => f()); }
    const skip = () => { tl.progress(1); };
    ov.addEventListener("click", skip);
    addEventListener("keydown", skip, { once: true });
  }
  lightsIntro();

  // velocity-reactive marquee (speeds up and skews with scroll)
  function marquee(root) {
    const track = root.querySelector(".marquee-track");
    if (!track) return;
    const loop = g.to(track, { xPercent: -50, ease: "none", duration: 40, repeat: -1 });
    const skew = g.quickTo(track, "skewX", { duration: 0.4, ease: "power3" });
    ScrollTrigger.create({ trigger: track, start: "top bottom", end: "bottom top", onUpdate: (st) => {
      const v = st.getVelocity();
      g.to(loop, { timeScale: 1 + Math.min(Math.abs(v) / 250, 6) * Math.sign(v || 1), duration: 0.2, overwrite: true });
      skew(g.utils.clamp(-12, 12, v / -140));
    } });
    ScrollTrigger.addEventListener("scrollEnd", () => { g.to(loop, { timeScale: 1, duration: 0.8 }); skew(0); });
    track.parentElement.addEventListener("pointerenter", () => g.to(loop, { timeScale: 0.15, duration: 0.5 }));
    track.parentElement.addEventListener("pointerleave", () => g.to(loop, { timeScale: 1, duration: 0.5 }));
  }

  /* ---------------------------------------------------------------- per view */
  const VIEWS = {
    race(root) {
      const hero = root.querySelector(".hero");
      if (hero) {
        const tl = g.timeline({ delay: 0.05 });
        splitHeadline(hero.querySelector("h1"), 0.1);
        tl.from(hero.querySelector(".eyebrow"), { opacity: 0, x: -20, duration: 0.6 }, 0)
          .from(hero.querySelectorAll(".chip"), { opacity: 0, y: 14, stagger: 0.06, duration: 0.6 }, 0.35)
          .from(hero.querySelectorAll(".flipclock .flap"), { opacity: 0, rotateX: -90, transformPerspective: 500, transformOrigin: "50% 0%", stagger: 0.05, duration: 0.7, ease: "back.out(1.6)" }, 0.45)
          .from(hero.querySelector(".trackcard"), { opacity: 0, x: 30, duration: 0.8 }, 0.55)
          .from(hero.querySelector(".hero-note"), { opacity: 0, y: 12, duration: 0.6 }, 0.7)
          .from(hero.querySelector(".hero-portrait img"), { opacity: 0, y: 80, scale: 1.06, duration: 1.4, ease: "expo.out" }, 0.1)
          .from(hero.querySelector(".hero-portrait .tag"), { opacity: 0, y: 20, scale: 0.9, duration: 0.7, ease: "back.out(2)" }, 0.9)
          .from(hero.querySelector(".flag-wave"), { opacity: 0, xPercent: 12, duration: 1.6, ease: "power3.out" }, 0);
        // parallax: decorative layers only
        g.to(hero.querySelector(".flag-wave"), { yPercent: 14, ease: "none", scrollTrigger: { trigger: hero, start: "top top", end: "bottom top", scrub: true } });
        g.to(hero.querySelector(".hero-portrait img"), { yPercent: -7, ease: "none", scrollTrigger: { trigger: hero, start: "top top", end: "bottom top", scrub: true } });
      }
      marquee(root);
      // engineer console: keys wave in, LED chase, knobs spin to value
      const con = root.querySelector(".console");
      if (con) {
        const keys = con.querySelectorAll(".key");
        g.set(keys, { opacity: 0, y: 18 });
        onEnter(con, () => {
          const tl = g.timeline();
          tl.from(con, { y: 40, opacity: 0, duration: 0.9 })
            .to(keys, { opacity: 1, y: 0, duration: 0.6, stagger: { each: 0.025, grid: [2, 11], from: "center" }, ease: "back.out(1.8)" }, 0.2);
          keys.forEach((k, i) => tl.add(() => { k.classList.add("chase"); setTimeout(() => k.classList.remove("chase"), 160); }, 0.9 + i * 0.035));
          con.querySelectorAll(".knob").forEach((kn) => {
            const target = kn.style.getPropertyValue("--a") || "0deg";
            kn.style.setProperty("--a", "-135deg");
            tl.to(kn, { "--a": target, duration: 1.6, ease: "elastic.out(1, 0.55)" }, 0.5);
          });
          tl.from(con.querySelectorAll(".dial b"), { opacity: 0, y: 8, stagger: 0.1, duration: 0.5 }, 0.8);
        }, "top 85%");
      }
      // podium: rise with P1 last and highest, then 3D tilt
      const pods = root.querySelectorAll(".pod");
      if (pods.length) {
        const order = [...pods].sort((a, b) => (b.classList.contains("p1") ? -1 : 0) - (a.classList.contains("p1") ? -1 : 0));
        g.set(pods, { opacity: 0, y: 60, rotateX: 18, transformPerspective: 900 });
        onEnter(root.querySelector(".podium"), () => g.to(order.reverse(), { opacity: 1, y: 0, rotateX: 0, duration: 1.1, stagger: 0.14 }));
        pods.forEach((p) => tilt(p, 6));
      }
      root.querySelectorAll(".stat .v").forEach((v) => onEnter(v, countUp));
      root.querySelectorAll(".pod .big span").forEach((v) => onEnter(v, countUp));
      growBars(root);
      batchRise(root, "tbody tr", 12);
      // heatmap ripple (nerd)
      const cells = root.querySelectorAll(".heat .hc");
      if (cells.length) {
        g.set(cells, { opacity: 0, scale: 0.4 });
        const n = Math.round(Math.sqrt(cells.length));
        onEnter(root.querySelector(".heat"), () => g.to(cells, { opacity: 1, scale: 1, duration: 0.5, ease: "power3.out", stagger: { each: 0.004, grid: "auto", from: "start" } }));
      }
    },
    strategy(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      root.querySelectorAll(".stat .v").forEach((v) => onEnter(v, countUp));
      const lanes = root.querySelectorAll(".lane");
      lanes.forEach((lane, i) => {
        const st = lane.querySelectorAll(".lane-stint"), win = lane.querySelector(".lane-win");
        g.set(st, { scaleX: 0, transformOrigin: "left center" });
        if (win) g.set(win, { opacity: 0 });
        onEnter(lane, () => {
          g.to(st, { scaleX: 1, duration: 0.55, stagger: 0.28, ease: "power2.out", delay: (i % 6) * 0.04 });
          if (win) g.to(win, { opacity: 1, duration: 0.5, delay: 0.8 });
        }, "top 95%");
      });
      drawPaths(root);
      batchRise(root, "tbody tr", 10);
    },
    drivers(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      const f = root.querySelector(".award-feature");
      if (f) {
        g.from(f.querySelector("img"), { x: 80, opacity: 0, duration: 1.3, delay: 0.2 });
        g.from(f.querySelectorAll(".copy > span"), { y: 24, opacity: 0, stagger: 0.08, duration: 0.8, delay: 0.35 });
        g.to(f.querySelector("img"), { yPercent: -6, ease: "none", scrollTrigger: { trigger: f, start: "top bottom", end: "bottom top", scrub: true } });
      }
      batchRise(root, ".award-list .rowbtn", 14);
      // gallery: masked wipe-up with image settle
      const cards = root.querySelectorAll(".gcard");
      g.set(cards, { clipPath: "inset(100% 0% 0% 0%)" });
      ScrollTrigger.batch([...cards], { start: "top 95%", once: true, onEnter: (b) => {
        g.to(b, { clipPath: "inset(0% 0% 0% 0%)", duration: 1, stagger: 0.06, ease: "expo.inOut" });
        g.from(b.map((c) => c.querySelector("img")).filter(Boolean), { scale: 1.25, duration: 1.4, stagger: 0.06 });
      } });
      cards.forEach((c) => tilt(c, 5));
      batchRise(root, ".stack > .rowbtn", 10);
      growBars(root);
      drawPaths(root);
      batchRise(root, "tbody tr", 10);
    },
    dvc(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      const f = root.querySelector(".award-feature");
      if (f) g.from(f.querySelector("img"), { x: 80, opacity: 0, duration: 1.3, delay: 0.2 });
      // each driver slides from where the car should be to where they finished
      root.querySelectorAll(".dvc-row").forEach((row) => {
        const holder = row.children[1];
        if (!holder) return;
        const [, band, car, drv] = holder.children;
        if (!drv || !car) return;
        onEnter(row, () => {
          const dx = car.getBoundingClientRect().left - drv.getBoundingClientRect().left + 2;
          g.from(drv, { x: dx, duration: 1.2, ease: "expo.inOut", delay: 0.1 });
          g.from(car, { scale: 0, duration: 0.5, ease: "back.out(3)", transformOrigin: "center" });
          g.from(band, { scaleX: 0, transformOrigin: dx > 0 ? "right center" : "left center", duration: 1.2, ease: "expo.inOut", delay: 0.1 });
          g.from(row.querySelector(".dvc-val"), { opacity: 0, x: 10, duration: 0.5, delay: 0.8 });
        }, "top 96%");
      });
      drawPaths(root);
      batchRise(root, "tbody tr", 10);
    },
    teams(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      batchRise(root, ".teamcard", 20);
      ScrollTrigger.batch($$(".axisbars .b i", root), { start: "top 95%", once: true,
        onEnter: (b) => g.fromTo(b, { scaleX: 0 }, { scaleX: 1, transformOrigin: "left center", duration: 1.2, stagger: 0.02 }) });
      root.querySelectorAll(".teamcard .score b").forEach((v) => onEnter(v, countUp));
      const radar = root.querySelector("svg[aria-label='constructor comparison radar']");
      if (radar) {
        radar.querySelectorAll("polygon").forEach((p, i) => g.set(p, { transformBox: "view-box", transformOrigin: "50% 51%" }));
        onEnter(radar, () => {
          const polys = [...radar.querySelectorAll("polygon")];
          const grid = polys.filter((p) => p.getAttribute("fill") === "none"), teams = polys.filter((p) => p.getAttribute("fill") !== "none");
          g.from(grid, { scale: 0, opacity: 0, duration: 0.8, stagger: 0.08 });
          g.from(teams, { scale: 0, duration: 1.3, stagger: 0.15, ease: "elastic.out(1, 0.6)", delay: 0.3 });
          g.from(radar.querySelectorAll("circle"), { scale: 0, transformOrigin: "center", transformBox: "fill-box", duration: 0.4, stagger: 0.02, delay: 0.9, ease: "back.out(3)" });
        });
      }
      drawPaths(root);
      batchRise(root, "tbody tr", 10);
    },
    season(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      growBars(root);
      root.querySelectorAll(".strip").forEach((s) => {
        const sq = s.querySelectorAll("i");
        g.set(sq, { scale: 0 });
        onEnter(s, () => g.to(sq, { scale: 1, duration: 0.45, stagger: 0.03, ease: "back.out(2.5)" }), "top 96%");
      });
      batchRise(root, "tbody tr", 10);
    },
    accuracy(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      root.querySelectorAll(".stat .v").forEach((v) => onEnter(v, countUp));
      batchRise(root, ".call", 12);
      drawPaths(root);
      batchRise(root, "tbody tr", 10);
    },
    lab(root) {
      splitHeadline(root.querySelector(".section-head h1"));
      const steps = root.querySelectorAll(".flow .step, .flow .arr");
      g.from(steps, { opacity: 0, x: -16, stagger: 0.07, duration: 0.6, delay: 0.3 });
      growBars(root);
      drawPaths(root);
    },
  };

  function play(tab, root) {
    ctx?.revert();
    ctx = g.context(() => {
      (VIEWS[tab] || (() => {}))(root);
      root.querySelectorAll(".seg button, .c-btn, .pill-toggle").forEach((b) => magnetic(b, 0.15));
      g.from(root.querySelectorAll(".panel:not(.award-feature)"), { opacity: 0, y: 18, duration: 0.8, stagger: 0.06, clearProps: "transform" });
    }, root);
    requestAnimationFrame(() => ScrollTrigger.refresh());
  }

  document.addEventListener("view:render", (e) => {
    const { tab, root, mode } = e.detail;
    const key = `${tab}|${mode}`;
    if (key === last) return;           // same view re-rendered by a sort/selection: no entrance replay
    last = key;
    const run = () => fontsReady.then(() => { if (document.contains(root)) play(tab, root); });   // SplitText must measure the real font
    if (introDone) run(); else pending.push(run);
  });

  // driver drawer
  document.addEventListener("drawer:open", (e) => {
    const body = e.detail.body;
    const tl = g.timeline();
    tl.from(body.querySelector(".dhead > img"), { x: 120, opacity: 0, duration: 1 })
      .from(body.querySelectorAll(".dhead .info > *"), { y: 20, opacity: 0, stagger: 0.06, duration: 0.6 }, 0.1)
      .from(body.querySelectorAll(".dstats > div"), { y: 16, opacity: 0, stagger: 0.05, duration: 0.5 }, 0.2)
      .from(body.querySelectorAll(".dsec"), { y: 16, opacity: 0, stagger: 0.07, duration: 0.6 }, 0.3);
    const bars = body.querySelectorAll("svg.chart rect[data-tip]");
    tl.from(bars, { scaleY: 0, transformOrigin: "50% 100%", transformBox: "fill-box", stagger: 0.02, duration: 0.6, ease: "back.out(1.6)" }, 0.35);
    tl.from(body.querySelectorAll(".rbar i"), { scaleX: 0, transformOrigin: "left center", stagger: 0.05, duration: 0.8 }, 0.5);
  });
}
