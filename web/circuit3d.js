// f1.h 3D circuit (igloo.inc-inspired): the reference lap as a floating ribbon in haze, coloured by
// speed, lifted by (exaggerated) elevation, with a car lapping at its real telemetry speed.
// Loaded on demand by app.js; renders only while on screen. mount() throws if WebGL is unavailable,
// and the caller keeps the flat SVG map in that case.

const THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.min.js";
const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
const LIFT = 7;          // elevation exaggeration: real relief is a few metres over kilometres
const WIDTH = 17;        // ribbon width in track units (1000 = the circuit's longest side)

// slow -> fast: deep blue, ice, white, amber, red
const RAMP = [[0, [0.16, 0.32, 0.95]], [0.35, [0.45, 0.82, 1.0]], [0.6, [0.93, 0.96, 1.0]], [0.8, [1.0, 0.72, 0.25]], [1, [1.0, 0.18, 0.12]]];
export function speedColor(k) {
  k = Math.max(0, Math.min(1, k));
  for (let i = 1; i < RAMP.length; i++) {
    if (k <= RAMP[i][0]) {
      const [k0, a] = RAMP[i - 1], [k1, b] = RAMP[i], t = (k - k0) / (k1 - k0);
      return a.map((v, j) => v + (b[j] - v) * t);
    }
  }
  return RAMP[RAMP.length - 1][1];
}

export async function mount(host, track, { accent = "#e10600", onSpeed } = {}) {
  const probe = document.createElement("canvas");
  if (!(probe.getContext("webgl2") || probe.getContext("webgl"))) throw new Error("no webgl");
  const THREE = await import(THREE_URL);

  const W = track.w, H = track.h, n = track.points.length;
  // layout-only tracks (traced official map, no telemetry): flat, one colour, no car - nothing invented
  const tel = Array.isArray(track.speed) && Array.isArray(track.z);
  const vmin = tel ? Math.min(...track.speed) : 0, vmax = tel ? Math.max(...track.speed) : 1;
  const zAt = (i) => (tel ? track.z[i] : 0);
  // centre line in world units (1 unit = 1000 track units), y up
  const P = track.points.map(([x, y], i) => new THREE.Vector3((x - W / 2) / 1000, (zAt(i) * LIFT) / 1000, (y - H / 2) / 1000));

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.75));
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  host.prepend(renderer.domElement);
  renderer.domElement.setAttribute("aria-hidden", "true");

  const scene = new THREE.Scene();
  scene.fog = new THREE.FogExp2(0x0b0d11, 0.3);
  const camera = new THREE.PerspectiveCamera(34, 1, 0.01, 50);
  const root = new THREE.Group();
  scene.add(root);

  // ---- ribbon (speed-coloured) + edge lines + translucent curtain down to the ground
  const rib = new Float32Array(n * 2 * 3), ribCol = new Float32Array(n * 2 * 3);
  const edgeL = [], edgeR = [], curtain = new Float32Array(n * 2 * 3);
  const half = WIDTH / 2000;
  for (let i = 0; i < n; i++) {
    const a = P[(i - 1 + n) % n], b = P[(i + 1) % n];
    const t = new THREE.Vector3().subVectors(b, a).setY(0).normalize();
    const nrm = new THREE.Vector3(-t.z, 0, t.x);
    const l = P[i].clone().addScaledVector(nrm, half), r = P[i].clone().addScaledVector(nrm, -half);
    rib.set([l.x, l.y, l.z, r.x, r.y, r.z], i * 6);
    const c = tel ? speedColor((track.speed[i] - vmin) / (vmax - vmin)) : [0.86, 0.9, 0.97];
    ribCol.set([...c, ...c], i * 6);
    edgeL.push(l); edgeR.push(r);
    curtain.set([P[i].x, P[i].y, P[i].z, P[i].x, -0.02, P[i].z], i * 6);
  }
  const strip = (count) => { const idx = []; for (let i = 0; i < count; i++) { const j = (i + 1) % count; idx.push(i * 2, i * 2 + 1, j * 2, i * 2 + 1, j * 2 + 1, j * 2); } return idx; };
  const ribGeo = new THREE.BufferGeometry();
  ribGeo.setAttribute("position", new THREE.BufferAttribute(rib, 3));
  ribGeo.setAttribute("color", new THREE.BufferAttribute(ribCol, 3));
  ribGeo.setIndex(strip(n));
  root.add(new THREE.Mesh(ribGeo, new THREE.MeshBasicMaterial({ vertexColors: true, side: THREE.DoubleSide, transparent: true, opacity: 0.92 })));
  const lineMat = new THREE.LineBasicMaterial({ color: 0xeef3ff, transparent: true, opacity: 0.55 });
  root.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(edgeL), lineMat));
  root.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(edgeR), lineMat));
  const curGeo = new THREE.BufferGeometry();
  curGeo.setAttribute("position", new THREE.BufferAttribute(curtain, 3));
  curGeo.setIndex(strip(n));
  root.add(new THREE.Mesh(curGeo, new THREE.MeshBasicMaterial({ color: 0x9fb4d8, transparent: true, opacity: 0.07, side: THREE.DoubleSide, depthWrite: false })));
  // shadow of the circuit on the ground
  root.add(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(P.map((p) => new THREE.Vector3(p.x, -0.02, p.z))),
    new THREE.LineBasicMaterial({ color: 0x5b6b86, transparent: true, opacity: 0.35 })));

  // ---- ground grid + drifting dust
  const grid = new THREE.GridHelper(3.2, 32, 0x2a3140, 0x1a1f29);
  grid.position.y = -0.021;
  grid.material.transparent = true; grid.material.opacity = 0.55;
  root.add(grid);
  const DUST = 700, dpos = new Float32Array(DUST * 3);
  for (let i = 0; i < DUST; i++) dpos.set([(Math.random() - 0.5) * 3, Math.random() * 0.9, (Math.random() - 0.5) * 3], i * 3);
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute("position", new THREE.BufferAttribute(dpos, 3));
  const dust = new THREE.Points(dustGeo, new THREE.PointsMaterial({ color: 0xdfe8ff, size: 0.006, transparent: true, opacity: 0.45, depthWrite: false }));
  scene.add(dust);

  // ---- start/finish gantry
  const sf = P[0];
  root.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(sf.x, -0.02, sf.z), new THREE.Vector3(sf.x, sf.y + (tel ? 0.12 : 0.035), sf.z)]),
    new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.8 })));

  // ---- the car: glowing head + fading trail, advanced by telemetry speed
  const col = new THREE.Color(accent);
  const car = new THREE.Mesh(new THREE.SphereGeometry(0.011, 16, 12), new THREE.MeshBasicMaterial({ color: 0xffffff }));
  const glowTex = (() => { const c = document.createElement("canvas"); c.width = c.height = 64; const g = c.getContext("2d"); const r = g.createRadialGradient(32, 32, 0, 32, 32, 32); r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.3, "rgba(255,255,255,.5)"); r.addColorStop(1, "rgba(255,255,255,0)"); g.fillStyle = r; g.fillRect(0, 0, 64, 64); return new THREE.CanvasTexture(c); })();
  const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: col, transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  glow.scale.setScalar(0.16);
  car.add(glow);
  if (tel) root.add(car);
  const TRAIL = 46, trailPos = new Float32Array(TRAIL * 3), trailCol = new Float32Array(TRAIL * 3);
  const trailGeo = new THREE.BufferGeometry();
  trailGeo.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
  trailGeo.setAttribute("color", new THREE.BufferAttribute(trailCol, 3));
  if (tel) root.add(new THREE.Line(trailGeo, new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, blending: THREE.AdditiveBlending, depthWrite: false })));
  // arc length along the centre line (world units) so the car moves at a speed-proportional pace
  const seg = P.map((p, i) => p.distanceTo(P[(i + 1) % n]));
  const total = seg.reduce((a, b) => a + b, 0);
  let s = 0, idx = 0;
  const lift = 0.006;
  const posAt = (i, f) => P[i].clone().lerp(P[(i + 1) % n], f).setY(P[i].y + (P[(i + 1) % n].y - P[i].y) * f + lift);
  const history = [];

  // ---- corner labels (HTML, projected each frame)
  const sfEl = document.createElement("span");
  sfEl.className = "c3-turn c3-sf"; sfEl.textContent = "START / FINISH";
  host.querySelector(".c3-labels").append(sfEl);
  const labels = [{ el: sfEl, p: P[0].clone().setY(P[0].y + (tel ? 0.13 : 0.05)) }, ...track.corners.map((k) => {
    const el = document.createElement("span");
    el.className = "c3-turn"; el.textContent = `T${k.n}`;
    host.querySelector(".c3-labels").append(el);
    // snap each corner label to the nearest point on the centre line so it rides the elevation
    let best = 0, bd = Infinity;
    track.points.forEach(([x, y], i) => { const d = (x - k.x) ** 2 + (y - k.y) ** 2; if (d < bd) { bd = d; best = i; } });
    return { el, p: P[best].clone().setY(P[best].y + 0.012) };
  })];

  // ---- camera rig: slow auto-orbit, drag to rotate (with inertia), scroll tilts the view
  let az = -0.6, vel = 0, tilt = 0.9, dragging = false, lastX = 0, lastY = 0, tiltDrag = 0;
  const cv = renderer.domElement;
  cv.style.touchAction = "pan-y";
  cv.addEventListener("pointerdown", (e) => { dragging = true; lastX = e.clientX; lastY = e.clientY; cv.setPointerCapture(e.pointerId); host.classList.add("grabbing"); });
  cv.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    vel = (e.clientX - lastX) * -0.006; az += vel;
    tiltDrag = Math.max(-0.35, Math.min(0.35, tiltDrag + (e.clientY - lastY) * 0.004));
    lastX = e.clientX; lastY = e.clientY; if (!running) frame(performance.now());
  });
  const end = () => { dragging = false; host.classList.remove("grabbing"); };
  cv.addEventListener("pointerup", end); cv.addEventListener("pointercancel", end);

  let aspect = 1;
  const resize = () => {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false); aspect = w / h;
    camera.aspect = aspect;
    // portrait: lift the circuit into the upper part so the HUD below doesn't sit on it
    if (aspect < 1) camera.setViewOffset(w, h, 0, h * 0.13, w, h); else camera.clearViewOffset();
    camera.updateProjectionMatrix();
    if (!running) frame(performance.now());
  };

  let running = false, raf = 0, t0 = performance.now(), shown = -1;
  const v3 = new THREE.Vector3();
  function frame(now) {
    const dt = Math.min(0.05, (now - t0) / 1000); t0 = now;
    if (!dragging) { vel *= 0.94; az += vel + (REDUCED ? 0 : dt * 0.06); }
    // scroll progress through the viewport tilts the camera from overhead to low and dramatic
    const r = host.getBoundingClientRect(), prog = Math.max(0, Math.min(1, 1 - (r.top + r.height / 2) / (innerHeight + r.height / 2)));
    const target = 1.05 - prog * 0.55 + tiltDrag;
    tilt += (target - tilt) * 0.08;
    // fit the circuit at any rotation: its half-diagonal is ~0.6 units, plus perspective margin
    const tv = Math.tan((camera.fov * Math.PI) / 360), dist = Math.max((0.62 / tv) * 0.78, 0.66 / (tv * aspect));
    scene.fog.density = 0.3 * (1.6 / dist);   // same haze whatever the framing distance
    camera.position.set(Math.sin(az) * Math.cos(tilt) * dist, Math.sin(tilt) * dist, Math.cos(az) * Math.cos(tilt) * dist);
    camera.lookAt(0, 0, 0);

    // car: advance by the reference lap's speed at this point (lap compressed to ~9 s)
    if (!REDUCED && tel) {
      const kmh = track.speed[idx];
      s += (kmh / 3.6) * dt * (total / (track.length_m || 5000)) * (track.lap_time / 9);
      while (s > seg[idx]) { s -= seg[idx]; idx = (idx + 1) % n; }
    }
    const pos = posAt(idx, seg[idx] ? s / seg[idx] : 0);
    car.position.copy(pos);
    history.unshift(pos); if (history.length > TRAIL) history.pop();
    for (let i = 0; i < TRAIL; i++) {
      const p = history[Math.min(i, history.length - 1)] || pos, k = 1 - i / TRAIL;
      trailPos.set([p.x, p.y, p.z], i * 3);
      trailCol.set([col.r * k, col.g * k, col.b * k], i * 3);
    }
    trailGeo.attributes.position.needsUpdate = true; trailGeo.attributes.color.needsUpdate = true;
    const v = tel ? Math.round(track.speed[idx]) : null;
    if (tel && onSpeed && v !== shown) { shown = v; onSpeed(v, (v - vmin) / (vmax - vmin)); }

    dust.rotation.y += dt * 0.01;
    renderer.render(scene, camera);

    // project corner labels; fade the ones facing away / far
    const hw = host.clientWidth / 2, hh = host.clientHeight / 2;
    labels.forEach(({ el, p }) => {
      v3.copy(p).applyMatrix4(root.matrixWorld).project(camera);
      const depth = camera.position.distanceTo(p);
      el.style.transform = `translate(${(v3.x * hw + hw).toFixed(1)}px, ${(-v3.y * hh + hh).toFixed(1)}px)`;
      el.style.opacity = v3.z < 1 ? String(Math.max(0.2, Math.min(1, 2.3 - depth))) : "0";
    });
    if (running) raf = requestAnimationFrame(frame);
  }
  const start = () => { if (running || REDUCED) return; running = true; t0 = performance.now(); raf = requestAnimationFrame(frame); };
  const stop = () => { running = false; cancelAnimationFrame(raf); };
  let inView = false;
  const io = new IntersectionObserver(([e]) => { inView = e.isIntersecting; inView && !document.hidden ? start() : stop(); }, { threshold: 0 });
  io.observe(host);
  const onVis = () => (document.hidden || !inView ? stop() : start());
  document.addEventListener("visibilitychange", onVis);
  const ro = new ResizeObserver(resize);
  ro.observe(host);
  resize();
  frame(performance.now());

  return () => {   // dispose when the view re-renders
    stop(); io.disconnect(); ro.disconnect(); document.removeEventListener("visibilitychange", onVis);
    scene.traverse((o) => { o.geometry?.dispose(); o.material?.map?.dispose(); o.material?.dispose(); });
    renderer.dispose(); renderer.forceContextLoss?.();
  };
}
