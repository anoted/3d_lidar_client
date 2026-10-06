import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

// Ring buffer of points; each point remembers when it arrived so the shader can
// fade/expire it. Trail length is then just a uniform -- no reallocation.
const CAPACITY = 2_000_000;
const $ = (id) => document.getElementById(id);

// ---------- scene ----------
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.setClearColor(0x0b0d12);
document.body.prepend(renderer.domElement);

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.05, 500);
camera.up.set(0, 0, 1); // Livox/ROS convention: x forward, y left, z up
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

function resetView() {
  camera.position.set(-7, -3, 5);
  controls.target.set(3, 0, 0);
  controls.update();
}
function topView() {
  camera.position.set(-0.01, 0, 28);
  controls.target.set(0, 0, 0);
  controls.update();
}
resetView();

const grid = new THREE.GridHelper(40, 40, 0x3a4152, 0x1c212c);
grid.rotation.x = Math.PI / 2;
scene.add(grid);
scene.add(new THREE.AxesHelper(0.6)); // sensor origin: red x (forward), green y, blue z

const positions = new THREE.BufferAttribute(new Float32Array(CAPACITY * 3), 3).setUsage(THREE.DynamicDrawUsage);
const intensities = new THREE.BufferAttribute(new Float32Array(CAPACITY), 1).setUsage(THREE.DynamicDrawUsage);
const births = new THREE.BufferAttribute(new Float32Array(CAPACITY).fill(-1e9), 1).setUsage(THREE.DynamicDrawUsage);
const geometry = new THREE.BufferGeometry();
geometry.setAttribute('position', positions);
geometry.setAttribute('intensity', intensities);
geometry.setAttribute('birth', births);
geometry.setDrawRange(0, 0);

const TURBO_GLSL = /* glsl */`
vec3 turbo(float x) {
  const vec4 kR4 = vec4(0.13572138, 4.61539260, -42.66032258, 132.13108234);
  const vec4 kG4 = vec4(0.09140261, 2.19418839, 4.84296658, -14.18503333);
  const vec4 kB4 = vec4(0.10667330, 12.64194608, -60.58204836, 110.36276771);
  const vec2 kR2 = vec2(-152.94239396, 59.28637943);
  const vec2 kG2 = vec2(4.27729857, 2.82956604);
  const vec2 kB2 = vec2(-89.90310912, 27.34824973);
  x = clamp(x, 0.0, 1.0);
  vec4 v4 = vec4(1.0, x, x * x, x * x * x);
  vec2 v2 = v4.zw * v4.z;
  return vec3(dot(v4, kR4) + dot(v2, kR2), dot(v4, kG4) + dot(v2, kG2), dot(v4, kB4) + dot(v2, kB2));
}`;

const material = new THREE.ShaderMaterial({
  uniforms: {
    uNow: { value: 0 }, uPersist: { value: 1 }, uSize: { value: 2 }, uPixelRatio: { value: renderer.getPixelRatio() },
    uMode: { value: 0 }, uLo: { value: 0 }, uHi: { value: 1 },
  },
  vertexShader: /* glsl */`
    attribute float intensity;
    attribute float birth;
    uniform float uNow, uPersist, uSize, uPixelRatio, uMode, uLo, uHi;
    varying vec3 vColor;
    ${TURBO_GLSL}
    void main() {
      float age = uNow - birth;
      if (age > uPersist || age < -1.0) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); gl_PointSize = 0.0; return; }
      vec4 world = modelMatrix * vec4(position, 1.0);
      float v = uMode < 0.5 ? world.z : (uMode < 1.5 ? intensity : length(position));
      vec3 c = uMode > 2.5 ? vec3(0.62, 0.8, 1.0) : turbo((v - uLo) / max(uHi - uLo, 1e-3));
      vColor = c * mix(1.0, 0.3, age / uPersist); // older points dim toward the background
      gl_Position = projectionMatrix * viewMatrix * world;
      gl_PointSize = uSize * uPixelRatio;
    }`,
  fragmentShader: /* glsl */`
    varying vec3 vColor;
    void main() {
      vec2 c = gl_PointCoord - 0.5;
      if (dot(c, c) > 0.25) discard;
      gl_FragColor = vec4(vColor, 1.0);
    }`,
});
const cloud = new THREE.Points(geometry, material);
cloud.frustumCulled = false;
scene.add(cloud);

// ---------- ingest ----------
let head = 0, filled = 0, paused = false;
const t0 = performance.now();
const now = () => (performance.now() - t0) / 1000;

// Smoothed colour ranges per mode (height, intensity, distance) + ground estimate.
const ranges = [{ lo: -1, hi: 2 }, { lo: 0, hi: 100 }, { lo: 0, hi: 15 }];
let groundZ = null, snapRanges = true;
const ema = (a, b, k = 0.1) => a + (b - a) * k;
function percentiles(arr, ps) {
  const s = Float32Array.from(arr).sort();
  return ps.map((p) => s[Math.min(s.length - 1, Math.floor(p * s.length))]);
}

function ingest(buf) {
  const dv = new DataView(buf);
  const n = dv.getUint32(4, true);
  const xyz = new Int16Array(buf, 16, n * 3);
  const inten = new Uint8Array(buf, 16 + n * 6, n);
  const t = now();
  const P = positions.array, I = intensities.array, B = births.array;
  const m = cloud.matrix.elements; // world z = m[2]*x + m[6]*y + m[10]*z
  const start = head;
  const step = Math.max(1, Math.floor(n / 800)), zs = [], ds = [], is = [];
  for (let i = 0; i < n; i++) {
    const x = xyz[3 * i] / 100, y = xyz[3 * i + 1] / 100, z = xyz[3 * i + 2] / 100;
    P[3 * head] = x; P[3 * head + 1] = y; P[3 * head + 2] = z;
    I[head] = inten[i]; B[head] = t;
    if (i % step === 0) { zs.push(m[2] * x + m[6] * y + m[10] * z); ds.push(Math.hypot(x, y, z)); is.push(inten[i]); }
    head = (head + 1) % CAPACITY;
  }
  filled = Math.min(CAPACITY, filled + n);
  geometry.setDrawRange(0, filled);
  markUpdated(start, n);

  if (zs.length > 20) {
    const [z02, z98] = percentiles(zs, [0.02, 0.98]);
    const [d95] = percentiles(ds, [0.95]);
    const [i98] = percentiles(is, [0.98]);
    const k = snapRanges ? 1 : 0.1;
    ranges[0].lo = ema(ranges[0].lo, z02, k); ranges[0].hi = ema(ranges[0].hi, z98, k);
    ranges[1].hi = ema(ranges[1].hi, Math.max(i98, 10), k);
    ranges[2].hi = ema(ranges[2].hi, Math.max(d95, 2), k);
    groundZ = groundZ === null || snapRanges ? z02 : ema(groundZ, z02, 0.05);
    grid.position.z = groundZ;
    snapRanges = false;
  }
}

function markUpdated(start, n) {
  const spans = start + n <= CAPACITY ? [[start, n]] : [[start, CAPACITY - start], [0, start + n - CAPACITY]];
  for (const attr of [positions, intensities, births]) {
    for (const [s, c] of spans) attr.addUpdateRange(s * attr.itemSize, c * attr.itemSize);
    attr.needsUpdate = true;
  }
}

// ---------- link ----------
let frames = 0, bytes = 0, lastN = 0, lastStatus = {};
function connect() {
  const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') { renderStatus(JSON.parse(ev.data)); return; }
    frames++; bytes += ev.data.byteLength;
    lastN = new DataView(ev.data).getUint32(4, true);
    if (!paused) ingest(ev.data);
  };
  ws.onopen = () => renderStatus(lastStatus);
  ws.onclose = () => {
    renderStatus({ ...lastStatus, ssh: 'web server unreachable' });
    setTimeout(connect, 1500);
  };
}

function renderStatus(s) {
  lastStatus = s;
  const b = s.bridge || {};
  $('s-target').textContent = s.target || '—';
  $('s-link').textContent = s.ssh || '—';
  $('s-source').textContent = b.detail || '—';
  $('s-source').title = b.detail || '';
  setGravity(b.gravity);
  $('orient').options[0].disabled = !gravity;
  if (!gravity && $('orient').value === 'imu') $('orient').title = 'waiting for IMU data from the sensor';
  if (b.points_per_s !== undefined) $('s-pps').textContent = `${(b.points_per_s / 1000).toFixed(0)}k pts/s`;
  $('err').textContent = s.error || b.error || '';
  const dot = $('dot');
  const live = s.ssh === 'connected' && b.points_per_s > 0;
  dot.className = live ? 'ok' : s.ssh === 'connected' || s.ssh === 'connecting' ? 'warn' : 'bad';
}

setInterval(() => {
  $('s-fps').textContent = `${frames} /s`;
  $('s-npts').textContent = lastN ? lastN.toLocaleString() : '—';
  $('s-bw').textContent = `${(bytes / 1e6).toFixed(2)} MB/s`;
  if (frames === 0 && lastStatus.ssh === 'connected') $('dot').className = 'warn';
  frames = 0; bytes = 0;
}, 1000);

// ---------- UI ----------
const LEGEND = [['Height', 'm'], ['Intensity', ''], ['Distance', 'm']];
function turboCss(x) {
  const c = (a, b) => a[0] + a[1] * x + a[2] * x * x + a[3] * x ** 3 + b[0] * x ** 4 + b[1] * x ** 5;
  const r = c([0.13572138, 4.6153926, -42.66032258, 132.13108234], [-152.94239396, 59.28637943]);
  const g = c([0.09140261, 2.19418839, 4.84296658, -14.18503333], [4.27729857, 2.82956604]);
  const b = c([0.1066733, 12.64194608, -60.58204836, 110.36276771], [-89.90310912, 27.34824973]);
  const f = (v) => Math.round(Math.max(0, Math.min(1, v)) * 255);
  return `rgb(${f(r)},${f(g)},${f(b)})`;
}
$('legend-bar').style.background =
  `linear-gradient(90deg, ${Array.from({ length: 11 }, (_, i) => turboCss(i / 10)).join(',')})`;

function bindRange(id, fmt, apply) {
  const el = $(id), out = $(id + '-v');
  const update = () => { out.textContent = fmt(+el.value); apply(+el.value); };
  el.addEventListener('input', update);
  update();
}
bindRange('size', (v) => v.toFixed(1), (v) => { material.uniforms.uSize.value = v; });
bindRange('persist', (v) => v.toFixed(1), (v) => { material.uniforms.uPersist.value = v; });
$('color').addEventListener('change', (e) => { material.uniforms.uMode.value = +e.target.value; });
// Orientation: level with the lidar's own accelerometer (rotate measured "up" onto +z,
// minimal rotation so forward stays forward), or show the raw / flipped sensor frame.
let gravity = null;
const Z_UP = new THREE.Vector3(0, 0, 1);
function applyOrientation() {
  const mode = $('orient').value;
  if (mode === 'imu' && gravity) cloud.quaternion.setFromUnitVectors(gravity, Z_UP);
  else if (mode === 'flip') cloud.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI);
  else cloud.quaternion.identity();
  cloud.updateMatrix();
}
function setGravity(g) {
  if (!g) return;
  const v = new THREE.Vector3(...g).normalize();
  if (gravity && gravity.angleTo(v) < 0.005) return; // ignore sub-0.3° jitter
  const first = !gravity;
  gravity = v;
  applyOrientation();
  if (first) snapRanges = true;
}
$('orient').addEventListener('change', () => { applyOrientation(); snapRanges = true; });
$('grid').addEventListener('change', (e) => { grid.visible = e.target.checked; });
function togglePause() {
  paused = !paused;
  $('pause').textContent = paused ? 'Resume' : 'Pause';
  $('pause').classList.toggle('on', paused);
}
$('pause').addEventListener('click', togglePause);
$('reset').addEventListener('click', resetView);
$('top').addEventListener('click', topView);
$('collapse').addEventListener('click', () => {
  const c = $('panel').classList.toggle('collapsed');
  $('collapse').textContent = c ? '+' : '–';
});
addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePause(); }
  if (e.key === 'r' || e.key === 'R') resetView();
  if (e.key === 'm' || e.key === 'M') setMeasuring(!measuring);
  if (e.key === 'Escape') { if (pending) cancelPending(); else if (measuring) setMeasuring(false); }
});
addEventListener('resize', () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

// ---------- measure ----------
// Endpoints are stored in the sensor frame as children of `cloud`, so lines stay
// glued to the points when orientation changes. Clicks snap to a visible lidar point.
const measureGroup = new THREE.Group();
cloud.add(measureGroup);
const MEASURE_COLOR = 0xffd54a;
const lineMat = new THREE.LineBasicMaterial({ color: MEASURE_COLOR, depthTest: false, transparent: true });
const dotMat = new THREE.PointsMaterial({ color: MEASURE_COLOR, size: 9, sizeAttenuation: false, depthTest: false, transparent: true });
const measurements = []; // { a, b, obj, label }
let measuring = false, pending = null, hoverPoint = null;
const hintDefault = $('hint').textContent;

function makeSegment(a, b) {
  const g = new THREE.Group();
  const pts = b ? [a, b] : [a];
  const dots = new THREE.Points(new THREE.BufferGeometry().setFromPoints(pts), dotMat);
  g.add(dots);
  if (b) g.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), lineMat));
  g.renderOrder = 10;
  g.children.forEach((c) => { c.renderOrder = 10; c.frustumCulled = false; });
  measureGroup.add(g);
  return g;
}
function disposeObj(g) {
  measureGroup.remove(g);
  g.children.forEach((c) => c.geometry.dispose());
}

// Nearest-to-camera visible point within PICK_PX of the cursor, in sensor frame.
const PICK_PX = 10;
const pvm = new THREE.Matrix4();
function pick(clientX, clientY) {
  const rect = renderer.domElement.getBoundingClientRect();
  const mx = clientX - rect.left, my = clientY - rect.top, w = rect.width, h = rect.height;
  cloud.updateMatrixWorld();
  pvm.multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse).multiply(cloud.matrixWorld);
  const e = pvm.elements, P = positions.array, B = births.array;
  const tNow = material.uniforms.uNow.value, persist = material.uniforms.uPersist.value;
  let best = -1, bestDepth = Infinity;
  for (let i = 0; i < filled; i++) {
    const age = tNow - B[i];
    if (age > persist || age < -1) continue;
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    const cw = e[3] * x + e[7] * y + e[11] * z + e[15];
    if (cw <= 0) continue;
    const sx = ((e[0] * x + e[4] * y + e[8] * z + e[12]) / cw * 0.5 + 0.5) * w;
    const sy = (0.5 - (e[1] * x + e[5] * y + e[9] * z + e[13]) / cw * 0.5) * h;
    const dx = sx - mx, dy = sy - my;
    if (dx * dx + dy * dy > PICK_PX * PICK_PX || cw >= bestDepth) continue;
    best = i; bestDepth = cw;
  }
  return best < 0 ? null : new THREE.Vector3(P[3 * best], P[3 * best + 1], P[3 * best + 2]);
}

function newLabel() {
  const el = document.createElement('div');
  el.className = 'mlabel';
  $('labels').appendChild(el);
  return el;
}
const wa = new THREE.Vector3(), wb = new THREE.Vector3(), mid = new THREE.Vector3();
function placeLabel(el, a, b) {
  wa.copy(a).applyMatrix4(cloud.matrixWorld);
  wb.copy(b).applyMatrix4(cloud.matrixWorld);
  const d = wa.distanceTo(wb), horiz = Math.hypot(wb.x - wa.x, wb.y - wa.y), vert = Math.abs(wb.z - wa.z);
  const inch = (m) => (m / 0.0254).toFixed(1);
  el.innerHTML = `<b>${d.toFixed(2)} m</b> <b class="in">${inch(d)} in</b>` +
    `<small>↔ ${horiz.toFixed(2)} m / ${inch(horiz)} in · ↕ ${vert.toFixed(2)} m / ${inch(vert)} in</small>`;
  mid.addVectors(wa, wb).multiplyScalar(0.5).project(camera);
  el.style.display = mid.z > 1 ? 'none' : '';
  el.style.left = `${(mid.x * 0.5 + 0.5) * innerWidth}px`;
  el.style.top = `${(0.5 - mid.y * 0.5) * innerHeight}px`;
}
function updateLabels() {
  for (const m of measurements) placeLabel(m.label, m.a, m.b);
  if (pending) {
    if (hoverPoint) placeLabel(pending.label, pending.a, hoverPoint);
    pending.label.style.visibility = hoverPoint ? 'visible' : 'hidden';
  }
}

function cancelPending() {
  if (!pending) return;
  disposeObj(pending.obj);
  if (pending.rubber) disposeObj(pending.rubber);
  pending.label.remove();
  pending = null;
}
function setMeasuring(on) {
  measuring = on;
  if (!on) { cancelPending(); setHover(null); }
  $('measure').classList.toggle('on', on);
  renderer.domElement.classList.toggle('measuring', on);
  $('hint').textContent = on ? 'click two points to measure · Esc: cancel · tip: pause (space) for precise picks' : hintDefault;
}
let hoverObj = null;
function setHover(p) {
  hoverPoint = p;
  if (hoverObj) { disposeObj(hoverObj); hoverObj = null; }
  if (pending?.rubber) { disposeObj(pending.rubber); pending.rubber = null; }
  if (!p) return;
  hoverObj = makeSegment(p);
  if (pending) pending.rubber = makeSegment(pending.a, p);
}

// Treat as a click only if the pointer barely moved (otherwise it was an orbit drag).
let downAt = null;
renderer.domElement.addEventListener('pointerdown', (e) => { downAt = e.button === 0 ? [e.clientX, e.clientY] : null; });
renderer.domElement.addEventListener('pointerup', (e) => {
  if (!measuring || !downAt || Math.hypot(e.clientX - downAt[0], e.clientY - downAt[1]) > 4) return;
  const p = pick(e.clientX, e.clientY);
  if (!p) return;
  if (!pending) {
    pending = { a: p, obj: makeSegment(p), label: newLabel(), rubber: null };
  } else {
    measurements.push({ a: pending.a, b: p, obj: makeSegment(pending.a, p), label: pending.label });
    disposeObj(pending.obj);
    if (pending.rubber) disposeObj(pending.rubber);
    pending.label.style.visibility = 'visible';
    pending = null;
  }
  setHover(null);
});
let hoverQueued = null;
renderer.domElement.addEventListener('pointermove', (e) => {
  if (!measuring || e.buttons) return;
  if (!hoverQueued) requestAnimationFrame(() => { setHover(pick(hoverQueued.x, hoverQueued.y)); hoverQueued = null; });
  hoverQueued = { x: e.clientX, y: e.clientY };
});
$('measure').addEventListener('click', () => setMeasuring(!measuring));
$('clear-measure').addEventListener('click', () => {
  cancelPending();
  for (const m of measurements) { disposeObj(m.obj); m.label.remove(); }
  measurements.length = 0;
});

let pausedAt = 0;
function frame() {
  requestAnimationFrame(frame);
  // While paused, freeze the clock so the trail stays on screen.
  if (paused) { if (!pausedAt) pausedAt = now(); } else pausedAt = 0;
  const u = material.uniforms, mode = u.uMode.value;
  u.uNow.value = paused ? pausedAt : now();
  if (mode < 3) {
    u.uLo.value = ranges[mode].lo; u.uHi.value = ranges[mode].hi;
    $('legend-title').textContent = LEGEND[mode][0] + (LEGEND[mode][1] ? ` (${LEGEND[mode][1]})` : '');
    $('legend-lo').textContent = ranges[mode].lo.toFixed(mode === 1 ? 0 : 1);
    $('legend-hi').textContent = ranges[mode].hi.toFixed(mode === 1 ? 0 : 1);
  }
  $('legend').classList.toggle('hidden', mode === 3);
  controls.update();
  renderer.render(scene, camera);
  updateLabels();
}

connect();
frame();
