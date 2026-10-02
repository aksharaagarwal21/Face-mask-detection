/* ═══════════════════════════════════════════════
   field.js — field mode: phone camera → /api/predict → overlay

   - back camera at high resolution; optical zoom when the phone exposes it,
     otherwise a digital crop that keeps the sensor's full resolution
   - tap to aim the zoom, pinch to zoom, double-tap to reset
   - one frame in flight at a time; the server tracks faces per session, so
     labels are smoothed and each person keeps an id
   - flagged faces are shown enlarged, so a person far away can be seen
   - freeze / photo / save work on the exact pixels that were analysed
   ═══════════════════════════════════════════════ */

'use strict';

const $ = id => document.getElementById(id);
const COLORS = { with_mask: '#22c55e', without_mask: '#ef4444', mask_weared_incorrect: '#f59e0b' };
const NAMES = { with_mask: 'MASK', without_mask: 'NO MASK', mask_weared_incorrect: 'INCORRECT' };
const MAX_SEND_SIDE = 1920;     // longer side of the frames sent for analysis
const MAX_PHOTO_SIDE = 4096;
const JPEG_QUALITY = 0.85;
const MAX_ZOOM = 8;
const MIN_MARKER = 16;          // px on screen: tiny distant faces still get a visible marker

const settings = loadSettings();
const state = {
  stream: null, track: null,
  hwZoom: null, hwApplied: 1,   // optical/hardware zoom range and the value in effect
  zoom: 1, center: { x: 0.5, y: 0.5 },
  far: false,
  still: null, stillHw: 1, photo: false, stillDirty: false,
  session: randomId(), reset: true,
  result: null,                 // { data, rect, scale, canvas }
  layout: null,                 // last drawn { d, rect, src }
  seen: new Set(), times: [],
  wakeLock: null, audio: null, hwHintShown: false,
};

// ── Settings ─────────────────────────────────────────────────────────────────
function loadSettings() {
  const defaults = { camera: 'environment', resolution: 1920, sound: true, vibrate: true,
                     labelAll: false, wake: true };
  try { return { ...defaults, ...JSON.parse(localStorage.getItem('fieldSettings') || '{}') }; }
  catch { return defaults; }
}
function saveSettings() {
  try { localStorage.setItem('fieldSettings', JSON.stringify(settings)); } catch { /* private mode */ }
}

// ── Small helpers ────────────────────────────────────────────────────────────
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
const sleep = ms => new Promise(r => setTimeout(r, ms));
function randomId() {
  const a = new Uint8Array(9);
  crypto.getRandomValues(a);
  return Array.from(a, b => b.toString(16).padStart(2, '0')).join('');
}
let toastTimer = null;
function toast(text) {
  const t = $('toast');
  t.textContent = text;
  t.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.remove('show'), 2800);
}
function setStatus(kind, text) {
  $('status').dataset.state = kind;
  $('statusText').textContent = text;
}
function showMessage(text, html = false) {
  if (html) $('messageText').innerHTML = text; else $('messageText').textContent = text;
  $('message').hidden = false;
}
function hideMessage() { $('message').hidden = true; }

// ── Camera ───────────────────────────────────────────────────────────────────
async function startCamera() {
  stopCamera();
  if (!window.isSecureContext || !navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    showMessage('The live camera needs a secure (https) link. On the computer, start the server with ' +
                '<code>python app.py --field</code> and open the phone link it shows. ' +
                'Photo mode works here too.', true);
    setStatus('error', 'No camera');
    return;
  }
  setStatus('starting', 'Starting camera');
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: { ideal: settings.camera },
        width: { ideal: settings.resolution },
        height: { ideal: Math.round(settings.resolution * 9 / 16) },
      },
    });
    state.stream = stream;
    state.track = stream.getVideoTracks()[0];
    const video = $('video');
    video.srcObject = stream;
    await video.play().catch(() => {});
    const caps = state.track.getCapabilities ? state.track.getCapabilities() : {};
    state.hwZoom = caps.zoom ? { min: caps.zoom.min, max: caps.zoom.max } : null;
    state.hwApplied = (state.track.getSettings && state.track.getSettings().zoom) || 1;
    hideMessage();
    changed();
    applyHardwareZoom();
    requestWakeLock();
  } catch (e) {
    const text = e.name === 'NotAllowedError'
      ? 'Camera permission was denied. Allow camera access for this site in the browser settings, or take a photo instead.'
      : (e.name === 'NotFoundError' || e.name === 'OverconstrainedError')
        ? 'No suitable camera was found. Take a photo instead.'
        : `Camera error: ${e.message}`;
    showMessage(text);
    setStatus('error', 'No camera');
  }
}

function stopCamera() {
  if (state.stream) state.stream.getTracks().forEach(t => t.stop());
  state.stream = null;
  state.track = null;
}

async function requestWakeLock() {
  if (!settings.wake || !('wakeLock' in navigator) || document.hidden) return;
  try { state.wakeLock = await navigator.wakeLock.request('screen'); } catch { /* not allowed */ }
}

// ── Source, zoom and crop ────────────────────────────────────────────────────
function source() {
  if (state.still) return { el: state.still, w: state.still.width, h: state.still.height };
  const v = $('video');
  if (state.stream && v.videoWidth) return { el: v, w: v.videoWidth, h: v.videoHeight };
  return null;
}

function digitalZoom() {
  return Math.max(1, state.zoom / (state.still ? state.stillHw : state.hwApplied));
}

function cropRect(src) {
  const z = digitalZoom();
  const w = src.w / z, h = src.h / z;
  const cx = clamp(state.center.x * src.w, w / 2, src.w - w / 2);
  const cy = clamp(state.center.y * src.h, h / 2, src.h - h / 2);
  return { x: cx - w / 2, y: cy - h / 2, w, h };
}

const sameRect = (a, b) => a && b && ['x', 'y', 'w', 'h'].every(k => Math.abs(a[k] - b[k]) < 1);

function changed() {
  // The analysed region changed: coordinates restart, so does tracking
  state.reset = true;
  state.result = null;
  if (state.still) state.stillDirty = true;
}

function setZoom(z) {
  state.zoom = clamp(Math.round(z * 10) / 10, 1, MAX_ZOOM);
  if (state.zoom === 1) state.center = { x: 0.5, y: 0.5 };
  $('zoom').value = state.zoom;
  $('zoomPill').textContent = `${state.zoom.toFixed(1)}×`;
  changed();
  applyHardwareZoom();
}

let hwTimer = null;
function applyHardwareZoom() {
  if (!state.track || !state.hwZoom || state.still) return;
  clearTimeout(hwTimer);
  hwTimer = setTimeout(async () => {
    const want = clamp(state.zoom, state.hwZoom.min, state.hwZoom.max);
    try {
      await state.track.applyConstraints({ advanced: [{ zoom: want }] });
      state.hwApplied = state.track.getSettings().zoom || want;
    } catch {
      state.hwZoom = null;
      state.hwApplied = 1;
    }
    changed();
  }, 120);
}

/** Aim the zoom at a point given in normalised source coordinates. */
function aimAt(p, zoomTo) {
  state.center = { x: clamp(p.x, 0, 1), y: clamp(p.y, 0, 1) };
  const optical = !state.still && state.hwZoom && (zoomTo || state.zoom) <= state.hwZoom.max;
  if (optical && !state.hwHintShown) {
    toast('This phone zooms optically: point the camera at the person.');
    state.hwHintShown = true;
  }
  setZoom(zoomTo || state.zoom);
}

function viewToSource(clientX, clientY) {
  const L = state.layout;
  if (!L) return null;
  const box = $('view').getBoundingClientRect();
  const x = clientX - box.left, y = clientY - box.top;
  if (x < L.d.x || y < L.d.y || x > L.d.x + L.d.w || y > L.d.y + L.d.h) return null;
  return {
    x: (L.rect.x + (x - L.d.x) / L.d.w * L.rect.w) / L.src.w,
    y: (L.rect.y + (y - L.d.y) / L.d.h * L.rect.h) / L.src.h,
  };
}

// ── Drawing ──────────────────────────────────────────────────────────────────
function drawFaces(ctx, faces, toScreen, unit = 1) {
  const placed = [];            // label rectangles already drawn
  const maxX = ctx.canvas.width / (ctx.getTransform().a || 1);   // drawing width in our units
  // violations first, most confident first, so they get the label spots in a crowd
  const order = [...faces].sort((a, b) => (b.violation - a.violation) || (b.confidence - a.confidence));
  for (const f of order) {
    const masked = f.label === 'with_mask';
    const color = COLORS[f.label] || '#fff';
    let { x, y, w, h } = toScreen(f.box);
    ctx.strokeStyle = color;
    ctx.lineWidth = (masked ? 2 : 3) * unit;
    if (Math.max(w, h) < MIN_MARKER * unit) {
      // too small to see as a box: ring around it instead
      ctx.beginPath();
      ctx.arc(x + w / 2, y + h / 2, MIN_MARKER * unit / 2 + 4 * unit, 0, Math.PI * 2);
      ctx.stroke();
      x -= (MIN_MARKER * unit - w) / 2;
      y -= (MIN_MARKER * unit - h) / 2;
    } else {
      ctx.strokeRect(x, y, w, h);
    }
    if (masked && !settings.labelAll) continue;

    const text = `${NAMES[f.label] || f.label} ${Math.round(f.confidence * 100)}%` +
                 (f.track_id !== undefined ? ` #${f.track_id}` : '');
    ctx.font = `700 ${12 * unit}px system-ui, sans-serif`;
    const tw = ctx.measureText(text).width + 8 * unit, th = 18 * unit;
    const ty = y - th - 2 * unit >= 0 ? y - th - 2 * unit : y + Math.max(h, MIN_MARKER * unit) + 2 * unit;
    x = clamp(x, 0, Math.max(0, maxX - tw));                       // keep the label on screen
    // in a crowd, skip labels that would cover another one (the flagged list has every face)
    if (placed.some(p => x < p.x + p.w && x + tw > p.x && ty < p.y + p.h && ty + th > p.y)) continue;
    placed.push({ x, y: ty, w: tw, h: th });
    ctx.fillStyle = color;
    ctx.fillRect(x, ty, tw, th);
    ctx.fillStyle = masked ? '#03140a' : '#fff';
    ctx.fillText(text, x + 4 * unit, ty + 13 * unit);
  }
}

function draw() {
  const canvas = $('view');
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const cw = canvas.clientWidth, ch = canvas.clientHeight;
  if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
    canvas.width = Math.round(cw * dpr);
    canvas.height = Math.round(ch * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, cw, ch);

  const src = source();
  if (src && cw && ch) {
    const rect = cropRect(src);
    const s = Math.min(cw / rect.w, ch / rect.h);
    const d = { w: rect.w * s, h: rect.h * s };
    d.x = (cw - d.w) / 2;
    d.y = (ch - d.h) / 2;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(src.el, rect.x, rect.y, rect.w, rect.h, d.x, d.y, d.w, d.h);
    state.layout = { d, rect, src };

    const r = state.result;
    if (r && sameRect(r.rect, rect)) {
      const k = d.w / r.rect.w / r.scale;          // sent-frame px → screen px
      drawFaces(ctx, r.data.faces, b => ({ x: d.x + b[0] * k, y: d.y + b[1] * k,
                                           w: (b[2] - b[0]) * k, h: (b[3] - b[1]) * k }));
    }
  }
  requestAnimationFrame(draw);
}

// ── Analysis ─────────────────────────────────────────────────────────────────
async function analyze() {
  const src = source();
  if (!src) { await sleep(150); return; }
  const rect = cropRect(src);
  const scale = Math.min(1, MAX_SEND_SIDE / Math.max(rect.w, rect.h));
  const canvas = document.createElement('canvas');
  canvas.width = Math.max(1, Math.round(rect.w * scale));
  canvas.height = Math.max(1, Math.round(rect.h * scale));
  const c = canvas.getContext('2d');
  c.imageSmoothingQuality = 'high';
  c.drawImage(src.el, rect.x, rect.y, rect.w, rect.h, 0, 0, canvas.width, canvas.height);
  if (state.still) state.stillDirty = false;

  const blob = await new Promise(r => canvas.toBlob(r, 'image/jpeg', JPEG_QUALITY));
  const params = new URLSearchParams({ session: state.session, range: state.far ? 'far' : 'near' });
  const resetting = state.reset;
  if (resetting) params.set('reset', '1');
  state.reset = false;

  const t0 = performance.now();
  let res;
  try {
    res = await fetch(`/api/predict?${params}`, {
      method: 'POST', body: blob, headers: { 'Content-Type': 'image/jpeg' }, credentials: 'same-origin',
    });
  } catch {
    setStatus('error', 'Offline');
    state.reset = true;
    await sleep(1000);
    return;
  }
  if (res.status === 401) {
    setStatus('error', 'Locked');
    showMessage('This link has expired or the server was restarted with a new key. ' +
                'Scan the QR code on the computer again.');
    await sleep(3000);
    return;
  }
  if (!res.ok) {
    setStatus('error', `Server error ${res.status}`);
    await sleep(1000);
    return;
  }
  const data = await res.json();
  const rtt = performance.now() - t0;

  // The user zoomed or moved while this frame was in flight: it no longer matches
  const now = source();
  if (!now || !sameRect(cropRect(now), rect) || (state.reset && !resetting)) return;

  if (resetting) state.seen.clear();
  state.result = { data, rect, scale, canvas };
  updatePanels(data, canvas, !resetting);

  state.times.push(performance.now());
  state.times = state.times.slice(-6);
  const fps = state.times.length > 1
    ? (state.times.length - 1) / ((state.times[state.times.length - 1] - state.times[0]) / 1000) : 0;
  $('metrics').textContent = state.still
    ? `${Math.round(rtt)} ms`
    : `${fps.toFixed(1)} fps · ${Math.round(rtt)} ms`;
  if (state.still) setStatus('paused', state.photo ? 'Photo' : 'Frozen');
  else setStatus('live', 'Live');
}

async function loop() {
  for (;;) {
    const live = !state.still && state.stream && !document.hidden;
    if (live || (state.still && state.stillDirty)) {
      try { await analyze(); } catch (e) { console.error(e); await sleep(500); }
    } else {
      await sleep(150);
    }
  }
}

// ── Counts, flagged faces, alerts ────────────────────────────────────────────
function updatePanels(data, canvas, alertNew) {
  const c = data.counts;
  $('nTotal').textContent = c.total;
  $('nMask').textContent = c.with_mask || 0;
  $('nNoMask').textContent = c.without_mask || 0;
  $('nIncorrect').textContent = c.mask_weared_incorrect || 0;

  const flagged = data.faces.filter(f => f.violation).sort((a, b) => b.confidence - a.confidence);
  const fresh = new Set();
  for (const f of flagged) {
    if (!state.seen.has(f.track_id)) {
      state.seen.add(f.track_id);
      fresh.add(f.track_id);
    }
  }
  if (alertNew && fresh.size) alertUser();

  const list = $('flagged');
  list.textContent = '';
  if (!flagged.length) {
    const p = document.createElement('p');
    p.className = 'empty';
    p.textContent = c.total ? 'None in view' : 'No faces in view';
    list.append(p);
    return;
  }
  for (const f of flagged) {
    const card = document.createElement('button');
    card.type = 'button';
    card.className = 'face-card' + (f.label === 'mask_weared_incorrect' ? ' incorrect' : '') +
                     (alertNew && fresh.has(f.track_id) ? ' new' : '');
    card.title = 'Zoom in on this person';
    const thumb = document.createElement('canvas');
    thumb.width = thumb.height = 168;
    drawThumb(thumb, canvas, f.box);
    const tag = document.createElement('span');
    tag.className = 'tag';
    tag.style.color = COLORS[f.label];
    tag.textContent = NAMES[f.label];
    const conf = document.createElement('span');
    conf.className = 'conf';
    conf.textContent = `${Math.round(f.confidence * 100)}%` + (f.track_id !== undefined ? ` · #${f.track_id}` : '');
    card.append(thumb, tag, conf);
    card.addEventListener('click', () => focusFace(f));
    list.append(card);
  }
}

function drawThumb(thumb, canvas, box) {
  // square crop around the face with context, enlarged
  const [x1, y1, x2, y2] = box;
  const side = Math.max(x2 - x1, y2 - y1) * 1.7;
  const cx = (x1 + x2) / 2, cy = (y1 + y2) / 2;
  const t = thumb.getContext('2d');
  t.fillStyle = '#000';
  t.fillRect(0, 0, thumb.width, thumb.height);
  t.imageSmoothingQuality = 'high';
  t.drawImage(canvas, cx - side / 2, cy - side / 2, side, side, 0, 0, thumb.width, thumb.height);
}

function focusFace(f) {
  const r = state.result, L = state.layout;
  if (!r || !L) return;
  const cx = r.rect.x + (f.box[0] + f.box[2]) / 2 / r.scale;
  const cy = r.rect.y + (f.box[1] + f.box[3]) / 2 / r.scale;
  aimAt({ x: cx / L.src.w, y: cy / L.src.h }, Math.min(MAX_ZOOM, Math.max(2, state.zoom * 2)));
}

function alertUser() {
  if (settings.vibrate && navigator.vibrate) navigator.vibrate([120, 60, 120]);
  if (settings.sound && state.audio) {
    const ctx = state.audio, o = ctx.createOscillator(), g = ctx.createGain();
    o.frequency.value = 880;
    g.gain.setValueAtTime(0.0001, ctx.currentTime);
    g.gain.exponentialRampToValueAtTime(0.25, ctx.currentTime + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + 0.25);
    o.connect(g).connect(ctx.destination);
    o.start();
    o.stop(ctx.currentTime + 0.26);
  }
}

// ── Freeze, photo, save ──────────────────────────────────────────────────────
function updateModeUI() {
  const still = !!state.still;
  $('btnFreeze').setAttribute('aria-pressed', String(still));
  $('freezeIco').textContent = still ? '▶' : '⏸';
  $('freezeLbl').textContent = still ? 'Live' : 'Freeze';
  $('modePill').hidden = !still;
  $('modePill').textContent = state.photo ? 'PHOTO' : 'FROZEN';
}

function freeze() {
  const v = $('video');
  if (!state.stream || !v.videoWidth) { toast('No live camera to freeze'); return; }
  const c = document.createElement('canvas');
  c.width = v.videoWidth;
  c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  state.still = c;
  state.stillHw = state.hwApplied;
  state.photo = false;
  changed();
  updateModeUI();
  toast('Frozen: zoom or tap to look closer');
}

function goLive() {
  state.still = null;
  state.photo = false;
  setZoom(1);
  updateModeUI();
  if (!state.stream) startCamera();
}

function loadPhoto(file) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = () => {
    const s = Math.min(1, MAX_PHOTO_SIDE / Math.max(img.naturalWidth, img.naturalHeight));
    const c = document.createElement('canvas');
    c.width = Math.round(img.naturalWidth * s);
    c.height = Math.round(img.naturalHeight * s);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    URL.revokeObjectURL(url);
    state.still = c;
    state.stillHw = 1;
    state.photo = true;
    hideMessage();
    setZoom(1);
    updateModeUI();
  };
  img.onerror = () => { URL.revokeObjectURL(url); toast('That file could not be opened as an image'); };
  img.src = url;
}

async function saveSnapshot() {
  const r = state.result;
  if (!r) { toast('Nothing analysed yet'); return; }
  const footer = 44;
  const out = document.createElement('canvas');
  out.width = r.canvas.width;
  out.height = r.canvas.height + footer;
  const ctx = out.getContext('2d');
  ctx.drawImage(r.canvas, 0, 0);
  const unit = Math.max(1, r.canvas.width / 900);
  drawFaces(ctx, r.data.faces, b => ({ x: b[0], y: b[1], w: b[2] - b[0], h: b[3] - b[1] }), unit);
  ctx.fillStyle = '#05070c';
  ctx.fillRect(0, r.canvas.height, out.width, footer);
  ctx.fillStyle = '#f8fafc';
  ctx.font = '600 14px system-ui, sans-serif';
  const c = r.data.counts;
  const stamp = new Date();
  ctx.fillText(`${stamp.toLocaleString()} · faces ${c.total} · mask ${c.with_mask || 0} · ` +
               `no mask ${c.without_mask || 0} · incorrect ${c.mask_weared_incorrect || 0}`, 10, r.canvas.height + 18);
  ctx.fillStyle = '#94a3b8';
  ctx.font = '12px system-ui, sans-serif';
  ctx.fillText('Automated flags: verify in person before acting.', 10, r.canvas.height + 36);

  const blob = await new Promise(res => out.toBlob(res, 'image/jpeg', 0.92));
  const pad = n => String(n).padStart(2, '0');
  const name = `mask-check-${stamp.getFullYear()}${pad(stamp.getMonth() + 1)}${pad(stamp.getDate())}-` +
               `${pad(stamp.getHours())}${pad(stamp.getMinutes())}${pad(stamp.getSeconds())}.jpg`;
  const file = new File([blob], name, { type: 'image/jpeg' });
  if (navigator.canShare && navigator.canShare({ files: [file] })) {
    try { await navigator.share({ files: [file], title: 'Mask check' }); return; }
    catch (e) { if (e.name === 'AbortError') return; }
  }
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  toast('Saved to this device');
}

// ── Gestures ─────────────────────────────────────────────────────────────────
function initGestures() {
  const vp = $('viewport');
  const pointers = new Map();
  let pinch = null, lastTap = 0;
  const dist = () => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };
  vp.addEventListener('pointerdown', e => {
    if (e.target.closest('.zoom, .message')) return;
    vp.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, t: performance.now() });
    if (pointers.size === 2) pinch = { d: dist(), z: state.zoom };
    if (!state.audio && settings.sound && window.AudioContext) state.audio = new AudioContext();
  });
  vp.addEventListener('pointermove', e => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    p.x = e.clientX;
    p.y = e.clientY;
    if (pinch && pointers.size === 2) setZoom(pinch.z * dist() / pinch.d);
  });
  const end = e => {
    const p = pointers.get(e.pointerId);
    if (!p) return;
    pointers.delete(e.pointerId);
    if (pinch) { if (pointers.size < 2) pinch = null; return; }
    const moved = Math.hypot(p.x - p.x0, p.y - p.y0);
    if (moved > 12 || performance.now() - p.t > 400) return;
    const now = performance.now();
    if (now - lastTap < 320) { lastTap = 0; setZoom(1); return; }      // double-tap: reset
    lastTap = now;
    const target = viewToSource(e.clientX, e.clientY);
    if (target) aimAt(target, state.zoom < 1.5 ? 2 : state.zoom);
  };
  vp.addEventListener('pointerup', end);
  vp.addEventListener('pointercancel', e => { pointers.delete(e.pointerId); pinch = null; });
}

// ── Wiring ───────────────────────────────────────────────────────────────────
function initControls() {
  $('zoom').addEventListener('input', e => setZoom(parseFloat(e.target.value)));
  $('zoomIn').addEventListener('click', () => setZoom(state.zoom * 1.25));
  $('zoomOut').addEventListener('click', () => setZoom(state.zoom / 1.25));

  $('btnRange').addEventListener('click', () => {
    state.far = !state.far;
    $('btnRange').setAttribute('aria-pressed', String(state.far));
    $('rangePill').textContent = state.far ? 'FAR' : 'NEAR';
    $('rangePill').classList.toggle('pill-far', state.far);
    if (state.still) state.stillDirty = true;
    toast(state.far ? 'Far range: finds smaller, distant faces (slower)' : 'Near range');
  });
  $('btnFreeze').addEventListener('click', () => (state.still ? goLive() : freeze()));
  $('btnSave').addEventListener('click', saveSnapshot);
  $('photoInput').addEventListener('change', e => {
    const file = e.target.files[0];
    e.target.value = '';
    if (file) loadPhoto(file);
  });

  const dialog = $('settings');
  $('btnSettings').addEventListener('click', () => {
    $('setCamera').value = settings.camera;
    $('setResolution').value = String(settings.resolution);
    $('setSound').checked = settings.sound;
    $('setVibrate').checked = settings.vibrate;
    $('setLabelAll').checked = settings.labelAll;
    $('setWake').checked = settings.wake;
    dialog.showModal();
  });
  dialog.addEventListener('close', () => {
    const restart = settings.camera !== $('setCamera').value ||
                    settings.resolution !== parseInt($('setResolution').value, 10);
    settings.camera = $('setCamera').value;
    settings.resolution = parseInt($('setResolution').value, 10);
    settings.sound = $('setSound').checked;
    settings.vibrate = $('setVibrate').checked;
    settings.labelAll = $('setLabelAll').checked;
    settings.wake = $('setWake').checked;
    saveSettings();
    if (restart && !state.still) startCamera();
    if (settings.wake) requestWakeLock();
    else if (state.wakeLock) { state.wakeLock.release().catch(() => {}); state.wakeLock = null; }
  });

  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && state.stream) requestWakeLock();
  });

  $('btnPhone').addEventListener('click', openPhoneSheet);
  $('btnCopyLink').addEventListener('click', () => {
    navigator.clipboard.writeText($('phoneLink').textContent)
      .then(() => toast('Link copied'), () => toast('Copy failed: select the link instead'));
  });
}

// ── Open on a phone: the link with the access key, as a QR code ──
async function openPhoneSheet() {
  let url = '';
  try {
    const res = await fetch('/api/connect');
    if (res.ok) url = ((await res.json()).phone_urls || [])[0] || '';
  } catch { /* offline */ }

  // a private network address means a --field server (same Wi-Fi, self-signed certificate)
  const lan = /^https?:\/\/(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(url);
  let how = lan
    ? 'Connect the phone to the same Wi-Fi, scan the code, then accept the certificate warning once (Advanced → Proceed).'
    : 'Scan the code with the phone camera. Works on any network.';
  if (!window.qrcode) how = 'QR code unavailable offline: open this link on the phone.';
  $('phoneNote').textContent = url
    ? `${how} The link contains the access key, so share it only with your team.`
    : 'No phone link: start the server with python app.py --tunnel (any network) or --field (same Wi-Fi).';

  $('phoneQr').innerHTML = '';
  if (url && window.qrcode) {
    const qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    $('phoneQr').innerHTML = qr.createSvgTag({ cellSize: 5, margin: 3, scalable: true });
  }
  $('phoneLink').textContent = url;
  $('phoneLink').hidden = $('btnCopyLink').hidden = !url;
  $('phoneSheet').showModal();
}

document.addEventListener('DOMContentLoaded', () => {
  // The access key came in the link; it now lives in a cookie, so hide it from the address bar
  if (new URLSearchParams(location.search).has('key')) history.replaceState(null, '', location.pathname);
  initControls();
  initGestures();
  requestAnimationFrame(draw);
  loop();
  startCamera();
});
