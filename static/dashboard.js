/* ═══════════════════════════════════════════════
   dashboard.js — Face Mask Detection Dashboard Controller
   ═══════════════════════════════════════════════ */

'use strict';

// ── State ─────────────────────────────────────────────────────────────────────
let donutChart = null;
let timelineChart = null;
let pollInterval = null;
const POLL_MS = 1000;

// ── Init ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  initDonutChart();
  initTimelineChart();
  startPolling();
  refreshLogs();
  initPhotoAnalysis();
  loadModelCard();
  showToast('Dashboard loaded — click ▶ Start to begin detection', 'info');
});

// ── Charts ────────────────────────────────────────────────────────────────────
function initDonutChart() {
  const ctx = document.getElementById('donutChart').getContext('2d');
  donutChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['With Mask', 'No Mask', 'Incorrect'],
      datasets: [{
        data: [0, 0, 0],
        backgroundColor: ['#10b981', '#ef4444', '#f59e0b'],
        borderColor: ['#0a0d14', '#0a0d14', '#0a0d14'],
        borderWidth: 3,
        hoverOffset: 6
      }]
    },
    options: {
      cutout: '72%',
      responsive: false,
      animation: { animateRotate: true, duration: 600 },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => ` ${ctx.label}: ${ctx.raw} faces`
          },
          backgroundColor: 'rgba(17,24,39,0.95)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 10,
          cornerRadius: 8
        }
      }
    }
  });
}

function initTimelineChart() {
  const ctx = document.getElementById('timelineChart').getContext('2d');
  timelineChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: [],
      datasets: [
        {
          label: 'Compliance %',
          data: [],
          borderColor: '#00d4ff',
          backgroundColor: 'rgba(0, 212, 255, 0.08)',
          fill: true,
          tension: 0.4,
          borderWidth: 2,
          pointRadius: 2,
          pointHoverRadius: 5,
          pointBackgroundColor: '#00d4ff'
        },
        {
          label: 'With Mask',
          data: [],
          borderColor: '#10b981',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [4, 2]
        },
        {
          label: 'No Mask',
          data: [],
          borderColor: '#ef4444',
          backgroundColor: 'transparent',
          fill: false,
          tension: 0.4,
          borderWidth: 1.5,
          pointRadius: 0,
          borderDash: [4, 2]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 400 },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: { color: '#475569', maxTicksLimit: 8, maxRotation: 0, font: { size: 10 } }
        },
        y: {
          min: 0,
          max: 100,
          grid: { color: 'rgba(255,255,255,0.04)', drawBorder: false },
          ticks: {
            color: '#475569',
            font: { size: 10 },
            callback: v => v + '%'
          }
        }
      },
      plugins: {
        legend: {
          labels: {
            color: '#94a3b8', font: { size: 11 },
            boxWidth: 16, padding: 14
          }
        },
        tooltip: {
          backgroundColor: 'rgba(17,24,39,0.95)',
          borderColor: 'rgba(255,255,255,0.08)',
          borderWidth: 1,
          titleColor: '#f1f5f9',
          bodyColor: '#94a3b8',
          padding: 10,
          cornerRadius: 8
        }
      }
    }
  });
}

// ── Update Charts from Stats ──────────────────────────────────────────────────
function updateDonut(stats) {
  const d = donutChart.data.datasets[0];
  d.data = [stats.with_mask || 0, stats.without_mask || 0, stats.mask_weared_incorrect || 0];
  donutChart.update('none');
}

function updateTimeline(timeline) {
  if (!Array.isArray(timeline) || timeline.length === 0) return;
  const labels = timeline.map(t => t.ts);
  const compliance = timeline.map(t => t.compliance_pct || 0);
  const withMask = timeline.map(t => t.with_mask || 0);
  const noMask = timeline.map(t => t.without_mask || 0);

  timelineChart.data.labels = labels;
  timelineChart.data.datasets[0].data = compliance;
  timelineChart.data.datasets[1].data = withMask;
  timelineChart.data.datasets[2].data = noMask;
  timelineChart.update('none');
}

// ── Polling ────────────────────────────────────────────────────────────────────
function startPolling() {
  if (pollInterval) clearInterval(pollInterval);
  pollInterval = setInterval(poll, POLL_MS);
  poll(); // immediate first call
}

async function poll() {
  try {
    const [statsRes, timelineRes] = await Promise.all([
      fetch('/api/stats'),
      fetch('/api/timeline')
    ]);
    if (!statsRes.ok || !timelineRes.ok) return;

    const stats = await statsRes.json();
    const timeline = await timelineRes.json();

    updateStatsUI(stats);
    updateDonut(stats);
    updateTimeline(timeline);
  } catch (e) {
    // Network error — ignore silently
  }
}

// ── Update UI Elements ─────────────────────────────────────────────────────────
function updateStatsUI(stats) {
  setText('statTotal', stats.total || 0);
  setText('statMask', stats.with_mask || 0);
  setText('statNoMask', stats.without_mask || 0);
  setText('statIncorrect', stats.mask_weared_incorrect || 0);

  const pct = stats.compliance_pct || 0;
  setText('donutPct', pct.toFixed(0) + '%');
  document.getElementById('donutPct').style.color = complianceColor(pct);

  setText('sumFrames', (stats.frames_processed || 0).toLocaleString());
  setText('sumFps', (stats.fps || 0).toFixed(1));
  setText('sumActive', stats.stream_active ? '✅ Yes' : '❌ No');
  setText('sumCompliance', pct.toFixed(1) + '%');
  setText('fpsBadge', (stats.fps || 0).toFixed(0) + ' FPS');
  setText('complianceBarPct', pct.toFixed(1) + '%');
  document.getElementById('complianceFill').style.width = pct + '%';

  if (stats.session_id) {
    const sid = stats.session_id.slice(0, 20);
    setText('sessionId', `Session: ${sid}…`);
  }

  // Stream status badge
  const dot = document.getElementById('streamStatus').querySelector('.status-dot');
  const txt = document.getElementById('streamStatus').querySelector('.status-text');
  if (stats.stream_active) {
    dot.classList.add('active');
    dot.classList.remove('inactive');
    txt.textContent = 'Live';
  } else {
    dot.classList.remove('active');
    dot.classList.add('inactive');
    txt.textContent = 'Inactive';
  }
}

function complianceColor(pct) {
  if (pct >= 80) return '#10b981';
  if (pct >= 50) return '#f59e0b';
  return '#ef4444';
}

function setText(id, val) {
  const el = document.getElementById(id);
  if (el && el.textContent !== String(val)) el.textContent = val;
}

// ── Stream Controls ────────────────────────────────────────────────────────────
async function startStream() {
  try {
    const res = await fetch('/api/stream/start', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'started' || data.status === 'already_running') {
      document.getElementById('btnStart').disabled = true;
      document.getElementById('btnStop').disabled = false;
      showToast('🟢 Stream started', 'success');
      // Reload image src to reconnect MJPEG stream
      const feed = document.getElementById('videoFeed');
      feed.src = '/video_feed?' + Date.now();
    }
  } catch (e) {
    showToast('❌ Failed to start stream', 'error');
  }
}

async function stopStream() {
  try {
    await fetch('/api/stream/stop', { method: 'POST' });
    document.getElementById('btnStart').disabled = false;
    document.getElementById('btnStop').disabled = true;
    showToast('🔴 Stream stopped', 'info');
  } catch (e) {
    showToast('❌ Failed to stop stream', 'error');
  }
}

// ── Control Buttons ────────────────────────────────────────────────────────────
async function takeScreenshot() {
  try {
    const res = await fetch('/api/screenshot', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'saved') {
      showToast(`📸 Screenshot saved: ${data.path.split(/[\\/]/).pop()}`, 'success');
    } else {
      showToast('⚠ No frame available', 'warning');
    }
  } catch (e) {
    showToast('❌ Screenshot failed', 'error');
  }
}

async function saveHeatmap() {
  try {
    const res = await fetch('/api/heatmap', { method: 'POST' });
    const data = await res.json();
    if (data.status === 'saved') {
      showToast(`🗺 Heatmap saved: ${data.path.split(/[\\/]/).pop()}`, 'success');
    } else {
      showToast('⚠ No frame available', 'warning');
    }
  } catch (e) {
    showToast('❌ Heatmap generation failed', 'error');
  }
}

async function resetStats() {
  if (!confirm('Reset all session statistics and logs?')) return;
  try {
    await fetch('/api/reset', { method: 'POST' });
    // Clear charts
    timelineChart.data.labels = [];
    timelineChart.data.datasets.forEach(d => d.data = []);
    timelineChart.update();
    donutChart.data.datasets[0].data = [0, 0, 0];
    donutChart.update();
    document.getElementById('logBody').innerHTML =
      '<tr class="log-empty"><td colspan="5">No violations logged yet.</td></tr>';
    showToast('🔄 Stats reset', 'info');
  } catch (e) {
    showToast('❌ Reset failed', 'error');
  }
}

async function refreshLogs() {
  try {
    const res = await fetch('/api/logs?n=50');
    const rows = await res.json();
    renderLogs(rows);
  } catch (e) { /* pass */ }
}

function renderLogs(rows) {
  const tbody = document.getElementById('logBody');
  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr class="log-empty"><td colspan="5">No violations logged yet.</td></tr>';
    return;
  }
  const violations = rows.filter(r => r.label !== 'with_mask').reverse().slice(0, 50);
  if (violations.length === 0) {
    tbody.innerHTML = '<tr class="log-empty"><td colspan="5">✅ No mask violations detected!</td></tr>';
    return;
  }
  tbody.innerHTML = violations.map(r => {
    const ts = r.timestamp ? r.timestamp.replace('T', ' ').slice(0, 19) : '—';
    const conf = r.confidence ? (parseFloat(r.confidence) * 100).toFixed(1) + '%' : '—';
    const badge = badgeHTML(r.label);
    return `<tr>
      <td>${ts}</td>
      <td>${r.face_id ?? '—'}</td>
      <td>${badge}</td>
      <td>${conf}</td>
      <td>${r.frame_number ?? '—'}</td>
    </tr>`;
  }).join('');
}

function badgeHTML(label) {
  if (label === 'with_mask') return `<span class="badge badge-green">✅ With Mask</span>`;
  if (label === 'without_mask') return `<span class="badge badge-red">❌ No Mask</span>`;
  if (label === 'mask_weared_incorrect') return `<span class="badge badge-orange">⚠️ Incorrect</span>`;
  return `<span class="badge">${label}</span>`;
}

function downloadLog() {
  window.location.href = '/api/logs/download';
  showToast('⬇ Downloading violations.csv…', 'info');
}

// ── Analyze a Photo ────────────────────────────────────────────────────────────
const BOX_COLORS = { with_mask: '#10b981', without_mask: '#ef4444', mask_weared_incorrect: '#f59e0b' };
const SHORT_LABELS = { with_mask: 'mask', without_mask: 'no mask', mask_weared_incorrect: 'incorrect' };

function initPhotoAnalysis() {
  const input = document.getElementById('photoInput');
  const drop = document.getElementById('analyzeDrop');
  input.addEventListener('change', () => { if (input.files[0]) analyzePhoto(input.files[0]); input.value = ''; });
  ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => {
    e.preventDefault(); drop.classList.remove('dragover');
  }));
  drop.addEventListener('drop', e => {
    const file = e.dataTransfer.files[0];
    if (file) analyzePhoto(file);
  });
}

async function analyzePhoto(file) {
  if (!file.type.startsWith('image/')) { showToast('Please choose an image file', 'warning'); return; }
  const summary = document.getElementById('analyzeSummary');
  summary.textContent = 'Analysing…';
  const form = new FormData();
  form.append('image', file);
  try {
    const res = await fetch('/api/predict', { method: 'POST', body: form });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.statusText);
    await drawAnalysis(file, data);
    renderAnalysisSummary(data);
  } catch (e) {
    summary.textContent = '';
    showToast(`Analysis failed: ${e.message}`, 'error');
  }
}

function drawAnalysis(file, data) {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.onload = () => {
      const canvas = document.getElementById('analyzeCanvas');
      const ctx = canvas.getContext('2d');
      // Small photos are drawn larger so crowded labels stay readable
      const scale = Math.max(1, 800 / img.naturalWidth);
      canvas.width = Math.round(img.naturalWidth * scale);
      canvas.height = Math.round(img.naturalHeight * scale);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      const line = Math.max(2, Math.round(canvas.width / 400));
      ctx.font = `${Math.max(11, line * 6)}px Inter, sans-serif`;
      data.faces.forEach(f => {
        const [x1, y1, x2, y2] = f.box.map(v => v * scale);
        const color = BOX_COLORS[f.label] || '#ffffff';
        ctx.lineWidth = line;
        ctx.strokeStyle = color;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${SHORT_LABELS[f.label] || f.label} ${(f.confidence * 100).toFixed(0)}%`;
        const tw = ctx.measureText(text).width + 6;
        const th = parseInt(ctx.font, 10) + 4;
        ctx.fillStyle = color;
        ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(text, x1 + 3, Math.max(th - 4, y1 - 4));
      });
      document.getElementById('analyzeHint').hidden = true;
      canvas.hidden = false;
      URL.revokeObjectURL(url);
      resolve();
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('the browser could not display this image')); };
    img.src = url;
  });
}

function renderAnalysisSummary(data) {
  const c = data.counts;
  const summary = document.getElementById('analyzeSummary');
  if (!c.total) {
    summary.textContent = `No faces found (${data.inference_ms} ms).`;
    return;
  }
  summary.innerHTML = [
    `<span><b>${c.total}</b> face${c.total === 1 ? '' : 's'}</span>`,
    badgeHTML('with_mask') + ` ${c.with_mask || 0}`,
    badgeHTML('without_mask') + ` ${c.without_mask || 0}`,
    badgeHTML('mask_weared_incorrect') + ` ${c.mask_weared_incorrect || 0}`,
    `<span>Compliance <b style="color:${complianceColor(data.compliance_pct)}">${data.compliance_pct}%</b></span>`,
    `<span class="mono">${data.inference_ms} ms</span>`,
  ].join('<span>·</span>');
}

// ── Model Card ─────────────────────────────────────────────────────────────────
async function loadModelCard() {
  const grid = document.getElementById('modelGrid');
  try {
    const m = await (await fetch('/api/model')).json();
    const pct = v => (v == null ? '—' : `${(v * 100).toFixed(2)}%`);
    const e2e = (m.end_to_end || []).find(r => r.backend === 'yunet') || {};
    const items = [
      ['Backbone', m.backbone || '—'],
      ['Input', m.input_size ? `${m.input_size[0]}×${m.input_size[1]}` : '—'],
      ['Test accuracy', pct(m.test && m.test.accuracy)],
      ['Test macro-F1', m.test && m.test.macro_f1 != null ? m.test.macro_f1.toFixed(3) : '—'],
      ['Calibration (ECE)', m.test && m.test.ece != null ? m.test.ece.toFixed(3) : '—'],
      ['End-to-end (photos)', pct(e2e.end_to_end_accuracy)],
    ];
    grid.innerHTML = items.map(([k, v]) =>
      `<div class="summary-item"><span class="summary-label">${k}</span>` +
      `<span class="summary-value mono">${v}</span></div>`).join('');
  } catch (e) {
    grid.innerHTML = '<div class="summary-item"><span class="summary-label">Model info unavailable</span></div>';
  }
}

// ── Toast ──────────────────────────────────────────────────────────────────────
let toastTimer = null;
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  const colors = {
    success: 'rgba(16, 185, 129, 0.25)',
    error: 'rgba(239, 68, 68, 0.25)',
    warning: 'rgba(245, 158, 11, 0.25)',
    info: 'rgba(0, 212, 255, 0.15)'
  };
  toast.textContent = message;
  toast.style.borderLeftColor = { success: '#10b981', error: '#ef4444', warning: '#f59e0b', info: '#00d4ff' }[type] || '#00d4ff';
  toast.style.borderLeftWidth = '3px';
  toast.style.borderLeftStyle = 'solid';
  toast.style.background = colors[type] || colors.info;
  toast.classList.add('show');
  if (toastTimer) clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 3500);
}

// ── Auto-refresh logs every 10 seconds ────────────────────────────────────────
setInterval(refreshLogs, 10000);

// ── Connect a phone ────────────────────────────────────────────────────────────
async function openPhoneDialog() {
  const body = document.getElementById('phoneBody');
  body.innerHTML = '<p class="phone-note">Loading…</p>';
  document.getElementById('phoneDialog').showModal();
  let urls = [];
  try {
    const res = await fetch('/api/connect');
    if (res.ok) urls = (await res.json()).phone_urls || [];
  } catch (e) { /* offline */ }

  if (!urls.length) {
    body.innerHTML = `
      <p class="phone-note">Phones need the server on the local network over HTTPS. Stop this
        server and start it in field mode:</p>
      <pre class="phone-cmd">python app.py --field</pre>
      <p class="phone-note">Then open this dialog again and scan the code with the phone.
        <a href="/field">Open field mode on this computer</a> to try it here.</p>`;
    return;
  }
  const url = urls[0];
  // a private network address means a local --field server (same Wi-Fi, self-signed certificate)
  const lan = /^https?:\/\/(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)/.test(url);
  let qrSvg = '';
  if (window.qrcode) {
    const qr = qrcode(0, 'M');
    qr.addData(url);
    qr.make();
    qrSvg = qr.createSvgTag({ cellSize: 5, margin: 3, scalable: true });
  }
  body.innerHTML = `
    <div class="phone-grid">
      <div class="phone-qr">${qrSvg || '<p class="phone-note">QR library unavailable offline: type the link instead.</p>'}</div>
      <ol class="phone-steps">${lan ? `
        <li>Connect the phone to the <b>same Wi-Fi</b> as this computer.</li>
        <li>Scan the code with the phone camera.</li>
        <li>The browser warns about the certificate once: tap <b>Advanced → Proceed</b>.</li>` : `
        <li>Scan the code with the phone camera (works on any network).</li>`}
        <li>Allow camera access. Add it to the home screen for a full-screen app.</li>
      </ol>
    </div>
    <p class="phone-note">Link (contains the access key, so share it only with your team):</p>
    <pre class="phone-cmd" id="phoneUrl"></pre>
    <button class="btn btn-sm btn-secondary" onclick="copyPhoneUrl()">⧉ Copy link</button>`;
  document.getElementById('phoneUrl').textContent = url;
}

function copyPhoneUrl() {
  const text = document.getElementById('phoneUrl').textContent;
  navigator.clipboard.writeText(text).then(
    () => showToast('Link copied', 'success'),
    () => showToast('Copy failed: select the link and copy it', 'warning'));
}
