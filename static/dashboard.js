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
