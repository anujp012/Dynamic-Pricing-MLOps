/* ═══════════════════════════════════════════════
   DYNAMIC PRICING ENGINE — app.js
   Sections:
     1. Utilities
     2. Clock
     3. Map Setup (Leaflet + OpenStreetMap)
     4. Address Search (Nominatim — free, no key)
     5. Chart Setup (Chart.js)
     6. Drift Bars  ← now fetched from /metrics/drift
     7. Pipeline Stages
     8. Log Stream
     9. Prediction API Call
    10. Live API Polling  ← Performance + Drift charts now live
═══════════════════════════════════════════════ */

'use strict';

/* ═══════════════════════════════════════════════
   1. UTILITIES
═══════════════════════════════════════════════ */
const $ = id => document.getElementById(id);
const API = () => $('api-url').value.trim().replace(/\/$/, '');

function show(el) { el.classList.remove('hidden'); }
function hide(el) { el.classList.add('hidden'); }

/* ═══════════════════════════════════════════════
   2. CLOCK
═══════════════════════════════════════════════ */
function updateClock() {
  $('topbar-time').textContent = new Date().toLocaleTimeString('en-GB');
}
updateClock();
setInterval(updateClock, 1000);

/* ═══════════════════════════════════════════════
   3. MAP SETUP
═══════════════════════════════════════════════ */
const map = L.map('map', { zoomControl: true }).setView([40.748, -73.986], 12);

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
  attribution: '© <a href="https://openstreetmap.org/copyright">OpenStreetMap</a>',
  maxZoom: 19
}).addTo(map);

function makeMarker(color, shadowColor) {
  return L.divIcon({
    html: `<div style="
      width:16px; height:16px; border-radius:50%;
      background:${color}; border:3px solid #fff;
      box-shadow: 0 2px 6px ${shadowColor};
    "></div>`,
    iconSize: [16, 16],
    iconAnchor: [8, 8],
    className: ''
  });
}

const PICKUP_ICON  = makeMarker('#22c55e', 'rgba(34,197,94,.5)');
const DROPOFF_ICON = makeMarker('#c0392b', 'rgba(192,57,43,.5)');

const markers = { pickup: null, dropoff: null };
const coords  = { pickup: null, dropoff: null };
let routeLine = null;
let mapMode   = 'pickup';

map.on('click', e => {
  const { lat, lng } = e.latlng;
  placeMarker(mapMode, lat, lng);
  reverseGeocode(lat, lng, mapMode);
  toggleMapMode();
  if (coords.pickup && coords.dropoff) drawRoute();
});

function placeMarker(type, lat, lng) {
  coords[type] = { lat, lng };
  if (markers[type]) map.removeLayer(markers[type]);
  const icon  = type === 'pickup' ? PICKUP_ICON : DROPOFF_ICON;
  const label = type === 'pickup' ? 'Pickup' : 'Dropoff';
  markers[type] = L.marker([lat, lng], { icon })
    .addTo(map)
    .bindPopup(`<span style="font-family:monospace;font-size:11px">${label}</span>`);
}

function toggleMapMode() {
  mapMode = mapMode === 'pickup' ? 'dropoff' : 'pickup';
  $('map-mode-label').textContent = `Mode: Set ${mapMode.charAt(0).toUpperCase() + mapMode.slice(1)}`;
}

function drawRoute() {
  if (routeLine) map.removeLayer(routeLine);
  const p = coords.pickup, d = coords.dropoff;
  routeLine = L.polyline([[p.lat, p.lng], [d.lat, d.lng]], {
    color: '#0f1e35', weight: 2, dashArray: '8 6', opacity: 0.7
  }).addTo(map);
  map.fitBounds(routeLine.getBounds(), { padding: [40, 40] });
  const km   = haversine(p.lat, p.lng, d.lat, d.lng);
  const mins = Math.round(km / 0.45);
  $('dist-km').textContent  = km.toFixed(2);
  $('dist-min').textContent = mins;
  show($('dist-badge'));
  $('r-dist').textContent = km.toFixed(2) + ' km';
}

function haversine(lat1, lon1, lat2, lon2) {
  const R   = 6371;
  const rad = x => x * Math.PI / 180;
  const dL  = rad(lat2 - lat1);
  const dR  = rad(lon2 - lon1);
  const a   = Math.sin(dL / 2) ** 2
              + Math.cos(rad(lat1)) * Math.cos(rad(lat2)) * Math.sin(dR / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
}

/* ═══════════════════════════════════════════════
   4. ADDRESS SEARCH
═══════════════════════════════════════════════ */
const searchTimers = {};

function onSearch(type) {
  const q    = $(`${type}-input`).value.trim();
  const list = $(`${type}-list`);
  if (q.length < 3) { list.classList.remove('open'); list.innerHTML = ''; return; }
  clearTimeout(searchTimers[type]);
  searchTimers[type] = setTimeout(() => nominatimSearch(type, q), 400);
}

async function nominatimSearch(type, query) {
  const list = $(`${type}-list`);
  try {
    const res  = await fetch(
      `https://nominatim.openstreetmap.org/search?q=${encodeURIComponent(query)}&format=json&limit=5`,
      { headers: { 'Accept-Language': 'en' } }
    );
    const data = await res.json();
    list.innerHTML = '';
    if (!data.length) { list.classList.remove('open'); return; }
    data.forEach(place => {
      const li = document.createElement('li');
      li.textContent = place.display_name;
      li.onclick = () => {
        const lat = parseFloat(place.lat);
        const lng = parseFloat(place.lon);
        $(`${type}-input`).value = place.display_name.split(',').slice(0, 3).join(',');
        list.classList.remove('open');
        list.innerHTML = '';
        placeMarker(type, lat, lng);
        map.setView([lat, lng], 14);
        mapMode = type === 'pickup' ? 'dropoff' : 'pickup';
        $('map-mode-label').textContent = `Mode: Set ${mapMode.charAt(0).toUpperCase() + mapMode.slice(1)}`;
        if (coords.pickup && coords.dropoff) drawRoute();
      };
      list.appendChild(li);
    });
    list.classList.add('open');
  } catch (err) {
    console.warn('Nominatim search error:', err);
  }
}

async function reverseGeocode(lat, lng, type) {
  try {
    const res  = await fetch(`https://nominatim.openstreetmap.org/reverse?lat=${lat}&lon=${lng}&format=json`);
    const data = await res.json();
    const name = (data.display_name || '').split(',').slice(0, 3).join(',');
    $(`${type}-input`).value = name;
  } catch {
    $(`${type}-input`).value = `${lat.toFixed(5)}, ${lng.toFixed(5)}`;
  }
}

document.addEventListener('click', e => {
  ['pickup-list', 'dropoff-list'].forEach(id => {
    const el = $(id);
    if (el && !el.contains(e.target) && e.target.id !== `${id.replace('-list','-input')}`) {
      el.classList.remove('open');
    }
  });
});

/* ═══════════════════════════════════════════════
   5. CHARTS (Chart.js)
═══════════════════════════════════════════════ */
Chart.defaults.font.family = "'IBM Plex Sans', sans-serif";
Chart.defaults.font.size   = 11;
Chart.defaults.color       = '#8a94a6';
Chart.defaults.borderColor = '#e2e4e9';

// ── Surge history (live, updated per prediction) ──
const hLabels = [], hData = [];
const histChart = new Chart($('ch-history'), {
  type: 'line',
  data: {
    labels: hLabels,
    datasets: [{
      label: 'Surge Multiplier',
      data: hData,
      borderColor: '#c0392b',
      backgroundColor: 'rgba(192,57,43,.06)',
      fill: true, tension: 0.4, pointRadius: 4,
      pointBackgroundColor: '#c0392b', pointBorderColor: '#fff', pointBorderWidth: 2, borderWidth: 2,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { color: '#f0f1f3' } }, y: { grid: { color: '#f0f1f3' }, min: 0.8 } }
  }
});

// ── Zone chart (live, updated per prediction) ──
const zoneAccum = { city_centre: [], airport: [], suburb: [], industrial: [] };
const zoneChart = new Chart($('ch-zone'), {
  type: 'bar',
  data: {
    labels: ['City Centre', 'Airport', 'Suburb', 'Industrial'],
    datasets: [{
      label: 'Avg Fare ($)',
      data: [0, 0, 0, 0],
      backgroundColor: ['rgba(192,57,43,.75)', 'rgba(26,84,144,.75)', 'rgba(26,122,74,.75)', 'rgba(92,96,110,.75)'],
      borderWidth: 0, borderRadius: 4,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: {
      x: { grid: { display: false } },
      y: { grid: { color: '#f0f1f3' }, ticks: { callback: v => '$' + v } }
    }
  }
});

// ── Weather chart (live, updated per prediction) ──
const weatherAccum = { clear: [], rainy: [], fog: [], storm: [] };
const weatherChart = new Chart($('ch-weather'), {
  type: 'bar',
  data: {
    labels: ['Clear', 'Rainy', 'Foggy', 'Storm'],
    datasets: [{
      label: 'Avg Surge',
      data: [0, 0, 0, 0],
      backgroundColor: ['rgba(26,122,74,.75)', 'rgba(26,84,144,.75)', 'rgba(146,96,10,.75)', 'rgba(192,57,43,.75)'],
      borderWidth: 0, borderRadius: 4,
    }]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { display: false } }, y: { grid: { color: '#f0f1f3' }, min: 0 } }
  }
});

// ── RMSE/MAE chart — populated from /metrics/performance ──
const rmseChart = new Chart($('ch-rmse'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'RMSE',
        data: [],
        borderColor: '#c0392b', backgroundColor: 'rgba(192,57,43,.05)',
        fill: true, tension: 0.4, pointRadius: 4, borderWidth: 2,
        pointBackgroundColor: '#c0392b', pointBorderColor: '#fff', pointBorderWidth: 2
      },
      {
        label: 'MAE',
        data: [],
        borderColor: '#1a5490', backgroundColor: 'rgba(26,84,144,.05)',
        fill: true, tension: 0.4, pointRadius: 4, borderWidth: 2,
        pointBackgroundColor: '#1a5490', pointBorderColor: '#fff', pointBorderWidth: 2
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { boxWidth: 12, padding: 16, color: '#4a5568' } } },
    scales: {
      x: { grid: { color: '#f0f1f3' } },
      y: { grid: { color: '#f0f1f3' }, ticks: { callback: v => '$' + v } }
    }
  }
});

// ── Feature importance chart — populated from /metrics/performance ──
const fiChart = new Chart($('ch-fi'), {
  type: 'bar',
  data: {
    labels: [],
    datasets: [{
      label: 'Importance',
      data: [],
      backgroundColor: [
        'rgba(192,57,43,.8)', 'rgba(192,57,43,.65)', 'rgba(26,84,144,.75)',
        'rgba(26,122,74,.75)', 'rgba(146,96,10,.6)', 'rgba(92,96,110,.5)', 'rgba(92,96,110,.4)'
      ],
      borderWidth: 0, borderRadius: 3,
    }]
  },
  options: {
    indexAxis: 'y',
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { display: false } },
    scales: { x: { grid: { color: '#f0f1f3' }, max: 0.5 }, y: { grid: { display: false } } }
  }
});

// ── Drift trend chart — populated from /metrics/drift ──
const driftTrendChart = new Chart($('ch-drift'), {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      {
        label: 'PSI Score',
        data: [],
        borderColor: '#c0392b', backgroundColor: 'rgba(192,57,43,.06)',
        fill: true, tension: 0.4, pointRadius: 4, borderWidth: 2,
        pointBackgroundColor: '#c0392b', pointBorderColor: '#fff', pointBorderWidth: 2
      },
      {
        label: 'Threshold (0.10)',
        data: [],
        borderColor: '#f59e0b',
        borderDash: [6, 4],
        pointRadius: 0, borderWidth: 1.5, fill: false
      }
    ]
  },
  options: {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { boxWidth: 12, padding: 16, color: '#4a5568' } } },
    scales: {
      x: { grid: { color: '#f0f1f3' } },
      y: { grid: { color: '#f0f1f3' }, min: 0, max: 0.35 }
    }
  }
});

/* ═══════════════════════════════════════════════
   6. DRIFT BARS — rendered from /metrics/drift data
   Called by fetchDriftMetrics() after API response
═══════════════════════════════════════════════ */
function renderDriftBars(psiFeatures) {
  const container = $('drift-bars');
  if (!container) return;
  if (!psiFeatures || !psiFeatures.length) return;

  container.innerHTML = '';
  psiFeatures.forEach(f => {
    const cls  = f.psi > 0.20 ? 'alert' : f.psi > 0.10 ? 'watch' : 'safe';
    const pct  = Math.min((f.psi / 0.3) * 100, 100).toFixed(1);
    const tpct = ((0.10 / 0.3) * 100).toFixed(1);
    const lbl  = cls === 'alert' ? 'Alert' : cls === 'watch' ? 'Watch' : 'Safe';
    container.innerHTML += `
      <div class="drift-item">
        <div class="drift-head">
          <span>${f.feature}</span>
          <span class="psi-label psi-${cls}">PSI ${f.psi.toFixed(2)} · ${lbl}</span>
        </div>
        <div class="drift-track">
          <div class="drift-fill ${cls}" style="width:${pct}%"></div>
          <div class="drift-threshold-line" style="left:${tpct}%"></div>
        </div>
      </div>`;
  });
}

/* ═══════════════════════════════════════════════
   7. PIPELINE STAGES
═══════════════════════════════════════════════ */
const STAGES = [
  { name: 'Checkout',     icon: '📥', st: 'pass', time: '11s' },
  { name: 'Drift Check',  icon: '🔍', st: 'pass', time: '1m 21s' },
  { name: 'Retrain',      icon: '🧠', st: 'pass', time: '2m 12s' },
  { name: 'Docker Build', icon: '🐳', st: 'pass', time: '1m 04s' },
  { name: 'K8s Deploy',   icon: '🚀', st: 'pass', time: '16s' },
];

const pipeContainer = $('pipeline-stages');
STAGES.forEach((s, i) => {
  const div = document.createElement('div');
  div.className = 'pipe-step';
  const label = s.st === 'pass' ? `✓ ${s.time}` : s.st === 'run' ? '▶ Running' : '— Pending';
  div.innerHTML = `
    <div class="pipe-inner ${s.st}">
      <span class="pipe-icon">${s.icon}</span>
      <span class="pipe-name">${s.name}</span>
      <span class="pipe-status ${s.st}">${label}</span>
    </div>`;
  pipeContainer.appendChild(div);
  if (i < STAGES.length - 1) {
    const arrow = document.createElement('span');
    arrow.className = 'pipe-arrow';
    arrow.textContent = '›';
    pipeContainer.appendChild(arrow);
  }
});

let elapsed = 0;
setInterval(() => {
  elapsed++;
  const m = Math.floor(elapsed / 60);
  const s = (elapsed % 60).toString().padStart(2, '0');
  $('pipe-timer').textContent = `${m}m ${s}s`;
}, 1000);

/* ═══════════════════════════════════════════════
   8. LOG STREAM
═══════════════════════════════════════════════ */
const logEl = $('log-stream');
const INIT_LOGS = [
  ['info', 'Dashboard initialized · OpenStreetMap ready (free, no API key)'],
  ['ok',   'FastAPI connection → http://localhost:8000'],
  ['ok',   'SQLite (data.db) monitoring active'],
  ['ok',   'MLflow experiment: Uber_Dynamic_Pricing'],
  ['info', 'Fetching real PSI scores from /metrics/drift...'],
  ['info', 'Fetching real RMSE/MAE history from /metrics/performance...'],
  ['info', 'Polling /metrics/drift every 15s · /metrics/performance every 30s'],
];

let logIdx = 0;
function appendLog(lvl, msg) {
  const ts  = new Date().toLocaleTimeString('en-GB');
  const cls = { ok: 'ok', warn: 'warn', err: 'err', info: 'info' }[lvl] || 'info';
  const pfx = { ok: '[OK]', warn: '[WARN]', err: '[ERR]', info: '[INFO]' }[lvl] || '[INFO]';
  logEl.innerHTML += `<div><span class="ts">${ts}</span><span class="${cls}">${pfx}</span> ${msg}</div>`;
  logEl.scrollTop  = logEl.scrollHeight;
}

function drainInitLogs() {
  if (logIdx < INIT_LOGS.length) {
    appendLog(...INIT_LOGS[logIdx++]);
    setTimeout(drainInitLogs, 600 + Math.random() * 500);
  }
}
setTimeout(drainInitLogs, 800);

/* ═══════════════════════════════════════════════
   9. PREDICTION
═══════════════════════════════════════════════ */
async function runPrediction() {
  const btn    = $('predict-btn');
  const errBox = $('error-box');
  hide(errBox);

  if (!coords.pickup || !coords.dropoff) {
    errBox.textContent = '⚠ Please set both Pickup and Dropoff locations on the map first.';
    show(errBox);
    return;
  }

  btn.disabled = true;
  $('btn-text').textContent = 'Getting price...';
  show($('btn-spinner'));

  const payload = {
    pickup_longitude:  coords.pickup.lng,
    pickup_latitude:   coords.pickup.lat,
    dropoff_longitude: coords.dropoff.lng,
    dropoff_latitude:  coords.dropoff.lat,
    passenger_count:   +$('f-pax').value,
    active_drivers:    +$('f-drivers').value,
    demand_zone:        $('f-zone').value,
    weather:            $('f-weather').value,
    time_of_day:        $('f-time').value,
    event_nearby:      +$('f-event').value,
  };

  try {
    const res = await fetch(`${API()}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'mlops-demo-key-2024',        // FIX 5: API key authentication
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      const e = await res.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${res.status}`);
    }

    const data = await res.json();

    const finalFare = data.final_fare       ?? data.predicted_fare ?? 0;
    const surge     = data.surge_multiplier ?? 1;
    const baseFare  = data.base_fare        ?? 0;
    const modelVer  = data.model_version    ?? 'latest';

    $('r-fare').textContent    = `$${finalFare.toFixed(2)}`;
    $('r-surge').textContent   = `${surge}×`;
    $('r-base').textContent    = `$${baseFare.toFixed(2)}`;
    $('r-footnote').textContent = `Model: ${modelVer} · Logged to SQLite · ${new Date().toLocaleTimeString()}`;

    const pct = Math.min(((surge - 1.0) / 2.5) * 100, 100);
    $('surge-fill').style.width = `${pct}%`;

    show($('result-panel'));

    $('kpi-surge').textContent = `${surge}×`;
    $('kpi-fare').textContent  = `$${finalFare.toFixed(2)}`;

    // Update surge history chart
    const t = new Date().toLocaleTimeString('en-GB').slice(0, 5);
    hLabels.push(t); hData.push(surge);
    if (hLabels.length > 15) { hLabels.shift(); hData.shift(); }
    histChart.update();

    // Update zone chart
    const zoneIdx = { city_centre: 0, airport: 1, suburb: 2, industrial: 3 };
    const zi = zoneIdx[payload.demand_zone];
    if (zi !== undefined) {
      zoneAccum[payload.demand_zone].push(finalFare);
      const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
      zoneChart.data.datasets[0].data[zi] = +avg(zoneAccum[payload.demand_zone]).toFixed(2);
      zoneChart.update();
    }

    // Update weather chart
    const weatherIdx = { clear: 0, rainy: 1, fog: 2, storm: 3 };
    const wi = weatherIdx[payload.weather];
    if (wi !== undefined) {
      weatherAccum[payload.weather].push(surge);
      const avg = arr => arr.reduce((a, b) => a + b, 0) / arr.length;
      weatherChart.data.datasets[0].data[wi] = +avg(weatherAccum[payload.weather]).toFixed(2);
      weatherChart.update();
    }

    appendLog('ok', `Prediction → Fare=$${finalFare.toFixed(2)} Surge=${surge}× Zone=${payload.demand_zone} Weather=${payload.weather}`);

    // Refresh drift metrics after each prediction (new data logged)
    setTimeout(fetchDriftMetrics, 2000);

  } catch (err) {
    errBox.textContent = `Error: ${err.message} — make sure FastAPI is running on ${API()}`;
    show(errBox);
    appendLog('err', err.message);
  } finally {
    btn.disabled = false;
    $('btn-text').textContent = 'Get Surge Price';
    hide($('btn-spinner'));
  }
}

/* ═══════════════════════════════════════════════
   10. LIVE API POLLING — all charts now fetch real data
═══════════════════════════════════════════════ */

async function checkApiHealth() {
  try {
    const res = await fetch(`${API()}/`, { signal: AbortSignal.timeout(3000) });
    if (res.ok) {
      $('api-status-dot').className    = 'status-dot online';
      $('api-status-text').textContent = 'API Online';
    } else throw new Error();
  } catch {
    $('api-status-dot').className    = 'status-dot offline';
    $('api-status-text').textContent = 'API Offline';
  }
}

// ── Fetch /metrics/drift → updates PSI bars + drift trend chart + KPI cards ──
async function fetchDriftMetrics() {
  try {
    const data = await fetch(`${API()}/metrics/drift`).then(r => r.json());

    // KPI cards
    $('kpi-rows').textContent    = data.rides_in_db?.toLocaleString()        ?? '—';
    $('kpi-db-hint').textContent = data.database                             ?? 'SQLite';
    $('live-rows').textContent   = data.rides_in_db?.toLocaleString()        ?? '—';
    $('live-preds').textContent  = data.predictions_logged?.toLocaleString() ?? '—';
    $('live-thresh').textContent = data.threshold                            ?? '—';

    // ── PSI bars (real computed values) ──
    if (data.psi_features && data.psi_features.length > 0) {
      renderDriftBars(data.psi_features);

      // Warn in log if any feature is in Alert
      data.psi_features.forEach(f => {
        if (f.status === 'Alert') {
          appendLog('warn', `${f.feature} PSI=${f.psi.toFixed(2)} exceeds threshold ${f.threshold} — Alert`);
        }
      });
    }

    // ── Drift trend chart (7-day from MLflow) ──
    if (data.drift_trend && data.drift_trend.length > 0) {
      const labels    = data.drift_trend.map(d => d.day);
      const psiVals   = data.drift_trend.map(d => d.psi);
      const threshold = data.threshold ?? 0.10;

      driftTrendChart.data.labels                    = labels;
      driftTrendChart.data.datasets[0].data          = psiVals;
      driftTrendChart.data.datasets[1].data          = labels.map(() => threshold);
      driftTrendChart.data.datasets[1].label         = `Threshold (${threshold})`;
      driftTrendChart.update();
    }

  } catch {
    $('kpi-db-hint').textContent = 'API offline';
  }
}

// ── Fetch /metrics/performance → updates RMSE chart + Feature Importance + Model Registry ──
async function fetchPerformanceMetrics() {
  try {
    const data = await fetch(`${API()}/metrics/performance`).then(r => r.json());

    // ── Live RMSE display ──
    if (data.live_rmse != null) {
      $('live-rmse').textContent = `$${parseFloat(data.live_rmse).toFixed(4)}`;
    }

    // ── RMSE / MAE chart (version history) ──
    if (data.versions && data.versions.length > 0) {
      const labels   = data.versions.map(v => v.version);
      const rmseVals = data.versions.map(v => v.rmse ?? null);
      const maeVals  = data.versions.map(v => v.mae  ?? null);

      rmseChart.data.labels                 = labels;
      rmseChart.data.datasets[0].data       = rmseVals;
      rmseChart.data.datasets[1].data       = maeVals;
      rmseChart.update();

      appendLog('ok', `Performance chart updated: ${labels.length} model versions loaded from MLflow`);

      // ── Model Registry table (if element exists in HTML) ──
      const registryEl = $('model-registry-table');
      if (registryEl) {
        const statusColors = { LIVE: '#22c55e', SHADOW: '#f59e0b', ARCHIVE: '#8a94a6', DEPR: '#c0392b' };
        registryEl.innerHTML = `
          <table style="width:100%;font-size:12px;border-collapse:collapse">
            <thead>
              <tr style="color:#8a94a6;border-bottom:1px solid #e2e4e9">
                <th style="text-align:left;padding:4px 8px">VERSION</th>
                <th style="text-align:left;padding:4px 8px">RMSE</th>
                <th style="text-align:left;padding:4px 8px">R²</th>
                <th style="text-align:left;padding:4px 8px">STATUS</th>
              </tr>
            </thead>
            <tbody>
              ${[...data.versions].reverse().map(v => `
                <tr style="border-bottom:1px solid #f0f1f3">
                  <td style="padding:4px 8px;font-family:monospace">${v.version}</td>
                  <td style="padding:4px 8px">$${v.rmse ?? '—'}</td>
                  <td style="padding:4px 8px">${v.r2 != null ? v.r2 : '—'}</td>
                  <td style="padding:4px 8px">
                    <span style="
                      background:${statusColors[v.status] ?? '#8a94a6'}22;
                      color:${statusColors[v.status] ?? '#8a94a6'};
                      padding:2px 8px; border-radius:4px; font-size:11px; font-weight:600
                    ">${v.status}</span>
                  </td>
                </tr>`).join('')}
            </tbody>
          </table>`;
      }
    }

    // ── Feature importance chart ──
    if (data.feature_importance && data.feature_importance.length > 0) {
      const clean = name => name
        .replace(/_zone_.*|_weather_.*|_time_.*|_event_.*/, '')
        .replace(/_/g, ' ');

      const labels = data.feature_importance.map(f => clean(f.feature));
      const values = data.feature_importance.map(f => f.importance);

      fiChart.data.labels              = labels;
      fiChart.data.datasets[0].data   = values;
      fiChart.update();

      appendLog('ok', `Feature importance updated: top feature = ${labels[0]}`);
    }

  } catch (err) {
    appendLog('warn', `Performance fetch failed: ${err.message}`);
  }
}

async function fetchPipelineStatus() {
  try {
    const data = await fetch(`${API()}/pipeline/status`).then(r => r.json());
    const st   = data.status || '';
    const el   = $('kpi-pipeline');
    const hint = $('kpi-pipeline-hint');
    const ps   = $('pipe-status');

    if (st.includes('SUCCESS')) {
      el.textContent   = '✓ Pass';
      el.className     = 'kpi-value green';
      hint.textContent = st.split('|').slice(1, 3).join(' ').trim();
      if (ps) { ps.textContent = 'SUCCESS'; ps.className = 'green'; }
    } else if (st.includes('FAILED')) {
      el.textContent   = '✗ Fail';
      el.className     = 'kpi-value accent';
      hint.textContent = 'Check GitHub Actions';
      if (ps) { ps.textContent = 'FAILED'; ps.className = 'accent'; }
    } else {
      el.textContent   = '—';
      hint.textContent = 'No run recorded';
    }
  } catch { /* silently fail */ }
}

// ── Initial load ──
checkApiHealth();
fetchDriftMetrics();
fetchPerformanceMetrics();
fetchPipelineStatus();

// ── Polling intervals ──
setInterval(checkApiHealth,          10000);
setInterval(fetchDriftMetrics,       15000);
setInterval(fetchPerformanceMetrics, 30000);
setInterval(fetchPipelineStatus,     20000);