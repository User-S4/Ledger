/**
 * ZeroTrace Ledger — Security Operations Center Frontend (Person 5)
 * 100% Dynamic Telemetry, In-Browser Simulation Controls, Live Logs Feed, and Real-Time Threat Attribution.
 */

const API_BASE = window.location.origin.includes("http") 
    ? window.location.origin 
    : "http://127.0.0.1:8000";

const MAX_HISTORY = 30;

// Streaming Data Buffers
let timeLabels = [];
let coverageHistory = [];
let velocityHistory = [];
let rpsHistory = [];
let lastTotalRequests = null;
let lastTimestamp = Date.now();

// Chart Instances
let coverageChartInstance = null;
let velocityChartInstance = null;
let rpsChartInstance = null;

Chart.defaults.color = '#8a99ad';
Chart.defaults.font.family = "'JetBrains Mono', monospace";

function initCharts() {
    // 1. Cumulative Manifold Coverage (LIVE)
    const cCtx = document.getElementById('coverageChart').getContext('2d');
    coverageChartInstance = new Chart(cCtx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [{
                label: 'Cumulative Cells Revealed',
                data: coverageHistory,
                borderColor: '#00e5ff',
                backgroundColor: 'rgba(0, 229, 255, 0.08)',
                borderWidth: 2.5,
                tension: 0.3,
                fill: true,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Total Latent Cells', color: '#566579' }
                },
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 6 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // 2. Discovery Velocity Stream (LIVE)
    const vCtx = document.getElementById('velocityChart').getContext('2d');
    velocityChartInstance = new Chart(vCtx, {
        type: 'line',
        data: {
            labels: timeLabels,
            datasets: [{
                label: 'New Cells / Window',
                data: velocityHistory,
                borderColor: '#2979ff',
                backgroundColor: 'rgba(41, 121, 255, 0.1)',
                borderWidth: 2,
                tension: 0.35,
                fill: true,
                pointRadius: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Discoveries in Window', color: '#566579' }
                },
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 5 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // 3. Traffic Throughput RPS (LIVE)
    const rCtx = document.getElementById('rpsChart').getContext('2d');
    rpsChartInstance = new Chart(rCtx, {
        type: 'bar',
        data: {
            labels: timeLabels,
            datasets: [{
                label: 'Requests / Second',
                data: rpsHistory,
                backgroundColor: 'rgba(0, 230, 118, 0.65)',
                borderColor: '#00e676',
                borderWidth: 1,
                borderRadius: 4,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: false,
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    title: { display: true, text: 'Requests / Sec', color: '#566579' }
                },
                x: {
                    grid: { display: false },
                    ticks: { maxTicksLimit: 5 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });
}

// Telemetry Polling Loop
async function pollStats() {
    try {
        const res = await fetch(`${API_BASE}/stats`);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const stats = await res.json();

        updateKPIs(stats);
        updateThreatAlert(stats);
        updateLiveCharts(stats);
        updateFlaggedTable(stats);
        updateSimBadge(stats);

        // Header Status
        document.getElementById('api-status-dot').className = 'status-dot green';
        document.getElementById('api-status-text').innerText = 'CONNECTED';
        document.getElementById('run-id-badge').innerText = stats.run_id || 'DEFAULT';

        // Defense Button
        const defBtn = document.getElementById('toggle-defense-btn');
        const defLabel = document.getElementById('defense-btn-state');
        if (stats.defense_enabled) {
            defBtn.className = 'defense-toggle-btn active';
            defLabel.innerText = `ON (${(stats.defense_mode || 'POISON').toUpperCase()})`;
        } else {
            defBtn.className = 'defense-toggle-btn disabled';
            defLabel.innerText = 'OFF (BYPASSED)';
        }

    } catch (err) {
        document.getElementById('api-status-dot').className = 'status-dot red';
        document.getElementById('api-status-text').innerText = 'OFFLINE';
    }
}

// Live Logs Feed Polling Loop
async function pollLogs() {
    try {
        const res = await fetch(`${API_BASE}/logs/recent?limit=30`);
        if (!res.ok) return;
        const logs = await res.json();
        updateLogsTable(logs);
    } catch (e) {
        // quiet error
    }
}

function updateKPIs(stats) {
    document.getElementById('kpi-total-requests').innerText = (stats.total_requests || 0).toLocaleString();
    document.getElementById('kpi-accounts-tracked').innerText = (stats.accounts_tracked || 0).toLocaleString();
    
    // Threat Posture KPI
    const threatEl = document.getElementById('kpi-threat-posture');
    const badgeEl = document.getElementById('threat-posture-badge');
    if (threatEl && badgeEl) {
        if (stats.threat_level === "attack") {
            threatEl.innerText = "UNDER ATTACK";
            threatEl.style.color = "var(--accent-red)";
            badgeEl.style.color = "var(--accent-red)";
            badgeEl.style.borderColor = "var(--accent-red)";
        } else {
            threatEl.innerText = "NOMINAL (SECURE)";
            threatEl.style.color = "var(--accent-green)";
            badgeEl.style.color = "var(--accent-green)";
            badgeEl.style.borderColor = "rgba(0, 230, 118, 0.3)";
        }
    }

    // Defense Interceptions KPI
    const interceptEl = document.getElementById('kpi-defense-interceptions');
    if (interceptEl) {
        const interceptions = stats.defense_interceptions || 0;
        interceptEl.innerText = interceptions.toLocaleString();
        if (interceptions > 0) {
            interceptEl.style.color = "var(--accent-red)";
        } else {
            interceptEl.style.color = "var(--text-primary)";
        }
    }
}

function updateThreatAlert(stats) {
    const banner = document.getElementById('threat-alert-banner');
    const icon = document.getElementById('threat-icon');
    const title = document.getElementById('threat-title');
    const desc = document.getElementById('threat-desc');
    const metricNum = document.getElementById('threat-metric-num');

    const suspicious = stats.suspicious_accounts || 0;
    metricNum.innerText = suspicious;

    if (stats.threat_level === "attack") {
        banner.className = 'threat-banner danger';
        icon.innerText = '🚨';
        title.innerText = 'DISTRIBUTED EXTRACTION DETECTED — ACTIVE FIGHTBACK';
        desc.innerText = `ZeroTrace Spatial Ledger has isolated coordinated latent exploration across ${suspicious} keys. Silent decision-boundary poisoning (swap_top2) is feeding compromised supervision to attackers.`;
    } else {
        banner.className = 'threat-banner normal';
        icon.innerText = '🛡️';
        title.innerText = 'SYSTEM SECURE — NORMAL TRAFFIC';
        desc.innerText = 'Manifold cell discoveries are within expected baseline rates. Multi-tenant office and business requests conform to typical operational clusters.';
    }
}

function updateLiveCharts(stats) {
    const now = Date.now();
    const timeStr = new Date(now).toTimeString().split(' ')[0];

    // Compute live RPS
    let rps = 0;
    if (lastTotalRequests !== null) {
        const deltaReq = (stats.total_requests || 0) - lastTotalRequests;
        const deltaSec = Math.max(0.2, (now - lastTimestamp) / 1000.0);
        rps = Math.max(0, Math.round(deltaReq / deltaSec));
    }
    lastTotalRequests = stats.total_requests || 0;
    lastTimestamp = now;

    // Update tag in header
    const rpsTag = document.getElementById('live-rps-tag');
    if (rpsTag) {
        rpsTag.innerText = `${rps} req/s`;
    }

    // Shift buffers
    timeLabels.push(timeStr);
    coverageHistory.push(stats.cells_revealed || 0);
    velocityHistory.push(stats.discoveries_in_window || 0);
    rpsHistory.push(rps);

    if (timeLabels.length > MAX_HISTORY) {
        timeLabels.shift();
        coverageHistory.shift();
        velocityHistory.shift();
        rpsHistory.shift();
    }

    // Update Coverage Chart
    if (coverageChartInstance) {
        if (stats.threat_level === "attack") {
            coverageChartInstance.data.datasets[0].borderColor = '#ff1744';
            coverageChartInstance.data.datasets[0].backgroundColor = 'rgba(255, 23, 68, 0.12)';
        } else {
            coverageChartInstance.data.datasets[0].borderColor = '#00e5ff';
            coverageChartInstance.data.datasets[0].backgroundColor = 'rgba(0, 229, 255, 0.08)';
        }
        coverageChartInstance.update();
    }

    // Update Velocity Chart
    if (velocityChartInstance) {
        velocityChartInstance.data.datasets[0].borderColor = stats.threat_level === "attack" ? '#ff1744' : '#2979ff';
        velocityChartInstance.update();
    }

    // Update RPS Chart
    if (rpsChartInstance) {
        rpsChartInstance.data.datasets[0].backgroundColor = stats.threat_level === "attack" 
            ? 'rgba(255, 23, 68, 0.7)' 
            : 'rgba(0, 230, 118, 0.65)';
        rpsChartInstance.data.datasets[0].borderColor = stats.threat_level === "attack" ? '#ff1744' : '#00e676';
        rpsChartInstance.update();
    }
}

function updateFlaggedTable(stats) {
    const tbody = document.getElementById('flagged-tbody');
    const countTag = document.getElementById('flagged-count-tag');
    const keys = stats.flagged_accounts || [];

    countTag.innerText = `${stats.suspicious_accounts || 0} Suspect Keys`;

    if (keys.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="3">No suspicious keys identified. All current traffic conforms to expected operational profiles.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = keys.slice(0, 15).map(k => `
        <tr>
            <td class="key-tag">${k}</td>
            <td><span class="badge-tag threat">ATTACK POOL</span></td>
            <td><span class="badge-tag poison">SILENT POISON</span></td>
        </tr>
    `).join('');
}

function updateLogsTable(logs) {
    const tbody = document.getElementById('live-logs-tbody');
    if (!tbody) return;

    if (!logs || logs.length === 0) {
        tbody.innerHTML = `
            <tr class="empty-row">
                <td colspan="7">No requests logged in this session. Start a traffic stream to watch queries live.</td>
            </tr>
        `;
        return;
    }

    tbody.innerHTML = logs.map(row => {
        const isPoison = row.action === "POISONED";
        const badgeClass = isPoison ? "log-badge-poison" : "log-badge-clean";
        return `
            <tr>
                <td>${row.time}</td>
                <td class="key-tag">${row.key_id}</td>
                <td>${row.ip}</td>
                <td><strong>${row.label}</strong> <span style="color:var(--text-muted)">(${row.confidence})</span></td>
                <td>${row.cell_id}</td>
                <td><span class="${badgeClass}">${row.action}</span></td>
                <td>${row.latency_ms} ms</td>
            </tr>
        `;
    }).join('');
}

function updateSimBadge(stats) {
    const badge = document.getElementById('sim-status-badge');
    const sim = stats.sim_status || {};
    if (sim.running) {
        if (sim.mode === "honest") {
            badge.className = "sim-badge active";
            badge.innerText = `STREAMING HONEST MULTI-PROFILE TRAFFIC (~10 req/s) [${sim.requests_sent || 0}]`;
        } else {
            badge.className = "sim-badge attack";
            badge.innerText = `STREAMING 400-KEY ATTACK (~25 req/s) [${sim.requests_sent || 0}]`;
        }
    } else {
        badge.className = "sim-badge";
        badge.innerText = "STATUS: IDLE";
    }
}

// Interactive Simulation Controls
function setupSimButtons() {
    document.getElementById('btn-sim-honest').addEventListener('click', async () => {
        await fetch(`${API_BASE}/sim/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'honest' })
        });
        pollStats();
        pollLogs();
    });

    document.getElementById('btn-sim-attack').addEventListener('click', async () => {
        await fetch(`${API_BASE}/sim/start`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mode: 'attack' })
        });
        pollStats();
        pollLogs();
    });

    document.getElementById('btn-sim-stop').addEventListener('click', async () => {
        await fetch(`${API_BASE}/sim/stop`, { method: 'POST' });
        pollStats();
        pollLogs();
    });

    document.getElementById('btn-sim-reset').addEventListener('click', async () => {
        await fetch(`${API_BASE}/sim/reset`, { method: 'POST' });
        timeLabels.length = 0;
        coverageHistory.length = 0;
        velocityHistory.length = 0;
        rpsHistory.length = 0;
        lastTotalRequests = null;
        pollStats();
        pollLogs();
    });

    document.getElementById('toggle-defense-btn').addEventListener('click', async () => {
        await fetch(`${API_BASE}/defense/toggle`, { method: 'POST' });
        pollStats();
    });
}

// App Launch
window.addEventListener('DOMContentLoaded', () => {
    initCharts();
    setupSimButtons();
    pollStats();
    pollLogs();
    setInterval(pollStats, 1000);
    setInterval(pollLogs, 1000);
});
