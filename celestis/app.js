// ═══════════════════════════════════════════════════════════════════
// I-STAR Mission Control — Core Logic
// ═══════════════════════════════════════════════════════════════════

const BACKEND_URL = 'http://127.0.0.1:8001';
const NUM_SATS = 36;
let liveMode = false;
let autoAuth = false;
let istarFleet = [];
let altChart = null;
let pollTimeout = null;

// Initialize Fleet
function initFleet() {
    for (let i = 1; i <= NUM_SATS; i++) {
        istarFleet.push({
            id: `I-STAR-${i.toString().padStart(2, '0')}`,
            plane: Math.floor((i - 1) / 12) + 1,
            alt: 500,
            fuel: 98 + (Math.random() * 2), // 98 - 100%
            status: 'nominal',
            dragDrop: 0,
            colaRisk: false
        });
    }
    renderFleetGrid();
}

function renderFleetGrid() {
    const grid = document.getElementById('fleet-grid');
    if (!grid) return;
    grid.innerHTML = istarFleet.map(sat => `<div class="node ${sat.status}" onclick="selectSat('${sat.id}')" title="${sat.id}\nAlt: ${sat.alt.toFixed(1)}km\nFuel: ${sat.fuel.toFixed(1)}%">${sat.id.split('-')[2]}</div>`).join('');
    updateFleetStats();
}

function updateFleetStats() {
    let n = 0, d = 0, c = 0, m = 0;
    istarFleet.forEach(s => {
        if (s.status === 'nominal') n++;
        else if (s.status === 'drifting') d++;
        else if (s.status === 'critical') c++;
        else if (s.status === 'maneuvering') m++;
    });
    const gn = document.getElementById('fs-nominal'); if (gn) gn.textContent = n;
    const gd = document.getElementById('fs-drifting'); if (gd) gd.textContent = d;
    const gc = document.getElementById('fs-critical'); if (gc) gc.textContent = c;
    const gm = document.getElementById('fs-maneuver'); if (gm) gm.textContent = m;

    const gf = document.getElementById('god-fleet');
    if (gf) {
        if (c > 0) { gf.textContent = 'CRITICAL'; gf.style.color = 'var(--red)'; }
        else if (d > 0) { gf.textContent = 'DRIFTING'; gf.style.color = 'var(--amber)'; }
        else { gf.textContent = 'NOMINAL'; gf.style.color = 'var(--green)'; }
    }
}

let selectedSat = 'I-STAR-01';
window.selectSat = function (id) {
    selectedSat = id;
    updateFuelUI();
    openDrawer(id);
}

window.closeDrawer = function () {
    const drawer = document.getElementById('sat-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (drawer) drawer.classList.remove('open');
    if (backdrop) backdrop.classList.remove('open');
}

function calcDeltaV(sat) {
    const dh = sat.dragDrop || 0;
    const mu = 3.986e14;
    const r = (6371 + sat.alt) * 1000;
    const v = Math.sqrt(mu / r);
    const dv = 0.5 * v * (dh * 1000) / r;   // Hohmann approx in m/s
    const isp = sat.isp_s || 1800;           // specific impulse (s)
    const g0 = 9.80665;
    const m0 = sat.mass_kg || 450;
    const mf = m0 * (1 - Math.exp(-dv / (isp * g0)));
    const thrust_mN = sat.thrust_mN || 60;
    const burnDur = (mf * 1000 * isp * g0) / thrust_mN; // seconds
    return { dh, dv: parseFloat(dv.toFixed(3)), burnDur: parseFloat(burnDur.toFixed(1)), fuelCost: parseFloat(mf.toFixed(4)) };
}

function openDrawer(id) {
    const sat = istarFleet.find(s => s.id === id);
    if (!sat) return;
    const drawer = document.getElementById('sat-drawer');
    const backdrop = document.getElementById('drawer-backdrop');
    if (!drawer) return;

    const el = (n, v) => { const e = document.getElementById(n); if (e) e.textContent = v; };

    // Header
    el('drawer-sat-id', sat.id);
    el('drawer-sat-role', (sat.role || 'Unknown') + ' - Plane ' + sat.plane);

    // Status banner
    const statusText = sat.status === 'critical' ? 'CRITICAL  DECAYING' :
        sat.status === 'drifting' ? 'DRIFTING' :
            sat.status === 'maneuvering' ? 'MANEUVERING' : 'NOMINAL ORBIT';
    const iconClass = 'drawer-status-icon status-' + (sat.status || 'nominal');
    el('drawer-status-text', statusText);
    const iconEl = document.getElementById('drawer-status-icon');
    if (iconEl) { iconEl.textContent = ''; iconEl.className = iconClass; }
    const banner = document.getElementById('drawer-status-banner');
    if (banner) banner.style.borderColor =
        sat.status === 'critical' ? 'var(--red)' :
            sat.status === 'drifting' ? 'var(--amber)' :
                sat.status === 'maneuvering' ? 'var(--cyan)' : 'var(--green)';

    // Orbital Telemetry
    const period = 2 * Math.PI * Math.sqrt(Math.pow((6371 + sat.alt) * 1000, 3) / 3.986e14) / 60;
    const velocity = Math.sqrt(3.986e14 / ((6371 + sat.alt) * 1000)) / 1000;
    el('dw-alt', sat.alt.toFixed(1) + ' km');
    el('dw-vel', velocity.toFixed(2) + ' km/s');
    el('dw-period', period.toFixed(1) + ' min');
    el('dw-drop', sat.dragDrop ? sat.dragDrop.toFixed(2) + ' km' : '0.00 km');
    const dropEl = document.getElementById('dw-drop');
    if (dropEl) dropEl.className = sat.dragDrop > 0 ? 'dc-val text-red' : 'dc-val';

    // Digital Twin Config
    el('dw-mass', sat.mass_kg ? sat.mass_kg + ' kg' : '-- kg');
    el('dw-fuel-type', sat.fuel_type ? sat.fuel_type.toUpperCase() : '--');
    el('dw-isp', sat.isp_s ? sat.isp_s + ' s' : '-- s');
    el('dw-thrust', sat.thrust_mN ? sat.thrust_mN + ' mN' : '-- mN');
    el('dw-area', sat.area_m2 ? sat.area_m2 + ' m2 (Cd ' + (sat.Cd || 2.2) + ')' : '--');
    el('dw-role', sat.role || '--');

    // Subsystem health (simulated live telemetry)
    const fuelKg = sat.fuel_kg || 12;
    const fuelPct = Math.min(100, (fuelKg / 15) * 100);
    const battV = 22 + Math.random() * 10;
    const battPct = Math.min(100, (battV / 35) * 100);
    const thrustTemp = Math.round(170 + Math.random() * 60);
    const thrustPct = Math.min(100, Math.max(0, ((thrustTemp - 100) / 200) * 100));
    const solarPct = Math.round(90 + Math.random() * 10);
    const commDbm = Math.round(-65 - Math.random() * 15);
    const commPct = Math.min(100, Math.max(0, (commDbm + 100) * 2));

    const setBar = (barId, valId, pct, val, color) => {
        const b = document.getElementById(barId);
        if (b) { b.style.width = Math.max(0, pct) + '%'; b.style.background = color; }
        el(valId, val);
    };
    setBar('dw-bat-bar', 'dw-bat', battPct, battV.toFixed(1) + ' V', battV > 24 ? 'var(--green)' : 'var(--amber)');
    setBar('dw-fuel-bar', 'dw-fuel-mass', fuelPct, fuelKg.toFixed(2) + ' kg', fuelPct > 30 ? 'var(--cyan)' : 'var(--red)');
    setBar('dw-thruster-bar', 'dw-thruster-temp', thrustPct, thrustTemp + ' C', thrustTemp > 220 ? 'var(--amber)' : 'var(--green)');
    setBar('dw-solar-bar', 'dw-solar', solarPct, solarPct + '%', 'var(--green)');
    setBar('dw-comms-bar', 'dw-comms', commPct, commDbm + ' dBm', commDbm > -80 ? 'var(--cyan)' : 'var(--amber)');

    // Delta-V calculation
    const dv = calcDeltaV(sat);
    el('dw-dh', dv.dh.toFixed(2) + ' km');
    el('dw-dv', dv.dv.toFixed(2) + ' m/s');
    el('dw-burn', dv.burnDur.toFixed(1) + ' s');
    el('dw-fuel-cost', dv.fuelCost.toFixed(3) + ' kg');
    el('dw-fuel-after', Math.max(0, fuelKg - dv.fuelCost).toFixed(3) + ' kg');

    // Maneuver log
    const logEl = document.getElementById('dw-maneuver-log');
    if (logEl) {
        logEl.innerHTML = (sat.maneuverHistory && sat.maneuverHistory.length > 0)
            ? sat.maneuverHistory.slice(-3).reverse()
                .map(m => '<div class="ml-entry"><span class="ml-time">' + m.time + '</span><span class="ml-desc">' + m.desc + '</span></div>')
                .join('')
            : '<span class="ml-empty">No maneuvers recorded this session.</span>';
    }

    const verEl = document.getElementById('drawer-verify');
    if (verEl) verEl.style.display = 'none';

    drawer.classList.add('open');
    if (backdrop) backdrop.classList.add('open');
}

function updateFuelUI() {
    const sat = istarFleet.find(s => s.id === selectedSat);
    if (!sat) return;

    const sn = document.getElementById('sel-sat-name'); if (sn) sn.textContent = sat.id;
    const sf = document.getElementById('sel-sat-fuel'); if (sf) sf.textContent = `${sat.fuel.toFixed(1)}% (Argon)`;
    const fb = document.getElementById('fuel-bar'); if (fb) fb.style.width = `${sat.fuel}%`;

    if (sat.fuel < 20) {
        if (fb) fb.style.background = 'var(--red)';
        if (sf) sf.style.color = 'var(--red)';
    } else if (sat.fuel < 50) {
        if (fb) fb.style.background = 'var(--amber)';
        if (sf) sf.style.color = 'var(--amber)';
    } else {
        if (fb) fb.style.background = 'var(--cyan)';
        if (sf) sf.style.color = 'var(--cyan)';
    }

    // Auth button + Ripple button logic
    const btn = document.getElementById('auth-btn');
    const rippleBtn = document.getElementById('ripple-trigger-btn');
    if (!btn) return;
    if (sat.status === 'critical' || sat.status === 'drifting') {
        btn.disabled = false;
        btn.textContent = 'AUTHORIZE ISTRAC UPLINK';
        document.getElementById('req-dv').textContent = (sat.dragDrop * 2.5).toFixed(2) + ' m/s';
        document.getElementById('req-fuel').textContent = (sat.dragDrop * 0.08).toFixed(2) + ' kg';
        if (rippleBtn) rippleBtn.style.display = 'block'; // show ripple button when burn needed
    } else {
        btn.disabled = true;
        btn.textContent = 'MANEUVER NOT REQUIRED';
        document.getElementById('req-dv').textContent = '0.0 m/s';
        document.getElementById('req-fuel').textContent = '0.0 kg';
        if (rippleBtn) rippleBtn.style.display = 'none';
    }
}

// ═══════════════════════════════════════════════════════════════════
// MANEUVER RIPPLE SIMULATOR
// Computes how a single satellite's burn affects the entire constellation
// ═══════════════════════════════════════════════════════════════════

function _rippleCalcOrbitalPeriod(altKm) {
    const GM = 3.986e14, Re = 6371000;
    const r = (Re + altKm * 1000);
    return 2 * Math.PI * Math.sqrt(Math.pow(r, 3) / GM) / 60; // minutes
}

function _rippleCalcGroundSpeed(altKm) {
    const GM = 3.986e14, Re = 6371000;
    return Math.sqrt(GM / (Re + altKm * 1000)) / 1000; // km/s
}

function _rippleSimulate(targetSat, dvMs) {
    const mu = 3.986e14;
    const Re = 6371;

    // ── New orbit after burn (vis-viva equation) ──────────────────────────────
    const r1 = (Re + targetSat.alt) * 1000;      // m
    const v1 = Math.sqrt(mu / r1);               // m/s  current velocity
    const v2 = v1 + dvMs;                         // m/s  after burn
    const a2 = mu / (2 * mu / r1 - v2 * v2);     // m    new semi-major axis
    const newAlt = (a2 / 1000) - Re;              // km

    const oldPeriod = _rippleCalcOrbitalPeriod(targetSat.alt);  // minutes
    const newPeriod = _rippleCalcOrbitalPeriod(newAlt);
    const periodDeltaMin = newPeriod - oldPeriod; // minutes — longer period = larger orbit

    // How many sats are also drifting/critical (fleet-wide context)
    const pendingBurns = istarFleet.filter(s => s.id !== targetSat.id &&
        (s.status === 'critical' || s.status === 'drifting')).length;

    const results = [];
    let totalRevisitDeltaSec = 0;
    let affectedCount = 0;

    istarFleet.forEach(other => {
        if (other.id === targetSat.id) return;

        const samePlane = other.plane === targetSat.plane;
        const satsInPlane = istarFleet.filter(s => s.plane === targetSat.plane).length;

        // ── Baseline inter-satellite angular spacing (degrees) ────────────────
        // In a Walker constellation, sats in same plane are equally spaced
        const angularSpacingDeg = 360 / satsInPlane;

        // At orbital velocity, this angular spacing translates to a time gap:
        // time_gap = period * (angular_spacing / 360)
        const revisitGapMinBefore = oldPeriod * (angularSpacingDeg / 360);
        const revisitGapMinAfter = newPeriod * (angularSpacingDeg / 360);

        let revisitDeltaSec, spacingDeltaKm, spacingPct;

        if (samePlane) {
            // After burn: target satellite drifts to higher orbit with longer period.
            // Its slot in the constellation pattern opens a gap.
            // The adjacent satellites' effective revisit time changes.
            revisitDeltaSec = (revisitGapMinAfter - revisitGapMinBefore) * 60;
            const orbCirc = 2 * Math.PI * (Re + targetSat.alt);
            const newCirc = 2 * Math.PI * (Re + newAlt);
            spacingDeltaKm = (newCirc - orbCirc) / (satsInPlane - 1);
            spacingPct = (spacingDeltaKm / (orbCirc / satsInPlane)) * 100;
        } else {
            // Cross-plane: the target satellite's period change shifts when it
            // crosses the equatorial node relative to other planes.
            // This creates a timing drift that degrades multi-plane coordination.
            revisitDeltaSec = Math.abs(periodDeltaMin) * 60 * 0.15; // 15% coupling factor
            const orbCirc = 2 * Math.PI * (Re + targetSat.alt);
            spacingDeltaKm = revisitDeltaSec * _rippleCalcGroundSpeed(other.alt);
            spacingPct = (spacingDeltaKm / (orbCirc / 3)) * 100;
        }

        totalRevisitDeltaSec += Math.abs(revisitDeltaSec);

        // ── Risk based on RELATIVE spacing change (%) ─────────────────────────
        // thresholds: >2% HIGH, >0.5% MEDIUM, >0.15% LOW, else NONE
        const absPct = Math.abs(spacingPct);
        const risk = absPct > 2.0 ? 'high' :
            absPct > 0.5 ? 'medium' :
                absPct > 0.15 ? 'low' : 'none';
        if (risk !== 'none') affectedCount++;

        results.push({
            id: other.id, plane: other.plane,
            revisitDeltaSec, spacingDeltaKm, spacingPct,
            risk, samePlane,
        });
    });

    // Sort: same-plane first, then by impact magnitude
    results.sort((a, b) => {
        if (a.samePlane !== b.samePlane) return b.samePlane - a.samePlane;
        return Math.abs(b.spacingPct) - Math.abs(a.spacingPct);
    });

    // ── Fleet-level scores ────────────────────────────────────────────────────
    const avgRevisitDeltaSec = totalRevisitDeltaSec / results.length;
    // Coverage score: penalise for revisit degradation, for pending burns from other sats
    const covBefore = 100;
    const singleBurnImpact = Math.min(15, (avgRevisitDeltaSec / oldPeriod / 60) * 100 * 25);
    const fleetBurnMultiplier = 1 + pendingBurns * 0.12; // each pending burn adds 12% more stress
    const covAfter = Math.max(55, covBefore - singleBurnImpact * fleetBurnMultiplier);

    // Old inter-sat spacing for display
    const satsInTargetPlane = istarFleet.filter(s => s.plane === targetSat.plane).length;
    const baseCirc = 2 * Math.PI * (Re + targetSat.alt);
    const avgSpacingBefore = baseCirc / satsInTargetPlane;
    const avgSpacingAfter = avgSpacingBefore + results
        .filter(r => r.samePlane)
        .reduce((s, r) => s + r.spacingDeltaKm, 0) / Math.max(1, satsInTargetPlane - 1);

    return {
        results, avgSpacingBefore, avgSpacingAfter,
        covBefore, covAfter,
        affectedCount, newAlt, dvMs,
        avgRevisitDeltaSec, pendingBurns,
        oldPeriod, newPeriod, periodDeltaMin,
    };
}

window.openRippleSimulator = function () {
    const sat = istarFleet.find(s => s.id === selectedSat);
    if (!sat || (sat.status !== 'critical' && sat.status !== 'drifting')) return;

    const dvMs = sat.dragDrop * 2.5;
    const sim = _rippleSimulate(sat, dvMs);

    // Subtitle
    const sub = document.getElementById('ripple-subtitle');
    if (sub) sub.textContent =
        `${sat.id}  ·  ${dvMs.toFixed(2)} m/s Prograde  ·  ${sat.alt.toFixed(1)} km → ${sim.newAlt.toFixed(1)} km  ·  ΔT = ${(sim.periodDeltaMin * 60).toFixed(1)}s longer orbit`;

    const el = id => document.getElementById(id);

    // ── Summary card 1: Inter-satellite gap ──────────────────────────────────
    if (el('rs-gap-before')) el('rs-gap-before').textContent = sim.avgSpacingBefore.toFixed(1) + ' km';
    if (el('rs-gap-after')) el('rs-gap-after').textContent = sim.avgSpacingAfter.toFixed(1) + ' km';
    const gapDeltaKm = sim.avgSpacingAfter - sim.avgSpacingBefore;
    const gapPct = (gapDeltaKm / sim.avgSpacingBefore) * 100;
    if (el('rs-gap-delta')) {
        el('rs-gap-delta').textContent = `${gapPct >= 0 ? '+' : ''}${gapPct.toFixed(3)}%`;
        el('rs-gap-delta').style.color = Math.abs(gapPct) > 2 ? 'var(--red)' :
            Math.abs(gapPct) > 0.5 ? 'var(--amber)' : 'var(--green)';
    }

    // ── Summary card 2: Fleet coverage score ─────────────────────────────────
    if (el('rs-cov-before')) el('rs-cov-before').textContent = sim.covBefore.toFixed(1) + '%';
    if (el('rs-cov-after')) el('rs-cov-after').textContent = sim.covAfter.toFixed(1) + '%';
    const covDelta = sim.covAfter - sim.covBefore;
    if (el('rs-cov-delta')) {
        el('rs-cov-delta').textContent = (covDelta >= 0 ? '+' : '') + covDelta.toFixed(1) + '%';
        el('rs-cov-delta').style.color = covDelta < -8 ? 'var(--red)' :
            covDelta < -3 ? 'var(--amber)' : 'var(--green)';
    }

    // ── Summary card 3: Satellites affected ──────────────────────────────────
    if (el('rs-impacted')) el('rs-impacted').textContent =
        `${sim.affectedCount} / ${sim.results.length}`;

    // ── Warning card: only if pending burns make it worse ────────────────────
    const warnCard = el('rs-gap-warn');
    if (sim.pendingBurns > 6 || sim.affectedCount > 8) {
        if (warnCard) warnCard.style.display = 'block';
        const gapMin = (sim.avgRevisitDeltaSec * sim.pendingBurns / 60).toFixed(1);
        if (el('rs-gap-window')) el('rs-gap-window').textContent =
            `${sim.pendingBurns} pending burns · cumulative gap ~${gapMin} min`;
    } else {
        if (warnCard) warnCard.style.display = 'none';
    }

    // ── Satellite rows ────────────────────────────────────────────────────────
    const list = el('ripple-list');
    if (list) {
        list.innerHTML = sim.results.map(r => {
            const absPct = Math.abs(r.spacingPct);
            const barPct = Math.min(100, absPct / 3 * 100);  // 3% = full bar
            const barColor = r.risk === 'high' ? 'var(--red)' :
                r.risk === 'medium' ? 'var(--amber)' :
                    r.risk === 'low' ? 'var(--cyan)' : 'var(--green)';
            const planeColor = { 1: '#00d2ff', 2: '#a855f7', 3: '#22c55e' }[r.plane] || '#fff';
            const revisitSign = r.revisitDeltaSec >= 0 ? '+' : '';
            const pctSign = r.spacingPct >= 0 ? '+' : '';
            // Show revisit time in the spacing column for clarity
            const revisitStr = Math.abs(r.revisitDeltaSec) < 60
                ? `${revisitSign}${r.revisitDeltaSec.toFixed(1)}s revisit`
                : `${revisitSign}${(r.revisitDeltaSec / 60).toFixed(2)}min revisit`;
            return `<div class="ripple-row">
                <span class="rr-id">${r.id}</span>
                <span class="rr-plane" style="color:${planeColor}">P${r.plane}</span>
                <span class="rr-spacing" style="color:${r.risk === 'none' ? 'var(--green)' : barColor}">
                    ${pctSign}${r.spacingPct.toFixed(3)}%
                </span>
                <div>
                  <div class="rr-bar-wrap"><div class="rr-bar" style="width:${barPct}%;background:${barColor}"></div></div>
                  <span style="font-size:9px;color:var(--text-muted);margin-top:2px;display:block">
                    ${revisitStr} · ${r.samePlane ? 'same plane — direct' : 'cross-plane — drift'}
                  </span>
                </div>
                <span class="rr-risk risk-${r.risk}">${r.risk.toUpperCase()}</span>
            </div>`;
        }).join('');
    }

    // Open modal
    el('ripple-backdrop').classList.add('open');
    el('ripple-modal').style.display = 'flex';
    requestAnimationFrame(() => el('ripple-modal').classList.add('open'));
};

window.closeRippleSimulator = function () {
    const modal = document.getElementById('ripple-modal');
    const backdrop = document.getElementById('ripple-backdrop');
    if (modal) { modal.classList.remove('open'); setTimeout(() => modal.style.display = 'none', 250); }
    if (backdrop) backdrop.classList.remove('open');
};

window.confirmRippleBurn = function () {
    closeRippleSimulator();
    authorizeUplink();
    logTerminal('[RIPPLE] Constellation impact reviewed. Executing burn...', 'warn');
};

// Logging Terminal
function logTerminal(msg, type = 'info') {
    const term = document.getElementById('action-log');
    if (!term) return;
    const ts = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const el = document.createElement('div');
    el.className = `log-line ${type}`;
    el.innerHTML = `[${ts}] ${msg}`;
    term.appendChild(el);
    if (term.children.length > 50) term.removeChild(term.firstChild);
    term.scrollTop = term.scrollHeight;
}

// Autonomy Toggle
window.toggleAutonomy = function (isChecked) {
    autoAuth = isChecked;
    const label = document.getElementById('auth-label');
    const authBtn = document.getElementById('auth-btn');
    if (isChecked) {
        if (label) { label.textContent = 'AUTONOMOUS (M2M)'; label.className = 'auth-label auto'; }
        if (authBtn) authBtn.style.display = 'none';
        logTerminal('[SYS] Full Autonomous M2M Mode Engaged. Human authorization bypassed.', 'warn');
        if (liveMode) processAutoManeuvers();
    } else {
        if (label) { label.textContent = 'HUMAN-ON-THE-LOOP'; label.className = 'auth-label'; }
        if (authBtn) authBtn.style.display = 'block';
        logTerminal('[SYS] Human-On-The-Loop Mode Engaged. Awaiting manual authorization.', 'info');
    }
}

function processAutoManeuvers() {
    if (!autoAuth) return;
    let actionTaken = false;
    istarFleet.forEach(sat => {
        if (sat.status === 'critical' || sat.status === 'drifting') {
            const dv = (sat.dragDrop * 2.5).toFixed(2);
            logTerminal(`[AUTO] Instructing ${sat.id} to execute ${dv}m/s burn.`, 'warn');
            sat.status = 'maneuvering';
            actionTaken = true;

            setTimeout(() => {
                sat.status = 'nominal';
                sat.fuel -= (sat.dragDrop * 0.08);
                sat.alt += sat.dragDrop;
                sat.dragDrop = 0;
                sat.colaRisk = false;
                logTerminal(`[M2M] ${sat.id} burn complete. Nominal orbit restored.`, 'succ');
                renderFleetGrid();
                updateFuelUI();
                updateCOLA();
            }, 3000 + Math.random() * 2000);
        }
    });
    if (actionTaken) {
        renderFleetGrid();
        updateFuelUI();
    }
}

window.authorizeUplink = function () {
    const sat = istarFleet.find(s => s.id === selectedSat);
    if (!sat || (sat.status !== 'critical' && sat.status !== 'drifting')) return;

    const btn = document.getElementById('auth-btn');
    if (btn) { btn.disabled = true; btn.textContent = 'TRANSMITTING OP-CODES...'; }
    const dv = (sat.dragDrop * 2.5).toFixed(2);
    logTerminal(`[UPLINK] Human Authorized: Transmitting ${dv}m/s burn op-codes to ${sat.id}...`, 'info');

    sat.status = 'maneuvering';
    renderFleetGrid();

    setTimeout(() => {
        sat.status = 'nominal';
        sat.fuel -= (sat.dragDrop * 0.08);
        sat.alt += sat.dragDrop;
        sat.dragDrop = 0;
        sat.colaRisk = false;
        logTerminal(`[TELEMETRY] ${sat.id} confirmed burn. Altitude corrected.`, 'succ');
        renderFleetGrid();
        updateFuelUI();
        updateCOLA();
        updateChart();
    }, 4000);
}

// Live Data Polling
window.toggleLive = async function () {
    liveMode = !liveMode;
    const btn = document.getElementById('live-mode-btn');
    const dot = document.getElementById('live-dot');

    if (liveMode) {
        if (btn) { btn.style.borderColor = 'rgba(239, 68, 68, 0.4)'; btn.style.color = 'var(--red)'; btn.style.background = 'rgba(239, 68, 68, 0.1)'; }
        if (dot) { dot.className = 'live-btn-dot pulse'; dot.style.background = 'var(--red)'; }
        const ll = document.getElementById('live-btn-label'); if (ll) ll.textContent = 'SECURE FEED LIVE';
        logTerminal('[NET] Connecting to NOAA SWPC / ISTRAC Backend...', 'info');
        pollData();
    } else {
        if (btn) btn.style = '';
        if (dot) { dot.className = 'live-btn-dot'; dot.style = ''; }
        const ll = document.getElementById('live-btn-label'); if (ll) ll.textContent = 'CONNECT LIVE';
        const lu = document.getElementById('live-updated'); if (lu) lu.textContent = 'OFFLINE';
        clearTimeout(pollTimeout);
        logTerminal('[NET] Secure live connection severed.', 'warn');
        resetDashboard();
    }
}

async function pollData() {
    if (!liveMode) return;
    try {
        const res = await fetch(`${BACKEND_URL}/api/live`);
        if (!res.ok) throw new Error('Backend offline');
        const data = await res.json();

        const ago = new Date(data.last_updated);
        const lu = document.getElementById('live-updated');
        if (lu) lu.textContent = `Sync: ${ago.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}`;

        processLiveData(data);
    } catch (e) {
        // Fallback to simulation if backend unconnected
        const lu = document.getElementById('live-updated');
        if (lu) lu.textContent = `Offline Matrix (Sim)`;
        simulateStorm();
    }

    pollTimeout = setTimeout(pollData, 10000); // Poll faster for demo
}

function processLiveData(data) {
    const s = data.storm || {};
    const gkp = document.getElementById('god-kp'); if (gkp) gkp.textContent = (s.kp || 0).toFixed(1);
    const gv = document.getElementById('god-vsw'); if (gv) gv.textContent = Math.round(s.solar_wind_speed || 0) + ' km/s';
    if (gkp) gkp.style.color = s.kp >= 7 ? 'var(--red)' : s.kp >= 5 ? 'var(--amber)' : 'var(--text-primary)';

    let phase = 'QUIET';
    let baseRho = 1.8e-13;
    let mlRho = 1.8e-13;
    let pct = 0;

    if (s.kp >= 7) { phase = 'MAIN PHASE'; baseRho = 3.2e-13; mlRho = 5.8e-13; pct = 80; }
    else if (s.kp >= 5) { phase = 'COMMENCEMENT'; baseRho = 2.4e-13; mlRho = 3.1e-13; pct = 30; }

    updateAtmos(phase, s.kp, baseRho, mlRho, pct);
    applyStormImpact(pct);
}

// ── Natural Storm Lifecycle Engine ───────────────────────────────────────────
// Storm builds and recedes naturally over ~5-7 minutes per full cycle.
// Each poll tick (10s) nudges ONE satellite at most, based on current intensity.
// Phases: quiet → pre-storm → commencement → main phase → recovery → quiet
// Satellites stay drifting/critical until operator acts — no auto-recovery.

const STORM_LIFECYCLE = [
    { name: 'QUIET', kpRange: [2.5, 3.8], pct: 0, duration: 6 },
    { name: 'PRE-STORM', kpRange: [3.8, 5.2], pct: 8, duration: 5 },
    { name: 'SUDDEN COMMENCEMENT', kpRange: [5.2, 6.5], pct: 28, duration: 5 },
    { name: 'MAIN PHASE', kpRange: [6.5, 9.0], pct: 75, duration: 10 },
    { name: 'RECOVERY PHASE', kpRange: [4.0, 5.5], pct: 15, duration: 6 },
];

let _stormPhaseIdx = 0;   // start quiet
let _stormPhaseTicks = 0;  // ticks spent in current phase
let _stormKp = 3.2; // smoothed Kp

function simulateStorm() {
    const phase = STORM_LIFECYCLE[_stormPhaseIdx];
    _stormPhaseTicks++;

    // Smooth Kp toward target range centre with jitter
    const kpTarget = (phase.kpRange[0] + phase.kpRange[1]) / 2 + (Math.random() - 0.5) * 0.6;
    _stormKp = _stormKp * 0.75 + kpTarget * 0.25;
    const kp = parseFloat(_stormKp.toFixed(1));

    // Update god bar Kp + solar wind
    const gkp = document.getElementById('god-kp');
    if (gkp) { gkp.textContent = kp.toFixed(1); gkp.style.color = kp >= 7 ? 'var(--red)' : kp >= 5 ? 'var(--amber)' : 'var(--text-primary)'; }
    const gv = document.getElementById('god-vsw');
    if (gv) gv.textContent = Math.round(350 + kp * 55 + Math.random() * 60) + ' km/s';

    // Atmospheric density
    let baseRho = 1.8e-13, mlRho = 1.8e-13, pct = phase.pct;
    if (kp >= 7) { baseRho = 3.2e-13 + Math.random() * 8e-14; mlRho = baseRho * 1.8; }
    else if (kp >= 5) { baseRho = 2.4e-13 + Math.random() * 3e-14; mlRho = baseRho * 1.3; }

    updateAtmos(phase.name, kp, baseRho, mlRho, pct);

    // Gradually impact satellites — 1-2 per tick based on storm intensity
    _gradualImpact(pct, kp);

    // Advance phase when duration elapsed
    if (_stormPhaseTicks >= phase.duration) {
        _stormPhaseTicks = 0;
        _stormPhaseIdx = (_stormPhaseIdx + 1) % STORM_LIFECYCLE.length;
        const nextPhase = STORM_LIFECYCLE[_stormPhaseIdx];
        logTerminal('[STORM] Phase transition -> ' + nextPhase.name, _stormPhaseIdx >= 3 ? 'warn' : 'info');
        // Satellites do NOT auto-recover — they stay drifting/critical until operator acts
    }
}

function _gradualImpact(pct, kp) {
    if (pct === 0) return; // QUIET phase — never touch satellites

    // Probability per 10s tick scales with storm intensity
    const chance = pct < 15 ? 0.35 : pct < 40 ? 0.70 : 0.95;
    if (Math.random() > chance) return;

    // Hit 1-2 satellites per tick during main phase
    const maxHits = (kp >= 6.5 && Math.random() < 0.5) ? 2 : 1;

    for (let h = 0; h < maxHits; h++) {
        const nominals = istarFleet.filter(s => s.status === 'nominal');
        const driftings = kp >= 7.0 ? istarFleet.filter(s => s.status === 'drifting') : [];

        let target = null, escalating = false;
        if (driftings.length > 0 && Math.random() < 0.4) {
            target = driftings[Math.floor(Math.random() * driftings.length)];
            escalating = true;
        } else if (nominals.length > 0) {
            target = nominals[Math.floor(Math.random() * nominals.length)];
        }
        if (!target) break;

        // Main phase (kp >= 6.5): 50% chance of critical
        const toStatus = ((kp >= 6.5 && Math.random() < 0.50) || escalating) ? 'critical' : 'drifting';
        const drop = (pct / 100) * (Math.random() * 1.5 + 0.4);
        target.dragDrop = (target.dragDrop || 0) + drop;
        target.alt -= drop;
        target.status = toStatus;

        // ── Globe bridge: mirror status change into 3D visualization ──────────
        window.dispatchEvent(new CustomEvent('celestis:satupdate', {
            detail: { id: target.id, status: toStatus, driftRate: drop * 24, alt_km: target.alt }
        }));

        // All drifting/critical satellites get COLA risk
        target.colaRisk = true;

        logTerminal(
            '[ATMOS] ' + target.id + ' drag spike +' + drop.toFixed(2) + ' km/day. Status -> ' + toStatus.toUpperCase(),
            toStatus === 'critical' ? 'crit' : 'warn'
        );
    }

    renderFleetGrid();
    updateFuelUI();
    updateChart();
    updateCOLA();

    if (autoAuth) processAutoManeuvers();
}

function _beginRecovery() {
    // No-op: satellites stay drifting/critical until operator authorises a maneuver.
    logTerminal('[STORM] Storm cycle complete. Satellites remain in current states - awaiting operator action.', 'warn');
}


function updateAtmos(phase, kp, baseRho, mlRho, pct) {
    const atp = document.getElementById('at-phase');
    if (atp) { atp.textContent = phase; atp.className = kp >= 7 ? 'acard-val text-red' : kp >= 5 ? 'acard-val text-amber' : 'acard-val text-green'; }
    const ab = document.getElementById('at-rho-base'); if (ab) ab.textContent = baseRho.toExponential(2);
    const am = document.getElementById('at-rho-ml'); if (am) am.textContent = mlRho.toExponential(2);

    // Row 1 (Physics): shrinks to show what NRLMSISE-00 predicts
    const physBar = document.getElementById('dd-bar-phys');
    const physPct = document.getElementById('dd-phys-pct');
    if (physBar) physBar.style.width = (100 - pct) + '%';
    if (physPct) physPct.textContent = (100 - pct).toFixed(0) + '%';

    // Row 2 (AI): base same width + glowing extension for the correction
    const aiBase = document.getElementById('dd-bar-ai-base');
    const aiGlow = document.getElementById('dd-bar-ml');
    const aiPct = document.getElementById('dd-ai-pct');
    if (aiBase) aiBase.style.width = (100 - pct) + '%';
    if (aiGlow) aiGlow.style.width = pct + '%';
    if (aiPct) aiPct.textContent = pct > 0 ? '+' + pct.toFixed(0) + '%' : '+0%';

    // Badge colour changes with severity
    const badge = document.getElementById('dd-pct');
    if (badge) {
        if (pct >= 60) {
            badge.textContent = '+' + pct.toFixed(0) + '% STORM';
            badge.style.cssText = 'color:var(--red);background:rgba(255,51,102,0.12);border:1px solid rgba(255,51,102,0.4)';
        } else if (pct >= 20) {
            badge.textContent = '+' + pct.toFixed(0) + '% ACTIVE';
            badge.style.cssText = 'color:var(--amber);background:rgba(255,184,0,0.10);border:1px solid rgba(255,184,0,0.3)';
        } else if (pct > 0) {
            badge.textContent = '+' + pct.toFixed(0) + '% ELEVATED';
            badge.style.cssText = 'color:var(--cyan);background:rgba(0,210,255,0.08);border:1px solid rgba(0,210,255,0.25)';
        } else {
            badge.textContent = 'QUIET';
            badge.style.cssText = 'color:var(--text-muted);background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08)';
        }
    }
}



function updateCOLA() {
    const list = document.getElementById('cola-list');
    if (!list) return;
    let hasRisk = false;
    list.innerHTML = '';

    istarFleet.forEach(sat => {
        if (sat.colaRisk) {
            hasRisk = true;
            list.innerHTML += `<div class="cola-item">
                <strong class="text-red">${sat.id}</strong> trajectory intersects catalog debris (CZ-5B Deriv) due to unmodeled dragging.<br>
                <span class="text-amber">TCA: 04h 12m</span> | <span class="text-red">PoC: ${(Math.random() * 0.005).toExponential(2)}</span>
            </div>`;
        }
    });

    const cs = document.getElementById('cola-status');
    if (!hasRisk) {
        list.innerHTML = `<div class="cola-empty">No collision threats detected in decay path.</div>`;
        if (cs) { cs.textContent = 'CLEAR'; cs.className = 'cola-badge'; }
    } else {
        if (cs) { cs.textContent = 'CRITICAL CONJUNCTION'; cs.className = 'cola-badge danger'; }
        logTerminal('[ALERT] Collision Avoidance (COLA) warning triggered for decaying nodes.', 'crit');
    }
}

function resetDashboard() {
    const gk = document.getElementById('god-kp'); if (gk) { gk.textContent = '—'; gk.style.color = ''; }
    const gv = document.getElementById('god-vsw'); if (gv) gv.textContent = '— km/s';
    const ap = document.getElementById('at-phase'); if (ap) { ap.textContent = 'OFFLINE'; ap.className = 'acard-val text-muted'; }
    const arb = document.getElementById('at-rho-base'); if (arb) arb.textContent = '1.8e-13';
    const arm = document.getElementById('at-rho-ml'); if (arm) arm.textContent = '1.8e-13';
    const dbp = document.getElementById('dd-bar-phys'); if (dbp) dbp.style.width = '100%';
    const dbm = document.getElementById('dd-bar-ml'); if (dbm) dbm.style.width = '0%';
    const ddp = document.getElementById('dd-pct'); if (ddp) ddp.textContent = '0% Difference';

    // reset fleet
    istarFleet.forEach(s => { s.status = 'nominal'; s.dragDrop = 0; s.colaRisk = false; });
    renderFleetGrid();
    updateFuelUI();
    updateChart();
    updateCOLA();
}

function initChart() {
    const ctx = document.getElementById('altitudeChart');
    if (!ctx) return;
    const labels = Array.from({ length: 24 }, (_, i) => `T+${i}h`);

    const dNominal = new Array(24).fill(500);
    const dPhys = labels.map((_, i) => 500 - (i * 0.05));
    const dML = labels.map((_, i) => 500 - (i * 0.15) - (Math.random() * 0.05));

    altChart = new Chart(ctx.getContext('2d'), {
        type: 'line',
        data: {
            labels,
            datasets: [
                { label: 'Nominal 500km', data: dNominal, borderColor: 'rgba(255,255,255,0.2)', borderDash: [5, 5], borderWidth: 1, pointRadius: 0 },
                { label: 'Physics (NRLMSISE)', data: dPhys, borderColor: '#a3b8cc', borderWidth: 2, pointRadius: 0, tension: 0.1 },
                { label: 'AI Risk Prediction', data: dML, borderColor: '#b057ff', borderWidth: 2, pointRadius: 0, tension: 0.1 }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#a3b8cc', font: { family: 'JetBrains Mono', size: 10 } } }
            },
            scales: {
                y: { min: 495, max: 501, ticks: { color: '#5c7999', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } },
                x: { ticks: { color: '#5c7999', font: { family: 'JetBrains Mono', size: 10 } }, grid: { color: 'rgba(255,255,255,0.05)' } }
            }
        }
    });
}

function updateChart() {
    if (!altChart) return;
    const drop = istarFleet.filter(s => s.status !== 'nominal').reduce((sum, s) => sum + s.dragDrop, 0);
    if (drop > 0) {
        const avgDrop = drop / istarFleet.filter(s => s.status !== 'nominal').length;
        const slopeP = avgDrop / 24;
        const slopeM = (avgDrop * 1.5) / 24;
        altChart.data.datasets[1].data = altChart.data.labels.map((_, i) => 500 - (slopeP * i));
        altChart.data.datasets[2].data = altChart.data.labels.map((_, i) => 500 - (slopeM * i) - (Math.random() * 0.1));
    } else {
        altChart.data.datasets[1].data = altChart.data.labels.map((_, i) => 500 - (i * 0.05));
        altChart.data.datasets[2].data = altChart.data.labels.map((_, i) => 500 - (i * 0.15) - (Math.random() * 0.05));
    }
    altChart.update();
}

window.addEventListener('DOMContentLoaded', () => {
    loadFleetConfig();
    initChart();
    startClock();

    // Reveal window controls if in Electron
    if (window.celestisApp) {
        const wc = document.getElementById('win-controls');
        if (wc) wc.style.display = 'flex';
    }
});

// IST Clock — updates every second
function startClock() {
    function tick() {
        const now = new Date();
        // IST = UTC + 5:30
        const istOffset = 5 * 60 + 30;
        const utcMin = now.getUTCHours() * 60 + now.getUTCMinutes() + istOffset;
        const hh = Math.floor(utcMin / 60) % 24;
        const mm = utcMin % 60;
        const ss = now.getUTCSeconds();
        const pad = n => String(n).padStart(2, '0');
        const el = document.getElementById('ist-time');
        if (el) el.textContent = pad(hh) + ':' + pad(mm) + ':' + pad(ss);
    }
    tick();
    setInterval(tick, 1000);
}

// Load fleet from JSON config, fall back to initFleet() if unavailable
async function loadFleetConfig() {
    try {
        const res = await fetch('./istar_fleet_config.json');
        if (!res.ok) throw new Error('Config not found');
        const cfg = await res.json();
        const sats = cfg.satellites || [];
        if (sats.length === 0) throw new Error('Empty config');

        istarFleet = sats.map(s => ({
            id: s.id,
            plane: s.plane || 1,
            alt: s.alt_km || 500,
            fuel: s.fuel_pct || 98,
            status: 'nominal',
            dragDrop: 0,
            colaRisk: false,
            role: s.role || 'Unknown',
            mass_kg: s.mass_kg || null,
            fuel_kg: s.fuel_kg || null,
            fuel_type: s.fuel_type || 'Argon',
            isp_s: s.isp_s || null,
            thrust_mN: s.thrust_mN || null,
            area_m2: s.area_m2 || null,
            Cd: s.Cd || 2.2,
        }));

        renderFleetGrid();
        logTerminal('[SYS] I-STAR Autonomous Maneuver Engine Initialized.', 'succ');
        logTerminal('[SYS] Awaiting live telemetry connection...', 'info');
    } catch (e) {
        // Fall back to generated fleet
        initFleet();
        logTerminal('[SYS] Fleet config unavailable - using default fleet.', 'warn');
    }
}
