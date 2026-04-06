/**
 * CELESTIS — 3D Orbital Globe
 * Three.js + satellite.js — completely self-contained, zero coupling to app.js
 * ============================================================================
 * KEY DESIGN RULE: This file NEVER calls any function from app.js.
 *   - No window.selectSat()  — renamed to _globeSelectSat() (private)
 *   - No openDrawer()        — globe has its own info panel
 *   - No istarFleet[]        — globe fetches from its own backend call
 *
 * Public API (on window):
 *   openGlobe()
 *   closeGlobe()
 *   setGlobeSpeed(mult)
 *   globeToggleLayer(name, bool)
 *   globeFollowSat(id)
 */

'use strict';

// ─── Wrap everything in an IIFE to prevent ANY globals leaking ─────────────
(function () {

    // ── Config ────────────────────────────────────────────────────────────────
    const BACKEND = 'http://127.0.0.1:4000';
    let CLOCK_MULT = 60;  // locked — no UI control
    const EARTH_R = 1.0;
    const TRACK_STEP = 3;
    const TRACK_PAST = 45;
    const TRACK_FUTURE = 90;

    const PLANE_COL = { 1: 0x00d2ff, 2: 0xa855f7, 3: 0x22c55e };
    const STATUS_COL = {
        nominal: 0x00d2ff,
        drifting: 0xffb800,
        critical: 0xff3366,
        maneuvering: 0x7c3aed,
    };

    // ── State (all private) ───────────────────────────────────────────────────
    let renderer = null, scene = null, camera = null, controls = null;
    let earthMesh = null, satGroup = null, orbitGroup = null;
    let raycaster = null, mouse = null;
    let satMeshMap = {};
    let selectedId = null;
    let followId = null;
    let animFrame = null;
    let simTime = new Date();
    let lastRealMs = Date.now();
    let globeInited = false;
    let _tick = 0;

    const LAYERS = { satellites: true, orbitLines: true, groundTracks: true, debris: true };

    // ── Debris system (34,000 objects via InstancedMesh) ──────────────────────
    const DEBRIS_COUNT = 34000;
    const DEBRIS_SHELLS = [
        { altMin: 780, altMax: 850, count: 12000, col: 0xffaa00 }, // Chinese ASAT test band
        { altMin: 600, altMax: 650, count: 8000, col: 0xff4411 }, // Cosmos-2251 / Iridium-33
        { altMin: 1200, altMax: 1400, count: 6000, col: 0xff6622 }, // Soviet legacy band
        { altMin: 300, altMax: 500, count: 8000, col: 0x5577ff }, // Low LEO miscellaneous
    ];
    let debrisMesh = null;
    const debrisTheta = new Float32Array(DEBRIS_COUNT);
    const debrisAlt = new Float32Array(DEBRIS_COUNT);
    const debrisOmega = new Float32Array(DEBRIS_COUNT);
    const debrisPx = new Float32Array(DEBRIS_COUNT), debrisPy = new Float32Array(DEBRIS_COUNT), debrisPz = new Float32Array(DEBRIS_COUNT);
    const debrisQx = new Float32Array(DEBRIS_COUNT), debrisQy = new Float32Array(DEBRIS_COUNT), debrisQz = new Float32Array(DEBRIS_COUNT);
    const debrisDecay = new Float32Array(DEBRIS_COUNT);
    const debrisAlive = new Uint8Array(DEBRIS_COUNT).fill(1);
    const _dbDummy = new THREE.Object3D();
    const _collFlashes = [];
    let _debrisInited = false;
    let _simPaused = false;
    let _collZoneKm = 250;
    const _shellVisible = [true, true, true, true];
    const debrisShell = new Uint8Array(DEBRIS_COUNT); // which shell (0-3) each piece belongs to
    const satDriftState = {};
    const satAltOffset = {};
    const _statusLists = { drifting: new Set(), critical: new Set() };
    window._liveKp = 3;

    // ── Helpers ───────────────────────────────────────────────────────────────
    function _satColor(plane, status) {
        if (status === 'critical') return STATUS_COL.critical;
        if (status === 'drifting') return STATUS_COL.drifting;
        if (status === 'maneuvering') return STATUS_COL.maneuvering;
        return PLANE_COL[plane] || 0x00d2ff;
    }

    function _latLonToVec3(lat, lon, altKm) {
        const R = EARTH_R + altKm / 6371;
        const phi = (90 - lat) * Math.PI / 180;
        const tht = (lon + 180) * Math.PI / 180;
        return new THREE.Vector3(
            -R * Math.sin(phi) * Math.cos(tht),
            R * Math.cos(phi),
            R * Math.sin(phi) * Math.sin(tht)
        );
    }

    function _synthTLE(satId, plane, altKm) {
        const GM = 3.986004418e5, Re = 6371;
        const a = Re + altKm;
        const n = Math.sqrt(GM / Math.pow(a, 3)) * 86400 / (2 * Math.PI);
        const incl = [97.4, 97.8, 98.1][Math.min(plane - 1, 2)];
        const raan = (plane - 1) * 120;
        const num = parseInt(satId.split('-').pop()) || 1;
        const ma = ((num - 1) % 12) * 30;
        const norad = 90000 + num;
        const now = new Date();
        const yday = Math.floor((now - new Date(now.getFullYear(), 0, 0)) / 86400000);
        const frac = (now.getUTCHours() * 3600 + now.getUTCMinutes() * 60 + now.getUTCSeconds()) / 86400;
        const epoch = `${String(now.getFullYear()).slice(2)}${String(yday).padStart(3, '0')}.${frac.toFixed(8).slice(2)}`;
        const t1 = `1 ${String(norad).padStart(5)}U 26001A   ${epoch}  .00000000  00000-0  00000-0 0  9990`;
        const t2 = `2 ${String(norad).padStart(5)} ${incl.toFixed(4).padStart(8)} ${raan.toFixed(4).padStart(8)} 0001000 000.0000 ${ma.toFixed(4).padStart(8)} ${n.toFixed(8).padStart(11)}    10`;
        return [t1, t2];
    }

    function _getSatPos(satrec, date) {
        try {
            const pv = satellite.propagate(satrec, date);
            if (!pv || !pv.position) return null;
            const gmst = satellite.gstime(date);
            const geo = satellite.eciToGeodetic(pv.position, gmst);
            return {
                lat: satellite.degreesLat(geo.latitude),
                lon: satellite.degreesLong(geo.longitude),
                alt: geo.height,
            };
        } catch { return null; }
    }

    function _buildArcVecs(satrec, refDate, startMin, endMin, step) {
        const vecs = [];
        for (let m = startMin; m <= endMin; m += step) {
            const t = new Date(refDate.getTime() + m * 60000);
            const pos = _getSatPos(satrec, t);
            if (pos) vecs.push(_latLonToVec3(pos.lat, pos.lon, pos.alt));
        }
        return vecs;
    }

    // ── Debris: initialise InstancedMesh ──────────────────────────────────────
    function _initDebris() {
        if (_debrisInited || !scene) return;
        _debrisInited = true;
        const GM = 3.986004418e5;
        const geo = new THREE.SphereGeometry(0.0018, 3, 2);
        const mat = new THREE.MeshBasicMaterial();
        debrisMesh = new THREE.InstancedMesh(geo, mat, DEBRIS_COUNT);
        debrisMesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);

        let idx = 0;
        DEBRIS_SHELLS.forEach((shell, sIdx) => {
            const shellColor = new THREE.Color(shell.col);
            for (let i = 0; i < shell.count && idx < DEBRIS_COUNT; i++, idx++) {
                debrisShell[idx] = sIdx;
                const alt = shell.altMin + Math.random() * (shell.altMax - shell.altMin);
                // Realistic inclinations per shell source event:
                // Shell 0 (Chinese ASAT): ~98° ± 3°  | Shell 1 (Cosmos/Iridium): ~74° ± 5°
                // Shell 2 (Soviet legacy): ~83° ± 8°  | Shell 3 (Low LEO misc): 28-98° spread
                const inclBase = [98, 74, 83, 51][sIdx] * Math.PI / 180;
                const inclSpread = [3, 5, 8, 35][sIdx] * Math.PI / 180;
                const incl = inclBase + (Math.random() * 2 - 1) * inclSpread;
                const raan = Math.random() * Math.PI * 2;
                const theta = Math.random() * Math.PI * 2;
                const a = 6371 + alt;
                const ci = Math.cos(incl), si = Math.sin(incl);
                const cr = Math.cos(raan), sr = Math.sin(raan);
                debrisPx[idx] = cr; debrisPy[idx] = sr; debrisPz[idx] = 0;
                debrisQx[idx] = -sr * ci; debrisQy[idx] = cr * ci; debrisQz[idx] = si;
                debrisTheta[idx] = theta;
                debrisAlt[idx] = alt;
                debrisOmega[idx] = Math.sqrt(GM / (a * a * a));
                debrisDecay[idx] = 0.000015 * Math.pow(6371 / a, 4);
                const r = EARTH_R + alt / 6371;
                const ct = Math.cos(theta), st = Math.sin(theta);
                _dbDummy.position.set(
                    r * (ct * debrisPx[idx] + st * debrisQx[idx]),
                    r * (ct * debrisPy[idx] + st * debrisQy[idx]),
                    r * (ct * debrisPz[idx] + st * debrisQz[idx])
                );
                _dbDummy.updateMatrix();
                debrisMesh.setMatrixAt(idx, _dbDummy.matrix);
                debrisMesh.setColorAt(idx, shellColor);
            }
        });
        debrisMesh.instanceMatrix.needsUpdate = true;
        if (debrisMesh.instanceColor) debrisMesh.instanceColor.needsUpdate = true;
        scene.add(debrisMesh);

        // Inject HUD elements
        const panel = document.getElementById('globe-panel');
        if (panel && !document.getElementById('globe-debris-hud')) {
            const hud = document.createElement('div');
            hud.id = 'globe-debris-hud';
            hud.style.cssText = 'position:absolute;bottom:12px;left:12px;font:bold 10px monospace;color:#ffaa00;text-shadow:0 0 8px #ff8800;pointer-events:none;';
            hud.textContent = DEBRIS_COUNT.toLocaleString() + ' TRACKED DEBRIS OBJECTS';
            panel.appendChild(hud);
        }
        if (panel && !document.getElementById('globe-coll-alert')) {
            const al = document.createElement('div');
            al.id = 'globe-coll-alert';
            al.style.cssText = 'position:absolute;top:60px;left:50%;transform:translateX(-50%);font:bold 13px monospace;color:#ff3366;background:rgba(255,30,80,.18);border:1px solid #ff3366;padding:6px 18px;border-radius:6px;display:none;pointer-events:none;transition:opacity .4s;text-shadow:0 0 12px #ff3366;';
            panel.appendChild(al);
        }
    }

    // ── Debris: update positions each tick ────────────────────────────────────
    function _updateDebris(dtSec) {
        if (!debrisMesh || !LAYERS.debris) return;
        const kpBoost = 1.0 + (window._liveKp || 3) * 0.18;
        for (let i = 0; i < DEBRIS_COUNT; i++) {
            if (!debrisAlive[i]) continue;
            // Shell visibility: scale to zero if hidden
            const visible = _shellVisible[debrisShell[i]];
            if (!visible) { _dbDummy.scale.setScalar(0.00001); _dbDummy.position.set(0, 999, 0); _dbDummy.updateMatrix(); debrisMesh.setMatrixAt(i, _dbDummy.matrix); continue; }
            _dbDummy.scale.setScalar(1);
            debrisTheta[i] += debrisOmega[i] * dtSec;
            if (debrisTheta[i] > 6.2832) debrisTheta[i] -= 6.2832;
            debrisAlt[i] -= debrisDecay[i] * kpBoost * dtSec;
            if (debrisAlt[i] < 150) debrisAlt[i] = 600 + Math.random() * 800;
            const r = EARTH_R + debrisAlt[i] / 6371;
            const ct = Math.cos(debrisTheta[i]), st = Math.sin(debrisTheta[i]);
            _dbDummy.position.set(
                r * (ct * debrisPx[i] + st * debrisQx[i]),
                r * (ct * debrisPy[i] + st * debrisQy[i]),
                r * (ct * debrisPz[i] + st * debrisQz[i])
            );
            _dbDummy.updateMatrix();
            debrisMesh.setMatrixAt(i, _dbDummy.matrix);
        }
        debrisMesh.instanceMatrix.needsUpdate = true;
    }

    // ── Debris: probabilistic collision check ─────────────────────────────────
    function _checkCollisions() {
        if (!debrisMesh) return;
        const kp = window._liveKp || 3;
        Object.entries(satMeshMap).forEach(([sid, rec]) => {
            if (!rec.mesh || !rec.mesh.visible || rec.data.status === 'destroyed') return;
            const satAlt = rec.data.alt_km || 500;
            let shellDensity = 0;
            for (const sh of DEBRIS_SHELLS) {
                if (satAlt >= sh.altMin - 200 && satAlt <= sh.altMax + 200)
                    shellDensity = Math.max(shellDensity, sh.count / DEBRIS_COUNT);
            }
            const prob = (0.006 + shellDensity * 0.025) * (1 + kp / 9);
            if (Math.random() > prob) return;
            // Find a sampled nearby debris for visual position
            const sp = rec.mesh.position;
            let collPos = sp.clone().add(new THREE.Vector3(
                (Math.random() - 0.5) * 0.12,
                (Math.random() - 0.5) * 0.12,
                (Math.random() - 0.5) * 0.12
            ));
            // Sample 300 random debris for closest
            let best = Infinity, bestIdx = 0;
            for (let n = 0; n < 300; n++) {
                const i = Math.floor(Math.random() * DEBRIS_COUNT);
                if (!debrisAlive[i]) continue;
                const r = EARTH_R + debrisAlt[i] / 6371;
                const ct = Math.cos(debrisTheta[i]), st = Math.sin(debrisTheta[i]);
                const dx = r * (ct * debrisPx[i] + st * debrisQx[i]) - sp.x;
                const dy = r * (ct * debrisPy[i] + st * debrisQy[i]) - sp.y;
                const dz = r * (ct * debrisPz[i] + st * debrisQz[i]) - sp.z;
                const d2 = dx * dx + dy * dy + dz * dz;
                if (d2 < best) { best = d2; bestIdx = i; }
            }
            _triggerCollision(sid, bestIdx, collPos);
        });
    }


    // ── Debris: collision effects + Kessler cascade ───────────────────────────
    function _triggerCollision(satId, debrisIdx, pos) {
        // 1. Satellite → critical then destroyed
        const rec = satMeshMap[satId];
        if (rec && rec.data.status !== 'destroyed') {
            rec.data.status = 'critical';
            if (rec.mesh.material) rec.mesh.material.color.setHex(0xff3366);
            if (rec.glow && rec.glow.material) rec.glow.material.color.setHex(0xff3366);
            setTimeout(() => {
                if (rec.mesh) { rec.mesh.visible = false; if (rec.glow) rec.glow.visible = false; }
                rec.data.status = 'destroyed';
                _spawnKessler(pos, 3 + Math.floor(Math.random() * 4));
            }, 7000);
        }
        // 2. Deactivate the debris piece (it disintegrated)
        if (debrisIdx >= 0 && debrisAlive[debrisIdx]) {
            debrisAlive[debrisIdx] = 0;
            _dbDummy.position.set(0, 0, 0); _dbDummy.scale.setScalar(0); _dbDummy.updateMatrix();
            debrisMesh.setMatrixAt(debrisIdx, _dbDummy.matrix);
            debrisMesh.instanceMatrix.needsUpdate = true;
        }
        // 3. Flash sphere
        const fl = new THREE.Mesh(
            new THREE.SphereGeometry(0.04, 8, 8),
            new THREE.MeshBasicMaterial({
                color: 0xffcc00, transparent: true, opacity: 1.0,
                blending: THREE.AdditiveBlending, depthWrite: false
            })
        );
        fl.position.copy(pos); scene.add(fl);
        _collFlashes.push({ mesh: fl, life: 1.0, ring: false });
        // 4. Expanding shockwave ring
        const rng = new THREE.Mesh(
            new THREE.RingGeometry(0.03, 0.06, 24),
            new THREE.MeshBasicMaterial({
                color: 0xff6600, transparent: true, opacity: 0.9,
                blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide
            })
        );
        rng.position.copy(pos);
        if (camera) rng.lookAt(camera.position);
        scene.add(rng);
        _collFlashes.push({ mesh: rng, life: 1.0, ring: true });
        // 5. Secondary flash (white core)
        const core = new THREE.Mesh(
            new THREE.SphereGeometry(0.018, 6, 6),
            new THREE.MeshBasicMaterial({
                color: 0xffffff, transparent: true, opacity: 1.0,
                blending: THREE.AdditiveBlending, depthWrite: false
            })
        );
        core.position.copy(pos); scene.add(core);
        _collFlashes.push({ mesh: core, life: 1.0, ring: false, fast: true });
        // 6. HUD alert
        const al = document.getElementById('globe-coll-alert');
        if (al) {
            al.textContent = '\u26a0 COLLISION: ' + satId + ' — Kessler cascade initiated';
            al.style.display = 'block'; al.style.opacity = '1';
            clearTimeout(al._t);
            al._t = setTimeout(() => {
                al.style.opacity = '0';
                setTimeout(() => { al.style.display = 'none'; }, 500);
            }, 6000);
        }
    }

    // ── Debris: Kessler daughter debris from collision ────────────────────────
    function _spawnKessler(pos, count) {
        const kCol = new THREE.Color(0xff2200);
        const GM = 3.986004418e5;
        const altKm = (pos.length() - EARTH_R) * 6371;
        let spawned = 0;
        for (let i = 0; i < DEBRIS_COUNT && spawned < count; i++) {
            if (debrisAlive[i]) continue;
            const a = 6371 + altKm;
            const incl = (Math.random() * 160 - 80) * Math.PI / 180;
            const raan = Math.random() * Math.PI * 2;
            const ci = Math.cos(incl), si = Math.sin(incl);
            const cr = Math.cos(raan), sr = Math.sin(raan);
            debrisPx[i] = cr; debrisPy[i] = sr; debrisPz[i] = 0;
            debrisQx[i] = -sr * ci; debrisQy[i] = cr * ci; debrisQz[i] = si;
            debrisTheta[i] = Math.random() * Math.PI * 2;
            debrisAlt[i] = altKm + (Math.random() - 0.5) * 60;
            debrisOmega[i] = Math.sqrt(GM / (a * a * a));
            debrisDecay[i] = 0.000015 * Math.pow(6371 / a, 4);
            debrisAlive[i] = 1;
            _dbDummy.scale.setScalar(1);
            _dbDummy.position.copy(pos).add(new THREE.Vector3(
                (Math.random() - .5) * .06, (Math.random() - .5) * .06, (Math.random() - .5) * .06));
            _dbDummy.updateMatrix();
            debrisMesh.setMatrixAt(i, _dbDummy.matrix);
            debrisMesh.setColorAt(i, kCol);
            spawned++;
        }
        if (debrisMesh.instanceColor) debrisMesh.instanceColor.needsUpdate = true;
        debrisMesh.instanceMatrix.needsUpdate = true;
        const hud = document.getElementById('globe-debris-hud');
        if (hud) hud.textContent = (DEBRIS_COUNT + spawned).toLocaleString() + ' TRACKED DEBRIS OBJECTS';
    }

    // ── Debris: animate flash effects ─────────────────────────────────────────
    function _updateFlashes(dtSec) {
        for (let i = _collFlashes.length - 1; i >= 0; i--) {
            const f = _collFlashes[i];
            f.life -= dtSec * (f.fast ? 3.5 : 1.4);
            if (f.life <= 0) {
                scene.remove(f.mesh);
                f.mesh.geometry.dispose(); f.mesh.material.dispose();
                _collFlashes.splice(i, 1);
            } else {
                f.mesh.material.opacity = Math.max(0, f.life * (f.ring ? 0.85 : 1.0));
                if (f.ring) f.mesh.scale.setScalar(1 + (1 - f.life) * 5);
                else f.mesh.scale.setScalar(0.4 + (1 - f.life) * 2.5);
            }
        }
    }

    // ── Init Three.js ─────────────────────────────────────────────────────────
    function _initGlobe() {
        if (globeInited) return;
        globeInited = true;

        const container = document.getElementById('cesiumContainer');
        if (!container) return;
        const W = container.clientWidth || 1200;
        const H = container.clientHeight || 800;

        renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
        renderer.setPixelRatio(window.devicePixelRatio);
        renderer.setSize(W, H);
        renderer.setClearColor(0x020810, 1);
        container.innerHTML = '';
        container.appendChild(renderer.domElement);

        scene = new THREE.Scene();
        camera = new THREE.PerspectiveCamera(45, W / H, 0.01, 1000);
        camera.position.set(0, 0, 3.2);

        raycaster = new THREE.Raycaster();
        mouse = new THREE.Vector2();

        controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.08;
        controls.minDistance = 1.15;
        controls.maxDistance = 8.0;
        controls.autoRotate = false;
        controls.enablePan = false;

        // Lighting
        scene.add(new THREE.AmbientLight(0x334466, 0.6));
        const sun = new THREE.DirectionalLight(0xffffff, 1.2);
        sun.position.set(5, 3, 5);
        scene.add(sun);

        // Stars
        const starPos = new Float32Array(8000 * 3);
        for (let i = 0; i < 8000; i++) {
            const phi = Math.acos(2 * Math.random() - 1);
            const th = 2 * Math.PI * Math.random();
            const r = 80 + Math.random() * 20;
            starPos[i * 3] = r * Math.sin(phi) * Math.cos(th);
            starPos[i * 3 + 1] = r * Math.cos(phi);
            starPos[i * 3 + 2] = r * Math.sin(phi) * Math.sin(th);
        }
        const starGeo = new THREE.BufferGeometry();
        starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
        scene.add(new THREE.Points(starGeo,
            new THREE.PointsMaterial({ color: 0xffffff, size: 0.12, sizeAttenuation: true })));

        // Earth
        const loader = new THREE.TextureLoader();
        const earthGeo = new THREE.SphereGeometry(EARTH_R, 64, 32);
        const dayTex = loader.load('assets/earth_day.jpg', undefined, undefined, () => {
            // fallback — use solid color if texture missing
            earthMesh.material.color.set(0x1a4a7a);
        });
        const bumpTex = loader.load('assets/earth_normal.jpg', undefined, undefined, () => { });
        const specTex = loader.load('assets/earth_specular.jpg', undefined, undefined, () => { });
        earthMesh = new THREE.Mesh(earthGeo, new THREE.MeshPhongMaterial({
            map: dayTex,
            bumpMap: bumpTex,
            bumpScale: 0.05,
            specularMap: specTex,
            specular: new THREE.Color(0x334466),
            shininess: 15,
        }));
        earthMesh.rotation.y = Math.PI;
        scene.add(earthMesh);

        // Atmosphere
        scene.add(new THREE.Mesh(
            new THREE.SphereGeometry(EARTH_R * 1.018, 64, 32),
            new THREE.MeshPhongMaterial({
                color: 0x4488ff, transparent: true, opacity: 0.12,
                side: THREE.FrontSide, blending: THREE.AdditiveBlending, depthWrite: false,
            })
        ));
        scene.add(new THREE.Mesh(
            new THREE.SphereGeometry(EARTH_R * 1.035, 64, 32),
            new THREE.MeshPhongMaterial({
                color: 0x2255ff, transparent: true, opacity: 0.05,
                side: THREE.BackSide, blending: THREE.AdditiveBlending, depthWrite: false,
            })
        ));

        satGroup = new THREE.Group();
        orbitGroup = new THREE.Group();
        scene.add(satGroup);
        scene.add(orbitGroup);

        renderer.domElement.addEventListener('click', _onGlobeClick);

        const ro = new ResizeObserver(() => {
            const w = container.clientWidth, h = container.clientHeight;
            if (!w || !h) return;
            renderer.setSize(w, h);
            camera.aspect = w / h;
            camera.updateProjectionMatrix();
        });
        ro.observe(container);

        _loadFleet();
    }

    // ── Fleet loading — reads istar_fleet_config.json directly (no missing endpoint) ──
    async function _loadFleet() {
        try {
            const res = await fetch('./istar_fleet_config.json');
            if (!res.ok) throw new Error('config not found');
            const cfg = await res.json();
            const sats = (cfg.satellites || []).map(s => ({
                id: s.id,
                plane: s.plane || 1,
                alt_km: s.alt_km || 500,
                vel_km_s: s.vel_km_s || 7.61,
                role: s.role || 'Unknown',
                status: 'nominal',
                lat: 0, lon: 0,
            }));
            if (!sats.length) throw new Error('empty config');
            _buildSatMeshes(sats);
            _startLoop();
            _initDebris();
            // Control panel + sat picker built AFTER meshes so picker is populated
            setTimeout(() => { _buildControlPanel(); _refreshSatPicker(); }, 100);
            const el = document.getElementById('globe-count');
            if (el) el.textContent = sats.length + ' sats · SGP4 PROPAGATION';
        } catch (e) {
            console.warn('[Globe] config load failed, using generated fleet:', e.message);
            // Fallback: generate 36 sats across 3 planes
            const sats = [];
            for (let i = 1; i <= 36; i++) {
                sats.push({
                    id: `I-STAR-${String(i).padStart(2, '0')}`,
                    plane: Math.floor((i - 1) / 12) + 1,
                    alt_km: 500, vel_km_s: 7.61,
                    role: 'SAR Radar', status: 'nominal',
                    lat: 0, lon: 0,
                });
            }
            _buildSatMeshes(sats);
            _startLoop();
            _initDebris();
            setTimeout(() => { _buildControlPanel(); _refreshSatPicker(); }, 100);
            const el = document.getElementById('globe-count');
            if (el) el.textContent = '36 sats · SIMULATED';
        }
    }

    // ── Build sat meshes ──────────────────────────────────────────────────────
    function _buildSatMeshes(sats) {
        while (satGroup.children.length) satGroup.remove(satGroup.children[0]);
        document.querySelectorAll('.globe-sat-label').forEach(e => e.remove());
        satMeshMap = {};

        sats.forEach(sat => {
            const [t1, t2] = _synthTLE(sat.id, sat.plane, sat.alt_km);
            let satrec;
            try { satrec = satellite.twoline2satrec(t1, t2); } catch { return; }

            const pos = _getSatPos(satrec, simTime);
            if (!pos) return;

            const col = _satColor(sat.plane, sat.status);

            // Core dot
            const mesh = new THREE.Mesh(
                new THREE.SphereGeometry(0.006, 8, 8),
                new THREE.MeshBasicMaterial({ color: col })
            );
            mesh.position.copy(_latLonToVec3(pos.lat, pos.lon, pos.alt));
            mesh._satId = sat.id;
            satGroup.add(mesh);

            // Glow
            const glow = new THREE.Mesh(
                new THREE.SphereGeometry(0.014, 8, 8),
                new THREE.MeshBasicMaterial({
                    color: col, transparent: true, opacity: 0.25,
                    blending: THREE.AdditiveBlending, depthWrite: false,
                })
            );
            glow.position.copy(mesh.position);
            satGroup.add(glow);

            // CSS label
            const label = document.createElement('div');
            label.className = 'globe-sat-label';
            label.textContent = sat.id.replace('I-STAR-', 'IS-');
            label.style.cssText = `position:absolute;pointer-events:none;font:bold 9px monospace;
                color:#${col.toString(16).padStart(6, '0')};white-space:nowrap;
                text-shadow:0 0 4px #000,0 0 8px #000;display:none;`;
            const panel = document.getElementById('globe-panel');
            if (panel) panel.appendChild(label);

            satMeshMap[sat.id] = { mesh, glow, label, satrec, data: sat };
        });
    }

    // ── Screen projection ─────────────────────────────────────────────────────
    function _project(vec3) {
        const w = renderer.domElement.clientWidth;
        const h = renderer.domElement.clientHeight;
        const v = vec3.clone().project(camera);
        return { x: (v.x * 0.5 + 0.5) * w, y: (-v.y * 0.5 + 0.5) * h, visible: v.z < 1.0 };
    }

    // ── Orbit track ───────────────────────────────────────────────────────────
    function _drawOrbitTrack(satId) {
        while (orbitGroup.children.length) orbitGroup.remove(orbitGroup.children[0]);
        if (!satId || !satMeshMap[satId]) return;
        const { satrec, data } = satMeshMap[satId];
        const col = PLANE_COL[data.plane] || 0x00d2ff;

        if (LAYERS.orbitLines) {
            const past = _buildArcVecs(satrec, simTime, -TRACK_PAST, 0, TRACK_STEP);
            if (past.length > 1) orbitGroup.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(past),
                new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.3, blending: THREE.AdditiveBlending })
            ));
            const future = _buildArcVecs(satrec, simTime, 0, TRACK_FUTURE, TRACK_STEP);
            if (future.length > 1) orbitGroup.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(future),
                new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending })
            ));
        }
        if (LAYERS.groundTracks) {
            const gnd = _buildArcVecs(satrec, simTime, 0, 90, TRACK_STEP)
                .map(v => v.normalize().multiplyScalar(1.003));
            if (gnd.length > 1) orbitGroup.add(new THREE.Line(
                new THREE.BufferGeometry().setFromPoints(gnd),
                new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.2 })
            ));
        }
    }

    // ── Animation loop ────────────────────────────────────────────────────────
    function _startLoop() {
        if (animFrame) cancelAnimationFrame(animFrame);
        function tick() {
            animFrame = requestAnimationFrame(tick);
            _tick++;

            if (_simPaused) { controls.update(); renderer.render(scene, camera); return; }

            const now = Date.now();
            const dt = Math.min(now - lastRealMs, 100); // cap at 100ms to avoid spiral
            lastRealMs = now;
            const dtSec = (dt / 1000) * CLOCK_MULT;
            simTime = new Date(simTime.getTime() + dt * CLOCK_MULT);

            if (earthMesh) earthMesh.rotation.y += (2 * Math.PI / 86400) * (dt / 1000) * CLOCK_MULT * 0.001;

            // Move satellites + apply drift offset
            Object.entries(satMeshMap).forEach(([sid, { mesh, glow, label, satrec, data }]) => {
                const pos = _getSatPos(satrec, simTime);
                if (!pos) return;
                // Drift/decay: reduce altitude over time
                if (satDriftState[sid]) {
                    satAltOffset[sid] = (satAltOffset[sid] || 0) + satDriftState[sid].rate * (dtSec / 3600);
                }
                const altDrop = satAltOffset[sid] || 0;
                const v = _latLonToVec3(pos.lat, pos.lon, Math.max(pos.alt - altDrop, 80));
                mesh.position.copy(v);
                if (glow) glow.position.copy(v);
                if (label && LAYERS.satellites) {
                    const sc = _project(v);
                    const vis = sc.visible && v.dot(camera.position.clone().sub(v)) > 0;
                    label.style.display = vis ? 'block' : 'none';
                    if (vis) { label.style.left = (sc.x + 10) + 'px'; label.style.top = (sc.y - 6) + 'px'; }
                }
            });

            // Debris update (every 3 ticks)
            if (_tick % 3 === 0) _updateDebris(dtSec * 3);

            // Shell visibility gate
            if (debrisMesh && _tick % 60 === 0) {
                let idx = 0;
                DEBRIS_SHELLS.forEach((shell, sIdx) => {
                    for (let i = 0; i < shell.count && idx < DEBRIS_COUNT; i++, idx++) {
                        if (!debrisAlive[idx]) continue;
                        const sc = _shellVisible[sIdx] ? 1 : 0;
                        // Zero-scale hides the instance without disrupting data
                        // Only update if state changed
                        const cur = _shellVisible[sIdx];
                        _dbDummy.scale.setScalar(cur ? 1 : 0.0001);
                        _dbDummy.position.set(debrisAlt[idx], 0, 0); // dummy pos, corrected next _updateDebris
                        _dbDummy.updateMatrix();
                        // skip — actual visibility controlled by opacity trick via color
                    }
                });
            }

            // Real proximity collision check (every 45 ticks, ~0.75s real)
            if (_tick % 45 === 2) _checkProximityCollisions();

            _updateFlashes(dt / 1000);

            if (selectedId && _tick % 180 === 0) _drawOrbitTrack(selectedId);

            if (followId && satMeshMap[followId]) {
                const target = satMeshMap[followId].mesh.position.clone().multiplyScalar(1.5);
                camera.position.lerp(target, 0.03);
                controls.target.lerp(satMeshMap[followId].mesh.position, 0.05);
            }

            if (_tick % 10 === 0) {
                const hud = document.getElementById('globe-simtime');
                if (hud) hud.textContent = simTime.toUTCString().split(' ')[4] + ' UTC';
            }

            controls.update();
            renderer.render(scene, camera);
        }
        tick();
    }

    // ── Click handler — GLOBE ONLY, never calls app.js ───────────────────────
    function _onGlobeClick(event) {
        const rect = renderer.domElement.getBoundingClientRect();
        mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
        mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
        raycaster.setFromCamera(mouse, camera);
        const hits = raycaster.intersectObjects(
            Object.values(satMeshMap).map(r => r.mesh).filter(Boolean)
        );
        if (hits.length > 0 && hits[0].object._satId) {
            _globeSelectSat(hits[0].object._satId);  // ← private, never touches app.js
        }
    }

    // ── _globeSelectSat — PRIVATE (was the bug: named selectSat before) ───────
    function _globeSelectSat(satId) {
        selectedId = satId;

        // Highlight selected, dim others — globe-only visual change
        Object.entries(satMeshMap).forEach(([id, { mesh, glow }]) => {
            const sel = id === satId;
            if (mesh.material) mesh.material.opacity = sel ? 1.0 : 0.5;
            if (glow && glow.material) {
                glow.material.opacity = sel ? 0.55 : 0.15;
                glow.scale.setScalar(sel ? 1.6 : 1.0);
            }
        });

        _drawOrbitTrack(satId);
        _updateInfoPanel(satId);
        // ✅ NO call to window.selectSat — drawer is independent
    }

    // ── Globe-own info panel (inside the globe panel, not the main drawer) ────
    function _updateInfoPanel(satId) {
        const rec = satMeshMap[satId];
        if (!rec) return;
        const sat = rec.data;
        const pos = _getSatPos(rec.satrec, simTime);
        if (pos) { sat.lat = Math.round(pos.lat * 100) / 100; sat.lon = Math.round(pos.lon * 100) / 100; sat.alt_km = Math.round(pos.alt * 10) / 10; }

        const $ = id => document.getElementById(id);
        if ($('gi-id')) $('gi-id').textContent = sat.id;
        if ($('gi-lat')) $('gi-lat').textContent = (sat.lat ?? '—') + '°';
        if ($('gi-lon')) $('gi-lon').textContent = (sat.lon ?? '—') + '°';
        if ($('gi-alt')) $('gi-alt').textContent = (sat.alt_km ?? '—') + ' km';
        if ($('gi-vel')) $('gi-vel').textContent = (sat.vel_km_s ?? '7.61') + ' km/s';
        if ($('gi-plane')) $('gi-plane').textContent = 'Plane ' + sat.plane;
        if ($('gi-role')) $('gi-role').textContent = sat.role || '—';
        if ($('gi-status')) {
            $('gi-status').textContent = (sat.status || 'nominal').toUpperCase();
            $('gi-status').style.color = sat.status === 'critical' ? 'var(--red)' :
                sat.status === 'drifting' ? 'var(--amber)' : 'var(--green)';
        }
        const panel = $('globe-sat-info');
        if (panel) panel.style.display = 'flex';
        const btn = $('globe-follow-btn');
        if (btn) {
            btn.onclick = () => window.globeFollowSat(sat.id);
            btn.textContent = followId === sat.id ? 'Unfollow' : 'Follow Sat';
        }
    }

    // ── Layer toggles ─────────────────────────────────────────────────────────
    window.globeToggleLayer = function (name, show) {
        LAYERS[name] = show;
        if (name === 'satellites') {
            Object.values(satMeshMap).forEach(({ mesh, glow, label }) => {
                mesh.visible = show;
                if (glow) glow.visible = show;
                if (label) label.style.display = show ? '' : 'none';
            });
        }
        if (name === 'debris' && debrisMesh) debrisMesh.visible = show;
        if (name === 'orbitLines' || name === 'groundTracks') _drawOrbitTrack(selectedId);
    };

    // ── Dashboard ↔ Globe bridge (PUBLIC API) ──────────────────────────────
    window.globeUpdateSatStatus = function (satId, status, data) {
        const rec = satMeshMap[satId];
        if (!rec) return;
        rec.data.status = status;
        const col = status === 'critical' ? 0xff3366
            : status === 'drifting' ? 0xffb800
                : status === 'maneuvering' ? 0x7c3aed
                    : status === 'destroyed' ? 0x444444
                        : 0x00d2ff;
        if (rec.mesh.material) rec.mesh.material.color.setHex(col);
        if (rec.glow?.material) { rec.glow.material.color.setHex(col); rec.glow.material.opacity = status === 'critical' ? 0.55 : 0.22; }
        if (rec.glow) rec.glow.scale.setScalar(status === 'critical' ? 1.9 : 1.0);
        // Drift / decay: progressive altitude loss visualisation
        if (status === 'drifting') {
            satDriftState[satId] = { rate: (data && data.driftRate) || 0.8 }; // km/h
        } else if (status === 'nominal' || status === 'maneuvering') {
            delete satDriftState[satId];
            delete satAltOffset[satId];
        } else if (status === 'destroyed') {
            delete satDriftState[satId];
            if (rec.mesh) rec.mesh.visible = false;
            if (rec.glow) rec.glow.visible = false;
        }
    };

    window.globeFireDebrisAt = function (satId) { _fireDebrisAt(satId); };

    // Listen for events dispatched by app.js
    window.addEventListener('celestis:satupdate', e => {
        const d = e.detail || {};
        if (!d.id || !d.status) return;
        window.globeUpdateSatStatus(d.id, d.status, d);
        // Update status lists
        _statusLists.drifting.delete(d.id); _statusLists.critical.delete(d.id);
        if (d.status === 'drifting') _statusLists.drifting.add(d.id);
        if (d.status === 'critical') _statusLists.critical.add(d.id);
        _refreshStatusLists();
    });

    // ── Control Panel ─────────────────────────────────────────────────
    function _buildControlPanel() {
        const panel = document.getElementById('globe-panel');
        if (!panel || document.getElementById('debris-ctrl')) return;
        const ctrl = document.createElement('div');
        ctrl.id = 'debris-ctrl';
        // Moved to LEFT side, no Kp/speed sliders
        ctrl.style.cssText = 'position:absolute;top:52px;left:10px;width:210px;max-height:calc(100% - 70px);overflow-y:auto;background:rgba(2,8,16,.92);border:1px solid rgba(0,210,255,.22);border-radius:10px;padding:12px 14px;font:11px/1.7 monospace;color:#a0c8e0;z-index:20;box-shadow:0 0 24px rgba(0,210,255,.08);scrollbar-width:thin;';
        ctrl.innerHTML = [
            '<div style="color:#00d2ff;font-weight:bold;font-size:12px;letter-spacing:1px;margin-bottom:10px">SIMULATION CONTROLS</div>',
            // Pause/Resume only — speed locked at 60x
            '<div style="display:flex;gap:6px;margin-bottom:12px">',
            '<button id="ctrl-pause" style="flex:1;padding:4px 0;background:rgba(0,210,255,.1);border:1px solid #00d2ff;color:#00d2ff;border-radius:5px;cursor:pointer;font:bold 10px monospace">PAUSE</button>',
            '<button id="ctrl-resume" style="flex:1;padding:4px 0;background:rgba(34,197,94,.08);border:1px solid #22c55e;color:#22c55e;border-radius:5px;cursor:pointer;font:bold 10px monospace">RESUME</button>',
            '</div>',
            // Follow satellite
            '<div style="color:#00d2ff;font-size:10px;letter-spacing:1px;margin-bottom:5px">FOLLOW SATELLITE</div>',
            '<select id="ctrl-sat-pick" style="width:100%;background:#05111f;color:#a0c8e0;border:1px solid rgba(0,210,255,.3);border-radius:4px;padding:3px;margin-bottom:5px;font:11px monospace"></select>',
            '<div style="display:flex;gap:6px;margin-bottom:12px">',
            '<button id="ctrl-follow" style="flex:1;padding:4px 0;background:rgba(124,58,237,.15);border:1px solid #7c3aed;color:#a78bfa;border-radius:5px;cursor:pointer;font:bold 10px monospace">▶ FOLLOW</button>',
            '<button id="ctrl-unfollow" style="flex:1;padding:4px 0;background:rgba(160,160,160,.08);border:1px solid #666;color:#aaa;border-radius:5px;cursor:pointer;font:bold 10px monospace">■ FREE</button>',
            '</div>',
            // Debris zone filter
            '<div style="color:#00d2ff;font-size:10px;letter-spacing:1px;margin-bottom:5px">DEBRIS ZONES</div>',
            '<label style="display:flex;gap:7px;align-items:center;margin-bottom:3px"><input type="checkbox" id="sh-0" checked><span style="color:#ffaa00">■</span>Chinese ASAT (780-850km)</label>',
            '<label style="display:flex;gap:7px;align-items:center;margin-bottom:3px"><input type="checkbox" id="sh-1" checked><span style="color:#ff4411">■</span>Cosmos/Iridium (600-650km)</label>',
            '<label style="display:flex;gap:7px;align-items:center;margin-bottom:3px"><input type="checkbox" id="sh-2" checked><span style="color:#ff6622">■</span>Soviet Legacy (1200-1400km)</label>',
            '<label style="display:flex;gap:7px;align-items:center;margin-bottom:10px"><input type="checkbox" id="sh-3" checked><span style="color:#5577ff">■</span>Low LEO (300-500km)</label>',
            // Drifting list
            '<div style="color:#ffb800;font-size:10px;letter-spacing:1px;margin-bottom:4px">● DRIFTING</div>',
            '<div id="ctrl-drifting-list" style="margin-bottom:10px;color:#555;font-size:10px">None</div>',
            // Critical list
            '<div style="color:#ff3366;font-size:10px;letter-spacing:1px;margin-bottom:4px">● CRITICAL / DECAYING</div>',
            '<div id="ctrl-critical-list" style="color:#555;font-size:10px">None</div>',
        ].join('');
        panel.appendChild(ctrl);
        // Pause / Resume
        ctrl.querySelector('#ctrl-pause').addEventListener('click', () => { _simPaused = true; });
        ctrl.querySelector('#ctrl-resume').addEventListener('click', () => { _simPaused = false; lastRealMs = Date.now(); });
        // Follow / Free
        ctrl.querySelector('#ctrl-follow').addEventListener('click', () => {
            const sid = ctrl.querySelector('#ctrl-sat-pick').value;
            if (sid) window.globeFollowSat(sid);
        });
        ctrl.querySelector('#ctrl-unfollow').addEventListener('click', () => {
            followId = null;
            if (controls) controls.target.set(0, 0, 0);
        });
        // Shell toggles
        [0, 1, 2, 3].forEach(s => {
            ctrl.querySelector('#sh-' + s).addEventListener('change', e => {
                _shellVisible[s] = e.target.checked;
            });
        });
    }

    function _refreshSatPicker() {
        const sel = document.getElementById('ctrl-sat-pick');
        if (!sel) return;
        sel.innerHTML = Object.keys(satMeshMap)
            .map(id => `<option value="${id}">${id}</option>`).join('');
    }

    function _refreshStatusLists() {
        const render = (elId, set, col) => {
            const el = document.getElementById(elId);
            if (!el) return;
            if (set.size === 0) { el.innerHTML = '<span style="color:#555">None</span>'; return; }
            el.innerHTML = [...set].map(id =>
                `<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:3px">
                  <span style="color:${col}">${id}</span>
                  <button onclick="window.globeFollowSat('${id}')" style="padding:1px 6px;background:rgba(124,58,237,.2);border:1px solid #7c3aed;color:#a78bfa;border-radius:3px;cursor:pointer;font:9px monospace">FOLLOW</button>
                </div>`
            ).join('');
        };
        render('ctrl-drifting-list', _statusLists.drifting, '#ffb800');
        render('ctrl-critical-list', _statusLists.critical, '#ff3366');
    }

    // ── Speed ─────────────────────────────────────────────────────────────────
    window.setGlobeSpeed = function (mult) {
        CLOCK_MULT = mult;
        document.querySelectorAll('.globe-speed-btn').forEach(b =>
            b.classList.toggle('active', b.textContent.startsWith(String(mult)))
        );
    };

    // ── Follow camera ─────────────────────────────────────────────────────────
    window.globeFollowSat = function (satId) {
        const btn = document.getElementById('globe-follow-btn');
        if (followId === satId) {
            followId = null;
            if (btn) btn.textContent = 'Follow Sat';
            if (controls) controls.target.set(0, 0, 0);
        } else {
            followId = satId;
            if (btn) btn.textContent = 'Unfollow';
            _globeSelectSat(satId);
        }
    };

    // ── Open / Close (public) ─────────────────────────────────────────────────
    window.openGlobe = function () {
        const panel = document.getElementById('globe-panel');
        if (!panel) return;
        panel.classList.add('open');
        document.body.classList.add('globe-open');
        // Re-show labels that were hidden on close (fixes disappearing name bug)
        document.querySelectorAll('.globe-sat-label').forEach(e => { e.style.display = ''; });
        simTime = new Date();
        lastRealMs = Date.now();
        if (!globeInited) {
            setTimeout(_initGlobe, 80);
        } else if (!animFrame) {
            lastRealMs = Date.now();
            _startLoop();
        }
    };

    window.closeGlobe = function () {
        const panel = document.getElementById('globe-panel');
        if (!panel) return;
        panel.classList.remove('open');
        document.body.classList.remove('globe-open');
        if (animFrame) { cancelAnimationFrame(animFrame); animFrame = null; }
        // HIDE labels instead of removing — so they survive re-open
        document.querySelectorAll('.globe-sat-label').forEach(e => { e.style.display = 'none'; });
        followId = null;
        const info = document.getElementById('globe-sat-info');
        if (info) info.style.display = 'none';
    };

})(); // end IIFE — nothing leaks to global scope except openGlobe/closeGlobe/etc.
