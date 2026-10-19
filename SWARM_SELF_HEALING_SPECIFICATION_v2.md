# CELESTIS-HEAL v2 — Space-Weather-Aware Self-Healing for LEO Satellite Constellations
### Revised Specification, Critical Review of v1, Novelty Thesis & Execution Plan
**Author:** Srinjay Singh · Inceptrix SAT · October 2026
**Status:** Planning document (no code). Supersedes `SWARM_SELF_HEALING_SPECIFICATION.md` (v1).

---

## 0. One-Paragraph Summary

CELESTIS's unique asset is **storm-time drag knowledge**: live space-weather ingestion, a drag-residual PINN, and TLE-derived ground truth across 26 storms. v1 of this spec built its self-healing feature around techniques that are already published (differential-drag phasing, Pontryagin neural networks, multi-agent CBFs, capability-aware auctions) and attached performance claims that the physics does not support. v2 re-centres the feature on the one problem CELESTIS is best placed to solve and that is poorly served today: **during a geomagnetic storm, a constellation cannot tell whether a satellite is falling out of its slot because of the weather or because it is broken — and it therefore cannot heal correctly.** CELESTIS-HEAL v2 = (1) storm-aware fault detection & isolation using the fleet itself as a thermosphere sensor, (2) closed-loop density assimilation back into the PINN, (3) uncertainty-calibrated, storm-forecast-scheduled in-plane re-phasing, and (4) capability-aware graceful degradation of the Walker geometry, all validated against real storm replays with model mismatch.

---

## 1. Critical Review of v1

### 1.1 Factual / Physics Errors

| # | v1 Claim | Problem | Correction |
|---|---|---|---|
| E1 | HCW + J₂ relative dynamics for the 36-satellite I-STAR fleet | HCW is a **close-formation linearisation** (valid to tens of km). I-STAR is a Walker constellation: in-plane neighbours are ~30° apart (~3,600 km), planes are 120° apart in RAAN. | Model the constellation in **mean orbital elements** (mean argument of latitude *u*, semi-major axis *a*, RAAN Ω). HCW/ROE only for close approaches. |
| E2 | Satellites can swap slots across planes | Cross-plane transfer of 120° RAAN is physically infeasible. Even differential-J₂ drift with a 50 km altitude offset gives δΩ̇ ≈ 5×10⁻⁹ rad/s → ~13 years for 120°. | **Healing is in-plane only** (12 satellites per plane). Cross-plane healing is out of scope. |
| E3 | "Edge-on → stays higher → moves faster" | Inverted. A higher orbit has a **lower** mean motion. More drag → lower *a* → higher mean motion → satellite moves **ahead**. | Use δn = −(3n / 2a)·δa; drag-induced phase drift Δy ≈ (3/2)·f·t². |
| E4 | "Hohmann contradiction" in `app.js` `calcDeltaV()` | For small ΔH (0.1–5 km) the Edelbaum low-thrust Δv between near-circular orbits equals the first-order Hohmann Δv (½·v·Δh/r). The Δv budget in `app.js` is approximately correct; only burn **scheduling** differs. | Keep the Δv formula; add burn-arc duration and multi-revolution scheduling. |
| E5 | Fleet parameters 450 kg, 60 mN, A = 0.4–3.2 m² | `istar_fleet_config.json` specifies **~150 kg, 5 mN, A = 0.8 m²**. README and config disagree. | Single source of truth = `istar_fleet_config.json`; reconcile README. |
| E6 | `u* = −λ_v/‖λ_v‖` as the fuel-optimal law | That is the minimum-time / thrust-direction law. Fuel-optimal low-thrust control is bang–off–bang with a switching function. | If optimal control is needed, state the correct cost and switching structure. |
| E7 | CELESTIS "already has a real-time density PINN" | The PINN predicts a **storm-integrated altitude-drop residual** (scalar per storm per satellite) from 7–15 tabular features, trained on 294 rows; OOD R² is 0.39–0.63. It is not a control-grade, along-track density predictor and has no uncertainty output. | Add uncertainty quantification, retrain on more data, and treat it as a **prior**, not ground truth. |

### 1.2 Unsupported Performance Claims (Removed)

| v1 Claim | Reality Check (I-STAR config: 150 kg, 5 mN, Cd 2.2, A ≈ 0.2–0.8 m², 500 km) |
|---|---|
| "Propellant-neutral healing; up to 78% fuel saved" | Differential-drag acceleration ≈ ½ρv²ΔB ≈ **1.3×10⁻⁷ m/s²** (ρ ≈ 5×10⁻¹³ kg/m³). Thruster acceleration ≈ **3.3×10⁻⁵ m/s²** (~260× larger). |
| "Coverage restored in < 12 hours" | Re-phasing after losing 1 of 12 satellites needs shifts up to ~15° (~1,800 km). Drag only: **~50 days** (quiet), ~2–3 weeks with storm enhancement. Thrust: **~3 days**, Δv ≈ 9 m/s, **≈ 0.08 kg Xe** of 12 kg on board. |
| "300% lifetime extension" | Fuel cost of thrust healing is already small. The real cost of drag-based healing is **lost altitude** (orbital energy), not fuel. |
| "Healing in < 60 s"; "planning in < 15 ms" | Planning latency is irrelevant when the manoeuvre takes days. The meaningful timescales are detection latency (orbits) and time-to-heal (days). |
| "Mathematically proven 0% collision" | CBF guarantees hold only under model, delay, and actuator-feasibility assumptions; with 3×10⁻⁵ m/s² of control authority, a 10 Hz CBF cannot change relative motion meaningfully. In-plane members never pass each other during re-phasing. The real hazards are **debris** and **polar plane-crossing timing**. |

### 1.3 Novelty Claims That Are Prior Art

| v1 "Novelty" | Existing Work (verify each in a literature review) |
|---|---|
| Differential-drag formation/constellation phasing | Leonard (1989); Bevilacqua & Romano (2008); operational on Planet Labs Flock (Foster et al., 2018); CYGNSS; ORBCOMM. |
| "Opportunistic" use of storm density for phasing | Discussed informally in operator literature; the idea alone is not defensible novelty. |
| P²-INN (Pontryagin PINN) | **"Pontryagin Neural Networks" (PoNN)** — Schiassi, Furfaro, D'Ambrosio et al. (2021–2023), including orbit transfers. Real-time DNN guidance for low-thrust: Izzo & Öztürk (2021), "G&CNETs". The name and concept are taken. |
| Multi-agent / delay-robust CBFs | Wang, Ames & Egerstedt (2017); spacecraft CBFs — Breeden & Panagou; time-delay CBFs — Molnár, Ames et al. |
| Capability-aware task re-allocation (Γ-matrix + CBBA) | CBBA with heterogeneous agents (Choi, Brunet & How, 2009); health-aware planning (Bethke & How). Cosine similarity on a 5×3 matrix is also technically unsound: it is scale-invariant, so a satellite at 10% on every subsystem scores the same as one at 100%. |

### 1.4 What v1 Got Right (Retained)
- Moving from single-satellite advisory to **fleet-level** decisions.
- Treating partial degradation as **graded, per-subsystem** rather than binary.
- Human-on-the-loop approval and an audit trail.
- Using CELESTIS's space-weather pipeline as the differentiator.

---

## 2. The Real Problem Statement

> **During a geomagnetic storm, a LEO constellation sees one or more satellites decay out of their slots. Operators must decide, under large density uncertainty and limited contact time: Is this weather or a fault? Which satellites are actually degraded? Should we heal now, wait for the storm to pass, or use the storm? How do we re-space the plane so coverage over the area of interest is preserved without burning lifetime or risking a decay below the safe floor?**

Today this decision is made manually, after the storm, using density models that are 30–80% wrong at storm peak. A satellite with a stuck panel or under-performing thruster looks identical to a healthy satellite in a denser patch of thermosphere. Misdiagnosis leads to two failure modes:
- **False fault:** a healthy satellite is safed or demoted, opening an unnecessary coverage gap.
- **Missed fault:** a degraded satellite is commanded with the standard recovery plan, under-delivers, and keeps decaying.

---

## 3. Novelty Thesis (v2)

The defensible contribution is not any single algorithm. It is a **closed loop that couples space-weather uncertainty, fleet-level fault isolation, and constellation reconfiguration** — something CELESTIS can do because it already owns the storm-drag model and data.

```
            ┌──────────────────────────────────────────────────────────────┐
            │                CELESTIS-HEAL v2 CLOSED LOOP                  │
            │                                                              │
  NOAA/DONKI│   ┌───────────────┐    ┌───────────────┐    ┌─────────────┐  │
  OMNI2 ────┼──►│ PINN density  │───►│ N1 Fleet FDI  │───►│ N4 Capability│  │
  forecasts │   │ prior + UQ    │    │ weather vs    │    │ aware slot   │  │
            │   └──────▲────────┘    │ fault, B_i    │    │ allocation   │  │
            │          │             └──────┬────────┘    └──────┬──────┘  │
            │   ┌──────┴────────┐           │                    │         │
            │   │ N2 Fleet-as-  │◄──────────┘                    ▼         │
            │   │ sensor density│                    ┌──────────────────┐  │
            │   │ assimilation  │                    │ N3 Storm-scheduled│  │
            │   └───────────────┘                    │ chance-constrained│  │
            │                                        │ re-phasing        │  │
            │                                        └────────┬─────────┘  │
            │                                                 ▼            │
            │                         Safety check (debris COLA, altitude  │
            │                         floor, plane-crossing) → Operator    │
            └──────────────────────────────────────────────────────────────┘
```

### N1 — Storm-Aware Fleet Fault Detection & Isolation ("Weather or Fault?")
**Idea.** Satellites in the same plane at the same altitude fly through nearly the same thermosphere within minutes of each other. Their along-track decelerations therefore share a **common-mode** density term. Each satellite's deviation from that common mode isolates **satellite-specific** causes.

**Formulation.**
- Observed drag acceleration for satellite *i*: aᵢ = ½·ρ(t, uᵢ)·v²·Bᵢ + εᵢ, with Bᵢ = Cd·Aᵢ/mᵢ.
- Factor density as ρ(t, u) = ρ_PINN(t, u)·κ(t), where κ is a shared fleet correction (storm-time model error).
- Jointly estimate {κ(t), B₁…B₁₂} per plane from GNSS/TLE orbit-determination residuals (multi-satellite filter or sliding-window least squares / factor graph).
- **Weather:** κ deviates, all Bᵢ stay within nominal bounds.
- **ADCS / attitude fault:** a single Bᵢ deviates (effective area changed).
- **Propulsion fault:** commanded Δv vs. OD-estimated Δv mismatch on a manoeuvre arc, with drag explained by κ·Bᵢ.
- Decision thresholds come from **calibrated intervals** (conformal prediction), giving a controlled false-alarm rate during storms.

**Why it is new (to verify).** Drag-based density estimation from satellites and ballistic-coefficient estimation each exist separately. Using **fleet common-mode separation with an ML storm prior to isolate actuator/attitude faults during storms**, with a calibrated false-alarm guarantee, is the candidate contribution.

### N2 — The Constellation as a Thermosphere Sensor (Closed-Loop Assimilation)
**Idea.** The κ(t) estimated in N1 is a real-time, multi-point storm-density measurement. Feed it back to:
- update the PINN's predictions online (Bayesian/ensemble update or conformal recalibration),
- tighten the uncertainty used by N3 and by the existing COLA engine,
- log it as new training data (directly attacking the 294-row dataset limitation).

**Contribution.** A self-improving loop in which the act of monitoring the fleet improves the storm model that protects the fleet.

### N3 — Uncertainty-Calibrated, Storm-Scheduled Re-phasing
**Idea.** Treat healing as a **decision over a horizon that includes the space-weather forecast** (DONKI CME arrival, NOAA Kp forecast), not as an immediate manoeuvre.

**Decision options per plane:**
1. **Heal now with thrust** — fastest, small Xe cost.
2. **Wait** — if a CME is forecast, defer so the plan is not invalidated by the storm.
3. **Storm-assisted phasing** — schedule high-/low-drag attitude windows to coincide with forecast density enhancement, plus thrust trim.
4. **Pre-emptive raise** — before a forecast storm, raise satellites predicted to approach the altitude floor.

**Formulation (chance-constrained).**
- Decision variables: thrust arcs, attitude (area) schedule, timing.
- Objective: minimise weighted (time-to-restore revisit, Δv, altitude lost, fleet altitude/fuel variance).
- Constraints: P(final phase error ≤ ε) ≥ 1 − δ and P(altitude ≥ h_floor) ≥ 1 − δ, using density scenarios sampled from the calibrated PINN + forecast uncertainty (scenario approach or sample-average approximation).
- In-plane phasing law: shift Δu via a temporary semi-major-axis offset δa (δn = −3nδa/2a), e.g. δa = 5 km → ~15° in ~2.5 days for Δv ≈ 5.5 m/s.

**Contribution.** Re-phasing decisions that explicitly reason about storm forecasts and **calibrated** ML density uncertainty, with empirically verifiable constraint-satisfaction rates.

### N4 — Capability-Aware Graceful Degradation of the Walker Plane
**Idea.** When a satellite is degraded (from N1), re-plan the **plane**, not just the satellite.

- **Capability model:** per-subsystem health hᵢ ∈ [0, 1] (propulsion, ADCS, power, payload type, link) used as **hard feasibility constraints** (e.g. a SAR slot requires SAR payload health ≥ 0.8 and power ≥ 0.7) plus a utility term. No cosine similarity.
- **Re-phasing strategies compared:**
  - *Global re-space:* 11 satellites evenly at 32.7°.
  - *Local repair:* only neighbours shift to split the gap.
  - *Role swap:* a degraded SAR node takes an optical/relay role; a healthy node moves into the SAR-critical phase.
- **Objective:** revisit/coverage over defined areas of interest (e.g. LAC, Arunachal, Indian Ocean Region) minus Δv/altitude cost, with lifetime-equity terms.
- **Solver:** 12-satellite in-plane assignment is a small MILP / Hungarian problem solved centrally in milliseconds. Distributed consensus is an **optional** autonomy mode, justified by ground-contact gaps, not compute.

### Safety Layer (Required, Not Claimed as Novelty)
- Screen every planned trajectory with the existing `cola_engine.py`, upgraded to **anisotropic RIC covariance** projected onto the B-plane.
- Altitude-floor chance constraint (from N3).
- Polar plane-crossing timing check between planes during re-phasing.
- Human-on-the-loop approval, with every recommendation logged.

---

## 4. Feature Set Inside CELESTIS

| ID | Feature | Description |
|---|---|---|
| F1 | **Truth Simulator & Anomaly Injection** | 36-satellite propagator with J₂ and attitude-dependent drag. "Truth" density from replayed real storms (Swarm/GRACE-FO accelerometer density or JB2008), while CELESTIS "believes" its PINN — deliberate model mismatch. Inject faults: thruster efficiency, stuck attitude (area), power loss, payload loss. |
| F2 | **"Weather or Fault?" Panel** | Per-plane common-mode density factor κ(t) with uncertainty band; per-satellite Bᵢ trend; fault classification with confidence and calibrated false-alarm rate. |
| F3 | **Healing Planner** | Side-by-side options (heal now / wait / storm-assisted / pre-raise) with time-to-restore, Δv, Xe kg, altitude lost, and probability of meeting phase and altitude constraints. |
| F4 | **Coverage & Revisit HUD** | Revisit time and gap duration over configured areas of interest: nominal → at failure → projected after healing. |
| F5 | **Globe Overlay** | Plane gap visualisation, planned phase shifts, uncertainty tubes on predicted positions, flagged satellites. |
| F6 | **Storm Replay Benchmark Mode** | Replay Halloween 2003, Starlink Feb-2022, Gannon May-2024 with injected faults; score every strategy automatically. |
| F7 | **Operator Approval & Audit Log** | Human-on-the-loop authorisation; full decision trace (inputs, uncertainty, chosen option). |
| F8 | **Assimilation Monitor** | Shows PINN prior vs. fleet-assimilated density and how predictions tighten over a storm. |

### Proposed API Additions (FastAPI, port 8001)
| Endpoint | Method | Purpose |
|---|---|---|
| `/api/heal/sim/state` | GET | Truth-sim fleet state (mean elements, health, altitude) |
| `/api/heal/sim/inject` | POST | Inject fault / select storm replay |
| `/api/heal/fdi` | GET | κ(t), Bᵢ estimates, fault classifications + confidence |
| `/api/heal/plan` | POST | Generate and compare healing options for a plane |
| `/api/heal/approve` | POST | Operator approval → execute in sim |
| `/api/heal/coverage` | GET | Revisit/gap metrics over areas of interest |
| `/api/heal/benchmark` | POST | Run a storm-replay Monte Carlo and return the scorecard |

---

## 5. Prerequisite Fixes in the Existing Codebase

| # | Issue | Required Action |
|---|---|---|
| P1 | Fleet parameters inconsistent (config 150 kg / 5 mN / 0.8 m² vs. README 450 kg / 60 mN) | Make `istar_fleet_config.json` the single source of truth; add min/max drag area per satellite; correct README. |
| P2 | Backend propagates 5 real satellites; frontend simulates 36 I-STAR satellites with random values | Backend must own the 36-satellite state through one simulator. |
| P3 | PINN has no uncertainty output; 294 samples; OOD R² 0.39–0.63 | Add deep ensembles or MC-dropout + conformal calibration; expand the dataset; report calibration (coverage of intervals). |
| P4 | PINN predicts storm-integrated ΔH, not density along track | Add a density-scaling formulation (κ factor) usable by the propagator. |
| P5 | COLA uses isotropic σ = max(d/3, R) | Use anisotropic RIC covariance (TLE-age-based growth) projected onto the B-plane. |
| P6 | No J₂ or attitude-dependent area in the propagator | Adopt a validated library (Orekit with box-wing drag, or Basilisk) as the truth propagator. |
| P7 | Port inconsistencies (4000 vs. 8001) across docs/files | Standardise on one port. |
| P8 | Subsystem telemetry in `app.js` is `Math.random()` | Replace with simulator-driven health state. |

---

## 6. Execution Plan

| Phase | Weeks | Scope | Exit Criteria |
|---|---|---|---|
| **0. Literature & Prior-Art Review** | 1–2 | Read and summarise: differential-drag phasing (Leonard, Bevilacqua, Planet/Foster), drag-derived density (Doornbos, Picone, Sutton), ballistic-coefficient estimation, constellation reconfiguration / graceful degradation, PoNN, conformal prediction for time series, scenario-based chance constraints. | Written gap analysis confirming or narrowing N1–N4. |
| **1. Foundations** | 3–5 | Fix P1, P2, P7, P8. Truth simulator (Orekit/Basilisk) for 36 satellites with J₂ + box-wing drag. Storm replay using OMNI2 drivers and real/alternate density. | Simulator reproduces historical TLE decay of ≥ 3 reference satellites within agreed error. |
| **2. Uncertainty-Aware Density** | 6–8 | P3, P4. Ensemble PINN + conformal intervals; κ-factor formulation; expand training data. | Empirical interval coverage within ±5% of nominal on held-out storms. |
| **3. N1 + N2: Fleet FDI & Assimilation** | 9–12 | Joint κ/Bᵢ estimator; fault classifier; online assimilation loop. | Fault isolation accuracy and false-alarm rate beat a single-satellite threshold baseline across all replayed storms. |
| **4. N3 + N4: Planner & Allocator** | 13–17 | In-plane phasing optimiser; chance constraints via density scenarios; capability-aware re-spacing (global / local / role swap); safety layer (P5, altitude floor, plane crossing). | Constraint-satisfaction rate ≥ 1 − δ in Monte Carlo; revisit restored with lower altitude/Δv cost than baselines. |
| **5. CELESTIS Integration** | 18–21 | API endpoints; UI features F1–F8; human-on-the-loop flow. | End-to-end demo: Gannon-2024 replay + injected ADCS fault → correct diagnosis → approved plan → restored coverage. |
| **6. Evaluation & Write-Up** | 22–26 | Full benchmark campaign; ablations (no PINN, no fleet common-mode, no forecast); paper draft. | Results tables, figures, reproducible scripts, manuscript. |

---

## 7. Evaluation Protocol

**Scenarios:** 3 historical storms × {no fault, ADCS stuck-area fault, 50% thruster degradation, total satellite loss, two simultaneous faults} × Monte Carlo over density-model error and OD noise.

**Truth vs. belief:** truth density from accelerometer-derived data or an independent model (JB2008/DTM); CELESTIS sees only NRLMSISE-00 + PINN + fleet observations.

**Baselines:**
1. No healing.
2. Thrust-only immediate re-phasing with NRLMSISE-00.
3. Differential-drag phasing with NRLMSISE-00 (Planet-style).
4. Single-satellite threshold fault detection.

**Metrics:**
| Category | Metric |
|---|---|
| Diagnosis | Fault isolation accuracy, detection latency (orbits), false-alarm rate during storms |
| Healing | Time to restore revisit target, max/mean revisit gap over AOIs |
| Cost | Δv and Xe per satellite, altitude lost, fleet lifetime variance |
| Safety | Empirical P(altitude < floor), P(phase error > ε), COLA alerts on planned trajectories |
| Model | Density interval coverage, sharpness, improvement from assimilation |

**Ablations:** remove PINN prior; remove fleet common-mode; remove forecast scheduling; replace conformal with uncalibrated intervals.

---

## 8. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| N1 identifiability: κ and Bᵢ trade off | Use ≥ 3 satellites per plane, attitude-known periods, and manoeuvre arcs; report observability analysis. |
| PINN too weak to be a useful prior | Fall back to NRLMSISE/JB2008 prior; novelty of N1/N2 holds with any prior. |
| Prior art found during Phase 0 | Narrow the claim to the specific coupling (calibrated FDI + forecast-scheduled reconfiguration). |
| Fictional fleet limits credibility | Also run on a real constellation analogue (e.g. a Planet or Starlink shell with public TLEs) for N1 validation. |
| Scope creep | Distributed consensus, learned guidance, and CBFs are explicitly deferred. |

---

## 9. What You Need

**Data:** OMNI2 (cached), NOAA SWPC/DONKI (live), Space-Track TLEs (cached), Swarm / GRACE-FO / CHAMP accelerometer-derived density (TU Delft thermosphere data portal), JB2008 drivers.
**Tools:** Orekit (Python wrapper) or Basilisk; PyTorch; a MILP/QP solver (OR-Tools, CVXPY with OSQP/ECOS); existing CELESTIS stack.
**Skills to strengthen:** orbit determination & estimation (EKF / batch least squares), conformal prediction, scenario-based stochastic optimisation, constellation coverage analysis.

---

## 10. Publication Positioning

**Working title:** *"Weather or Fault? Storm-Aware Fault Isolation and Uncertainty-Calibrated Reconfiguration for LEO Constellations."*
**Target venues:** AIAA Journal of Guidance, Control, and Dynamics; Acta Astronautica; Advances in Space Research; AGU Space Weather; AAS/AIAA Astrodynamics Specialist Conference.
**Claims to make only after Phase 0 confirms the gap:**
1. Fleet common-mode separation with an ML storm prior for actuator/attitude fault isolation during storms, with calibrated false-alarm control.
2. Closed-loop assimilation of fleet-derived density into the storm model.
3. Forecast-scheduled, chance-constrained in-plane re-phasing under calibrated ML density uncertainty.

Patent filing only after a formal prior-art search.

---

<div align="center"><i>CELESTIS-HEAL v2 · Srinjay Singh · Inceptrix SAT · 2026</i></div>
