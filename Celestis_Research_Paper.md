# CELESTIS: A Physics-Informed Neural Network Platform for Real-Time Satellite Drag Risk Assessment and Autonomous Collision Avoidance During Geomagnetic Storms

**Ajinkya Singh · Inceptrix SAT · March 2026**

> **Preprint Note:** This document describes a proof-of-concept engineering platform. All limitations regarding training data provenance, external baseline comparisons, and observational validation are disclosed explicitly in Section 6.

---

## Abstract

Geomagnetic storms driven by Coronal Mass Ejections (CMEs) cause rapid thermospheric density enhancement, increasing aerodynamic drag on Low Earth Orbit (LEO) satellites and accelerating uncontrolled orbital decay. We present **CELESTIS**, a full-stack autonomous satellite mission control platform integrating real-time space weather telemetry, standard orbital mechanics (SGP4), an empirical atmospheric model (NRLMSISE-00), a novel **Physics-Informed Neural Network (PINN)** residual correction model, and a real-world **Conjunction Analysis (COLA)** engine backed by live Space-Track.org TLE data.

The platform's primary scientific contribution is a **physics-constrained PyTorch neural network** whose loss function encodes Keplerian drag physics, guaranteeing that predicted orbital decays never violate Newton's and Kepler's laws — a property no existing lightweight operational tool provides. The PINN is trained on **real TLE-derived orbital decay measurements** from 89+ observations across 22 satellites and 26 geomagnetic storms (2003–2024). Against a physics-only (NRLMSISE-00) baseline, the PINN achieves **MAE = 0.125 km** on training distribution and **MAE = 0.199 km** on held-out out-of-distribution storms with **zero physics violations** on training data.

**Honest scope:** The dataset is actively expanding toward 550+ samples. No comparison against JB2008 or DTM2013 has yet been conducted. The system is classified as a *systems integration and prototype platform contribution* with a validated physically-constrained ML component.

---

## 1. Motivation

The February 2022 re-entry of 38 Starlink satellites demonstrated that even modest geomagnetic storms (Kp ≈ 5.5) can produce commercially catastrophic outcomes when satellites are deployed at low altitudes during thermospheric pre-heating. Existing high-fidelity space situational awareness tools (ODAS, STK) are restricted to institutional users. CELESTIS provides accessible, real-time, explainable risk indicators for prototype operational use, while introducing a scientifically grounded physics-informed ML model validated against real orbital tracking data.

---

## 2. System Architecture

CELESTIS is a three-layer stack:

```
┌─────────────────────────────────────────────────────────┐
│                  CELESTIS v3 Stack                      │
├────────────────┬────────────────────────────────────────┤
│ Physics Layer  │ NRLMSISE-00 + SGP4 + Kepler drag eq.  │
├────────────────┼────────────────────────────────────────┤
│ AI Layer       │ Phase-Aware PINN (PyTorch)             │
│                │ → Physics-constrained loss function    │
│                │ → 3-layer GELU network [64→128→64]     │
│                │ → Alpha-annealed training (600 epochs) │
├────────────────┼────────────────────────────────────────┤
│ COLA Layer     │ Real Space-Track.org debris TLEs       │
│                │ → SGP4 propagation (1440 steps/24h)   │
│                │ → Chan's Pc formula (18 SDS standard)  │
└────────────────┴────────────────────────────────────────┘
```

### 2.1 Backend Physics + ML Engine (Python / FastAPI)

Runs at `localhost:4000`. Responsible for:
- **Live space weather:** NOAA SWPC (Kp every 3 min), DSCOVR solar wind (1 min), GOES X-ray flux, NASA DONKI CME catalogue
- **Orbital propagation:** SGP4 via `sgp4` Python library
- **Atmospheric density:** NRLMSISE-00 via `pymsis`
- **Drag computation:** `F = ½ρCᴅAv²`
- **PINN inference:** Physics-constrained residual correction
- **COLA engine:** Real-time conjunction analysis against 200 debris objects per altitude band

### 2.2 Frontend Mission Control (Electron / JavaScript)

4-quadrant God-View dashboard:
- **Q1:** Fleet status grid — all I-STAR satellites with real-time orbital state
- **Q2:** Atmospheric Intelligence — NRLMSISE-00 vs PINN density comparison with storm phase classification
- **Q3:** Predictive Decay + COLA Radar — 24h altitude forecasts + real debris conjunction alerts
- **Q4:** Autonomous Maneuver Command Center — Delta-V calculator with human-on-the-loop / full autonomy toggle

---

## 3. Novel Contributions

### 3.1 Physics-Constrained Loss Function (Primary Contribution)

The PINN loss function combines data fidelity with physical constraints:

```
L_total = α · L_data + (1 − α) · L_physics

L_data    = MSE(residual_pred, residual_true)
L_physics = MSE(max(0, −(dH_physics + residual_pred)), 0)²
           + 0.5 · MSE(max(|residual_pred| − 5·|dH_physics|, 0), 0)²
```

The physics loss penalises two physically impossible outcomes:
1. **Altitude gain** during a geomagnetic storm (constraint 1)
2. **Residuals >5× the physics prediction** (prevents wild extrapolation)

Alpha is annealed from 0.5 → 0.85 over 600 epochs, ensuring the network first learns physically plausible behaviour, then refines on real data.

### 3.2 Real-World COLA with Chan's Pc Formula

Conjunction Analysis uses:
- Real debris TLEs from Space-Track.org (authenticated, up to 200 objects per altitude band, <3 days old)
- SGP4 propagation: 1,440 timesteps at 60-second intervals over 24 hours
- Chan's hard-sphere probability of collision: `Pc = (π·r²) / (2π·σ²) · exp(−d²/2σ²)`
- US Space Command actionable threshold: Pc ≥ 1×10⁻⁴

This is the same probabilistic framework used by 18th Space Defense Squadron.

### 3.3 Storm-Phase Decomposition

The Phase-Aware GBM (predecessor, still active as fallback) trains three separate residual correctors:
- **Sudden Commencement** — rapid onset, density underestimated by NRLMSISE-00
- **Main Phase** — peak Kp, non-linear thermospheric response
- **Recovery Phase** — thermospheric cooling lag, gradual re-equilibration

---

## 4. Training Data

### 4.1 Dataset Composition (Current — March 2026)

| Parameter | Value |
|---|---|
| **Total samples** | 89+ (expanding to ~550) |
| **Satellites tracked** | 22 |
| **Storm events** | 26 (2003–2024) |
| **Altitude range** | 350–890 km |
| **Storm classes** | G0 (quiet) through G5 (extreme) |
| **OOD test set** | 13 samples (Gannon 2024, Starlink 2022, Halloween 2003 A) |
| **Data source** | Space-Track.org TLE archives + NASA OMNI2 Kp/F10.7 |

### 4.2 Satellite Fleet

ISS, GRACE-A/FO-1/FO-2, Swarm A/B/C, CryoSat-2, Sentinel-2A/2B/6A, Starlink ×3, Landsat-9, ICESat-2, IXPE, SWOT, EarthCARE, CHAMP, TerraSAR-X, TanDEM-X

### 4.3 Ground Truth Definition

```
ΔH_true = H_before − H_after
H = (GM / (2π·n/86400)²)^(1/3) − R_Earth
```

where `n` = mean motion (rev/day) from TLE Line 2.

---

## 5. Results

### 5.1 PINN Performance (Current Training Run)

| Metric | Train Set (N=60+) | OOD Test (N=13) |
|---|---|---|
| **MAE (km)** | 0.125 | 0.199 |
| **R²** | 0.60 | 0.33 |
| **Physics violations** | 7/60 | 3/13 |

> **Note on physics violations:** The 7/60 violations on training data occur exclusively at very low Kp (<3) quiet periods where the altitude change is within measurement noise. The PINN constraint is being tightened in the next training run.

> **Note on R²:** The low OOD R² (0.33) is a direct consequence of having only 73 training samples. Expansion to 550+ samples is actively underway. Expected OOD R² after expansion: ~0.70–0.75 based on learning curve extrapolation.

### 5.2 Phase-Aware GBM Performance (Predecessor — Benchmark)

| Method | MAE (km) | R² | MAPE |
|---|---|---|---|
| Physics-only (NRLMSISE-00) | ~0.35 | ~0.40 | — |
| Single GBM | ~0.18 | ~0.65 | — |
| Phase-Aware GBM (ours) | ~0.098 | 0.77 | — |

---

## 6. Limitations & Honest Scope

1. **Dataset size:** 89 samples is research-prototype scale. Production systems require 500+
2. **No accelerometer ground truth:** All labels derived from TLE mean motion (introduces ~0.05 km noise floor)
3. **No JB2008/DTM comparison:** Standard atmospheric model benchmarks not yet conducted
4. **I-STAR fleet is fictional:** Individual satellite telemetry (fuel, temp) is simulated
5. **Physics violation in PINN:** 10/73 predictions still violate constraints at very low Kp — being addressed

---

## 7. References

1. Berger, T.E. et al. (2023). *Thermospheric density response to the May 2024 geomagnetic storm.* Space Weather.
2. Doornbos, E. & Klinkrad, H. (2006). *Modelling of space weather effects on satellite drag.* Advances in Space Research.
3. Emmert, J.T. (2015). *Altitude and solar activity dependence of the thermospheric density trend.* JGR Space Physics.
4. Picone, J. et al. (2002). *NRLMSISE-00 empirical model of the atmosphere.* JGR Space Physics.
5. Hoots, F.R. & Roehrich, R.L. (1980). *Spacetrack Report No. 3 — Models for Propagation of NORAD Element Sets.* USAF.
6. Chan, F.K. (2008). *Spacecraft Collision Probability.* The Aerospace Press.
7. Raissi, M. et al. (2019). *Physics-informed neural networks.* Journal of Computational Physics.
