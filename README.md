# CELESTIS — Autonomous Satellite Mission Control Platform

> **Physics-Informed AI for real-time orbital decay prediction, collision avoidance, and autonomous maneuver planning for ISRO's I-STAR constellation.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2-red?logo=pytorch)
![Electron](https://img.shields.io/badge/Electron-29-cyan?logo=electron)
![License](https://img.shields.io/badge/license-MIT-yellow)

---

## The Problem

During geomagnetic storms, the upper atmosphere expands dramatically — dragging satellites out of orbit faster than ground teams can respond. The global standard model (NRLMSISE-00) underestimates this drag by **30–80%** during peak storm phases.

In 2022, SpaceX lost 38 Starlink satellites in a single storm. India's upcoming I-STAR constellation of 36 LEO satellites faces the same threat.

---

## What CELESTIS Does

CELESTIS is a full-stack autonomous mission control platform that:

- 🌞 **Watches the Sun 24/7** — Fuses 5 live government feeds (NOAA SWPC, NASA DONKI, GOES, DSCOVR, Space-Track) every 55 seconds
- 🧠 **Predicts Orbital Decay** — Physics-Informed Neural Network (PINN) corrects NRLMSISE-00 with a physics-constrained loss that cannot hallucinate impossible orbits
- 💥 **Detects Debris Collisions** — COLA engine scans 200+ real debris objects every 15 min using SGP4 + Chan's Pc formula
- 🚀 **Plans & Executes Maneuvers** — Auto-calculates Delta-V per satellite; supports Human-on-the-Loop or Full Autonomy
- 🖥️ **God-View Dashboard** — Electron 4-quadrant mission control with Three.js 3D orbital globe (36 satellites + 34,000 debris)

---

## Results

| Method | MAE (km) | R² | vs Physics |
|---|---|---|---|
| Physics-only (NRLMSISE-00) | 0.234 | — | baseline |
| Single GBM | 0.181 | 0.61 | +22.7% |
| **Phase-Aware GBM — ours** | **0.098** | **0.77** | **+58.3%** |

> Tested on **3 storms the model never saw**: Gannon 2024 (strongest in 21 years), Starlink 2022, Halloween 2003.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  CELESTIS v3 SYSTEM                     │
├──────────────┬──────────────────────────────────────────┤
│ 📡 DATA      │ NOAA SWPC · NASA DONKI · GOES · DSCOVR  │
│ LAYER        │ Space-Track.org — refresh every 55s      │
├──────────────┼──────────────────────────────────────────┤
│ ⚙️ PHYSICS   │ NRLMSISE-00 → PINN [64→128→64]          │
│ + AI LAYER   │ SGP4 → Chan's Pc (COLA)                 │
│ (FastAPI)    │ Tsiolkovsky Delta-V Calculator           │
├──────────────┼──────────────────────────────────────────┤
│ 🖥️ MISSION   │ 4-Quadrant God-View Dashboard           │
│ CONTROL UI   │ Three.js 3D Orbital Globe               │
│ (Electron)   │ Human-on-the-Loop / Full Autonomy        │
└──────────────┴──────────────────────────────────────────┘
```

---

## Project Structure

```
celestis/
├── data_pipeline/          # Data acquisition + AI training
│   ├── space_track_client.py   # Space-Track.org TLE fetcher
│   ├── kp_client.py            # NOAA Kp/F10.7 fetcher  
│   ├── build_dataset.py        # Ground-truth dataset builder
│   ├── train_pinn.py           # Physics-Informed Neural Network
│   ├── benchmark.py            # Phase-Aware GBM benchmark
│   └── output/                 # Results: CSVs + model weights
├── backend/                # FastAPI inference server
│   ├── api.py                  # REST endpoints (port 4000)
│   └── cola_engine.py          # Collision avoidance logic
├── electron/               # Desktop app shell
│   └── main.js                 # Electron entry point
├── app.js                  # 4-quadrant dashboard controller
├── globe.js                # Three.js 3D orbital visualization
├── style.css               # Mission control dark UI
└── index.html              # Dashboard HTML
```

---

## Quickstart

### 1. Backend (FastAPI)
```bash
cd backend
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 4000 --reload
```

### 2. Data Pipeline & Model Training
```bash
cd data_pipeline
pip install -r ../backend/requirements.txt
python build_dataset.py    # Build ground-truth dataset
python benchmark.py        # Train Phase-Aware GBM
python train_pinn.py       # Train PINN model
```

### 3. Mission Control Dashboard
```bash
npm install
npm start                  # Launches Electron app
```

> Requires the FastAPI backend running on port 4000.

---

## Fleet Configuration

CELESTIS manages **36 I-STAR satellites** across 3 orbital planes:
- **Plane A** (12 sats) — 500km SSO, 97.4° inclination
- **Plane B** (12 sats) — 550km SSO, 97.6° inclination  
- **Plane C** (12 sats) — 600km SSO, 97.8° inclination

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Model | PyTorch (PINN) + scikit-learn (Phase-Aware GBM) |
| Physics | NRLMSISE-00, SGP4, Chan's Pc, Tsiolkovsky |
| Backend | FastAPI (async), Python 3.11 |
| Frontend | Electron 29, Three.js, vanilla JS |
| Data | Space-Track.org, NOAA SWPC, NASA DONKI, GOES, DSCOVR |

---

## Research

This project introduces two novel contributions:

1. **Storm-Phase-Aware Residual Decomposition** — Separate GBM models for onset, main phase, and recovery — exploiting physically distinct atmospheric dynamics in each phase
2. **Physics-Constrained PINN Loss** — Neural network architecturally prevented from predicting thermodynamically impossible orbital states

See [`Celestis_Research_Paper.md`](Celestis_Research_Paper.md) and [`SOLARSHEILD_TECHNICAL_REFERENCE.md`](SOLARSHEILD_TECHNICAL_REFERENCE.md) for full details.

---

## Built By

**Inceptrix SAT Team** — Ajinkya Singh & Srinjay  
*Built for ISRO's I-STAR constellation | April 2025*
