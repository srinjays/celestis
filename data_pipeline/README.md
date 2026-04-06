# CELESTIS Data Pipeline — README
## v3.0 · March 2026 · Inceptrix SAT

---

## Overview

The CELESTIS data pipeline builds the ground-truth dataset used to train the Physics-Informed Neural Network (PINN) for atmospheric drag residual correction. It fetches real TLE data from Space-Track.org, computes actual satellite altitude drops during geomagnetic storms, and pairs them with real space weather indices from NASA OMNI2.

---

## Prerequisites

1. **Space-Track.org account** — Free at https://www.space-track.org  
   Add credentials to `backend/.env`:
   ```
   SPACETRACK_USER=your@email.com
   SPACETRACK_PASS=yourpassword
   ```

2. **Python dependencies:**
   ```bash
   pip install torch scikit-learn numpy requests python-dotenv sgp4
   ```

---

## Step 1 — Build the Ground Truth Dataset

```bash
cd d:\inceptrix_sat
python data_pipeline/build_dataset.py
```

**What it does:**
- Fetches TLEs for **22 satellites** across **26 storm events** (2003–2024)
- Computes true altitude drop from TLE mean motion (Kepler's third law)
- Fetches real Kp/F10.7 from NASA OMNI2
- **Incremental writes** — crash-safe, resumes from where it stopped
- **Expected output:** `output/ground_truth_dataset.csv` (~550 rows when complete)
- **Space-Track quota:** 300 req/hr free. Re-run after 45 min if quota hit.

---

## Step 2 — Train the PINN

```bash
python data_pipeline/train_pinn.py
```

**What it does:**
- Trains a 3-layer PyTorch PINN [64→128→64, GELU] with physics-constrained loss
- 600 epochs, alpha annealing (0.5→0.85), AdamW + cosine LR schedule
- **Outputs:**
  - `output/pinn_model.pt` — PyTorch state dict
  - `output/pinn_scaler.pkl` — StandardScaler for feature normalisation
  - `output/pinn_benchmark.json` — accuracy report
- **Training time:** ~3 min CPU, ~25 sec GPU

---

## Step 3 — Run the GBM Benchmark (Optional)

```bash
python data_pipeline/benchmark.py
```

Trains the Phase-Aware GBM (predecessor model) and produces the benchmark comparison table. Output: `output/benchmark_results.csv`

---

## Step 4 — Start the Backend (PINN Auto-Loads)

```bash
cd d:\inceptrix_sat\backend
python -m uvicorn api:app --reload --port 4000
```

The API auto-detects `pinn_model.pt` on startup and switches predictions to PINN mode.  
Check: `http://localhost:4000/api/pinn/status`

---

## Dataset Composition (Current)

| Item | Value |
|---|---|
| Satellites | 22 (ISS, GRACE-A/FO, Swarm A/B/C, Starlink ×3, Landsat-9, ICESat-2, CryoSat-2, Sentinel-2A/2B/6A, IXPE, SWOT, EarthCARE, CHAMP, TerraSAR-X, TanDEM-X) |
| Storm events | 26 (2003–2024, G0 through G5) |
| Altitude range | 350–890 km |
| Current rows | 89+ (expanding to ~550) |
| OOD held-out | Gannon 2024, Starlink 2022, Halloween 2003 A |
| Source | Space-Track.org + NASA OMNI2 |

---

## Ground Truth Definition

```
ΔH_true (km) = H_before − H_after

where:
  H = (GM / (2π·n/86400)²)^(1/3) − R_Earth
  n = mean motion (rev/day) from TLE Line 2
  GM = 398600.4418 km³/s²
  R_Earth = 6371.0 km
```

This is the standard definition from Doornbos & Klinkrad (2006), reproducible from any NORAD TLE archive.

---

## Novel Scientific Contributions

### 1. Physics-Informed Neural Network (PINN)
The loss function encodes orbital physics — the network **cannot predict physically impossible orbits** (altitude gain during storm). This is the key property that makes CELESTIS trustworthy for aerospace applications.

### 2. Phase-Aware Storm Decomposition
Three separate residual models (sudden commencement / main phase / recovery phase) exploit the physically different thermospheric dynamics in each storm phase — novel in lightweight real-time drag prediction systems.

### 3. Real-World COLA
Chan's Pc formula applied to live Space-Track debris TLEs propagated via SGP4 for 24 hours — same mathematical framework as US Space Command (18 SDS).

---

## OOD Test Events (Never Used in Training)

| Event | Date | Kp | Why Held Out |
|---|---|---|---|
| Halloween 2003 A | 2003-10-28 | 9.0 | Extreme G5, earliest in catalogue |
| Starlink 2022 | 2022-02-03 | 5.5 | Density pre-heating outlier |
| Gannon 2024 | 2024-05-10 | 9.0 | Most recent G5, future-unseen |
