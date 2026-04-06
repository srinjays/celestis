# CELESTIS — Setup & Installation Guide

Complete guide to get CELESTIS running from scratch on your machine.

---

## Prerequisites

Before you start, make sure you have the following installed:

| Tool | Version | Download |
|---|---|---|
| Python | 3.11+ | https://python.org/downloads |
| Node.js | 18+ | https://nodejs.org |
| Git | Any | https://git-scm.com |

> **Check your versions:**
> ```bash
> python --version    # should show 3.11+
> node --version      # should show 18+
> git --version
> ```

---

## Step 1 — Clone the Repository

```bash
git clone https://github.com/srinjays/FAHH.git
cd FAHH
```

---

## Step 2 — Set Up the Python Backend

### 2a. Create a virtual environment (recommended)

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS / Linux
python -m venv .venv
source .venv/bin/activate
```

### 2b. Install Python dependencies

```bash
cd backend
pip install -r requirements.txt
```

This installs:
- `fastapi` — Web framework for the API server
- `uvicorn` — ASGI server to run FastAPI
- `torch` — PyTorch for the PINN neural network
- `scikit-learn` — Phase-Aware GBM model
- `sgp4` — Orbital propagation for debris tracking
- `pymsis` — NRLMSISE-00 atmospheric density model
- `numpy`, `requests`, `httpx` — Core utilities

> ⚠️ **PyTorch note:** The above installs CPU-only PyTorch. For GPU support, visit https://pytorch.org/get-started and install the CUDA version matching your GPU.

### 2c. Set up environment variables

Create a file called `.env` inside the `backend/` folder:

```
backend/.env
```

Add the following:

```env
SPACETRACK_USER=your_spacetrack_email
SPACETRACK_PASS=your_spacetrack_password
```

> You need a free account at **https://www.space-track.org** to fetch live debris TLE data. Registration is free and instant.

### 2d. Start the backend server

```bash
# From inside the backend/ folder
uvicorn api:app --host 0.0.0.0 --port 4000 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:4000
INFO:     Application startup complete.
```

> The backend must be running on port **4000** before you launch the dashboard.

---

## Step 3 — Set Up the Data Pipeline (Optional — for AI model training)

> Skip this step if you just want to run the dashboard with the pre-trained model. The backend loads the model automatically.

### 3a. Go to the data pipeline folder

```bash
cd data_pipeline
```

### 3b. Build the ground-truth dataset

```bash
python build_dataset.py
```

This fetches real TLEs from Space-Track.org and pairs them with NOAA Kp/F10.7 solar data to build `output/ground_truth_dataset.csv`.

> ⚠️ Requires the `.env` credentials from Step 2c. Takes ~5–10 minutes depending on your connection.

### 3c. Train the Phase-Aware GBM model

```bash
python benchmark.py
```

Trains 3 storm-phase-specific GBM models and saves results to:
- `output/benchmark_results.csv` — accuracy table
- `output/crossval_results.json` — 5-fold CV results

### 3d. Train the PINN model

```bash
python train_pinn.py
```

Trains the Physics-Informed Neural Network with physics-constrained loss function.

### 3e. View results

```bash
python show_results.py
```

Prints the full benchmark comparison table to the console.

---

## Step 4 — Set Up the Mission Control Dashboard

### 4a. Go back to the root folder

```bash
cd ..   # from backend/ or data_pipeline/
```

### 4b. Install Node.js dependencies

```bash
npm install
```

This installs:
- `electron` — Desktop app framework
- `three` — Three.js for the 3D orbital globe
- `electron-builder` — For packaging into a .exe (optional)

### 4c. Launch the dashboard

```bash
npm start
```

The CELESTIS God-View mission control dashboard will open as a native desktop application.

> ✅ Make sure the FastAPI backend (Step 2d) is running before launching the dashboard.

---

## Full Startup Sequence (Quick Reference)

Open **two terminals** and run:

**Terminal 1 — Backend:**
```bash
cd FAHH/backend
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
uvicorn api:app --host 0.0.0.0 --port 4000
```

**Terminal 2 — Dashboard:**
```bash
cd FAHH
npm install
npm start
```

---

## Project Structure

```
FAHH/
├── backend/
│   ├── api.py               # FastAPI server (port 4000)
│   ├── cola_engine.py       # Debris collision avoidance engine
│   ├── requirements.txt     # Python dependencies
│   └── .env                 # Your Space-Track credentials (create this)
│
├── data_pipeline/
│   ├── build_dataset.py     # Builds ground-truth training dataset
│   ├── train_pinn.py        # Trains the PINN model
│   ├── benchmark.py         # Trains Phase-Aware GBM + benchmarks
│   ├── show_results.py      # Prints accuracy table
│   └── output/              # CSV results + trained models
│
├── electron/
│   └── main.js              # Electron app entry point
│
├── app.js                   # 4-quadrant dashboard controller
├── globe.js                 # Three.js 3D orbital visualization
├── index.html               # Dashboard layout
├── style.css                # Mission control dark UI
├── dataset.js               # Fleet + debris data handler
├── istar_config.js          # I-STAR satellite fleet config
├── istar_fleet_config.json  # 36-satellite fleet parameters
└── package.json             # Node.js config
```

---

## Troubleshooting

**Dashboard shows "Backend Offline"**
→ Make sure `uvicorn api:app --port 4000` is running in a separate terminal.

**`pymsis` install fails**
→ Run: `pip install pymsis --no-build-isolation`

**Space-Track fetch fails / 401 error**
→ Check your `.env` credentials. Make sure your Space-Track account is active at https://www.space-track.org

**Electron window doesn't open**
→ Run `npm install` again, then `npm start`. Make sure Node.js 18+ is installed.

**`torch` install is slow**
→ Normal — PyTorch is a large package (~800MB). Let it finish.

**Port 4000 already in use**
→ Change the port in `uvicorn api:app --port 5000` and update the API URL in `app.js` accordingly.

---

## System Requirements

| Component | Minimum | Recommended |
|---|---|---|
| OS | Windows 10 / macOS 12 / Ubuntu 20 | Windows 11 / macOS 14 |
| RAM | 8 GB | 16 GB |
| Storage | 2 GB free | 5 GB free |
| GPU | Not required | CUDA-capable (for faster PINN training) |
| Internet | Required (for live feeds) | Stable broadband |

---

*Built by Inceptrix SAT — Ajinkya Singh & Srinjay | April 2025*
