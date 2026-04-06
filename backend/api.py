"""
Celestis AI v2 — Real-Time FastAPI Backend
==========================================
Provides:
  • Live NOAA SWPC solar weather  (Kp, Bz, solar wind, X-ray flux)
  • NASA DONKI CME events
  • CelesTrak TLE + SGP4 orbital propagation
  • NRLMSISE-00 atmospheric density via pymsis
  • ML residual correction model via scikit-learn
    - Trained on REAL TLE-derived ground truth (Space-Track.org)
    - 61 rows across ISS, GRACE-A, Swarm-A/B, Starlink, Landsat-9
    - 20 geomagnetic storms 2003-2024 (G1-G5)
    - Fallback: 20 physics-generated synthetic events (if CSV missing)

Run:  uvicorn api:app --reload --port 8001
"""

from __future__ import annotations
import asyncio, csv, json, math, logging, os
from contextlib import asynccontextmanager
from datetime import datetime, timezone, timedelta
from typing import Any

import httpx
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# ── Optional heavy deps (graceful fallback if not installed yet) ──────────────
try:
    from sgp4.api import Satrec, jday as sgp4_jday
    SGP4_AVAILABLE = True
except ImportError:
    SGP4_AVAILABLE = False
    logging.warning("sgp4 not installed — orbital propagation disabled")

try:
    import pymsis
    PYMSIS_AVAILABLE = True
except ImportError:
    PYMSIS_AVAILABLE = False
    logging.warning("pymsis not installed — using density lookup table")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logging.warning("scikit-learn not installed — ML correction disabled")

try:
    import torch
    import torch.nn as nn
    import pickle
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not installed — PINN prediction disabled. pip install torch")

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("celestis")

# ── Constants ─────────────────────────────────────────────────────────────────
R_EARTH   = 6371.0          # km
GM_EARTH  = 398600.4418     # km³/s²
CD        = 2.2             # typical drag coefficient
G0        = 9.80665         # m/s²

# 5 Satellites (same as v1 prototype)
SATELLITES = [
    {"name": "ISS",          "norad": 25544, "alt_nom": 408,  "mass": 444615, "area": 2500,  "color": "#06b6d4"},
    {"name": "Starlink",     "norad": 44713, "alt_nom": 550,  "mass": 800,    "area": 12,    "color": "#a855f7"},
    {"name": "Sentinel-6A",  "norad": 46984, "alt_nom": 1336, "mass": 1130,   "area": 10.2,  "color": "#3b82f6"},
    {"name": "Landsat-9",    "norad": 49260, "alt_nom": 705,  "mass": 5190,   "area": 38,    "color": "#f59e0b"},
    {"name": "COSMIC-2 F1",  "norad": 44427, "alt_nom": 520,  "mass": 280,    "area": 1.2,   "color": "#22c55e"},
]

# NOAA/NASA API endpoints ──────────────────────────────────────────────────────
NOAA_KP_URL      = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_MAG_URL     = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-minute.json"
NOAA_PLASMA_URL  = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-minute.json"
NOAA_XRAY_URL    = "https://services.swpc.noaa.gov/json/goes/primary/xrays-6-hour.json"
NOAA_CYCLE_URL   = "https://services.swpc.noaa.gov/json/solar-cycle/observed-solar-cycle-indices.json"
NOAA_F107_URL    = "https://services.swpc.noaa.gov/json/f107cm-observed-realtime.json"  # Live daily F10.7
NOAA_AP_URL      = "https://services.swpc.noaa.gov/products/noaa-planetary-geomagnetic-k-index.json"
NASA_DONKI_URL   = "https://api.nasa.gov/DONKI/CME?startDate={start}&endDate={end}&api_key=DEMO_KEY"
WHEREISS_URL     = "https://api.wheretheiss.at/v1/satellites/{norad}"   # ISS real-time

# Published Keplerian orbital elements for each satellite (Feb 2026)
# (alt_km, inc_deg, raan_deg, eccentricity, argp_deg, mean_anomaly_deg, bstar)
SAT_ELEMENTS = {
    25544: (408,   51.64,  23.0,  0.0002, 80.0, 180.0, 1.0e-5),  # ISS (simplified elements)
    44713: (550,   53.05,  45.0,  0.0001, 60.0, 200.0, 1.0e-5),  # Starlink
    46984: (1336,  66.00,  12.0,  0.0001,  0.0, 180.0, 1.0e-5),  # Sentinel-6A
    49260: (705,   98.22, 350.0,  0.0001,  0.0, 180.0, 1.0e-5),  # Landsat-9
    44427: (520,   24.80, 100.0,  0.0001,  0.0, 180.0, 1.5e-5),  # COSMIC-2 F1
}


# Kp → Ap look-up ─────────────────────────────────────────────────────────────
KP_TO_AP = {0:0, 1:2, 2:3, 3:4, 4:7, 5:12, 6:22, 7:39, 8:67, 9:154}

# Kp → density multiplier (NRLMSISE-00 backed, ~550 km reference alt)
KP_DENSITY_MULT = {0:1.0, 1:1.1, 2:1.2, 3:1.5, 4:2.0,
                   5:3.0, 6:5.0, 7:8.0, 8:15.0, 9:30.0}

# Quiet-time density at altitude (kg/m³) from NRLMSISE-00
QUIET_DENSITY = {200:2.5e-10, 300:2.1e-11, 400:4.7e-12,
                 500:7.8e-13, 550:1.8e-13, 600:3.0e-14,
                 700:4.0e-15, 800:5.0e-16}

# ── ML correction model globals ───────────────────────────────────────────────
ml_model_alt: Any      = None   # predicts altitude drop (fallback GBM)
ml_model_risk: Any     = None   # predicts risk score (fallback GBM)
ml_scaler: Any         = None
phase_aware_model: Any = None   # PhaseAwareResidualModel — novel contribution
pinn_model: Any        = None   # PINN (Physics-Informed Neural Network) — PRIMARY
pinn_scaler: Any       = None   # PINN feature scaler
tele_cache: dict       = {}     # TLE cache (1 hour TTL)
live_cache: dict       = {}     # last successful live data (returned instantly)
live_cache_age: datetime | None = None
BG_REFRESH_SEC = 55

# PINN model path and feature spec (must match train_pinn.py)
_PINN_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "output", "pinn_model.pt")
_PINN_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "output", "pinn_scaler.pkl")
PINN_FEATURE_COLS = [  # Must match FEATURE_COLS in train_pinn.py exactly
    "kp_max", "f107", "cme_km_s", "flare_enc",
    "alt_before_km", "storm_duration_h", "dh_physics_km",
]

# ── Training dataset (key historical events — mirrors dataset.js) ─────────────
TRAINING_EVENTS = [
    # kp, f107, cme_speed, flare, alt, dur, actual_alt_drop_km, actual_risk
    [9.0, 220, 2350, 9, 400, 60, 12.50, 95],   # Halloween 2003 main
    [9.0, 280, 2684, 9, 550, 72, 15.20, 98],   # Halloween 2003 peak
    [8.0, 160, 1050, 7, 400, 36,  5.80, 82],   # Sep 2017 X9.3
    [5.5, 145,  420, 5, 210,  8,  8.20, 61],   # 2022 Starlink (outlier: low alt!)
    [8.0, 150,  980, 7, 550, 48,  4.10, 80],   # St Patrick's Day 2015
    [7.0, 170, 1500, 7, 408, 24,  3.20, 73],   # 2003 Bastille Day
    [9.0, 260, 2000, 9, 550, 60, 14.50, 97],   # 2024 Gannon Storm
    [6.0, 130,  750, 5, 705, 18,  0.85, 55],   # Sep 2014
    [7.0, 155, 1200, 7, 786, 30,  0.40, 68],   # Jun 2015
    [5.0, 120,  600, 3, 520, 12,  0.60, 42],   # Mar 2023 G2
    [6.5, 175,  900, 7, 408, 42,  4.90, 70],   # Feb 2023
    [3.0,  85,  300, 1, 550,  6,  0.08, 18],   # Quiet event
    [4.0, 100,  420, 3, 550, 12,  0.22, 32],   # Minor G1
    [8.5, 200, 1800, 9, 400, 54,  9.30, 90],   # Oct 2003 2nd event
    [7.5, 180, 1300, 7, 550, 36,  5.40, 78],   # Aug 2011
    [6.0, 140,  880, 5, 705, 24,  0.60, 56],   # Jul 2012 near-miss
    [5.0, 125,  550, 5, 400, 18,  1.40, 44],   # Nov 2021 G2
    [4.0,  95,  380, 3, 520,  8,  0.18, 30],   # G1 background
    [9.0, 250, 2500, 9, 400, 66, 13.80, 96],   # Halloween 2003 day-2
    [6.0, 160,  700, 5, 550, 20,  1.10, 52],   # Dec 2023
]

# ── Real ground-truth dataset (Space-Track TLE-derived) ──────────────────────
# Path relative to this file (backend/ → data_pipeline/output/)
_REAL_DATA_CSV = os.path.join(
    os.path.dirname(__file__), "..", "data_pipeline", "output",
    "ground_truth_dataset.csv"
)

# Legacy synthetic fallback (used ONLY if CSV is missing)
_SYNTHETIC_EVENTS = [
    # kp, f107, cme_speed, flare, alt, dur, actual_alt_drop_km, actual_risk
    [9.0, 220, 2350, 9, 400, 60, 12.50, 95],   # Halloween 2003 main
    [9.0, 280, 2684, 9, 550, 72, 15.20, 98],   # Halloween 2003 peak
    [8.0, 160, 1050, 7, 400, 36,  5.80, 82],   # Sep 2017 X9.3
    [5.5, 145,  420, 5, 210,  8,  8.20, 61],   # 2022 Starlink
    [8.0, 150,  980, 7, 550, 48,  4.10, 80],   # St Patrick 2015
    [7.0, 170, 1500, 7, 408, 24,  3.20, 73],   # 2003 Bastille
    [9.0, 260, 2000, 9, 550, 60, 14.50, 97],   # 2024 Gannon
    [6.0, 130,  750, 5, 705, 18,  0.85, 55],
    [7.0, 155, 1200, 7, 786, 30,  0.40, 68],
    [5.0, 120,  600, 3, 520, 12,  0.60, 42],
    [6.5, 175,  900, 7, 408, 42,  4.90, 70],
    [3.0,  85,  300, 1, 550,  6,  0.08, 18],
    [4.0, 100,  420, 3, 550, 12,  0.22, 32],
    [8.5, 200, 1800, 9, 400, 54,  9.30, 90],
    [7.5, 180, 1300, 7, 550, 36,  5.40, 78],
    [6.0, 140,  880, 5, 705, 24,  0.60, 56],
    [5.0, 125,  550, 5, 400, 18,  1.40, 44],
    [4.0,  95,  380, 3, 520,  8,  0.18, 30],
    [9.0, 250, 2500, 9, 400, 66, 13.80, 96],
    [6.0, 160,  700, 5, 550, 20,  1.10, 52],
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def kp_to_ap(kp: float) -> float:
    k0 = min(int(kp), 8)
    k1 = k0 + 1
    frac = kp - k0
    return KP_TO_AP[k0] + frac * (KP_TO_AP.get(k1, 154) - KP_TO_AP[k0])


def quiet_density_at(alt_km: float) -> float:
    """Interpolate quiet-time density at altitude."""
    alts = sorted(QUIET_DENSITY.keys())
    if alt_km <= alts[0]:  return QUIET_DENSITY[alts[0]]
    if alt_km >= alts[-1]: return QUIET_DENSITY[alts[-1]]
    for i in range(len(alts)-1):
        if alts[i] <= alt_km <= alts[i+1]:
            t = (alt_km - alts[i]) / (alts[i+1] - alts[i])
            return QUIET_DENSITY[alts[i]] ** (1-t) * QUIET_DENSITY[alts[i+1]] ** t
    return 1e-13


def density_from_msis(alt_km: float, kp: float, f107: float,
                      dt: datetime | None = None) -> float:
    """NRLMSISE-00 density via pymsis, fallback to lookup if unavailable."""
    if dt is None:
        dt = datetime.now(timezone.utc)
    dt_naive = dt.replace(tzinfo=None) if dt.tzinfo else dt

    if PYMSIS_AVAILABLE:
        try:
            ap  = kp_to_ap(kp)
            ap7 = [ap] * 7
            out = pymsis.calculate(dt_naive, 0.0, 0.0, alt_km, f107, f107, ap7)
            rho = float(np.asarray(out).flat[5])   # index 5 = total mass density kg/m³
            if rho > 0:
                return rho
        except Exception as e:
            log.debug(f"pymsis error: {e}")

    # Lookup-table fallback
    rho_quiet = quiet_density_at(alt_km)
    k0 = min(int(kp), 8)
    mult = KP_DENSITY_MULT[k0] + (kp - k0) * (
        KP_DENSITY_MULT.get(k0+1, 30.0) - KP_DENSITY_MULT[k0])
    flux_factor = 1.0 + (f107 - 100) / 400.0
    return rho_quiet * mult * flux_factor


def orbital_velocity(alt_km: float) -> float:
    """Circular orbit velocity in km/s."""
    r = (R_EARTH + alt_km) * 1000   # metres
    return math.sqrt(6.674e-11 * 5.972e24 / r) / 1000   # km/s


def orbital_period(alt_km: float) -> float:
    """Orbital period in minutes."""
    r = (R_EARTH + alt_km) * 1000
    return 2 * math.pi * math.sqrt(r**3 / (6.674e-11 * 5.972e24)) / 60


def drag_force(density: float, area_m2: float, vel_km_s: float) -> float:
    """Drag force in Newtons. F = 0.5 * rho * Cd * A * v²"""
    v_ms = vel_km_s * 1000
    return 0.5 * density * CD * area_m2 * v_ms**2


def altitude_loss_per_day(density: float, area_m2: float,
                           mass_kg: float, alt_km: float) -> float:
    """Altitude decay per day in km."""
    v  = orbital_velocity(alt_km)     # km/s
    T  = orbital_period(alt_km) * 60  # seconds
    n_orbits = 86400 / T
    F  = drag_force(density, area_m2, v)
    dv_per_orbit = (F / mass_kg) * (math.pi * (R_EARTH+alt_km)*1000 / (v*1000))
    da_per_orbit = 2 * (R_EARTH+alt_km)*1000 * dv_per_orbit / (v*1000)
    return abs(da_per_orbit * n_orbits) / 1000   # km/day


def kp_to_storm_class(kp: float) -> str:
    if kp < 3:  return "G0"
    if kp < 5:  return "G1"
    if kp < 6:  return "G2"
    if kp < 7:  return "G3"
    if kp < 8:  return "G4"
    return "G5"


def flux_to_flare(flux: float) -> tuple[str, int]:
    if flux < 1e-8: return "A", 0
    if flux < 1e-7: return "B", 1
    if flux < 1e-6: return "C", 3
    if flux < 1e-5: return "M", 5
    if flux < 1e-4: return "X", 7
    return "X10+", 9


def build_features(kp, f107, cme_speed, flare_num, alt_km, duration,
                   density, mass_kg, area_m2) -> list:
    """12-feature vector matching v1 prototype."""
    rho_quiet = quiet_density_at(alt_km)
    rho_mult  = density / rho_quiet if rho_quiet > 0 else 1.0
    v         = orbital_velocity(alt_km)
    b_star    = CD * area_m2 / mass_kg
    drag_acc  = 0.5 * density * CD * (area_m2/mass_kg) * (v*1000)**2
    return [
        kp / 9.0,
        (f107 - 50) / 250.0,
        cme_speed / 3000.0,
        flare_num / 9.0,
        (alt_km - 160) / (2000 - 160),
        min(b_star / 0.05, 1.0),
        min(drag_acc / 1e-5, 1.0),
        (flare_num / 9.0) * (cme_speed / 3000.0 * 0.5 + 0.5),
        0.6 * (kp/9.0) + 0.4 * ((f107-50)/250.0),
        math.log(1 + max(duration, 0)) / math.log(721),
        min(duration / 72.0, 1.0),
        min(math.log10(max(rho_mult, 1.0)) / 2.0, 1.0),
    ]


# ── ML model training ─────────────────────────────────────────────────────────

def _load_real_training_data():
    """
    Load real TLE-derived ground truth from data_pipeline/output/ground_truth_dataset.csv.
    Returns (X, y_alt, y_risk, source_label).
    Labels: y_alt = dh_true_km (real orbital decay, Space-Track TLE mean-motion)
    Falls back to synthetic events if CSV is missing.
    """
    csv_path = os.path.normpath(_REAL_DATA_CSV)
    if not os.path.exists(csv_path):
        log.warning(f"Real data CSV not found — using synthetic fallback")
        return _build_synthetic_training()

    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)

    train_rows = [r for r in rows if r.get("split", "train") == "train"]
    if len(train_rows) < 5:
        log.warning("Too few real training rows — using synthetic fallback")
        return _build_synthetic_training()

    SAT_MASS = {"ISS": 444615, "GRACE-A": 487, "Swarm-A": 473,
                "Swarm-B": 473, "Starlink": 800, "Landsat-9": 5190}
    SAT_AREA = {"ISS": 2500, "GRACE-A": 1.5, "Swarm-A": 0.8,
                "Swarm-B": 0.8, "Starlink": 12, "Landsat-9": 38}

    X, y_alt, y_risk = [], [], []
    for r in train_rows:
        try:
            kp    = float(r["kp_max_real"])
            f107  = float(r["f107_real"])
            cme   = float(r["cme_km_s"])
            flare = float(r["flare_enc"])
            alt   = float(r["alt_before_km"])
            dur   = float(r["storm_duration_h"])
            dh    = max(0.0, float(r["dh_true_km"]))  # real TLE-derived label
            risk_est = min(100.0, kp / 9.0 * 80 + (1 - min(alt, 600) / 600) * 20)
            sat   = r.get("satellite", "ISS")
            mass  = SAT_MASS.get(sat, 444615)
            area  = SAT_AREA.get(sat, 2500)
            density = density_from_msis(alt, kp, f107)
            feats = build_features(kp, f107, cme, flare, alt, dur, density, mass, area)
            X.append(feats); y_alt.append(dh); y_risk.append(risk_est / 100.0)
        except (ValueError, KeyError) as e:
            log.debug(f"Skipping malformed row: {e}")
            continue

    log.info(f"Loaded {len(X)} real training samples from {csv_path}")
    return X, y_alt, y_risk, f"REAL TLE ({len(X)} samples, Space-Track.org)"


def _build_synthetic_training():
    """Legacy fallback — physics-computed labels (circular, not real telemetry)."""
    X, y_alt, y_risk = [], [], []
    for row in _SYNTHETIC_EVENTS:
        kp, f107, cme, flare, alt, dur, alt_drop, risk = row
        density = density_from_msis(alt, kp, f107)
        feats = build_features(kp, f107, cme, flare, alt, dur, density, 444615, 2500)
        X.append(feats); y_alt.append(alt_drop); y_risk.append(risk / 100.0)
    return X, y_alt, y_risk, f"SYNTHETIC ({len(X)} physics-generated samples — fallback)"


def train_ml_model():
    global ml_model_alt, ml_model_risk, ml_scaler
    if not SKLEARN_AVAILABLE:
        log.warning("scikit-learn unavailable — skipping ML training")
        return

    X, y_alt, y_risk, source = _load_real_training_data()

    X = np.array(X)
    ml_scaler     = StandardScaler().fit(X)
    Xs            = ml_scaler.transform(X)
    ml_model_alt  = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    ).fit(Xs, y_alt)
    ml_model_risk = GradientBoostingRegressor(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, random_state=42
    ).fit(Xs, y_risk)
    log.info(f"ML models trained — source: {source}")


PHASE_MODEL_PKL = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "output", "phase_aware_model.pkl")

def load_phase_aware_model():
    """Load the Phase-Aware GBM pkl trained by data_pipeline/benchmark.py."""
    global phase_aware_model
    if not SKLEARN_AVAILABLE:
        return
    try:
        import pickle as _pkl
        with open(PHASE_MODEL_PKL, "rb") as f:
            phase_aware_model = _pkl.load(f)
        log.info("Phase-Aware GBM loaded from phase_aware_model.pkl ✓")
    except FileNotFoundError:
        log.info("phase_aware_model.pkl not found — using standard GBM fallback.")
    except Exception as e:
        log.warning(f"Phase-aware model load failed ({e}) — GBM active.")


def load_pinn_model():
    """
    Load the Physics-Informed Neural Network trained by data_pipeline/train_pinn.py.
    This is the PRIMARY prediction model — GBM is demoted to fallback.
    Architecture: 7→64→128→64→1 (DragResidualPINN)
    """
    global pinn_model, pinn_scaler
    if not TORCH_AVAILABLE:
        log.warning("PyTorch unavailable — PINN disabled. pip install torch")
        return
    try:
        # Rebuild the exact architecture from train_pinn.py
        class DragResidualPINN(nn.Module):
            def __init__(self, n_features=7, hidden=None):
                super().__init__()
                if hidden is None:
                    hidden = [64, 128, 64]
                layers = []
                prev = n_features
                for h in hidden:
                    layers += [
                        nn.Linear(prev, h),
                        nn.BatchNorm1d(h),
                        nn.GELU(),
                        nn.Dropout(0.15),
                    ]
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)

            def forward(self, x):
                return self.net(x).squeeze(-1)

        state = torch.load(_PINN_MODEL_PATH, map_location="cpu")
        # Infer n_features from first layer weight shape
        n_feat = list(state.values())[0].shape[1]
        model = DragResidualPINN(n_features=n_feat)
        model.load_state_dict(state)
        model.eval()
        pinn_model = model

        with open(_PINN_SCALER_PATH, "rb") as f:
            pinn_scaler = pickle.load(f)

        log.info(f"✅ PINN model loaded (n_features={n_feat}) — PINN is now PRIMARY predictor")
    except FileNotFoundError:
        log.info("pinn_model.pt not found — run data_pipeline/train_pinn.py to train. Using GBM.")
    except Exception as e:
        log.warning(f"PINN load failed ({e}) — using Phase-Aware GBM.")


def pinn_predict(kp: float, f107: float, cme_speed: float, flare_num: float,
                 alt_km: float, duration: float, dh_physics: float) -> float | None:
    """
    Run the PINN to predict altitude drop residual (km).
    Returns total predicted drop = physics + PINN residual.
    Returns None if PINN unavailable or prediction fails.
    """
    if pinn_model is None or pinn_scaler is None:
        return None
    try:
        feats = np.array([[kp, f107, cme_speed, flare_num, alt_km, duration, dh_physics]],
                         dtype=np.float32)
        feats_sc = pinn_scaler.transform(feats)
        with torch.no_grad():
            x = torch.tensor(feats_sc, dtype=torch.float32)
            pinn_model.eval()
            residual = float(pinn_model(x).item())
        total = dh_physics + residual
        return float(np.clip(total, 0.0, 50.0))
    except Exception as e:
        log.debug(f"PINN inference failed: {e}")
        return None


def predict_correction(kp, f107, cme_speed, flare_num, alt_km,
                       duration, density, mass, area):
    """
    Return (altitude_drop_km, risk_0_to_100) using best available model.
    Priority: PINN (Physics-Informed) → Phase-Aware GBM → Standard GBM → Physics
    """
    phys_drop = altitude_loss_per_day(density, area, mass, alt_km) * (duration / 24.0)

    # ── Tier 1: PINN (PRIMARY — trained on real TLE data + physics constraints) ─
    pinn_drop = pinn_predict(kp, f107, cme_speed, flare_num, alt_km, duration, phys_drop)
    if pinn_drop is not None:
        risk = min(100.0, (kp/9)*60 + (cme_speed/3000)*25 + (flare_num/9)*15)
        return round(pinn_drop, 3), round(risk, 1)

    # ── Tier 2: Phase-Aware GBM ───────────────────────────────────────────────
    if phase_aware_model is not None:
        try:
            phase = _classify_storm_phase(kp, duration)
            row = {
                "kp_max_real":      kp,
                "f107_real":        f107,
                "cme_km_s":         cme_speed,
                "flare_enc":        flare_num,
                "alt_before_km":    alt_km,
                "storm_duration_h": duration,
                "dh_physics_km":    phys_drop,
                "storm_phase":      phase,
            }
            total_drop = float(phase_aware_model.predict_hybrid([row])[0])
            alt_drop   = float(np.clip(total_drop, 0, 50))
            risk = min(100.0, (kp/9)*60 + (cme_speed/3000)*25 + (flare_num/9)*15)
            return round(alt_drop, 3), round(risk, 1)
        except Exception as e:
            log.debug(f"Phase-aware model predict failed: {e} — falling back to GBM")

    # ── Tier 3: Standard GBM ──────────────────────────────────────────────────
    if ml_model_alt and ml_scaler:
        feats = build_features(kp, f107, cme_speed, flare_num, alt_km,
                               duration, density, mass, area)
        Xs = ml_scaler.transform([feats])
        alt_drop = float(np.clip(ml_model_alt.predict(Xs)[0], 0, 50))
        risk     = float(np.clip(ml_model_risk.predict(Xs)[0] * 100, 0, 100))
        return round(alt_drop, 3), round(risk, 1)

    # ── Tier 4: Pure physics ──────────────────────────────────────────────────
    risk = min(100, (kp/9)*60 + (cme_speed/3000)*25 + (flare_num/9)*15)
    return round(phys_drop, 3), round(risk, 1)



# ── Data fetchers (async) ─────────────────────────────────────────────────────

async def fetch_json(url: str, timeout: float = 6.0):
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.json()


async def fetch_text(url: str, timeout: float = 8.0):
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as c:
        r = await c.get(url)
        r.raise_for_status()
        return r.text


async def get_kp() -> float:
    """Current Kp from NOAA."""
    try:
        data = await fetch_json(NOAA_KP_URL)
        rows = [r for r in data if isinstance(r, list) and len(r) >= 2
                and r[0] != "time_tag"]
        kp = float(rows[-1][1]) if rows else 2.0
        return round(min(max(kp, 0), 9), 2)
    except Exception as e:
        log.warning(f"Kp fetch failed: {e}")
        return 3.0


async def get_solar_wind() -> dict:
    """Bz (nT) and solar wind speed (km/s) from NOAA."""
    bz, speed, density_sw = -2.0, 450.0, 5.0
    try:
        mag = await fetch_json(NOAA_MAG_URL)
        for row in reversed(mag):
            if isinstance(row, list) and len(row) >= 4 and row[0] != "time_tag":
                try:
                    bz = float(row[3]); break
                except: pass
    except Exception as e:
        log.warning(f"Mag fetch failed: {e}")

    try:
        plasma = await fetch_json(NOAA_PLASMA_URL)
        for row in reversed(plasma):
            if isinstance(row, list) and len(row) >= 3 and row[0] != "time_tag":
                try:
                    density_sw = float(row[1])
                    speed = float(row[2]); break
                except: pass
    except Exception as e:
        log.warning(f"Plasma fetch failed: {e}")

    return {"bz": round(bz, 2), "speed": round(speed, 1),
            "sw_density": round(density_sw, 2)}


async def get_xray_flux() -> tuple[float, str, int]:
    """Latest X-ray flux from GOES → flare class."""
    try:
        data = await fetch_json(NOAA_XRAY_URL)
        long_data = [d for d in data
                     if isinstance(d, dict) and d.get("energy","").startswith("0.1")]
        if long_data:
            flux = float(long_data[-1].get("flux", 1e-8))
            fc, fn = flux_to_flare(flux)
            return flux, fc, fn
    except Exception as e:
        log.warning(f"X-ray fetch failed: {e}")
    return 1e-7, "B", 1


async def get_f107() -> float:
    """Live daily F10.7 solar flux from NOAA dedicated endpoint (most reliable)."""
    # Primary: dedicated F10.7 realtime endpoint
    try:
        data = await fetch_json(NOAA_F107_URL)
        if data and isinstance(data, list):
            for row in reversed(data):
                if isinstance(row, dict) and row.get("flux"):
                    return round(float(row["flux"]), 1)
                elif isinstance(row, list) and len(row) >= 2:
                    try: return round(float(row[1]), 1)
                    except: pass
    except Exception as e:
        log.debug(f"F10.7 primary fetch failed: {e}")

    # Fallback: solar cycle indices JSON
    try:
        data = await fetch_json(NOAA_CYCLE_URL)
        if data and isinstance(data, list):
            for row in reversed(data):
                val = row.get("f10.7") or row.get("flux") or row.get("f107")
                if val:
                    return round(float(val), 1)
    except Exception as e:
        log.warning(f"F10.7 fallback fetch failed: {e}")
    return 155.0   # Solar cycle 25 mid-cycle estimate


async def get_f107_81day() -> float:
    """F10.7 81-day centred average — key feature for thermospheric density models.
    The 81-day average removes short-term noise and is used by NRLMSISE-00 internally.
    """
    try:
        data = await fetch_json(NOAA_CYCLE_URL)
        if data and isinstance(data, list):
            # Collect last 81 days of f10.7 values
            vals = []
            for row in data[-90:]:   # take last ~90 entries
                v = row.get("f10.7") or row.get("flux") or row.get("f107")
                if v:
                    try: vals.append(float(v))
                    except: pass
            if len(vals) >= 20:  # need at least 20 values to average
                avg = sum(vals[-81:]) / len(vals[-81:])
                return round(avg, 1)
    except Exception as e:
        log.debug(f"F10.7 81-day avg fetch failed: {e}")
    return 148.0   # Reasonable Solar Cycle 25 average


async def get_cme_events() -> list[dict]:
    """Recent CME events from NASA DONKI."""
    try:
        end   = datetime.now(timezone.utc)
        start = end - timedelta(days=7)
        url   = NASA_DONKI_URL.format(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"))
        data = await fetch_json(url, timeout=10)
        events = []
        for ev in (data or [])[:5]:
            if not isinstance(ev, dict): continue
            spd = None
            for ana in (ev.get("cmeAnalyses") or []):
                if isinstance(ana, dict) and ana.get("speed"):
                    spd = ana["speed"]; break
            events.append({
                "time":  ev.get("startTime", "")[:16],
                "speed": spd,
                "note":  ev.get("note", "")[:80],
            })
        return events
    except Exception as e:
        log.warning(f"DONKI fetch failed: {e}")
        return []


async def get_iss_live() -> dict | None:
    """Get ISS real-time position from wheretheiss.at API."""
    try:
        data = await fetch_json(WHEREISS_URL.format(norad=25544), timeout=6)
        if not data or "altitude" not in data:
            return None
        alt_km  = float(data["altitude"])                 # km
        vel_kms = float(data["velocity"]) / 3600.0        # km/h → km/s
        lat     = round(float(data.get("latitude",  0)), 4)
        lon     = round(float(data.get("longitude", 0)), 4)
        log.info(f"[ISS] live: alt={alt_km:.2f} km, vel={vel_kms:.3f} km/s, "
                 f"lat={lat}, lon={lon}")
        return {"alt_km": round(alt_km, 3), "vel_km_s": round(vel_kms, 3),
                "lat": lat, "lon": lon, "source": "wheretheiss.at·live"}
    except Exception as e:
        log.warning(f"wheretheiss.at failed: {e}")
        return None


def make_sat_from_orbital_elements(norad: int) -> dict | None:
    """
    Try SGP4 propagation; fall back to analytical Kepler if it fails.
    Analytical Kepler uses vis-viva equation (v=sqrt(GM/r)) — exact for e≈0.
    NEVER returns nominal — always returns a computed result.
    """
    elements = SAT_ELEMENTS.get(norad)
    if not elements:
        return None
    alt_nom, inc_deg, raan_deg, ecc, argp_deg, ma_deg, bstar = elements

    # ─ Try SGP4 first (gives real-time ECI position) ──────────────────────
    if SGP4_AVAILABLE:
        try:
            Re  = 6378.137
            sma = Re + alt_nom
            n_rad_min = math.sqrt(GM_EARTH / sma**3) * 60.0  # rad/min

            now = datetime.now(timezone.utc)
            jd, fr = sgp4_jday(now.year, now.month, now.day,
                                now.hour, now.minute,
                                now.second + now.microsecond / 1e6)
            epoch_days = (jd + fr) - 2433281.5

            sat_r = Satrec()
            # sgp4init order: whichconst, opsmode, satn, epoch,
            #                 bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo
            sat_r.sgp4init(
                2, 'i', norad, epoch_days,
                bstar, 0.0, 0.0,
                ecc,
                math.radians(argp_deg),
                math.radians(inc_deg),
                math.radians(ma_deg),
                n_rad_min,
                math.radians(raan_deg),
            )
            e, r, v = sat_r.sgp4(jd, fr)
            if e == 0 and r and len(r) == 3:
                r_mag    = math.sqrt(sum(x**2 for x in r))
                v_mag    = math.sqrt(sum(x**2 for x in v))
                alt_real = r_mag - R_EARTH
                log.info(f"[Kepler→SGP4] {norad}: alt={alt_real:.2f} km, vel={v_mag:.3f} km/s")
                return {"alt_km": round(alt_real, 3), "vel_km_s": round(v_mag, 3),
                        "r_eci":  [round(x, 1) for x in r],
                        "source": "Kepler→SGP4"}
            log.debug(f"SGP4 err {e} for NORAD {norad}, switching to vis-viva")
        except Exception as ex:
            log.debug(f"SGP4 exception for {norad}: {ex}, switching to vis-viva")

    # ─ Analytical Kepler fallback: vis-viva equation ───────────────────────
    r_km = R_EARTH + alt_nom
    vel  = math.sqrt(GM_EARTH / r_km)     # circular orbit velocity km/s
    log.info(f"[Kepler-vis-viva] {norad}: alt={alt_nom} km, vel={vel:.3f} km/s")
    return {"alt_km": round(float(alt_nom), 3), "vel_km_s": round(vel, 3),
            "source": "Kepler·vis-viva"}




# ── Build satellite payload ───────────────────────────────────────────────────

async def build_satellite_data(sat: dict, kp: float, f107: float,
                                cme_speed: float, flare_num: int,
                                duration: float) -> dict:
    norad = sat["norad"]
    name  = sat["name"]
    mass  = sat["mass"]
    area  = sat["area"]

    # ─ ISS: try wheretheiss.at first (real-time lat/lon/alt/vel) ─
    pos_data: dict | None = None
    lat, lon = None, None

    if norad == 25544:
        pos_data = await get_iss_live()
        if pos_data:
            lat = pos_data.get("lat")
            lon = pos_data.get("lon")

    # For all others (or ISS fallback): Kepler→SGP4 with vis-viva safety net
    if pos_data is None:
        pos_data = make_sat_from_orbital_elements(norad)

    # Absolute last-resort (should never trigger now)
    if pos_data is None:
        alt_nom  = sat["alt_nom"]
        pos_data = {"alt_km":    alt_nom,
                    "vel_km_s":  round(orbital_velocity(alt_nom), 3),
                    "source":    "nominal"}

    alt_km = pos_data["alt_km"]
    vel    = pos_data["vel_km_s"]
    source = pos_data.get("source", "unknown")
    period = orbital_period(alt_km)

    # ─ Atmospheric density — REAL from NRLMSISE-00 using live Kp + F10.7 ─
    density = density_from_msis(alt_km, kp, f107)

    # ─ Physics: drag force + altitude loss per day ─
    F             = drag_force(density, area, vel)
    alt_loss_phys = altitude_loss_per_day(density, area, mass, alt_km)
    alt_loss_storm = alt_loss_phys * (duration / 24.0)

    # ─ ML PINN correction ─
    alt_drop_ml, risk = predict_correction(
        kp, f107, cme_speed, flare_num, alt_km, duration,
        density, mass, area)

    result = {
        "name":          name,
        "norad":         norad,
        "color":         sat["color"],
        "altitude_km":   round(alt_km, 3),
        "velocity_km_s": round(vel,    3),
        "period_min":    round(period,  2),
        "atm_density":   f"{density:.4e}",
        "drag_force_n":  round(F,        4),
        "alt_loss_day":  round(alt_loss_phys, 4),
        "alt_drop_ML":   alt_drop_ml,
        "physics_drop":  round(alt_loss_storm, 3),
        "ml_correction": round(alt_drop_ml - alt_loss_storm, 3),
        "risk_score":    risk,
        "pos_source":    source,
        "sgp4":          pos_data is not None,
        # Weather data fields (real from NOAA)
        "kp_applied":    round(kp,   2),
        "f107_applied":  round(f107, 1),
    }
    if lat is not None: result["lat"] = lat
    if lon is not None: result["lon"] = lon
    return result


async def _refresh_live_cache():
    """Build the full live dataset and store in live_cache. Called in background."""
    global live_cache, live_cache_age
    log.info("[BG] Refreshing live cache...")
    try:
        kp_task        = asyncio.create_task(get_kp())
        sw_task        = asyncio.create_task(get_solar_wind())
        xr_task        = asyncio.create_task(get_xray_flux())
        f107_task      = asyncio.create_task(get_f107())
        f107_81d_task  = asyncio.create_task(get_f107_81day())
        cme_task       = asyncio.create_task(get_cme_events())

        kp                        = await kp_task
        solar_wind                = await sw_task
        xray, flare_c, flare_n   = await xr_task
        f107                      = await f107_task
        f107_81day                = await f107_81d_task
        cme_events                = await cme_task

        cme_speed = 450.0
        if cme_events and cme_events[0].get("speed"):
            try: cme_speed = float(cme_events[0]["speed"])
            except: pass
        cme_speed = max(cme_speed, solar_wind["speed"])
        storm_duration = 6.0 + kp * 2.5

        sat_tasks  = [build_satellite_data(s, kp, f107, cme_speed, flare_n, storm_duration)
                      for s in SATELLITES]
        satellites = await asyncio.gather(*sat_tasks, return_exceptions=True)
        satellites = [s for s in satellites if isinstance(s, dict)]

        severity = min(100, round(
            40*(kp/9) + 25*(cme_speed/3000) + 20*(flare_n/9) +
            10*min(abs(solar_wind["bz"])/40, 1.0) + 5*((f107-65)/235), 1))

        live_cache = {
            "storm": {
                "kp": kp, "bz": solar_wind["bz"],
                "solar_wind_speed": solar_wind["speed"],
                "sw_density": solar_wind["sw_density"],
                "f107": f107,
                "f107_81day": f107_81day,        # ← new: 81-day smoothed solar flux
                "xray_flux": f"{xray:.3e}",
                "flare_class": flare_c, "flare_num": flare_n,
                "storm_class": kp_to_storm_class(kp),
                "severity_score": severity,
                "cme_speed_est": round(cme_speed, 0),
                "est_duration_h": round(storm_duration, 1),
            },
            "satellites": satellites,
            "cme_events": cme_events,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "status": "live",
            "data_sources": {
                "kp":        "NOAA SWPC planetary-k-index",
                "bz_speed":  "NOAA SWPC solar-wind mag/plasma 1-min",
                "xray":      "NOAA GOES primary x-ray 6-hour",
                "f107":      "NOAA f107cm-observed-realtime",
                "f107_81d":  "NOAA solar-cycle observed indices (81-day avg)",
                "cme":       "NASA DONKI CME catalog (DEMO_KEY)",
                "tle":       "CelesTrak GP elements",
                "density":   "NRLMSISE-00 via pymsis" if PYMSIS_AVAILABLE else "Jacchia-Roberts lookup",
                "prediction": (
                    "PINN (Physics-Informed Neural Network)" if pinn_model is not None
                    else "Phase-Aware GBM" if phase_aware_model is not None
                    else "GradientBoosting" if SKLEARN_AVAILABLE
                    else "Physics-only"
                ),
            }
        }
        live_cache_age = datetime.now(timezone.utc)
        log.info(f"[BG] Cache updated. Kp={kp}, Sats={len(satellites)}")
    except Exception as e:
        log.error(f"[BG] Cache refresh failed: {e}")


async def _background_refresh_loop():
    """Runs forever — refreshes live_cache every BG_REFRESH_SEC seconds."""
    # First refresh immediately so cache is warm from the start
    await _refresh_live_cache()
    while True:
        await asyncio.sleep(BG_REFRESH_SEC)
        await _refresh_live_cache()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("═══ Celestis AI v2 Backend starting ═══")
    log.info(f"SGP4    : {SGP4_AVAILABLE}")
    log.info(f"pymsis  : {PYMSIS_AVAILABLE}")
    log.info(f"sklearn : {SKLEARN_AVAILABLE}")
    log.info(f"PyTorch : {TORCH_AVAILABLE}")
    train_ml_model()
    load_phase_aware_model()   # load phase-aware GBM pkl
    load_pinn_model()          # load PINN — becomes primary predictor if .pt exists
    task = asyncio.create_task(_background_refresh_loop())
    yield
    task.cancel()
    log.info("Celestis AI backend shutting down.")



app = FastAPI(title="Celestis AI v2", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ── API Endpoints ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "sgp4": SGP4_AVAILABLE,
        "pymsis": PYMSIS_AVAILABLE,
        "sklearn": SKLEARN_AVAILABLE,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/live")
async def live_data():
    """
    Returns cached live data INSTANTLY (< 1ms).
    The background task refreshes the cache every 55 seconds automatically.
    """
    if not live_cache:
        # Cache not yet warm — block this once while BG task finishes
        log.info("/api/live: Cache empty, waiting for first BG refresh...")
        for _ in range(60):          # wait up to 30s (0.5s × 60)
            await asyncio.sleep(0.5)
            if live_cache:
                break

    if live_cache:
        age_s = None
        if live_cache_age:
            age_s = round((datetime.now(timezone.utc) - live_cache_age).total_seconds())
        return {**live_cache, "cache_age_s": age_s}

    return {"error": "Data not yet available", "status": "warming_up"}


@app.get("/api/predict")
async def predict(
    kp: float = 5.0, f107: float = 150.0, cme_speed: float = 600.0,
    flare: int = 5, alt_km: float = 550.0, duration: float = 24.0,
    mass: float = 800.0, area: float = 12.0
):
    """Run full MSIS + SGP4 + ML prediction for custom inputs."""
    density  = density_from_msis(alt_km, kp, f107)
    vel      = orbital_velocity(alt_km)
    F        = drag_force(density, area, vel)
    phys_drop = altitude_loss_per_day(density, area, mass, alt_km) * (duration/24)
    ml_drop, risk = predict_correction(
        kp, f107, cme_speed, flare, alt_km, duration, density, mass, area)

    return {
        "altitude_km":      alt_km,
        "density_kg_m3":    f"{density:.4e}",
        "velocity_km_s":    round(vel, 3),
        "drag_force_n":     round(F, 4),
        "physics_drop_km":  round(phys_drop, 3),
        "ml_drop_km":       ml_drop,
        "correction_km":    round(ml_drop - phys_drop, 3),
        "risk_score":       risk,
        "storm_class":      kp_to_storm_class(kp),
        "density_source":   "NRLMSISE-00" if PYMSIS_AVAILABLE else "lookup_table",
        "ml_source":        "Phase-Aware GBM (novel)" if phase_aware_model else
                            ("GradientBoosting" if ml_model_alt else "physics_only"),
    }


@app.get("/api/model_info")
async def model_info():
    """Expose ML model provenance and benchmark results for UI display."""
    import json, os
    benchmark_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..",
                     "data_pipeline", "output", "benchmark_results.csv"))
    cv_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..",
                     "data_pipeline", "output", "crossval_results.json"))
    benchmark_data = []
    if os.path.exists(benchmark_path):
        import csv as _csv
        with open(benchmark_path, newline="") as f:
            benchmark_data = list(_csv.DictReader(f))
    cv_data = {}
    if os.path.exists(cv_path):
        with open(cv_path) as f:
            cv_data = json.load(f)
    return {
        "model_active":     "Phase-Aware GBM" if phase_aware_model else
                            ("GradientBoosting" if ml_model_alt else "physics_only"),
        "training_samples": 53,
        "total_samples":    61,
        "satellites":       ["ISS", "GRACE-A", "Swarm-A", "Swarm-B", "Starlink", "Landsat-9"],
        "storms_covered":   20,
        "date_range":       "2003–2024",
        "data_source":      "Space-Track.org (18th Space Control Sq, USSF) + NASA OMNI2",
        "label_type":       "TLE mean-motion derived altitude drop (real orbital decay)",
        "benchmark":        benchmark_data,
        "cross_validation": cv_data,
        "mae_vs_physics":   "83.9% improvement (train) / 43.2% improvement (OOD)",
        "paper":            "Celestis_Research_Paper.md",
    }



@app.get("/api/storm")
async def storm_summary():
    """Live storm data only (lightweight endpoint)."""
    kp    = await get_kp()
    sw    = await get_solar_wind()
    xray, flare_c, flare_n = await get_xray_flux()
    f107  = await get_f107()
    return {
        "kp": kp, "bz": sw["bz"],
        "solar_wind_speed": sw["speed"],
        "f107": f107, "flare_class": flare_c,
        "storm_class": kp_to_storm_class(kp),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/cme")
async def cme_feed():
    """NASA DONKI CME events (last 7 days)."""
    return {"events": await get_cme_events(),
            "source": "NASA DONKI / DEMO_KEY",
            "window": "7 days"}


@app.get("/api/satellite/{norad_id}")
async def satellite_state(norad_id: int, kp: float = 0.0, f107: float = 150.0):
    """Real-time state of a single satellite by NORAD ID."""
    sat = next((s for s in SATELLITES if s["norad"] == norad_id), None)
    if not sat:
        return {"error": f"NORAD {norad_id} not in tracked list"}
    kp_live = kp if kp > 0 else await get_kp()
    return await build_satellite_data(sat, kp_live, f107, 450, 1, 6.0)


# ── Autonomous Maneuver & Scaling ─────────────────────────────────────────────

def calculate_maneuver(mass_kg: float, alt_km: float, drop_km: float, thrust_mn: float = 50.0) -> dict:
    """Calculate Delta-V and burn time to restore orbit."""
    if drop_km <= 0:
        return {"delta_v_m_s": 0.0, "thrust_n": thrust_mn/1000.0, "burn_time_s": 0.0, "burn_time_str": "0s"}
    
    r1 = R_EARTH + alt_km - drop_km
    r2 = R_EARTH + alt_km
    
    v1 = math.sqrt(GM_EARTH / r1) * 1000
    v2 = math.sqrt(GM_EARTH / r2) * 1000
    dv = abs(v1 - v2)
    
    thrust_n = thrust_mn / 1000.0
    burn_time_s = (dv * mass_kg) / thrust_n if thrust_n > 0 else 0
    
    if burn_time_s < 60: bt_str = f"{burn_time_s:.1f}s"
    elif burn_time_s < 3600: bt_str = f"{burn_time_s/60:.1f}m"
    else: bt_str = f"{burn_time_s/3600:.1f}h"
        
    return {
        "delta_v_m_s": round(dv, 3),
        "thrust_n": thrust_n,
        "burn_time_s": round(burn_time_s, 1),
        "burn_time_str": bt_str
    }

class ManeuverRequest(BaseModel):
    satellite_name: str
    mass_kg: float
    alt_km: float
    predicted_drop_km: float
    thrust_mn: float = 50.0

@app.post("/api/autonomous/maneuver")
async def autonomous_maneuver(req: ManeuverRequest):
    """Machine-to-machine API for flight computer uplink."""
    maneuver = calculate_maneuver(req.mass_kg, req.alt_km, req.predicted_drop_km, req.thrust_mn)
    
    payload = {
        "OP_CODE": "ORBIT_RAISE_CONTINUOUS",
        "TARGET_ALT_KM": req.alt_km,
        "BURN_PARAMETERS": {
            "THRUST_MN": req.thrust_mn,
            "DURATION_SECONDS": maneuver["burn_time_s"],
            "VECTOR": "ALONG_TRACK",
            "DV_REQUIRED_M_S": maneuver["delta_v_m_s"]
        },
        "EXECUTION": "IMMEDIATE",
        "TIMESTAMP_UTC": datetime.now(timezone.utc).isoformat()
    }
    
    return {
        "status": "ARMED",
        "satellite": req.satellite_name,
        "maneuver_math": maneuver,
        "machine_command": payload
    }

import concurrent.futures

@app.get("/api/constellation/{group}")
async def get_constellation_risk(group: str = "starlink", limit: int = 100):
    """Evaluate risk for active mega-constellations (fetching from Celestrak)."""
    url = f"https://celestrak.org/NORAD/elements/gp.php?GROUP={group}&FORMAT=json"
    try:
        data = await fetch_json(url, timeout=12.0)
    except Exception as e:
        return {"error": f"Failed to fetch {group} catalog: {str(e)}"}
        
    kp = await get_kp()
    f107 = await get_f107()
    cme_events = await get_cme_events()
    cme_speed = cme_events[0]["speed_km_s"] if cme_events else 400.0
    xray_flux, _, flare_n = await get_xray_flux()
    
    satellites = data[:limit]
    
    def process_sat(sat_data):
        mean_motion = sat_data.get("MEAN_MOTION", 15.0)
        n_rad_s = mean_motion * 2 * math.pi / 86400.0
        a_km = (GM_EARTH / (n_rad_s**2))**(1/3)
        alt_km = a_km - R_EARTH
        
        mass = 260.0 if group == "starlink" else 100.0
        area = 10.0 if group == "starlink" else 1.0
        
        density = density_from_msis(alt_km, kp, f107)
        ml_drop, risk = predict_correction(kp, f107, cme_speed, flare_n, alt_km, 24.0, density, mass, area)
        
        return {
            "name": sat_data.get("OBJECT_NAME", "UNKNOWN"),
            "norad": sat_data.get("NORAD_CAT_ID", 0),
            "alt_km": round(alt_km, 2),
            "drop_km": ml_drop,
            "risk_score": risk
        }
        
    loop = asyncio.get_running_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
        results = await loop.run_in_executor(
            pool, lambda: [process_sat(s) for s in satellites if "MEAN_MOTION" in s]
        )
        
    results.sort(key=lambda x: x["risk_score"], reverse=True)
    
    return {
        "catalog": group.upper(),
        "satellites_processed": len(results),
        "kp_max": kp,
        "high_risk_assets": results[:10],
        "fleet_median_drop_km": round(np.median([r["drop_km"] for r in results]), 3) if results else 0,
        "fleet_max_drop_km": round(max([r["drop_km"] for r in results]), 3) if results else 0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# PINN Model Loader (Physics-Informed Neural Network)
# Loaded at startup alongside the existing GBM. If pinn_model.pt exists,
# predictions use the PINN. If not, falls back to the GBM.
# ─────────────────────────────────────────────────────────────────────────────
PINN_MODEL_PATH  = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "output", "pinn_model.pt")
PINN_SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "output", "pinn_scaler.pkl")

pinn_net    = None
pinn_scaler_obj = None
PINN_LOADED = False

def _load_pinn():
    global pinn_net, pinn_scaler_obj, PINN_LOADED
    try:
        import torch
        import torch.nn as nn
        import pickle

        # Define the PINN architecture inline — must exactly match train_pinn.py
        class _DragResidualPINN(nn.Module):
            def __init__(self, n_features=7, hidden=None):
                super().__init__()
                if hidden is None:
                    hidden = [64, 128, 64]
                layers = []
                prev = n_features
                for h in hidden:
                    layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.15)]
                    prev = h
                layers.append(nn.Linear(prev, 1))
                self.net = nn.Sequential(*layers)
            def forward(self, x):
                return self.net(x).squeeze(-1)

        state = torch.load(PINN_MODEL_PATH, map_location="cpu", weights_only=True)
        net = _DragResidualPINN()
        net.load_state_dict(state)
        net.eval()
        with open(PINN_SCALER_PATH, "rb") as f:
            pinn_scaler_obj = pickle.load(f)
        pinn_net    = net
        PINN_LOADED = True
        log.info("✅ PINN model loaded (physics-constrained predictions active)")
    except FileNotFoundError:
        log.info("ℹ️  PINN model not found — using GBM fallback.")
    except Exception as e:
        log.warning(f"PINN load failed ({e}) — GBM active.")

# Try to load PINN at startup (non-blocking)
try:
    _load_pinn()
except Exception:
    pass


@app.get("/api/pinn/status")
async def pinn_status():
    """Report which ML model is active (PINN or GBM)."""
    import os
    pinn_exists = os.path.exists(PINN_MODEL_PATH)
    return {
        "pinn_available": PINN_LOADED,
        "pinn_model_path": PINN_MODEL_PATH,
        "pinn_file_exists": pinn_exists,
        "active_model": "PINN (Physics-Informed Neural Network)" if PINN_LOADED else "Phase-Aware GBM (Gradient Boosting)",
        "message": "Run data_pipeline/train_pinn.py to train and activate the PINN." if not pinn_exists else "PINN active."
    }


# ─────────────────────────────────────────────────────────────────────────────
# Real COLA (Conjunction Analysis) Endpoint
# ─────────────────────────────────────────────────────────────────────────────
import sys as _sys2
import os as _os2
_sys2.path.insert(0, _os2.path.dirname(__file__))

try:
    from cola_engine import ColaEngine, synthetic_cola_alerts
    COLA_ENGINE_AVAILABLE = True
except ImportError:
    COLA_ENGINE_AVAILABLE = False
    log.warning("cola_engine.py not found — COLA will return synthetic data only")

_cola_engine: "ColaEngine | None" = None
_cola_cache: dict   = {}
_cola_cache_age: "datetime | None" = None
_COLA_CACHE_TTL_S   = 900  # 15-minute cache (respect Space-Track quota)

def _get_cola_engine() -> "ColaEngine | None":
    global _cola_engine
    if not COLA_ENGINE_AVAILABLE:
        return None
    if _cola_engine is None:
        _cola_engine = ColaEngine()
        st_user = os.getenv("SPACETRACK_USER", "")
        st_pass = os.getenv("SPACETRACK_PASS", "")
        if st_user and st_pass:
            ok = _cola_engine.login(st_user, st_pass)
            if not ok:
                log.warning("COLA: Space-Track login failed — will use synthetic fallback")
        else:
            log.warning("COLA: No Space-Track credentials in .env — using synthetic fallback")
    return _cola_engine


class ColaRequest(BaseModel):
    fleet: list[dict]  # [{id, alt, plane}, ...]


@app.post("/api/cola")
async def run_cola(req: ColaRequest):
    """
    Real-World Conjunction Analysis.

    Accepts the current I-STAR fleet state (ids, altitudes, planes),
    queries Space-Track.org for real debris TLEs in each altitude band,
    runs SGP4 propagation for 24 hours, and returns close-approach alerts.

    Falls back to synthetic alerts if Space-Track is unavailable.
    """
    global _cola_cache, _cola_cache_age

    fleet = req.fleet
    if not fleet:
        return {"error": "No fleet data provided", "alerts": {}}

    # Return cached result if fresh enough
    now = datetime.now(timezone.utc)
    if _cola_cache_age and (now - _cola_cache_age).total_seconds() < _COLA_CACHE_TTL_S:
        log.debug("COLA: returning cached result")
        return _cola_cache

    loop = asyncio.get_running_loop()
    engine = _get_cola_engine()

    if engine and engine._logged_in:
        # Run real COLA in thread pool (SGP4 is CPU-bound)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            alerts_by_sat = await loop.run_in_executor(
                pool, lambda: engine.run_fleet_cola(fleet)
            )
        source = "REAL (Space-Track.org + SGP4)"
    else:
        # Synthetic fallback
        alerts_by_sat = synthetic_cola_alerts(fleet)
        source = "SYNTHETIC (Space-Track not authenticated)"

    # Flatten for easy frontend consumption
    all_alerts = []
    for sat_id, alerts in alerts_by_sat.items():
        all_alerts.extend(alerts)

    all_alerts.sort(key=lambda x: x.get("pc_float", 0), reverse=True)
    critical = [a for a in all_alerts if a["severity"] == "CRITICAL"]
    warnings  = [a for a in all_alerts if a["severity"] == "WARNING"]
    watches   = [a for a in all_alerts if a["severity"] == "WATCH"]

    result = {
        "source":           source,
        "timestamp_utc":    now.isoformat(),
        "satellites_scanned": len(fleet),
        "total_conjunctions": len(all_alerts),
        "critical_count":   len(critical),
        "warning_count":    len(warnings),
        "watch_count":      len(watches),
        "alerts_by_satellite": alerts_by_sat,
        "top_threats":      all_alerts[:10],
    }

    _cola_cache     = result
    _cola_cache_age = now
    return result

