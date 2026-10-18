"""
CELESTIS — GNSS Orbit Determination + Feature Engineering
===========================================================
Takes filtered long-format GNSS Parquet → computes:
  1. Receiver ECEF position via Weighted Least Squares (pseudoranges)
  2. ECEF → WGS84 altitude conversion
  3. Altitude timeseries for PINN training
  4. Ionospheric TEC proxy from iono delay statistics
  5. Joins with NOAA space weather (Kp, F10.7)

Usage:
    python data_pipeline/preprocess.py
"""

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
PROCESSED_DIR = Path("data_pipeline/processed")
FEATURES_DIR  = Path("data_pipeline/features")
SPEED_OF_LIGHT = 2.99792458e8  # m/s

# WGS84 ellipsoid parameters
WGS84_A = 6378137.0          # semi-major axis (m)
WGS84_F = 1 / 298.257223563  # flattening
WGS84_B = WGS84_A * (1 - WGS84_F)
WGS84_E2 = 2*WGS84_F - WGS84_F**2  # eccentricity squared


# ── Orbit Determination ─────────────────────────────────────────────────────

def weighted_least_squares_position(
    sat_pos: np.ndarray,          # (N, 3) GPS satellite ECEF positions
    pseudoranges: np.ndarray,     # (N,)   measured pseudoranges (m)
    clock_corr_ns: np.ndarray,    # (N,)   satellite clock corrections (ns)
    cno: np.ndarray,              # (N,)   C/N0 values (dB-Hz) for weighting
    x0: Optional[np.ndarray] = None,  # (4,) initial guess [X, Y, Z, dt]
    max_iter: int = 10,
    tol: float = 1e-4
) -> Tuple[np.ndarray, float, bool]:
    """
    Compute receiver ECEF position from pseudoranges using WLS.
    
    State vector: x = [X_recv, Y_recv, Z_recv, c·dt_recv]
    Measurement:  ρ_i = |r_sat_i - r_recv| + c·dt_recv - c·dt_sat_i
    
    Weights: W_i = 10^(C/N0_i / 10) — higher C/N0 = more trusted
    
    Returns:
        position: (3,) ECEF receiver position in meters
        clock_bias_m: receiver clock bias in meters
        converged: whether solution converged
    """
    N = len(pseudoranges)
    if N < 4:
        return None, None, False
    
    # Apply satellite clock corrections
    clock_corr_m = clock_corr_ns * 1e-9 * SPEED_OF_LIGHT
    corrected_pr = pseudoranges - clock_corr_m
    
    # Measurement weights (linear scale from dB)
    weights = np.power(10, cno / 10.0)
    W = np.diag(weights)
    
    # Initial state guess: Earth center if not provided
    if x0 is None:
        # Use mean of satellite positions as rough initial guess
        x0 = np.array([
            np.mean(sat_pos[:, 0]) * 0.01,
            np.mean(sat_pos[:, 1]) * 0.01,
            np.mean(sat_pos[:, 2]) * 0.01,
            0.0
        ])
    
    x = x0.copy()
    converged = False
    
    for iteration in range(max_iter):
        r_recv = x[:3]
        c_dt   = x[3]
        
        # Geometric ranges
        diffs  = sat_pos - r_recv[np.newaxis, :]
        ranges = np.linalg.norm(diffs, axis=1)  # (N,)
        
        # Design matrix H (Jacobian)
        H = np.zeros((N, 4))
        H[:, :3] = -diffs / ranges[:, np.newaxis]  # unit vectors
        H[:, 3]  = 1.0                               # clock column
        
        # Residuals
        predicted = ranges + c_dt
        residuals = corrected_pr - predicted
        
        # WLS update: δx = (H^T W H)^{-1} H^T W δρ
        HtW  = H.T @ W
        HtWH = HtW @ H
        HtWr = HtW @ residuals
        
        try:
            dx = np.linalg.solve(HtWH, HtWr)
        except np.linalg.LinAlgError:
            break
        
        x = x + dx
        
        if np.linalg.norm(dx[:3]) < tol:
            converged = True
            break
    
    return x[:3], x[3], converged


def ecef_to_geodetic(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Convert ECEF (X, Y, Z) in meters to geodetic (lat, lon, alt).
    
    Uses Bowring's iterative method for WGS84.
    
    Returns:
        lat_deg: latitude in degrees
        lon_deg: longitude in degrees  
        alt_km:  altitude above WGS84 ellipsoid in kilometers
    """
    lon = np.arctan2(y, x)
    p   = np.sqrt(x**2 + y**2)
    lat = np.arctan2(z, p * (1 - WGS84_E2))  # initial estimate
    
    for _ in range(5):
        sinlat = np.sin(lat)
        N_curv = WGS84_A / np.sqrt(1 - WGS84_E2 * sinlat**2)
        lat    = np.arctan2(z + WGS84_E2 * N_curv * sinlat, p)
    
    sinlat = np.sin(lat)
    N_curv = WGS84_A / np.sqrt(1 - WGS84_E2 * sinlat**2)
    alt_m  = p / np.cos(lat) - N_curv
    
    return np.degrees(lat), np.degrees(lon), alt_m / 1000.0  # km


def compute_orbit_from_epoch(epoch_df: pd.DataFrame) -> Optional[dict]:
    """
    Given all channel measurements for one epoch, compute:
    - Receiver ECEF position
    - Geodetic altitude
    - Velocity estimate from Doppler
    
    Returns dict with orbital state or None if insufficient data.
    """
    # Need at least 4 good measurements
    valid = epoch_df.dropna(subset=["pseudorange_m", "sat_x_m", "sat_y_m", "sat_z_m"])
    if len(valid) < 4:
        return None
    
    sat_pos = valid[["sat_x_m", "sat_y_m", "sat_z_m"]].values.astype(float)
    pr      = valid["pseudorange_m"].values.astype(float)
    clk     = valid["sat_clock_corr_ns"].fillna(0).values.astype(float)
    cno     = valid["cno_dbhz"].fillna(30).values.astype(float)
    
    # Solve position
    pos, clock_bias, converged = weighted_least_squares_position(
        sat_pos, pr, clk, cno
    )
    
    if pos is None or not converged:
        return None
    
    # Convert to geodetic
    lat, lon, alt_km = ecef_to_geodetic(*pos)
    
    # Velocity from Doppler (km/s approximate)
    # Doppler = -v_rel / λ, λ ≈ 0.19m (GPS L1)
    doppler_vals = valid["doppler_hz"].dropna().values
    if len(doppler_vals) >= 4:
        # RMS Doppler as velocity proxy
        lambda_l1 = SPEED_OF_LIGHT / 1.57542e9  # GPS L1 wavelength
        vel_ms     = doppler_vals * lambda_l1
        speed_ms  = np.sqrt(np.mean(vel_ms**2))
    else:
        speed_ms = np.nan
    
    # Ionospheric TEC proxy
    iono_vals = valid["iono_delay_m"].dropna().values
    iono_mean = float(np.mean(iono_vals)) if len(iono_vals) > 0 else np.nan
    iono_max  = float(np.max(iono_vals))  if len(iono_vals) > 0 else np.nan
    iono_std  = float(np.std(iono_vals))  if len(iono_vals) > 0 else np.nan
    
    return {
        "epoch_idx":    int(valid["epoch_idx"].iloc[0]),
        "towc_s":       float(valid["towc_s"].iloc[0]),
        "utc_time":     valid["utc_time"].iloc[0] if "utc_time" in valid else pd.NaT,
        "n_tracked":    int(valid["n_tracked"].iloc[0]) if "n_tracked" in valid else len(valid),
        "n_used":       len(valid),
        # Orbital state
        "recv_x_m":     float(pos[0]),
        "recv_y_m":     float(pos[1]),
        "recv_z_m":     float(pos[2]),
        "latitude_deg": float(lat),
        "longitude_deg":float(lon),
        "altitude_km":  float(alt_km),
        "speed_ms":     float(speed_ms),
        "clock_bias_m": float(clock_bias),
        # Space weather proxies
        "iono_delay_mean_m": iono_mean,
        "iono_delay_max_m":  iono_max,
        "iono_delay_std_m":  iono_std,
        # Signal quality
        "mean_cno":     float(np.mean(cno)),
        "min_cno":      float(np.min(cno)),
        "range_resid_rms": float(np.sqrt(np.mean(valid["range_residual_m"].dropna()**2)))
                           if len(valid["range_residual_m"].dropna()) > 0 else np.nan,
    }


def build_orbit_timeseries(parquet_path: Path) -> pd.DataFrame:
    """
    Process all epochs in a Parquet file to produce altitude timeseries.
    
    This is the main output for PINN training.
    """
    print(f"\n  🔭 Computing orbits from: {parquet_path.name}")
    
    df = pd.read_parquet(parquet_path)
    print(f"     Loaded {len(df):,} channel-measurements, {df['epoch_idx'].nunique():,} epochs")
    
    results = []
    epochs  = df["epoch_idx"].unique()
    
    for i, epoch_idx in enumerate(sorted(epochs)):
        if i % 5000 == 0:
            pct = 100 * i / len(epochs)
            print(f"     Processing epoch {i:,}/{len(epochs):,} ({pct:.0f}%)...", end="\r")
        
        epoch_df = df[df["epoch_idx"] == epoch_idx]
        result   = compute_orbit_from_epoch(epoch_df)
        if result:
            results.append(result)
    
    print(f"\n     Solved {len(results):,} / {len(epochs):,} epochs")
    orbit_df = pd.DataFrame(results)
    return orbit_df


def compute_delta_h(orbit_df: pd.DataFrame, window_min: int = 60) -> pd.DataFrame:
    """
    Compute altitude change rate — the key target variable for PINN.
    
    ΔH = altitude drop over <window_min> minutes
    Positive ΔH = satellite is decaying (altitude is dropping — normal in drag)
    """
    orbit_df = orbit_df.sort_values("towc_s").reset_index(drop=True)
    
    # Rolling window altitude change
    window_samples = window_min * 60  # at 1 Hz
    orbit_df["alt_smooth_km"] = (
        orbit_df["altitude_km"]
        .rolling(window=60, center=True, min_periods=10)
        .mean()
    )
    
    # ΔH per hour: negative derivative of altitude
    orbit_df["dH_dt_km_per_hr"] = (
        -orbit_df["alt_smooth_km"].diff()
        .rolling(window=3600, min_periods=60)
        .mean()
        * 3600  # convert per-second to per-hour
    )
    
    return orbit_df


def engineer_features(orbit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create final feature set for PINN training.
    
    Features aligned with existing CELESTIS PINN:
    - altitude_km       → current altitude
    - iono_delay_mean_m → ionospheric TEC proxy (local Kp proxy)
    - iono_delay_std_m  → ionospheric variability
    - mean_cno          → signal quality (proxy for atmosphere density)
    - speed_ms          → orbital velocity
    
    Target:
    - dH_dt_km_per_hr   → altitude decay rate (for PINN residual training)
    """
    df = orbit_df.copy()
    
    # Lag features: altitude 1h and 6h ago
    df["alt_lag_1h"]  = df["altitude_km"].shift(3600)
    df["alt_lag_6h"]  = df["altitude_km"].shift(21600)
    df["dH_1h"]       = df["altitude_km"] - df["alt_lag_1h"]
    df["dH_6h"]       = df["altitude_km"] - df["alt_lag_6h"]
    
    # Iono TEC: rolling 10-min stats
    df["iono_roll_mean"] = df["iono_delay_mean_m"].rolling(600, min_periods=60).mean()
    df["iono_roll_max"]  = df["iono_delay_max_m"].rolling(600, min_periods=60).max()
    
    # Hour of day and day of year for solar flux correlation
    if "utc_time" in df.columns:
        df["hour_of_day"]  = df["utc_time"].dt.hour + df["utc_time"].dt.minute / 60
        df["day_of_year"]  = df["utc_time"].dt.dayofyear
    
    return df


def run_preprocessing(processed_dir: Path, features_dir: Path):
    """Process all Parquet GNSS files into orbit + feature datasets."""
    parquet_files = sorted(processed_dir.glob("*_gnss.parquet"))
    
    if not parquet_files:
        print(f"⚠️  No GNSS Parquet files found in {processed_dir}")
        print(f"   Run ingest.py first.")
        return
    
    features_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🛰️  CELESTIS Orbit Determination & Feature Engineering")
    print(f"   Processing {len(parquet_files)} GNSS Parquet file(s)\n")
    
    all_orbits = []
    for pf in parquet_files:
        out_path = features_dir / pf.name.replace("_gnss.parquet", "_orbits.parquet")
        
        if out_path.exists():
            print(f"  ✅ Already computed: {out_path.name}")
            all_orbits.append(pd.read_parquet(out_path))
            continue
        
        orbit_df = build_orbit_timeseries(pf)
        orbit_df = compute_delta_h(orbit_df)
        orbit_df = engineer_features(orbit_df)
        
        orbit_df.to_parquet(out_path, index=False)
        print(f"  💾 Saved: {out_path.name} ({out_path.stat().st_size/1e6:.1f} MB)")
        all_orbits.append(orbit_df)
    
    # Combine all orbit files into one master dataset
    if all_orbits:
        master = pd.concat(all_orbits, ignore_index=True).sort_values("towc_s")
        master_path = features_dir / "master_orbit_features.parquet"
        master.to_parquet(master_path, index=False)
        
        print(f"\n✅ Master dataset: {master_path}")
        print(f"   Rows:   {len(master):,}")
        print(f"   Altitudes: {master['altitude_km'].min():.1f} – {master['altitude_km'].max():.1f} km")
        print(f"   Time span: {master['towc_s'].min():.0f}s – {master['towc_s'].max():.0f}s")
        print(f"   Orbit solutions: {master['altitude_km'].notna().sum():,}")


if __name__ == "__main__":
    run_preprocessing(PROCESSED_DIR, FEATURES_DIR)
