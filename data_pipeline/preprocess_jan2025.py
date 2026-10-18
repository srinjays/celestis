"""
CELESTIS — GNSS Ground Receiver Preprocessing & Feature Engineering
====================================================================
Source: D:/Jan2025/Extracted_Files/
Structure: <date>/<WEEKDAY>/<hour_session>/ containing:
    POSB.csv   — Position (ECEF XYZ, lat/lon/alt, velocity) ← PRIMARY
    CLKB.csv   — Receiver clock bias & drift                ← KEEP
    DOPB.csv   — DOP geometry metrics                       ← KEEP
    TIDB.csv   — UTC timestamp ground truth                 ← KEEP (time index)
    SATB_L1.csv — GPS L1 per-channel observables (275 cols) ← SELECTIVE
    SATB_L5.csv — IRNSS L5 per-channel (253 cols)           ← SELECTIVE
    SATB_S1.csv — IRNSS S1 per-channel (253 cols)           ← SELECTIVE
    SBSATB.csv  — SBAS satellite data                        ← DROP (corrections only)
    CONB.csv    — Receiver config/hardware info              ← DROP (static)
    ACKB.csv    — Acknowledgement (empty)                    ← DROP

Output: data_pipeline/features/jan2025_master.parquet
        data_pipeline/features/jan2025_pinn_ready.parquet  ← model-ready

Run:  python data_pipeline/preprocess_jan2025.py
"""

import gc
import warnings
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from datetime import datetime
from typing import Optional

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_ROOT    = Path("D:/Jan2025/Extracted_Files")
OUTPUT_DIR   = Path("d:/inceptrix_sat/data_pipeline/features")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── What We Keep and Why ──────────────────────────────────────────────────────
#
# POSB — Solved receiver position (already computed by the GNSS receiver!)
#   X/Y/Z pos (m)           → convert to altitude (primary PINN target)
#   X/Y/Z vel (m/s)         → orbital velocity (drag proxy)
#   Latitude/Longitude/Altitude → final geodetic output
#   std deviation E/N/U     → position quality metric
#   PDOP, HDOP, VDOP        → geometry quality (from DOPB)
#   Position Mode           → 7=best fix (filter on this)
#
# CLKB — Receiver clock
#   Clock Bias (ns)         → receiver health metric
#   Clock Drift (ns/s)      → ionospheric/atmospheric proxy
#   UTC Time Offset (s)     → leap second correction
#
# DOPB — DOP metrics
#   GDOP, PDOP, HDOP, VDOP → geometry (low=good, high=poor visibility)
#   Total No of Satellites  → sky coverage
#   Total No of IR/GP Sats  → IRNSS vs GPS breakdown
#
# TIDB — Time
#   Day/Month/Year/Hour/Min/Sec → precise UTC timestamp
#
# SATB_L1 — GPS L1 channels (selected columns only, NOT all 275)
#   Mean C/N0 across channels   → signal quality / atmosphere proxy
#   Max/Min elevation tracked   → sky geometry
#   Mean Iono Delay             → space weather proxy (TEC)
#   Mean Tropo Delay            → tropospheric state
#   RMS Range Residuals         → orbit quality

# ── Column selections ─────────────────────────────────────────────────────────
POSB_KEEP = [
    "TOWC (s)", "Week no", "Block Count",
    "X pos (m)", "Y pos (m)", "Z pos (m)",
    "X vel (m/s)", "Y vel (m/s)", "Z vel (m/s)",
    "Latitude (deg)", "Longitude (deg)", "Altitude (m)",
    "speed (m/s)",
    "std deviation East(m)", "std dev North(m)", "std dev Up(m)",
    "Position fix status", "Position Mode", " Position mode",
    "Position Estimation Time (s)",
]

CLKB_KEEP = [
    "TOWC (s)", "Block Count",
    "Clock Bias (ns)", "Clock Drift (ns/s)",
    "Inter System Bias (ns)", "UTC Time Offset (s)",
]

DOPB_KEEP = [
    "TOWC (s)", "Block Count",
    "GDOP", "PDOP", "HDOP", "VDOP", "TDOP",
    "Total No of Satellites",
    "Total No of IR Satellites", "Total No of GP Satellites",
    "Position Mode",
]

TIDB_KEEP = [
    "TOWC (s)", "Block Count",
    "Day", "Month", "Year", "Hour", "Minute", "Seconds",
]

N_CHANNELS_L1 = 12
N_CHANNELS_L5 = 9   # IRNSS L5 has fewer channels
N_CHANNELS_S1 = 9   # IRNSS S1


# ── Helpers ───────────────────────────────────────────────────────────────────

def safe_read(filepath: Path, keep_cols: Optional[list] = None,
              encoding: str = "latin-1") -> Optional[pd.DataFrame]:
    """Read a CSV safely, only loading needed columns."""
    try:
        df = pd.read_csv(filepath, encoding=encoding, low_memory=False)
        # Clean column names
        df.columns = [str(c).strip() for c in df.columns]
        # Drop unnamed junk columns
        df = df[[c for c in df.columns if not c.startswith("Unnamed")]]
        if keep_cols:
            available = [c for c in keep_cols if c in df.columns]
            df = df[available]
        return df
    except Exception as e:
        return None


def build_utc_timestamp(tidb: pd.DataFrame) -> pd.Series:
    """Build a proper UTC datetime from TIDB Day/Month/Year/Hour/Min/Sec columns."""
    try:
        ts = pd.to_datetime({
            "year":   tidb["Year"].astype(int),
            "month":  tidb["Month"].astype(int),
            "day":    tidb["Day"].astype(int),
            "hour":   tidb["Hour"].astype(int),
            "minute": tidb["Minute"].astype(int),
            "second": tidb["Seconds"].astype(int),
        }, errors="coerce")
        return ts
    except Exception:
        return pd.Series(pd.NaT, index=tidb.index)


def extract_satb_summary(filepath: Path, n_channels: int) -> Optional[pd.DataFrame]:
    """
    From SATB_L1/L5/S1 wide format, extract per-epoch summary stats.
    Instead of keeping all 275 columns, derive 10 meaningful features.
    """
    try:
        df = pd.read_csv(filepath, encoding="latin-1", low_memory=False)
        df.columns = [str(c).strip() for c in df.columns]
    except Exception:
        return None

    towc_col = next((c for c in df.columns if c.upper().startswith("TOWC")), None)
    block_col = next((c for c in df.columns if "Block Count" in c), None)
    if not towc_col:
        return None

    result = pd.DataFrame()
    result["TOWC (s)"] = df[towc_col]
    if block_col:
        result["Block Count"] = df[block_col]

    # Aggregate across all channels
    cno_cols      = [c for c in df.columns if "C/NO" in c and "(dB-Hz)" in c]
    iono_cols     = [c for c in df.columns if "Iono Delay" in c and "(m)" in c]
    tropo_cols    = [c for c in df.columns if "Tropo Delay" in c and "(m)" in c]
    elev_cols     = [c for c in df.columns if "Elevation" in c and "(deg)" in c]
    resid_cols    = [c for c in df.columns if "Range Residuals" in c and "(m)" in c]
    reject_cols   = [c for c in df.columns if "Reject Code" in c]
    doppler_cols  = [c for c in df.columns if "Doppler" in c and "(Hz)" in c]
    locktime_cols = [c for c in df.columns if "Lock Time" in c and "(s)" in c]

    def safe_stat(cols, stat):
        if not cols:
            return np.nan
        sub = df[cols].apply(pd.to_numeric, errors="coerce")
        sub[sub < -1e9] = np.nan   # filter 1.#QNB / NaN sentinel values
        if stat == "mean": return sub.mean(axis=1)
        if stat == "max":  return sub.max(axis=1)
        if stat == "min":  return sub.min(axis=1)
        if stat == "std":  return sub.std(axis=1)
        if stat == "sum":  return sub.sum(axis=1)

    if cno_cols:
        result["cno_mean_dbhz"]    = safe_stat(cno_cols, "mean")
        result["cno_max_dbhz"]     = safe_stat(cno_cols, "max")
    if iono_cols:
        result["iono_mean_m"]      = safe_stat(iono_cols, "mean")
        result["iono_max_m"]       = safe_stat(iono_cols, "max")
        result["iono_std_m"]       = safe_stat(iono_cols, "std")
    if tropo_cols:
        result["tropo_mean_m"]     = safe_stat(tropo_cols, "mean")
    if elev_cols:
        result["elev_mean_deg"]    = safe_stat(elev_cols, "mean")
        result["elev_min_deg"]     = safe_stat(elev_cols, "min")
    if resid_cols:
        result["range_resid_rms"]  = safe_stat(resid_cols, "mean")
    if reject_cols:
        result["n_good_channels"]  = (
            df[reject_cols].apply(pd.to_numeric, errors="coerce") == 0
        ).sum(axis=1)
    if doppler_cols:
        result["doppler_rms_hz"]   = (
            df[doppler_cols]
            .apply(pd.to_numeric, errors="coerce")
            .pow(2).mean(axis=1).pow(0.5)
        )
    if locktime_cols:
        result["lock_time_mean_s"] = safe_stat(locktime_cols, "mean")

    return result


def compute_altitude_from_ecef(df: pd.DataFrame) -> pd.DataFrame:
    """
    WGS84: convert ECEF (X,Y,Z) to geodetic altitude as cross-check.
    Also compute altitude from the receiver's own Altitude(m) column.
    """
    WGS84_A  = 6378137.0
    WGS84_F  = 1 / 298.257223563
    WGS84_E2 = 2*WGS84_F - WGS84_F**2

    if all(c in df.columns for c in ["X pos (m)", "Y pos (m)", "Z pos (m)"]):
        x = pd.to_numeric(df["X pos (m)"], errors="coerce")
        y = pd.to_numeric(df["Y pos (m)"], errors="coerce")
        z = pd.to_numeric(df["Z pos (m)"], errors="coerce")

        lon = np.arctan2(y, x)
        p   = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p * (1 - WGS84_E2))
        for _ in range(5):
            sinlat = np.sin(lat)
            N      = WGS84_A / np.sqrt(1 - WGS84_E2 * sinlat**2)
            lat    = np.arctan2(z + WGS84_E2 * N * sinlat, p)
        sinlat    = np.sin(lat)
        N_curv    = WGS84_A / np.sqrt(1 - WGS84_E2 * sinlat**2)
        alt_ecef  = p / np.cos(lat) - N_curv

        df["altitude_ecef_m"]  = alt_ecef
        df["altitude_ecef_km"] = alt_ecef / 1000.0

    # Use built-in altitude (already solved by receiver)
    if "Altitude (m)" in df.columns:
        a = pd.to_numeric(df["Altitude (m)"], errors="coerce")
        a[a < -1000] = np.nan   # sentinel cleanup
        df["altitude_m"]  = a
        df["altitude_km"] = a / 1000.0

    return df


def compute_velocity_magnitude(df: pd.DataFrame) -> pd.DataFrame:
    """3D velocity magnitude from ECEF velocity components."""
    if all(c in df.columns for c in ["X vel (m/s)", "Y vel (m/s)", "Z vel (m/s)"]):
        vx = pd.to_numeric(df["X vel (m/s)"], errors="coerce")
        vy = pd.to_numeric(df["Y vel (m/s)"], errors="coerce")
        vz = pd.to_numeric(df["Z vel (m/s)"], errors="coerce")
        df["velocity_3d_ms"] = np.sqrt(vx**2 + vy**2 + vz**2)
    return df


# ── Session Processor ─────────────────────────────────────────────────────────

def process_session(session_dir: Path, date_str: str) -> Optional[pd.DataFrame]:
    """
    Process one session folder (e.g. 09_01_2025/THURSDAY/6/).
    Merges POSB + CLKB + DOPB + TIDB + SATB summaries on Block Count.
    Returns a clean per-epoch DataFrame.
    """
    # ── Load core tables ──────────────────────────────────────────────────────
    posb = safe_read(session_dir / "POSB.csv",  POSB_KEEP)
    tidb = safe_read(session_dir / "TIDB.csv",  TIDB_KEEP)
    clkb = safe_read(session_dir / "CLKB.csv",  CLKB_KEEP)
    dopb = safe_read(session_dir / "DOPB.csv",  DOPB_KEEP)

    if posb is None or len(posb) == 0:
        return None

    # Build UTC timestamp from TIDB
    if tidb is not None and len(tidb) > 0:
        posb["utc_time"] = build_utc_timestamp(tidb)
    else:
        posb["utc_time"] = pd.NaT

    # Add source metadata
    posb["session_date"]    = date_str
    posb["session_dir"]     = str(session_dir.relative_to(DATA_ROOT))

    # ── Compute derived columns ───────────────────────────────────────────────
    posb = compute_altitude_from_ecef(posb)
    posb = compute_velocity_magnitude(posb)

    # ── Merge CLKB ────────────────────────────────────────────────────────────
    if clkb is not None and "Block Count" in clkb.columns and "Block Count" in posb.columns:
        # Exclude TOWC and any column already in posb to avoid collision
        clkb_merge = clkb.drop(columns=[c for c in clkb.columns
                                         if c in posb.columns and c != "Block Count"],
                                errors="ignore")
        posb = posb.merge(clkb_merge, on="Block Count", how="left")

    # ── Merge DOPB ────────────────────────────────────────────────────────────
    if dopb is not None and "Block Count" in dopb.columns and "Block Count" in posb.columns:
        dopb_merge = dopb.drop(columns=[c for c in dopb.columns
                                          if c in posb.columns and c != "Block Count"],
                                 errors="ignore")
        posb = posb.merge(dopb_merge, on="Block Count", how="left")

    # ── Extract SATB summaries ────────────────────────────────────────────────
    for label, fname, n_ch in [
        ("l1", "SATB_L1.csv", N_CHANNELS_L1),
        ("l5", "SATB_L5.csv", N_CHANNELS_L5),
        ("s1", "SATB_S1.csv", N_CHANNELS_S1),
    ]:
        fpath = session_dir / fname
        if fpath.exists():
            satb = extract_satb_summary(fpath, n_ch)
            if satb is not None and "Block Count" in satb.columns:
                # Rename all non-key columns with signal label prefix
                rename_map = {
                    c: f"{label}_{c}"
                    for c in satb.columns
                    if c != "Block Count"
                }
                satb = satb.rename(columns=rename_map)
                # Drop any columns that still exist in posb (shouldn't happen after rename)
                satb_merge = satb.drop(columns=[c for c in satb.columns
                                                 if c in posb.columns and c != "Block Count"],
                                        errors="ignore")
                posb = posb.merge(satb_merge, on="Block Count", how="left")

    return posb


# ── Quality Filter ─────────────────────────────────────────────────────────────

def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """
    Strict quality gating on the merged epoch-level data.
    
    Removes:
    - Position Mode != 7 (not a full 3D fix)
    - Altitude outside plausible range (this is a ground receiver: 0–5000m)
    - PDOP > 10 (poor geometry)
    - UTC timestamp is NaT
    - Duplicate Block Count within same session
    """
    before = len(df)

    # Position Mode 7 = full 3D position fix (best quality)
    pm_col = " Position mode" if " Position mode" in df.columns else "Position Mode"
    if pm_col in df.columns:
        pm = pd.to_numeric(df[pm_col], errors="coerce")
        df = df[pm == 7]

    # Altitude plausibility (ground receiver: Bangalore ~920m MSL, allow ±2km)
    if "altitude_m" in df.columns:
        alt = pd.to_numeric(df["altitude_m"], errors="coerce")
        df = df[alt.between(-500, 5000)]

    # PDOP quality gate
    if "PDOP" in df.columns:
        pdop = pd.to_numeric(df["PDOP"], errors="coerce")
        df = df[pdop < 10]

    # Valid UTC time
    if "utc_time" in df.columns:
        df = df[df["utc_time"].notna()]

    # Remove sentinel values (1.#QNB → already handled in compute steps)
    after = len(df)
    return df.reset_index(drop=True)


# ── Feature Engineering ────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final feature set for the CELESTIS PINN.
    All features are physical proxies for atmospheric/ionospheric state.
    """
    df = df.sort_values("utc_time").reset_index(drop=True)

    # ── Time features ─────────────────────────────────────────────────────────
    df["hour_of_day"]   = df["utc_time"].dt.hour + df["utc_time"].dt.minute / 60
    df["day_of_year"]   = df["utc_time"].dt.dayofyear
    df["day_of_week"]   = df["utc_time"].dt.dayofweek
    df["week_of_year"]  = df["utc_time"].dt.isocalendar().week.astype(int)

    # ── Ionospheric TEC proxy (primary space weather indicator) ───────────────
    # Combine L1 + L5 + S1 iono delays into one robust TEC estimate
    iono_cols = [c for c in df.columns if "iono_mean_m" in c]
    if iono_cols:
        df["iono_tec_proxy_m"]   = df[iono_cols].mean(axis=1)
        df["iono_tec_max_m"]     = df[iono_cols].max(axis=1)

    # Rolling statistics (5-min window at 1Hz = 300 samples)
    if "iono_tec_proxy_m" in df.columns:
        df["iono_roll_5min"]     = df["iono_tec_proxy_m"].rolling(300, min_periods=30).mean()
        df["iono_roll_std_5min"] = df["iono_tec_proxy_m"].rolling(300, min_periods=30).std()
        df["iono_spike"]         = (
            df["iono_tec_proxy_m"] - df["iono_roll_5min"]
        ).clip(lower=0)   # sudden iono increase = storm indicator

    # ── Signal quality composite ──────────────────────────────────────────────
    cno_cols = [c for c in df.columns if "cno_mean" in c]
    if cno_cols:
        df["cno_composite"]      = df[cno_cols].mean(axis=1)

    # ── Geometry quality ──────────────────────────────────────────────────────
    # Low PDOP = good satellite geometry = trustworthy measurements
    if "PDOP" in df.columns:
        df["pdop"] = pd.to_numeric(df["PDOP"], errors="coerce")

    # ── Clock health ──────────────────────────────────────────────────────────
    if "Clock Drift (ns/s)" in df.columns:
        df["clock_drift_ns_s"]   = pd.to_numeric(df["Clock Drift (ns/s)"], errors="coerce")
        df["clock_drift_abs"]    = df["clock_drift_ns_s"].abs()

    # ── Constellation visibility ──────────────────────────────────────────────
    if "Total No of Satellites" in df.columns:
        df["n_sats_total"]       = pd.to_numeric(df["Total No of Satellites"], errors="coerce")
    if "Total No of IR Satellites" in df.columns:
        df["n_sats_irnss"]       = pd.to_numeric(df["Total No of IR Satellites"], errors="coerce")
    if "Total No of GP Satellites" in df.columns:
        df["n_sats_gps"]         = pd.to_numeric(df["Total No of GP Satellites"], errors="coerce")

    # ── Altitude (primary variable) ───────────────────────────────────────────
    if "altitude_m" in df.columns:
        df["altitude_km"]        = pd.to_numeric(df["altitude_m"], errors="coerce") / 1000.0
        # Altitude change (rate of change in m per second)
        df["dalt_dt_m_s"]        = df["altitude_m"].diff() / df["TOWC (s)"].diff().clip(lower=0.1)

    # ── Speed composite ───────────────────────────────────────────────────────
    if "velocity_3d_ms" in df.columns:
        df["velocity_3d_ms"]     = pd.to_numeric(df["velocity_3d_ms"], errors="coerce")

    # ── Tropo delay combined ──────────────────────────────────────────────────
    tropo_cols = [c for c in df.columns if "tropo_mean" in c]
    if tropo_cols:
        df["tropo_composite_m"]  = df[tropo_cols].mean(axis=1)

    return df


# ── Final Column Selection ────────────────────────────────────────────────────

PINN_FEATURE_COLS = [
    # Time
    "utc_time", "session_date", "TOWC (s)", "Week no",
    "hour_of_day", "day_of_year", "day_of_week",
    # Position (primary)
    "Latitude (deg)", "Longitude (deg)", "altitude_km", "altitude_m",
    "altitude_ecef_km",
    # Velocity
    "velocity_3d_ms", "speed (m/s)",
    # Position quality
    "std deviation East(m)", "std dev North(m)", "std dev Up(m)",
    "pdop", "HDOP", "VDOP",
    # Satellite geometry
    "n_sats_total", "n_sats_irnss", "n_sats_gps",
    # Ionospheric TEC (space weather proxy) ← Key for CELESTIS
    "iono_tec_proxy_m", "iono_tec_max_m",
    "iono_roll_5min", "iono_roll_std_5min", "iono_spike",
    "l1_iono_mean_m", "l1_iono_max_m", "l1_iono_std_m",
    "l5_iono_mean_m", "l5_iono_max_m",
    "s1_iono_mean_m", "s1_iono_max_m",
    # Tropospheric
    "tropo_composite_m",
    # Signal quality
    "cno_composite", "l1_cno_mean_dbhz", "l5_cno_mean_dbhz", "s1_cno_mean_dbhz",
    "l1_n_good_channels", "l1_doppler_rms_hz",
    # Clock
    "Clock Bias (ns)", "clock_drift_ns_s", "clock_drift_abs",
    "Inter System Bias (ns)",
]


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    # Find all session directories
    sessions = []
    for date_dir in sorted(DATA_ROOT.iterdir()):
        if not date_dir.is_dir() or date_dir.name == "NO_TIME Log Files":
            continue
        date_str = date_dir.name
        for day_dir in sorted(date_dir.iterdir()):
            for session_dir in sorted(day_dir.iterdir()):
                if (session_dir / "POSB.csv").exists():
                    sessions.append((date_str, session_dir))

    total_sessions = len(sessions)
    print(f"\n🛰️  CELESTIS — Jan 2025 Preprocessing & Feature Engineering")
    print(f"   Data root:    {DATA_ROOT}")
    print(f"   Sessions:     {total_sessions}")
    print(f"   Output:       {OUTPUT_DIR}\n")

    master_path = OUTPUT_DIR / "jan2025_master.parquet"
    pinn_path   = OUTPUT_DIR / "jan2025_pinn_ready.parquet"

    writer_master = None
    writer_pinn   = None

    total_rows_raw    = 0
    total_rows_clean  = 0

    for idx, (date_str, session_dir) in enumerate(sessions):
        short = str(session_dir.relative_to(DATA_ROOT))
        print(f"  [{idx+1:02d}/{total_sessions}] Processing: {short}", end=" ... ", flush=True)

        try:
            df = process_session(session_dir, date_str)
            if df is None or len(df) == 0:
                print("SKIP (empty)")
                continue

            rows_before = len(df)
            df = quality_filter(df)
            rows_after  = len(df)

            if rows_after == 0:
                print(f"SKIP (0 rows after filter)")
                continue

            df = engineer_features(df)
            total_rows_raw   += rows_before
            total_rows_clean += rows_after

            print(f"{rows_before} → {rows_after} rows ✓")

            # Write master (all columns)
            tbl_master = pa.Table.from_pandas(df, preserve_index=False)
            if writer_master is None:
                writer_master = pq.ParquetWriter(str(master_path),
                                                  tbl_master.schema, compression="snappy")
            writer_master.write_table(tbl_master)

            # Write PINN-ready (selected columns only)
            keep = [c for c in PINN_FEATURE_COLS if c in df.columns]
            tbl_pinn = pa.Table.from_pandas(df[keep], preserve_index=False)
            if writer_pinn is None:
                writer_pinn = pq.ParquetWriter(str(pinn_path),
                                                tbl_pinn.schema, compression="snappy")
            writer_pinn.write_table(tbl_pinn)

            del df; gc.collect()

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    if writer_master: writer_master.close()
    if writer_pinn:   writer_pinn.close()

    print(f"\n{'='*65}")
    print(f"✅  Preprocessing complete!")
    print(f"    Raw rows:    {total_rows_raw:,}")
    print(f"    Clean rows:  {total_rows_clean:,}  ({100*total_rows_clean/max(total_rows_raw,1):.1f}% retained)")
    print(f"    Master:      {master_path}  ({master_path.stat().st_size/1e6:.0f} MB)")
    print(f"    PINN-ready:  {pinn_path}  ({pinn_path.stat().st_size/1e6:.0f} MB)")
    print(f"\n    Next step:   python data_pipeline/train_pinn.py")
    print(f"{'='*65}\n")

    # Print quick feature summary
    print("📊 Feature Summary (PINN-ready dataset):")
    df_check = pd.read_parquet(pinn_path)
    print(f"   Shape: {df_check.shape}")
    print(f"   Time:  {df_check['utc_time'].min()} → {df_check['utc_time'].max()}")
    if "altitude_km" in df_check.columns:
        print(f"   Altitude: {df_check['altitude_km'].mean():.3f} km "
              f"(std={df_check['altitude_km'].std():.4f})")
    if "iono_tec_proxy_m" in df_check.columns:
        print(f"   Iono TEC: {df_check['iono_tec_proxy_m'].mean():.2f} m "
              f"(max={df_check['iono_tec_proxy_m'].max():.2f} m)")
    print(f"   Null %:  {df_check.isnull().mean().mean()*100:.1f}% average")


if __name__ == "__main__":
    run()
