"""
CELESTIS — Memory-Safe GNSS Telemetry Ingestor (Streaming)
============================================================
Handles large Excel files by streaming row-by-row (no full RAM load).
Auto-detects TOWC week rollovers across January (week boundaries every Sunday).

Drop ALL your January .xlsx files into:  data_pipeline/raw/
Then run:                                python data_pipeline/ingest.py

GPS Week reference for January 2026:
  Week 2347: Sun Jan  4, 2026 → Sat Jan 10, 2026
  Week 2348: Sun Jan 11, 2026 → Sat Jan 17, 2026
  Week 2349: Sun Jan 18, 2026 → Sat Jan 24, 2026
  Week 2350: Sun Jan 25, 2026 → Sat Jan 31, 2026
"""

import os
import gc
import sys
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import openpyxl
from pathlib import Path
from datetime import datetime, timedelta
from typing import Iterator, List, Optional

# ── Config ───────────────────────────────────────────────────────────────────
RAW_DIR      = Path("data_pipeline/raw")
OUTPUT_DIR   = Path("data_pipeline/processed")
CHUNK_SIZE   = 10_000          # rows to accumulate before writing to Parquet
N_CHANNELS   = 12

# GPS epoch
GPS_EPOCH         = datetime(1980, 1, 6, 0, 0, 0)
GPS_LEAPSECONDS   = 18         # GPS - UTC offset as of 2025
WEEK_SECONDS      = 604800     # seconds per GPS week
TOWC_ROLLOVER_GAP = 3600       # if TOWC drops by more than this → new week

# January 2026 GPS week map (Sunday date → GPS week number)
JANUARY_2026_WEEKS = {
    "2026-01-04": 2347,
    "2026-01-11": 2348,
    "2026-01-18": 2349,
    "2026-01-25": 2350,
}
GPS_WEEK_START = 2347  # Jan 4, 2026

# ── Column Detection ─────────────────────────────────────────────────────────

def build_column_map(headers: List[str]) -> dict:
    """
    Map raw Excel headers to canonical names.
    Handles duplicated TOWC columns and any column ordering.
    """
    col_map = {}
    towc_seen = 0
    
    for i, h in enumerate(headers):
        h_str = str(h).strip()
        
        if h_str.upper().startswith("TOWC"):
            if towc_seen == 0:
                col_map["towc_primary"] = i
            towc_seen += 1
        elif "System Status" in h_str:
            col_map["system_status"] = i
        elif "Block Count" in h_str:
            col_map["block_count"] = i
        elif "No of tracked" in h_str:
            col_map["n_tracked"] = i
        elif "Acquisition status word1" in h_str:
            col_map["acq_word1"] = i
        elif "Acquisition status word2" in h_str:
            col_map["acq_word2"] = i
        else:
            # Per-channel columns (Chan-N pattern)
            for n in range(1, N_CHANNELS + 1):
                suffix = f"-{n}"
                if h_str.endswith(suffix) or f"-{n} " in h_str or f"-{n})" in h_str:
                    key = h_str.replace(f"-{n}", "").strip()
                    col_map[f"ch{n}_{key}"] = i
                    break
    
    return col_map


# ── Streaming Excel Reader ───────────────────────────────────────────────────

def stream_excel_rows(filepath: Path, skip_rows: int = 0) -> Iterator[tuple]:
    """
    Stream rows from Excel using openpyxl read_only mode.
    This uses O(1) memory — only one row at a time is in RAM.
    
    Yields: (row_values_list,)
    """
    wb = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    ws = wb.active
    
    for i, row in enumerate(ws.iter_rows(values_only=True)):
        if i < skip_rows:
            continue
        yield row
    
    wb.close()


def get_excel_header(filepath: Path) -> List[str]:
    """Read only the first row (header) from Excel."""
    wb  = openpyxl.load_workbook(str(filepath), read_only=True, data_only=True)
    ws  = wb.active
    header = next(ws.iter_rows(values_only=True))
    wb.close()
    return [str(h) if h is not None else "" for h in header]


# ── TOWC → UTC Conversion ────────────────────────────────────────────────────

class TOWCConverter:
    """
    Converts GPS Time of Week Count (seconds) to UTC datetime.
    Handles week rollovers automatically by detecting TOWC resets.
    """
    def __init__(self, start_gps_week: int = GPS_WEEK_START):
        self.gps_week    = start_gps_week
        self.prev_towc   = None
    
    def convert(self, towc: float) -> Optional[datetime]:
        if towc is None or np.isnan(float(towc)):
            return None
        
        towc = float(towc)
        
        # Detect week rollover: TOWC dropped by a large amount
        if self.prev_towc is not None and (self.prev_towc - towc) > TOWC_ROLLOVER_GAP:
            self.gps_week += 1
            print(f"\n  🔄 GPS week rollover detected: TOWC {self.prev_towc:.0f}→{towc:.0f} "
                  f"| Now on GPS week {self.gps_week}")
        
        self.prev_towc = towc
        
        # GPS time in seconds since GPS epoch
        gps_secs = self.gps_week * WEEK_SECONDS + towc - GPS_LEAPSECONDS
        return GPS_EPOCH + timedelta(seconds=gps_secs)


# ── Wide Row → Channel Records ────────────────────────────────────────────────

CHANNEL_FIELD_MAP = {
    "Chan No":                      "channel_no",
    "PRN":                          "prn",
    "Channel Tracking Status":      "tracking_status",
    "Doppler (Hz)":                 "doppler_hz",
    "C/NO (dB-Hz)":                 "cno_dbhz",
    "Azimuth (deg)":                "azimuth_deg",
    "Elevation (deg)":              "elevation_deg",
    "PR (m)":                       "pseudorange_m",
    "DR (m)":                       "delta_range_m",
    "Reject Code":                  "reject_code",
    "Lock Time (s)":                "lock_time_s",
    "Iono Delay (m)":               "iono_delay_m",
    "Tropo Delay (m)":              "tropo_delay_m",
    "Carrier Delay (cycles)":       "carrier_cycles",
    "Satellite X Position (m)":     "sat_x_m",
    "Satellite Y Position (m)":     "sat_y_m",
    "Satellite Z Position (m)":     "sat_z_m",
    "Satellite X Velocity (m/s)":   "sat_vx_ms",
    "Satellite Y Velocity (m/s)":   "sat_vy_ms",
    "Satellite Z Velocity (m/s)":   "sat_vz_ms",
    "Range Residuals (m)":          "range_residual_m",
    "Satellite Clock Corrections (ns)": "sat_clock_corr_ns",
}


def row_to_records(
    row_values: tuple,
    col_map: dict,
    epoch_idx: int,
    towc_s: float,
    utc_time: Optional[datetime],
    n_tracked: int,
) -> List[dict]:
    """
    Convert one wide-format row to N channel record dicts.
    Only returns channels that have a valid PRN.
    """
    records = []
    
    for n in range(1, N_CHANNELS + 1):
        rec = {
            "epoch_idx": epoch_idx,
            "towc_s":    towc_s,
            "utc_time":  utc_time,
            "n_tracked": n_tracked,
            "channel":   n,
        }
        
        # Pull each per-channel field
        for raw_key, canonical in CHANNEL_FIELD_MAP.items():
            col_key = f"ch{n}_{raw_key}"
            if col_key in col_map:
                val = row_values[col_map[col_key]]
                rec[canonical] = float(val) if val is not None else np.nan
            else:
                rec[canonical] = np.nan
        
        # Only keep channels with a real PRN
        if not np.isnan(rec.get("prn", np.nan)) and rec.get("prn", 0) > 0:
            records.append(rec)
    
    return records


# ── Parquet Writer ────────────────────────────────────────────────────────────

OUTPUT_SCHEMA = pa.schema([
    ("epoch_idx",        pa.int64()),
    ("towc_s",           pa.float64()),
    ("utc_time",         pa.timestamp("s")),
    ("n_tracked",        pa.float32()),
    ("channel",          pa.int32()),
    ("prn",              pa.float32()),
    ("tracking_status",  pa.float32()),
    ("doppler_hz",       pa.float32()),
    ("cno_dbhz",         pa.float32()),
    ("azimuth_deg",      pa.float32()),
    ("elevation_deg",    pa.float32()),
    ("pseudorange_m",    pa.float64()),
    ("delta_range_m",    pa.float32()),
    ("reject_code",      pa.float32()),
    ("lock_time_s",      pa.float32()),
    ("iono_delay_m",     pa.float32()),
    ("tropo_delay_m",    pa.float32()),
    ("carrier_cycles",   pa.float64()),
    ("sat_x_m",          pa.float64()),
    ("sat_y_m",          pa.float64()),
    ("sat_z_m",          pa.float64()),
    ("sat_vx_ms",        pa.float32()),
    ("sat_vy_ms",        pa.float32()),
    ("sat_vz_ms",        pa.float32()),
    ("range_residual_m", pa.float32()),
    ("sat_clock_corr_ns",pa.float32()),
])


def records_to_table(records: List[dict]) -> pa.Table:
    df = pd.DataFrame(records)
    if "utc_time" in df.columns:
        df["utc_time"] = pd.to_datetime(df["utc_time"], errors="coerce")
    for field in OUTPUT_SCHEMA:
        if field.name not in df.columns:
            df[field.name] = np.nan
    return pa.Table.from_pandas(df[OUTPUT_SCHEMA.names], schema=OUTPUT_SCHEMA,
                                preserve_index=False)


# ── Main Processing Function ──────────────────────────────────────────────────

def process_file(xlsx_path: Path, output_dir: Path,
                 start_gps_week: int = GPS_WEEK_START) -> Path:
    """
    Stream-process one Excel file → Parquet.
    Memory usage: O(CHUNK_SIZE) rows at a time regardless of file size.
    """
    stem     = xlsx_path.stem
    out_path = output_dir / f"{stem}_gnss.parquet"
    
    if out_path.exists():
        size = out_path.stat().st_size / 1e6
        print(f"  ✅ Already processed: {out_path.name} ({size:.0f} MB) — skipping")
        return out_path
    
    file_size_mb = xlsx_path.stat().st_size / 1e6
    print(f"\n{'='*65}")
    print(f"  📂 {xlsx_path.name}  ({file_size_mb:.0f} MB)")
    print(f"{'='*65}")
    
    # Step 1: Read header
    print(f"  Reading column headers...", end=" ", flush=True)
    headers = get_excel_header(xlsx_path)
    col_map = build_column_map(headers)
    print(f"{len(headers)} columns detected, {len(col_map)} mapped")
    
    if "towc_primary" not in col_map:
        print(f"  ⚠️  WARNING: No TOWC column found! Check column names.")
    
    # Step 2: Stream rows and write Parquet in chunks
    output_dir.mkdir(parents=True, exist_ok=True)
    converter  = TOWCConverter(start_gps_week=start_gps_week)
    writer     = None
    buffer     = []
    epoch_idx  = 0
    rows_read  = 0
    rows_kept  = 0
    
    print(f"  Streaming rows: ", end="", flush=True)
    
    for row in stream_excel_rows(xlsx_path, skip_rows=1):  # skip header
        rows_read += 1
        
        # Progress print every 10k rows
        if rows_read % 10_000 == 0:
            print(f"\r  Streaming rows: {rows_read:,} read | "
                  f"{rows_kept:,} channel records | "
                  f"GPS week {converter.gps_week}", end="", flush=True)
        
        # Extract TOWC and epoch-level metadata
        towc_idx = col_map.get("towc_primary")
        towc_s   = float(row[towc_idx]) if (towc_idx is not None 
                                             and row[towc_idx] is not None) else np.nan
        if np.isnan(towc_s):
            continue  # blank row
        
        utc_time  = converter.convert(towc_s)
        n_tracked = row[col_map["n_tracked"]] if "n_tracked" in col_map else np.nan
        
        # Convert wide row → per-channel records
        channel_recs = row_to_records(
            row, col_map, epoch_idx, towc_s, utc_time,
            float(n_tracked) if n_tracked is not None else np.nan
        )
        buffer.extend(channel_recs)
        rows_kept  += len(channel_recs)
        epoch_idx  += 1
        
        # Flush buffer to Parquet when full
        if len(buffer) >= CHUNK_SIZE * N_CHANNELS:
            table = records_to_table(buffer)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema,
                                          compression="snappy")
            writer.write_table(table)
            buffer.clear()
            gc.collect()
    
    # Flush remaining
    if buffer:
        table = records_to_table(buffer)
        if writer is None:
            writer = pq.ParquetWriter(str(out_path), table.schema,
                                      compression="snappy")
        writer.write_table(table)
        buffer.clear()
    
    if writer:
        writer.close()
    
    out_size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0
    print(f"\r  ✅ Done: {rows_read:,} epochs → {rows_kept:,} channel records "
          f"| Parquet: {out_size_mb:.0f} MB")
    
    return out_path


def run_ingestion(raw_dir: Path, output_dir: Path, start_gps_week: int = GPS_WEEK_START):
    """Discover and process all Excel files in raw_dir."""
    xlsx_files = sorted(
        list(raw_dir.glob("*.xlsx")) + list(raw_dir.glob("*.xls"))
    )
    
    if not xlsx_files:
        print(f"\n⚠️  No Excel files found in: {raw_dir.resolve()}")
        print(f"   Drop your .xlsx files into that folder and re-run.")
        return []
    
    print(f"\n🛰️  CELESTIS — GNSS Telemetry Ingestor")
    print(f"   Files found:   {len(xlsx_files)}")
    print(f"   Starting week: GPS {start_gps_week} "
          f"(Sun Jan 4, 2026)")
    print(f"   Output dir:    {output_dir.resolve()}")
    
    parquet_files = []
    current_week  = start_gps_week
    
    for xlsx_path in xlsx_files:
        try:
            out = process_file(xlsx_path, output_dir, start_gps_week=current_week)
            parquet_files.append(out)
            # Each subsequent file may start on the next GPS week
            # (converter handles rollovers mid-file automatically)
        except Exception as e:
            print(f"\n  ❌ Failed on {xlsx_path.name}: {e}")
            import traceback; traceback.print_exc()
    
    total_size = sum(p.stat().st_size for p in parquet_files if p.exists()) / 1e9
    print(f"\n✅ Ingestion complete.")
    print(f"   Parquet files: {len(parquet_files)}")
    print(f"   Total size:    {total_size:.2f} GB")
    print(f"   Next step:     python data_pipeline/preprocess.py")
    return parquet_files


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CELESTIS GNSS Telemetry Ingestor")
    parser.add_argument("--input",    default=str(RAW_DIR),    help="Dir with .xlsx files")
    parser.add_argument("--output",   default=str(OUTPUT_DIR), help="Output Parquet dir")
    parser.add_argument("--gps_week", default=GPS_WEEK_START, type=int,
                        help="Starting GPS week (default: 2347 = Jan 4, 2026)")
    args = parser.parse_args()
    run_ingestion(Path(args.input), Path(args.output), args.gps_week)
