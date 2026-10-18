"""
CELESTIS — Quick Data Profiler
================================
Run this FIRST on any new Excel file to understand its structure
before running the full 8.7GB pipeline.

Usage:
    python data_pipeline/profile_data.py path/to/your_file.xlsx
    python data_pipeline/profile_data.py data_pipeline/raw/january.xlsx
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path

def profile_gnss_file(filepath: str, n_rows: int = 500):
    """
    Quickly profile a GNSS Excel file without loading all 8.7GB.
    Reads first n_rows to understand structure, stats, and data quality.
    """
    path = Path(filepath)
    if not path.exists():
        print(f"❌ File not found: {filepath}")
        return
    
    size_mb = path.stat().st_size / 1e6
    print(f"\n🔍 CELESTIS Data Profiler")
    print(f"{'='*60}")
    print(f"File:   {path.name}")
    print(f"Size:   {size_mb:.1f} MB")
    
    # Load sample
    print(f"\nLoading first {n_rows} rows...")
    df = pd.read_excel(filepath, nrows=n_rows, engine='openpyxl')
    
    print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Timestamp analysis
    towc_cols = [c for c in df.columns if "TOWC" in str(c).upper()]
    print(f"\n⏱️  Timestamp columns: {len(towc_cols)}")
    for tc in towc_cols[:3]:
        vals = df[tc].dropna()
        if len(vals) > 1:
            diff = vals.diff().dropna()
            print(f"   {tc}: min={vals.min():.1f}s  max={vals.max():.1f}s  "
                  f"Δt_median={diff.median():.2f}s (sample rate)")
    
    # Channel coverage
    print(f"\n📡 Channel Analysis:")
    for n in range(1, 13):
        prn_col  = f"PRN-{n}"
        cno_col  = f"C/NO-{n} (dB-Hz)"
        rej_col  = f"Reject Code-{n}"
        
        if prn_col not in df.columns:
            continue
        
        prn_counts = df[prn_col].notna().sum()
        if cno_col in df.columns:
            cno_mean = df[cno_col].mean()
            cno_str  = f"  C/N0={cno_mean:.1f} dB-Hz"
        else:
            cno_str = ""
        
        if rej_col in df.columns:
            good_pct = 100 * (df[rej_col] == 0).mean()
            rej_str  = f"  good={good_pct:.0f}%"
        else:
            rej_str = ""
        
        print(f"   Chan {n:>2}: {prn_counts:>4} readings{cno_str}{rej_str}")
    
    # Iono delay statistics
    iono_cols = [c for c in df.columns if "Iono Delay" in str(c)]
    if iono_cols:
        print(f"\n🌐 Ionospheric Delay Stats (space weather proxy):")
        all_iono = df[iono_cols].values.flatten()
        all_iono = all_iono[~np.isnan(all_iono)]
        if len(all_iono) > 0:
            print(f"   Mean: {np.mean(all_iono):.2f} m")
            print(f"   Max:  {np.max(all_iono):.2f} m")
            print(f"   Std:  {np.std(all_iono):.2f} m")
    
    # Pseudorange stats
    pr_cols = [c for c in df.columns if c.startswith("PR-") and "(m)" in c]
    if pr_cols:
        print(f"\n📏 Pseudorange Statistics:")
        all_pr = df[pr_cols].values.flatten()
        all_pr = all_pr[~np.isnan(all_pr)]
        if len(all_pr) > 0:
            print(f"   Range: {np.min(all_pr)/1e6:.1f} – {np.max(all_pr)/1e6:.1f} Mm")
            print(f"   Mean:  {np.mean(all_pr)/1e6:.1f} Mm")
    
    # Data completeness
    print(f"\n📊 Data Completeness:")
    key_cols = [c for c in df.columns if any(k in str(c) for k in 
                ["PRN", "C/NO", "PR-", "Iono", "Doppler"])]
    for col in key_cols[:15]:
        pct = 100 * df[col].notna().mean()
        bar = "█" * int(pct/10) + "░" * (10 - int(pct/10))
        print(f"   {col[:35]:35s} [{bar}] {pct:.0f}%")
    
    # Extrapolate full file size
    rows_per_mb   = n_rows / size_mb
    est_total_rows = int(rows_per_mb * size_mb)
    sampling_rate  = df[towc_cols[0]].diff().median() if towc_cols else 1.0
    duration_days  = (est_total_rows * sampling_rate) / 86400
    
    print(f"\n📈 Estimated Full Dataset:")
    print(f"   Total rows:   ~{est_total_rows:,}")
    print(f"   Sample rate:  {sampling_rate:.1f}s")
    print(f"   Duration:     ~{duration_days:.1f} days")
    print(f"   Features/row: {df.shape[1]}")
    print(f"\n✅ Profile complete. Drop all Excel files into:")
    print(f"   d:\\inceptrix_sat\\data_pipeline\\raw\\")
    print(f"   Then run: python data_pipeline/ingest.py\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Try to find any xlsx in raw/
        raw_dir = Path("data_pipeline/raw")
        xlsx_files = list(raw_dir.glob("*.xlsx"))
        if xlsx_files:
            profile_gnss_file(str(xlsx_files[0]))
        else:
            print("Usage: python data_pipeline/profile_data.py <path_to_excel_file.xlsx>")
            print("Or drop Excel files into data_pipeline/raw/ and run without arguments")
    else:
        profile_gnss_file(sys.argv[1])
