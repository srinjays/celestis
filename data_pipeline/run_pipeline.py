"""
CELESTIS — Master Data Pipeline Runner
========================================
One command to go from raw Excel → trained PINN model.

Usage:
    python data_pipeline/run_pipeline.py               # full pipeline
    python data_pipeline/run_pipeline.py --skip_ingest # if already ingested
    python data_pipeline/run_pipeline.py --only_ingest # just convert Excel
    python data_pipeline/run_pipeline.py --only_train  # just retrain model
    python data_pipeline/run_pipeline.py --profile     # profile raw files only
"""

import argparse
import time
from pathlib import Path

RAW_DIR       = Path("data_pipeline/raw")
PROCESSED_DIR = Path("data_pipeline/processed")
FEATURES_DIR  = Path("data_pipeline/features")
MODELS_DIR    = Path("data_pipeline/models")

BANNER = """
╔══════════════════════════════════════════════════════════════╗
║          🛰️  CELESTIS — Full Data Pipeline                   ║
║          Inceptrix SAT · Ajinkya Singh · 2026                ║
╚══════════════════════════════════════════════════════════════╝
"""

def hr(label=""):
    width = 65
    if label:
        pad = (width - len(label) - 4) // 2
        print(f"\n{'─'*pad}[ {label} ]{'─'*pad}")
    else:
        print(f"\n{'─'*width}")


def stage(num, total, label):
    print(f"\n{'='*65}")
    print(f"  STAGE {num}/{total}: {label}")
    print(f"{'='*65}")


def check_raw_files():
    xlsx = list(RAW_DIR.glob("*.xlsx")) + list(RAW_DIR.glob("*.xls"))
    if not xlsx:
        print(f"  ⚠️  No Excel files found in {RAW_DIR.resolve()}")
        print(f"\n  👉 DROP your January .xlsx files into:")
        print(f"     {RAW_DIR.resolve()}")
        print(f"  Then re-run this script.\n")
        return False
    
    total_gb = sum(f.stat().st_size for f in xlsx) / 1e9
    print(f"  Found {len(xlsx)} Excel file(s) | Total: {total_gb:.2f} GB")
    for f in xlsx:
        print(f"    📄 {f.name}  ({f.stat().st_size/1e6:.0f} MB)")
    return True


def run_stage_profile():
    """Quick profile of raw files — no processing."""
    print(f"\n{BANNER}")
    hr("PROFILING RAW DATA")
    
    xlsx = list(RAW_DIR.glob("*.xlsx"))
    if not xlsx:
        print(f"No files in {RAW_DIR}")
        return
    
    # Profile the first file
    from data_pipeline.profile_data import profile_gnss_file
    profile_gnss_file(str(xlsx[0]), n_rows=200)


def run_stage_ingest():
    """Stage 1: Excel → Parquet."""
    from data_pipeline.ingest import run_ingestion
    parquet_files = run_ingestion(RAW_DIR, PROCESSED_DIR)
    return parquet_files


def run_stage_preprocess():
    """Stage 2: Parquet GNSS → Orbit timeseries + features."""
    from data_pipeline.preprocess import run_preprocessing
    run_preprocessing(PROCESSED_DIR, FEATURES_DIR)


def run_stage_train(args):
    """Stage 3: Train PINN on orbit features."""
    import sys
    sys.argv = [
        "train_pinn.py",
        f"--epochs={args.epochs}",
        f"--batch={args.batch}",
        f"--lr={args.lr}",
    ]
    from data_pipeline.train_pinn import train_pinn
    train_pinn(args)


def main():
    parser = argparse.ArgumentParser(description="CELESTIS Full Data Pipeline")
    parser.add_argument("--skip_ingest",  action="store_true", help="Skip Excel→Parquet step")
    parser.add_argument("--only_ingest",  action="store_true", help="Only run ingestion")
    parser.add_argument("--only_train",   action="store_true", help="Only run training")
    parser.add_argument("--profile",      action="store_true", help="Profile raw files only")
    parser.add_argument("--epochs",       type=int,   default=600)
    parser.add_argument("--batch",        type=int,   default=512)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--alpha_start",  type=float, default=0.5)
    parser.add_argument("--alpha_end",    type=float, default=0.85)
    args = parser.parse_args()
    
    print(BANNER)
    t0 = time.time()
    
    # ── Profile only ──────────────────────────────────────────────────────────
    if args.profile:
        run_stage_profile()
        return
    
    # ── Only train ────────────────────────────────────────────────────────────
    if args.only_train:
        stage(1, 1, "PINN Training")
        run_stage_train(args)
        return
    
    # ── Check input files ─────────────────────────────────────────────────────
    hr("INPUT FILES")
    if not check_raw_files():
        return
    
    n_stages = 3 if not (args.skip_ingest or args.only_ingest) else 2
    
    # ── Stage 1: Ingest ───────────────────────────────────────────────────────
    if not args.skip_ingest:
        stage(1, n_stages, "INGESTION — Excel → Parquet (streaming, memory-safe)")
        print("  ⏳ This is the slowest step — ~1-3 hours per GB of Excel.")
        print("  💡 Safely Ctrl+C and re-run — already-converted files are skipped.\n")
        t1 = time.time()
        run_stage_ingest()
        print(f"  ⏱️  Ingestion took: {(time.time()-t1)/60:.1f} minutes")
    
    if args.only_ingest:
        print(f"\n✅ Ingestion complete. Run preprocessing next:")
        print(f"   python data_pipeline/run_pipeline.py --skip_ingest")
        return
    
    # ── Stage 2: Preprocess ───────────────────────────────────────────────────
    stage(2, n_stages, "ORBIT DETERMINATION — Pseudoranges → Altitude Timeseries")
    print("  Computing WLS position solutions per epoch...")
    print("  Extracting ionospheric TEC proxies...")
    t2 = time.time()
    run_stage_preprocess()
    print(f"  ⏱️  Preprocessing took: {(time.time()-t2)/60:.1f} minutes")
    
    # ── Stage 3: Train ────────────────────────────────────────────────────────
    stage(3, n_stages, f"PINN TRAINING — Physics-Constrained Neural Network ({args.epochs} epochs)")
    t3 = time.time()
    run_stage_train(args)
    print(f"  ⏱️  Training took: {(time.time()-t3)/60:.1f} minutes")
    
    # ── Summary ───────────────────────────────────────────────────────────────
    total_min = (time.time() - t0) / 60
    print(f"\n{'═'*65}")
    print(f"  🎉 CELESTIS Pipeline Complete!")
    print(f"  Total time: {total_min:.0f} minutes")
    print(f"  Model:      data_pipeline/models/")
    print(f"  Features:   data_pipeline/features/master_orbit_features.parquet")
    print(f"{'═'*65}\n")


if __name__ == "__main__":
    main()
