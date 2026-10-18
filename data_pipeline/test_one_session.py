"""Quick test of one session before full run."""
import sys
sys.path.insert(0, 'd:/inceptrix_sat')
from pathlib import Path
from data_pipeline.preprocess_jan2025 import (
    process_session, quality_filter, engineer_features, PINN_FEATURE_COLS
)

session = Path('D:/Jan2025/Extracted_Files/09_01_2025/THURSDAY/6')
print('Testing on:', session)

df = process_session(session, '09_01_2025')
if df is None:
    print('ERROR: session returned None')
    sys.exit(1)

print(f'Raw rows: {len(df)}, columns: {len(df.columns)}')
df_q = quality_filter(df.copy())
print(f'After quality filter: {len(df_q)}')
df_f = engineer_features(df_q)
keep = [c for c in PINN_FEATURE_COLS if c in df_f.columns]
print(f'PINN features available: {len(keep)}/{len(PINN_FEATURE_COLS)}')
missing = [c for c in PINN_FEATURE_COLS if c not in df_f.columns]
if missing:
    print('Missing:', missing)

print('\nSample PINN feature stats:')
check_cols = ['altitude_km','iono_tec_proxy_m','cno_composite','pdop',
              'n_sats_total','velocity_3d_ms','clock_drift_ns_s']
for c in check_cols:
    if c in df_f.columns:
        print(f'  {c}: mean={df_f[c].mean():.4f}  std={df_f[c].std():.4f}  nulls={df_f[c].isna().sum()}')

print('\nUTC time range:')
print(f'  {df_f["utc_time"].min()} → {df_f["utc_time"].max()}')
print('\nAll columns in output:')
print(df_f.columns.tolist())
