import pandas as pd
from pathlib import Path

print('=== PINN-READY DATASET ===')
df = pd.read_parquet('d:/inceptrix_sat/data_pipeline/features/jan2025_pinn_ready.parquet')
print(f'Shape:    {df.shape}')
print(f'Time:     {df.utc_time.min()} to {df.utc_time.max()}')
print(f'Sessions: {df.session_date.nunique()} unique dates')
print(f'Dates:    {sorted(df.session_date.unique())}')
print()
print('FEATURE STATS:')
for col in df.select_dtypes(include=['float64','float32','int64','int32']).columns:
    print(f'  {col:35s}  mean={df[col].mean():>10.3f}  std={df[col].std():>9.3f}  null%={df[col].isna().mean()*100:>4.1f}%')

print()
pinn_size   = Path('d:/inceptrix_sat/data_pipeline/features/jan2025_pinn_ready.parquet').stat().st_size
master_size = Path('d:/inceptrix_sat/data_pipeline/features/jan2025_master.parquet').stat().st_size
print(f'Parquet size (PINN-ready): {pinn_size/1e6:.1f} MB')
print(f'Parquet size (master):     {master_size/1e6:.1f} MB')
