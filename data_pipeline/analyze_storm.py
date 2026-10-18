"""Analyze Jan 2026 storm signature in GNSS iono data."""
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_parquet('d:/inceptrix_sat/data_pipeline/features/jan2025_pinn_ready.parquet')

summary = df.groupby('session_date').agg(
    rows=('utc_time','count'),
    iono_mean=('iono_tec_proxy_m','mean'),
    iono_max=('iono_tec_proxy_m','max'),
    iono_std=('iono_tec_proxy_m','std'),
    iono_spike_max=('iono_spike','max'),
    clock_drift_abs=('clock_drift_abs','mean'),
    cno=('cno_composite','mean'),
).round(4)

print("=== IONO TEC BY SESSION DATE ===")
print(summary.to_string())

# Quiet baseline: Jan 2-17 (before storm)
quiet_dates = [d for d in summary.index if d <= '17_01_2025']
storm_dates = [d for d in summary.index if d >= '20_01_2025']
quiet_iono  = summary.loc[quiet_dates, 'iono_mean'].mean()
storm_iono  = summary.loc[storm_dates, 'iono_mean'].mean()

print("\n=== STORM ANALYSIS — Jan 18 2026 X-class CME ===")
print("CME speed: ~1700 km/s | Travel time: ~25h | Impact: Jan 19 2026")
print()
print("Quiet baseline  (Jan 2-17):  {:.4f} m avg iono".format(quiet_iono))
print("Storm window    (Jan 20-30): {:.4f} m avg iono".format(storm_iono))
print("Storm elevation:             +{:.4f} m  ({:.1f}% increase)".format(
    storm_iono - quiet_iono, (storm_iono / quiet_iono - 1) * 100))

peak_idx  = df['iono_tec_proxy_m'].idxmax()
spike_idx = df['iono_spike'].idxmax()
clk_idx   = df['clock_drift_abs'].idxmax()

print()
print("Peak iono TEC:   {:.4f} m  @ {}".format(
    df['iono_tec_proxy_m'].max(), df.loc[peak_idx, 'session_date']))
print("Peak iono spike: {:.4f} m  @ {}".format(
    df['iono_spike'].max(), df.loc[spike_idx, 'session_date']))
print("Peak clk drift:  {:.2f}  ns/s  @ {}".format(
    df['clock_drift_abs'].max(), df.loc[clk_idx, 'session_date']))

# Estimated TEC in TECU (1 TECU = 0.162 m delay at L1)
print()
print("=== TEC ESTIMATES (TECU, L1 vertical equivalent) ===")
for date, row in summary.iterrows():
    tecu = row['iono_mean'] / 0.162
    flag = " <-- STORM PEAK" if row['iono_mean'] == summary['iono_mean'].max() else ""
    print("  {}: {:.1f} TECU{}".format(date, tecu, flag))
