"""Full storm analysis — write to file to avoid encoding issues."""
import sys
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

out = []

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

out.append("=== IONO TEC BY SESSION DATE ===")
for date, row in summary.iterrows():
    out.append("  {} | rows={:5d} | iono_mean={:.4f}m | iono_max={:.4f}m | spike_max={:.4f}m | clk_drift={:.1f}".format(
        date, int(row['rows']), row['iono_mean'], row['iono_max'],
        row['iono_spike_max'], row['clock_drift_abs']))

quiet_dates = [d for d in summary.index if d <= '17_01_2025']
storm_dates = [d for d in summary.index if d >= '20_01_2025']
quiet_iono  = summary.loc[quiet_dates, 'iono_mean'].mean()
storm_iono  = summary.loc[storm_dates, 'iono_mean'].mean()

out.append("")
out.append("=== STORM ANALYSIS: Jan 18 2026 X-class CME (1700 km/s, 25h travel) ===")
out.append("  Quiet baseline  (Jan 02-17): {:.4f} m avg iono = {:.1f} TECU".format(quiet_iono, quiet_iono/0.162))
out.append("  Storm window    (Jan 20-30): {:.4f} m avg iono = {:.1f} TECU".format(storm_iono, storm_iono/0.162))
out.append("  Elevation vs baseline:       +{:.4f}m  ({:.1f}% increase)".format(
    storm_iono - quiet_iono, (storm_iono/quiet_iono - 1)*100))

peak_idx  = df['iono_tec_proxy_m'].idxmax()
spike_idx = df['iono_spike'].idxmax()
clk_idx   = df['clock_drift_abs'].idxmax()

out.append("")
out.append("  Peak iono TEC   : {:.4f} m = {:.1f} TECU  @ session {}".format(
    df['iono_tec_proxy_m'].max(), df['iono_tec_proxy_m'].max()/0.162,
    df.loc[peak_idx, 'session_date']))
out.append("  Peak iono spike : {:.4f} m  @ session {}".format(
    df['iono_spike'].max(), df.loc[spike_idx, 'session_date']))
out.append("  Peak clock drift: {:.2f} ns/s  @ session {}".format(
    df['clock_drift_abs'].max(), df.loc[clk_idx, 'session_date']))

out.append("")
out.append("=== STORM SESSION COMPARISON ===")
for date, row in summary.iterrows():
    tecu = row['iono_mean'] / 0.162
    is_storm = " *** STORM PEAK ***" if row['iono_mean'] == summary['iono_mean'].max() else ""
    is_window = " (storm window)" if date >= '20_01_2025' else " (quiet)"
    out.append("  {} : {:.1f} TECU{}{}".format(date, tecu, is_window, is_storm))

result = "\n".join(out)
print(result)

# Also save to file
with open('d:/inceptrix_sat/data_pipeline/storm_analysis.txt', 'w') as f:
    f.write(result)
print("\nSaved to data_pipeline/storm_analysis.txt")
