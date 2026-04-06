import csv, json

print("=== BENCHMARK RESULTS (Real TLE Ground Truth) ===\n")
rows = list(csv.DictReader(open("output/benchmark_results.csv")))
header = f"{'Method':<44} {'N':>4} {'MAE(km)':>9} {'RMSE(km)':>9} {'R2':>7} {'vs Physics':>12}"
print(header)
print("-"*85)
for r in rows:
    print(f"{r['label']:<44} {r['N']:>4} {r['MAE_km']:>9} {r['RMSE_km']:>9} {r['R2']:>7} {r['pct_improvement_vs_physics']:>10}%")

print()
print("=== 5-FOLD CROSS-VALIDATION ===")
cv = json.load(open("output/crossval_results.json"))
for k, v in cv.items():
    print(f"  {k}: {v}")
