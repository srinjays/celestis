"""Run enrich_and_retrain with full error output captured to file."""
import sys, io, traceback

sys.path.insert(0, 'd:/inceptrix_sat/celestis/data_pipeline')
sys.path.insert(0, 'd:/inceptrix_sat/celestis')

log_lines = []

def log(msg):
    print(msg)
    log_lines.append(msg)

try:
    import numpy as np
    import pandas as pd
    import json, pickle
    from pathlib import Path

    ROOT         = Path('d:/inceptrix_sat/celestis')
    GROUND_TRUTH = ROOT / "data_pipeline/output/ground_truth_dataset.csv"
    ENRICHED_CSV = ROOT / "data_pipeline/output/ground_truth_enriched.csv"
    OUTPUT_DIR   = ROOT / "data_pipeline/output"

    BANGALORE_QUIET_TECU   = 62.7
    BANGALORE_QUIET_IONO_M = 10.164
    L1_TECU_TO_METERS      = 0.162

    def kp_f107_to_tec(kp, f107, storm_duration_h=24.0):
        f107_ref  = 120.0
        tec_quiet = BANGALORE_QUIET_TECU * (f107 / f107_ref) ** 0.6
        if kp <= 4:
            ratio = 1.0 + 0.008 * kp
        elif kp <= 6:
            ratio = 1.0 + 0.012 * kp + 0.002 * (kp - 4) ** 2
        elif kp <= 8:
            ratio = 1.0 + 0.01 * kp + 0.005 * (kp - 6) ** 2
        else:
            ratio = 1.0 + 0.10 + 0.02 * (kp - 6) ** 2
        dur_factor = min(1.0 + 0.005 * max(0, storm_duration_h - 12), 1.3)
        ratio *= dur_factor
        tec_storm = tec_quiet * ratio
        return {
            "tec_quiet_tecu": round(tec_quiet, 2),
            "tec_storm_tecu": round(tec_storm, 2),
            "tec_delta_tecu": round(tec_storm - tec_quiet, 2),
            "tec_ratio":      round(ratio, 4),
            "iono_storm_m":   round(tec_storm * L1_TECU_TO_METERS, 4),
            "iono_delta_m":   round((tec_storm - tec_quiet) * L1_TECU_TO_METERS, 4),
        }

    # Step 1: Load and enrich
    log("Loading ground truth...")
    df = pd.read_csv(GROUND_TRUTH)
    log("  Rows: {}  Cols: {}".format(len(df), len(df.columns)))

    tec_records = []
    for _, row in df.iterrows():
        tec = kp_f107_to_tec(float(row['kp_max_real']), float(row['f107_real']),
                               float(row['storm_duration_h']))
        tec_records.append(tec)

    tec_df   = pd.DataFrame(tec_records)
    enriched = pd.concat([df.reset_index(drop=True), tec_df], axis=1)
    enriched.to_csv(ENRICHED_CSV, index=False)
    log("Enriched CSV saved: {}".format(ENRICHED_CSV))
    log("Columns now: {}".format(list(enriched.columns)))

    # TEC summary by Kp bucket
    log("\nTEC by Kp bucket:")
    bins   = [0, 4, 6, 7.5, 9.1]
    labels = ['G1(Kp<4)', 'G2(Kp4-6)', 'G3(Kp6-7.5)', 'G4-G5(Kp>7.5)']
    enriched['kp_bin'] = pd.cut(enriched['kp_max_real'], bins=bins, labels=labels)
    for label, grp in enriched.groupby('kp_bin', observed=True):
        log("  {}: n={:3d} | quiet={:.1f}T | storm={:.1f}T | ratio={:.3f} | iono_storm={:.3f}m".format(
            label, len(grp),
            grp['tec_quiet_tecu'].mean(),
            grp['tec_storm_tecu'].mean(),
            grp['tec_ratio'].mean(),
            grp['iono_storm_m'].mean()))

    # Step 2: Train
    log("\nStarting PINN training...")
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score

    ALL_FEATURES = [
        "kp_max_real", "f107_real", "cme_km_s", "flare_enc",
        "alt_before_km", "storm_duration_h", "dh_physics_km",
        "tec_storm_tecu", "tec_delta_tecu", "tec_ratio", "iono_storm_m",
    ]
    TARGET = "residual_km"

    train_df = enriched[enriched['split'] == 'train'].dropna(subset=ALL_FEATURES + [TARGET])
    ood_df   = enriched[enriched['split'] == 'ood'].dropna(subset=ALL_FEATURES + [TARGET])

    log("  Train rows: {}  OOD rows: {}".format(len(train_df), len(ood_df)))

    X_train  = train_df[ALL_FEATURES].values.astype(np.float32)
    y_train  = train_df[TARGET].values.astype(np.float32)
    dp_train = train_df['dh_physics_km'].values.astype(np.float32)
    X_ood    = ood_df[ALL_FEATURES].values.astype(np.float32)
    y_ood    = ood_df[TARGET].values.astype(np.float32)
    dp_ood   = ood_df['dh_physics_km'].values.astype(np.float32)

    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_od_sc = scaler.transform(X_ood)

    class PINN(nn.Module):
        def __init__(self, n):
            super().__init__()
            layers = []
            prev = n
            for h in [64, 128, 64]:
                layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(0.15)]
                prev = h
            layers.append(nn.Linear(prev, 1))
            self.net = nn.Sequential(*layers)
        def forward(self, x):
            return self.net(x).squeeze(-1)

    def pinn_loss(pred, y_true, dh_phys, alpha):
        L_d = nn.MSELoss()(pred, y_true)
        tot = dh_phys + pred
        L_p = torch.mean(torch.clamp(-tot, min=0.0) ** 2)
        L_p += 0.5 * torch.mean(torch.clamp(torch.abs(pred) - torch.abs(dh_phys) * 5, min=0.0) ** 2)
        return alpha * L_d + (1 - alpha) * L_p

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = PINN(len(ALL_FEATURES)).to(device)
    opt    = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    sched  = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=600, eta_min=1e-5)

    X_t  = torch.tensor(X_tr_sc, dtype=torch.float32).to(device)
    y_t  = torch.tensor(y_train,  dtype=torch.float32).to(device)
    dp_t = torch.tensor(dp_train, dtype=torch.float32).to(device)

    batch  = max(8, min(32, len(X_train) // 4))
    loader = DataLoader(TensorDataset(X_t, y_t, dp_t), batch_size=batch,
                        shuffle=True, drop_last=True)

    log("  Device: {}  Batch: {}  Features: {}".format(device.upper(), batch, len(ALL_FEATURES)))
    log("{:>6} {:>12} {:>8}".format("Epoch", "Loss", "Alpha"))

    best_loss, best_state = float('inf'), None
    for epoch in range(1, 601):
        model.train()
        alpha = 0.5 + 0.35 * (epoch / 600)
        ep_loss = 0.0
        for xb, yb, dpb in loader:
            opt.zero_grad()
            loss = pinn_loss(model(xb), yb, dpb, alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            ep_loss += loss.item()
        sched.step()
        avg = ep_loss / len(loader)
        if avg < best_loss:
            best_loss  = avg
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 100 == 0 or epoch == 1:
            log("{:>6} {:>12.6f} {:>8.3f}".format(epoch, avg, alpha))

    model.load_state_dict(best_state)

    def evaluate(X_sc, y_true, dp, label):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(X_sc, dtype=torch.float32).to(device)).cpu().numpy()
        tot_true = dp + y_true
        tot_pred = dp + pred
        mae  = mean_absolute_error(tot_true, tot_pred)
        r2   = r2_score(tot_true, tot_pred)
        viols = int(np.sum(tot_pred < 0))
        log("  {}: MAE={:.5f}km  R2={:.4f}  violations={}/{}".format(
            label, mae, r2, viols, len(pred)))
        return {"label": label, "mae_km": round(float(mae),5),
                "r2": round(float(r2),4), "violations": viols}

    log("\n=== EVALUATION (TEC-ENRICHED PINN) ===")
    results = []
    results.append(evaluate(X_tr_sc, y_train, dp_train, "Train"))
    if len(X_ood) > 0:
        results.append(evaluate(X_od_sc, y_ood, dp_ood, "OOD"))

    baseline_path = OUTPUT_DIR / "pinn_benchmark.json"
    if baseline_path.exists():
        with open(baseline_path) as f:
            baseline = json.load(f)
        log("\n=== COMPARISON vs BASELINE ===")
        for b, r in zip(baseline, results):
            improvement = r['r2'] - b.get('r2_total', 0)
            log("  {}: OLD R2={:.4f}  NEW R2={:.4f}  ({:+.4f})".format(
                r['label'][:20], b.get('r2_total', 0), r['r2'], improvement))

    # Save
    torch.save({
        "state_dict":  model.state_dict(),
        "features":    ALL_FEATURES,
        "n_features":  len(ALL_FEATURES),
    }, OUTPUT_DIR / "pinn_model_tec.pt")

    with open(OUTPUT_DIR / "pinn_scaler_tec.pkl", 'wb') as f:
        pickle.dump(scaler, f)

    with open(OUTPUT_DIR / "pinn_benchmark_tec.json", 'w') as f:
        json.dump(results, f, indent=2)

    log("\nSaved pinn_model_tec.pt")
    log("Saved pinn_benchmark_tec.json")
    log("\nDONE. Next: update backend/api.py to load pinn_model_tec.pt")

except Exception:
    log("\nERROR:")
    log(traceback.format_exc())

with open('d:/inceptrix_sat/data_pipeline/retrain_log.txt', 'w') as f:
    f.write('\n'.join(log_lines))
print("\nFull log saved to data_pipeline/retrain_log.txt")
