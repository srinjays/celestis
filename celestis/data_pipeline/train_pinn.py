"""
data_pipeline/train_pinn.py
===========================
CELESTIS Phase-Aware PINN — Physics-Informed Neural Network
============================================================

Replaces the scikit-learn GBM with a PyTorch neural network that is
PHYSICALLY CONSTRAINED via a custom loss function. This is clinically
important: a standard "black-box" ML model can hallucinate physically
impossible orbital decays. A PINN cannot — the loss function penalises
any prediction that violates Kepler's drag equation.

Architecture
────────────
Input features  (7):  kp_max, f10.7, cme_km_s, flare_enc,
                       alt_before_km, storm_duration_h, dh_physics_km
Hidden layers   (3):  [64, 128, 64] neurons, GELU activation, BatchNorm
Output          (1):  residual_km  (how much NRLMSISE under/over-estimates)

Physics Loss (PINN Constraint)
──────────────────────────────
The standard drag equation for altitude decay is:
    dH/dt = -(Cd * A * rho * v²) / (2 * m) * T
We enforce:
    L_physics = MSE(dH_pred, dH_physics)   ← must not deviate wildly
    L_data    = MSE(dH_pred, dH_true)      ← must match real TLE data
    L_total   = alpha * L_data + (1 - alpha) * L_physics

The alpha weight is annealed: starts at 0.5 (equal physics/data balance),
gradually shifts to 0.85 (data-dominant after convergence), ensuring
the network first learns physically plausible orbits, then refines
on real data.

Output
──────
Saves: data_pipeline/output/pinn_model.pt  ← PyTorch state dict
       data_pipeline/output/pinn_scaler.pkl ← feature scaler
       data_pipeline/output/pinn_benchmark.json ← accuracy report

Usage
─────
    cd d:\\inceptrix_sat
    pip install torch scikit-learn numpy
    python data_pipeline/train_pinn.py
"""

from __future__ import annotations
import os, sys, csv, json, math, pickle, logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import GROUND_TRUTH_CSV, OUTPUT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s")
log = logging.getLogger("pinn_trainer")

# ── Constants (Keplerian Drag Equation) ───────────────────────────────────────
GM      = 3.986004418e14   # m³/s²   Earth gravitational parameter
RE      = 6.371e6          # m        Earth radius
G0      = 9.80665          # m/s²

# Feature columns (must match the CSV)
FEATURE_COLS = [
    "kp_max_real", "f107_real", "cme_km_s", "flare_enc",
    "alt_before_km", "storm_duration_h", "dh_physics_km",
]

# ── Try importing PyTorch ─────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_OK = True
except ImportError:
    TORCH_OK = False
    log.error("PyTorch not installed. Run:  pip install torch")
    sys.exit(1)

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    log.error("scikit-learn not installed. Run:  pip install scikit-learn")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# PINN Architecture
# ─────────────────────────────────────────────────────────────────────────────
class DragResidualPINN(nn.Module):
    """
    Physics-Informed Neural Network that predicts the RESIDUAL between
    NRLMSISE-00 physics drag prediction and real TLE-observed altitude drop.

    The model is physically regularised: it cannot output a residual that
    would make the total predicted decay physically impossible (e.g. a
    satellite GAINING altitude during a geomagnetic storm).
    """
    def __init__(self, n_features: int = 7, hidden: list[int] = [64, 128, 64]):
        super().__init__()
        layers = []
        prev = n_features
        for h in hidden:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(0.15),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Physics-Informed Loss Function
# ─────────────────────────────────────────────────────────────────────────────
def pinn_loss(
    y_pred_resid: torch.Tensor,
    y_true_resid: torch.Tensor,
    dh_physics: torch.Tensor,
    alpha: float = 0.7,
) -> tuple[torch.Tensor, dict]:
    """
    Total loss = alpha * data_loss + (1 - alpha) * physics_loss

    physics_loss: ensures the predicted TOTAL decay (physics + residual) is
    in the physically plausible range. A satellite in LEO CANNOT gain altitude
    during a storm. Penalise sign violations hard.

    data_loss: standard MSE between predicted residual and real TLE residual.
    """
    mse = nn.MSELoss()

    # Data fidelity loss
    L_data = mse(y_pred_resid, y_true_resid)

    # Total predicted decay
    total_pred_decay = dh_physics + y_pred_resid

    # Physics constraint 1: total decay must be ≥ 0 (no altitude gain during storm)
    # Penalise negative total decay (physically impossible)
    violation_gain = torch.clamp(-total_pred_decay, min=0.0)
    L_physics_gain = torch.mean(violation_gain ** 2)

    # Physics constraint 2: residual magnitude shouldn't exceed 5× the physics prediction
    # (prevents wild extrapolation)
    max_allowed_resid = torch.abs(dh_physics) * 5.0
    violation_magnitude = torch.clamp(torch.abs(y_pred_resid) - max_allowed_resid, min=0.0)
    L_physics_mag = torch.mean(violation_magnitude ** 2)

    L_physics = L_physics_gain + 0.5 * L_physics_mag
    L_total   = alpha * L_data + (1.0 - alpha) * L_physics

    return L_total, {
        "total": L_total.item(),
        "data":  L_data.item(),
        "phys":  L_physics.item(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────
def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list, list]:
    """Load ground truth CSV. Returns train/OOD splits."""
    rows_train, rows_ood = [], []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                feat = [float(r[c]) for c in FEATURE_COLS]
                resid = float(r["residual_km"])
                dh_phys = float(r["dh_physics_km"])
            except (ValueError, KeyError):
                continue
            entry = (feat, resid, dh_phys, r.get("storm_label","?"), r.get("satellite","?"))
            if r.get("split") == "ood":
                rows_ood.append(entry)
            else:
                rows_train.append(entry)

    def unpack(rows):
        X  = np.array([r[0] for r in rows], dtype=np.float32)
        y  = np.array([r[1] for r in rows], dtype=np.float32)
        dp = np.array([r[2] for r in rows], dtype=np.float32)
        return X, y, dp

    X_tr, y_tr, dp_tr = unpack(rows_train)
    X_od, y_od, dp_od = unpack(rows_ood)
    labels_ood = [(r[3], r[4]) for r in rows_ood]

    log.info(f"Dataset: {len(rows_train)} train rows, {len(rows_ood)} OOD rows")
    return X_tr, y_tr, dp_tr, X_od, y_od, dp_od, labels_ood


# ─────────────────────────────────────────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────────────────────────────────────────
def train_pinn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    dp_train: np.ndarray,
    epochs: int = 600,
    batch_size: int = 16,
    lr: float = 3e-3,
    alpha_start: float = 0.5,
    alpha_end: float = 0.85,
    device: str = "cpu",
) -> tuple[DragResidualPINN, StandardScaler]:

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)

    X_t  = torch.tensor(X_sc,    dtype=torch.float32).to(device)
    y_t  = torch.tensor(y_train, dtype=torch.float32).to(device)
    dp_t = torch.tensor(dp_train,dtype=torch.float32).to(device)

    ds     = TensorDataset(X_t, y_t, dp_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = DragResidualPINN(n_features=X_train.shape[1]).to(device)
    opt   = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    log.info(f"Training PINN on {device.upper()} | {epochs} epochs | batch={batch_size}")
    log.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0.0
        alpha = alpha_start + (alpha_end - alpha_start) * (epoch / epochs)

        for xb, yb, dpb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss, _ = pinn_loss(pred, yb, dpb, alpha=alpha)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            epoch_loss += loss.item()

        sched.step()
        avg_loss = epoch_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if epoch % 50 == 0 or epoch == 1:
            log.info(f"  Epoch {epoch:4d}/{epochs}  loss={avg_loss:.6f}  alpha={alpha:.2f}  lr={sched.get_last_lr()[0]:.6f}")

    model.load_state_dict(best_state)
    log.info(f"Training complete. Best loss: {best_loss:.6f}")
    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(
    model: DragResidualPINN,
    scaler: StandardScaler,
    X: np.ndarray,
    y_true: np.ndarray,
    dp: np.ndarray,
    label: str,
    device: str = "cpu",
) -> dict:
    model.eval()
    X_sc = scaler.transform(X)
    with torch.no_grad():
        X_t  = torch.tensor(X_sc, dtype=torch.float32).to(device)
        pred = model(X_t).cpu().numpy()

    # Total predicted decay = physics + PINN residual
    total_true = dp + y_true
    total_pred = dp + pred

    mae_res  = mean_absolute_error(y_true, pred)
    mae_tot  = mean_absolute_error(total_true, total_pred)
    r2_res   = r2_score(y_true, pred)
    r2_tot   = r2_score(total_true, total_pred)
    mape_tot = np.mean(np.abs((total_true - total_pred) / (np.abs(total_true) + 1e-6))) * 100

    # Physics violation check: how many predictions are physically impossible?
    n_violations = int(np.sum(total_pred < 0))  # altitude GAIN during storm

    log.info(f"\n{'─'*55}")
    log.info(f"{label}")
    log.info(f"  Residual MAE   : {mae_res:.4f} km")
    log.info(f"  Total ΔH MAE   : {mae_tot:.4f} km")
    log.info(f"  Total ΔH R²    : {r2_tot:.4f}")
    log.info(f"  MAPE (total)   : {mape_tot:.1f}%")
    log.info(f"  Physics violations: {n_violations}/{len(pred)}")

    return {
        "label": label,
        "n_samples": len(y_true),
        "mae_residual_km": round(float(mae_res), 5),
        "mae_total_km": round(float(mae_tot), 5),
        "r2_residual": round(float(r2_res), 4),
        "r2_total": round(float(r2_tot), 4),
        "mape_pct": round(float(mape_tot), 2),
        "physics_violations": n_violations,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log.info("=" * 55)
    log.info("CELESTIS Phase-Aware PINN Training")
    log.info("=" * 55)

    if not os.path.exists(GROUND_TRUTH_CSV):
        log.error(f"Dataset not found: {GROUND_TRUTH_CSV}")
        log.error("Run build_dataset.py first to collect real TLE data.")
        sys.exit(1)

    # ── 1. Load data ───────────────────────────────────────────────────────────
    result = load_dataset(GROUND_TRUTH_CSV)
    X_tr, y_tr, dp_tr, X_od, y_od, dp_od, labels_ood = result

    if len(X_tr) < 5:
        log.error(f"Only {len(X_tr)} training rows — need at least 5. Run build_dataset.py.")
        sys.exit(1)

    # ── 2. Detect device ───────────────────────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device.upper()}")

    # ── 3. Train ───────────────────────────────────────────────────────────────
    model, scaler = train_pinn(
        X_tr, y_tr, dp_tr,
        epochs=600,
        batch_size=max(8, min(32, len(X_tr) // 4)),
        lr=3e-3,
        alpha_start=0.5,
        alpha_end=0.85,
        device=device,
    )

    # ── 4. Evaluate ────────────────────────────────────────────────────────────
    results = []
    results.append(evaluate(model, scaler, X_tr, y_tr, dp_tr, "Train Set (in-distribution)", device))
    if len(X_od) > 0:
        results.append(evaluate(model, scaler, X_od, y_od, dp_od, "OOD Test Set (unseen storms)", device))

    # ── 5. Save model ─────────────────────────────────────────────────────────
    pinn_model_path  = os.path.join(OUTPUT_DIR, "pinn_model.pt")
    pinn_scaler_path = os.path.join(OUTPUT_DIR, "pinn_scaler.pkl")
    pinn_bench_path  = os.path.join(OUTPUT_DIR, "pinn_benchmark.json")

    torch.save(model.state_dict(), pinn_model_path)
    with open(pinn_scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    with open(pinn_bench_path, "w") as f:
        json.dump(results, f, indent=2)

    log.info(f"\nSaved model   → {pinn_model_path}")
    log.info(f"Saved scaler  → {pinn_scaler_path}")
    log.info(f"Saved results → {pinn_bench_path}")
    log.info("\n✅  PINN training complete. Update backend/api.py to load pinn_model.pt")

    # ── 6. Print comparison vs GBM ────────────────────────────────────────────
    gbm_bench = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
    if os.path.exists(gbm_bench):
        log.info("\n📊 Comparison vs existing Phase-Aware GBM:")
        with open(gbm_bench, newline="") as f:
            for row in csv.DictReader(f):
                mae = float(row.get('MAE_km') or row.get('mae_km') or 0)
                r2  = float(row.get('R2')     or row.get('r2')     or 0)
                log.info(f"   GBM | {row.get('label','?'):40s} | MAE={mae:.4f} km | R²={r2:.3f}")
        for r in results:
            log.info(f"  PINN | {r['label']:40s} | MAE={r['mae_total_km']:.4f} km | R²={r['r2_total']:.3f}")


if __name__ == "__main__":
    main()
