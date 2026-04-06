"""
data_pipeline/benchmark.py
Trains the phase-aware GBM model on real ground-truth data and produces
the paper-grade benchmark comparison table.

Novel Contribution: Storm-Phase-Aware Residual Correction Model
  Instead of one GBM, we train THREE: one per storm phase
  (sudden_commencement, main_phase, recovery_phase).
  This exploits the physically different density dynamics in each phase
  which a single global model cannot capture.

Benchmark Table (paper Table 2):
  Method                   | MAE (km) | RMSE (km) | R²  | MAPE (%)
  ─────────────────────────┼──────────┼───────────┼─────┼─────────
  Physics-only baseline    |          |           |     |
  Single GBM (prior work)  |          |           |     |
  Phase-Aware GBM (ours)   |          |           |     |
  ─────────────────────────┴──────────┴───────────┴─────┴─────────
  OOD (Starlink 2022)      |          |           |     |
  OOD (Gannon 2024)        |          |           |     |
  OOD (Halloween 2003 A)   |          |           |     |
"""

import os
import sys
import csv
import json
import math
import pickle
import logging
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from config import GROUND_TRUTH_CSV, BENCHMARK_CSV, PHASE_MODEL_PATH, OUTPUT_DIR

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(message)s")

try:
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False
    log.error("scikit-learn not installed. Run: pip install scikit-learn")


FEATURE_COLS = [
    "kp_max_real", "f107_real", "cme_km_s", "flare_enc",
    "alt_before_km", "storm_duration_h", "dh_physics_km",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_csv(path: str) -> list[dict]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _to_float(row: dict, cols: list) -> list[float]:
    return [float(row[c]) for c in cols]


def _metrics(y_true, y_pred, label: str) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae    = mean_absolute_error(y_true, y_pred)
    rmse   = math.sqrt(mean_squared_error(y_true, y_pred))
    r2     = r2_score(y_true, y_pred)
    mape   = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + 1e-6))) * 100
    # 95% confidence interval on MAE via bootstrap
    np.random.seed(42)
    boot_maes = []
    for _ in range(5000):
        idx = np.random.choice(len(y_true), len(y_true), replace=True)
        boot_maes.append(mean_absolute_error(y_true[idx], y_pred[idx]))
    ci_low  = np.percentile(boot_maes, 2.5)
    ci_high = np.percentile(boot_maes, 97.5)
    return {
        "label":   label,
        "N":       len(y_true),
        "MAE_km":  round(mae,  4),
        "RMSE_km": round(rmse, 4),
        "R2":      round(r2,   4),
        "MAPE_pct":round(mape, 2),
        "MAE_CI_95": f"[{ci_low:.4f}, {ci_high:.4f}]",
    }


# ── Phase-Aware Model (Novel Contribution) ────────────────────────────────────

class PhaseAwareResidualModel:
    """
    Storm-phase-decomposed GBM residual correction model.

    Novelty: Instead of one global correction, we train separate GBMs for:
      - sudden_commencement : rapid Kp rise, underestimation by density models
      - main_phase          : peak Kp, highly non-linear density response
      - recovery_phase      : gradual decay, thermospheric cooling lag

    At inference: determine phase from Kp trajectory, apply correct sub-model.
    """
    PHASES = ["sudden_commencement", "main_phase", "recovery_phase"]

    def __init__(self):
        self.models   = {}   # phase → GBM
        self.scalers  = {}   # phase → StandardScaler
        self.global_model  = None
        self.global_scaler = None

    def fit(self, rows_train: list[dict]):
        # ── Global single GBM (prior-work baseline) ───────────────────────
        X_all = np.array([_to_float(r, FEATURE_COLS) for r in rows_train])
        y_all = np.array([float(r["residual_km"]) for r in rows_train])

        self.global_scaler = StandardScaler().fit(X_all)
        Xs = self.global_scaler.transform(X_all)
        self.global_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ).fit(Xs, y_all)

        # ── Phase-specific models ─────────────────────────────────────────
        for phase in self.PHASES:
            phase_rows = [r for r in rows_train if r["storm_phase"] == phase]
            if len(phase_rows) < 3:
                log.warning(f"Phase '{phase}' has only {len(phase_rows)} samples — using global model")
                self.models[phase]  = self.global_model
                self.scalers[phase] = self.global_scaler
                continue

            X = np.array([_to_float(r, FEATURE_COLS) for r in phase_rows])
            y = np.array([float(r["residual_km"]) for r in phase_rows])
            sc = StandardScaler().fit(X)
            Xs_ = sc.transform(X)
            m   = GradientBoostingRegressor(
                n_estimators=150, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=42
            ).fit(Xs_, y)
            self.models[phase]  = m
            self.scalers[phase] = sc
            log.info(f"  Phase '{phase}': trained on {len(phase_rows)} samples")

    def predict_hybrid(self, rows: list[dict], phase_key: str | None = None) -> np.ndarray:
        """Predict total altitude drop = physics + phase-specific ML residual."""
        results = []
        for r in rows:
            phase   = phase_key or r.get("storm_phase", "main_phase")
            m       = self.models.get(phase, self.global_model)
            sc      = self.scalers.get(phase, self.global_scaler)
            X       = np.array([_to_float(r, FEATURE_COLS)])
            res_hat = m.predict(sc.transform(X))[0]
            total   = float(r["dh_physics_km"]) + res_hat
            results.append(total)
        return np.array(results)

    def predict_global(self, rows: list[dict]) -> np.ndarray:
        """Predict using single global GBM (prior-work comparable)."""
        X = np.array([_to_float(r, FEATURE_COLS) for r in rows])
        Xs = self.global_scaler.transform(X)
        res_hat = self.global_model.predict(Xs)
        physics = np.array([float(r["dh_physics_km"]) for r in rows])
        return physics + res_hat

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        log.info(f"Phase-aware model saved → {path}")

    @staticmethod
    def load(path: str) -> "PhaseAwareResidualModel":
        with open(path, "rb") as f:
            return pickle.load(f)


# ── Cross-Validation ──────────────────────────────────────────────────────────

def cross_validate_global(rows: list[dict]) -> dict:
    """5-fold CV on the global GBM model."""
    X = np.array([_to_float(r, FEATURE_COLS) for r in rows])
    y = np.array([float(r["residual_km"]) for r in rows])
    sc = StandardScaler().fit(X)
    Xs = sc.transform(X)
    m  = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                    learning_rate=0.05, random_state=42)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    neg_mae  = cross_val_score(m, Xs, y, cv=cv, scoring="neg_mean_absolute_error")
    neg_rmse = cross_val_score(m, Xs, y, cv=cv, scoring="neg_root_mean_squared_error")
    return {
        "cv_mae_mean":   round(-neg_mae.mean(),  4),
        "cv_mae_std":    round( neg_mae.std(),   4),
        "cv_rmse_mean":  round(-neg_rmse.mean(), 4),
        "cv_rmse_std":   round( neg_rmse.std(),  4),
    }


# ── Main benchmark runner ─────────────────────────────────────────────────────

def run_benchmark():
    if not SKLEARN_OK:
        return

    if not os.path.exists(GROUND_TRUTH_CSV):
        log.error(f"Ground truth CSV not found: {GROUND_TRUTH_CSV}")
        log.error("Run build_dataset.py first")
        return

    rows = _load_csv(GROUND_TRUTH_CSV)
    log.info(f"Loaded {len(rows)} rows from {GROUND_TRUTH_CSV}")

    train_rows = [r for r in rows if r["split"] == "train"]
    ood_rows   = [r for r in rows if r["split"] == "ood"]
    log.info(f"Train: {len(train_rows)}   OOD: {len(ood_rows)}")

    if len(train_rows) < 5:
        log.error("Insufficient training data. Need ≥ 5 samples.")
        return

    y_train_true = np.array([float(r["dh_true_km"])    for r in train_rows])
    y_ood_true   = np.array([float(r["dh_true_km"])    for r in ood_rows])
    y_phys_train = np.array([float(r["dh_physics_km"]) for r in train_rows])
    y_phys_ood   = np.array([float(r["dh_physics_km"]) for r in ood_rows])

    # ── Train phase-aware model ───────────────────────────────────────────
    log.info("Training phase-aware residual model...")
    model = PhaseAwareResidualModel()
    model.fit(train_rows)
    model.save(PHASE_MODEL_PATH)

    # ── 5-fold cross-validation on training set ───────────────────────────
    log.info("Running 5-fold cross-validation...")
    cv_results = cross_validate_global(train_rows)
    log.info(f"  CV MAE  = {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f} km")
    log.info(f"  CV RMSE = {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f} km")

    # ── Generate predictions ──────────────────────────────────────────────
    y_global_train  = model.predict_global(train_rows)
    y_hybrid_train  = model.predict_hybrid(train_rows)
    y_global_ood    = model.predict_global(ood_rows)
    y_hybrid_ood    = model.predict_hybrid(ood_rows)

    # ── Compute benchmark table ───────────────────────────────────────────
    results = [
        _metrics(y_train_true, y_phys_train,    "Physics-only (train)"),
        _metrics(y_train_true, y_global_train,  "Single GBM — prior work (train)"),
        _metrics(y_train_true, y_hybrid_train,  "Phase-Aware GBM — ours (train)"),
    ]
    if len(ood_rows) >= 2:
        results += [
            _metrics(y_ood_true, y_phys_ood,    "Physics-only (OOD)"),
            _metrics(y_ood_true, y_global_ood,  "Single GBM — prior work (OOD)"),
            _metrics(y_ood_true, y_hybrid_ood,  "Phase-Aware GBM — ours (OOD)"),
        ]

    # % improvement over physics baseline
    phys_mae_train = results[0]["MAE_km"]
    for r in results:
        if phys_mae_train > 0:
            pct = (1 - r["MAE_km"] / phys_mae_train) * 100
            r["pct_improvement_vs_physics"] = round(pct, 1)
        else:
            r["pct_improvement_vs_physics"] = 0.0

    # ── Print table ───────────────────────────────────────────────────────
    print("\n" + "="*90)
    print("BENCHMARK COMPARISON TABLE — Celestis Phase-Aware Residual Model")
    print("="*90)
    header = f"{'Method':<42} {'N':>4} {'MAE':>8} {'RMSE':>8} {'R²':>6} {'MAPE':>7} {'vs Physics':>12}"
    print(header)
    print("-" * 90)
    for r in results:
        print(f"{r['label']:<42} {r['N']:>4} "
              f"{r['MAE_km']:>8.4f} {r['RMSE_km']:>8.4f} "
              f"{r['R2']:>6.3f} {r['MAPE_pct']:>6.1f}% "
              f"{r['pct_improvement_vs_physics']:>+10.1f}%")
    print("="*90)
    print(f"\n5-Fold Cross-Validation (on training set, real ground truth):")
    print(f"  MAE  = {cv_results['cv_mae_mean']:.4f} ± {cv_results['cv_mae_std']:.4f} km")
    print(f"  RMSE = {cv_results['cv_rmse_mean']:.4f} ± {cv_results['cv_rmse_std']:.4f} km")
    print()

    # ── Save CSV ──────────────────────────────────────────────────────────
    all_cols = list(results[0].keys())
    with open(BENCHMARK_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_cols)
        writer.writeheader()
        writer.writerows(results)
    log.info(f"Benchmark saved → {BENCHMARK_CSV}")

    # Save CV results separately
    cv_out = os.path.join(OUTPUT_DIR, "crossval_results.json")
    with open(cv_out, "w") as f:
        json.dump({**cv_results, "n_folds": 5, "model": "GBM-global"}, f, indent=2)
    log.info(f"Cross-validation results saved → {cv_out}")


if __name__ == "__main__":
    run_benchmark()
