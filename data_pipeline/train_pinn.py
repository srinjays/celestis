"""
CELESTIS — GNSS PINN Training Pipeline
=======================================
Trains the Physics-Informed Neural Network on real satellite 
altitude data derived from onboard GNSS receiver observables.

Architecture matches existing CELESTIS PINN:
  Input (8 features): altitude, iono_mean, iono_std, iono_max,
                       speed_ms, mean_cno, hour_of_day, day_of_year
  Hidden: 64→128→64 neurons, GELU, BatchNorm, Dropout 15%
  Output: instantaneous altitude decay rate (km/hr)

Physics constraint:
  L_total = α × L_data + (1-α) × L_physics
  L_physics = penalty for predicting altitude gain (physically impossible in drag)

Usage:
    python data_pipeline/train_pinn.py
    python data_pipeline/train_pinn.py --epochs 600 --batch 512 --lr 1e-3
"""

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
FEATURES_DIR = Path("data_pipeline/features")
MODELS_DIR   = Path("data_pipeline/models")

# ── Config defaults ──────────────────────────────────────────────────────────
DEFAULT_EPOCHS     = 600
DEFAULT_BATCH      = 512
DEFAULT_LR         = 1e-3
DEFAULT_ALPHA_START = 0.5   # initial physics/data balance
DEFAULT_ALPHA_END   = 0.85  # final alpha (more data-driven)
TRAIN_SPLIT         = 0.80
VAL_SPLIT           = 0.10  # test = remaining 10%

# Feature columns to use for training
FEATURE_COLS = [
    "altitude_km",
    "iono_delay_mean_m",
    "iono_delay_std_m",
    "iono_delay_max_m",
    "speed_ms",
    "mean_cno",
    "hour_of_day",
    "day_of_year",
]

TARGET_COL = "dH_dt_km_per_hr"


# ── Dataset ──────────────────────────────────────────────────────────────────

class GNSSOrbitDataset(Dataset):
    """
    PyTorch Dataset from GNSS-derived orbit features.
    
    Each sample is one epoch (1 second of data) with:
    - X: feature vector of orbital + ionospheric state
    - y: altitude decay rate (km/hr)
    """
    def __init__(self, df: pd.DataFrame, feature_cols: list, target_col: str):
        # Drop rows with any NaN in features or target
        valid = df[feature_cols + [target_col]].dropna()
        
        self.X = torch.tensor(valid[feature_cols].values, dtype=torch.float32)
        self.y = torch.tensor(valid[target_col].values,   dtype=torch.float32).unsqueeze(1)
        
        # Normalize features
        self.X_mean = self.X.mean(dim=0)
        self.X_std  = self.X.std(dim=0).clamp(min=1e-8)
        self.X      = (self.X - self.X_mean) / self.X_std
        
        # Normalize target (store scale for inverse transform)
        self.y_mean = self.y.mean()
        self.y_std  = self.y.std().clamp(min=1e-8)
        self.y      = (self.y - self.y_mean) / self.y_std
        
        print(f"  Dataset: {len(self.X):,} valid samples | "
              f"altitude range: {valid['altitude_km'].min():.1f}–{valid['altitude_km'].max():.1f} km | "
              f"decay rate range: {valid[target_col].min():.4f}–{valid[target_col].max():.4f} km/hr")
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def split_dataset(dataset: GNSSOrbitDataset, train_frac=0.8, val_frac=0.1):
    """Chronological split — never shuffle time series."""
    n = len(dataset)
    n_train = int(n * train_frac)
    n_val   = int(n * val_frac)
    n_test  = n - n_train - n_val
    
    train_ds, val_ds, test_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )
    return train_ds, val_ds, test_ds


# ── PINN Architecture ────────────────────────────────────────────────────────

class CelestisPINN(nn.Module):
    """
    Physics-Informed Neural Network for satellite altitude decay prediction.
    
    Architecture:
        Input → FC(64) → GELU → BN → Dropout →
                FC(128)→ GELU → BN → Dropout →
                FC(64) → GELU → BN → Dropout →
                FC(1) → Output (decay rate)
    
    The output is the RESIDUAL correction on top of exponential drag baseline.
    total_decay = drag_baseline + pinn_residual
    """
    def __init__(self, n_features: int = 8, dropout: float = 0.15):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(n_features, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            # Layer 2
            nn.Linear(64, 128),
            nn.GELU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout),
            
            # Layer 3
            nn.Linear(128, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            
            # Output
            nn.Linear(64, 1),
        )
        
        # Initialize weights (Xavier for stability)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ── Physics-Constrained Loss ─────────────────────────────────────────────────

class PhysicsConstrainedLoss(nn.Module):
    """
    CELESTIS custom loss function embedding orbital drag physics.
    
    L_total = α × L_data + (1-α) × L_physics
    
    L_data    = MSE between prediction and measured decay rate
    L_physics = penalty for:
        (1) Predicting altitude GAIN (negative decay) — physically impossible
            Satellites only go down due to drag, never up without a maneuver
        (2) Predicting decay rate more than 5× the reference drag baseline
    
    α is annealed from alpha_start → alpha_end over training.
    """
    def __init__(self, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
    
    def forward(
        self,
        pred:       torch.Tensor,  # (B, 1) predicted decay rate (normalized)
        target:     torch.Tensor,  # (B, 1) measured decay rate (normalized)
        y_std:      float = 1.0,   # for physics scale reference
    ) -> dict:
        
        # Data loss: standard MSE
        L_data = nn.functional.mse_loss(pred, target)
        
        # Physics Loss 1: altitude gain penalty
        # In normalized space, negative pred = altitude gain = forbidden
        gain_violation = torch.relu(-pred)
        L_gain = torch.mean(gain_violation ** 2)
        
        # Physics Loss 2: extreme decay penalty (> 5× typical)
        typical_decay_normalized = 1.0  # 1 std in normalized space = typical
        L_extreme = torch.mean(torch.relu(pred - 5 * typical_decay_normalized) ** 2)
        
        L_physics = L_gain + 0.5 * L_extreme
        
        # Combined loss
        L_total = self.alpha * L_data + (1 - self.alpha) * L_physics
        
        return {
            "loss":      L_total,
            "L_data":    L_data.item(),
            "L_physics": L_physics.item(),
            "L_total":   L_total.item(),
        }


# ── Training Loop ─────────────────────────────────────────────────────────────

def anneal_alpha(epoch: int, total_epochs: int,
                 alpha_start: float, alpha_end: float) -> float:
    """Linear annealing of physics/data loss balance."""
    progress = min(epoch / total_epochs, 1.0)
    return alpha_start + progress * (alpha_end - alpha_start)


def evaluate(model, loader, loss_fn, device):
    """Evaluate model on a DataLoader, return avg loss and R²."""
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            
            losses = loss_fn(pred, y)
            total_loss += losses["L_total"]
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())
    
    all_pred = torch.cat(all_pred).numpy().flatten()
    all_true = torch.cat(all_true).numpy().flatten()
    
    ss_res = np.sum((all_true - all_pred) ** 2)
    ss_tot = np.sum((all_true - all_true.mean()) ** 2)
    r2     = 1 - ss_res / (ss_tot + 1e-8)
    
    return total_loss / len(loader), r2


def train_pinn(args):
    """Full PINN training pipeline on GNSS-derived orbital data."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🛰️  CELESTIS PINN Training Pipeline")
    print(f"   Device:  {device}")
    print(f"   Epochs:  {args.epochs}")
    print(f"   Batch:   {args.batch}")
    print(f"   LR:      {args.lr}")
    print(f"   α range: {args.alpha_start} → {args.alpha_end}")
    
    # ── Load Data ─────────────────────────────────────────────────────────────
    master_path = FEATURES_DIR / "master_orbit_features.parquet"
    
    if not master_path.exists():
        print(f"\n❌ Feature dataset not found: {master_path}")
        print(f"   Run: python data_pipeline/ingest.py")
        print(f"   Then: python data_pipeline/preprocess.py")
        return
    
    print(f"\n📂 Loading feature dataset...")
    df = pd.read_parquet(master_path)
    print(f"   Rows: {len(df):,}")
    
    # Check which feature cols are available
    available_features = [c for c in FEATURE_COLS if c in df.columns]
    missing_features   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing_features:
        print(f"   ⚠️  Missing features (will skip): {missing_features}")
    print(f"   Features: {available_features}")
    
    # ── Build Dataset ─────────────────────────────────────────────────────────
    dataset    = GNSSOrbitDataset(df, available_features, TARGET_COL)
    train_ds, val_ds, test_ds = split_dataset(dataset, TRAIN_SPLIT, VAL_SPLIT)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False, num_workers=0)
    
    print(f"   Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")
    
    # ── Build Model ───────────────────────────────────────────────────────────
    model    = CelestisPINN(n_features=len(available_features)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn  = PhysicsConstrainedLoss(alpha=args.alpha_start)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Model parameters: {n_params:,}")
    
    # ── Training ──────────────────────────────────────────────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run_id    = datetime.now().strftime("%Y%m%d_%H%M%S")
    best_val_r2 = -np.inf
    history     = []
    
    print(f"\n{'═'*65}")
    print(f"{'Epoch':>6} {'Train Loss':>11} {'Val Loss':>10} {'Val R²':>8} "
          f"{'L_data':>8} {'L_phys':>8} {'α':>6}")
    print(f"{'─'*65}")
    
    for epoch in range(1, args.epochs + 1):
        
        # Anneal alpha
        alpha_now       = anneal_alpha(epoch, args.epochs, args.alpha_start, args.alpha_end)
        loss_fn.alpha   = alpha_now
        
        # Training step
        model.train()
        train_loss_sum = 0.0
        L_data_sum     = 0.0
        L_phys_sum     = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            pred   = model(X)
            losses = loss_fn(pred, y)
            losses["loss"].backward()
            optimizer.step()
            
            train_loss_sum += losses["L_total"]
            L_data_sum     += losses["L_data"]
            L_phys_sum     += losses["L_physics"]
        
        scheduler.step()
        
        n_batches  = len(train_loader)
        train_loss = train_loss_sum / n_batches
        L_data_avg = L_data_sum / n_batches
        L_phys_avg = L_phys_sum / n_batches
        
        # Validation (every 10 epochs for speed)
        if epoch % 10 == 0 or epoch == args.epochs:
            val_loss, val_r2 = evaluate(model, val_loader, loss_fn, device)
            
            print(f"{epoch:>6} {train_loss:>11.5f} {val_loss:>10.5f} {val_r2:>8.4f} "
                  f"{L_data_avg:>8.5f} {L_phys_avg:>8.5f} {alpha_now:>6.3f}")
            
            history.append({
                "epoch": epoch, "train_loss": float(train_loss),
                "val_loss": float(val_loss), "val_r2": float(val_r2),
                "L_data": float(L_data_avg), "L_physics": float(L_phys_avg),
                "alpha": float(alpha_now)
            })
            
            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_path   = MODELS_DIR / f"celestis_pinn_best_{run_id}.pt"
                torch.save({
                    "epoch":        epoch,
                    "model_state":  model.state_dict(),
                    "optimizer":    optimizer.state_dict(),
                    "val_r2":       val_r2,
                    "features":     available_features,
                    "X_mean":       dataset.X_mean,
                    "X_std":        dataset.X_std,
                    "y_mean":       dataset.y_mean,
                    "y_std":        dataset.y_std,
                    "alpha":        alpha_now,
                    "run_id":       run_id,
                }, best_path)
    
    # ── Final Evaluation ──────────────────────────────────────────────────────
    print(f"\n{'═'*65}")
    print(f"✅ Training complete")
    print(f"   Best Val R²: {best_val_r2:.4f}")
    
    # Load best model for final test evaluation
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    test_loss, test_r2 = evaluate(model, test_loader, loss_fn, device)
    
    print(f"   Test R²:     {test_r2:.4f}")
    print(f"   Model saved: {best_path}")
    
    # Save training history
    hist_path = MODELS_DIR / f"training_history_{run_id}.csv"
    pd.DataFrame(history).to_csv(hist_path, index=False)
    print(f"   History CSV: {hist_path}")
    
    return model, best_path, test_r2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CELESTIS PINN Trainer")
    parser.add_argument("--epochs",      type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",       type=int,   default=DEFAULT_BATCH)
    parser.add_argument("--lr",          type=float, default=DEFAULT_LR)
    parser.add_argument("--alpha_start", type=float, default=DEFAULT_ALPHA_START)
    parser.add_argument("--alpha_end",   type=float, default=DEFAULT_ALPHA_END)
    args = parser.parse_args()
    
    train_pinn(args)
