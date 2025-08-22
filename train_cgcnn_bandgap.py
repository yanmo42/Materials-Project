#!/usr/bin/env python3
"""
train_cgcnn_bandgap.py
Train a CGCNN-style regressor on the dataset produced by make_mp_bandgap_dataset.py.

Expected CSVs in --data-root: train.csv, val.csv, test.csv with columns:
  id, band_gap, cif_path

Key features:
- On-disk graph caching (first epoch builds graphs from CIFs, later loads .pt)
- Mixed precision (--amp) + cosine or plateau scheduler
- Non-negativity option for predictions (--nonneg uses Softplus head)
- Reproducible seeds, num_workers/pin_memory controls, CSV logging, best checkpoint by val MAE

Usage:
  python train_cgcnn_bandgap.py --data-root testingballs123 --device cuda --epochs 80 --batch-size 32 --y-norm --amp
"""
import os
import json
import math
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PYGDataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure

# ---------------------- Utils ---------------------- #

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mae(pred, true): return torch.mean(torch.abs(pred-true)).item()
def rmse(pred, true): return math.sqrt(torch.mean((pred-true)**2).item())
def r2(pred, true):
    ss_res = torch.sum((true - pred)**2)
    ss_tot = torch.sum((true - torch.mean(true))**2) + 1e-9
    return (1 - ss_res/ss_tot).item()


def gaussian_expansion(dist: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    # dist: (E,), centers: (K,)
    return torch.exp(-0.5 * ((dist.view(-1, 1) - centers.view(1, -1)) / width) ** 2)


def build_graph(struct: Structure, cutoff: float, max_neighbors: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns: (z, edge_index, dists)
      z: (N,) atomic numbers (1..)
      edge_index: (2, E)
      dists: (E,)
    """
    N = len(struct)
    z = torch.tensor([site.specie.Z for site in struct.sites], dtype=torch.long)

    send, recv, dists = [], [], []
    for i in range(N):
        neighs = struct.get_neighbors(struct[i], r=cutoff)
        neighs.sort(key=lambda x: x.nn_distance)
        # Take up to max_neighbors nearest neighbors
        for n in neighs[:max_neighbors]:
            j = int(n.index)
            send.append(i); recv.append(j); dists.append(float(n.nn_distance))

    if len(dists) == 0:
        # Fallback: self loops (degenerate cell)
        send, recv, dists = list(range(N)), list(range(N)), [0.0] * N

    edge_index = torch.tensor([send, recv], dtype=torch.long)
    dists = torch.tensor(dists, dtype=torch.float)
    return z, edge_index, dists


# ---------------------- Dataset with caching ---------------------- #

class CrystalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: Path,
        cutoff: float = 8.0,
        max_neighbors: int = 12,
        n_gauss: int = 50,
        r_max: float = 8.0,
        width: float = 0.4,
        y_norm: bool = True,
        cache_dir: Optional[Path] = None,
    ):
        df = pd.read_csv(csv_path)
        self.ids = df["id"].tolist()
        self.paths = df["cif_path"].tolist()
        self.y = df["band_gap"].astype(float).values

        self.cutoff = float(cutoff)
        self.max_neighbors = int(max_neighbors)
        self.centers = torch.linspace(0, r_max, n_gauss)
        self.width = float(width)
        self.y_norm = y_norm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        if y_norm:
            self.y_mean = float(np.mean(self.y))
            self.y_std = float(np.std(self.y) + 1e-8)
        else:
            self.y_mean = 0.0
            self.y_std = 1.0

    def __len__(self): return len(self.ids)

    def _cache_path(self, idx: int) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        # Use the material id as filename
        return self.cache_dir / f"{self.ids[idx]}.pt"

    def _load_or_build_graph(self, idx: int) -> Data:
        cache_path = self._cache_path(idx)
        if cache_path and cache_path.exists():
            try:
                data = torch.load(cache_path)
                if isinstance(data, Data) and hasattr(data, "edge_attr"):
                    return data
            except Exception:
                pass  # fall through to rebuild

        # Build from CIF
        struct = Structure.from_file(self.paths[idx])
        z, edge_index, dists = build_graph(struct, self.cutoff, self.max_neighbors)
        edge_attr = gaussian_expansion(dists, self.centers, self.width)
        data = Data(z=z, edge_index=edge_index, edge_attr=edge_attr, n_atoms=torch.tensor([len(struct)], dtype=torch.long))

        if cache_path:
            try:
                torch.save(data, cache_path)
            except Exception:
                pass
        return data

    def __getitem__(self, idx):
        data = self._load_or_build_graph(idx)
        # Attach target and metadata on the fly
        y_val = (self.y[idx] - self.y_mean) / self.y_std if self.y_norm else self.y[idx]
        data.y = torch.tensor([y_val], dtype=torch.float)
        data.id = self.ids[idx]
        return data


# ---------------------- Model ---------------------- #

class CGCNNRegressor(nn.Module):
    def __init__(
        self,
        emb_dim=64,
        edge_feat_dim=50,
        n_convs=3,
        hidden=128,
        max_Z=118,
        dropout=0.0,
        nonneg=False,
    ):
        super().__init__()
        self.node_emb = nn.Embedding(max_Z + 1, emb_dim, padding_idx=0)  # Z ∈ [1..max_Z]
        self.convs = nn.ModuleList([CGConv(emb_dim, dim=edge_feat_dim, batch_norm=True) for _ in range(n_convs)])
        self.act = nn.SiLU()
        self.nonneg = nonneg
        head = [
            nn.Linear(emb_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        ]
        if nonneg:
            head.append(nn.Softplus())  # ensures output >= 0
        self.readout = nn.Sequential(*head)

    def forward(self, z, edge_index, edge_attr, batch):
        # Clamp Z to embedding range just in case
        z = z.clamp(min=0, max=self.node_emb.num_embeddings - 1)
        x = self.node_emb(z)
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = self.act(x)
        g = global_mean_pool(x, batch)  # (B, emb_dim)
        yhat = self.readout(g).squeeze(-1)  # (B,)
        return yhat


# ---------------------- Train / Eval ---------------------- #

def run_epoch(model, loader, device, optimizer=None, scaler=None, y_denorm: Optional[Tuple[float, float]] = None):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, nobs = 0.0, 0
    preds_list, trues_list = [], []
    ids_list = []

    crit = nn.SmoothL1Loss()  # robust to outliers

    for batch in loader:
        batch = batch.to(device)
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            pred = model(batch.z, batch.edge_index, batch.edge_attr, batch.batch)
            loss = crit(pred, batch.y.view(-1))

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += float(loss.item()) * batch.num_graphs
        nobs += batch.num_graphs
        preds_list.append(pred.detach().cpu())
        trues_list.append(batch.y.detach().cpu())
        # PyG keeps custom Python attributes as lists in the batch
        ids_batch = batch.id if isinstance(batch.id, list) else [batch.id]
        ids_list.extend(ids_batch)

    preds = torch.cat(preds_list)
    trues = torch.cat(trues_list)

    # Optional inverse transform for metrics
    if y_denorm is not None:
        mean, std = y_denorm
        preds = preds * std + mean
        trues = trues * std + mean

    return total_loss / max(nobs, 1), preds, trues, ids_list


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cutoff", type=float, default=8.0)
    ap.add_argument("--max-neighbors", type=int, default=12)
    ap.add_argument("--n-gauss", type=int, default=50)
    ap.add_argument("--r-max", type=float, default=8.0)
    ap.add_argument("--width", type=float, default=0.4)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--n-convs", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--y-norm", action="store_true", help="Normalize target during training (recommended).")
    ap.add_argument("--run-dir", type=Path, default=None, help="Where to save checkpoints & logs.")
    ap.add_argument("--cache-dir", type=Path, default=None, help="Graph cache directory (defaults to <data-root>/cache).")
    ap.add_argument("--num-workers", type=int, default=0, help="DataLoader workers. >0 speeds up with caching.")
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision.")
    ap.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="plateau")
    ap.add_argument("--nonneg", action="store_true", help="Enforce non-negative predictions via Softplus head.")
    args = ap.parse_args()

    set_seed(args.seed)

    run_dir = args.run_dir or (args.data_root / "runs" / f"cgcnn_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Make args JSON-serializable (cast Path objects to str)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))

    device = torch.device(args.device)

    # Datasets
    default_cache = args.cache_dir or (args.data_root / "cache")
    train_ds = CrystalDataset(
        args.data_root / "train.csv",
        cutoff=args.cutoff, max_neighbors=args.max_neighbors,
        n_gauss=args.n_gauss, r_max=args.r_max, width=args.width,
        y_norm=args.y_norm, cache_dir=default_cache
    )
    val_ds = CrystalDataset(
        args.data_root / "val.csv",
        cutoff=args.cutoff, max_neighbors=args.max_neighbors,
        n_gauss=args.n_gauss, r_max=args.r_max, width=args.width,
        y_norm=args.y_norm, cache_dir=default_cache
    )
    test_ds = CrystalDataset(
        args.data_root / "test.csv",
        cutoff=args.cutoff, max_neighbors=args.max_neighbors,
        n_gauss=args.n_gauss, r_max=args.r_max, width=args.width,
        y_norm=args.y_norm, cache_dir=default_cache
    )

    y_mean, y_std = train_ds.y_mean, train_ds.y_std
    (run_dir / "target_norm.json").write_text(json.dumps(
        {"y_mean": y_mean, "y_std": y_std, "normalized": bool(args.y_norm)}, indent=2))

    pin = (device.type == "cuda")
    train_loader = PYGDataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                 num_workers=args.num_workers, pin_memory=pin, persistent_workers=bool(args.num_workers))
    val_loader = PYGDataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, pin_memory=pin, persistent_workers=bool(args.num_workers))
    test_loader = PYGDataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=pin, persistent_workers=bool(args.num_workers))

    model = CGCNNRegressor(
        emb_dim=args.emb_dim, edge_feat_dim=args.n_gauss, n_convs=args.n_convs,
        hidden=args.hidden, dropout=args.dropout, max_Z=118, nonneg=args.nonneg
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    if args.scheduler == "plateau":
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=8, verbose=True)
    elif args.scheduler == "cosine":
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    else:
        sched = None

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = float("inf")
    best_path = run_dir / "best.pt"
    log_rows = []

    print(f"Model parameters: {count_parameters(model):,}")

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_pred, tr_true, _ = run_epoch(
            model, train_loader, device, optimizer=opt, scaler=scaler,
            y_denorm=(y_mean, y_std) if args.y_norm else None
        )
        va_loss, va_pred, va_true, _ = run_epoch(
            model, val_loader, device, optimizer=None, scaler=None,
            y_denorm=(y_mean, y_std) if args.y_norm else None
        )

        tr_mae, tr_rmse, tr_r2 = mae(tr_pred, tr_true), rmse(tr_pred, tr_true), r2(tr_pred, tr_true)
        va_mae, va_rmse, va_r2 = mae(va_pred, va_true), rmse(va_pred, va_true), r2(va_pred, va_true)

        lr_now = opt.param_groups[0]["lr"]
        print(f"[{epoch:03d}/{args.epochs}] LR={lr_now:.2e} "
              f"train: L={tr_loss:.4f} MAE={tr_mae:.3f} RMSE={tr_rmse:.3f} R2={tr_r2:.3f} | "
              f"val: L={va_loss:.4f} MAE={va_mae:.3f} RMSE={va_rmse:.3f} R2={va_r2:.3f}")

        log_rows.append({
            "epoch": epoch, "lr": lr_now,
            "train_loss": tr_loss, "train_mae": tr_mae, "train_rmse": tr_rmse, "train_r2": tr_r2,
            "val_loss": va_loss, "val_mae": va_mae, "val_rmse": va_rmse, "val_r2": va_r2
        })

        # Step schedulers
        if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sched.step(va_mae)  # minimize MAE
        elif sched is not None:
            sched.step()

        # Save best by val MAE
        if va_mae < best_val:
            best_val = va_mae
            torch.save({"model": model.state_dict(), "args": vars(args)}, best_path)

    # Save logs
    pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)

    # Test with best checkpoint
    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])

    te_loss, te_pred, te_true, te_ids = run_epoch(
        model, test_loader, device, optimizer=None, scaler=None,
        y_denorm=(y_mean, y_std) if args.y_norm else None
    )

    te_mae, te_rmse, te_r2 = mae(te_pred, te_true), rmse(te_pred, te_true), r2(te_pred, te_true)
    print(f"TEST: MAE={te_mae:.3f} eV  RMSE={te_rmse:.3f} eV  R2={te_r2:.3f}")

    # Write predictions aligned with ids
    pd.DataFrame({
        "id": te_ids,
        "band_gap_true": te_true.numpy().reshape(-1),
        "band_gap_pred": te_pred.numpy().reshape(-1)
    }).to_csv(run_dir / "preds_test.csv", index=False)

    # Save final (last) checkpoint too
    torch.save({"model": model.state_dict(), "args": vars(args)}, run_dir / "last.pt")
    print(f"Saved checkpoints to: {run_dir}")


if __name__ == "__main__":
    main()
