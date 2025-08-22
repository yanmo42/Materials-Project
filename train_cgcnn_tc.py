#!/usr/bin/env python3
"""
train_cgcnn_tc.py — Train a CGCNN-style model to predict Curie temperature (Tc)

Dataset layout (as created by build_tc_dataset.py):
  dataset_tc/
    structures/<sample_id>.cif
    id_prop.csv            # columns: id, Tc_C   (or: id, Tc_K)
    splits/
      train_ids.csv        # one id per line
      val_ids.csv
      test_ids.csv

Install (CPU example):
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  pip install pymatgen pandas numpy tqdm

Usage:
  python train_cgcnn_tc.py --data-root dataset_tc --epochs 300 --batch-size 16 --radius 8.0 --max-nbr 12 --device cuda
  # use Kelvin targets if id_prop.csv contains Tc_K:
  python train_cgcnn_tc.py --data-root dataset_tc --target-col Tc_K
  # enable disk caching of processed graphs:
  python train_cgcnn_tc.py --data-root dataset_tc --cache
"""

from __future__ import annotations
import argparse, os, math, random, time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from pymatgen.core import Structure

# -------------------------- Utils & Repro --------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def human_time(s: float) -> str:
    m, s = divmod(int(s), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

# -------------------------- Graph Builder --------------------------

def gaussian_expansion(dist: torch.Tensor, centers: torch.Tensor, width: float) -> torch.Tensor:
    # dist: (E,), centers: (G,)
    return torch.exp(-((dist.unsqueeze(1) - centers.unsqueeze(0)) ** 2) / (width ** 2))

def build_graph_from_structure(struct: Structure, radius: float, max_nbr: int,
                               gcenters: torch.Tensor, gwidth: float) -> Dict[str, torch.Tensor]:
    """
    Returns (CPU tensors):
      x      : (N,) atomic numbers (int64) -> embedding indices
      edge_i : (E,) destination node indices
      edge_j : (E,) source node indices
      edge_f : (E, G) edge (distance) features, Gaussian expanded
    """
    # Atom features = atomic numbers (embedding lookup later)
    Z = torch.tensor([site.specie.Z for site in struct], dtype=torch.long)
    N = len(struct)

    # Neighbor list (periodic)
    try:
        ii, jj, offsets, dists = struct.get_neighbor_list(r=radius)
        # Keep only nearest max_nbr per destination atom
        ii = np.asarray(ii); jj = np.asarray(jj); dists = np.asarray(dists)
        order = np.lexsort((dists, ii))  # sort primarily by ii then by distance
        ii, jj, dists = ii[order], jj[order], dists[order]

        kept_mask = np.zeros_like(ii, dtype=bool)
        counts = np.zeros(N, dtype=np.int32)
        for k in range(len(ii)):
            i = ii[k]
            if i == jj[k]:
                continue
            if counts[i] < max_nbr:
                kept_mask[k] = True
                counts[i] += 1
        ii, jj, dists = ii[kept_mask], jj[kept_mask], dists[kept_mask]
    except Exception:
        # Fallback: per-site neighbor query (may be slower)
        ii_list, jj_list, dd_list = [], [], []
        for i, site in enumerate(struct.sites):
            neighs = struct.get_neighbors(site, r=radius)
            neighs = sorted([n for n in neighs if n.index != i], key=lambda n: n.nn_distance)[:max_nbr]
            for n in neighs:
                ii_list.append(i); jj_list.append(n.index); dd_list.append(float(n.nn_distance))
        ii = np.array(ii_list, dtype=np.int64)
        jj = np.array(jj_list, dtype=np.int64)
        dists = np.array(dd_list, dtype=np.float32)

    # If nothing found (tiny cells or edge cases), make a minimal nearest-neighbor graph (non-periodic)
    if len(ii) == 0:
        coords = np.array([s.coords for s in struct.sites], dtype=float)  # cartesian
        if N >= 2:
            diffs = coords[:, None, :] - coords[None, :, :]
            D = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(D, np.inf)
            nn_idx = np.argmin(D, axis=1)
            ii = np.arange(N, dtype=np.int64)
            jj = nn_idx.astype(np.int64)
            dists = D[ii, jj].astype(np.float32)
        else:
            ii = np.array([0], dtype=np.int64)
            jj = np.array([0], dtype=np.int64)
            dists = np.array([1.0], dtype=np.float32)

    edge_i = torch.tensor(ii, dtype=torch.long)
    edge_j = torch.tensor(jj, dtype=torch.long)
    edge_d = torch.tensor(dists, dtype=torch.float32)
    edge_f = gaussian_expansion(edge_d, gcenters, gwidth)  # (E, G)

    return {"x": Z, "edge_i": edge_i, "edge_j": edge_j, "edge_f": edge_f}

# -------------------------- Dataset --------------------------

class CrystalDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split_ids: List[str], target_map: Dict[str, float],
                 radius: float = 8.0, max_nbr: int = 12, gmin: float = 0.0, gmax: float = 8.0,
                 gstep: float = 0.2, gwidth: float = 0.2, cache: bool = False):
        self.root = Path(root)
        self.ids = split_ids
        self.target_map = target_map
        self.radius = radius
        self.max_nbr = max_nbr
        self.gcenters = torch.arange(gmin, gmax + 1e-6, gstep, dtype=torch.float32)  # inclusive
        self.gwidth = gwidth
        self.cache = cache
        self.cache_dir = self.root / "cache"
        if self.cache:
            self.cache_dir.mkdir(exist_ok=True)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int):
        sid = self.ids[idx]
        cif_path = self.root / "structures" / f"{sid}.cif"
        y = float(self.target_map[sid])

        cache_file = self.cache_dir / f"{sid}.pt"
        if self.cache and cache_file.exists():
            data = torch.load(cache_file, map_location="cpu")
        else:
            struct = Structure.from_file(str(cif_path))
            data = build_graph_from_structure(
                struct, self.radius, self.max_nbr, self.gcenters, self.gwidth
            )
            if self.cache:
                torch.save({k: v for k, v in data.items()}, cache_file)

        # Return CPU tensors; we'll move to device in the main process (avoid CUDA in workers)
        return {"id": sid, "x": data["x"], "edge_i": data["edge_i"],
                "edge_j": data["edge_j"], "edge_f": data["edge_f"],
                "y": torch.tensor([y], dtype=torch.float32)}

def collate_graphs(batch: List[Dict[str, torch.Tensor]]):
    """
    CPU-only collation. DO NOT move tensors to CUDA here (runs inside worker processes).
    """
    xs, e_i, e_j, e_f, ys, batch_index = [], [], [], [], [], []
    n_off = 0
    for b_idx, item in enumerate(batch):
        x = item["x"]; ei = item["edge_i"]; ej = item["edge_j"]; ef = item["edge_f"]; y = item["y"]
        xs.append(x); ys.append(y)
        e_i.append(ei + n_off); e_j.append(ej + n_off); e_f.append(ef)
        batch_index.append(torch.full((x.shape[0],), b_idx, dtype=torch.long))
        n_off += x.shape[0]
    x = torch.cat(xs, dim=0)             # CPU
    edge_i = torch.cat(e_i, dim=0)       # CPU
    edge_j = torch.cat(e_j, dim=0)       # CPU
    edge_f = torch.cat(e_f, dim=0)       # CPU
    y = torch.cat(ys, dim=0)             # CPU  (B,1)
    batch_index = torch.cat(batch_index, dim=0)  # CPU
    return {"x": x, "edge_i": edge_i, "edge_j": edge_j, "edge_f": edge_f, "batch": batch_index, "y": y}

# -------------------------- Model --------------------------

class EdgeNetwork(nn.Module):
    """MLP over concatenated (x_j, edge_f) to produce messages."""
    def __init__(self, in_atom: int, in_edge: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_atom + in_edge, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU()
        )

    def forward(self, x_j: torch.Tensor, edge_f: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x_j, edge_f], dim=1))

class CGCNNBlock(nn.Module):
    """
    Message passing block:
      m_ij = EdgeMLP([x_j, e_ij])  -> (E, hidden)
      agg_i = sum_j m_ij           -> (N, hidden)
      x_i' = SiLU(x_i + W * Dropout(agg_i))  with W: hidden->atom_dim
    """
    def __init__(self, atom_dim: int, edge_dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.edge_net = EdgeNetwork(atom_dim, edge_dim, hidden)
        self.lin = nn.Linear(hidden, atom_dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = hidden  # keep for shape creation

    def forward(self, x: torch.Tensor, edge_i: torch.Tensor, edge_j: torch.Tensor, edge_f: torch.Tensor) -> torch.Tensor:
        x_j = x[edge_j]  # (E, atom_dim)
        m = self.edge_net(x_j, edge_f)  # (E, hidden)
        # FIX: aggregate in hidden-dim buffer (not atom_dim)
        agg = torch.zeros((x.size(0), self.hidden), device=x.device, dtype=x.dtype)
        agg.index_add_(0, edge_i, m)  # (N, hidden)
        x = F.silu(x + self.lin(self.dropout(agg)))  # map hidden->atom_dim, residual
        return x

class CrystalGraphNetwork(nn.Module):
    def __init__(self, atom_emb_dim: int = 128, edge_feat_dim: int = 41,
                 num_convs: int = 3, hidden: int = 256, dropout: float = 0.1):
        super().__init__()
        # Atom embedding for Z up to 118 (index 0 unused)
        self.atom_embed = nn.Embedding(119, atom_emb_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            CGCNNBlock(atom_emb_dim, edge_feat_dim, hidden, dropout=dropout)
            for _ in range(num_convs)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(atom_emb_dim) for _ in range(num_convs)])
        self.head = nn.Sequential(
            nn.Linear(atom_emb_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden//2),
            nn.SiLU(),
            nn.Linear(hidden//2, 1),
        )

    def forward(self, Z: torch.Tensor, edge_i: torch.Tensor, edge_j: torch.Tensor, edge_f: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        x = self.atom_embed(Z)  # (N, atom_emb_dim)
        for conv, ln in zip(self.convs, self.norms):
            x = conv(x, edge_i, edge_j, edge_f)
            x = ln(x)
        # Pool by mean over nodes per crystal
        B = int(batch.max().item() + 1) if batch.numel() > 0 else 1
        sums = torch.zeros((B, x.size(1)), device=x.device)
        counts = torch.zeros((B, 1), device=x.device)
        sums.index_add_(0, batch, x)
        counts.index_add_(0, batch, torch.ones((x.size(0), 1), device=x.device))
        pooled = sums / counts.clamp_min(1.0)
        out = self.head(pooled)  # (B, 1)
        return out

# -------------------------- Metrics --------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device: torch.device, scaler=None) -> Dict[str, float]:
    model.eval()
    mae, mse, n = 0.0, 0.0, 0
    for batch in loader:
        # tensors arrive on CPU; move to device here (main process)
        for k in ("x", "edge_i", "edge_j", "edge_f", "batch", "y"):
            batch[k] = batch[k].to(device, non_blocking=True)
        y = batch["y"]  # (B,1)
        pred = model(batch["x"], batch["edge_i"], batch["edge_j"], batch["edge_f"], batch["batch"])
        if scaler is not None:
            pred = pred * scaler["std"] + scaler["mean"]
        diff = (pred - y)
        mae += torch.abs(diff).sum().item()
        mse += (diff ** 2).sum().item()
        n += y.numel()
    rmse = math.sqrt(mse / max(n, 1))
    mae = mae / max(n, 1)
    return {"MAE": mae, "RMSE": rmse}

# -------------------------- Training --------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="Path to dataset root (dataset_tc)")
    ap.add_argument("--target-col", default="Tc_C", choices=["Tc_C", "Tc_K"], help="Target column in id_prop.csv")
    ap.add_argument("--radius", type=float, default=8.0, help="Neighbor cutoff radius (Å)")
    ap.add_argument("--max-nbr", type=int, default=12, help="Max neighbors per atom")
    ap.add_argument("--gmax", type=float, default=8.0, help="Max distance for Gaussian centers (Å)")
    ap.add_argument("--gstep", type=float, default=0.2, help="Step for Gaussian centers (Å)")
    ap.add_argument("--gwidth", type=float, default=0.2, help="Gaussian width (Å)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-convs", type=int, default=3)
    ap.add_argument("--atom-emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--cache", action="store_true", help="Cache processed graphs to disk")
    # argparse turns dashes into underscores; set dest explicitly and use args.no_standardize
    ap.add_argument("--no-standardize", dest="no_standardize", action="store_true",
                    help="Disable target standardization")
    ap.add_argument("--early-stop", type=int, default=40, help="Early stopping patience (epochs)")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    root = Path(args.data_root)
    idprop = pd.read_csv(root / "id_prop.csv")
    if args.target_col not in idprop.columns:
        alt = "Tc_K" if args.target_col == "Tc_C" else "Tc_C"
        if alt in idprop.columns:
            print(f"[info] Target column '{args.target_col}' not found; using '{alt}' instead.")
            args.target_col = alt
        else:
            raise SystemExit(f"id_prop.csv must contain '{args.target_col}' (or 'Tc_C'/'Tc_K').")

    id2y = {r["id"]: float(r[args.target_col]) for _, r in idprop.iterrows()}

    def read_split(name: str) -> List[str]:
        p = root / "splits" / f"{name}_ids.csv"
        if not p.exists():
            return []
        s = [str(x).strip() for x in open(p, "r", encoding="utf-8").read().splitlines() if str(x).strip()]
        return [i for i in s if i in id2y]

    train_ids = read_split("train")
    val_ids   = read_split("val")
    test_ids  = read_split("test")

    print(f"Train/Val/Test sizes: {len(train_ids)}/{len(val_ids)}/{len(test_ids)}")

    ds_args = dict(radius=args.radius, max_nbr=args.max_nbr, gmin=0.0, gmax=args.gmax,
                   gstep=args.gstep, gwidth=args.gwidth, cache=args.cache)
    train_ds = CrystalDataset(root, train_ids, id2y, **ds_args)
    val_ds   = CrystalDataset(root, val_ids,   id2y, **ds_args)
    test_ds  = CrystalDataset(root, test_ids,  id2y, **ds_args)

    def make_loader(ds, shuffle: bool):
        # IMPORTANT: collate_fn returns CPU tensors; no CUDA in workers
        return torch.utils.data.DataLoader(
            ds, batch_size=args.batch_size, shuffle=shuffle,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=(device.type == "cuda"),
            collate_fn=collate_graphs
        )

    train_loader = make_loader(train_ds, True)
    val_loader   = make_loader(val_ds, False)
    test_loader  = make_loader(test_ds, False)

    # edge features dim = number of Gaussian centers (inclusive range)
    edge_feat_dim = int(np.floor(args.gmax / args.gstep)) + 1
    model = CrystalGraphNetwork(atom_emb_dim=args.atom_emb_dim, edge_feat_dim=edge_feat_dim,
                                num_convs=args.num_convs, hidden=args.hidden, dropout=args.dropout).to(device)
    print(model)

    # Target scaler (standardization improves optimization)
    scaler = None
    if not args.no_standardize:
        ys = torch.tensor([id2y[i] for i in train_ids], dtype=torch.float32, device=device).unsqueeze(1)
        mean = ys.mean(dim=0)
        std = ys.std(dim=0).clamp_min(1e-6)
        scaler = {"mean": mean, "std": std}
        print(f"[scaler] mean={mean.item():.4f}, std={std.item():.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, verbose=True)
    loss_fn = nn.L1Loss()  # MAE

    best_val = float("inf")
    best_state = None
    patience = args.early_stop
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            # Move to device in main process (not in workers)
            for k in ("x", "edge_i", "edge_j", "edge_f", "batch", "y"):
                batch[k] = batch[k].to(device, non_blocking=True)

            y = batch["y"]  # (B,1)
            target = y
            if scaler is not None:
                target = (target - scaler["mean"]) / scaler["std"]

            pred = model(batch["x"], batch["edge_i"], batch["edge_j"], batch["edge_f"], batch["batch"])
            loss = loss_fn(pred, target)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item() * y.size(0)
            n_samples += y.size(0)

        train_loss = total_loss / max(1, n_samples)

        # Validation
        val_metrics = evaluate(model, val_loader, device, scaler=scaler)
        scheduler.step(val_metrics["MAE"])

        improved = val_metrics["MAE"] < best_val - 1e-6
        if improved:
            best_val = val_metrics["MAE"]
            best_state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler,
                "args": vars(args),
            }

        print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} | "
              f"val_MAE={val_metrics['MAE']:.3f} | val_RMSE={val_metrics['RMSE']:.3f} | "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        if not improved:
            patience -= 1
            if patience <= 0:
                print("Early stopping triggered.")
                break
        else:
            patience = args.early_stop

    # Save best checkpoint
    out_dir = Path(args.data_root) / "checkpoints"
    out_dir.mkdir(exist_ok=True)
    ckpt_path = out_dir / "best_cgcnn_tc.pt"
    if best_state is None:
        best_state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler,
            "args": vars(args),
        }
    torch.save(best_state, ckpt_path)
    print(f"[saved] {ckpt_path}")

    # Load best & evaluate on test
    model.load_state_dict(best_state["model"])
    test_metrics = evaluate(model, test_loader, device, scaler=scaler)
    print(f"[test] MAE={test_metrics['MAE']:.3f}  RMSE={test_metrics['RMSE']:.3f}  "
          f"(units: {'K' if args.target_col=='Tc_K' else '°C'})")
    print(f"[time] total {human_time(time.time()-t0)}")

if __name__ == "__main__":
    main()
