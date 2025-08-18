#!/usr/bin/env python3
# Minimal CGCNN-style trainer with epoch timing, AMP, and workers
import argparse, os, json, time, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import CGConv, global_mean_pool
from pymatgen.core import Structure

try:
    from torch.cuda.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except Exception:
    AMP_AVAILABLE = False

# ------------------
# Graph construction
# ------------------
def gaussian_expansion(dists, centers, width):
    # dists: (E,), centers: (K,), returns (E, K)
    d = dists[:, None] - centers[None, :]
    return np.exp(-0.5 * (d / width) ** 2).astype(np.float32)

def structure_to_graph(struct: Structure, cutoff=6.0, n_gauss=50):
    """Build a radius graph with Gaussian-expanded distance edge features."""
    num_sites = len(struct)
    z = np.array([site.specie.Z for site in struct.sites], dtype=np.int64)

    # shortest directed edges within cutoff
    shortest = {}
    for i, site in enumerate(struct.sites):
        for nb in struct.get_neighbors(site, cutoff):
            j = int(nb.index)
            if i == j:
                continue
            d = float(nb.nn_distance)
            key = (i, j)
            if key not in shortest or d < shortest[key]:
                shortest[key] = d

    if not shortest:
        # fallback: increase cutoff a bit
        for i, site in enumerate(struct.sites):
            for nb in struct.get_neighbors(site, cutoff * 1.5):
                j = int(nb.index)
                if i == j:
                    continue
                d = float(nb.nn_distance)
                key = (i, j)
                if key not in shortest or d < shortest[key]:
                    shortest[key] = d
        if not shortest:
            raise RuntimeError("No neighbors found; try larger --cutoff.")

    ij = np.array(list(shortest.keys()), dtype=np.int64)  # (E,2)
    edge_index = ij.T
    dists = np.array(list(shortest.values()), dtype=np.float32)

    centers = np.linspace(0.0, cutoff, n_gauss, dtype=np.float32)
    width = (centers[1] - centers[0]) if n_gauss > 1 else 0.5
    edge_attr = gaussian_expansion(dists, centers, width)

    data = Data(
        z=torch.as_tensor(z, dtype=torch.long),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
    )
    return data

# -------------
# PyG Dataset
# -------------
class CrystalDataset(Dataset):
    def __init__(self, split_csv, cutoff=6.0, n_gauss=50):
        self.df = pd.read_csv(split_csv)
        self.cutoff = cutoff
        self.n_gauss = n_gauss

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cif_path = row["cif_path"]
        y = float(row["tc_C"])  # Celsius
        struct = Structure.from_file(cif_path)
        data = structure_to_graph(struct, cutoff=self.cutoff, n_gauss=self.n_gauss)
        data.y = torch.tensor([y], dtype=torch.float32)
        # keep the sample id for downstream analysis
        data.sample_id = int(row["sample_id"])
        return data

# -------------
# The model
# -------------
class SimpleCGCNN(nn.Module):
    def __init__(self, emb_dim=64, n_layers=6, edge_dim=50, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(119, emb_dim)  # atomic number up to 118
        self.convs = nn.ModuleList([CGConv(emb_dim, dim=edge_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.emb(data.z)  # (N, emb_dim)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch)  # (B, emb_dim)
        x = self.mlp(x).squeeze(-1)         # (B,)
        return x

# -------------
# Train / Eval
# -------------
def run_epoch(model, loader, optimizer=None, device="cpu", scaler=None, use_amp=False):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total_mae, total_graphs = 0.0, 0
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                with autocast(enabled=use_amp):
                    out = model(batch)
                    loss = torch.mean(torch.abs(out - batch.y))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(batch)
                loss = torch.mean(torch.abs(out - batch.y))
                loss.backward()
                optimizer.step()
        else:
            with torch.no_grad():
                if scaler is not None:
                    with autocast(enabled=use_amp):
                        out = model(batch)
                        loss = torch.mean(torch.abs(out - batch.y))
                else:
                    out = model(batch)
                    loss = torch.mean(torch.abs(out - batch.y))
        total_mae += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs
    return (total_mae / max(total_graphs, 1)), total_graphs

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="splits", help="dir with train.csv/val.csv/test.csv")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--cutoff", type=float, default=6.0)
    ap.add_argument("--n-gauss", type=int, default=50)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--emb-dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--outdir", default="runs/cgcnn")
    ap.add_argument("--amp", action="store_true", help="use mixed precision on CUDA")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader workers")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    with open(Path(args.outdir) / "hparams.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # datasets & loaders
    train_set = CrystalDataset(Path(args.splits)/"train.csv", cutoff=args.cutoff, n_gauss=args.n_gauss)
    val_set   = CrystalDataset(Path(args.splits)/"val.csv",   cutoff=args.cutoff, n_gauss=args.n_gauss)
    test_set  = CrystalDataset(Path(args.splits)/"test.csv",  cutoff=args.cutoff, n_gauss=args.n_gauss)

    pin = args.device.startswith("cuda")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)

    # model/opt
    model = SimpleCGCNN(emb_dim=args.emb_dim, n_layers=args.layers,
                        edge_dim=args.n_gauss, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # AMP scaler
    use_amp = bool(args.amp and args.device.startswith("cuda") and AMP_AVAILABLE)
    scaler = GradScaler(enabled=use_amp) if use_amp else None

    # metrics log
    metrics_path = Path(args.outdir) / "metrics.csv"
    with open(metrics_path, "w") as f:
        f.write("epoch,train_mae,val_mae,secs,graphs_per_sec,best_val\n")

    best_val, best_path = float("inf"), Path(args.outdir)/"best.pt"
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        tr_mae, tr_graphs = run_epoch(model, train_loader, optimizer=opt,
                                      device=args.device, scaler=scaler, use_amp=use_amp)
        va_mae, va_graphs = run_epoch(model, val_loader, optimizer=None,
                                      device=args.device, scaler=scaler, use_amp=use_amp)
        secs = time.time() - t0
        gps = (tr_graphs + va_graphs) / max(secs, 1e-6)
        is_best = va_mae < best_val
        if is_best:
            best_val = va_mae
            torch.save(model.state_dict(), best_path)
        print(f"[{epoch:03d}] train MAE: {tr_mae:.2f} °C | val MAE: {va_mae:.2f} °C "
              f"| {secs:.1f}s/epoch | ~{gps:.1f} graphs/s | best val: {best_val:.2f} °C")
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_mae:.6f},{va_mae:.6f},{secs:.3f},{gps:.2f},{best_val:.6f}\n")

    # Test with best checkpoint
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    te_mae, _ = run_epoch(model, test_loader, optimizer=None,
                          device=args.device, scaler=scaler, use_amp=use_amp)
    print(f"[TEST] MAE: {te_mae:.2f} °C")

    # Save test predictions
    model.eval()
    preds, targets, ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(args.device, non_blocking=True)
            if use_amp:
                with autocast():
                    out = model(batch)
            else:
                out = model(batch)
            preds.extend(out.cpu().numpy().tolist())
            targets.extend(batch.y.cpu().numpy().tolist())
            ids.extend([getattr(g, "sample_id") for g in batch.to_data_list()])
    pd.DataFrame({"sample_id": ids, "tc_C_true": targets, "tc_C_pred": preds}) \
      .to_csv(Path(args.outdir)/"test_predictions.csv", index=False)
    print(f"Saved: {best_path}, {metrics_path}, and test_predictions.csv")

if __name__ == "__main__":
    main()
