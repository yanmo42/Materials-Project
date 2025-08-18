#!/usr/bin/env python3
# CGCNN trainer (strict mode + optional target normalization)
import argparse, os, json, time, random, math
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
from torch import amp as _amp  # PyTorch 2.6+ AMP API

# ------------------
# Graph construction
# ------------------
def gaussian_expansion(dists, centers, width):
    d = dists[:, None] - centers[None, :]
    return np.exp(-0.5 * (d / width) ** 2).astype(np.float32)

def _radius_edges(struct: Structure, cutoff: float):
    """Collect shortest directed edges within 'cutoff' Å."""
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
    return shortest

def structure_to_graph(struct: Structure, cutoff=6.0, n_gauss=50):
    """Strict: build radius graph at 'cutoff' only; raise if no edges."""
    num_sites = len(struct)
    z = np.array([site.specie.Z for site in struct.sites], dtype=np.int64)

    shortest = _radius_edges(struct, cutoff)
    if not shortest:
        raise RuntimeError("No neighbors at cutoff.")

    ij = np.array(list(shortest.keys()), dtype=np.int64)  # (E,2)
    edge_index = ij.T
    dists = np.array(list(shortest.values()), dtype=np.float32)

    centers = np.linspace(0.0, max(cutoff, dists.max() + 1e-3), n_gauss, dtype=np.float32)
    width = (centers[1] - centers[0]) if n_gauss > 1 else 0.5
    edge_attr = gaussian_expansion(dists, centers, width)

    data = Data(
        z=torch.as_tensor(z, dtype=torch.long),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        edge_attr=torch.as_tensor(edge_attr, dtype=torch.float32),
    )
    data.num_nodes = int(num_sites)  # silence PyG inference warning
    return data

# -------------
# Dataset
# -------------
class CrystalDataset(Dataset):
    def __init__(self, split_csv, cutoff=6.0, n_gauss=50, normalize=False, y_mu=0.0, y_std=1.0):
        self.df = pd.read_csv(split_csv)
        self.cutoff = cutoff
        self.n_gauss = n_gauss
        self.normalize = normalize
        self.y_mu = float(y_mu)
        self.y_std = float(y_std if y_std and math.isfinite(y_std) else 1.0)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        struct = Structure.from_file(row["cif_path"])
        data = structure_to_graph(struct, cutoff=self.cutoff, n_gauss=self.n_gauss)
        y_C = float(row["tc_C"])
        y = (y_C - self.y_mu) / self.y_std if self.normalize else y_C
        data.y = torch.tensor([y], dtype=torch.float32)      # used for loss
        data.y_C = torch.tensor([y_C], dtype=torch.float32)  # used for metrics (°C)
        data.sample_id = int(row["sample_id"])
        return data

# -------------
# Model
# -------------
class SimpleCGCNN(nn.Module):
    def __init__(self, emb_dim=64, n_layers=6, edge_dim=50, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(119, emb_dim)
        self.convs = nn.ModuleList([CGConv(emb_dim, dim=edge_dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data):
        x = self.emb(data.z)
        for conv in self.convs:
            x = conv(x, data.edge_index, data.edge_attr)
        x = global_mean_pool(x, data.batch)
        return self.mlp(x).squeeze(-1)

# -------------
# Train / Eval
# -------------
def run_epoch(model, loader, optimizer=None, device="cpu", scaler=None, use_amp=False,
              normalize=False, y_mu=0.0, y_std=1.0):
    train = optimizer is not None
    model.train() if train else model.eval()
    total_mae_C, total_graphs = 0.0, 0

    for batch in loader:
        batch = batch.to(device, non_blocking=True)

        def _forward():
            with _amp.autocast("cuda", enabled=use_amp):
                out = model(batch)  # normalized or °C
                loss = torch.mean(torch.abs(out - batch.y))  # train on normalized if enabled
            return out, loss

        if train:
            optimizer.zero_grad(set_to_none=True)
            out, loss = _forward()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()
        else:
            with torch.no_grad():
                out, loss = _forward()

        # Convert predictions to °C for metrics
        if normalize:
            out_C = out * y_std + y_mu
        else:
            out_C = out
        mae_C = torch.mean(torch.abs(out_C - batch.y_C))

        total_mae_C += mae_C.item() * batch.num_graphs
        total_graphs += batch.num_graphs

    return total_mae_C / max(total_graphs, 1), total_graphs

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _has_edges(cif_path: str, cutoff: float) -> bool:
    try:
        s = Structure.from_file(cif_path)
        return bool(_radius_edges(s, cutoff))
    except Exception:
        return False

def _has_finite_label(v) -> bool:
    try:
        x = float(v); return math.isfinite(x)
    except Exception:
        return False

def _prefilter_splits(splits_dir: Path, outdir: Path, cutoff: float):
    """Drop rows with no neighbors at cutoff OR non-finite tc_C; return new dir with filtered CSVs."""
    filt_dir = outdir / "splits_filtered"
    filt_dir.mkdir(parents=True, exist_ok=True)
    skipped = {"no_neighbors": {}, "bad_label": {}}
    sizes = {}
    for name in ["train", "val", "test"]:
        df = pd.read_csv(splits_dir / f"{name}.csv").copy()

        keep_mask, bad_nn_ids, bad_y_ids = [], [], []
        for _, row in df.iterrows():
            y_ok = _has_finite_label(row["tc_C"])
            if not y_ok:
                keep_mask.append(False); bad_y_ids.append(int(row["sample_id"])); continue
            nn_ok = _has_edges(row["cif_path"], cutoff)
            if not nn_ok:
                keep_mask.append(False); bad_nn_ids.append(int(row["sample_id"])); continue
            keep_mask.append(True)

        out = df[keep_mask]
        out.to_csv(filt_dir / f"{name}.csv", index=False)
        sizes[name] = (len(df), len(out))
        if bad_nn_ids: skipped["no_neighbors"][name] = bad_nn_ids
        if bad_y_ids:  skipped["bad_label"][name]    = bad_y_ids

    (outdir / "skipped_samples.json").write_text(json.dumps(skipped, indent=2))
    print("Prefilter (strict):", {k: f"{v[1]}/{v[0]} kept" for k, v in sizes.items()})
    if any(skipped["bad_label"].values()):
        print("Note: dropped rows with non-finite tc_C (see skipped_samples.json).")
    return filt_dir

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
    ap.add_argument("--normalize-target", action="store_true",
                    help="standardize tc_C using TRAIN mean/std for loss; metrics remain in °C")
    args = ap.parse_args()

    set_seed(args.seed)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Strict prefilter
    splits_dir = Path(args.splits)
    filt_dir = _prefilter_splits(splits_dir, outdir, args.cutoff)

    # Compute normalization from TRAIN ONLY (to avoid leakage)
    train_df = pd.read_csv(filt_dir / "train.csv")
    y_train = pd.to_numeric(train_df["tc_C"], errors="coerce").dropna()
    y_mu = float(y_train.mean())
    y_std = float(y_train.std())
    if not math.isfinite(y_std) or y_std == 0.0: y_std = 1.0
    norm_info = {"enabled": bool(args.normalize_target), "mean_C": y_mu, "std_C": y_std}
    (outdir / "target_norm.json").write_text(json.dumps(norm_info, indent=2))

    # Save hparams
    h = vars(args).copy(); h.update({"y_mean_C": y_mu, "y_std_C": y_std})
    (outdir / "hparams.json").write_text(json.dumps(h, indent=2))

    # datasets & loaders (use filtered splits)
    pin = args.device.startswith("cuda")
    ds_kwargs = dict(cutoff=args.cutoff, n_gauss=args.n_gauss,
                     normalize=args.normalize_target, y_mu=y_mu, y_std=y_std)
    train_set = CrystalDataset(filt_dir / "train.csv", **ds_kwargs)
    val_set   = CrystalDataset(filt_dir / "val.csv",   **ds_kwargs)
    test_set  = CrystalDataset(filt_dir / "test.csv",  **ds_kwargs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)
    test_loader  = DataLoader(test_set,  batch_size=args.batch_size,
                              num_workers=args.workers, persistent_workers=(args.workers>0),
                              pin_memory=pin)

    # model/opt/amp
    model = SimpleCGCNN(emb_dim=args.emb_dim, n_layers=args.layers,
                        edge_dim=args.n_gauss, dropout=args.dropout).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    use_amp = bool(args.amp and args.device.startswith("cuda"))
    scaler = _amp.GradScaler("cuda", enabled=use_amp) if use_amp else None

    # metrics log
    metrics_path = outdir / "metrics.csv"
    metrics_path.write_text("epoch,train_mae,val_mae,secs,graphs_per_sec,best_val\n")

    best_val, best_path = float("inf"), outdir / "best.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_mae, tr_graphs = run_epoch(model, train_loader, optimizer=opt,
                                      device=args.device, scaler=scaler, use_amp=use_amp,
                                      normalize=args.normalize_target, y_mu=y_mu, y_std=y_std)
        va_mae, va_graphs = run_epoch(model, val_loader, optimizer=None,
                                      device=args.device, scaler=scaler, use_amp=use_amp,
                                      normalize=args.normalize_target, y_mu=y_mu, y_std=y_std)
        secs = time.time() - t0
        gps = (tr_graphs + va_graphs) / max(secs, 1e-6)
        if va_mae < best_val:
            best_val = va_mae
            torch.save(model.state_dict(), best_path)
        print(f"[{epoch:03d}] train MAE: {tr_mae:.2f} °C | val MAE: {va_mae:.2f} °C "
              f"| {secs:.1f}s/epoch | ~{gps:.1f} graphs/s | best val: {best_val:.2f} °C")
        with open(metrics_path, "a") as f:
            f.write(f"{epoch},{tr_mae:.6f},{va_mae:.6f},{secs:.3f},{gps:.2f},{best_val:.6f}\n")

    # test (load best)
    model.load_state_dict(torch.load(best_path, map_location=args.device))
    te_mae, _ = run_epoch(model, test_loader, optimizer=None,
                          device=args.device, scaler=scaler, use_amp=use_amp,
                          normalize=args.normalize_target, y_mu=y_mu, y_std=y_std)
    print(f"[TEST] MAE: {te_mae:.2f} °C")

    # predictions (in °C)
    model.eval()
    preds, targets, ids = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(args.device, non_blocking=True)
            with _amp.autocast("cuda", enabled=use_amp):
                out = model(batch)
            out_C = out * y_std + y_mu if args.normalize_target else out
            preds.extend(out_C.cpu().numpy().tolist())
            targets.extend(batch.y_C.cpu().numpy().tolist())
            ids.extend([getattr(g, "sample_id") for g in batch.to_data_list()])
    pd.DataFrame({"sample_id": ids, "tc_C_true": targets, "tc_C_pred": preds}) \
      .to_csv(outdir / "test_predictions.csv", index=False)
    print(f"Saved: {best_path}, {metrics_path}, target_norm.json, and test_predictions.csv")

if __name__ == "__main__":
    main()
