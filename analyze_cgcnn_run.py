#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

# Use a non-interactive backend (safe on headless servers)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def mae(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.mean(np.abs(a-b)))
def rmse(a,b): a=np.asarray(a); b=np.asarray(b); return float(np.sqrt(np.mean((a-b)**2)))
def r2(y, yhat):
    y=np.asarray(y); yhat=np.asarray(yhat)
    ss_res = np.sum((y - yhat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

def pick_splits_dir(run_dir: Path, fallback: Path):
    cand = run_dir / "splits_filtered"
    return cand if (cand / "train.csv").exists() else fallback

def load_splits(splits_dir: Path):
    train = pd.read_csv(splits_dir / "train.csv")
    val   = pd.read_csv(splits_dir / "val.csv")
    test  = pd.read_csv(splits_dir / "test.csv")
    for df in (train, val, test):
        df["tc_C"] = pd.to_numeric(df["tc_C"], errors="coerce")
        df.dropna(subset=["tc_C"], inplace=True)
    return train, val, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/cgcnn", help="directory of a finished training run")
    ap.add_argument("--splits", default="splits", help="original splits dir (used if no filtered splits)")
    ap.add_argument("--outdir", default=None, help="where to write analysis (default: <run-dir>/analysis)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir) if args.outdir else (run_dir / "analysis")
    outdir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    preds_path   = run_dir / "test_predictions.csv"
    if not metrics_path.exists() or not preds_path.exists():
        raise SystemExit("Could not find metrics.csv or test_predictions.csv in run dir.")

    metrics = pd.read_csv(metrics_path)
    preds   = pd.read_csv(preds_path).dropna(subset=["tc_C_true","tc_C_pred"])

    splits_dir = pick_splits_dir(run_dir, Path(args.splits))
    train, val, test = load_splits(splits_dir)

    mu  = float(train["tc_C"].mean())
    med = float(train["tc_C"].median())
    base_val_mae_mean  = mae(val["tc_C"],  mu)
    base_test_mae_mean = mae(test["tc_C"], mu)
    base_val_mae_med   = mae(val["tc_C"],  med)
    base_test_mae_med  = mae(test["tc_C"], med)

    y_true = preds["tc_C_true"].to_numpy()
    y_pred = preds["tc_C_pred"].to_numpy()
    test_mae  = mae(y_true, y_pred)
    test_rmse = rmse(y_true, y_pred)
    test_r2   = r2(y_true, y_pred)
    spearman  = float(pd.Series(y_true).corr(pd.Series(y_pred), method="spearman"))

    best_val  = float(metrics["val_mae"].min()) if "val_mae" in metrics else float("nan")
    last_train = float(metrics["train_mae"].iloc[-1]) if "train_mae" in metrics else float("nan")
    last_val   = float(metrics["val_mae"].iloc[-1]) if "val_mae" in metrics else float("nan")

    imp_vs_mean = 100.0 * (1.0 - (test_mae / base_test_mae_mean)) if base_test_mae_mean>0 else float("nan")
    imp_vs_med  = 100.0 * (1.0 - (test_mae / base_test_mae_med))  if base_test_mae_med>0  else float("nan")

    # ---------- Plots ----------
    # 1) Pred vs True
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    pad = 0.05*(lims[1]-lims[0]) if lims[1]>lims[0] else 1.0
    lims = [lims[0]-pad, lims[1]+pad]

    fig = plt.figure(figsize=(6,6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot(lims, lims, linestyle="--")                 # y=x
    plt.hlines(mu, lims[0], lims[1], linestyles=":", linewidth=1)  # train mean baseline
    plt.xlabel("True Tc (°C)"); plt.ylabel("Predicted Tc (°C)")
    plt.title("Predicted vs True Tc")
    plt.xlim(lims); plt.ylim(lims); plt.tight_layout()
    fig.savefig(outdir / "pred_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2) Error histogram
    err = y_pred - y_true
    fig = plt.figure(figsize=(6,4))
    plt.hist(err, bins=40)
    plt.xlabel("Error (Pred - True) °C"); plt.ylabel("Count")
    plt.title("Prediction Error Histogram")
    plt.tight_layout()
    fig.savefig(outdir / "error_hist.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3) Residuals vs True
    fig = plt.figure(figsize=(6,4))
    plt.scatter(y_true, err, s=10, alpha=0.5)
    plt.hlines(0, xmin=y_true.min(), xmax=y_true.max(), linestyles="--")
    plt.xlabel("True Tc (°C)"); plt.ylabel("Residual (Pred - True) °C")
    plt.title("Residuals vs True Tc")
    plt.tight_layout()
    fig.savefig(outdir / "residuals_vs_true.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Summary ----------
    summary = {
        "splits_used": str(splits_dir),
        "counts": {
            "train": int(len(train)), "val": int(len(val)), "test": int(len(test)),
            "test_predictions": int(len(preds)),
        },
        "label_stats_train": {
            "mean_tc_C": mu, "median_tc_C": med, "std_tc_C": float(train["tc_C"].std())
        },
        "baselines": {
            "val_mae_mean":  base_val_mae_mean,  "test_mae_mean": base_test_mae_mean,
            "val_mae_median": base_val_mae_med,  "test_mae_median": base_test_mae_med
        },
        "model": {
            "last_train_mae": last_train, "last_val_mae": last_val, "best_val_mae": best_val,
            "test_mae": test_mae, "test_rmse": test_rmse, "test_r2": test_r2, "test_spearman": spearman,
            "improvement_vs_mean_baseline_pct": imp_vs_mean,
            "improvement_vs_median_baseline_pct": imp_vs_med
        }
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    lines = []
    lines.append(f"Splits used: {splits_dir}")
    lines.append(f"Counts — train:{len(train)}  val:{len(val)}  test:{len(test)}  (preds:{len(preds)})")
    lines.append(f"Train label mean/median/std (°C): {mu:.1f} / {med:.1f} / {float(train['tc_C'].std()):.1f}")
    lines.append("")
    lines.append("Baselines (predict constants from TRAIN labels):")
    lines.append(f"  Val MAE (mean):   {base_val_mae_mean:.1f} °C")
    lines.append(f"  Test MAE (mean):  {base_test_mae_mean:.1f} °C")
    lines.append(f"  Val MAE (median): {base_val_mae_med:.1f} °C")
    lines.append(f"  Test MAE (median):{base_test_mae_med:.1f} °C")
    lines.append("")
    lines.append("Model:")
    lines.append(f"  Last train/val MAE: {last_train:.1f} / {last_val:.1f} °C")
    lines.append(f"  Best val MAE:       {best_val:.1f} °C")
    lines.append(f"  Test MAE / RMSE:    {test_mae:.1f} / {test_rmse:.1f} °C")
    lines.append(f"  R2 / Spearman:      {test_r2:.3f} / {spearman:.3f}")
    lines.append(f"  Improvement vs mean baseline:   {imp_vs_mean:.1f}%")
    lines.append(f"  Improvement vs median baseline: {imp_vs_med:.1f}%")
    (outdir / "summary.txt").write_text("\n".join(lines))

    print("\n".join(lines))
    print(f"\nWrote plots to: {outdir}/pred_vs_true.png, error_hist.png, residuals_vs_true.png")
    print(f"Full summary:   {outdir}/summary.json")

if __name__ == "__main__":
    main()
