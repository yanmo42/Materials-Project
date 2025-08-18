#!/usr/bin/env python3
"""
Create leakage-safe train/val/test splits for CGCNN.

Input : mp_cif_map.csv (from the linker), needs at least:
        row_idx, cif_path, tc_C, chemical_composition, mp_id, status
Output: splits/train.csv, splits/val.csv, splits/test.csv
        Each row: sample_id,cif_path,tc_C,chemical_composition,mp_id
"""

from __future__ import annotations
import argparse
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def _norm_comp(s: str) -> str:
    """Normalize a composition string for grouping."""
    return re.sub(r"\s+", "", str(s).strip()).lower()


def _write_split(df: pd.DataFrame, path: Path) -> None:
    keep = ["sample_id", "cif_path", "tc_C", "chemical_composition", "mp_id"]
    df[keep].to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser(description="Make group-aware splits for CGCNN.")
    ap.add_argument("--map", default="mp_cif_map.csv", help="Linker output CSV")
    ap.add_argument("--outdir", default="splits", help="Directory for output CSVs")
    ap.add_argument("--group-by", choices=["mp_id", "composition"], default="mp_id",
                    help="Grouping key used to avoid leakage across splits")
    ap.add_argument("--val-size", type=float, default=0.10, help="Fraction for validation set")
    ap.add_argument("--test-size", type=float, default=0.10, help="Fraction for test set")
    ap.add_argument("--min-cif", action="store_true",
                    help="Drop rows without valid CIF path; also require status == 'OK' if present")
    ap.add_argument("--check-exists", action="store_true",
                    help="Additionally require that CIF files exist on disk")
    ap.add_argument("--max-comp-l1", type=float, default=None,
                    help="Optional: keep only rows with comp_frac_l1 <= threshold (if column exists)")
    ap.add_argument("--seed1", type=int, default=42, help="Random seed for test split")
    ap.add_argument("--seed2", type=int, default=24, help="Random seed for val split")
    args = ap.parse_args()

    if not (0 < args.val_size < 1) or not (0 < args.test_size < 1) or args.val_size + args.test_size >= 1:
        raise SystemExit("val_size and test_size must be in (0,1) and sum to < 1.")

    df = pd.read_csv(args.map)

    # Minimal usability filtering
    if args.min_cif:
        if "status" in df.columns:
            df = df[df["status"] == "OK"].copy()
        df = df[df["cif_path"].notna()].copy()
        if args.check_exists:
            df = df[df["cif_path"].apply(lambda p: Path(str(p)).is_file())].copy()

    if args.max_comp_l1 is not None and "comp_frac_l1" in df.columns:
        comp_l1 = pd.to_numeric(df["comp_frac_l1"], errors="coerce")
        df = df[comp_l1.notna() & (comp_l1 <= args.max_comp_l1)].copy()

    # Ensure required columns
    if "row_idx" in df.columns:
        df["sample_id"] = df["row_idx"]
    else:
        df = df.reset_index().rename(columns={"index": "sample_id"})
    required = ["cif_path", "tc_C", "chemical_composition", "mp_id", "sample_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required columns in {args.map}: {missing}")

    # Build grouping labels (no leakage across these)
    if args.group_by == "mp_id":
        groups = df["mp_id"].astype(str)
    else:  # composition
        groups = df["chemical_composition"].map(_norm_comp)
    df["group_key"] = groups  # for diagnostics only

    # Need at least 3 distinct groups to split into train/val/test
    n_groups = df["group_key"].nunique()
    if n_groups < 3:
        raise SystemExit(f"Not enough unique groups ({n_groups}) for 3-way split. "
                         f"Try a different grouping or gather more data.")

    # Split: first carve out test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed1)
    train_val_idx, test_idx = next(gss1.split(df, groups=groups))
    train_val = df.iloc[train_val_idx].reset_index(drop=True)
    test = df.iloc[test_idx].reset_index(drop=True)

    # Split train vs val from remaining pool
    rel_val = args.val_size / (1.0 - args.test_size)  # relative to train_val pool
    gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_val, random_state=args.seed2)
    train_idx, val_idx = next(gss2.split(train_val, groups=train_val["group_key"]))
    train = train_val.iloc[train_idx].reset_index(drop=True)
    val = train_val.iloc[val_idx].reset_index(drop=True)

    # Write outputs
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    _write_split(train, outdir / "train.csv")
    _write_split(val,   outdir / "val.csv")
    _write_split(test,  outdir / "test.csv")

    # Diagnostics
    print(f"Saved splits to {outdir}/")
    print(f"train: {len(train)} | val: {len(val)} | test: {len(test)}")
    print(f"Unique groups — train:{train['group_key'].nunique()}, "
          f"val:{val['group_key'].nunique()}, test:{test['group_key'].nunique()}")
    tset, vset, sset = set(train["group_key"]), set(val["group_key"]), set(test["group_key"])
    print(f"Overlap train∩val:  {len(tset & vset)}")
    print(f"Overlap train∩test: {len(tset & sset)}")
    print(f"Overlap val∩test:   {len(vset & sset)}")


if __name__ == "__main__":
    main()
