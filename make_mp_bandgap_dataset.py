#!/usr/bin/env python3
"""
make_mp_bandgap_dataset.py
Create a CGCNN-ready band-gap dataset from Materials Project.

Output layout:
  <outdir>/
    cifs/<material_id>.cif
    meta.csv                 # id, formula, reduced_formula, band_gap, energy_above_hull, cif_path
    train.csv                # id, band_gap, cif_path   (grouped by reduced_formula)
    val.csv
    test.csv
    README.md

Usage:
  export MP_API_KEY="..."
  python make_mp_bandgap_dataset.py --outdir datasets/mp-bandgap-v1 --n-max 50000 --stable-only --exclude-metals --e-above-hull-max 0.03
"""
import argparse
import os
from math import ceil
from pathlib import Path

import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester
from pymatgen.core import Structure, Composition
from sklearn.model_selection import GroupShuffleSplit

from dotenv import load_dotenv
load_dotenv()

FIELDS = [
    "material_id",
    "formula_pretty",
    "band_gap",
    "is_stable",
    "energy_above_hull",
    "structure",
    "is_gap_direct",
    "deprecated",
]


def write_readme(outdir, n_attempted_fetch, n_kept, args):
    text = f"""# MP band-gap dataset

Attempted to fetch (max): {n_attempted_fetch}
Kept after filters & writing: {n_kept}

Filters:
- stable_only: {args.stable_only}
- exclude_metals: {args.exclude_metals}
- band_gap range: [{args.min_gap}, {args.max_gap if args.max_gap is not None else 'None'}] eV
- energy_above_hull_max: {args.e_above_hull_max}
- n_max: {args.n_max if args.n_max else 'None'}
- chunk_size: {args.chunk_size}

Notes:
- We request only a limited number of chunks from the API, so we don't download the entire corpus.
- Splits are **grouped by reduced_formula** to avoid polymorph leakage.
"""
    (outdir / "README.md").write_text(text)


def group_splits(df, train_size=0.8, val_size=0.1, seed=123):
    # Split by reduced_formula grouping
    gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=seed)
    train_idx, temp_idx = next(gss.split(df, groups=df["reduced_formula"]))
    train_df, temp_df = df.iloc[train_idx].copy(), df.iloc[temp_idx].copy()

    remain = val_size / (1 - train_size)
    gss2 = GroupShuffleSplit(n_splits=1, train_size=remain, random_state=seed + 1)
    val_idx, test_idx = next(gss2.split(temp_df, groups=temp_df["reduced_formula"]))
    val_df, test_df = temp_df.iloc[val_idx].copy(), temp_df.iloc[test_idx].copy()
    return train_df, val_df, test_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-key", default=os.getenv("MP_API_KEY"), help="Materials Project API key (or set MP_API_KEY).")
    ap.add_argument("--outdir", required=True, type=Path)
    ap.add_argument("--n-max", type=int, default=None, help="Write at most this many rows (also limits fetch size).")
    ap.add_argument("--chunk-size", type=int, default=1000, help="Results per chunk/page to fetch from API.")
    ap.add_argument("--stable-only", action="store_true", help="Keep only is_stable=True & energy_above_hull <= threshold.")
    ap.add_argument("--e-above-hull-max", type=float, default=0.03, help="Max energy_above_hull (eV/atom) if --stable-only.")
    ap.add_argument("--exclude-metals", action="store_true", help="Drop band_gap <= 0.")
    ap.add_argument("--min-gap", type=float, default=0.0)
    ap.add_argument("--max-gap", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    if not args.api_key:
        raise SystemExit("No API key. Provide --api-key or set MP_API_KEY.")

    outdir = args.outdir
    (outdir / "cifs").mkdir(parents=True, exist_ok=True)

    # Build server-side filters to ensure `fields` is respected and to avoid deprecated docs
    query_params = {
        "deprecated": False,  # exclude deprecated docs
    }
    # Band gap filter (also acts as "non-empty query" so 'fields' is honored)
    if args.exclude_metals:
        low = max(args.min_gap, 1e-6)
        high = args.max_gap if args.max_gap is not None else None
        query_params["band_gap"] = (low, high)
    else:
        query_params["band_gap"] = (args.min_gap, args.max_gap) if args.max_gap is not None else (args.min_gap, None)

    if args.stable_only:
        query_params["is_stable"] = True
        query_params["energy_above_hull"] = (0, args.e_above_hull_max)

    # Decide how many chunks to fetch to honor n_max without pulling the full corpus
    if args.n_max is not None and args.n_max > 0:
        num_chunks = ceil(args.n_max / args.chunk_size)
        max_write = args.n_max
    else:
        num_chunks = None
        max_write = None  # no cap
    attempted_fetch = (num_chunks or 0) * args.chunk_size if num_chunks else "all"

    rows = []

    # Use raw dicts (no Pydantic validation) to avoid NaN type validation errors in deprecated/odd docs.
    with MPRester(args.api_key, monty_decode=False, use_document_model=False) as mpr:
        docs = mpr.materials.summary.search(
            fields=FIELDS,
            chunk_size=args.chunk_size,
            num_chunks=num_chunks,
            **query_params,
        )

        # Iterate with a progress bar and write CIFs + rows until we hit max_write
        for d in tqdm(docs, desc="Filtering + writing CIFs"):
            # d is a dict because use_document_model=False
            if d.get("deprecated", False):
                continue

            bg = d.get("band_gap", None)
            if bg is None:
                continue
            # Extra sanity check for metals even if we filtered server-side
            if args.exclude_metals and (bg is None or bg <= 0):
                continue
            if bg < args.min_gap:
                continue
            if args.max_gap is not None and bg > args.max_gap:
                continue
            if args.stable_only:
                if not d.get("is_stable", False):
                    continue
                eah = d.get("energy_above_hull", None)
                if eah is not None and eah > args.e_above_hull_max:
                    continue

            struct_obj = d.get("structure", None)
            if not struct_obj:
                continue

            # Write CIF (skip if already present)
            mid = d["material_id"]
            cif_path = outdir / "cifs" / f"{mid}.cif"
            if not cif_path.exists():
                try:
                    Structure.from_dict(struct_obj).to(filename=str(cif_path))
                except Exception:
                    # Skip malformed structures just in case
                    continue

            formula = d.get("formula_pretty", "")
            try:
                rf = Composition(formula).reduced_formula
            except Exception:
                # Fallback if formula parsing fails
                rf = formula or str(mid)

            rows.append({
                "id": mid,
                "formula": formula,
                "reduced_formula": rf,
                "band_gap": float(bg),
                "energy_above_hull": float(d["energy_above_hull"]) if d.get("energy_above_hull") is not None else None,
                "is_gap_direct": bool(d["is_gap_direct"]) if d.get("is_gap_direct") is not None else None,
                "cif_path": str(cif_path)
            })

            if max_write is not None and len(rows) >= max_write:
                break

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("No rows after filtering. Loosen filters or check API key/quota.")

    df.to_csv(outdir / "meta.csv", index=False)

    # Grouped splits (by reduced formula) to prevent polymorph leakage
    train_df, val_df, test_df = group_splits(df, train_size=0.8, val_size=0.1, seed=args.seed)
    for name, part in [("train", train_df), ("val", val_df), ("test", test_df)]:
        part[["id", "band_gap", "cif_path"]].to_csv(outdir / f"{name}.csv", index=False)

    write_readme(outdir, n_attempted_fetch=attempted_fetch, n_kept=len(df), args=args)
    print(f"Wrote dataset to: {outdir}")

    # Print summary of numeric columns (compatible across pandas versions)
    num_df = df.select_dtypes(include=["number"])
    if not num_df.empty:
        print(num_df.describe())
    else:
        print("No numeric columns to summarize.")


if __name__ == "__main__":
    main()
