#!/usr/bin/env python3
"""
tc_cleaner.py — minimal Curie-temperature (°C) cleaner for CGCNN

Output columns (default):
  chemical_composition, phase, space_group, lattice_parameters, tc_C
Optional:
  --keep-id           -> adds 'id'
  --keep-source-year  -> extracts 'source_year' from 'id' like 'YYYY/HASH' if present

Notes
- Leaves duplicates intact (no dedupe, no n_obs).
- Keeps only what's relevant for CGCNN labeling and later CIF linkage.
- Curie temps remain in °C (no K conversion).
"""

import argparse
from pathlib import Path
import re
import pandas as pd

BASE_COLS = ["chemical_composition", "phase", "space_group", "lattice_parameters"]

def norm_header(c: str) -> str:
    return c.strip().lower().replace(" ", "_")

def try_extract_year(id_val: str):
    if not isinstance(id_val, str):
        return None
    m = re.match(r"^\s*(\d{4})\s*/", id_val)
    return int(m.group(1)) if m else None

def main():
    ap = argparse.ArgumentParser(description="Slim cleaner for Curie temperature (°C).")
    ap.add_argument("--in", dest="inp", required=True, help="Path to raw extractor CSV")
    ap.add_argument("--out", dest="out", default="cleaned_tc.csv", help="Output CSV path")
    ap.add_argument("--keep-id", action="store_true", help="Keep 'id' column if present")
    ap.add_argument("--keep-source-year", action="store_true",
                    help="Extract 'source_year' from 'id' like 'YYYY/HASH'")
    args = ap.parse_args()

    inp = Path(args.inp)
    if not inp.exists():
        raise SystemExit(f"Input not found: {inp}")

    df = pd.read_csv(inp, encoding="utf-8-sig", engine="python")
    df.columns = [norm_header(c) for c in df.columns]

    if "curie_temperature" not in df.columns:
        raise SystemExit("Missing required column 'curie_temperature' in input.")

    # Curie temperature in °C
    df["tc_C"] = pd.to_numeric(df["curie_temperature"], errors="coerce")
    df = df[df["tc_C"].notna()].copy()

    # Start with structural/context columns that actually exist
    keep = [c for c in BASE_COLS if c in df.columns] + ["tc_C"]

    # Optional provenance
    if args.keep_id and "id" in df.columns:
        keep.append("id")
    if args.keep_source_year and "id" in df.columns:
        df["source_year"] = df["id"].apply(try_extract_year)
        keep.append("source_year")

    # Always include composition if present (needed for later CIF resolution)
    if "chemical_composition" not in keep and "chemical_composition" in df.columns:
        keep.insert(0, "chemical_composition")

    df = df[keep]

    # Write
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"[tc_cleaner] kept rows : {len(df)}")
    print(f"[tc_cleaner] columns   : {list(df.columns)}")
    if len(df):
        print(f"[tc_cleaner] tc_C min/med/max: {df['tc_C'].min():.2f} / "
              f"{df['tc_C'].median():.2f} / {df['tc_C'].max():.2f}")
        print(df.head(5).to_string(index=False))

if __name__ == "__main__":
    main()
