#!/usr/bin/env python3
"""
tc_postprocess.py — clean Curie-temperature rows for CGCNN training

What it does
1) Fix Kelvin/Celsius mix-ups by detecting ~273.15 K clusters *within each (ID, composition)*.
2) Filter to oxide-like ferroelectrics (configurable).
3) Exclude incipient/non-FE compositions (exact-name blacklist).
4) Enforce Tc plausibility window and required numeric Tc.
5) De-duplicate within each paper (ID) on (composition, Tc), keeping the most complete row.
6) Optional global de-duplication across papers on (composition, round(Tc, ndigits)).

Usage
  python tc_postprocess.py results.csv -o results_clean.csv
  # include non-oxide salts (e.g., KDP) if you want a broader set:
  python tc_postprocess.py results.csv -o results_clean.csv --no-oxide-only
  # enable global dedupe across papers (rounded to 1 decimal by default):
  python tc_postprocess.py results.csv -o results_clean.csv --global-dedupe --global-dedupe-dp 1
"""

import argparse, math, re, sys
from collections import defaultdict
from typing import List, Dict, Any, Tuple

import pandas as pd

NUMERIC_COLS = [
    "spontaneous_polarization",
    "remanent_polarization",
    "coercive_field",
    "curie_temperature",
    "dielectric_constant",
    "piezoelectric_constant",
]
AUX_COLS_FOR_COMPLETENESS = [
    "phase", "space_group", "lattice_parameters",
    "spontaneous_polarization", "remanent_polarization",
    "coercive_field", "dielectric_constant", "piezoelectric_constant"
]

DEFAULT_INCIPIENT = {"KTaO3", "SrTiO3", "CaTiO3"}  # strict equals
ANION_PATTERNS = ("CN", "NO3", "NO2", "SO4", "HSO4")
EXCLUDE_TOKENS = ("NH4", "·")  # ammonium, hydrates (dot-water)

def parse_args():
    p = argparse.ArgumentParser(description="Post-process Tc CSV for CGCNN.")
    p.add_argument("input", help="Input CSV (results.csv)")
    p.add_argument("-o", "--output", default="results_clean.csv", help="Output CSV path")
    p.add_argument("--oxide-only", action=argparse.BooleanOptionalAction, default=True,
                   help="Keep only oxide-like materials (default: True)")
    p.add_argument("--incipient", nargs="*", default=sorted(DEFAULT_INCIPIENT),
                   help="Exact formulas to exclude as incipient/non-FE (space-separated).")
    p.add_argument("--tc-min", type=float, default=-260.0, help="Minimum plausible Tc (°C)")
    p.add_argument("--tc-max", type=float, default=1200.0, help="Maximum plausible Tc (°C)")
    p.add_argument("--cluster-window", type=float, default=6.0,
                   help="Tolerance around 273.15 for K↔°C cluster detection")
    p.add_argument("--global-dedupe", action="store_true", help="Dedupe across papers globally")
    p.add_argument("--global-dedupe-dp", type=int, default=1,
                   help="Decimals to round Tc for global dedupe key (default: 1)")
    return p.parse_args()

def is_number(x) -> bool:
    return isinstance(x, (int, float)) and not (x is None or (isinstance(x, float) and math.isnan(x)))

def is_oxide_like(formula: str) -> bool:
    if not formula or "O" not in formula:
        return False
    if any(tok in formula for tok in EXCLUDE_TOKENS):
        return False
    if any(pat in formula for pat in ANION_PATTERNS):
        return False
    # Kick out obvious hydrogen-bonded salts (contains H with a digit)
    if re.search(r"H[0-9]", formula or ""):
        return False
    return True

def completeness_score(row: pd.Series) -> int:
    score = 0
    for c in AUX_COLS_FOR_COMPLETENESS:
        v = row.get(c, None)
        if pd.notna(v) and str(v).strip() != "":
            score += 1
    return score

def fix_kelvin_celsius_by_id(df: pd.DataFrame, tol: float) -> Tuple[pd.DataFrame, int]:
    """
    Within each (ID, chemical_composition), if a Tc has a partner lower value such that
    (tc_high - tc_low) ≈ 273.15 (±tol), convert the higher value to °C.
    """
    changed = 0
    if "ID" not in df.columns:
        return df, changed

    def convert_group(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal changed
        vals = [x for x in g["curie_temperature"].tolist() if is_number(x)]
        def should_convert(tc):
            if not is_number(tc):
                return False
            # convert this value if it looks like the "Kelvin" member of a 273.15-apart pair
            for v in vals:
                if is_number(v) and (tc - v) > 200 and abs((tc - v) - 273.15) < tol:
                    return True
            return False
        mask = g["curie_temperature"].apply(should_convert)
        changed += int(mask.sum())
        g.loc[mask, "curie_temperature"] = g.loc[mask, "curie_temperature"] - 273.15
        return g

    df = df.groupby(["ID", "chemical_composition"], dropna=False, as_index=False, group_keys=False).apply(convert_group)
    return df, changed


def fix_kelvin_celsius_global_by_composition(df: pd.DataFrame, tol: float) -> Tuple[pd.DataFrame, int]:
    """
    Across ALL papers, within each chemical_composition, convert values that are
    ≈273.15 higher than another value for the same composition.
    """
    changed = 0

    def convert_group(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal changed
        vals = [x for x in g["curie_temperature"].tolist() if is_number(x)]
        def should_convert(tc):
            if not is_number(tc):
                return False
            for v in vals:
                if is_number(v) and (tc - v) > 200 and abs((tc - v) - 273.15) < tol:
                    return True
            return False
        mask = g["curie_temperature"].apply(should_convert)
        changed += int(mask.sum())
        g.loc[mask, "curie_temperature"] = g.loc[mask, "curie_temperature"] - 273.15
        return g

    df = df.groupby(["chemical_composition"], dropna=False, as_index=False, group_keys=False).apply(convert_group)
    return df, changed



def filter_rows(df: pd.DataFrame, oxide_only: bool, incipient_set: set, tc_min: float, tc_max: float) -> Tuple[pd.DataFrame, Dict[str,int]]:
    reasons = {"no_tc":0, "incipient":0, "non_oxide":0, "out_of_range":0}
    keep_mask = []
    for _, r in df.iterrows():
        comp = (r.get("chemical_composition") or "").strip()
        tc = r.get("curie_temperature", None)

        if not is_number(tc):
            reasons["no_tc"] += 1
            keep_mask.append(False); continue

        if comp in incipient_set:
            reasons["incipient"] += 1
            keep_mask.append(False); continue

        if oxide_only and not is_oxide_like(comp):
            reasons["non_oxide"] += 1
            keep_mask.append(False); continue

        if tc < tc_min or tc > tc_max:
            reasons["out_of_range"] += 1
            keep_mask.append(False); continue

        keep_mask.append(True)

    return df[pd.Series(keep_mask, index=df.index)], reasons

def dedupe_within_paper(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    For each ID, keep one row per (composition, Tc), choosing the most complete row.
    """
    removed = 0
    if "ID" not in df.columns:
        return df, removed

    def choose(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal removed
        # rank by completeness
        g = g.copy()
        g["_score"] = g.apply(completeness_score, axis=1)
        # drop duplicates keeping idxmax by score
        # sort to ensure max score kept; in ties keep the first occurrence
        g = g.sort_values(["_score"], ascending=False)
        before = len(g)
        g = g.drop_duplicates(subset=["chemical_composition", "curie_temperature"], keep="first")
        removed += before - len(g)
        return g.drop(columns=["_score"])

    df = df.groupby("ID", as_index=False, group_keys=False).apply(choose)
    return df, removed

def global_dedupe(df: pd.DataFrame, dp: int) -> Tuple[pd.DataFrame, int]:
    """
    Dedupe across papers on (composition, round(Tc, dp)), keeping the row with the highest completeness.
    """
    removed = 0
    key_col = "_tc_key"
    df = df.copy()
    df[key_col] = df["curie_temperature"].round(dp)

    def choose(g: pd.DataFrame) -> pd.DataFrame:
        nonlocal removed
        g = g.copy()
        g["_score"] = g.apply(completeness_score, axis=1)
        g = g.sort_values(["_score"], ascending=False)
        before = len(g)
        g = g.drop_duplicates(subset=["chemical_composition", key_col], keep="first")
        removed += before - len(g)
        return g.drop(columns=["_score"])

    df = df.groupby(["chemical_composition", key_col], as_index=False, group_keys=False).apply(choose)
    return df.drop(columns=[key_col]), removed

def main():
    args = parse_args()

    # Read with strings, then coerce numeric columns
    df = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    input_cols = list(df.columns)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].replace({"": None, "null": None, "None": None}), errors="coerce")

    # 1) fix K↔°C clusters (per-ID, then global by composition)
    df, n_fixed_id  = fix_kelvin_celsius_by_id(df,  tol=args.cluster_window)
    df, n_fixed_comp = fix_kelvin_celsius_global_by_composition(df, tol=args.cluster_window)
    n_fixed = n_fixed_id + n_fixed_comp


    # 2) filter rows
    incipient_set = set(args.incipient)
    df, rm_reasons = filter_rows(
        df, oxide_only=args.oxide_only, incipient_set=incipient_set,
        tc_min=args.tc_min, tc_max=args.tc_max
    )

    # 3) dedupe within paper
    df, rm_dupes_in_paper = dedupe_within_paper(df)

    # 4) optional global dedupe
    rm_global = 0
    if args.global_dedupe:
        df, rm_global = global_dedupe(df, dp=args.global_dedupe_dp)

    # Ensure original column order; add back any missing numeric cols
    for c in NUMERIC_COLS:
        if c not in df.columns:
            df[c] = pd.NA
    # Reorder to input order if possible
    final_cols = [c for c in input_cols if c in df.columns] + [c for c in df.columns if c not in input_cols]
    df = df[final_cols]

    # Write
    df.to_csv(args.output, index=False)

    # Summary
    print("=== tc_postprocess summary ===")
    print(f"Input file      : {args.input}")
    print(f"Output file     : {args.output}")
    print(f"Rows in (raw)   : {sum(1 for _ in open(args.input, 'r', encoding='utf-8', errors='ignore')) - 1}")
    print(f"Kelvin→°C fixed : {n_fixed}  (per-ID: {n_fixed_id}, global-by-comp: {n_fixed_comp})")


    print("Removed counts  : " + ", ".join(f"{k}={v}" for k,v in rm_reasons.items()))
    print(f"Dedup (per ID)  : {rm_dupes_in_paper}")
    if args.global_dedupe:
        print(f"Dedup (global)  : {rm_global}")
    print(f"Rows out (clean): {len(df)}")

if __name__ == "__main__":
    sys.exit(main())
