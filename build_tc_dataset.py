#!/usr/bin/env python3
"""
build_tc_dataset.py — one-shot builder: results_clean.csv -> CGCNN dataset for Tc

Pipeline (single command):
  results_clean.csv
    -> validates & normalizes rows
    -> links each composition to a Materials Project structure (best candidate)
    -> saves CIFs into dataset/structures/<rowIdx>__<mpid>.cif
    -> writes id_prop.csv (id, Tc_C) or (id, Tc_K)
    -> writes splits/train_ids.csv, val_ids.csv, test_ids.csv
    -> writes meta/{link_map.csv, kept_rows.csv, dropped_placeholder_comps.csv, summary.txt}

Requirements:
  pip install pandas tqdm pymatgen mp-api scikit-learn python-dotenv

Usage:
  export MP_API_KEY="..."   # required unless you use --dry-run
  python build_tc_dataset.py results_clean.csv --out dataset_tc
  # Kelvin targets:
  python build_tc_dataset.py results_clean.csv --out dataset_tcK --kelvin
  # If you want to ignore SG hints (space_group) entirely:
  python build_tc_dataset.py results_clean.csv --out dataset_tc --ignore-spg
  # Throttle API calls:
  python build_tc_dataset.py results_clean.csv --out dataset_tc --sleep 0.1
  # Quick subset:
  python build_tc_dataset.py results_clean.csv --out dataset_tc --limit 100

Notes:
  - Excludes placeholder compositions like "Pb(Zr,Ti)O3" (commas in element list).
  - Uses the mp-api client when available; auto-falls back to legacy pymatgen MPRester.
  - Chooses the best MP candidate by stoichiometric closeness (+ small SG bonus if enabled).

CHANGELOG:
  - 2025-08-21: Robustified train/val/test splitting to avoid scikit-learn's
    "least populated class has only 1 member" error when stratifying by binned targets.
    The script now progressively reduces the number of quantile bins and falls back
    to random splits if stratification is not feasible.
"""

from __future__ import annotations
import argparse, os, sys, time, math, re, shutil
from pathlib import Path
from typing import Optional, Tuple, List

import pandas as pd
from tqdm import tqdm

# Optional .env support
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# sklearn is preferred; fall back to simple random splits if missing
try:
    from sklearn.model_selection import train_test_split
    HAVE_SK = True
except Exception:
    import random
    HAVE_SK = False

# pymatgen + MP API
try:
    from pymatgen.core import Composition
    HAVE_PM = True
except Exception:
    HAVE_PM = False

USING_NEW = True
try:
    # modern client
    from mp_api.client import MPRester as MPResterNew
except Exception:
    USING_NEW = False
    # legacy fallback
    try:
        from pymatgen.ext.matproj import MPRester as MPResterOld  # type: ignore
    except Exception:
        MPResterOld = None  # type: ignore


# ----------------------------- helpers -----------------------------

def safe_norm(s) -> str:
    """Normalize a value to a compact lowercase string; non-strings/NaN -> ''."""
    if s is None:
        return ""
    if isinstance(s, float) and (math.isnan(s) or not math.isfinite(s)):
        return ""
    s = str(s).strip()
    if s.lower() in ("", "nan", "none"):
        return ""
    return s.replace(" ", "").lower()

def get_spacegroup_symbol(doc) -> Optional[str]:
    """On mp-api SummaryDoc use symmetry.symbol. Dict fallback supported."""
    symm = None
    try:
        symm = getattr(doc, "symmetry", None)
    except Exception:
        symm = None
    if symm is None and isinstance(doc, dict):
        symm = doc.get("symmetry")
    if symm is None:
        return None
    # attr style
    try:
        symb = getattr(symm, "symbol")
        if symb:
            return str(symb)
    except Exception:
        pass
    # dict style
    if isinstance(symm, dict):
        return symm.get("symbol")
    return None

def get_formula_pretty(doc) -> Optional[str]:
    try:
        return getattr(doc, "formula_pretty")
    except Exception:
        return doc.get("formula_pretty") if isinstance(doc, dict) else None

def get_material_id(doc) -> str:
    try:
        return getattr(doc, "material_id")
    except Exception:
        return doc["material_id"]

def as_comp_obj(doc) -> Optional[Composition]:
    try:
        c = getattr(doc, "composition")
        if isinstance(c, Composition):
            return c
        try:
            return Composition(c)
        except Exception:
            pass
    except Exception:
        pass
    if isinstance(doc, dict):
        c = doc.get("composition")
        if c:
            try:
                return Composition(c)
            except Exception:
                pass
        f = doc.get("formula_pretty")
        if f:
            try:
                return Composition(f)
            except Exception:
                pass
    try:
        f = getattr(doc, "formula_pretty", None)
        if f:
            return Composition(f)
    except Exception:
        pass
    return None

def frac_distance(target: Composition, cand: Composition) -> float:
    t = target.fractional_composition.as_dict()
    c = cand.fractional_composition.as_dict()
    keys = set(t) | set(c)
    return sum(abs(t.get(k, 0.0) - c.get(k, 0.0)) for k in keys)

def best_candidate_for_comp(mpr, comp: Composition, spg_hint: Optional[str], use_spg: bool, max_results: int = 500):
    """Return (doc, parts) best match by stoichiometry + small SG bonus."""
    els = sorted([el.symbol for el in comp.elements])
    nelem = len(els)

    if USING_NEW:
        docs = mpr.materials.summary.search(
            elements=els,
            num_elements=nelem,
            fields=["material_id","formula_pretty","composition","symmetry","structure"],
            chunk_size=min(max_results, 500),
            num_chunks=1,
        )
        docs = list(docs)
    else:
        chemsys = "-".join(els)
        basic = mpr.query({"chemsys": chemsys},
                          properties=["material_id","pretty_formula","spacegroup","unit_cell_formula"])
        docs = []
        for d in basic[:max_results]:
            docs.append({
                "material_id": d["material_id"],
                "formula_pretty": d.get("pretty_formula"),
                "composition": d.get("unit_cell_formula") or {},
                "symmetry": {"symbol": ((d.get("spacegroup") or {}).get("symbol"))},
                "structure": None,
            })

    if not docs:
        return None, None

    spg_hint_norm = safe_norm(spg_hint) if (spg_hint and use_spg) else ""
    best, best_score, best_parts = None, -1e9, None
    for d in docs:
        ccomp = as_comp_obj(d)
        if ccomp is None:
            continue
        l1 = frac_distance(comp, ccomp)
        base = 1.0 - min(l1, 1.0)
        spg_cand = safe_norm(get_spacegroup_symbol(d))
        spg_bonus = 0.05 if (spg_hint_norm and spg_cand and spg_cand == spg_hint_norm) else 0.0
        score = base + spg_bonus
        if score > best_score:
            best, best_score = d, score
            best_parts = {"comp_frac_l1": float(l1), "spg_match": bool(spg_bonus > 0), "score": float(score)}
    return best, best_parts

def write_cif(mpr, doc, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    if USING_NEW:
        struct = getattr(doc, "structure", None)
        if struct is None:
            struct = mpr.materials.get_structure_by_material_id(get_material_id(doc))
    else:
        struct = mpr.get_structure_by_material_id(get_material_id(doc))  # type: ignore
    cif_str = struct.to(fmt="cif")
    with open(out_path, "w") as f:
        f.write(cif_str)
    return True

def sanitize_comp(s: str) -> str:
    return re.sub(r"\s+", "", str(s or ""))

def is_placeholder_comp(s: str) -> bool:
    # e.g., Pb(Zr,Ti)O3 — commas inside element list indicate unspecified ratios
    return bool(re.search(r"\([A-Za-z][A-Za-z]?,", str(s)))

def mk_sample_id(row_idx: int, mpid: str) -> str:
    return f"{int(row_idx):07d}__{mpid}"

# ------------------ robust splitting (fixes your error) ------------------

def _make_bins_safe(y: List[float], max_bins: int = 10) -> Tuple[Optional[pd.Series], Optional[int]]:
    """
    Create quantile bins with at least 2 members per bin for stratified splitting.
    If not possible, return (None, None) to signal no stratification.
    """
    n = len(y)
    if n < 4:  # too small to stratify meaningfully
        return None, None
    # start from sqrt(n), clamp to [2, max_bins]
    target_bins = min(max_bins, max(2, int(math.sqrt(n))))
    for b in range(target_bins, 1, -1):
        try:
            bins = pd.qcut(y, q=b, duplicates="drop")
        except Exception:
            continue
        counts = bins.value_counts()
        if (counts.min() >= 2) and (counts.nunique() > 1 or len(counts) > 1):
            return bins, len(counts)
    return None, None

def make_splits(ids: List[str], y: List[float], test_size: float, val_size: float, seed: int) -> Tuple[List[str], List[str], List[str], str]:
    """
    Robustly create train/val/test splits. Try stratified splits on binned targets; if that fails,
    progressively reduce bin count; finally fall back to random splits.

    Returns (train_ids, val_ids, test_ids, info_string).
    """
    info = []
    n = len(ids)

    if not HAVE_SK:
        # random fallback
        import random
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        n_test = int(round(test_size * n))
        n_val = int(round(val_size * n))
        ids_test = [ids[i] for i in order[:n_test]]
        ids_val  = [ids[i] for i in order[n_test:n_test+n_val]]
        ids_train= [ids[i] for i in order[n_test+n_val:]]
        info.append("Splits: random (scikit-learn not installed)")
        return ids_train, ids_val, ids_test, "; ".join(info)

    # --- First, make test split ---
    bins, nb = _make_bins_safe(y, max_bins=10)
    if bins is not None:
        try:
            ids_train, ids_test, y_train, y_test = train_test_split(
                ids, bins, test_size=test_size, random_state=seed, stratify=bins
            )
            info.append(f"Test split: stratified (bins={nb})")
        except ValueError:
            bins = None  # fall through to random
    if bins is None:
        # Random test split
        ids_train, ids_test = train_test_split(ids, test_size=test_size, random_state=seed, shuffle=True)
        # y_train only needed for val split
        y_train = [y[ids.index(i)] for i in ids_train]
        info.append("Test split: random (stratification not feasible)")

    # --- Now, make val split from remainder ---
    # Use the y values corresponding to ids_train
    y_train_numeric = y_train if isinstance(y_train, (list, pd.Series)) else [y[ids.index(i)] for i in ids_train]
    bins_val, nb_val = _make_bins_safe(y_train_numeric, max_bins=10)
    val_size_abs = int(round(val_size * n))
    if val_size_abs > 0 and len(ids_train) > 1:
        if bins_val is not None:
            try:
                ids_train2, ids_val, _, _ = train_test_split(
                    ids_train, bins_val, test_size=val_size_abs/len(ids_train),
                    random_state=seed, stratify=bins_val
                )
                ids_train = ids_train2
                info.append(f"Val split: stratified (bins={nb_val})")
            except ValueError:
                # random val split from the remainder
                ids_train, ids_val = train_test_split(ids_train, test_size=val_size_abs/len(ids_train),
                                                      random_state=seed, shuffle=True)
                info.append("Val split: random (stratification not feasible)")
        else:
            # random val split from the remainder
            ids_train, ids_val = train_test_split(ids_train, test_size=val_size_abs/len(ids_train),
                                                  random_state=seed, shuffle=True)
            info.append("Val split: random (stratification not feasible)")
    else:
        ids_val = []
        info.append("Val split: skipped (val_size=0 or too few samples)")

    return ids_train, ids_val, ids_test, "; ".join(info)

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Build a CGCNN Tc dataset from results_clean.csv in one step.")
    ap.add_argument("results_clean", help="CSV from tc_postprocess.py")
    ap.add_argument("--out", default="dataset_tc", help="Output dataset root")
    ap.add_argument("--kelvin", action="store_true", help="Write targets in Kelvin (Tc_K)")
    ap.add_argument("--ignore-spg", action="store_true", help="Ignore space_group hints entirely")
    ap.add_argument("--max-results", type=int, default=300, help="Max candidates per MP search")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep between MP lookups (seconds)")
    ap.add_argument("--limit", type=int, default=0, help="Process first N rows (0 = all)")
    ap.add_argument("--test-size", type=float, default=0.2, help="Test fraction")
    ap.add_argument("--val-size", type=float, default=0.1, help="Validation fraction (of remainder)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for splits")
    ap.add_argument("--dry-run", action="store_true", help="Parse & rank candidates but do not write CIFs/dataset")
    args = ap.parse_args()

    # Check MP availability
    api_key = os.getenv("MP_API_KEY", "").strip()
    if not api_key and not args.dry_run:
        print("ERROR: Provide Materials Project API key via MP_API_KEY or use --dry-run.", file=sys.stderr)
        sys.exit(1)
    if not HAVE_PM:
        print("ERROR: pymatgen not installed. pip install pymatgen", file=sys.stderr)
        sys.exit(1)
    if not USING_NEW and MPResterOld is None:
        print("ERROR: mp-api and legacy MPRester both unavailable.", file=sys.stderr)
        sys.exit(1)

    # IO dirs
    out = Path(args.out)
    structures_dir = out / "structures"
    splits_dir = out / "splits"
    meta_dir = out / "meta"
    for d in (out, structures_dir, splits_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Read input
    df = pd.read_csv(args.results_clean)
    # Required columns
    for col in ("chemical_composition", "curie_temperature"):
        if col not in df.columns:
            raise SystemExit(f"Missing required column in {args.results_clean}: {col}")

    # Normalize
    df["chemical_composition"] = df["chemical_composition"].astype(str).map(sanitize_comp)
    df["curie_temperature"] = pd.to_numeric(df["curie_temperature"], errors="coerce")

    # Drop invalid rows
    df = df[df["curie_temperature"].notna() & df["chemical_composition"].str.len().gt(0)].reset_index(drop=True)
    # Drop placeholder comps like Pb(Zr,Ti)O3
    mask_placeholder = df["chemical_composition"].map(is_placeholder_comp)
    df_bad = df[mask_placeholder].copy()
    df = df[~mask_placeholder].reset_index(drop=True)

    # Limit
    if args.limit > 0:
        df = df.iloc[:args.limit, :].reset_index(drop=True)

    # Open MP client
    if USING_NEW:
        mpr = MPResterNew(api_key)
    else:
        mpr = MPResterOld(api_key)  # type: ignore

    link_rows = []
    # Loop rows and link
    rows = list(df.itertuples(index=True))
    for row in tqdm(rows, total=len(rows), desc="Linking MP structures"):
        idx = row.Index
        comp_str = row.chemical_composition
        tc_C = float(row.curie_temperature)
        spg_hint = getattr(row, "space_group", None)
        use_spg = (not args.ignore_spg) and ("space_group" in df.columns)

        # Parse composition (supports Pb(Zr0.52Ti0.48)O3, etc.)
        try:
            target_comp = Composition(comp_str)
        except Exception:
            link_rows.append({
                "row_idx": idx, "chemical_composition": comp_str, "Tc_C": tc_C,
                "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                "spg_match": None, "score": None, "cif_path": None, "status": "UNPARSABLE_COMPOSITION"
            })
            continue

        # Search & choose best
        try:
            doc, parts = best_candidate_for_comp(mpr, target_comp, spg_hint, use_spg, max_results=args.max_results)
        except Exception as e:
            link_rows.append({
                "row_idx": idx, "chemical_composition": comp_str, "Tc_C": tc_C,
                "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                "spg_match": None, "score": None, "cif_path": None, "status": f"SEARCH_ERROR:{e}"
            })
            continue

        if not doc:
            link_rows.append({
                "row_idx": idx, "chemical_composition": comp_str, "Tc_C": tc_C,
                "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                "spg_match": None, "score": None, "cif_path": None, "status": "NO_CANDIDATE"
            })
            continue

        mpid = get_material_id(doc)
        formula_pretty = get_formula_pretty(doc)
        sample_id = mk_sample_id(idx, mpid)
        cif_dst = structures_dir / f"{sample_id}.cif"

        status = "OK"
        if not args.dry_run:
            try:
                write_cif(mpr, doc, cif_dst)
            except Exception as e:
                status = f"CIF_SAVE_ERROR:{e}"
                cif_dst = None

        link_rows.append({
            "row_idx": idx, "chemical_composition": comp_str, "Tc_C": tc_C,
            "mp_id": mpid, "formula_pretty": formula_pretty,
            **(parts or {"comp_frac_l1": None, "spg_match": None, "score": None}),
            "cif_path": str(cif_dst) if cif_dst else None, "status": status
        })

        if args.sleep:
            time.sleep(args.sleep)

    link_df = pd.DataFrame(link_rows)
    link_df.to_csv(meta_dir / "link_map.csv", index=False)

    # Keep only OK rows with existing CIFs (unless dry_run)
    if args.dry_run:
        ok_df = link_df[link_df["status"] == "OK"].copy()
    else:
        ok_df = link_df[(link_df["status"] == "OK") & link_df["cif_path"].notna()].copy()
        ok_df = ok_df[ok_df["cif_path"].map(lambda p: Path(str(p)).exists())]

    # Build id_prop
    ids, tcs = [], []
    for r in ok_df.itertuples(index=False):
        sample_id = mk_sample_id(int(r.row_idx), str(r.mp_id))
        ids.append(sample_id)
        tcs.append(float(r.Tc_C))

    if not ids:
        print("No CIFs linked successfully. See meta/link_map.csv and check your MP_API_KEY / network.", file=sys.stderr)
        # Write diagnostics
        if not df_bad.empty:
            df_bad.to_csv(meta_dir / "dropped_placeholder_comps.csv", index=False)
        sys.exit(2)

    idprop = pd.DataFrame({"id": ids, "Tc_C": tcs})
    target_col = "Tc_K" if args.kelvin else "Tc_C"
    if args.kelvin:
        idprop["Tc_K"] = idprop["Tc_C"] + 273.15
    idprop[["id", target_col]].to_csv(out / "id_prop.csv", index=False)

    # Splits (robust, stratified when feasible)
    train_ids, val_ids, test_ids, split_info = make_splits(
        ids=ids, y=idprop[target_col].tolist(),
        test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    pd.Series(train_ids).to_csv(splits_dir / "train_ids.csv", index=False, header=False)
    pd.Series(val_ids).to_csv(  splits_dir / "val_ids.csv",   index=False, header=False)
    pd.Series(test_ids).to_csv( splits_dir / "test_ids.csv", index=False, header=False)

    # Meta
    kept_rows = df.iloc[[int(i.split("__")[0]) for i in ids if i.split("__")[0].isdigit()]].copy()
    kept_rows.to_csv(meta_dir / "kept_rows.csv", index=False)
    if not df_bad.empty:
        df_bad.to_csv(meta_dir / "dropped_placeholder_comps.csv", index=False)

    # Summary
    unmatched = (link_df["status"] != "OK").sum()
    summary = [
        f"Input rows (after basic checks): {len(df)}",
        f"Linked OK                       : {len(ok_df)}",
        f"Unmatched/Errors                : {int(unmatched)}",
        f"Targets unit                    : {'Kelvin' if args.kelvin else 'Celsius'}",
        f"Splits (train/val/test)         : {len(train_ids)}/{len(val_ids)}/{len(test_ids)}",
        f"Split strategy                  : {split_info}",
        f"Dataset root                    : {out.resolve()}",
    ]
    (meta_dir / "summary.txt").write_text("\n".join(summary), encoding="utf-8")
    print("\n".join(summary))
    if args.dry_run:
        print("\nDRY RUN: CIFs not written; inspect meta/link_map.csv for candidate MP IDs.")

if __name__ == "__main__":
    sys.exit(main())
