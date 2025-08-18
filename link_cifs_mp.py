#!/usr/bin/env python3
# link_cifs_mp.py — robust MP linker (handles NaN space_group; uses symmetry; optional ignore-spg)

from __future__ import annotations
import argparse, os, sys, time, math
from pathlib import Path
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Composition

# Optional .env support
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

USING_NEW = True
try:
    from mp_api.client import MPRester as MPResterNew
except Exception:
    USING_NEW = False
    from pymatgen.ext.matproj import MPRester as MPResterOld  # legacy

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

def as_comp_obj(doc) -> Composition | None:
    """Return a pymatgen.Composition for a SummaryDoc/dict, if possible."""
    # mp-api SummaryDoc often exposes a Composition already
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
    # dict fallback
    if isinstance(doc, dict):
        c = doc.get("composition")
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
    # last resort: pretty formula via attribute
    try:
        f = getattr(doc, "formula_pretty", None)
        if f:
            return Composition(f)
    except Exception:
        pass
    return None

def get_spacegroup_symbol(doc) -> str | None:
    """SummaryDoc exposes symmetry, not spacegroup. Prefer symmetry.symbol."""
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

def get_formula_pretty(doc) -> str | None:
    try:
        return getattr(doc, "formula_pretty")
    except Exception:
        return doc.get("formula_pretty") if isinstance(doc, dict) else None

def get_material_id(doc) -> str:
    try:
        return getattr(doc, "material_id")
    except Exception:
        return doc["material_id"]

def frac_distance(target: Composition, cand: Composition) -> float:
    t = target.fractional_composition.as_dict()
    c = cand.fractional_composition.as_dict()
    keys = set(t) | set(c)
    return sum(abs(t.get(k, 0.0) - c.get(k, 0.0)) for k in keys)

def best_candidate_for_comp(mpr, comp: Composition, spg_hint: str | None, use_spg: bool, max_results: int = 500):
    els = sorted([el.symbol for el in comp.elements])
    nelem = len(els)

    if USING_NEW:
        # Use modern route; request symmetry (not spacegroup)
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

    spg_hint_norm = safe_norm(spg_hint) if use_spg else ""
    best, best_score, best_parts = None, -1e9, None

    for d in docs:
        ccomp = as_comp_obj(d)
        if ccomp is None:
            continue

        l1 = frac_distance(comp, ccomp)
        base = 1.0 - min(l1, 1.0)

        spg_cand = safe_norm(get_spacegroup_symbol(d))
        spg_bonus = (0.05 if (use_spg and spg_hint_norm and spg_cand and spg_cand == spg_hint_norm) else 0.0)
        score = base + spg_bonus

        if score > best_score:
            best, best_score = d, score
            best_parts = {"comp_frac_l1": float(l1), "spg_match": bool(spg_bonus > 0), "score": float(score)}
    return best, best_parts

def save_cif(mpr, doc, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return True
    if USING_NEW:
        struct = getattr(doc, "structure", None)
        if struct is None:
            struct = mpr.materials.get_structure_by_material_id(get_material_id(doc))
    else:
        struct = mpr.get_structure_by_material_id(get_material_id(doc))
    cif_str = struct.to(fmt="cif")
    with open(out_path, "w") as f:
        f.write(cif_str)
    return True

def main():
    ap = argparse.ArgumentParser(description="Link cleaned Tc rows to Materials Project CIFs.")
    ap.add_argument("--in", dest="inp", default="cleaned_tc.csv")
    ap.add_argument("--out", dest="out", default="mp_cif_map.csv")
    ap.add_argument("--cif-dir", dest="cif_dir", default="cifs_mp")
    ap.add_argument("--api-key", dest="api_key", default=os.getenv("MP_API_KEY"))
    ap.add_argument("--max-results", type=int, default=300)
    ap.add_argument("--sleep", type=float, default=0.1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--ignore-spg", action="store_true",
                    help="Ignore space_group hints entirely (avoids any string-normalization issues).")
    args = ap.parse_args()

    if not args.api_key:
        print("ERROR: Provide Materials Project API key via --api-key or MP_API_KEY", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(args.inp, encoding="utf-8-sig")
    for col in ["chemical_composition", "tc_C"]:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    spg_present = ("space_group" in df.columns)
    use_spg = (spg_present and (not args.ignore_spg))

    if USING_NEW:
        mpr = MPResterNew(args.api_key)
    else:
        mpr = MPResterOld(args.api_key)

    out_rows = []
    rows = list(df.itertuples(index=True))
    if args.limit > 0:
        rows = rows[:args.limit]

    for row in tqdm(rows, total=len(rows), desc="Linking CIFs"):
        idx = row.Index
        comp_str = getattr(row, "chemical_composition")
        tc_C = float(getattr(row, "tc_C"))
        spg_hint = getattr(row, "space_group") if spg_present else None

        # Parse composition (supports Pb(Zr0.55Ti0.45)O3 etc.)
        try:
            target_comp = Composition(comp_str)
        except Exception:
            out_rows.append({"row_idx": idx, "chemical_composition": comp_str, "tc_C": tc_C,
                             "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                             "spg_match": None, "score": None, "cif_path": None,
                             "status": "UNPARSABLE_COMPOSITION"})
            continue

        try:
            doc, parts = best_candidate_for_comp(mpr, target_comp, spg_hint, use_spg, max_results=args.max_results)
        except Exception as e:
            out_rows.append({"row_idx": idx, "chemical_composition": comp_str, "tc_C": tc_C,
                             "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                             "spg_match": None, "score": None, "cif_path": None,
                             "status": f"SEARCH_ERROR:{e}"})
            continue

        if not doc:
            out_rows.append({"row_idx": idx, "chemical_composition": comp_str, "tc_C": tc_C,
                             "mp_id": None, "formula_pretty": None, "comp_frac_l1": None,
                             "spg_match": None, "score": None, "cif_path": None,
                             "status": "NO_CANDIDATE"})
            continue

        mpid = get_material_id(doc)
        formula_pretty = get_formula_pretty(doc)
        cif_path = Path(args.cif_dir) / f"{idx:07d}__{mpid}.cif"

        try:
            save_cif(mpr, doc, cif_path)
            status = "OK"
        except Exception as e:
            status = f"CIF_SAVE_ERROR:{e}"
            cif_path = None

        out_rows.append({"row_idx": idx, "chemical_composition": comp_str, "tc_C": tc_C,
                         "mp_id": mpid, "formula_pretty": formula_pretty,
                         **(parts or {"comp_frac_l1": None, "spg_match": None, "score": None}),
                         "cif_path": str(cif_path) if cif_path else None, "status": status})

        if args.sleep:
            time.sleep(args.sleep)

    pd.DataFrame(out_rows).to_csv(args.out, index=False)
    print(f"\nWrote mapping to {args.out}")
    print(f"CIFs in {args.cif_dir}/")
    unmatched = sum(1 for r in out_rows if r["status"] != "OK")
    print(f"Matched: {len(out_rows) - unmatched} / {len(out_rows)}  |  Unmatched: {unmatched}")

if __name__ == "__main__":
    main()
