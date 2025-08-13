#!/usr/bin/env python3
import os, csv, re, sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple

from dotenv import load_dotenv
from mp_api.client import MPRester
from pymatgen.core import Composition

CRYS_TOKENS = {"triclinic","monoclinic","orthorhombic","tetragonal","trigonal","hexagonal","cubic"}
COMP_SYNONYMS = {"chemical_composition","composition","formula","compound","material","chem_composition"}

def norm_phases(s: str) -> List[str]:
    if not s: return []
    toks = re.split(r"[,\s/;]+", s.lower())
    return [t for t in toks if t in CRYS_TOKENS]

def parse_formula(maybe: str) -> Optional[Composition]:
    if not maybe: return None
    try:
        return Composition(str(maybe).strip())
    except Exception:
        return None

def comp_to_chemsys(comp: Composition) -> str:
    els = sorted([str(el) for el in comp.elements])
    return "-".join(els)

def choose_best(docs, desired_phases: List[str]):
    if not docs: return None
    pool = docs
    if desired_phases:
        pool_phase = [d for d in docs
                      if getattr(d, "symmetry", None)
                      and getattr(d.symmetry, "crystal_system", None)
                      and d.symmetry.crystal_system.lower() in desired_phases]
        if pool_phase:
            pool = pool_phase
    def key(d):
        eah = float(getattr(d, "energy_above_hull", 1e9) or 1e9)
        stable_rank = 0 if getattr(d, "is_stable", False) else 1
        return (stable_rank, eah)
    return sorted(pool, key=key)[0]

def normalize_headers(fieldnames: List[str]) -> Tuple[Dict[str,str], List[str]]:
    mapping = {}
    normed = []
    for c in fieldnames:
        n = (c or "").replace("\ufeff","").strip().lower()
        mapping[n] = c
        normed.append(n)
    return mapping, normed

def autodetect_comp_col(row: Dict[str,str]) -> Optional[str]:
    # try any cell that parses as a sensible inorganic formula
    for k, v in row.items():
        if not isinstance(v, str): continue
        comp = parse_formula(v)
        if comp and len(comp.elements) > 1 and len(str(v)) <= 40:
            return k
    return None

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("props_csv", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("--id-col", default="ID")
    ap.add_argument("--comp-col", default="chemical_composition")
    ap.add_argument("--phase-col", default="phase")
    ap.add_argument("--max-per-formula", type=int, default=1)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--show-cols", action="store_true", help="Print detected CSV columns and exit")
    args = ap.parse_args()

    load_dotenv()
    api_key = os.getenv("MP_API_KEY") or os.getenv("PMG_MAPI_KEY")
    if not api_key:
        print("[!] Set MP_API_KEY (or PMG_MAPI_KEY) in your environment or .env", file=sys.stderr)
        sys.exit(2)

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    cifs_dir = out / "mp_cifs"; cifs_dir.mkdir(parents=True, exist_ok=True)
    fmap = open(out / "cif_map.csv", "w", newline=""); wmap = csv.writer(fmap); wmap.writerow(["id","cif_path"])
    log  = open(out / "fetch_log.txt", "w")

    with open(args.props_csv, newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            print("[!] CSV appears empty or malformed.", file=sys.stderr); sys.exit(2)
        header_map, cols_norm = normalize_headers(reader.fieldnames)
        if args.show_cols:
            print("Detected columns:")
            for raw in reader.fieldnames:
                print(f"- '{raw}'")
            return
        # map requested columns robustly
        def resolve_col(name: str, fallbacks: List[str]) -> Optional[str]:
            name_n = name.replace("\ufeff","").strip().lower()
            if name_n in header_map: return header_map[name_n]
            for fb in fallbacks:
                if fb in header_map: return header_map[fb]
            return None

        id_col = resolve_col(args.id_col, ["id","pdf_id","source_id"])
        phase_col = resolve_col(args.phase_col, ["phase","crystal_system"])
        comp_col = resolve_col(args.comp_col, list(COMP_SYNONYMS))

        if not id_col:
            print("[!] Could not find an ID column. Use --id-col to specify.", file=sys.stderr); sys.exit(2)
        if not comp_col:
            # we'll try per-row autodetect below
            print("[i] Composition column not found; will autodetect per row.", file=sys.stderr)

        chosen: Dict[Tuple[str,str], str] = {}
        kept = 0

        with MPRester(api_key) as mpr:
            for i, row in enumerate(reader):
                rid = (row.get(id_col) or "").strip()
                raw_comp = (row.get(comp_col) or "").strip() if comp_col else ""
                if not raw_comp:
                    # try autodetect
                    guess_key = autodetect_comp_col(row)
                    raw_comp = (row.get(guess_key) or "").strip() if guess_key else ""
                if not rid or not raw_comp:
                    log.write(f"[row {i}] skip: id='{rid}' comp='{raw_comp}' (missing)\n")
                    continue

                phases = norm_phases(row.get(phase_col, "")) if phase_col else []
                comp = parse_formula(raw_comp)
                if comp is None:
                    log.write(f"[row {i} id={rid}] unparseable comp='{raw_comp}'; skipping.\n")
                    continue

                reduced = comp.reduced_formula
                phase_sig = ",".join(phases) if phases else "*"
                cache_key = (reduced, phase_sig)
                if cache_key in chosen:
                    wmap.writerow([rid, chosen[cache_key]]); kept += 1
                    if args.debug: print(f"[reuse] {rid} <- {reduced} ({phase_sig})")
                    continue

                # Try formula search first
                try:
                    docs = mpr.materials.summary.search(
                        formula=reduced,
                        fields=["material_id","formula_pretty","structure","symmetry","is_stable","energy_above_hull"]
                    )
                except Exception as e:
                    docs = []; log.write(f"[row {i} id={rid}] formula search error: {e}\n")

                if not docs:
                    # Fallback: chemsys
                    cs = comp_to_chemsys(comp)
                    try:
                        docs = mpr.materials.summary.search(
                            chemsys=cs,
                            fields=["material_id","formula_pretty","structure","symmetry","is_stable","energy_above_hull"]
                        )
                        docs = [d for d in docs if getattr(d, "formula_pretty", None) == reduced]
                    except Exception as e:
                        docs = []; log.write(f"[row {i} id={rid}] chemsys search error: {e}\n")

                if not docs:
                    log.write(f"[row {i} id={rid}] no MP docs for '{reduced}'\n")
                    if args.debug: print(f"[miss] {rid} {reduced}")
                    continue

                doc = choose_best(docs, phases) or choose_best(docs, [])
                if not doc:
                    log.write(f"[row {i} id={rid}] docs found but none selected.\n")
                    continue

                cif_path = cifs_dir / f"{doc.material_id}.cif"
                try:
                    doc.structure.to(fmt="cif", filename=str(cif_path))
                except Exception as e:
                    log.write(f"[row {i} id={rid}] failed to write CIF: {e}\n")
                    continue

                chosen[cache_key] = str(cif_path)
                wmap.writerow([rid, cif_path]); kept += 1
                if args.debug:
                    phase = getattr(doc.symmetry, "crystal_system", None) if getattr(doc, "symmetry", None) else None
                    print(f"[hit] {rid} <- {reduced}  mpid={doc.material_id}  phase={phase}")

    fmap.close(); log.close()
    print(f"[OK] Downloaded {len(set(chosen.values()))} CIF(s) to {cifs_dir}")
    print(f"[OK] Wrote mapping: {out/'cif_map.csv'}")
    print(f"[i] Details in {out/'fetch_log.txt'}")
    print("Note: organics/true mixtures will still be skipped until you have ICSD CIFs.")
if __name__ == "__main__":
    main()
