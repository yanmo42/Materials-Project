#!/usr/bin/env python3
"""
Fetch relaxed crystal structures from Materials Project and save as CIF.
Reads MP_API_KEY from a local `.env` file (via python-dotenv).

CSV requirements:
  • must contain columns  'ID'  and  'chemical_composition'
  • 'ID' becomes <ID>.cif
"""

import csv, os, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv                 # NEW ←─────────────

from mp_api.client import MPRester
from pymatgen.core import Composition

# -------------------------------------------------------------------- config
CSV_IN  = Path("metadata_with_id.csv")         # change if needed
OUT_DIR = Path("cifs")                         # will be created
load_dotenv()                                  # NEW ←─────────────
API_KEY = os.getenv("MP_API_KEY")              # comes from .env
# ---------------------------------------------------------------------------

if not API_KEY:
    sys.exit("❌  No MP_API_KEY found.  Add it to .env or your shell env.")

OUT_DIR.mkdir(exist_ok=True)

def reduced(formula: str) -> str:
    """Return a reduced formula string like 'BaTiO3'."""
    return Composition(formula).reduced_formula

# --- 1. collect IDs that still need a CIF -----------------------------------
pending = defaultdict(list)
with CSV_IN.open() as f:
    reader = csv.DictReader(f)
    for row in reader:
        cif_path = OUT_DIR / f"{row['ID']}.cif"
        if cif_path.exists():
            continue
        pending[reduced(row["chemical_composition"])].append(row["ID"])

todo = sum(len(v) for v in pending.values())
print(f"Need to fetch {todo} CIFs for {len(pending)} unique formulas")

# --- 2. query Materials Project --------------------------------------------
with MPRester(API_KEY) as mpr:
    for formula, ids in tqdm(pending.items(), unit="formula"):
        docs = mpr.materials.summary.search(
            formula=formula,
            fields=["structure"],
            num_chunks=1,     # one relaxed structure is sufficient
        )
        if not docs:
            print(f"⚠️  no MP entry for {formula}")
            continue

        struct = docs[0].structure
        for sid in ids:
            struct.to(filename=OUT_DIR / f"{sid}.cif")

print("✅  Finished.  CIF folder now contains",
      len(list(OUT_DIR.glob('*.cif'))), "files.")
