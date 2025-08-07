#!/usr/bin/env python3
"""
Match (or fetch) CIF files for every row in your data table
and write CGCNN-ready metadata files.

INPUT CSV must contain at least:
  • chemical_composition   (string)
  • <target_property>      (e.g. curie_temperature, coercive_field …)

After this script runs you will have:
  • metadata_with_id.csv   (original rows + ID)
  • id_prop.csv            (ID, target)  → drop straight into CGCNN
  • cifs/<ID>.cif          (one CIF per sample)

Edit the CONFIG block below to fit your filenames.
"""

# ------------------------- CONFIG -------------------------------------------
CSV_IN       = "data.csv"             # original file, NO 'ID' column yet
TARGET_COL   = "curie_temperature"    # the property CGCNN will learn first
CIFS_DIR     = "cifs"                 # CIF folder to fill / reuse
ADD_MP_FETCH = True                   # set False to skip Materials Project
# ---------------------------------------------------------------------------

import os, re, csv, sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from dotenv import load_dotenv
import pandas as pd

from pymatgen.core import Composition, Structure

# --- load Materials-Project key ---------------------------------------------
load_dotenv()
MP_KEY = os.getenv("MP_API_KEY")
if ADD_MP_FETCH and not MP_KEY:
    print("⚠️  ADD_MP_FETCH=True but no MP_API_KEY found in .env; will skip MP fetch.")
    ADD_MP_FETCH = False

if ADD_MP_FETCH:
    from mp_api.client import MPRester
    mpr = MPRester(MP_KEY)

# --- helper functions --------------------------------------------------------
def norm_formula(text: str) -> str:
    """Return reduced formula string (e.g. 'BaTiO3')."""
    tokens = re.findall(r"[A-Za-z][a-z]?[\d.]*", text)
    if not tokens:
        raise ValueError(f"Cannot parse composition '{text}'")
    return Composition("".join(tokens)).reduced_formula

def save_structure(struct: Structure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    struct.to(fmt="cif", filename=path)

# --- load CSV, build formula set --------------------------------------------
df = pd.read_csv(CSV_IN)
if TARGET_COL not in df.columns:
    sys.exit(f"❌  TARGET_COL '{TARGET_COL}' not found in {CSV_IN}")

df["__formula"] = df["chemical_composition"].apply(norm_formula)
Path(CIFS_DIR).mkdir(exist_ok=True)

# --- index existing CIFs -----------------------------------------------------
formula2cifs: dict[str, list[Path]] = defaultdict(list)
for cif in Path(CIFS_DIR).rglob("*.cif"):
    try:
        f = Structure.from_file(cif).composition.reduced_formula
        formula2cifs[f].append(cif)
    except Exception:
        print("⚠️  unreadable CIF skipped:", cif.name)

# --- main loop: map each row to one CIF -------------------------------------
ids, missing = [], 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="matching"):
    formula = row["__formula"]
    cifs = formula2cifs.get(formula, [])

    # fetch from MP if none on disk
    if not cifs and ADD_MP_FETCH:
        try:
            docs = mpr.materials.summary.search(
                formula=formula,
                fields=["material_id", "structure"],
                num_chunks=1,
            )
            if docs:
                cid = docs[0].material_id
                cif_path = Path(CIFS_DIR) / f"{cid}.cif"
                save_structure(docs[0].structure, cif_path)
                cifs = [cif_path]
                formula2cifs[formula].append(cif_path)
        except Exception as e:
            print(f"⚠️  MP fetch failed for {formula}: {e}")

    if cifs:
        # pick the smallest cell if several
        chosen = min(
            cifs, key=lambda p: Structure.from_file(p).num_sites)
        ids.append(chosen.stem)
    else:
        ids.append(None)
        missing += 1

df["ID"] = ids
df = df.dropna(subset=["ID"]).reset_index(drop=True)
print(f"✅  Matched {len(df)} rows  |  {missing} rows still missing a CIF")

# --- clean numeric target ----------------------------------------------------
df[TARGET_COL] = (
    df[TARGET_COL]
    .a
