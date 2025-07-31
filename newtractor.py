#!/usr/bin/env python3
# (shortened header text omitted for brevity — unchanged except for new flags doc)

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
from google import genai
from google.genai import types
from tqdm import tqdm
from colorama import init as colorama_init, Fore, Style

# -----------------------------------------------------------------------------
# Schema & constants
# -----------------------------------------------------------------------------
class FerroelectricMaterial(BaseModel):
    chemical_composition: Optional[str] = None
    phase: Optional[str] = None
    space_group: Optional[str] = None
    spontaneous_polarization: Optional[str] = None
    remanent_polarization: Optional[str] = None
    coercive_field: Optional[str] = None
    curie_temperature: Optional[str] = None
    dielectric_constant: Optional[str] = None
    piezoelectric_constant: Optional[str] = None


PROPERTIES_AND_EXPLANATIONS: dict[str, str] = {
    "chemical_composition":    "Material formula, e.g. BaTiO3.",
    "phase":                   "Crystallographic phase (cubic, tetragonal, orthorhombic, or rhombohedral), e.g. tetragonal.",
    "space_group":             "Symmetry space group, e.g. Pm3m, I4/mmm.",
    "lattice_parameters":      "Lattice vector lengths (a, b, c) in Å, e.g. 3.99, 3.99, 4.00.",
    "spontaneous_polarization": "Spontaneous polarization in µC/cm².",
    "remanent_polarization":    "Remanent polarization in µC/cm².",
    "coercive_field":           "Coercive field in kV/cm.",
    "curie_temperature":        "Curie temperature in °C (if in K, subtract 273.15).",
    "dielectric_constant":      "Relative permittivity (dimensionless).",
    "piezoelectric_constant":   "Piezoelectric coefficient in pC/N.",
}

# Example a single CSV row matching the above order:
EXAMPLE_ROW = [
    "BaTiO3",     # chemical_composition
    "tetragonal", # phase
    "P4mm",       # space_group
    "3.99,3.99,4.00", # lattice_parameters
    "26",         # spontaneous_polarization
    "17",         # remanent_polarization
    "1.8",        # coercive_field
    "120",        # curie_temperature
    "1200",       # dielectric_constant
    "190",        # piezoelectric_constant
]

# Full combined prompt
SYSTEM_INSTRUCTIONS_FULL = """
You are an expert at structured data extraction for ferroelectric materials.
Extract exactly these fields from each paper:

• chemical_composition      – Material formula (e.g. BaTiO3)
• phase                     – Crystallographic phase (cubic, tetragonal, orthorhombic, or rhombohedral)
• space_group               – Symmetry space group (e.g. Pm3m, I4/mmm)
• lattice_parameters        – Lattice vector lengths a,b,c in Å (e.g. 3.99,3.99,4.00)
• spontaneous_polarization  – in µC/cm²
• remanent_polarization     – in µC/cm²
• coercive_field            – in kV/cm
• curie_temperature         – in °C  (if provided in K, subtract 273.15)
• dielectric_constant       – relative permittivity (dimensionless)
• piezoelectric_constant    – in pC/N

Return a JSON array of objects matching that schema.
• Use standard chemical-formula notation for compositions.
• Convert all units as specified.
• If a property isn’t clearly reported, leave it blank.

Example CSV header + one row:
```
chemical_composition,phase,space_group,lattice_parameters,spontaneous_polarization,remanent_polarization,coercive_field,curie_temperature,dielectric_constant,piezoelectric_constant
BaTiO3,tetragonal,P4mm,3.99,3.99,4.00,26,17,1.8,120,1200,190
```
"""

# Functional-only prompt
SYSTEM_INSTRUCTIONS_FUNCTIONAL = """
You are an expert at structured data extraction for ferroelectric materials.
Extract exactly these functional properties from each paper:

• chemical_composition      – Material formula (e.g. BaTiO3)
• spontaneous_polarization  – in µC/cm²
• remanent_polarization     – in µC/cm²
• coercive_field            – in kV/cm
• curie_temperature         – in °C  (if provided in K, subtract 273.15)
• dielectric_constant       – relative permittivity (dimensionless)
• piezoelectric_constant    – in pC/N

Return a JSON array of objects with only these fields (plus ID downstream).
• Convert all units as specified.
• If a property isn’t clearly reported, leave it blank.

Example CSV header + one row:
```
chemical_composition,spontaneous_polarization,remanent_polarization,coercive_field,curie_temperature,dielectric_constant,piezoelectric_constant
BaTiO3,26,17,1.8,120,1200,190
```
"""

# Structural-only prompt
SYSTEM_INSTRUCTIONS_STRUCTURAL = """
You are an expert at structured data extraction for crystallography.
Extract exactly these structural fields from each paper:

• chemical_composition      – Material formula (e.g. BaTiO3)
• phase                     – Crystallographic phase (cubic, tetragonal, orthorhombic, or rhombohedral)
• space_group               – Symmetry space group (e.g. Pm3m, I4/mmm)
• lattice_parameters        – Lattice vector lengths a,b,c in Å (e.g. 3.99,3.99,4.00)

Return a JSON array of objects with only these fields (plus ID downstream).
• Use standard chemical-formula notation for compositions.
• If a property isn’t clearly reported, leave it blank.

Example CSV header + one row:
```
chemical_composition,phase,space_group,lattice_parameters
BaTiO3,tetragonal,P4mm,3.99,3.99,4.00
```
"""

# Aliased default (can be overridden in main)
SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_FULL


# -----------------------------------------------------------------------------
# Input walking & selection
# -----------------------------------------------------------------------------
@dataclass
class WorkItem:
    year: str
    folder: Path
    mode: str   # "json" or "pdf"
    path: Path


def iter_hash_folders(root: Path) -> Iterable[Path]:
    if not root.exists():
        logging.warning(f"Root directory does not exist: {root}")
        return
    for year_dir in sorted(root.iterdir()):
        if not year_dir.is_dir():
            continue
        for hash_dir in sorted(year_dir.iterdir()):
            if hash_dir.is_dir():
                yield hash_dir


def to_id(hash_dir: Path) -> str:
    return f"{hash_dir.parent.name}/{hash_dir.name}"


def choose_input(folder: Path) -> tuple[Optional[str], Optional[Path]]:
    jsons = list(folder.glob("*.json"))
    if jsons:
        jsons.sort(key=lambda p: p.stat().st_size, reverse=True)
        return "json", jsons[0]
    pdfs = list(folder.glob("*.pdf"))
    if pdfs:
        pdfs.sort(key=lambda p: p.stat().st_size, reverse=True)
        return "pdf", pdfs[0]
    return None, None


def load_text_from_json(path: Path) -> str:
    try:
        obj = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        logging.warning(f"Failed to parse JSON from {path}: {e}")
        return ""

    # handle different JSON schemas
    if isinstance(obj, dict) and isinstance(obj.get("page_texts"), list):
        parts = []
        for i, page in enumerate(obj["page_texts"], 1):
            if isinstance(page, str) and page.strip():
                parts.append(f"\n<page {i}>\n{page.strip()}\n</page {i}>")
        return "\n".join(parts)

    pages = obj.get("pages") if isinstance(obj, dict) else None
    if isinstance(pages, list):
        parts = []
        for i, p in enumerate(pages, 1):
            txt = p.get("text") if isinstance(p, dict) else None
            if isinstance(txt, str) and txt.strip():
                parts.append(f"\n<page {i}>\n{txt.strip()}\n</page {i}>")
        if parts:
            return "\n".join(parts)

    def gather(x) -> list[str]:
        out = []
        if isinstance(x, str):
            s = x.strip()
            if s:
                out.append(s)
        elif isinstance(x, dict):
            for v in x.values():
                out.extend(gather(v))
        elif isinstance(x, list):
            for v in x:
                out.extend(gather(v))
        return out

    return "\n".join(gather(obj))

# -----------------------------------------------------------------------------
# LLM calls
# -----------------------------------------------------------------------------
def init_client(api_key: str) -> genai.Client:
    return genai.Client(api_key=api_key)


def extract_with_text(client: genai.Client, model: str, text: str) -> str:
    content = f"<paper_start>\n{text}\n<paper_end>"
    try:
        resp = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=list[FerroelectricMaterial],
                temperature=0.0,                # <- add this

            ),
            contents=content,
        )
        return resp.text or "[]"
    except Exception as e:
        logging.error(f"Error in extract_with_text: {e}")
        return "[]"


def extract_with_pdf(client: genai.Client, model: str, pdf_bytes: bytes) -> str:
    try:
        resp = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTIONS,
                response_mime_type="application/json",
                response_schema=list[FerroelectricMaterial],
                temperature=0.0,                # <- add this

            ),
            contents=[
                types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
                "Extract the specified properties for any ferroelectric materials mentioned.",
            ],
        )
        return resp.text or "[]"
    except Exception as e:
        logging.error(f"Error in extract_with_pdf: {e}")
        return "[]"

# -----------------------------------------------------------------------------
# Core processing
# -----------------------------------------------------------------------------
def row_has_any_value(row: pd.Series, cols: list[str]) -> bool:
    for c in cols:
        v = row.get(c)
        if isinstance(v, str) and v.strip():
            return True
        if v is not None and not pd.isna(v):
            return True
    return False


def count_rows_with_noncomp_props(df_local: pd.DataFrame, property_cols: list[str]) -> int:
    noncomp = [c for c in property_cols if c != "chemical_composition"]
    if df_local.empty:
        return 0
    mask = df_local.apply(lambda r: row_has_any_value(r, noncomp), axis=1)
    return int(mask.sum())


def clean_to_rows(raw_json: str, property_cols: list[str]) -> pd.DataFrame:
    try:
        out = json.loads(raw_json)
    except json.JSONDecodeError as e:
        logging.warning(f"Failed to parse JSON response: {e}")
        return pd.DataFrame()

    if not isinstance(out, list):
        logging.warning("JSON response is not a list")
        return pd.DataFrame()

    df_local = pd.DataFrame(out)
    if df_local.empty:
        return df_local

    # ensure all columns exist
    for c in property_cols:
        if c not in df_local.columns:
            df_local[c] = None

    # clean text
    for c in df_local.select_dtypes(include=["object"]).columns:
        df_local[c] = df_local[c].apply(lambda x: x.strip() if isinstance(x, str) else x)

    df_local = df_local[property_cols]

    mask = df_local.apply(lambda r: row_has_any_value(r, property_cols), axis=1)
    return df_local[mask]


def run(
    root: Path,
    out_dir: Path,
    api_key: str,
    limit: int,
    flash_model: str,
    pro_model: str,
    flash_rpm: float,
    pro_rpm: float,
    use_flash_for_pdf: bool,
    resume: bool,
    update_existing: bool,
    fallback_pdf_on_empty: bool,
    fallback_pdf_on_weak: bool,
    min_props: int,
    dump_raw: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    in_progress_csv = out_dir / "ferroelectric_extraction_in_progress.csv"
    final_csv = out_dir / "ferroelectric_extraction_final.csv"
    fail_log = out_dir / "failed_attempts.log"

    dumps_dir = out_dir / "dumps"
    if dump_raw:
        dumps_dir.mkdir(exist_ok=True, parents=True)

    colorama_init()
    client = init_client(api_key)

    property_cols = list(PROPERTIES_AND_EXPLANATIONS.keys())
    cols = property_cols + ["ID"]

    # Load existing results
    df = pd.DataFrame(columns=cols)
    processed_ids: set[str] = set()
    seed_frames: list[pd.DataFrame] = []

    for existing in (in_progress_csv, final_csv):
        if existing.exists():
            try:
                tmp = pd.read_csv(existing)
                for c in cols:
                    if c not in tmp.columns:
                        tmp[c] = None
                tmp = tmp[cols]
                seed_frames.append(tmp)
                processed_ids.update(tmp["ID"].dropna().astype(str).tolist())
            except Exception as e:
                logging.warning("Failed to read %s: %s", existing, e)

    if seed_frames:
        df = pd.concat(seed_frames, ignore_index=True)
        df.drop_duplicates(inplace=True)

    # Build worklist
    all_folders = list(iter_hash_folders(root))
    if update_existing:
        folders = all_folders
    else:
        folders = [h for h in all_folders if to_id(h) not in processed_ids]
    if limit and limit > 0:
        folders = folders[:limit]

    # Rate limits
    next_ok_flash = time.time()
    next_ok_pro = time.time()
    flash_delay = 60.0 / max(1.0, flash_rpm)
    pro_delay = 60.0 / max(1.0, pro_rpm)

    pbar = tqdm(total=len(folders), desc="Processing", unit="paper")
    start_time = time.time()

    success_rows_total = 0
    fail_count = 0
    error_count = 0
    pdf_calls = 0
    flash_requests = 0
    pro_requests = 0
    fallback_tries = 0
    weak_fallback_tries = 0
    success_papers = 0

    for hash_dir in folders:
        year = hash_dir.parent.name
        paper_id = f"{year}/{hash_dir.name}"
        mode, path = choose_input(hash_dir)
        if not mode:
            pbar.update(1)
            continue

        pdf_candidate = None
        pdfs = list(hash_dir.glob("*.pdf"))
        if pdfs:
            pdf_candidate = max(pdfs, key=lambda p: p.stat().st_size)

        # First extraction attempt
        try:
            if mode == "json":
                time.sleep(max(0.0, next_ok_flash - time.time()))
                text = load_text_from_json(path)
                raw = extract_with_text(client, flash_model, text)
                flash_requests += 1
                next_ok_flash = time.time() + flash_delay
                if dump_raw:
                    (dumps_dir / f"{paper_id.replace('/','_')}_{flash_model}_json.json").write_text(raw)
            else:
                time.sleep(max(0.0, (next_ok_flash if use_flash_for_pdf else next_ok_pro) - time.time()))
                pdf_bytes = path.read_bytes()
                model = flash_model if use_flash_for_pdf else pro_model
                raw = extract_with_pdf(client, model, pdf_bytes)
                pdf_calls += 1
                if model == pro_model:
                    pro_requests += 1
                else:
                    flash_requests += 1
                if dump_raw:
                    (dumps_dir / f"{paper_id.replace('/','_')}_{model}_pdf.json").write_text(raw)
                if use_flash_for_pdf:
                    next_ok_flash = time.time() + flash_delay
                else:
                    next_ok_pro = time.time() + pro_delay

            out_df = clean_to_rows(raw, property_cols)

            # Fallback on empty JSON
            if out_df.empty and mode == "json" and fallback_pdf_on_empty and pdf_candidate:
                fallback_tries += 1
                time.sleep(max(0.0, next_ok_pro - time.time()))
                pdf_bytes = pdf_candidate.read_bytes()
                raw2 = extract_with_pdf(client, pro_model, pdf_bytes)
                pdf_calls += 1
                pro_requests += 1
                next_ok_pro = time.time() + pro_delay
                out_df = clean_to_rows(raw2, property_cols)

            # Weak fallback (no non-composition props)
            if (mode == "json" and fallback_pdf_on_weak and pdf_candidate and
                not out_df.empty and count_rows_with_noncomp_props(out_df, property_cols) < min_props):
                weak_fallback_tries += 1
                tqdm.write(Fore.YELLOW + f"↻ {paper_id} — weak JSON result; retrying PDF…" + Style.RESET_ALL)
                time.sleep(max(0.0, next_ok_pro - time.time()))
                pdf_bytes = pdf_candidate.read_bytes()
                raw3 = extract_with_pdf(client, pro_model, pdf_bytes)
                pdf_calls += 1
                pro_requests += 1
                next_ok_pro = time.time() + pro_delay
                out_df2 = clean_to_rows(raw3, property_cols)
                if count_rows_with_noncomp_props(out_df2, property_cols) >= count_rows_with_noncomp_props(out_df, property_cols):
                    out_df = out_df2

            if out_df.empty:
                with fail_log.open("a", encoding="utf-8-sig") as elog:
                    src = mode.upper()
                    elog.write(f"Paper: {paper_id}\nInfo: Empty after {src}.\n\n")
                fail_count += 1
                tqdm.write(Fore.RED + f"✗ {paper_id} — no rows" + Style.RESET_ALL)
                pbar.update(1)
                continue

            # Write successful rows
            rows_added = len(out_df)
            success_rows_total += rows_added
            success_papers += 1
            out_df["ID"] = paper_id
            df = pd.concat([df, out_df[cols]], ignore_index=True)
            df.drop_duplicates(inplace=True)
            df.to_csv(in_progress_csv, index=False, encoding="utf-8-sig")

            filled = count_rows_with_noncomp_props(out_df, property_cols)
            tqdm.write(Fore.GREEN + f"✓ {paper_id} — {rows_added} row(s), {filled} with props" + Style.RESET_ALL)

        except Exception as e:
            logging.exception("Error processing %s", paper_id)
            with fail_log.open("a", encoding="utf-8-sig") as elog:
                elog.write(f"Paper: {paper_id}\nError: {e}\n\n")
            fail_count += 1
            error_count += 1
            tqdm.write(Fore.RED + f"✗ {paper_id} — error" + Style.RESET_ALL)
        finally:
            pbar.update(1)

    pbar.close()

    # Finalize
    df.drop_duplicates(inplace=True)
    df.to_csv(final_csv, index=False, encoding="utf-8-sig")
    elapsed = time.time() - start_time
    total_targets = len(folders)

    tqdm.write("\n" + "="*60)
    tqdm.write("Summary")
    tqdm.write("="*60)
    tqdm.write(f"Targets: {total_targets}")
    tqdm.write(Fore.GREEN + f"  Papers with results: {success_papers}" + Style.RESET_ALL)
    tqdm.write(Fore.GREEN + f"  Rows extracted:     {success_rows_total}" + Style.RESET_ALL)
    tqdm.write(Fore.RED   + f"  No-result papers:   {fail_count}" + Style.RESET_ALL)
    tqdm.write(Fore.YELLOW+ f"  Weak fallbacks:     {weak_fallback_tries}" + Style.RESET_ALL)
    tqdm.write(Fore.YELLOW+ f"  Empty fallbacks:    {fallback_tries}" + Style.RESET_ALL)
    tqdm.write(Fore.RED   + f"  Errors:             {error_count}" + Style.RESET_ALL)
    tqdm.write(f"Model calls — Flash: {flash_requests}, Pro: {pro_requests}, PDF: {pdf_calls}")
    tqdm.write(f"Elapsed: {elapsed:.1f} s  (~{elapsed/60:.2f} min)")
    tqdm.write("Output:")
    tqdm.write(f"  Final CSV: {final_csv}")
    tqdm.write(f"  In-progress CSV: {in_progress_csv}")
    tqdm.write(f"  Fail log: {fail_log}")
    logging.info("Final saved: %s (%d rows)", final_csv, len(df))

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract ferroelectric properties from papers tree")
    p.add_argument("--root", type=Path, required=True,
                   help="Root folder containing papers/<year>/<hash>/")
    p.add_argument("--out-dir", type=Path, default=Path("./ferroelectric_extraction_results2"),
                   help="Directory to write CSVs and logs")
    p.add_argument("--limit", type=int, default=0,
                   help="Max number of folders to process (0 = all)")
    p.add_argument("--extraction-mode", choices=["full","functional","structural"], default="full",
                   help="Which prompt to use: full, functional, or structural")
    
    p.add_argument("--flash-model", default="gemini-2.5-flash", help="Model for JSON/text path")
    p.add_argument("--pro-model", default="gemini-2.5-pro", help="Model for PDF path")
    p.add_argument("--use-flash-for-pdf", action="store_true",
                   help="Use Flash for PDFs too (cheaper, may reduce accuracy)")

    p.add_argument("--flash-rpm", type=float, default=12.0, help="Max requests per minute for Flash path")
    p.add_argument("--pro-rpm", type=float, default=3.0, help="Max requests per minute for Pro path")

    p.add_argument("--resume", action="store_true",
                   help="(Deprecated) Prior CSVs are always loaded; kept for compatibility.")
    p.add_argument("--update-existing", action="store_true",
                   help="Reprocess IDs even if already present in previous CSVs.")

    p.add_argument("--fallback-pdf-on-empty", action="store_true",
                   help="If JSON path yields no rows, retry with PDF (Pro by default).")
    p.add_argument("--fallback-pdf-on-weak", action="store_true",
                   help="If JSON yields rows but no properties beyond composition, retry with PDF.")
    p.add_argument("--min-props", type=int, default=1,
                   help="Minimum number of rows with non-composition property required to accept JSON result.")

    p.add_argument("--dump-raw", action="store_true",
                   help="Write raw model JSON per paper for debugging.")

    p.add_argument("--api-key", default=None, help="API key (overrides env and .env)")
    p.add_argument("--env", type=Path, default=None, help="Path to a .env file")

    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                   help="Logging level (default: INFO)")
    return p.parse_args()


def load_api_key(arg_key: Optional[str], env_path: Optional[Path]) -> str:
    if env_path is not None:
        load_dotenv(dotenv_path=env_path)
    else:
        load_dotenv()
    key = arg_key or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GOOGLE_API_KEY not set. Use --api-key or set it in environment / .env")
    return key


def main() -> None:
    args = parse_args()
    # choose the prompt based on CLI flag
    global SYSTEM_INSTRUCTIONS
    if args.extraction_mode == "functional":
        SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_FUNCTIONAL
    elif args.extraction_mode == "structural":
        SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_STRUCTURAL
    else:
        SYSTEM_INSTRUCTIONS = SYSTEM_INSTRUCTIONS_FULL

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    api_key = load_api_key(args.api_key, args.env)

    run(
        root=args.root.resolve(),
        out_dir=args.out_dir.resolve(),
        api_key=api_key,
        limit=args.limit,
        flash_model=args.flash_model,
        pro_model=args.pro_model,
        flash_rpm=args.flash_rpm,
        pro_rpm=args.pro_rpm,
        use_flash_for_pdf=args.use_flash_for_pdf,
        resume=args.resume,
        update_existing=args.update_existing,
        fallback_pdf_on_empty=args.fallback_pdf_on_empty,
        fallback_pdf_on_weak=args.fallback_pdf_on_weak,
        min_props=args.min_props,
        dump_raw=args.dump_raw,
    )

if __name__ == "__main__":
    main()
