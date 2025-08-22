#!/usr/bin/env python3
"""
md_extractor.py — MD-first extractor (Gemini 2.5 Flash) with PDF fallback (Gemini 2.5 Pro)

- Outputs ONE streaming CSV with EXACT columns:
  chemical_composition,phase,space_group,lattice_parameters,spontaneous_polarization,
  remanent_polarization,coercive_field,curie_temperature,dielectric_constant,
  piezoelectric_constant,ID

- Discovers MinerU Markdown at: <mineru_root>/<year>/<hash>/<hash>/auto/*.md
- Pairs with PDFs at:           <pdf_root>/<year>/<hash>.pdf
- Prints a discovery summary BEFORE API calls (or only, with --dry-run)
- Pass 1: MD → Gemini 2.5 Flash
- Pass 2: PDF-only → Gemini 2.5 Pro
- Idempotent markers: out_dir/records/<year>/<id>.done

Notes:
- Supports .env via python-dotenv if present.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Optional

# Optional .env support
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from tqdm import tqdm

# Google GenAI (Gemini) SDK
from google import genai
from google.genai import types

# Pydantic for structured output schema
from pydantic import BaseModel
from typing import Optional as Opt

# ==========================
# CSV header (exact schema)
# ==========================
HEADERS = [
    "chemical_composition",
    "phase",
    "space_group",
    "lattice_parameters",
    "spontaneous_polarization",
    "remanent_polarization",
    "coercive_field",
    "curie_temperature",
    "dielectric_constant",
    "piezoelectric_constant",
    "ID",
]

_write_lock = threading.Lock()


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def discover_manifest(mineru_root: Path, pdf_root: Path, limit: Optional[int] = None) -> List[Dict]:
    rows: List[Dict] = []
    mineru_root = mineru_root.resolve()
    pdf_root = pdf_root.resolve()

    if not mineru_root.exists():
        raise SystemExit(f"ERROR: MinerU root not found: {mineru_root}")
    if not pdf_root.exists():
        print(f"[WARN] PDF root not found: {pdf_root}. PDF fallback may be skipped where PDFs are missing.")

    years = sorted([p for p in mineru_root.iterdir() if p.is_dir()])
    for y in years:
        year = y.name
        id_dirs = sorted([p for p in y.iterdir() if p.is_dir()])
        for id_dir in id_dirs:
            id_ = id_dir.name

            md_candidates = list((id_dir / id_ / "auto").glob("*.md"))
            md_path = max(md_candidates, key=lambda p: p.stat().st_size) if md_candidates else None

            pdf_path = pdf_root / year / f"{id_}.pdf"
            has_pdf = pdf_path.exists()

            stdout_path = id_dir / "stdout.txt"
            stderr_path = id_dir / "stderr.txt"

            rows.append({
                "id": id_,
                "year": year,
                "md_path": str(md_path) if md_path else None,
                "pdf_path": str(pdf_path) if has_pdf else None,
                "stdout_path": str(stdout_path) if stdout_path.exists() else None,
                "stderr_path": str(stderr_path) if stderr_path.exists() else None,
                "has_md": md_path is not None,
                "has_pdf": has_pdf,
            })
            if limit and len(rows) >= limit:
                return rows
    return rows


def print_summary(rows: List[Dict]) -> None:
    total = len(rows)
    has_md = sum(1 for r in rows if r["has_md"])
    md_and_pdf = sum(1 for r in rows if r["has_md"] and r["has_pdf"])
    pdf_only = sum(1 for r in rows if (not r["has_md"]) and r["has_pdf"])
    missing_both = sum(1 for r in rows if (not r["has_md"]) and (not r["has_pdf"]))

    print("\n=== Discovery summary ===")
    print(f"Total papers: {total}")
    print(f"MD available: {has_md}  (MD + PDF: {md_and_pdf})")
    print(f"PDF-only (no MD): {pdf_only}")
    print(f"Missing both MD and PDF: {missing_both}")


def append_csv_rows(csv_path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(csv_path.parent)
    is_new = not csv_path.exists()
    with _write_lock:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            for row in rows:
                safe = {h: ("" if row.get(h) is None else row.get(h)) for h in HEADERS}
                writer.writerow(safe)


# ----------------------------------------------------------
# Structured output schema (like newtractor.py uses)
# ----------------------------------------------------------
class MaterialRecord(BaseModel):
    chemical_composition: Opt[str] = None
    phase: Opt[str] = None
    space_group: Opt[str] = None
    lattice_parameters: Opt[str] = None
    spontaneous_polarization: Opt[float] = None
    remanent_polarization: Opt[float] = None
    coercive_field: Opt[float] = None
    curie_temperature: Opt[float] = None
    dielectric_constant: Opt[float] = None
    piezoelectric_constant: Opt[float] = None


SYSTEM_INSTRUCTIONS = """
You are extracting training data for a CGCNN (crystal graph convolutional neural network).
Your sole target label is the Curie temperature (ferroelectric → paraelectric transition) for inorganic crystalline materials.

OUTPUT ONLY records that are appropriate for CGCNN training AND contain a trustworthy Curie temperature in °C.

INCLUDE only if ALL are true:
- The sample is an inorganic crystalline or polycrystalline material (ceramic, single crystal, thin film).
- The chemical_composition is a valid chemical formula using element symbols and stoichiometric coefficients
  (e.g., BaTiO3, Pb(Zr0.52Ti0.48)O3, Ba0.95La0.05TiO3).
- If doping is reported, dopant(s) are element symbols with explicit stoichiometry/fraction; convert to a proper formula
  (e.g., “BaTiO3 with 5% La on Ba site” → “Ba0.95La0.05TiO3”). If you cannot convert, EXCLUDE.
- A Curie temperature for the ferroelectric → paraelectric transition is explicitly reported (not inferred),
  and can be returned as a single numeric value after unit conversion to °C.

EXCLUDE the following (do not output any record that matches any item below):
- Liquid crystals or soft matter (SmA, SmC, SmC*, nematic, cholesteric, LC), polymers, organic small molecules.
- Vague composites or host/dopant descriptions without element symbols (e.g., “PZT + epoxy”, “host A with 1a”).
- Coordination/complex salts, hydrates, ammonium salts, and related hydrogen- or anion-complex systems:
  examples include formulas with “·H2O”, “NH4”, “(CN)”, “NO3”, “NO2”, “SO4”, “HSO4”.
- Incipient/non-ferroelectrics reported with Curie–Weiss temperatures (T0) or extrapolated values, e.g., KTaO3, SrTiO3, CaTiO3,
  unless the composition is explicitly modified to become ferroelectric AND an FE→PE Curie temperature is stated.
- Temperatures for non-ferroelectric transitions (magnetic Curie temperature, Néel temperature, superconducting Tc,
  or structural transitions not associated with FE→PE).
- Tc given only as a plot with no numeric value, or only a range without a clear central value attributable to Tc.

CURIE TEMPERATURE RULES (to ensure ≤5 K label accuracy):
- curie_temperature MUST be a single NUMBER in °C. Convert K→°C by subtracting 273.15. Keep decimals; do not round.
- Prefer values explicitly labeled “Curie temperature” or clearly the FE→PE transition (dielectric permittivity peak, DSC/DTA).
- If multiple Tc values are reported for the same state:
  1) Prefer values explicitly labeled “Curie temperature” over generic “transition temperature”.
  2) Prefer the heating-cycle value over cooling if both are provided and differ, unless the paper states cooling is canonical.
  3) If bulk and thin film differ and composition is nominally the same, treat them as DISTINCT states and output separate records.
- If both Kelvin and Celsius values appear for the same composition/state, convert all to °C.
  If unlabeled numbers for the same composition/state differ by ~273 (±6), treat the higher as Kelvin and convert to °C.
- Negative °C values are allowed (e.g., Tc = 123 K → −150.15 °C).
- Plausibility window: discard Tc outside −260 °C to 1200 °C.

DE-DUPLICATION WITHIN A PAPER (avoid overweighting):
- Do not output duplicate rows for the same (chemical_composition AND curie_temperature) within the same source.
  If multiple entries are identical in those keys, keep the one with the most complete auxiliary fields (phase/space_group/etc.).

FIELDS and UNITS (schema must match exactly; leave non-Tc fields null if not reported):
- chemical_composition: valid chemical formula string using element symbols and optional parentheses; NO words like “host”, “doped with”, “wt%”.
- phase: concise phase/crystal label if given (e.g., tetragonal, rhombohedral); else null.
- space_group: Hermann–Mauguin symbol if given (e.g., P4mm); else null.
- lattice_parameters: single string if reported, e.g., "a=..., b=..., c=... Å; α=..., β=..., γ=...°"; else null.
- spontaneous_polarization: NUMBER (µC/cm²) or null.
- remanent_polarization: NUMBER (µC/cm²) or null.
- coercive_field: NUMBER (kV/cm) or null.
- curie_temperature: REQUIRED NUMBER (°C) after conversion. If uncertain or not explicitly stated, set null and EXCLUDE the record.
- dielectric_constant: NUMBER (dimensionless) or null.
- piezoelectric_constant: NUMBER (pC/N; usually d33) or null.

GENERAL RULES:
- Convert all units to the required ones. Strip “~”, “≈”, “±”, text, and units; return pure numbers in numeric fields.
- If a paper reports multiple distinct states (different compositions, doping levels, bulk vs thin film, annealing treatments), output multiple records—one per unique state.
- Do NOT invent values. Only extract values explicitly reported in text or tables (not from unlabeled figures).
- If a record fails ANY inclusion rule above, do not output it.
- Return ONLY JSON matching response_schema (list[MaterialRecord]) with the exact keys shown above.
""".strip()


def _records_to_csv_rows(id_: str, records: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for rec in (records or []):
        out.append({
            "chemical_composition": rec.get("chemical_composition"),
            "phase": rec.get("phase"),
            "space_group": rec.get("space_group"),
            "lattice_parameters": rec.get("lattice_parameters"),
            "spontaneous_polarization": rec.get("spontaneous_polarization"),
            "remanent_polarization": rec.get("remanent_polarization"),
            "coercive_field": rec.get("coercive_field"),
            "curie_temperature": rec.get("curie_temperature"),
            "dielectric_constant": rec.get("dielectric_constant"),
            "piezoelectric_constant": rec.get("piezoelectric_constant"),
            "ID": id_,
        })
    return out


def _marker_path(out_dir: Path, year: str, id_: str) -> Path:
    return out_dir / "records" / year / f"{id_}.done"


# ----------------------------------------------------------
# Gemini calls (mirror newtractor.py style)
# ----------------------------------------------------------
def call_gemini_md(client: genai.Client, model: str, md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8", errors="ignore")
    content = f"<paper_start>\n{text}\n<paper_end>"
    resp = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=list[MaterialRecord],
            temperature=0.0,
        ),
        contents=content,
    )
    return resp.text or "[]"


def call_gemini_pdf(client: genai.Client, model: str, pdf_path: Path) -> str:
    pdf_bytes = pdf_path.read_bytes()
    resp = client.models.generate_content(
        model=model,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            response_mime_type="application/json",
            response_schema=list[MaterialRecord],
            temperature=0.0,
        ),
        contents=[
            types.Part.from_bytes(data=pdf_bytes, mime_type="application/pdf"),
            "Extract the specified properties for any ferroelectric materials mentioned.",
        ],
    )
    return resp.text or "[]"


# =====================
# Per-item processors
# =====================
def process_one_md(row: Dict, out_dir: Path, client: genai.Client, model: str, force: bool) -> Dict:
    id_ = row["id"]
    year = row["year"]
    md_path = Path(row["md_path"]) if row.get("md_path") else None
    if md_path is None:
        return {"id": id_, "status": "skip:no_md", "rows": [], "err": "no_md"}

    marker = _marker_path(out_dir, year, id_)
    if marker.exists() and not force:
        return {"id": id_, "status": "skipped_exists", "rows": []}

    try:
        json_text = call_gemini_md(client, model, md_path)
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("Model did not return a JSON list")
        rows = _records_to_csv_rows(id_, parsed)
        ensure_dir(marker.parent)
        marker.write_text("ok\n", encoding="utf-8")
        return {"id": id_, "status": "ok", "rows": rows}
    except Exception as e:
        return {"id": id_, "status": "error", "rows": [], "err": f"{type(e).__name__}: {e}"}


def process_one_pdf(row: Dict, out_dir: Path, client: genai.Client, model: str, force: bool) -> Dict:
    id_ = row["id"]
    year = row["year"]
    pdf_path = Path(row["pdf_path"]) if row.get("pdf_path") else None
    if pdf_path is None:
        return {"id": id_, "status": "skip:no_pdf", "rows": [], "err": "no_pdf"}

    marker = _marker_path(out_dir, year, id_)
    if marker.exists() and not force:
        return {"id": id_, "status": "skipped_exists", "rows": []}

    try:
        json_text = call_gemini_pdf(client, model, pdf_path)
        parsed = json.loads(json_text)
        if isinstance(parsed, dict):
            parsed = [parsed]
        if not isinstance(parsed, list):
            raise ValueError("Model did not return a JSON list")
        rows = _records_to_csv_rows(id_, parsed)
        ensure_dir(marker.parent)
        marker.write_text("ok\n", encoding="utf-8")
        return {"id": id_, "status": "ok", "rows": rows}
    except Exception as e:
        return {"id": id_, "status": "error", "rows": [], "err": f"{type(e).__name__}: {e}"}


# =====================
# Pass runners
# =====================
def append_csv_rows(csv_path: Path, rows: Iterable[Dict]) -> None:
    ensure_dir(csv_path.parent)
    is_new = not csv_path.exists()
    with _write_lock:
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
            if is_new:
                writer.writeheader()
            for row in rows:
                safe = {h: ("" if row.get(h) is None else row.get(h)) for h in HEADERS}
                writer.writerow(safe)


def _already_done(out_dir: Path, row: Dict, force: bool) -> bool:
    if force:
        return False
    return _marker_path(out_dir, row["year"], row["id"]).exists()

def run_md_pass(rows: List[Dict], args, client: genai.Client) -> Dict:
    out_dir = Path(args.out_dir)
    md_rows = [r for r in rows if r.get("has_md")]
    if not md_rows:
        print("No MD rows to process.")
        return {"errors": []}

    # Filter out IDs that are already marked done (unless --force)
    pending = [r for r in md_rows if not _already_done(out_dir, r, args.force)]
    skipped = len(md_rows) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(f"MD-capable: {len(md_rows)} | already done: {skipped} | to process now: {len(pending)}")
    if not pending:
        print("Nothing to do (all MD items already processed).")
        return {"errors": []}

    print(f"Starting MD pass on {len(pending)} papers with model {args.model_flash}.")
    csv_path = Path(args.results_csv)
    errors = []

    with ThreadPoolExecutor(max_workers=args.md_workers) as ex:
        futures = [ex.submit(process_one_md, r, out_dir, client, args.model_flash, args.force) for r in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="MD pass"):
            res = fut.result()
            if res.get("rows"):
                append_csv_rows(csv_path, res["rows"])
            if res.get("status") == "error" and res.get("err"):
                errors.append((res["id"], res["err"]))
    return {"errors": errors}


def run_pdf_pass(rows: List[Dict], args, client: genai.Client) -> Dict:
    out_dir = Path(args.out_dir)
    pdf_rows = [r for r in rows if (not r.get("has_md")) and r.get("has_pdf")]
    if not pdf_rows:
        print("No PDF-only rows to process.")
        return {"errors": []}

    # Filter out IDs that are already marked done (unless --force)
    pending = [r for r in pdf_rows if not _already_done(out_dir, r, args.force)]
    skipped = len(pdf_rows) - len(pending)
    if args.limit:
        pending = pending[: args.limit]

    print(f"PDF-eligible: {len(pdf_rows)} | already done: {skipped} | to process now: {len(pending)}")
    if not pending:
        print("Nothing to do (all PDF-only items already processed).")
        return {"errors": []}

    print(f"Starting PDF pass on {len(pending)} papers with model {args.model_pro}.")
    csv_path = Path(args.results_csv)
    errors = []

    with ThreadPoolExecutor(max_workers=args.pdf_workers) as ex:
        futures = [ex.submit(process_one_pdf, r, out_dir, client, args.model_pro, args.force) for r in pending]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="PDF pass"):
            res = fut.result()
            if res.get("rows"):
                append_csv_rows(csv_path, res["rows"])
            if res.get("status") == "error" and res.get("err"):
                errors.append((res["id"], res["err"]))
    return {"errors": errors}

# =====================
# API key selection
# =====================
def resolve_api_key(cmd_key: Optional[str]) -> str:
    if cmd_key:
        return cmd_key
    if os.getenv("GOOGLE_API_KEY"):
        return os.environ["GOOGLE_API_KEY"]
    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    return ""


# =====================
# CLI
# =====================
def main() -> None:
    ap = argparse.ArgumentParser(description="MD-first extractor with PDF fallback (Gemini) — CSV output")
    ap.add_argument("--mineru-root", required=True, type=Path)
    ap.add_argument("--pdf-root", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--results-csv", type=Path, help="Path to streaming CSV (default: out_dir/results.csv)")

    ap.add_argument("--limit", type=int, default=None, help="Limit number of papers (testing)")
    ap.add_argument("--only-md", action="store_true", help="Only run the MD pass")
    ap.add_argument("--only-pdf", action="store_true", help="Only run the PDF pass (MD must be absent)")
    ap.add_argument("--force", action="store_true", help="Re-run even if marker exists")
    ap.add_argument("--dry-run", action="store_true", help="Just print the summary and exit")
    ap.add_argument("--probe", action="store_true", help="Quick auth/model probe")

    ap.add_argument("--md-workers", type=int, default=8)
    ap.add_argument("--pdf-workers", type=int, default=2)

    ap.add_argument("--model-flash", default="gemini-2.5-flash")
    ap.add_argument("--model-pro", default="gemini-2.5-pro")

    ap.add_argument("--api-key", default=None, help="Explicit API key; overrides env")
    ap.add_argument("--verbose-errors", type=int, default=5, help="Print the first N errors encountered per pass")

    args = ap.parse_args()

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit("ERROR: No API key found. Provide --api-key or set GOOGLE_API_KEY / GEMINI_API_KEY.")

    if not args.results_csv:
        args.results_csv = args.out_dir / "results.csv"
    ensure_dir(args.out_dir)

    # Discovery
    rows = discover_manifest(args.mineru_root, args.pdf_root, args.limit)
    print_summary(rows)

    if args.dry_run:
        return

    # Client
    client = genai.Client(api_key=api_key)

    # Optional probe (simple text round-trip — does not touch files)
    if args.probe:
        try:
            txt = client.models.generate_content(model=args.model_flash, contents="ping").text or ""
            print(f"Probe OK: {txt[:60]}{'...' if len(txt) > 60 else ''}")
        except Exception as e:
            raise SystemExit(f"Probe failed: {type(e).__name__}: {e}")

    # Mode selection
    all_errors: List[tuple[str, str]] = []

    if args.only_pdf and args.only_md:
        raise SystemExit("ERROR: Cannot set both --only-md and --only-pdf.")

    if args.only_pdf:
        result = run_pdf_pass(rows, args, client)
        all_errors.extend(result["errors"])
    elif args.only_md:
        result = run_md_pass(rows, args, client)
        all_errors.extend(result["errors"])
    else:
        md_res = run_md_pass(rows, args, client)
        all_errors.extend(md_res["errors"])
        pdf_res = run_pdf_pass(rows, args, client)
        all_errors.extend(pdf_res["errors"])

    # Error reporting
    if all_errors:
        print("\nErrors encountered (showing up to --verbose-errors):")
        for (i, (pid, err)) in enumerate(all_errors[: args.verbose_errors], start=1):
            print(f"{i:02d}. {pid}: {err}")
        by_type = Counter(e.split(":")[0] for _, e in all_errors if ":" in e)
        if by_type:
            print(f"Error summary by type: {dict(by_type)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    main()
