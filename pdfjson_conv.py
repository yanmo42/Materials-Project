#!/usr/bin/env python3
"""
PDF ➜ JSON converter for the Ferroelectric Materials Project
===========================================================
Version   : v0.5.0 (2025-07-21)
Python    : 3.9-3.12
Requires  :
    pip install google-genai pymupdf tqdm python-dotenv jsonschema

Highlights
----------
* **Cheap-first, smart-fallback**: uses Gemini Flash for ordinary PDFs,
  upgrades to Gemini Pro when a scan is detected.
* **Schema-constrained JSON**: leverages controlled generation with a JSON schema.
* **Windows-friendly concurrency**: ThreadPoolExecutor (network I/O bound).
* **Automatic cleanup**: Deletes uploaded files after processing.
* **.env** support via python-dotenv.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import time
import fitz  # PyMuPDF
from dotenv import load_dotenv
from jsonschema import validate as json_validate, ValidationError
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging 
import google.genai as genai
from google.genai import types
import threading
from collections import Counter
# ──────────────────────────────────────────────────────────────────────────────
# Initialize the Gen AI client
# ──────────────────────────────────────────────────────────────────────────────
def init_client() -> genai.Client:
    load_dotenv()  # load GOOGLE_API_KEY from .env if present
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        sys.exit("Error: GOOGLE_API_KEY not set (export or .env)")
    return genai.Client(api_key=api_key)

client = init_client()

# ──────────────────────────────────────────────────────────────────────────────
# Your Draft‑7 JSON Schema (for offline validation)
# ──────────────────────────────────────────────────────────────────────────────
JSON_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "metadata": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "doi": {"type": ["string", "null"]},
                "year": {"type": ["integer", "null"]},
            },
            "required": ["title"],
        },
        "sections": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "heading": {"type": "string"},
                    "text": {"type": "string"},
                },
                "required": ["heading", "text"],
            },
        },
    },
    "required": ["metadata", "sections"],
}

# ──────────────────────────────────────────────────────────────────────────────
# Vertex‑style schema (uppercase types + nullable) for the SDK
# ──────────────────────────────────────────────────────────────────────────────
VERTEX_SCHEMA: Dict[str, Any] = {
    "type": "OBJECT",
    "properties": {
        "metadata": {
            "type": "OBJECT",
            "properties": {
                "title": {"type": "STRING"},
                "doi": {"type": "STRING", "nullable": True},
                "year": {"type": "INTEGER", "nullable": True},
            },
            "required": ["title"],
        },
        "sections": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "heading": {"type": "STRING"},
                    "text": {"type": "STRING"},
                },
                "required": ["heading", "text"],
            },
        },
    },
    "required": ["metadata", "sections"],
}

MODEL_TIERS = SimpleNamespace(
    CHEAP="gemini-2.5-flash",   # switch to 2.5 Flash
    EXPENSIVE="gemini-2.5-pro", # switch to 2.5 Pro
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def safe_open_pdf(path: Path) -> fitz.Document | None:
    try:
        return fitz.open(path)
    except Exception as e:
        # If it's truly empty or corrupted, return None
        return None

def is_probably_scanned(pdf_path: Path, *, max_pages: int = 2, min_chars: int = 500) -> bool:
    """Low extractable text → likely scan → use expensive model."""
    try:
        doc = fitz.open(pdf_path)
        pages = min(max_pages, len(doc))
        chars = sum(len(doc.load_page(i).get_text("text")) for i in range(pages))
        return chars < min_chars * pages
    except Exception:
        return True

def call_gemini(pdf_path: Path, model_name: str, verbose: bool = False) -> Dict[str, Any]:
    """
    Upload a PDF, ask Gemini for schema-constrained JSON, and clean up.
    """
    if verbose:
        tqdm.write(f"  Uploading {pdf_path.name}...")
    # SDK wants 'file', not 'path'
    uploaded = client.files.upload(file=str(pdf_path))

    if verbose:
        start = time.time()
        tqdm.write(f"  [{time.strftime('%H:%M:%S')}] Invoking {model_name!r} on {pdf_path.name!r}…")

    try:
        prompt = (
            "You are a document converter. Extract the title, DOI, year, and "
            "reading-order text grouped into sections with headings. Ignore "
            "references and acknowledgments. Return ONLY JSON that validates "
            "against the provided schema."
        )
        gen_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=VERTEX_SCHEMA
        )

        # prompt first, then the uploaded file
        response = client.models.generate_content(
            model=model_name,
            contents=[prompt, uploaded],
            config=gen_config
        )

        if verbose:
            elapsed = time.time() - start
            tqdm.write(f"  [{time.strftime('%H:%M:%S')}] Model done in {elapsed:.1f}s")

        # 1) parse JSON
        data = json.loads(response.text)
        # 2) offline validate with your Draft‑7 schema
        json_validate(data, JSON_SCHEMA)
        return data

    finally:
        if verbose:
            tqdm.write(f"  Deleting uploaded file: {uploaded.name}")
        client.files.delete(name=uploaded.name)

# ──────────────────────────────────────────────────────────────────────────────
# Worker & CLI
# ──────────────────────────────────────────────────────────────────────────────
def process_pdf(pdf_path: Path, cfg: dict) -> None:
    """Process a single PDF: convert via Gemini, handle output, and record stats."""
    # Paths for output JSON and failure marker
    out_path = cfg["out_dir"] / pdf_path.with_suffix(".json").name
    fail_marker = cfg["fail_dir"] / pdf_path.name

    # 1) Skip if already succeeded
    if out_path.exists() and out_path.stat().st_size > 0:
        return
    # 2) Optionally skip if already failed
    if cfg.get("skip_failed", False) and fail_marker.exists():
        return

    # 3) Choose model tier based on scan detection
    tier = cfg["exp"] if is_probably_scanned(pdf_path) else cfg["cheap"]

    try:
        # 4) Call Gemini to get structured JSON
        data = call_gemini(pdf_path, tier, verbose=cfg["verb"])

        # 5) Serialize and write JSON atomically
        payload = json.dumps(data, ensure_ascii=False, indent=2)
        tmp_path = out_path.with_suffix(".json.tmp")
        tmp_path.write_text(payload, encoding="utf-8")
        tmp_path.replace(out_path)

        # 6) Update stats under lock
        with cfg["lock"]:
            cfg["stats"]["success"] += 1
            cfg["stats"]["model_usage"][tier] += 1

        tqdm.write(f"✔ {pdf_path.name} → {tier}")

    except Exception as e:
        # Remove any empty JSON file
        if out_path.exists() and out_path.stat().st_size == 0:
            out_path.unlink()
        # Touch failure marker
        fail_marker.touch()
        # Update failure count
        with cfg["lock"]:
            cfg["stats"]["fail"] += 1
        tqdm.write(f"❌ {pdf_path.name} failed: {e}")

def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Convert PDFs to JSON via Gemini")
    ap.add_argument("--input-dir",   type=Path, required=True, help="Folder with PDFs")
    ap.add_argument("--output-dir",  type=Path, default=Path("json_pdfs"))
    ap.add_argument("--threads",     type=int, default=os.cpu_count() or 8)
    ap.add_argument("--limit",       type=int, help="Process only first N PDFs (testing)")
    ap.add_argument("--cheap-model", default=MODEL_TIERS.CHEAP)
    ap.add_argument("--expensive-model", default=MODEL_TIERS.EXPENSIVE)
    ap.add_argument("--verbose",     action="store_true", help="Enable detailed logging")
    return ap.parse_args(argv)



def main(argv: List[str] | None = None) -> None:

    # ----- 1) Imports for PDF inspection -----
    from PyPDF2 import PdfReader

    # ----- 2) Silence MuPDF’s CSS warning -----
    logging.getLogger("fitz").setLevel(logging.ERROR)

    # ----- 3) Parse args & prep folders -----
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fail_dir = args.output_dir / "_failed"
    fail_dir.mkdir(exist_ok=True)

    quarantine = args.input_dir / "_quarantine"
    quarantine.mkdir(exist_ok=True)

    # ----- 4) Thread‑safe stats -----
    stats_lock = threading.Lock()
    stats = {"success": 0, "fail": 0, "model_usage": Counter()}
    cfg = {
        "out_dir":     args.output_dir,
        "fail_dir":    fail_dir,
        "cheap":       args.cheap_model,
        "exp":         args.expensive_model,
        "verb":        args.verbose,
        "stats":       stats,
        "lock":        stats_lock,
        "skip_failed": False,
    }

    # ----- 5) Gather & limit -----
    all_pdfs = sorted(args.input_dir.rglob("*.pdf"),
                      key=lambda p: p.stat().st_size)
    if args.limit:
        all_pdfs = all_pdfs[: args.limit]
    print(f"Found {len(all_pdfs)} PDFs (after --limit)")

    # ----- 6) Pre‑filter via PyPDF2 -----
    valid_pdfs = []
    removed = 0
    for p in all_pdfs:
        try:
            reader = PdfReader(str(p))
            page_count = len(reader.pages)
        except Exception as e:
            tqdm.write(f"⚠ Deleting unreadable PDF {p.name}: {e}")
            p.replace(quarantine / p.name)
            removed += 1
            continue

        if page_count == 0:
            tqdm.write(f"⚠ Removing blank PDF {p.name}")
            p.replace(quarantine / p.name)
            removed += 1
        else:
            valid_pdfs.append(p)

    print(f"{len(valid_pdfs)} valid PDFs, {removed} removed")

    if not valid_pdfs:
        sys.exit("No valid PDFs left after filtering blank/unreadable files.")

    # ----- 7) Sort remaining -----
    valid_pdfs.sort(key=lambda p: p.stat().st_size)

    # ----- 8) Convert with ThreadPool + tqdm -----
    with ThreadPoolExecutor(max_workers=args.threads) as executor, \
         tqdm(total=len(valid_pdfs), desc="Converting PDFs", unit="file") as pbar:

        fut_to_name = {
            executor.submit(process_pdf, pdf, cfg): pdf.name
            for pdf in valid_pdfs
        }

        for fut in as_completed(fut_to_name):
            name = fut_to_name[fut]
            try:
                fut.result()
                pbar.set_postfix(file=name, status="✔")
            except Exception:
                pbar.set_postfix(file=name, status="❌")
            pbar.update(1)

    # ----- 9) Final summary -----
    total = len(valid_pdfs)
    succ  = stats["success"]
    fail  = stats["fail"]

    print("\n=== Conversion Summary ===")
    print(f" Total PDFs attempted: {total}")
    print(f"   ✔ Successful: {succ}")
    print(f"   ❌ Failed:     {fail}")
    print("\n Model usage:")
    for model, cnt in stats["model_usage"].items():
        print(f"   • {model}: {cnt}")

if __name__ == "__main__":
    main()
