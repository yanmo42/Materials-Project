#!/usr/bin/env python3
"""
Robust PDF checker with PyMuPDF integration and improved checks.
"""

import argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import fitz  # pymupdf

# PyPDF2 compatibility
try:
    from PyPDF2 import PdfReader
except ImportError:
    from PyPDF2 import PdfFileReader as PdfReader

try:
    from PyPDF2.errors import PdfReadError
except ImportError:
    from PyPDF2.utils import PdfReadError

import logging
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


def has_eof(path: Path, tail_size=2048) -> bool:
    """Checks a larger tail of the PDF for '%EOF' marker."""
    try:
        with open(path, 'rb') as f:
            f.seek(-tail_size, 2)
            tail = f.read()
    except OSError:
        with open(path, 'rb') as f:
            tail = f.read()
    return b'%EOF' in tail


def is_valid_pdf_fitz(path: Path):
    """Checks PDF validity using PyMuPDF."""
    try:
        with fitz.open(path) as doc:
            if doc.page_count == 0:
                return False, "0 pages (fitz)"
        return True, "fitz valid"
    except Exception as e:
        return False, f"fitz error: {e}"


def is_bad_pdf(path: Path, min_pages: int, eof_check: bool, min_size: int|None) -> tuple[bool,str]:
    """Extended PDF validation."""
    size = path.stat().st_size

    if size == 0:
        return True, "zero-byte file"

    if min_size is not None and size < min_size:
        return True, f"{size} bytes (< {min_size})"

    if eof_check and not has_eof(path):
        return True, "missing '%EOF'"

    valid_fitz, fitz_info = is_valid_pdf_fitz(path)
    if not valid_fitz:
        return True, fitz_info

    try:
        reader = PdfReader(str(path), strict=True)
        num_pages = len(reader.pages)
        if num_pages < min_pages:
            return True, f"{num_pages} pages (< {min_pages})"
        return False, f"{num_pages} pages"
    except PdfReadError as e:
        return True, f"PyPDF2 error: {e}"
    except Exception as e:
        return True, f"Unexpected error: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="ferro_papers")
    parser.add_argument("--quarantine-dir", default="ferro_papers_quarantine")
    parser.add_argument("--min-pages", type=int, default=1)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--eof-check", action="store_true")
    parser.add_argument("--min-size", type=int, default=10240)
    parser.add_argument("--delete", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir).resolve()
    quarantine = Path(args.quarantine_dir).resolve()

    if not args.delete and quarantine.is_relative_to(input_dir):
        quarantine = input_dir.parent / f"{input_dir.name}_quarantine"
    if not args.delete:
        quarantine.mkdir(parents=True, exist_ok=True)

    all_pdfs = [
        p for p in input_dir.rglob("*.pdf")
        if not (not args.delete and p.resolve().is_relative_to(quarantine))
    ]
    to_scan = all_pdfs if args.max_files is None else all_pdfs[:args.max_files]

    bad_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        jobs = pool.map(
            lambda pdf: (pdf, *is_bad_pdf(pdf, args.min_pages, args.eof_check, args.min_size)),
            to_scan
        )

        for pdf, bad, info in tqdm(jobs, total=len(to_scan), desc="Scanning PDFs", unit="file"):
            if not bad:
                continue

            bad_count += 1
            if args.delete:
                pdf.unlink()
                print(f"❌ Deleted {pdf} ({info})")
            else:
                dest = quarantine / pdf.relative_to(input_dir)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(pdf), str(dest))
                print(f"❌ Quarantined {pdf} → {dest} ({info})")

    print(f"\n🔍 Scan complete. Scanned {len(to_scan)} files, "
          f"{bad_count} bad PDF(s) {'deleted' if args.delete else 'quarantined'}.")


if __name__ == "__main__":
    main()
