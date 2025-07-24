#!/usr/bin/env python3
"""
find_bad_pdfs.py

Scan a folder of PDFs (optionally limited), in parallel with a status bar,
detect corrupted/zero-page/missing-eof/small files, and either quarantine them
for manual review or delete them.

Usage:
  pip install PyPDF2 tqdm
  python find_bad_pdfs.py \
      --input-dir ferro_papers \
      --quarantine-dir ../ferro_papers_quarantine \
      --min-pages 1 \
      --max-files 500 \
      --threads 8 \
      --eof-check \
      --min-size 20480 \
      [--delete]
"""

import argparse, shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

# ——— PyPDF2 compatibility imports ———
try:
    from PyPDF2 import PdfReader
except ImportError:
    from PyPDF2 import PdfFileReader as PdfReader

try:
    from PyPDF2.errors import PdfReadError
except ImportError:
    from PyPDF2.utils import PdfReadError
# ————————————————————————

import logging
logging.getLogger("PyPDF2").setLevel(logging.ERROR)


def has_eof(path: Path) -> bool:
    """Return True if the last 1 KB of the file contains '%EOF'."""
    try:
        with open(path, 'rb') as f:
            f.seek(-1024, 2)
            tail = f.read()
    except OSError:
        # file smaller than 1 KB
        with open(path, 'rb') as f:
            tail = f.read()
    return b'%EOF' in tail

def is_bad_pdf(path: Path, min_pages: int,
               eof_check: bool, min_size: int|None) -> tuple[bool,str]:
    """
    Returns (True, reason) if the PDF is:
      - unreadable (parse error),
      - has fewer than min_pages pages,
      - is missing '%EOF' (when eof_check=True),
      - or is smaller than min_size bytes (when min_size set).
    Otherwise returns (False, info).
    """
    # 1) size check
    if min_size is not None:
        size = path.stat().st_size
        if size < min_size:
            return True, f"{size} bytes (< {min_size})"

    # 2) EOF marker check
    if eof_check and not has_eof(path):
        return True, "missing '%EOF'"

    # 3) try parsing/counting pages
    try:
        reader = PdfReader(str(path), strict=True)
        pages = getattr(reader, "pages", None) or getattr(reader, "getNumPages", lambda: [])()
        n = len(pages) if hasattr(pages, "__len__") else pages
        if n < min_pages:
            return True, f"{n} pages (< {min_pages})"
        return False, f"{n} pages"
    except PdfReadError as e:
        return True, f"PdfReadError: {e}"
    except Exception as e:
        return True, f"Error: {e}"

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir",  default="ferro_papers",
                   help="Root folder containing your downloaded PDFs")
    p.add_argument("--quarantine-dir", default="ferro_papers_quarantine",
                   help="Where to move bad PDFs (if not deleting)")
    p.add_argument("--min-pages",   type=int, default=1,
                   help="Minimum number of pages to count as valid")
    p.add_argument("--max-files",   type=int, default=None,
                   help="Max number of PDFs to scan (None = no limit)")
    p.add_argument("--threads",     type=int, default=4,
                   help="Number of worker threads to use")
    p.add_argument("--eof-check",   action="store_true",
                   help="Flag any PDF missing a trailing '%EOF'")
    p.add_argument("--min-size",    type=int, default=None,
                   help="Flag any PDF smaller than this many bytes")
    p.add_argument("--delete",      action="store_true",
                   help="Delete bad PDFs instead of quarantining")
    args = p.parse_args()

    input_dir  = Path(args.input_dir).resolve()
    quarantine = Path(args.quarantine_dir).resolve()

    # Prevent infinite nesting: bump quarantine outside input_dir if needed
    if not args.delete and quarantine.is_relative_to(input_dir):
        quarantine = input_dir.parent / f"{input_dir.name}_quarantine"
    if not args.delete:
        quarantine.mkdir(parents=True, exist_ok=True)

    # Gather all PDFs, skipping anything already in quarantine
    all_pdfs = [
        p for p in input_dir.rglob("*.pdf")
        if not (not args.delete and p.resolve().is_relative_to(quarantine))
    ]
    to_scan = all_pdfs if args.max_files is None else all_pdfs[:args.max_files]

    bad_count = 0
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        jobs = pool.map(
            lambda pdf: (pdf, *is_bad_pdf(
                pdf,
                args.min_pages,
                args.eof_check,
                args.min_size
            )),
            to_scan
        )

        for pdf, bad, info in tqdm(
            jobs,
            total=len(to_scan),
            desc="Scanning PDFs",
            unit="file"
        ):
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
