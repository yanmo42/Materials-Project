#!/usr/bin/env python3
"""
Bulk PDF → JSON text extraction into per-paper folders (copies PDF + writes JSON),
with progress bar, multiprocessing, optional limit, idempotent skips, and summary.

Each saved JSON has:
{
  "pdf_basename": "<id>",
  "rel_path": "year/ID.pdf",
  "num_pages": <int>,
  "char_count": <int>,
  "page_texts": ["...", "...", ...]   # one string per page
}
"""
import os
import argparse
import shutil
import pdfplumber
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import time
import json

# Suppress pdfminer/plumber color warnings
logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

# Minimum characters to consider a PDF "text-rich"
TEXT_THRESHOLD = 1000


def process_single(task):
    """
    task: (pdf_path, rel, input_dir, output_dir, pretty)
    Returns:
        True  -> saved JSON
        False -> processed but below threshold or error
        None  -> skipped (JSON already existed)
    """
    pdf_path, rel, input_dir, output_dir, pretty = task
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    rel_dir = os.path.dirname(rel)
    paper_folder = os.path.join(output_dir, rel_dir, base)
    out_pdf = os.path.join(paper_folder, base + '.pdf')
    out_json = os.path.join(paper_folder, base + '.json')

    # Skip if JSON already exists
    if os.path.exists(out_json):
        return None

    try:
        os.makedirs(paper_folder, exist_ok=True)

        # Copy PDF if not already there
        if not os.path.exists(out_pdf):
            shutil.copy2(pdf_path, paper_folder)

        # Extract text per page
        with pdfplumber.open(pdf_path) as pdf:
            page_texts = [(page.extract_text() or "") for page in pdf.pages]

        char_count = sum(len(p) for p in page_texts)

        # Save if above threshold
        if char_count >= TEXT_THRESHOLD:
            payload = {
                "pdf_basename": base,
                "rel_path": rel.replace("\\", "/"),
                "num_pages": len(page_texts),
                "char_count": char_count,
                "page_texts": page_texts,
            }
            with open(out_json, "w", encoding="utf-8") as f:
                if pretty:
                    json.dump(payload, f, ensure_ascii=False, indent=2)
                    f.write("\n")  # newline at EOF for readability
                else:
                    json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
            return True
        else:
            return False

    except Exception as e:
        logging.error(f"[ERROR] {rel}: {e}")
        # Cleanup on error
        shutil.rmtree(paper_folder, ignore_errors=True)
        return False


def extract_folder(input_dir, output_dir, workers, limit, pretty=False):
    # Collect tasks
    tasks = []
    total_input = 0
    for root, _, files in os.walk(input_dir):
        for fname in files:
            if not fname.lower().endswith('.pdf'):
                continue
            total_input += 1
            pdf_path = os.path.join(root, fname)
            rel = os.path.relpath(pdf_path, input_dir)
            tasks.append((pdf_path, rel, input_dir, output_dir, pretty))

    # Apply limit
    if limit and limit > 0:
        tasks = tasks[:limit]
    count_tasks = len(tasks)

    attempted = success = failure = 0

    # Use processes for true parallelism
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_single, t): t for t in tasks}
            for future in tqdm(
                as_completed(futures),
                total=count_tasks,
                desc="Extracting PDFs",
                unit="pdf",
                unit_scale=True,
            ):
                res = future.result()
                if res is True:
                    success += 1
                elif res is False:
                    failure += 1
                if res is not None:
                    attempted += 1
    else:
        for t in tqdm(
            tasks,
            total=count_tasks,
            desc="Extracting PDFs",
            unit="pdf",
            unit_scale=True,
        ):
            res = process_single(t)
            if res is True:
                success += 1
            elif res is False:
                failure += 1
            if res is not None:
                attempted += 1

    # Summarize output folder
    total_pdf_out = total_json = 0
    for _, _, files in os.walk(output_dir):
        for f in files:
            if f.lower().endswith('.pdf'):
                total_pdf_out += 1
            elif f.lower().endswith('.json'):
                total_json += 1
    total_failures = total_pdf_out - total_json

    return {
        "total_input": total_input,
        "attempted": attempted,
        "success": success,
        "failure": failure,
        "total_pdf_out": total_pdf_out,
        "total_json": total_json,
        "total_failures": total_failures,
    }


def main():
    p = argparse.ArgumentParser(
        description="Extract text from PDFs and save as JSON into per-paper folders"
    )
    p.add_argument("input_dir", help="Folder of PDFs (e.g. ferro_papers)")
    p.add_argument("output_dir", help="Destination for per-paper folders")
    p.add_argument("-w", "--workers", type=int, default=1, help="Process count")
    p.add_argument("-l", "--limit", type=int, default=None, help="Max PDFs to process")
    p.add_argument(
        "--pretty",
        action="store_true",
        help="Write indented, human-readable JSON (slightly larger/slower)",
    )
    args = p.parse_args()

    start = time.time()
    summary = extract_folder(
        args.input_dir, args.output_dir, args.workers, args.limit, args.pretty
    )
    elapsed = time.time() - start
    throughput = (summary["attempted"] / elapsed) if elapsed > 0 else 0.0

    print("\n=== Extraction Summary ===")
    print(f"Elapsed time: {elapsed:.2f} sec")
    print(f"Total PDFs found: {summary['total_input']}")
    print(f"Attempted this run: {summary['attempted']}")
    print(f"  - Success: {summary['success']}")
    print(f"  - Failure: {summary['failure']}")
    print(f"Total PDFs in output: {summary['total_pdf_out']}")
    print(f"Total .json saved: {summary['total_json']}")
    print(f"Total failures overall: {summary['total_failures']}")
    print(f"Throughput: {throughput:.2f} PDFs/sec")


if __name__ == "__main__":
    main()
