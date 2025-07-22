#!/usr/bin/env python3
"""
count_unique_papers_fast.py
Faster version of count_unique_papers.py: parallel SHA-1 hashing + progress bar.
By default only shows a progress bar and summary; use --verbose for details.
"""

import argparse
import hashlib
import sys
import time
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(
        description="Count unique PDFs by content SHA-1, in parallel with a progress bar."
    )
    p.add_argument(
        "--root", "-r",
        default="ferro_papers",
        help="Root folder to scan for PDFs"
    )
    p.add_argument(
        "--workers", "-w",
        type=int, default=4,
        help="Number of threads for hashing (I/O-bound)"
    )
    p.add_argument(
        "--chunk-size", "-c",
        type=int, default=4 * 1024 * 1024,
        help="Read chunk size in bytes (default 4 MiB)"
    )
    p.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show per-file errors and duplicate-group details"
    )
    return p.parse_args()

def sha1_file(path: Path, chunk_size: int):
    """Compute SHA-1 hash of a file; returns (hash, error_msg)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest().upper(), None
    except Exception as e:
        return None, str(e)

def main():
    args = parse_args()
    root = Path(args.root)
    if not root.is_dir():
        sys.exit(f"❌ Folder not found: {root.resolve()}")

    pdf_files = list(root.rglob("*.pdf"))
    total = len(pdf_files)
    if total == 0:
        sys.exit("❌ No PDF files found in the directory tree")

    if args.verbose:
        print(f"🔍  Scanning {total} PDF files with {args.workers} workers…")

    hash_to_paths = defaultdict(list)
    error_details = []

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(sha1_file, path, args.chunk_size): path
            for path in pdf_files
        }

        for future in tqdm(
            as_completed(futures),
            total=total,
            desc="Hashing PDFs",
            unit="file",
            ncols=80,
            leave=not args.verbose  # keep progress bar if not verbose
        ):
            path = futures[future]
            file_hash, err = future.result()
            if err:
                error_details.append((path, err))
            else:
                hash_to_paths[file_hash].append(path)

    total_time = time.time() - start_time
    processed = total - len(error_details)
    unique = len(hash_to_paths)
    duplicates = processed - unique

    # Always print summary last
    print(f"\n⏱️  Done in {total_time:.1f}s — {processed}/{total} processed, {len(error_details)} errors")
    print("\n📊  Results")
    print("────────────────────────────")
    print(f"Total files   : {total}")
    print(f"Processed     : {processed}")
    print(f"Errors        : {len(error_details)}")
    print(f"Unique files  : {unique}")
    print(f"Duplicates    : {duplicates}")

    if error_details and args.verbose:
        print("\n⚠️  Errors encountered:")
        for path, err in error_details:
            print(f"   • {path.relative_to(root)}: {err}")
    elif error_details:
        print("\n⚠️  (Use --verbose to see error details)")

    if duplicates > 0 and args.verbose:
        print("\n🔁  Duplicate groups (same content, different paths):")
        group_no = 0
        for h, paths in hash_to_paths.items():
            if len(paths) > 1:
                group_no += 1
                print(f"\nGroup {group_no} — SHA-1 {h} ({len(paths)} copies):")
                for p in paths:
                    size_mb = p.stat().st_size / (1024*1024)
                    print(f"   • {p.relative_to(root)} ({size_mb:.1f} MB)")
        savings = sum(
            paths[0].stat().st_size * (len(paths)-1)
            for paths in hash_to_paths.values() if len(paths)>1
        )
        print(f"\n💾  Potential space savings: {savings/(1024*1024):.1f} MB")
    elif duplicates > 0:
        print("\n🔁  (Use --verbose to list duplicate file groups)")

if __name__ == "__main__":
    main()
