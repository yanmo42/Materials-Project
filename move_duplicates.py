#!/usr/bin/env python3
"""
move_duplicates.py
Find and move duplicate PDFs into a separate folder, keeping one copy in place.
"""

import argparse, shutil
from pathlib import Path
from collections import defaultdict
import hashlib
from tqdm import tqdm

def sha1_file(path: Path, chunk_size=4 * 1024 * 1024):
    """Compute SHA-1 hash of a file"""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest().upper()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="ferro_papers", help="Root folder of PDFs")
    p.add_argument("--duplicates-dir", default="ferro_papers_duplicates",
                   help="Where to move duplicate PDFs")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true", help="Show what would happen but don’t move anything")
    args = p.parse_args()

    root = Path(args.root)
    dup_root = Path(args.duplicates_dir)
    dup_root.mkdir(parents=True, exist_ok=True)

    print(f"🔍 Scanning PDFs under {root} …")
    pdf_files = list(root.rglob("*.pdf"))

    # Hash all PDFs
    hash_to_paths = defaultdict(list)
    for pdf in tqdm(pdf_files, desc="Hashing PDFs", unit="file"):
        try:
            h = sha1_file(pdf)
            hash_to_paths[h].append(pdf)
        except Exception as e:
            print(f"⚠️ Error hashing {pdf}: {e}")

    # Identify duplicate groups
    duplicate_groups = {h: paths for h, paths in hash_to_paths.items() if len(paths) > 1}
    print(f"\n📊 Found {len(duplicate_groups)} groups of duplicates")

    total_moved = 0
    for h, paths in duplicate_groups.items():
        keep = paths[0]  # keep the first one
        for duplicate in paths[1:]:
            rel_path = duplicate.relative_to(root)
            target_path = dup_root / rel_path
            target_path.parent.mkdir(parents=True, exist_ok=True)

            if args.dry_run:
                print(f"Would move {duplicate} → {target_path}")
            else:
                shutil.move(str(duplicate), str(target_path))
                print(f"✅ Moved duplicate {duplicate} → {target_path}")
                total_moved += 1

    print(f"\n🎉 Done. Moved {total_moved} duplicate PDFs to {dup_root}")

if __name__ == "__main__":
    main()
