#!/usr/bin/env python3
"""
count_unique_papers.py
Count unique PDF files in ferro_papers/ by file content (SHA‑1 hash).
"""

import hashlib, sys
from pathlib import Path
from collections import defaultdict

ROOT = Path("ferro_papers")     # adjust if folder lives elsewhere
if not ROOT.is_dir():
    sys.exit(f"❌ Folder not found: {ROOT.resolve()}")

def sha1_file(path: Path, chunk=131072) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest().upper()

hash_to_paths = defaultdict(list)
pdf_files     = list(ROOT.glob("**/*.pdf"))

print(f"🔍  Scanning {len(pdf_files)} PDF files …")
for p in pdf_files:
    hash_to_paths[sha1_file(p)].append(p)

total    = len(pdf_files)
unique   = len(hash_to_paths)
dupes    = total - unique

print("\n📊  Results")
print("────────────────────────────")
print(f"Total files   : {total}")
print(f"Unique files  : {unique}")
print(f"Duplicates    : {dupes}")

if dupes:
    print("\n🔁  Duplicate groups (same content, different paths)")
    for h, paths in hash_to_paths.items():
        if len(paths) > 1:
            print(f"\nSHA‑1 {h}  ({len(paths)} copies)")
            for p in paths:
                print("   •", p.relative_to(ROOT))
