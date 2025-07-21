#!/usr/bin/env python3
"""
count_unique_papers.py
Count unique PDF files in ferro_papers/ by file content (SHA‑1 hash).
"""

import hashlib, sys, time
from pathlib import Path
from collections import defaultdict

ROOT = Path("ferro_papers")     # adjust if folder lives elsewhere
if not ROOT.is_dir():
    sys.exit(f"❌ Folder not found: {ROOT.resolve()}")

def sha1_file(path: Path, chunk=131072) -> str:
    """Calculate SHA-1 hash of file with error handling."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            while True:
                data = f.read(chunk)
                if not data:
                    break
                h.update(data)
        return h.hexdigest().upper()
    except (OSError, IOError) as e:
        print(f"⚠️  Error reading {path}: {e}")
        return None

hash_to_paths = defaultdict(list)
pdf_files = list(ROOT.glob("**/*.pdf"))

if not pdf_files:
    print("❌ No PDF files found in the directory tree")
    sys.exit(1)

print(f"🔍  Scanning {len(pdf_files)} PDF files …")

# Process files with progress indicator and error handling
processed = 0
errors = 0
start_time = time.time()

for i, p in enumerate(pdf_files, 1):
    # Show progress every 10 files or for large collections
    if i % max(1, len(pdf_files) // 20) == 0 or i == len(pdf_files):
        elapsed = time.time() - start_time
        rate = i / elapsed if elapsed > 0 else 0
        print(f"  Progress: {i}/{len(pdf_files)} ({i/len(pdf_files)*100:.1f}%) "
              f"- {rate:.1f} files/sec - Current: {p.name}")
    
    # Check if file is accessible and not empty
    if not p.exists():
        print(f"⚠️  File no longer exists: {p}")
        errors += 1
        continue
        
    if p.stat().st_size == 0:
        print(f"⚠️  Empty file skipped: {p}")
        errors += 1
        continue
    
    # Large file warning (>100MB)
    if p.stat().st_size > 100 * 1024 * 1024:
        print(f"📁  Processing large file ({p.stat().st_size / (1024*1024):.1f}MB): {p.name}")
    
    file_hash = sha1_file(p)
    if file_hash:
        hash_to_paths[file_hash].append(p)
        processed += 1
    else:
        errors += 1

total_time = time.time() - start_time

print(f"\n⏱️  Processing completed in {total_time:.2f} seconds")

total = len(pdf_files)
unique = len(hash_to_paths)
dupes = processed - unique

print("\n📊  Results")
print("────────────────────────────")
print(f"Total files   : {total}")
print(f"Processed     : {processed}")
print(f"Errors        : {errors}")
print(f"Unique files  : {unique}")
print(f"Duplicates    : {dupes}")

if dupes > 0:
    print(f"\n🔁  Duplicate groups (same content, different paths)")
    dupe_count = 0
    for h, paths in hash_to_paths.items():
        if len(paths) > 1:
            dupe_count += 1
            print(f"\nGroup {dupe_count} - SHA‑1 {h}  ({len(paths)} copies)")
            for p in paths:
                size_mb = p.stat().st_size / (1024 * 1024)
                print(f"   • {p.relative_to(ROOT)} ({size_mb:.1f}MB)")

# Summary of potential space savings
if dupes > 0:
    total_dupe_size = 0
    for h, paths in hash_to_paths.items():
        if len(paths) > 1:
            file_size = paths[0].stat().st_size
            total_dupe_size += file_size * (len(paths) - 1)
    
    print(f"\n💾  Potential space savings: {total_dupe_size / (1024*1024):.1f}MB by removing duplicates")