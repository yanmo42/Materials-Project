#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# Optional overall progress bar
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

def discover_pdfs(root: Path):
    return sorted(root.rglob("*.pdf"))

def out_dir_for(pdf: Path, in_root: Path, out_root: Path, flatten: bool):
    if flatten:
        return out_root / pdf.stem
    rel = pdf.relative_to(in_root)
    # preserve subdirs, put results beside the PDF name (without .pdf)
    return out_root / rel.with_suffix("")

def has_existing_output(dest: Path) -> bool:
    """Return True if this dest already contains any markdown (recursively)
    or our explicit success marker."""
    if (dest / "MINERU_OK").exists():
        return True
    if not dest.exists():
        return False
    # MinerU nests outputs (e.g., <dest>/<hash>/auto/*.md)
    for pattern in ("*.md", "*.markdown"):
        if any(dest.rglob(pattern)):
            return True
    return False

def run_one(mineru_bin: str, pdf: Path, dest: Path,
            backend: str, method: str, device: str,
            timeout: int, force: bool):
    # Guard: skip if output already exists and not forcing
    if not force and has_existing_output(dest):
        return 0, "skipped (exists)"

    dest.mkdir(parents=True, exist_ok=True)

    cmd = [mineru_bin, "-p", str(pdf), "-o", str(dest),
           "-b", backend, "-m", method, "-d", device]

    with open(dest / "stdout.txt", "w") as out, open(dest / "stderr.txt", "w") as err:
        try:
            rc = subprocess.run(cmd, timeout=timeout, stdout=out, stderr=err).returncode
            if rc == 0:
                # Mark success to be robust even if MD filenames change
                try:
                    (dest / "MINERU_OK").write_text("ok\n", encoding="utf-8")
                except Exception:
                    pass
            return rc, "ok" if rc == 0 else "fail"
        except subprocess.TimeoutExpired:
            err.write("\nTimed out.\n")
            return 124, "timeout"
        except Exception as e:
            err.write(f"\nRunner exception: {e}\n")
            return 1, "error"

def main():
    ap = argparse.ArgumentParser(description="Batch MinerU over downloaded-pdfs/**.pdf")
    ap.add_argument("--input",  default=str(Path.cwd() / "downloaded-pdfs"),
                    help="Input root (default: $PWD/downloaded-pdfs)")
    ap.add_argument("--output", default=str(Path.cwd() / "mineru-out"),
                    help="Output root (default: $PWD/mineru-out)")
    ap.add_argument("-w","--workers", type=int, default=6)
    ap.add_argument("-t","--timeout", type=int, default=900, help="Per-file timeout (s)")
    ap.add_argument("-b","--backend", default="pipeline")
    ap.add_argument("-m","--method",  default="auto", choices=["auto","txt","ocr"])
    ap.add_argument("-d","--device",  default="cpu",  choices=["cpu","cuda","mps"])
    ap.add_argument("--flatten", action="store_true",
                    help="Put each output in output/<pdf-stem>/ (ignore subdirs)")
    ap.add_argument("--force",   action="store_true",
                    help="Re-run even if output already exists")
    ap.add_argument("--limit", type=int, default=None,
                    help="Max number of PDFs to process (applied AFTER skipping already-done)")
    args = ap.parse_args()

    mineru_bin = shutil.which("mineru")
    if not mineru_bin:
        print("ERROR: 'mineru' not found on PATH for this env.", file=sys.stderr)
        print(f"Python: {sys.executable}", file=sys.stderr)
        print(f"PATH: {os.environ.get('PATH')}", file=sys.stderr)
        sys.exit(2)

    in_root  = Path(args.input).resolve()
    out_root = Path(args.output).resolve()
    if not in_root.exists():
        print(f"ERROR: input root not found: {in_root}", file=sys.stderr)
        sys.exit(3)
    out_root.mkdir(parents=True, exist_ok=True)

    all_pdfs = discover_pdfs(in_root)
    if not all_pdfs:
        print(f"No PDFs found under {in_root}")
        sys.exit(0)

    # Pre-filter items that already have output so --limit advances to NEW work
    work_items = []
    already_done = 0
    for pdf in all_pdfs:
        dest = out_dir_for(pdf, in_root, out_root, args.flatten)
        if not args.force and has_existing_output(dest):
            already_done += 1
            continue
        work_items.append((pdf, dest))

    if args.limit and args.limit > 0:
        work_items = work_items[:args.limit]

    print(f"MinerU: {mineru_bin}")
    print(f"Input : {in_root}")
    print(f"Output: {out_root}")
    print(f"Found {len(all_pdfs)} PDFs | already have output for {already_done} | to run now: {len(work_items)}")
    if not work_items:
        print("Nothing to do. Use --force to re-run everything.")
        sys.exit(0)

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(
            run_one, mineru_bin, pdf, dest,
            args.backend, args.method, args.device, args.timeout, args.force
        ): (pdf, dest) for (pdf, dest) in work_items}

        bar = tqdm(total=len(work_items), unit="pdf", dynamic_ncols=True, desc="Overall") if tqdm else None

        for fut in as_completed(futures):
            pdf, dest = futures[fut]
            rc, status = fut.result()
            mark = "✓" if rc == 0 else "✗"
            line = f"{mark} {pdf.relative_to(in_root)}  ->  {dest.relative_to(out_root)}  [{status}]"

            if rc == 0:
                ok += 1
            else:
                fail += 1

            if bar:
                bar.update(1)
                bar.set_postfix_str(f"{ok}✓ / {fail}✗", refresh=False)
                # keep the bar intact while logging lines
                tqdm.write(line)
            else:
                print(line)

        if bar:
            bar.close()

    print(f"\nDone. Success: {ok}  Failures: {fail}  Total: {ok+fail}")
    if fail:
        print("Check failing cases' stderr.txt for details.")

if __name__ == "__main__":
    main()
