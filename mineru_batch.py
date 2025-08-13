#!/usr/bin/env python3
import argparse, os, shutil, subprocess, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

def discover_pdfs(root: Path):
    return sorted(root.rglob("*.pdf"))

def out_dir_for(pdf: Path, in_root: Path, out_root: Path, flatten: bool):
    if flatten:
        return out_root / pdf.stem
    rel = pdf.relative_to(in_root)
    # preserve subdirs, put results beside the PDF name (without .pdf)
    return out_root / rel.with_suffix("")

def run_one(mineru_bin: str, pdf: Path, dest: Path, backend: str, method: str, device: str, timeout: int, force: bool):
    dest.mkdir(parents=True, exist_ok=True)
    md_glob = list(dest.glob("*.md"))
    if md_glob and not force:
        return 0, "skipped (exists)"

    cmd = [mineru_bin, "-p", str(pdf), "-o", str(dest), "-b", backend, "-m", method, "-d", device]
    with open(dest / "stdout.txt", "w") as out, open(dest / "stderr.txt", "w") as err:
        try:
            rc = subprocess.run(cmd, timeout=timeout).returncode
            return rc, "ok" if rc == 0 else "fail"
        except subprocess.TimeoutExpired:
            err.write("\nTimed out.\n")
            return 124, "timeout"
        except Exception as e:
            err.write(f"\nRunner exception: {e}\n")
            return 1, "error"

def main():
    ap = argparse.ArgumentParser(description="Batch MinerU over downloaded-pdfs/**.pdf")
    ap.add_argument("--input",  default=str(Path.cwd() / "downloaded-pdfs"), help="Input root (default: $PWD/downloaded-pdfs)")
    ap.add_argument("--output", default=str(Path.cwd() / "mineru-out"),     help="Output root (default: $PWD/mineru-out)")
    ap.add_argument("-w","--workers", type=int, default=6)
    ap.add_argument("-t","--timeout", type=int, default=900, help="Per-file timeout (s)")
    ap.add_argument("-b","--backend", default="pipeline")
    ap.add_argument("-m","--method",  default="auto", choices=["auto","txt","ocr"])
    ap.add_argument("-d","--device",  default="cpu",  choices=["cpu","cuda","mps"])
    ap.add_argument("--flatten", action="store_true", help="Put each output in output/<pdf-stem>/ (ignore subdirs)")
    ap.add_argument("--force",   action="store_true", help="Re-run even if a .md already exists in dest")
    # NEW: limit how many PDFs to process this run
    ap.add_argument("--limit", type=int, default=None, help="Max number of PDFs to process")
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

    pdfs = discover_pdfs(in_root)
    # Apply limit (if provided)
    if args.limit and args.limit > 0:
        pdfs = pdfs[:args.limit]

    if not pdfs:
        print(f"No PDFs found under {in_root}")
        sys.exit(0)

    print(f"MinerU: {mineru_bin}")
    print(f"Input : {in_root}")
    print(f"Output: {out_root}")
    print(f"Found {len(pdfs)} PDFs. Running with {args.workers} workers…\n")

    ok = fail = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {}
        for pdf in pdfs:
            dest = out_dir_for(pdf, in_root, out_root, args.flatten)
            futures[ex.submit(run_one, mineru_bin, pdf, dest, args.backend, args.method, args.device, args.timeout, args.force)] = (pdf, dest)
        for fut in as_completed(futures):
            pdf, dest = futures[fut]
            rc, status = fut.result()
            mark = "✓" if rc == 0 else "✗"
            print(f"{mark} {pdf.relative_to(in_root)}  ->  {dest.relative_to(out_root)}  [{status}]")
            if rc == 0: ok += 1
            else: fail += 1

    print(f"\nDone. Success: {ok}  Failures: {fail}  Total: {ok+fail}")
    if fail:
        print("Check failing cases' stderr.txt for details.")

if __name__ == "__main__":
    main()
