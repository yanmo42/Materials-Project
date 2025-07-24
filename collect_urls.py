#!/usr/bin/env python3
import argparse, json
from pathlib import Path
import requests
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor

OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"

def candidate_urls(js: dict) -> list[str]:
    locs = [js.get("best_oa_location"), *(js.get("oa_locations") or [])]
    urls = [l.get("url_for_pdf") or l.get("url") for l in locs if l]
    return list(dict.fromkeys(urls))  # dedupe

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True, help="Unpaywall contact email")
    ap.add_argument("--max", type=int, default=1000, help="Max number of works to scan")
    ap.add_argument("--threads", type=int, default=8, help="Number of threads for fetching OA URLs")
    ap.add_argument("--output", default="jobs.jsonl", help="where to write DOI/year/urls")
    args = ap.parse_args()

    out = Path(args.output)
    seen = set()
    mode = "w"
    if out.exists():
        # Load existing DOIs to avoid duplicates
        with out.open("r") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    seen.add(rec.get("doi"))
                except json.JSONDecodeError:
                    continue
        mode = "a"

    sess = requests.Session()
    sess.headers["User-Agent"] = "oa-sel 1.5"

    scanned = 0
    written = 0
    skipped = 0
    cursor = "*"
    write_lock = threading.Lock()

    print(f"🔎 Collecting up to {args.max} works → {out} (mode={'append' if mode=='a' else 'overwrite'})")
    pbar = tqdm(total=args.max, desc="Searching", unit="works", ncols=80)

    with out.open(mode) as fout:
        def process_work(work):
            nonlocal written, skipped
            doi = work.get("doi")
            if not doi or doi in seen:
                skipped += 1
                return
            year = work.get("publication_year") or "na"
            try:
                upw = sess.get(
                    UNPAYWALL.format(doi),
                    params={"email": args.email},
                    timeout=15
                )
                if upw.status_code != 200:
                    return
                urls = candidate_urls(upw.json())
                if not urls:
                    return
                record = {"doi": doi, "year": year, "urls": urls}
                with write_lock:
                    fout.write(json.dumps(record) + "\n")
                    written += 1
                    seen.add(doi)
            except Exception:
                return

        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            while scanned < args.max:
                js = sess.get(
                    OPENALEX,
                    params={
                      "search": "ferroelectric",
                      "filter": "open_access.is_oa:true",
                      "per-page": 200,
                      "cursor": cursor
                    },
                    timeout=30
                ).json()
                print(js.get("results", []))
                
                for work in js.get("results", []):
                    if scanned >= args.max:
                        break
                    scanned += 1
                    pbar.update(1)
                    executor.submit(process_work, work)

                cursor = js.get("meta", {}).get("next_cursor")
                if not cursor:
                    break
            executor.shutdown(wait=True)

    pbar.close()

    # Summary
    print(f"\n📊 Summary:")
    print(f"   🔍 Scanned works: {scanned} (limit {args.max})")
    print(f"   📥 New OA URLs written: {written}")
    print(f"   🚫 Skipped duplicates: {skipped}")
    print(f"   📄 Jobs in file: {len(seen)}")
    print(f"   📁 File location: {out.resolve()}")

if __name__ == "__main__":
    main()
