#!/usr/bin/env python3
# OA ferroelectric harvester – v1.3  (Windows‑optimised, cross‑platform)

import argparse, hashlib, queue, threading, time, re, urllib.parse as urlparse
from pathlib import Path
import requests
from tqdm import tqdm

# -------------------------------------------------------------------- #
# constants
# -------------------------------------------------------------------- #
OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"

OUT_DIR   = Path("ferro_papers")
FAIL_DIR  = Path("failed_papers")           # zero‑byte markers for hard fails

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

HARDBLOCK = {"pubs.acs.org", "onlinelibrary.wiley.com",
             "www.science.org", "advances.sciencemag.org", "pubs.aip.org"}

# -------------------------------------------------------------------- #
# helpers
# -------------------------------------------------------------------- #
def sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest().upper()
def pdf_name(doi: str) -> str: return f"{sha1(doi)}.pdf"

def candidate_urls(js: dict) -> list[str]:
    locs = [js.get("best_oa_location"), *(js.get("oa_locations") or [])]
    urls = [l["url_for_pdf"] or l["url"] for l in locs if l]
    return list(dict.fromkeys(urls))

def tried_before(sha: str) -> bool:
    """True if we already have this PDF or a .fail marker."""
    return (any(OUT_DIR.glob(f"**/{sha}.pdf")) or
            any(FAIL_DIR.glob(f"**/{sha}.fail")))

def try_direct(url: str, tgt: Path, timeout=30) -> bool:
    try:
        with requests.get(url, headers={"User-Agent": UA,
                                        "Accept": "application/pdf,*/*",
                                        "Referer": url},
                          stream=True, timeout=timeout,
                          allow_redirects=True) as r:
            if "pdf" not in r.headers.get("content-type", "").lower():
                return False
            tgt.parent.mkdir(parents=True, exist_ok=True)
            with open(tgt, "wb") as f:
                for chunk in r.iter_content(16384):
                    f.write(chunk)
        return tgt.stat().st_size > 1024
    except Exception:
        return False

# -------------------------------------------------------------------- #
# first‑pass worker (direct only)
# -------------------------------------------------------------------- #
def worker(q: queue.Queue, bar: tqdm, stats: dict, rescue_enabled: bool):
    while True:
        item = q.get()
        if item is None:
            break
        doi, year, urls = item
        sha = pdf_name(doi)[:-4].upper()
        tgt = OUT_DIR / str(year) / f"{sha}.pdf"
        ok  = tgt.exists()

        if not ok:
            for url in urls:
                host = urlparse.urlparse(url).netloc
                if host in HARDBLOCK:
                    continue          # leave for Selenium
                if try_direct(url, tgt):
                    ok = True
                    break

        # With --no-rescue, we don't create .fail markers for direct download failures
        # Only with --rescue do we track misses (they get passed to Selenium)
        if not ok:
            if rescue_enabled:
                stats["miss"] += 1  # Will be passed to Selenium
            # No .fail marker creation for --no-rescue
        else:
            stats["ok"] += 1

        print(f"{'✅' if ok else '➖'} {doi}")
        bar.update(1)
        q.task_done()

# -------------------------------------------------------------------- #
# Selenium utilities (only used if --rescue)
# -------------------------------------------------------------------- #
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

def driver_major(path: str) -> int:
    p = Path(path).resolve()
    for parent in [p.parent] + list(p.parents):
        m = re.match(r"^(\d+)", parent.name)
        if m: return int(m.group(1))
    raise RuntimeError(f"Cannot parse version from {path!s}")

def make_driver(tmp: Path, driver_path: str, timeout=45) -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(tmp),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True})
    drv = uc.Chrome(options=opts, driver_executable_path=driver_path)
    drv.set_page_load_timeout(timeout)
    drv.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": UA})
    drv.execute_cdp_cmd("Network.setExtraHTTPHeaders",
                        {"headers": {"Referer": "https://doi.org"}})
    drv.execute_cdp_cmd("Page.setDownloadBehavior",
                        {"behavior": "allow", "downloadPath": str(tmp)})
    return drv

# -------------------------------------------------------------------- #
# main
# -------------------------------------------------------------------- #
def main():
    
    start_time = time.time()


    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--rescue",     dest="rescue", action="store_true",
                     help="enable Selenium second pass (default)")
    grp.add_argument("--no-rescue",  dest="rescue", action="store_false",
                     help="skip Selenium pass")
    ap.set_defaults(rescue=True)
    ap.add_argument("--slow", action="store_true",
                    help="poll 20 s per Selenium page (default 6 s)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    FAIL_DIR.mkdir(exist_ok=True)

    # Get initial counts before starting
    initial_pdfs = len(list(OUT_DIR.glob("**/*.pdf")))
    initial_fails = len(list(FAIL_DIR.glob("**/*.fail")))

    sess = requests.Session()
    sess.headers["User-Agent"] = "oa-sel 1.3"

    jobs, scanned, cursor = [], 0, "*"
    print("🔎  Collecting DOIs + OA URLs …")

    while scanned < args.max:
        resp = sess.get(OPENALEX, params={
                "search": "ferroelectric",
                "filter": "open_access.is_oa:true",
                "per-page": 200,
                "cursor": cursor}, timeout=40)
        js = resp.json()

        for work in js["results"]:
            if scanned >= args.max:
                break
            scanned += 1
            if scanned % 100 == 0:
                print(f"  OpenAlex scanned {scanned}")

            doi  = work.get("doi")
            if not doi:
                continue
            year = work.get("publication_year") or "na"
            sha  = pdf_name(doi)[:-4].upper()
            if tried_before(sha):
                continue            # already have PDF or .fail marker

            upw = sess.get(UNPAYWALL.format(doi),
                            params={"email": args.email}, timeout=15)
            if upw.status_code != 200:
                continue
            urls = candidate_urls(upw.json())
            if urls:
                jobs.append((doi, year, urls))

        cursor = js["meta"]["next_cursor"]
        if not cursor:
            break

    print(f"•  {len(jobs)} works with candidate URLs")
    if not jobs:
        # Still show final counts even if no new jobs
        final_pdfs = len(list(OUT_DIR.glob("**/*.pdf")))
        final_fails = len(list(FAIL_DIR.glob("**/*.fail")))
        print(f"\n📊  This run: 0 PDFs added  /  0 fails added")
        print(f"🎉  Library totals: {final_pdfs} PDFs  /  {final_fails} fails")
        print(f"📁  PDFs saved to:   {OUT_DIR.resolve()}")
        print(f"📁  Fail markers:    {FAIL_DIR.resolve()}")
        return

    # ---------------- fast direct pass ----------------------------------
    q = queue.Queue()
    stats = {"ok": 0, "miss": 0}
    bar = tqdm(total=len(jobs), unit="pdf", ncols=80, desc="DIRECT")

    for _ in range(min(args.threads, len(jobs))):
        threading.Thread(target=worker,
                         args=(q, bar, stats, args.rescue),
                         daemon=True).start()
    for j in jobs:
        q.put(j)
    q.join()
    bar.close()

    # Track first pass results
    first_pass_pdfs = stats["ok"]
    # Only count misses if rescue is enabled (they get passed to Selenium)
    first_pass_misses = stats["miss"] if args.rescue else 0
    # Calculate how many papers were attempted but not downloaded in --no-rescue mode
    first_pass_attempted = len(jobs)
    first_pass_not_downloaded = first_pass_attempted - first_pass_pdfs if not args.rescue else 0

    # ---------------- optional Selenium rescue -------------------------
    selenium_pdfs = 0
    selenium_fails = 0
    
    missing = [j for j in jobs
               if not (OUT_DIR / str(j[1]) / pdf_name(j[0])).exists()]

    if args.rescue and missing:
        print(f"\n🕷  Selenium second pass on {len(missing)} remaining URLs …")
        drv_path = ChromeDriverManager().install()
        tmp_dir  = OUT_DIR / "_tmp"
        drv = make_driver(tmp_dir, drv_path,
                          timeout=60 if args.slow else 45)
        poll = 40 if args.slow else 12

        for doi, year, urls in tqdm(missing, unit="pdf", ncols=80):
            sha = pdf_name(doi)[:-4].upper()
            tgt = OUT_DIR / str(year) / f"{sha}.pdf"
            ok  = False
            for url in urls:
                try:
                    drv.get(url)
                except Exception:
                    continue
                for _ in range(poll):
                    tmp_pdfs = list(tmp_dir.glob("*.pdf"))
                    if tmp_pdfs:
                        tmp_pdfs[0].rename(tgt)
                        ok = True
                        break
                    time.sleep(0.5)
                if ok:
                    break
            if ok:
                selenium_pdfs += 1
            else:                            # still failed → .fail marker
                selenium_fails += 1
                marker = FAIL_DIR / str(year) / f"{sha}.fail"
                marker.parent.mkdir(parents=True, exist_ok=True)
                marker.touch()
            print(f"{'✅' if ok else '❌'} selenium {doi}")
        drv.quit()

    # ---------------- summary ------------------------------------------
    # Calculate final counts and this run's additions
    final_pdfs = len(list(OUT_DIR.glob("**/*.pdf")))
    final_fails = len(list(FAIL_DIR.glob("**/*.fail")))
    
    run_pdfs_added = first_pass_pdfs + selenium_pdfs
    run_fails_added = selenium_fails  # Only Selenium creates .fail markers
    
    not_found = scanned - len(jobs)
    
    print(f"\n📊  This run: {run_pdfs_added} PDFs added  /  {run_fails_added} fails added. {len(jobs)} URLs checked")
    if args.rescue:
        print(f"    ├─ Direct download: {first_pass_pdfs} PDFs, 0 fails")
        print(f"    └─ Selenium rescue: {selenium_pdfs} PDFs, {selenium_fails} fails")
        if first_pass_misses > 0:
            print(f"    Note: {first_pass_misses} papers failed direct download, passed to Selenium")
    else:
        print(f"    └─ Direct download: {first_pass_pdfs} PDFs, 0 fails")
        if first_pass_not_downloaded > 0:
            print(f"    Note: {first_pass_not_downloaded} papers could not be downloaded directly")

    print(f"📈  Search stats: --max {args.max}, scanned {scanned}, found {len(jobs)} URLs, {not_found} without URLs")
    print(f"🎉  Library totals: {final_pdfs} PDFs  /  {final_fails} fails")
    print(f"📁  PDFs saved to:   {OUT_DIR.resolve()}")
    print(f"📁  Fail markers:    {FAIL_DIR.resolve()}")

    # time calculations
    end_time = time.time()
    total_runtime = end_time - start_time
    
    
    hours = int(total_runtime // 3600)
    minutes = int((total_runtime % 3600) // 60)
    seconds = int(total_runtime % 60)
    
    if hours > 0:
        runtime_str = f"{hours}h {minutes}m {seconds}s"
    elif minutes > 0:
        runtime_str = f"{minutes}m {seconds}s"
    else:
        runtime_str = f"{seconds}s"
    
    print(f"⏱️  Total runtime: {runtime_str} ({total_runtime:.1f} seconds)")

if __name__ == "__main__":
    main()