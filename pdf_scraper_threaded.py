#!/usr/bin/env python3
# pdf_scraper
# 
"""
Test with:
  pdf_scraper.py --email you@example.com --max 60 --threads 12 --selenium-threads 4 --verbose

"""

import argparse, hashlib, queue, threading, time, re, urllib.parse as urlparse
from pathlib import Path
import requests
from tqdm import tqdm
from selenium.webdriver.chrome.service import Service

# -------------------------------------------------------------------- #
# constants
# -------------------------------------------------------------------- #
OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"

OUT_DIR   = Path("ferro_papers")
FAIL_DIR  = Path("failed_papers")           # zero‑byte markers for hard fails
TMP_DIR   = OUT_DIR / "_tmp"                # Selenium download folder

UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

HARDBLOCK = {"pubs.acs.org", "onlinelibrary.wiley.com",
             "www.science.org", "advances.sciencemag.org", "pubs.aip.org", 
             "www.sciencedirect.com", "sciencedirect.com", "linkinghub.elsevier.com"}

PDF_RE = re.compile(r"https?://[^\"'<> ]+\.pdf(?:\?[^\"'<> ]*)?", re.I)

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
                    continue  # leave for Selenium or helpers
                if try_direct(url, tgt):
                    ok = True
                    break

        if not ok and rescue_enabled:
            stats["miss"] += 1
        elif ok:
            stats["ok"] += 1

        bar.update(1)
        q.task_done()

# -------------------------------------------------------------------- #
# Selenium utilities
# -------------------------------------------------------------------- #
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc

COMMON_PDF_BUTTONS = [
    "Download PDF", "View PDF", "Full text PDF",
    "Article PDF", "Get PDF", "PDF"  # fallback short text
]


def make_driver(download_dir: Path, driver_path: str, timeout=45, thread_id=0, headless=False):
    opts = uc.ChromeOptions()
    # Keep visible for now (headless often blocks downloads)
    
    if headless:
        opts.add_argument("--headless=new")


    opts.add_argument("--disable-gpu")
    opts.add_argument("--window-size=1200,1000")
    # Offset windows so they don't overlap completely
    x_offset = (thread_id % 3) * 400
    y_offset = (thread_id // 3) * 300
    opts.add_argument(f"--window-position={x_offset},{y_offset}")
    
    opts.add_experimental_option("prefs", {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    })

    service = Service(executable_path=driver_path)
    drv = uc.Chrome(service=service, options=opts)


    drv.set_page_load_timeout(timeout)
    drv.execute_cdp_cmd("Network.setUserAgentOverride", {"userAgent": UA})
    drv.execute_cdp_cmd("Network.setExtraHTTPHeaders", {"headers": {"Referer": "https://doi.org"}})
    drv.execute_cdp_cmd("Page.setDownloadBehavior", {"behavior": "allow", "downloadPath": str(download_dir)})
    return drv


# ---------------- Selenium helpers ----------------

def extract_pdf_from_html(html: str) -> list[str]:
    """Return list of absolute PDF URLs found in raw HTML."""
    # 1) <meta name="citation_pdf_url" content="..."></meta>
    metas = re.findall(r'<meta[^>]+name=[\"\']citation_pdf_url[\"\'][^>]+content=[\"\']([^\"\']+)[\"\']', html, flags=re.I)
    urls = metas[:]
    # 2) any direct .pdf links
    urls.extend(PDF_RE.findall(html))
    # De‑duplicate while preserving order
    seen = set()
    unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            unique.append(u)
    return unique


def click_common_pdf_buttons(driver):
    for text in COMMON_PDF_BUTTONS:
        try:
            el = driver.find_element(By.XPATH, f"//a[normalize-space()='{text}'] | //button[normalize-space()='{text}']")
            el.click()
            return True
        except Exception:
            continue
    return False

# -------------------------------------------------------------------- #
# Selenium worker function for parallel processing
# -------------------------------------------------------------------- #

def selenium_worker(work_queue: queue.Queue, result_queue: queue.Queue, 
                   driver_path: str, thread_id: int, args, poll: int):
    """Worker function for parallel Selenium processing"""
    # Create unique download directory for this thread
    thread_tmp_dir = TMP_DIR / f"thread_{thread_id}"
    thread_tmp_dir.mkdir(exist_ok=True)
    
    driver = None
    try:
        driver = make_driver(thread_tmp_dir, driver_path, 
                           timeout=30 if args.slow else 15, 
                           thread_id=thread_id,
                           headless=args.headless)
        

        while True:
            try:
                # Get work item with timeout to avoid hanging
                work_item = work_queue.get(timeout=5)
                if work_item is None:  # Shutdown signal
                    break
                    
                doi, year, urls = work_item
                sha = pdf_name(doi)[:-4].upper()
                tgt = OUT_DIR / str(year) / f"{sha}.pdf"
                ok = False
                
                # Track what we tried for this paper
                attempts = []
                doi_short = doi.split('/')[-1] if '/' in doi else doi[:12]
                
                for url in urls:
                    try:
                        driver.get(url)
                        # time.sleep(1.5)
                        time.sleep(0.5 if not args.slow else 1.5)
                        title = driver.title.strip()
                        url_short = url if len(url) <= 100 else url[:97] + "…"

                        # Clean up title for display
                        if title and title != "New Tab" and len(title) > 5:
                            clean_title = title + "..." if len(title) > 50 else title
                            attempts.append(f"📄 {clean_title}")
                        else:
                            attempts.append("📄 (no title)")
                        attempts.append(f"🔗 {url_short}")

                        html = driver.page_source
                        
                        # Save debug HTML with thread ID to avoid conflicts
                        if args.verbose:
                            debug_file = f"selenium_debug_thread_{thread_id}.html"
                            with open(debug_file, "w", encoding="utf-8") as f: 
                                f.write(html)

                        # 1) click any obvious PDF button
                        button_clicked = click_common_pdf_buttons(driver)
                        if button_clicked:
                            attempts.append("🖱️ Clicked PDF download button")
                        time.sleep(0.5 if not args.slow else 2)

                        # 2) examine downloads directory (thread-specific)
                        for _ in range(poll):
                            tmp_pdfs = list(thread_tmp_dir.glob("*.pdf"))
                            if tmp_pdfs:
                                tmp_pdfs[0].rename(tgt)
                                ok = True
                                break
                            time.sleep(0.5)
                        if ok:
                            break

                        # 3) fallback: scrape HTML for PDF links
                        pdf_links = extract_pdf_from_html(html)
                        if pdf_links:
                            attempts.append(f"🔗 Found {len(pdf_links)} PDF link(s) in HTML")
                        
                        for pdf_link in pdf_links:
                            if try_direct(pdf_link, tgt):
                                ok = True
                                break
                        if ok:
                            break
                            
                    except Exception as e:
                        error_msg = str(e)[:40] + "..." if len(str(e)) > 40 else str(e)
                        attempts.append(f"⚠️ Error: {error_msg}")
                        continue
                
                # Send result back to main thread
                result = {
                    'success': ok,
                    'doi': doi,
                    'doi_short': doi_short,
                    'year': year,
                    'sha': sha,
                    'attempts': attempts,
                    'thread_id': thread_id
                }
                result_queue.put(result)
                work_queue.task_done()
                
            except queue.Empty:
                # No more work available, continue checking
                continue
            except Exception as e:
                # Handle any other errors
                print(f"Thread {thread_id} error: {e}")
                work_queue.task_done()
                continue
                
    except Exception as e:
        print(f"Thread {thread_id} failed to initialize: {e}")
    finally:
        # Clean up driver
        if driver:
            try:
                driver.quit()
            except Exception:
                pass
        # Clean up thread-specific temp directory
        try:
            for f in thread_tmp_dir.glob("*"):
                f.unlink()
            thread_tmp_dir.rmdir()
        except Exception:
            pass

# -------------------------------------------------------------------- #
# main
# -------------------------------------------------------------------- #

def main():
    start_time = time.time()

    # --- CLI ---
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8, help="Threads for direct download phase")
    ap.add_argument("--selenium-threads", type=int, default=3, help="Number of parallel Selenium instances")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--rescue", dest="rescue", action="store_true", help="enable Selenium second pass (default)")
    grp.add_argument("--no-rescue", dest="rescue", action="store_false", help="skip Selenium pass")
    ap.set_defaults(rescue=True)
    ap.add_argument("--slow", action="store_true", help="poll longer (useful for flaky sites)")
    ap.add_argument("--verbose", "-v", action="store_true", help="show page titles and extra details")
    ap.add_argument("--jobs-file", help="JSONL file of pre-collected (doi, year, urls) tuples")
    ap.add_argument("--headless", action="store_true", help="run all Selenium browsers in headless mode (may affect downloads)")
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    FAIL_DIR.mkdir(exist_ok=True)
    TMP_DIR.mkdir(exist_ok=True)

    sess = requests.Session(); sess.headers["User-Agent"] = "oa-sel 1.5"

   # --- load jobs ---
    if args.jobs_file:
        import json
        from pathlib import Path

            # 1) One-time scan of already-downloaded PDFs and .fail markers
        existing = {
                p.stem
                for p in OUT_DIR.glob("**/*.pdf")
            } | {
                f.stem
                for f in FAIL_DIR.glob("**/*.fail")
            }

        # 2) Load & filter the jobs.jsonl
        jobs = []
        print(f"🔎  Loading jobs from {args.jobs_file} …")
        with Path(args.jobs_file).open() as fin:
            for line in fin:
                rec = json.loads(line)
                doi, year, urls = rec["doi"], rec["year"], rec["urls"]
                sha = pdf_name(doi)[:-4].upper()
                if sha in existing:
                    continue
                jobs.append((doi, year, urls))


        scanned = len(jobs)
        print(f"• {scanned} works loaded from jobs file")

        if not jobs:
            print("Nothing new to download – exiting.")
            return
    
    
    else:
        #OG logic
        jobs, scanned, cursor = [], 0, "*"
        print("🔎  Collecting DOIs + OA URLs …")

        while scanned < args.max:
            js = sess.get(OPENALEX, params={"search": "ferroelectric", "filter": "open_access.is_oa:true", "per-page": 200, "cursor": cursor}, timeout=40).json()
            for work in js["results"]:
                if scanned >= args.max: break
                scanned += 1
                doi = work.get("doi"); year = work.get("publication_year") or "na"
                if not doi: continue
                sha = pdf_name(doi)[:-4].upper()
                if tried_before(sha): continue
                upw = sess.get(UNPAYWALL.format(doi), params={"email": args.email}, timeout=15)
                if upw.status_code != 200: continue
                urls = candidate_urls(upw.json())
                if urls: jobs.append((doi, year, urls))
            cursor = js["meta"]["next_cursor"]
            if not cursor: break

        print(f"•  {len(jobs)} works with candidate URLs")
        if not jobs: return


    # --- direct pass ---
    stats = {"ok": 0, "miss": 0}; bar = tqdm(total=len(jobs), unit="pdf", desc="DIRECT", ncols=80)
    q = queue.Queue()
    for _ in range(min(args.threads, len(jobs))):
        threading.Thread(target=worker, args=(q, bar, stats, args.rescue), daemon=True).start()
    for j in jobs: q.put(j)
    q.join(); bar.close()

    first_pass_pdfs           = stats["ok"]          # how many PDFs we actually saved
    first_pass_misses         = stats["miss"]        # direct downloads that failed
    first_pass_not_downloaded = len(jobs) - first_pass_pdfs  # only shown when --no-rescue

    missing = [j for j in jobs if not (OUT_DIR / str(j[1]) / pdf_name(j[0])).exists()]
    
    if args.rescue and missing:
        print(f"\n🕷  Selenium second pass on {len(missing)} remaining URLs with {args.selenium_threads} parallel instances…")
        drv_path = ChromeDriverManager().install()
        poll = 6

        # Set up parallel Selenium processing
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add all missing jobs to work queue
        for job in missing:
            work_queue.put(job)
        
        # Start worker threads
        selenium_threads = []
        for thread_id in range(args.selenium_threads):
            t = threading.Thread(
                target=selenium_worker, 
                args=(work_queue, result_queue, drv_path, thread_id, args, poll),
                daemon=True
            )
            t.start()
            selenium_threads.append(t)
        
        selenium_pdfs = selenium_fails = 0
        success_messages = []
        failure_messages = []
        
        # Process results with progress bar
        with tqdm(total=len(missing), unit="pdf", desc="SELENIUM") as pbar:
            processed = 0
            while processed < len(missing):
                try:
                    result = result_queue.get(timeout=60)  # 1 minute timeout
                    processed += 1
                    
                    if result['success']:
                        selenium_pdfs += 1
                        if args.verbose and result['attempts']:
                            success_messages.append(f"✅ [T{result['thread_id']}] {result['doi_short']}")
                            for attempt in result['attempts']:
                                success_messages.append(f"    {attempt}")
                        else:
                            success_messages.append(f"✅ [T{result['thread_id']}] {result['doi_short']}")
                    else:
                        selenium_fails += 1
                        # Create fail marker
                        marker = FAIL_DIR / str(result['year']) / f"{result['sha']}.fail"
                        marker.parent.mkdir(parents=True, exist_ok=True)
                        marker.touch()
                        
                        if args.verbose and result['attempts']:
                            failure_messages.append(f"❌ [T{result['thread_id']}] {result['doi_short']}")
                            for attempt in result['attempts']:
                                failure_messages.append(f"    {attempt}")
                        else:
                            failure_messages.append(f"❌ [T{result['thread_id']}] {result['doi_short']}")
                    
                    # Update progress bar
                    pbar.set_description(f"SELENIUM [{selenium_pdfs}✅ {selenium_fails}❌]")
                    pbar.update(1)
                    
                except queue.Empty:
                    print(f"Warning: Timeout waiting for results. {processed}/{len(missing)} completed.")
                    break
        
        # Signal threads to stop and wait for them
        for _ in range(args.selenium_threads):
            work_queue.put(None)
        
        for t in selenium_threads:
            t.join(timeout=30)  # Give threads time to clean up
        
        # Display results after progress bar is complete
        print(f"\n📋 Selenium Results Summary:")
        if success_messages:
            print(f"\n✅ Successfully downloaded ({selenium_pdfs}):")
            for msg in success_messages:
                print(f"  {msg}")
        
        if failure_messages:
            print(f"\n❌ Failed downloads ({selenium_fails}):")
            for msg in failure_messages:
                print(f"  {msg}")
    else:
        selenium_pdfs = selenium_fails = 0

    import gc
    gc.collect() # force garbage collection so driver shuts off

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
        print(f"    └─ Selenium rescue ({args.selenium_threads} parallel): {selenium_pdfs} PDFs, {selenium_fails} fails")
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