#!/usr/bin/env python3
# scraperv3.py — headless PDF harvester with debug logging

import argparse, hashlib, queue, threading, time, re, urllib.parse as urlparse
from pathlib import Path
import requests, subprocess, shutil, platform, os, tempfile
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import TimeoutException
from webdriver_manager.chrome import ChromeDriverManager

# -------------------------------------------------------------------- #
# constants & directories
# -------------------------------------------------------------------- #
OPENALEX   = "https://api.openalex.org/works"
UNPAYWALL  = "https://api.unpaywall.org/v2/{}"
OUT_DIR    = Path("downloaded_pdfs")
FAIL_DIR   = Path("failed_downloads")
TMP_DIR    = OUT_DIR / "_tmp"
UA         = ("Mozilla/5.0 (X11; Linux x86_64) "
              "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")
HARDBLOCK  = {
    "pubs.acs.org","onlinelibrary.wiley.com","www.science.org",
    "advances.sciencemag.org","pubs.aip.org","www.sciencedirect.com",
    "sciencedirect.com","linkinghub.elsevier.com"
}
PDF_RE     = re.compile(r"https?://[^\"'<> ]+\.pdf(?:\?[^\"'<> ]*)?", re.I)

# Common button texts to look for when trying to click a PDF link
COMMON_PDF_BUTTONS = [
    "Download PDF",
    "View PDF",
    "Full text PDF",
    "Article PDF",
    "Get PDF",
    "PDF"
]


DRIVER_INIT_LOCK = threading.Lock()

# -------------------------------------------------------------------- #
# helpers
# -------------------------------------------------------------------- #
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest().upper()

def pdf_name(doi: str) -> str:
    """Convert a DOI string to a SHA1-based filename."""
    return f"{sha1(doi)}.pdf"

def candidate_urls(js: dict) -> list[str]:
    locs = [js.get("best_oa_location"), *(js.get("oa_locations") or [])]
    urls = [l["url_for_pdf"] or l["url"] for l in locs if l]
    return list(dict.fromkeys(urls))

def try_direct(url: str, tgt: Path, timeout=30) -> bool:
    try:
        with requests.get(
            url,
            headers={"User-Agent": UA, "Accept": "application/pdf,*/*", "Referer": url},
            stream=True, timeout=timeout, allow_redirects=True
        ) as r:
            if "pdf" not in r.headers.get("content-type","").lower():
                return False
            tgt.parent.mkdir(parents=True, exist_ok=True)
            with open(tgt, "wb") as f:
                for chunk in r.iter_content(16384):
                    f.write(chunk)
        return tgt.stat().st_size > 1024
    except Exception:
        return False

def extract_pdf_from_html(html: str) -> list[str]:
    metas = re.findall(
        r'<meta[^>]+name=[\"\']citation_pdf_url[\"\'][^>]+content=[\"\']([^\"\']+)[\"\']',
        html, flags=re.I
    )
    urls = metas[:]
    urls.extend(PDF_RE.findall(html))
    seen = set(); unique = []
    for u in urls:
        if u not in seen:
            seen.add(u); unique.append(u)
    return unique

# -------------------------------------------------------------------- #
# Chrome setup
# -------------------------------------------------------------------- #
def get_chrome_binary() -> str:
    candidates = []
    if platform.system() == "Linux":
        candidates = [
            "/usr/bin/google-chrome","/usr/bin/google-chrome-stable",
            "/usr/bin/chromium","/usr/bin/chromium-browser","/usr/bin/brave-browser"
        ]
    else:  # Windows
        candidates = [
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"
        ]
    for name in ["google-chrome","chromium","chrome"]:
        path = shutil.which(name)
        if path and os.path.isfile(path):
            return path
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise RuntimeError("No Chrome binary found.")

def get_chrome_major() -> tuple[str,str]:
    if "CHROME_VERSION" in os.environ:
        v = os.environ["CHROME_VERSION"].split(".")[0]
        return v, get_chrome_binary()
    binary = get_chrome_binary()
    out = subprocess.check_output([binary,"--version"], stderr=subprocess.DEVNULL).decode()
    m = re.search(r"\b(\d+)\.", out)
    if m: return m.group(1), binary
    raise RuntimeError("Could not parse Chrome version.")

def make_driver(download_dir: Path, driver_path: str,
                timeout=45, thread_id=0, headless=False):
    chrome_major, chrome_binary = get_chrome_major()
    profile_base = Path(tempfile.gettempdir())/"pdf_scraper_chrome"
    profile_base.mkdir(exist_ok=True, parents=True)
    user_data = profile_base/f"profile_{thread_id}_{os.getpid()}_{int(time.time())}"
    user_data.mkdir(exist_ok=True, parents=True)

    opts = ChromeOptions()
    opts.page_load_strategy = 'eager'
    if platform.system()=="Linux":
        opts.add_argument("--no-sandbox")
        opts.add_argument("--disable-dev-shm-usage")
        opts.add_argument("--disable-gpu")
    if headless:
        opts.add_argument("--headless=new")
    x,y = (thread_id%3)*420, (thread_id//3)*320
    opts.add_argument(f"--window-position={x},{y}")
    opts.add_argument("--window-size=1280,1024")
    opts.add_argument(f"--user-data-dir={user_data}")
    opts.binary_location = chrome_binary
    opts.add_argument(f"--user-agent={UA}")
    prefs = {
        "download.default_directory": str(download_dir.resolve()),
        "plugins.always_open_pdf_externally": True,
    }
    opts.add_experimental_option("prefs", prefs)

    with DRIVER_INIT_LOCK:
        svc = Service(executable_path=driver_path)
        drv = webdriver.Chrome(service=svc, options=opts)
        drv.set_page_load_timeout(timeout)
        drv.implicitly_wait(5)
        try:
            drv.execute_cdp_cmd("Page.setDownloadBehavior", {
                "behavior":"allow",
                "downloadPath":str(download_dir.resolve())
            })
        except Exception:
            pass
    return drv, user_data

# -------------------------------------------------------------------- #
# Selenium worker with full debug & timing
# -------------------------------------------------------------------- #
from selenium.common.exceptions import TimeoutException, UnexpectedAlertPresentException, WebDriverException, InvalidSessionIdException
import queue, traceback

def selenium_worker(work_queue: queue.Queue,
                    result_queue: queue.Queue,
                    driver_path: str,
                    thread_id: int,
                    args,
                    poll: int):
    # prepare per-thread temp dir
    thread_tmp = TMP_DIR / f"thread_{thread_id}"
    thread_tmp.mkdir(exist_ok=True, parents=True)

    # initialize driver
    try:
        time.sleep(thread_id * 1.5)
        driver, profile_dir = make_driver(
            thread_tmp,
            driver_path,
            timeout=45 if args.slow else 25,
            thread_id=thread_id,
            headless=args.headless
        )
        print(f"[T{thread_id}] initialized — queue size at start: {work_queue.qsize()}")
    except Exception:
        print(f"[T{thread_id}] Failed to init driver, exiting thread")
        traceback.print_exc()
        return

    # main work loop: only exit when we get a None sentinel
    while True:
        try:
            work_item = work_queue.get(timeout=10)
            if work_item is None:
                # clean shutdown signal
                break

            doi, year, urls = work_item
            doi_short = doi.split("/")[-1]
            print(f"[T{thread_id}] got work_item: {doi_short}")

            t0 = time.time()
            ok = False
            attempts: list[str] = []

            # clear out old downloads
            for f in thread_tmp.glob("*"):
                try: f.unlink()
                except: pass

            # try each candidate URL
            for idx, url in enumerate(urls):
                attempts.append(f"🔗 URL {idx+1}/{len(urls)}: {url[:80]}")

                # 1) navigate with alert & timeout handling
                try:
                    start_nav = time.time()
                    driver.get(url)
                    nav_time = time.time() - start_nav
                    attempts.append(f"⏱ get() → {nav_time:.1f}s")
                except UnexpectedAlertPresentException:
                    # dismiss site alert and skip this URL
                    try:
                        alert = driver.switch_to.alert
                        alert.accept()
                        attempts.append("⚠️ alert dismissed")
                    except:
                        pass
                    continue
                except TimeoutException:
                    attempts.append("⚠️ get() TIMEOUT")
                    continue
                 # ── NEW: heal crashed tabs ─────────────────────────────────────────────
                except (InvalidSessionIdException, WebDriverException) as e:
                    msg = (e.msg or "").lower()
                    # restart if the browser itself died
                    if any(t in msg for t in ("tab crashed", "chrome not reachable",
                                            "devtoolsactiveport", "invalid session")):
                        attempts.append(f"⚠️ {msg[:60]} — restarting driver")
                        try:
                            driver.quit()
                        except Exception:
                            pass
                        try:
                            driver, profile_dir = make_driver(thread_tmp, driver_path,
                                                            timeout=45 if args.slow else 25,
                                                            thread_id=thread_id,
                                                            headless=args.headless)
                        except Exception as new_e:
                            attempts.append(f"❌ could not restart driver: {new_e}")
                            break          # give up on this DOI
                        continue           # retry same DOI
                    else:
                        attempts.append(f"⚠️ WebDriver error: {msg[:60]}")
                        continue

                except OSError as e:          # disk / permission errors
                    attempts.append(f"⚠️ OS error: {e}")
                    continue

                # allow page to stabilize
                time.sleep(2)
                html = driver.page_source
                if args.verbose:
                    (thread_tmp / f"debug_{thread_id}_{idx}.html")\
                        .write_text(html, encoding="utf-8")

                # 2) attempt to click known PDF buttons
                clicked = False
                for text in COMMON_PDF_BUTTONS:
                    try:
                        els = driver.find_elements(By.XPATH,
                            f"//a[contains(normalize-space(),'{text}')]|"
                            f"//button[contains(normalize-space(),'{text}')]")
                        for el in els:
                            if el.is_displayed() and el.is_enabled():
                                driver.execute_script("arguments[0].click();", el)
                                attempts.append(f"🖱️ clicked “{text}”")
                                clicked = True
                                break
                        if clicked:
                            break
                    except Exception:
                        continue

                if clicked:
                    # wait for the download to start
                    time.sleep(3)

                    for _ in range(poll):
                        pdf_list = list(thread_tmp.glob("*.pdf"))
                        if pdf_list and pdf_list[0].stat().st_size > 1024:
                            pdf_file = pdf_list[0]

                            # 1) capture size (bytes) before moving
                            byte_size = pdf_file.stat().st_size
                            kb_size = byte_size / 1024

                            # 2) move into your output folder
                            tgt = OUT_DIR / str(year) / f"{sha1(doi)}.pdf"
                            tgt.parent.mkdir(parents=True, exist_ok=True)
                            pdf_file.rename(tgt)

                            # 3) validate magic header
                            try:
                                with open(tgt, "rb") as f:
                                    magic = f.read(4)
                            except Exception as e:
                                print(f"[T{thread_id}] ⚠️ Could not open {tgt}: {e}")
                                ok = False
                            else:
                                if magic != b"%PDF":
                                    # Invalid PDF: delete and log
                                    tgt.unlink()
                                    ok = False
                                    print(f"[T{thread_id}] ⚠️ Invalid header {magic!r}, deleted ({kb_size:.1f} KB)")
                                else:
                                    ok = True
                                    print(f"[T{thread_id}] ✅ Valid PDF saved ({kb_size:.1f} KB)")

                            # 4) break out if it really was a PDF
                            if ok:
                                break

                        # not ready yet—try again
                        time.sleep(0.5)

                    # exit the click-loop if we succeeded
                    if ok:
                        break

                # 3) fallback: extract any .pdf links from HTML
                links = extract_pdf_from_html(html)
                if links:
                    attempts.append(f"🔍 found {len(links)} links")
                    for link in links[:3]:
                        tgt = OUT_DIR / str(year) / f"{sha1(doi)}.pdf"
                        if try_direct(link, tgt):
                            # 1) capture size
                            byte_size = tgt.stat().st_size
                            kb_size   = byte_size / 1024

                            # 2) validate magic header
                            with open(tgt, "rb") as f:
                                magic = f.read(4)

                            if magic != b"%PDF":
                                attempts.append(f"⚠️ Invalid header {magic!r}, deleting ({kb_size:.1f} KB)")
                                tgt.unlink()
                                ok = False
                            else:
                                attempts.append(f"✅ direct‐link PDF saved ({kb_size:.1f} KB)")
                                print   (f"[T{thread_id}] ✅ direct link PDF saved ({kb_size:.1f} KB)")
                                ok = True

                            break

                    if ok:
                        break

            # record result
            duration = time.time() - t0
            print(f"[T{thread_id}] DONE {doi_short} in {duration:.1f}s, success={ok}")
            result_queue.put({
                'success': ok,
                'doi': doi,
                'doi_short': doi_short,
                'year': year,
                'sha': sha1(doi),
                'attempts': attempts,
                'thread_id': thread_id
            })
            work_queue.task_done()

        except queue.Empty:
            # no work; retry until we see the sentinel
            continue

        except Exception:
            # catch-all: log and keep thread alive
            print(f"[T{thread_id}] UNCAUGHT EXCEPTION, skipping current job:")
            traceback.print_exc()
            # you may want to mark a failure here:
            # result_queue.put({ 'success': False, ... })
            continue

    # cleanup on sentinel
    try:
        driver.quit()
    except:
        pass
    # remove tmp files & profile
    for f in thread_tmp.glob("*"):
        try: f.unlink()
        except: pass
    try: thread_tmp.rmdir()
    except: pass
    if profile_dir and profile_dir.exists():
        shutil.rmtree(profile_dir, ignore_errors=True)
    print(f"[T{thread_id}] cleaned up")


# -------------------------------------------------------------------- #
# direct‐download worker (unchanged)
# -------------------------------------------------------------------- #
def worker(q, bar, stats, rescue_enabled):
    while True:
        item = q.get()
        if item is None:
            break
        doi, year, urls = item
        sha = sha1(doi)
        tgt = OUT_DIR/str(year)/f"{sha}.pdf"
        ok = tgt.exists()
        if not ok:
            for u in urls:
                host = urlparse.urlparse(u).netloc
                if host in HARDBLOCK: continue
                if try_direct(u, tgt):
                    ok = True
                    break
        if ok: stats["ok"] += 1
        elif rescue_enabled: stats["miss"] += 1
        bar.update(1)
        q.task_done()



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
    ap.add_argument("--only-selenium", action="store_true",  help="Skip direct-download phase and run only the Selenium rescue pass")
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

                if args.max and len(jobs) >= args.max:
                    break  # ✅ stop loading when --max reached

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

    # Test Chrome installation before proceeding
    try:
        chrome_major, chrome_binary = get_chrome_major()
        print(f"✅ Chrome binary found: {chrome_binary} (version {chrome_major})")
    except Exception as e:
        print(f"❌ Chrome detection failed: {e}")
        if args.rescue:
            print("   Selenium rescue pass will be skipped.")
            args.rescue = False


    if args.only_selenium:
        args.rescue = True
        missing = jobs[:]
        print(f"⏩ Skipping direct download; will run Selenium on {len(missing)} jobs")
        
    else:
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
        
        # Initialize ChromeDriver with better error handling
        max_retries = 3
        drv_path = None
        
        for attempt in range(max_retries):
            try:
                # Try to get matching version first, fall back to latest
                try:
                    drv_path = ChromeDriverManager(driver_version=chrome_major).install()
                    print(f"✅ ChromeDriver v{chrome_major} installed → {drv_path}")
                except ValueError as e:
                    if "No such driver version" in str(e):
                        print(f"⚠️  ChromeDriver v{chrome_major} not available, using latest...")
                        drv_path = ChromeDriverManager().install()
                        print(f"✅ ChromeDriver (latest) installed → {drv_path}")
                    else:
                        raise
                break
                
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"⚠️  ChromeDriver install attempt {attempt+1} failed → {e}; retrying in 3s...")
                    time.sleep(3)
                else:
                    print(f"❌ ChromeDriver installation failed after {max_retries} attempts → {e}")
                    print("   Try: pip install --upgrade webdriver-manager")
                    return

        # Test ChromeDriver
        try:
            test_service = Service(executable_path=drv_path)
            test_service.start()
            test_service.stop()
            print(f"✅ ChromeDriver test successful")
        except Exception as e:
            print(f"❌ ChromeDriver test failed: {e}")
            return
        
        poll = 15 if args.slow else 10

        # Set up parallel Selenium processing
        work_queue = queue.Queue()
        result_queue = queue.Queue()
        
        # Add all missing jobs to work queue
        for job in missing:
            work_queue.put(job)
        
        # Start worker threads with better error handling
        selenium_threads = []
        active_threads = 0
        
        for thread_id in range(args.selenium_threads):
            try:
                t = threading.Thread(
                    target=selenium_worker, 
                    args=(work_queue, result_queue, drv_path, thread_id, args, poll),
                    daemon=True
                )
                t.start()
                selenium_threads.append(t)
                active_threads += 1
                print(f"Started Selenium thread {thread_id}")
            except Exception as e:
                print(f"Failed to start Selenium thread {thread_id}: {e}")
        
        if active_threads == 0:
            print("❌ No Selenium threads could be started")
            return
        
        print(f"✅ {active_threads} Selenium threads running")
        
        selenium_pdfs = selenium_fails = 0
        success_messages = []
        failure_messages = []
        
        # Process results with progress bar
        with tqdm(total=len(missing), unit="pdf", desc="SELENIUM") as pbar:
            processed = 0
            timeout_count = 0
            max_timeouts = 30  # Allow some timeouts before giving up
            
            while processed < len(missing):
                try:
                    result = result_queue.get(timeout=120)  # 2 minute timeout
                    processed += 1
                    timeout_count = 0  # Reset timeout counter on successful result
                    
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
                    timeout_count += 1
                    print(f"⚠️  No results in 60s ({timeout_count}/{max_timeouts}) — {processed}/{len(missing)} done")
                    if timeout_count >= max_timeouts:
                        print("❌  Too many back‐to‐back waits, stopping early")
                        break
                    # Check if any threads are stil alive
                    alive_threads = sum(1 for t in selenium_threads if t.is_alive())
                    if alive_threads == 0:
                        print(f"⚠️  No Selenium threads are running, stopping")
                        break
        
        # Signal threads to stop and wait for them
        print(f"🛑 Signaling {active_threads} threads to stop...")
        for _ in range(active_threads):
            work_queue.put(None)
        
        # Wait for threads to finish with timeout
        for i, t in enumerate(selenium_threads):
            t.join(timeout=45)
            if t.is_alive():
                print(f"⚠️  Thread {i} did not stop gracefully")
        
        # Display results after progress bar is complete
        print(f"\n📋 Selenium Results Summary:")
        if success_messages:
            print(f"\n✅ Successfully downloaded ({selenium_pdfs}):")
            for msg in success_messages[:10]:  # Limit output
                print(f"  {msg}")
            if len(success_messages) > 10:
                print(f"  ... and {len(success_messages) - 10} more")
        
        if failure_messages:
            print(f"\n❌ Failed downloads ({selenium_fails}):")
            for msg in failure_messages[:10]:  # Limit output
                print(f"  {msg}")
            if len(failure_messages) > 10:
                print(f"  ... and {len(failure_messages) - 10} more")
    else:
        selenium_pdfs = selenium_fails = 0

    # Force cleanup
    import gc
    gc.collect()

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

    """
    Linux-specific run commands:
    
    # First run (headless for stability):
    python pdf_scraper_fixed.py --email your@email.com --max 1000 --threads 12 --selenium-threads 6 --jobs-file jobs.jsonl --verbose --headless
    
    # Second run (GUI mode for debugging):
    python pdf_scraper_fixed.py --email your@email.com --max 1000 --threads 8 --selenium-threads 4 --jobs-file jobs.jsonl --verbose
    
    # Slow/thorough run:
    python pdf_scraper_fixed.py --email your@email.com --max 1000 --threads 6 --selenium-threads 3 --jobs-file jobs.jsonl --verbose --slow
    """