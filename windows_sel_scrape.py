#!/usr/bin/env python3
# OA ferroelectric PDF harvester – fast direct pass + optional Selenium rescue
# v1.2  (Windows‑optimised, but works cross‑platform)

import argparse, hashlib, queue, threading, time, re, urllib.parse as urlparse
from pathlib import Path
import requests
from tqdm import tqdm

# ── constants ───────────────────────────────────────────────────────────────
OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"
OUT_DIR   = Path("ferro_papers")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/138.0.0.0 Safari/537.36")

HARDBLOCK = {"pubs.acs.org", "onlinelibrary.wiley.com",
             "www.science.org", "advances.sciencemag.org", "pubs.aip.org"}

# ── helpers ────────────────────────────────────────────────────────────────
def sha1(s: str) -> str: return hashlib.sha1(s.encode()).hexdigest().upper()
def pdf_name(doi: str) -> str: return f"{sha1(doi)}.pdf"

def candidate_urls(js: dict) -> list[str]:
    locs = [js.get("best_oa_location"), *(js.get("oa_locations") or [])]
    urls = [l["url_for_pdf"] or l["url"] for l in locs if l]
    return list(dict.fromkeys(urls))

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

# ── first‑pass worker (direct only) ─────────────────────────────────────────
def worker(q: queue.Queue, bar: tqdm, stats: dict):
    while True:
        item = q.get()
        if item is None:
            break
        doi, year, urls = item
        tgt = OUT_DIR / str(year) / pdf_name(doi)
        ok = tgt.exists()
        if not ok:
            for url in urls:
                host = urlparse.urlparse(url).netloc
                if host in HARDBLOCK:          # leave for Selenium pass
                    continue
                if try_direct(url, tgt):
                    ok = True; break
        stats["ok" if ok else "miss"] += 1
        print(f"{'✅' if ok else '➖'} {doi}")
        bar.update(1); q.task_done()

# ── Selenium utilities (only if rescue enabled) ─────────────────────────────
from webdriver_manager.chrome import ChromeDriverManager
import undetected_chromedriver as uc
def driver_major(path: str) -> int:
    """Return major version from any webdriver‑manager path (cross‑platform)."""
    p = Path(path).resolve()
    for parent in [p.parent] + list(p.parents):
        m = re.match(r"^(\d+)", parent.name)
        if m: return int(m.group(1))
    raise RuntimeError(f"Cannot parse version from {path!s}")

def make_driver(tmp: Path, driver_path: str, timeout=45) -> uc.Chrome:
    opts = uc.ChromeOptions()
    opts.add_argument("--headless=new"); opts.add_argument("--disable-gpu")
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

# ── main ────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--email", required=True)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--slow", action="store_true",
                    help="poll 20 s per Selenium page (default 6 s)")
    # new flag
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--rescue", dest="rescue", action="store_true",
                     help="enable Selenium second pass (default)")
    grp.add_argument("--no-rescue", dest="rescue", action="store_false",
                     help="skip Selenium pass (fastest)")
    ap.set_defaults(rescue=True)
    args = ap.parse_args()

    OUT_DIR.mkdir(exist_ok=True)
    already = len(list(OUT_DIR.glob("**/*.pdf")))     # <── NEW

    sess = requests.Session(); sess.headers["User-Agent"] = "oa-sel 1.2"

    # --- collect jobs -------------------------------------------------------
    jobs, scanned, cursor = [], 0, "*"
    print("🔎  Collecting DOIs + OA URLs …")
    while scanned < args.max:
        r = sess.get(OPENALEX, params={
            "search": "ferroelectric",
            "filter": "open_access.is_oa:true",
            "per-page": 200, "cursor": cursor}, timeout=40)
        js = r.json()
        for w in js["results"]:
            if scanned >= args.max: break
            scanned += 1
            if scanned % 100 == 0:
                print(f"  OpenAlex scanned {scanned}")
            doi = w.get("doi"); year = w.get("publication_year") or "na"
            if not doi: continue
            upw = sess.get(UNPAYWALL.format(doi),
                           params={"email": args.email}, timeout=15)
            if upw.status_code == 200:
                urls = candidate_urls(upw.json())
                if urls: jobs.append((doi, year, urls))
        cursor = js["meta"]["next_cursor"]
        if not cursor: break

    print(f"•  {len(jobs)} works with candidate URLs")
    if not jobs: return

    # --- fast first pass ----------------------------------------------------
    q = queue.Queue(); stats = {"ok": 0, "miss": 0}
    bar = tqdm(total=len(jobs), unit="pdf", ncols=80, desc="1st pass")
    for _ in range(min(args.threads, len(jobs))):
        threading.Thread(target=worker, args=(q, bar, stats),
                         daemon=True).start()
    for j in jobs: q.put(j)
    q.join(); bar.close()

    # --- optional Selenium rescue ------------------------------------------
    missing = [j for j in jobs
               if not (OUT_DIR / str(j[1]) / pdf_name(j[0])).exists()]

    if args.rescue and missing:
        print(f"\n🕷  Selenium second pass on {len(missing)} remaining URLs …")
        drv_path = ChromeDriverManager().install()
        drv = make_driver(OUT_DIR / "_tmp", drv_path,
                          timeout=60 if args.slow else 45)
        poll = 40 if args.slow else 12          # 20 s vs 6 s

        for doi, year, urls in tqdm(missing, unit="pdf", ncols=80):
            tgt = OUT_DIR / str(year) / pdf_name(doi); ok = False
            for url in urls:
                try: drv.get(url)
                except Exception: continue
                for _ in range(poll):
                    tmp_pdfs = list((OUT_DIR / "_tmp").glob("*.pdf"))
                    if tmp_pdfs:
                        tmp_pdfs[0].rename(tgt); ok = True; break
                    time.sleep(0.5)
                if ok: break
            print(f"{'✅' if ok else '❌'} selenium {doi}")
        drv.quit()

    # --- summary ------------------------------------------------------------
    final_total = len(list(OUT_DIR.glob("**/*.pdf")))
    new_ok      = final_total - already            # downloaded during this run
    new_fail    = len(jobs) - new_ok

    print(f"\n🎉  Session: {new_ok} ok / {new_fail} fail")
    print(f"📁  PDFs saved to: {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
