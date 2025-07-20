#!/usr/bin/env python3
"""
oa_ferroelectric_selenium_pool.py
──────────────────────────────────────────────────────────────────────────────
Headless‑Chrome scraper that reuses one browser per worker thread, avoiding
the “Could not determine browser executable” error after many launches.
"""

from __future__ import annotations
import argparse, hashlib, pathlib, queue, threading, time
from typing import List, Tuple
import urllib.parse as urlparse

import dns.exception, dns.resolver, requests, undetected_chromedriver as uc
from selenium.common.exceptions import TimeoutException
from tqdm import tqdm

# ── constants
OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"
OUT_DIR   = pathlib.Path("./ferroelectric_pdfs").expanduser()
HARDBLOCK = {"pubs.acs.org", "onlinelibrary.wiley.com"}

# ── tiny DNS cache
_dns: dict[str,str] = {}
def resolve(host:str)->str:
    if host in _dns: return _dns[host]
    try: ip=dns.resolver.resolve(host,"A",lifetime=5)[0].to_text()
    except dns.exception.DNSException: ip=host
    _dns[host]=ip; return ip

# ── helpers
def pdf_name(doi:str)->str:
    return f"{hashlib.sha1(doi.encode()).hexdigest().upper()}.pdf"

def candidate_urls(js:dict)->List[str]:
    locs=[js.get("best_oa_location"),*(js.get("oa_locations") or [])]
    seen,urls=set(),[]
    for l in locs or []:
        if not l: continue
        u=l.get("url_for_pdf") or l.get("url")
        if u and u not in seen: seen.add(u); urls.append(u)
    return urls

# ── one Chrome per thread ---------------------------------------------------
def make_driver(download_dir: pathlib.Path):
    import undetected_chromedriver as uc
    from pathlib import Path

    # ARM‑native Chrome + driver installed via Homebrew
    CHROME  = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    DRIVER  = Path("/opt/homebrew/bin/chromedriver")     # <- Brew’s 138 arm64

    if not CHROME.exists():
        raise FileNotFoundError("Chrome not found at default path.")
    if not DRIVER.exists():
        raise FileNotFoundError("chromedriver not found – brew install chromedriver")

    opts = uc.ChromeOptions()
    opts.binary_location = str(CHROME)
    opts.add_argument("--headless=new")
    opts.add_argument("--disable-gpu")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")

    prefs = {
        "download.default_directory": str(download_dir),
        "download.prompt_for_download": False,
        "plugins.always_open_pdf_externally": True,
    }
    opts.add_experimental_option("prefs", prefs)

    # Key lines ↓ : use *your* driver and skip any UC patching / copying
    driver = uc.Chrome(
        options=opts,
        browser_executable_path=str(CHROME),
        driver_executable_path=str(DRIVER),
        patch_drivers=False # <— NO patch → UC won't clone into cache
    )
    driver.set_page_load_timeout(60)
    return driver



# ── worker thread -----------------------------------------------------------
def worker(q:queue.Queue, bar:tqdm, ok_fail:dict):
    tmp_base=OUT_DIR/"_tmp"/threading.current_thread().name
    tmp_base.mkdir(parents=True,exist_ok=True)
    driver=make_driver(tmp_base)

    while True:
        item=q.get()
        if item is None: break
        doi,year,urls=item
        tgt=OUT_DIR/str(year)/pdf_name(doi)
        success=False; reason=""
        try:
            if tgt.exists():
                success=True; reason="exists"
            else:
                for url in urls:
                    dom=urlparse.urlparse(url).netloc.lower()
                    if any(b in dom for b in HARDBLOCK):
                        reason="blocked"; continue
                    try: driver.get(url)
                    except TimeoutException:
                        reason="timeout"; continue
                    time.sleep(4)
                    pdfs=list(tmp_base.glob("*.pdf"))
                    if pdfs:
                        pdfs[0].rename(tgt); success=True; reason="ok"; break
                    reason="no_pdf"
        finally:
            for f in tmp_base.glob("*.pdf"): f.unlink(missing_ok=True)
            bar.update(1)
            if success: ok_fail["ok"]+=1; print(f"✅ {ok_fail['ok']} {doi}")
            else:       ok_fail["fail"]+=1; print(f"❌ {ok_fail['fail']} {doi} ({reason})")
            q.task_done()
    driver.quit()

# ── OpenAlex iterator -------------------------------------------------------
def openalex_iter(sess, limit:int):
    cursor="*"; scanned=0
    while scanned<limit:
        r=sess.get(OPENALEX,params={
            "search":"ferroelectric","filter":"open_access.is_oa:true",
            "per-page":200,"cursor":cursor},timeout=40); r.raise_for_status()
        js=r.json()
        for w in js["results"]:
            if scanned>=limit: break
            scanned+=1
            if scanned%100==0: print(f"  OpenAlex scanned {scanned}")
            yield w
        cursor=js["meta"]["next_cursor"]
        if not cursor: break

# ── main --------------------------------------------------------------------
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--email",required=True)
    ap.add_argument("--max",type=int,default=1000)
    ap.add_argument("--threads",type=int,default=4)
    args=ap.parse_args()

    OUT_DIR.mkdir(parents=True,exist_ok=True)
    sess=requests.Session()
    sess.headers["User-Agent"]="oa-selenium/0.5"

    jobs=[]
    print("🔎  Collecting DOIs + OA URLs …")
    for w in openalex_iter(sess,args.max):
        doi=w.get("doi"); yr=w.get("publication_year") or "na"
        if not doi: continue
        upw=sess.get(UNPAYWALL.format(doi),params={"email":args.email},timeout=15)
        if upw.status_code!=200: continue
        urls=candidate_urls(upw.json())
        if urls: jobs.append((doi,yr,urls))
    print(f"•  {len(jobs)} works with candidate URLs")

    if not jobs: return
    q=queue.Queue()
    ok_fail={"ok":0,"fail":0}

    bar=tqdm(total=len(jobs),ncols=80,unit="pdf")
    threads=[]
    for _ in range(args.threads):
        t=threading.Thread(target=worker,args=(q,bar,ok_fail),daemon=True)
        t.start(); threads.append(t)
    for j in jobs: q.put(j)
    q.join()
    for _ in threads: q.put(None)
    for t in threads: t.join()
    bar.close()

    print("\n🎉  Done.")
    print(f"   ✅ {ok_fail['ok']} succeeded")
    print(f"   ❌ {ok_fail['fail']} failed")
    print("   📂 PDFs in", OUT_DIR.resolve())

if __name__=="__main__":
    main()
