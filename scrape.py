#!/usr/bin/env python3
"""
quick_ferroelectric_dump.py
Download a local corpus of OA PDFs that contain the word “ferroelectric”.
Uses OpenAlex for metadata and Unpaywall for OA links.
Resumable via SQLite (state.db).

Example
-------
python quick_ferroelectric_dump.py \
    --email you@uni.edu \
    --out ./ferro_papers \
    --max 5000 \
    --upw-threads 6 \
    --threads 12
"""

from __future__ import annotations

# ── stdlib
import argparse, contextlib, hashlib, pathlib, random, re, socket, sqlite3, sys, textwrap, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.parse as urlparse

# ── third‑party
import dns.exception, dns.resolver
import requests, ssl
from urllib3 import poolmanager
from urllib3.util import connection as urllib3_conn
from requests.adapters import HTTPAdapter

# ────────────────────────────────────────────────────────────────────────────
# CONFIG --------------------------------------------------------------------
# ────────────────────────────────────────────────────────────────────────────
OPENALEX  = "https://api.openalex.org/works"
UNPAYWALL = "https://api.unpaywall.org/v2/{}"

# publishers that almost always return 403 / pay‑wall
BLOCKED_DOMAINS = {
    'pubs.acs.org', 'pubs.aip.org', 'advances.sciencemag.org',
    'www.science.org', 'www.cell.com', 'www.sciencedirect.com',
    'onlinelibrary.wiley.com', 'iopscience.iop.org', 'www.worldscientific.com',
    'journals.aps.org', 'aip.scitation.org', 'www.pnas.org'
}

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_6) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.4 Safari/605.1.15",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0",
]

# ────────────────────────────────────────────────────────────────────────────
# DNS cache + DoH fallback
# ────────────────────────────────────────────────────────────────────────────
_host_cache: dict[str, str] = {}

def resolve_ip(host: str) -> str:
    if host in _host_cache:
        return _host_cache[host]
    try:
        ip = socket.gethostbyname(host)
    except socket.gaierror:
        try:
            r = dns.resolver.Resolver(configure=False)
            r.nameservers = ["1.1.1.1", "8.8.8.8"]
            ip = r.resolve(host, "A", lifetime=8)[0].to_text()
        except dns.exception.DNSException as e:
            raise socket.gaierror(f"DNS fallback failed for {host}: {e}") from None
    _host_cache[host] = ip
    return ip

# ────────────────────────────────────────────────────────────────────────────
# HTTPS adapter — dial IP, keep hostname for TLS SNI
# ────────────────────────────────────────────────────────────────────────────
class HostHeaderSSLAdapter(HTTPAdapter):
    def init_poolmanager(self, *a, **kw):
        kw["ssl_context"] = ssl.create_default_context()
        self.poolmanager = poolmanager.PoolManager(*a, **kw)

    def get_connection(self, url, proxies=None):
        host = urlparse.urlparse(url).hostname
        if host:
            ip = resolve_ip(host)
            orig = urllib3_conn.create_connection

            def patched(addr, *a, **kw):
                if addr[0] == host:
                    return orig((ip, addr[1]), *a, **kw, server_hostname=host)
                return orig(addr, *a, **kw)

            urllib3_conn.create_connection = patched
            try:
                return super().get_connection(url, proxies)
            finally:
                urllib3_conn.create_connection = orig
        return super().get_connection(url, proxies)

# ────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ────────────────────────────────────────────────────────────────────────────
def sha1(t: str) -> str: return hashlib.sha1(t.encode()).hexdigest().upper()
def pdf_name(doi: str) -> str: return f"{sha1(doi)}.pdf"

def ensure_tables(cur):
    cur.execute("CREATE TABLE IF NOT EXISTS done   (doi TEXT PRIMARY KEY, path TEXT, ts INT)")
    cur.execute("CREATE TABLE IF NOT EXISTS failed (doi TEXT PRIMARY KEY, error TEXT, ts INT)")
    cur.execute("CREATE TABLE IF NOT EXISTS skipped(doi TEXT PRIMARY KEY, reason TEXT, ts INT)")

def mark_row(cur, table: str, doi: str, message: str):
    cur.execute(f"INSERT OR REPLACE INTO {table} VALUES (?,?,?)",
                (doi, message, int(time.time())))

# ────────────────────────────────────────────────────────────────────────────
# ArXiv utilities
# ────────────────────────────────────────────────────────────────────────────
_ARXIV_PATTERNS = [
    (r'arxiv\.org/pdf/([0-9]{4}\.[0-9]{4,5})(?:v\d+)?$', r'arxiv.org/pdf/\1.pdf'),
    (r'arxiv\.org/pdf/([a-z-]+/[0-9]{7})(?:v\d+)?$',    r'arxiv.org/pdf/\1.pdf'),
]

def fix_arxiv_url(url: str) -> str:
    if "arxiv.org" not in url.lower(): return url
    for pat, repl in _ARXIV_PATTERNS:
        if re.search(pat, url, re.I):
            url = re.sub(pat, repl, url, flags=re.I); break
    return url.replace("http://", "https://")

# ────────────────────────────────────────────────────────────────────────────
# OpenAlex iterator
# ────────────────────────────────────────────────────────────────────────────
def openalex_iter(sess, cursor="*", per_page=200):
    while True:
        params = {"search": "ferroelectric",
                  "filter": "open_access.is_oa:true",
                  "per-page": per_page,
                  "cursor": cursor}
        r = sess.get(OPENALEX, params=params, timeout=40); r.raise_for_status()
        js = r.json(); yield from js["results"]
        cursor = js["meta"]["next_cursor"];  # None when finished
        if not cursor: break

# ────────────────────────────────────────────────────────────────────────────
# Unpaywall helpers
# ────────────────────────────────────────────────────────────────────────────
def best_pdf_url(upw: dict | None):
    if not upw: return None
    locs = [upw.get("best_oa_location"), *(upw.get("oa_locations") or [])]
    locs = [l for l in locs if l and l.get("url_for_pdf")]
    if not locs: return None
    locs.sort(key=lambda l: 0 if l["host_type"] == "repository" else 1)
    return locs[0]["url_for_pdf"]

def unpaywall_json(sess, doi: str, email: str):
    try:
        r = sess.get(UNPAYWALL.format(doi), params={"email": email}, timeout=15)
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

# ────────────────────────────────────────────────────────────────────────────
# Per‑download session (random UA)
# ────────────────────────────────────────────────────────────────────────────
def session_for_download(base: requests.Session):
    s = requests.Session()
    s.verify  = base.verify
    s.proxies = base.proxies
    s.headers = {"User-Agent": random.choice(USER_AGENTS),
                 "Accept": "application/pdf,*/*;q=0.9"}
    return s

# ────────────────────────────────────────────────────────────────────────────
# PDF downloader
# ────────────────────────────────────────────────────────────────────────────
def download_pdf(url: str, target: pathlib.Path, base: requests.Session,
                 doi: str, tries=3):
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".part")

    for a in range(tries):
        s = session_for_download(base)
        try:
            if a: time.sleep(random.uniform(2, 5) * a)
            with s.get(url, stream=True, timeout=60, allow_redirects=True) as r:
                if r.status_code in (403, 404):
                    raise requests.exceptions.HTTPError(str(r.status_code))
                r.raise_for_status()

                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(16384):
                        if chunk:
                            f.write(chunk)
                with open(tmp, "rb") as f:
                    if not f.read(4) == b"%PDF":
                        raise ValueError("Not PDF")
                tmp.rename(target)
                return
        except Exception as e:
            last = e
        finally:
            s.close()
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    raise last

# ────────────────────────────────────────────────────────────────────────────
# Gather jobs
# ────────────────────────────────────────────────────────────────────────────
def gather_jobs(pairs, email, sess, threads=4):
    jobs, skips = [], []

    def fetch(pair):
        doi, yr = pair
        js = unpaywall_json(sess, doi, email)
        url = best_pdf_url(js)
        if not url:                  return None, "no_url"
        url = fix_arxiv_url(url)
        dom = urlparse.urlparse(url).netloc.lower()
        if any(b in dom for b in BLOCKED_DOMAINS):
            return None, "blocked_publisher"
        return (doi, yr, url), None

    with ThreadPoolExecutor(max_workers=threads) as pool:
        futs = {pool.submit(fetch, p): p for p in pairs}
        for i, fut in enumerate(as_completed(futs), 1):
            result, reason = fut.result()
            if result: jobs.append(result)
            else:      skips.append((futs[fut][0], reason))
            if i % 100 == 0:
                print(f"  …checked {i}/{len(pairs)} DOIs "
                      f"({len(jobs)} downloadable, {len(skips)} skipped)")
    return jobs, skips

# ────────────────────────────────────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Dump OA ferroelectric PDFs.")
    ap.add_argument("--email", required=True)
    ap.add_argument("--out", default="ferro_papers")
    ap.add_argument("--max", type=int, default=5000)
    ap.add_argument("--upw-threads", type=int, default=6)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()

    out_dir = pathlib.Path(args.out).expanduser(); out_dir.mkdir(exist_ok=True)

    base = requests.Session()
    base.headers["User-Agent"] = f"oa-dump/1.0 ({args.email})"
    base.mount("https://", HostHeaderSSLAdapter())

    db = sqlite3.connect("state.db"); cur = db.cursor(); ensure_tables(cur)

    # 1. collect DOIs --------------------------------------------------------
    unseen=[]
    print("🔎  Paging OpenAlex …")
    for w in openalex_iter(base):
        if len(unseen) >= args.max: break
        doi = w.get("doi"); yr = w.get("publication_year") or "na"
        if doi:
            cur.execute("SELECT 1 FROM done WHERE doi=?", (doi,))
            if not cur.fetchone():
                unseen.append((doi, yr))
        if unseen and len(unseen) % 500 == 0:
            print(f"  collected {len(unseen)} DOIs …")
    print(f"✨  unseen DOIs: {len(unseen)}")
    if not unseen: return

    # 2. Unpaywall
    print("🌐  Querying Unpaywall …")
    jobs, skips = gather_jobs(unseen, args.email, base, args.upw_threads)

    for doi, rsn in skips:
        mark_row(cur, "skipped", doi, rsn)
    db.commit()

    print(f"📄  {len(jobs)} downloadable PDFs")
    print(f"⏭️   {len(skips)} skipped (blocked/no-url)")

    if not jobs:
        print("Nothing to download.")
        return

    # 3. download
    success = fail = 0
    print(f"⬇️   Downloading with {args.threads} threads …")
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        futs = {pool.submit(download_pdf, url, out_dir/str(yr)/pdf_name(doi),
                            base, doi): doi for doi, yr, url in jobs}
        for fut in as_completed(futs):
            doi = futs[fut]
            try:
                fut.result()
                mark_row(cur, "done", doi, "ok")
                success += 1
                print(f"✅ {success}/{len(jobs)} {doi}")
            except Exception as e:
                mark_row(cur, "failed", doi, str(e))
                fail += 1
                print(f"❌ {fail}/{len(jobs)} {doi}: {e}")
            db.commit()

    print("\n🎉  Finished.")
    print(f"    ✅ {success} succeeded")
    print(f"    ❌ {fail} failed (see 'failed' table)")
    print(f"    ⏭️  {len(skips)} skipped (see 'skipped' table)")
    print(f"    📂 Saved in {out_dir}")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted – progress saved.")
        sys.exit(1)
