"""
scripts/ingest_public_domain.py

More robust ingestion of public-domain materia medica pages.
Targets:
 - Kent (homeoint.org)
 - Boericke (homeoint.org)
 - Clarke (archive.org)
 - Allen (archive.org)

This script:
 - fetches index pages and follows links
 - supports simple archive.org 'details' pages and 'stream' text
 - uses a browser-like User-Agent
 - prints detailed logs and writes data/remedies_full.json

Note: archive.org sometimes serves page images or PDFs; this script attempts HTML/text extraction when available.
If large-scale scraping is blocked on GitHub runners for any site, consider running locally or in Colab.
"""
import os
import time
import json
import re
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/121.0 Safari/537.36"
}
OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "remedies_full.json")
FALLBACK_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "remedies_master.json")

# Index pages / sources
SOURCES = [
    # Kent index (homeoint)
    "https://homeoint.org/books/kent/index.htm",
    # Boericke (Pocket Manual)
    "https://homeoint.org/books/boericmm/index.htm",
    # Clarke - archive.org "Dictionary of Practical Materia Medica" item (details page)
    "https://archive.org/details/ClarkeDictionaryOfPracticalMateriaMedica",
    # Allen - Encyclopedia (example public domain item; may vary by host)
    "https://archive.org/details/encyclopediapure00alle",  # if item exists; otherwise archive pages collected
]

# Helper: fetch with retries
def fetch(url, tries=3, timeout=20):
    for i in range(tries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print(f"[fetch] error fetching {url} ({i+1}/{tries}): {e}")
            time.sleep(1.5)
    return None

# Parse homeoint-style index pages (Kent/Boericke)
def parse_homeoint_index(html, base_url):
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a['href'].strip()
        if href.lower().endswith((".htm", ".html")):
            full = href if href.startswith("http") else urljoin(base_url, href)
            links.append(full)
    # dedupe
    return list(dict.fromkeys(links))

# Try to extract text from an index/entry page
def extract_from_html(html, url):
    soup = BeautifulSoup(html, "html.parser")
    # remove script/style
    for s in soup(["script", "style", "noscript"]):
        s.decompose()
    # collect main text from p, li, div, pre, h1-h3
    parts = []
    for tag in soup.find_all(["h1","h2","h3","p","div","li","pre"]):
        text = tag.get_text(separator=" ", strip=True)
        if text and len(text) > 40:
            parts.append(text)
    # fallback: whole body text
    if not parts:
        body = soup.body
        if body:
            text = body.get_text(separator=" ", strip=True)
            if text and len(text)>100:
                parts = [text]
    content = "\n\n".join(parts[:400])
    if len(content.strip()) < 100:
        return None
    # title
    title = soup.title.string.strip() if soup.title and soup.title.string else url
    name = re.sub(r'[^A-Za-z\s]', '', title).strip()
    if not name:
        # try first line of content as name
        first = content.splitlines()[0].strip() if content else url
        name = re.sub(r'[^A-Za-z\s]', '', first)[:60]
    return {"id": url, "name": name[:60], "full_text": content, "source": url}

# Archive.org helpers: from a details page, collect candidate text links
def parse_archive_details(html, base):
    soup = BeautifulSoup(html, "html.parser")
    links = set()
    # find anchors to '/stream/' or '/details/' pages, and to files
    for a in soup.find_all("a", href=True):
        href = a['href']
        if "/stream/" in href or "/details/" in href or href.endswith(".txt") or href.endswith(".html"):
            full = href if href.startswith("http") else urljoin(base, href)
            links.add(full)
    # also try to find '/download/' or '/files/' links
    for a in soup.find_all("a", href=True):
        href = a['href']
        if "/download/" in href or "/files/" in href:
            full = href if href.startswith("http") else urljoin(base, href)
            links.add(full)
    return list(links)

# Try to extract from an archive.org stream or text URL
def extract_archive_text(url):
    html = fetch(url)
    if not html:
        return None
    # if this is a 'stream' page, it might contain <pre> or page images - try best-effort
    res = extract_from_html(html, url)
    if res:
        return res
    # sometimes archive pages embed a 'text' link inside; attempt to find <iframe> or <pre>
    soup = BeautifulSoup(html, "html.parser")
    # find tag with id="ocr_text" or similar
    ocr = soup.find(id=re.compile(r'ocr|page|viewer', re.I))
    if ocr:
        text = ocr.get_text(separator=" ", strip=True)
        if len(text) > 200:
            title = soup.title.string if soup.title else url
            name = re.sub(r'[^A-Za-z\s]', '', title).strip()
            return {"id": url, "name": name[:60], "full_text": text, "source": url}
    return None

# Main driver
def main():
    print("Starting ingestion...")
    all_links = []
    for src in SOURCES:
        print("Processing source:", src)
        html = fetch(src)
        if not html:
            print("  -> failed to fetch index:", src)
            continue

        if "homeoint.org" in src:
            base = "/".join(src.split("/")[:-1])
            links = parse_homeoint_index(html, base)
            print(f"  -> found {len(links)} homeoint links")
            all_links.extend(links)
        elif "archive.org" in src:
            # parse archive details -> candidate links
            base = src
            cand = parse_archive_details(html, base)
            print(f"  -> found {len(cand)} archive candidate links")
            all_links.extend(cand)
        else:
            # fallback: try to extract as an entry page itself
            all_links.append(src)

    # dedupe & filter
    unique_links = []
    for l in all_links:
        if l not in unique_links:
            # normalize archive URLs: prefer 'https://archive.org/stream/...' or '/details/...'
            unique_links.append(l)
    print("Total unique candidate links:", len(unique_links))

    remedies = []
    # scan links and try to extract remedies
    for link in unique_links:
        # small politeness
        time.sleep(0.12)
        try:
            # special handling for archive.org details -> try to fetch nested streams
            if "archive.org/details/" in link and "stream" not in link:
                # fetch details page and parse stream links
                html = fetch(link)
                if not html:
                    continue
                nested = parse_archive_details(html, link)
                # try nested streams first
                for n in nested[:8]:
                    try:
                        r = extract_archive_text(n)
                        if r:
                            remedies.append(r)
                            print("  + archive extracted from nested:", n)
                            break
                    except Exception:
                        continue
                continue

            # general html fetch + extract
            html = fetch(link)
            if not html:
                continue

            r = extract_from_html(html, link)
            if r:
                remedies.append(r)
                print("  + extracted:", link)
                continue

            # if no plain html, try archive extraction
            if "archive.org" in link:
                ra = extract_archive_text(link)
                if ra:
                    remedies.append(ra)
                    print("  + archive special extracted:", link)
                    continue

        except Exception as e:
            print("  ! error extracting", link, e)
            continue

    # dedupe remedies by name
    print("Raw remedies extracted:", len(remedies))
    merged = {}
    for r in remedies:
        key = r.get("name","").strip().lower()
        if not key:
            key = r.get("id")
        if key in merged:
            merged[key]["full_text"] += "\n\n" + (r.get("full_text",""))
            merged[key]["source"] += ";" + r.get("source","")
        else:
            merged[key] = r

    final = list(merged.values())
    print("Merged remedies count:", len(final))

    # Fallback to master if nothing found
    if len(final) == 0:
        print("No remedies extracted; using fallback remedies_master.json")
        if os.path.exists(FALLBACK_FILE):
            with open(FALLBACK_FILE, "r", encoding="utf-8") as f:
                final = json.load(f)

    # write output
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2, ensure_ascii=False)
    print("Wrote", OUT_FILE, "with", len(final), "entries")

if __name__ == "__main__":
    main()
