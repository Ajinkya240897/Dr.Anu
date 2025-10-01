"""
scripts/ingest_public_domain.py
Fetch public-domain materia medica index pages and extract remedy pages.
Saves data/remedies_full.json
"""
import os, requests, json, re, time
from bs4 import BeautifulSoup

SOURCES = [
    "https://homeoint.org/books/kent/index.htm",
    "https://homeoint.org/books/boericmm/index.htm",
    # Add more public-domain indexes as needed
]

OUT_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "remedies_full.json")

def fetch(url, tries=3, timeout=20):
    for i in range(tries):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return r.text
        except Exception as e:
            print("fetch error", url, e)
            time.sleep(1)
    return None

def parse_index(index_html, base):
    soup = BeautifulSoup(index_html, "html.parser")
    links = []
    for a in soup.find_all("a", href=True):
        href = a['href']
        if href.endswith(".htm") or href.endswith(".html"):
            if href.startswith("http"):
                links.append(href)
            else:
                links.append(base.rstrip('/') + '/' + href.lstrip('/'))
    return links

def extract_remedy(html, url):
    soup = BeautifulSoup(html, "html.parser")
    title = soup.title.string if soup.title else url
    parts = []
    for tag in soup.find_all(["h1","h2","h3","p","div","li"]):
        text = tag.get_text(separator=' ', strip=True)
        if text and len(text) > 30:
            parts.append(text)
    text = "\n\n".join(parts[:400])
    name_frag = re.sub(r'[^A-Za-z\\s]', '', title).strip()
    name = " ".join(name_frag.split()[:4]) if name_frag else url
    if len(text.strip()) < 100:
        return None
    return {"id": url, "name": name, "full_text": text, "source": url}

def main():
    all_links = []
    for src in SOURCES:
        print("Fetching index", src)
        html = fetch(src)
        if not html:
            print("Index fetch failed:", src)
            continue
        base = "/".join(src.split('/')[:-1])
        links = parse_index(html, base)
        print(" -> links found:", len(links))
        all_links.extend(links)
    all_links = list(dict.fromkeys(all_links))
    print("Total unique links to consider:", len(all_links))
    remedies = []
    for link in all_links:
        html = fetch(link)
        if not html:
            continue
        rem = extract_remedy(html, link)
        if rem:
            remedies.append(rem)
        time.sleep(0.08)
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(remedies, f, indent=2, ensure_ascii=False)
    print("Wrote", OUT_FILE, "with", len(remedies), "remedies")

if __name__ == '__main__':
    main()
