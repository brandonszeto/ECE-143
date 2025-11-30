import csv
import time
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.planecrashinfo.com/"
DATABASE_URL = urljoin(BASE_URL, "database.htm")
OUTPUT_CSV = "planecrashinfo_accidents.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; PlaneCrashInfoScraper/1.0; +https://example.com)"
}


def get_soup(url):
    """Fetch a URL and return a BeautifulSoup object."""
    resp = requests.get(url, headers=HEADERS, timeout=20)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def get_year_links():
    """
    From the main database page, extract links to each year (or year-range) page.
    Returns a list of absolute URLs.
    """
    soup = get_soup(DATABASE_URL)

    year_links = []
    # Assumption: years are in <a> tags on the database page.
    # We filter to .htm links that are not the main index itself.
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Basic heuristic: year pages are usually something like '1950/1950.htm' or '1960/1960.htm'
        if href.lower().endswith(".htm") and "database" not in href.lower():
            absolute_url = urljoin(BASE_URL, href)
            year_links.append(absolute_url)

    year_links = sorted(set(year_links))
    return year_links


def get_accident_links_for_year(year_url):
    """
    From a year page, extract links to each accident detail page (typically the date column).
    Returns a list of absolute URLs.
    """
    soup = get_soup(year_url)
    accident_links = []

    # Usually, accidents are displayed in a table, first column date linking to detail
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        # Accident pages are typically something like '1950/19500117.htm'
        if href.lower().endswith(".htm") and "database" not in href.lower():
            absolute_url = urljoin(year_url, href)
            accident_links.append(absolute_url)

    accident_links = sorted(set(accident_links))
    return accident_links


def parse_accident_detail(accident_url):
    """
    Parse an individual accident detail page into a dict of fields.
    The site typically uses a table with label/value rows (e.g., Date, Time, Location, etc.).
    """
    soup = get_soup(accident_url)

    record = {
        "detail_url": accident_url,
    }

    # Heuristic: find the main table that contains label/value rows
    table = soup.find("table")
    if not table:
        # Fallback: just dump full text
        record["raw_text"] = soup.get_text(separator=" ", strip=True)
        return record

    # Many pages use <tr><td>Label:</td><td>Value</td></tr>
    for row in table.find_all("tr"):
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        label = cells[0].get_text(" ", strip=True).rstrip(":")
        value = cells[1].get_text(" ", strip=True)

        # Normalize label to something CSV-friendly
        norm_label = (
            label.lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
            .strip("_")
        )
        if norm_label:
            record[norm_label] = value

    # Also capture summary paragraph if present (some pages have a dedicated summary area)
    summary = []
    for p in soup.find_all("p"):
        text = p.get_text(" ", strip=True)
        if text:
            summary.append(text)
    if summary and "summary" not in record:
        record["summary"] = " ".join(summary)

    return record


def main():
    print(f"Fetching year links from {DATABASE_URL} ...")
    year_links = get_year_links()
    print(f"Found {len(year_links)} year pages.")

    all_records = []
    visited_detail_urls = set()

    for i, year_url in enumerate(year_links, start=1):
        print(f"[{i}/{len(year_links)}] Processing year page: {year_url}")
        try:
            accident_links = get_accident_links_for_year(year_url)
        except Exception as e:
            print(f"  Error fetching accident links from {year_url}: {e}")
            continue

        print(f"  Found {len(accident_links)} accident pages for this year.")

        for j, acc_url in enumerate(accident_links, start=1):
            if acc_url in visited_detail_urls:
                continue

            print(f"    [{j}/{len(accident_links)}] Accident: {acc_url}")
            try:
                record = parse_accident_detail(acc_url)
                record["year_page_url"] = year_url
                all_records.append(record)
                visited_detail_urls.add(acc_url)
            except Exception as e:
                print(f"      Error parsing {acc_url}: {e}")

            # Be polite; don't hammer the server
            time.sleep(0.5)

        # Slight extra delay between year pages
        time.sleep(1.0)

    # Determine all keys for CSV header
    all_keys = set()
    for rec in all_records:
        all_keys.update(rec.keys())
    fieldnames = sorted(all_keys)

    print(f"Writing {len(all_records)} records to {OUTPUT_CSV} ...")
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in all_records:
            writer.writerow(rec)

    print("Done.")


if __name__ == "__main__":
    main()
