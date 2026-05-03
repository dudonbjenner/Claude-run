#!/usr/bin/env python3
"""
Lead Scraper
============
Scrapes contact information (emails, phone numbers, social links,
company names) from a list of websites and exports results to CSV.

Usage:
    python lead_scraper.py --urls urls.txt --output leads.csv
    python lead_scraper.py --url https://example.com --output leads.csv
"""

import re
import csv
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser
from dataclasses import dataclass, fields, asdict
from typing import Optional

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Run: pip install requests beautifulsoup4")
    raise

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("lead_scraper")

# ── Data model ────────────────────────────────────────────────────────────────
@dataclass
class Lead:
    url: str
    company_name: str = ""
    emails: str = ""          # semicolon-separated
    phones: str = ""          # semicolon-separated
    linkedin: str = ""
    twitter: str = ""
    facebook: str = ""
    instagram: str = ""
    address: str = ""
    description: str = ""
    status: str = "ok"        # ok | error | timeout

# ── Regex patterns ────────────────────────────────────────────────────────────
EMAIL_RE    = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
PHONE_RE    = re.compile(
    r"(?:\+?1[\s\-.]?)?"           # optional country code
    r"(?:\(?\d{3}\)?[\s\-.]?)"     # area code
    r"\d{3}[\s\-.]?\d{4}"
)
LINKEDIN_RE  = re.compile(r"https?://(?:www\.)?linkedin\.com/(?:company|in)/[^\s\"'<>]+")
TWITTER_RE   = re.compile(r"https?://(?:www\.)?(?:twitter|x)\.com/[^\s\"'<>]+")
FACEBOOK_RE  = re.compile(r"https?://(?:www\.)?facebook\.com/[^\s\"'<>]+")
INSTAGRAM_RE = re.compile(r"https?://(?:www\.)?instagram\.com/[^\s\"'<>]+")

# Pages worth crawling for contact info (relative paths)
CONTACT_PATHS = [
    "/contact", "/contact-us", "/contacts",
    "/about",   "/about-us",
    "/team",    "/our-team",
    "/info",    "/reach-us",
]

# ── Core scraper ──────────────────────────────────────────────────────────────
class LeadScraper:
    def __init__(
        self,
        delay: float = 1.5,
        timeout: int = 10,
        max_pages_per_site: int = 4,
        user_agent: str = "LeadScraperBot/1.0 (+https://github.com/example/lead-scraper)",
    ):
        self.delay = delay
        self.timeout = timeout
        self.max_pages = max_pages_per_site
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent})

    # ── robots.txt cache ──────────────────────────────────────────────────────
    _robots_cache: dict[str, RobotFileParser] = {}

    def _robots(self, base: str) -> RobotFileParser:
        if base not in self._robots_cache:
            rp = RobotFileParser()
            rp.set_url(base.rstrip("/") + "/robots.txt")
            try:
                rp.read()
            except Exception:
                pass  # treat as allow-all on fetch failure
            self._robots_cache[base] = rp
        return self._robots_cache[base]

    def _allowed(self, url: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        return self._robots(base).can_fetch(self.session.headers["User-Agent"], url)

    # ── HTTP helpers ──────────────────────────────────────────────────────────
    def _get(self, url: str, retries: int = 3) -> Optional[BeautifulSoup]:
        if not self._allowed(url):
            log.info("  Blocked by robots.txt: %s", url)
            return None
        delay = 2.0
        for attempt in range(1, retries + 1):
            try:
                r = self.session.get(url, timeout=self.timeout)
                r.raise_for_status()
                return BeautifulSoup(r.text, "html.parser")
            except requests.Timeout:
                log.warning("  Timeout (attempt %d/%d): %s", attempt, retries, url)
            except requests.HTTPError as e:
                code = e.response.status_code
                # Don't retry client errors (4xx) except 429
                if code != 429 and 400 <= code < 500:
                    log.warning("  HTTP %s: %s", code, url)
                    return None
                log.warning("  HTTP %s (attempt %d/%d): %s", code, attempt, retries, url)
            except Exception as e:
                log.warning("  Error fetching %s – %s (attempt %d/%d)", url, e, attempt, retries)
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
        return None

    # ── Extraction helpers ────────────────────────────────────────────────────
    @staticmethod
    def _extract_emails(text: str) -> list[str]:
        raw = EMAIL_RE.findall(text)
        # Filter out common false positives (image filenames, etc.)
        blocked = {"png", "jpg", "gif", "svg", "webp", "css", "js", "woff"}
        return list({
            e.lower() for e in raw
            if e.split(".")[-1].lower() not in blocked
        })

    @staticmethod
    def _extract_phones(text: str) -> list[str]:
        return list(set(PHONE_RE.findall(text)))

    @staticmethod
    def _extract_social(html_text: str) -> dict[str, str]:
        return {
            "linkedin":  next(iter(LINKEDIN_RE.findall(html_text)),  ""),
            "twitter":   next(iter(TWITTER_RE.findall(html_text)),   ""),
            "facebook":  next(iter(FACEBOOK_RE.findall(html_text)),  ""),
            "instagram": next(iter(INSTAGRAM_RE.findall(html_text)), ""),
        }

    @staticmethod
    def _extract_company_name(soup: BeautifulSoup, url: str) -> str:
        # 1. OG site_name
        og = soup.find("meta", property="og:site_name")
        if og and og.get("content"):
            return og["content"].strip()
        # 2. <title> (strip trailing tagline after " | " or " – ")
        if soup.title and soup.title.string:
            title = re.split(r"[|\-–—]", soup.title.string)[0].strip()
            if title:
                return title
        # 3. Fallback: domain name
        domain = urlparse(url).netloc.lstrip("www.")
        return domain.split(".")[0].title()

    @staticmethod
    def _extract_description(soup: BeautifulSoup) -> str:
        for attr in ("name", "property"):
            tag = soup.find("meta", {attr: "description"}) or \
                  soup.find("meta", {attr: "og:description"})
            if tag and tag.get("content"):
                return tag["content"].strip()[:300]
        return ""

    @staticmethod
    def _extract_address(soup: BeautifulSoup) -> str:
        # Schema.org PostalAddress
        addr_tag = soup.find(attrs={"itemprop": "address"})
        if addr_tag:
            return addr_tag.get_text(" ", strip=True)[:200]
        # <address> element
        addr_tag = soup.find("address")
        if addr_tag:
            return addr_tag.get_text(" ", strip=True)[:200]
        return ""

    # ── Sitemap discovery ─────────────────────────────────────────────────────
    _CONTACT_KEYWORDS = re.compile(
        r"/(?:contact|about|team|reach|info|people|staff|support|help)",
        re.IGNORECASE,
    )

    def _sitemap_contact_urls(self, base: str) -> list[str]:
        """Return up to 5 contact-ish URLs found in the site's sitemap."""
        found: list[str] = []
        for sitemap_path in ("/sitemap.xml", "/sitemap_index.xml"):
            sitemap_url = base.rstrip("/") + sitemap_path
            if not self._allowed(sitemap_url):
                continue
            try:
                r = self.session.get(sitemap_url, timeout=self.timeout)
                if not r.ok:
                    continue
                soup = BeautifulSoup(r.text, "xml")
                locs = [tag.get_text(strip=True) for tag in soup.find_all("loc")]
                found += [
                    loc for loc in locs
                    if self._CONTACT_KEYWORDS.search(loc) and loc not in found
                ]
                if found:
                    break
            except Exception:
                pass
        return found[:5]

    # ── Per-site scraping ─────────────────────────────────────────────────────
    def scrape(self, url: str) -> Lead:
        lead = Lead(url=url)
        base = f"{urlparse(url).scheme}://{urlparse(url).netloc}"

        all_emails: set[str] = set()
        all_phones: set[str] = set()
        socials: dict[str, str] = {}

        sitemap_urls = self._sitemap_contact_urls(base)
        static_urls  = [urljoin(base, p) for p in CONTACT_PATHS]
        # Sitemap hits first; static fallbacks fill gaps; homepage always first
        extra = sitemap_urls + [u for u in static_urls if u not in sitemap_urls]
        urls_to_visit = [url] + extra
        visited: set[str] = set()
        pages_scraped = 0

        for page_url in urls_to_visit:
            if page_url in visited or pages_scraped >= self.max_pages:
                break
            visited.add(page_url)

            log.info("  Fetching: %s", page_url)
            soup = self._get(page_url)
            if soup is None:
                continue

            pages_scraped += 1
            html_text = str(soup)

            # Gather data from the homepage
            if page_url == url:
                lead.company_name = self._extract_company_name(soup, url)
                lead.description  = self._extract_description(soup)
                lead.address      = self._extract_address(soup)

            all_emails.update(self._extract_emails(html_text))
            all_phones.update(self._extract_phones(html_text))

            for k, v in self._extract_social(html_text).items():
                if v and not socials.get(k):
                    socials[k] = v

            time.sleep(self.delay)

        lead.emails    = "; ".join(sorted(all_emails))
        lead.phones    = "; ".join(sorted(all_phones))
        lead.linkedin  = socials.get("linkedin",  "")
        lead.twitter   = socials.get("twitter",   "")
        lead.facebook  = socials.get("facebook",  "")
        lead.instagram = socials.get("instagram", "")
        return lead

    def scrape_many(self, urls: list[str], workers: int = 1) -> list[Lead]:
        cleaned: list[str] = []
        for url in urls:
            url = url.strip()
            if not url or url.startswith("#"):
                continue
            if not url.startswith("http"):
                url = "https://" + url
            cleaned.append(url)

        total = len(cleaned)

        def _scrape_one(idx_url: tuple[int, str]) -> Lead:
            i, url = idx_url
            log.info("[%d/%d] Scraping %s", i, total, url)
            try:
                return self.scrape(url)
            except Exception as e:
                log.error("  Unhandled error for %s: %s", url, e)
                return Lead(url=url, status="error")

        indexed = list(enumerate(cleaned, 1))

        if workers <= 1:
            return [_scrape_one(item) for item in indexed]

        results: list[Optional[Lead]] = [None] * total
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_scrape_one, item): item[0] - 1 for item in indexed}
            for future in as_completed(futures):
                results[futures[future]] = future.result()
        return [r for r in results if r is not None]


# ── CSV export ────────────────────────────────────────────────────────────────
def save_csv(leads: list[Lead], path: str) -> None:
    if not leads:
        log.warning("No leads to save.")
        return
    col_names = [f.name for f in fields(Lead)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_names)
        writer.writeheader()
        for lead in leads:
            writer.writerow(asdict(lead))
    log.info("Saved %d leads → %s", len(leads), path)


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Web lead scraper")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url",  help="Single URL to scrape")
    src.add_argument("--urls", help="Path to a text file with one URL per line")
    p.add_argument("--output",   default="leads.csv",  help="Output CSV file (default: leads.csv)")
    p.add_argument("--delay",    type=float, default=1.5, help="Seconds between requests (default: 1.5)")
    p.add_argument("--timeout",  type=int,   default=10,  help="Request timeout in seconds (default: 10)")
    p.add_argument("--max-pages",type=int,   default=4,   help="Max pages crawled per site (default: 4)")
    p.add_argument("--workers",  type=int,   default=1,   help="Parallel site workers (default: 1)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.url:
        urls = [args.url]
    else:
        with open(args.urls, encoding="utf-8") as fh:
            urls = [line.strip() for line in fh if line.strip()]

    log.info("Starting scrape of %d URL(s)", len(urls))
    scraper = LeadScraper(
        delay=args.delay,
        timeout=args.timeout,
        max_pages_per_site=args.max_pages,
    )
    leads = scraper.scrape_many(urls, workers=args.workers)
    save_csv(leads, args.output)

    # Summary
    ok  = sum(1 for l in leads if l.status == "ok")
    err = len(leads) - ok
    log.info("Done. %d scraped, %d errors.", ok, err)


if __name__ == "__main__":
    main()
