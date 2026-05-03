"""Unit tests for lead_scraper extraction helpers and core logic."""

import time
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from bs4 import BeautifulSoup

from lead_scraper import (
    LeadScraper,
    Lead,
    save_csv,
    EMAIL_RE,
    PHONE_RE,
    LINKEDIN_RE,
    TWITTER_RE,
    FACEBOOK_RE,
    INSTAGRAM_RE,
)


def _soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


class TestExtractEmails(unittest.TestCase):
    def test_plain_email(self):
        result = LeadScraper._extract_emails("Contact us at hello@example.com please.")
        self.assertIn("hello@example.com", result)

    def test_multiple_emails_deduped(self):
        result = LeadScraper._extract_emails("a@b.com a@b.com c@d.org")
        self.assertEqual(sorted(result), ["a@b.com", "c@d.org"])

    def test_image_extension_filtered(self):
        result = LeadScraper._extract_emails("src=\"logo@2x.png\" email@company.com")
        self.assertNotIn("logo@2x.png", result)
        self.assertIn("email@company.com", result)

    def test_no_emails(self):
        self.assertEqual(LeadScraper._extract_emails("no emails here"), [])

    def test_case_normalised(self):
        result = LeadScraper._extract_emails("Info@Example.COM")
        self.assertIn("info@example.com", result)


class TestExtractPhones(unittest.TestCase):
    def test_us_number(self):
        result = LeadScraper._extract_phones("Call us at (800) 555-1234.")
        self.assertTrue(any("555" in p for p in result))

    def test_dashes(self):
        result = LeadScraper._extract_phones("800-555-6789")
        self.assertTrue(len(result) >= 1)

    def test_no_phones(self):
        self.assertEqual(LeadScraper._extract_phones("nothing here"), [])


class TestExtractSocial(unittest.TestCase):
    def _html(self, *links):
        hrefs = " ".join(f'<a href="{l}">{l}</a>' for l in links)
        return f"<html><body>{hrefs}</body></html>"

    def test_linkedin(self):
        html = self._html("https://www.linkedin.com/company/acme")
        r = LeadScraper._extract_social(html)
        self.assertEqual(r["linkedin"], "https://www.linkedin.com/company/acme")

    def test_twitter_x_domain(self):
        html = self._html("https://x.com/acmecorp")
        r = LeadScraper._extract_social(html)
        self.assertEqual(r["twitter"], "https://x.com/acmecorp")

    def test_facebook(self):
        html = self._html("https://www.facebook.com/acme")
        r = LeadScraper._extract_social(html)
        self.assertEqual(r["facebook"], "https://www.facebook.com/acme")

    def test_instagram(self):
        html = self._html("https://www.instagram.com/acme")
        r = LeadScraper._extract_social(html)
        self.assertEqual(r["instagram"], "https://www.instagram.com/acme")

    def test_missing_returns_empty(self):
        r = LeadScraper._extract_social("<html></html>")
        self.assertEqual(r["linkedin"], "")
        self.assertEqual(r["twitter"], "")


class TestExtractCompanyName(unittest.TestCase):
    def test_og_site_name(self):
        soup = _soup('<meta property="og:site_name" content="Acme Corp">')
        self.assertEqual(LeadScraper._extract_company_name(soup, "https://acme.com"), "Acme Corp")

    def test_title_strips_tagline(self):
        soup = _soup("<title>Acme Corp | Best Products</title>")
        name = LeadScraper._extract_company_name(soup, "https://acme.com")
        self.assertEqual(name, "Acme Corp")

    def test_title_dash_separator(self):
        soup = _soup("<title>Acme Corp – We build things</title>")
        name = LeadScraper._extract_company_name(soup, "https://acme.com")
        self.assertEqual(name, "Acme Corp")

    def test_fallback_domain(self):
        soup = _soup("<html></html>")
        name = LeadScraper._extract_company_name(soup, "https://www.mycompany.io")
        self.assertEqual(name, "Mycompany")


class TestExtractDescription(unittest.TestCase):
    def test_meta_description(self):
        soup = _soup('<meta name="description" content="We make great things.">')
        self.assertEqual(LeadScraper._extract_description(soup), "We make great things.")

    def test_og_description_fallback(self):
        soup = _soup('<meta property="og:description" content="OG desc.">')
        self.assertEqual(LeadScraper._extract_description(soup), "OG desc.")

    def test_truncated_to_300(self):
        long = "x" * 400
        soup = _soup(f'<meta name="description" content="{long}">')
        self.assertEqual(len(LeadScraper._extract_description(soup)), 300)

    def test_missing_returns_empty(self):
        self.assertEqual(LeadScraper._extract_description(_soup("<html></html>")), "")


class TestExtractAddress(unittest.TestCase):
    def test_itemprop_address(self):
        soup = _soup('<span itemprop="address">123 Main St, Springfield</span>')
        self.assertIn("123 Main St", LeadScraper._extract_address(soup))

    def test_address_element(self):
        soup = _soup("<address>456 Oak Ave<br>Boston, MA</address>")
        self.assertIn("456 Oak Ave", LeadScraper._extract_address(soup))

    def test_no_address(self):
        self.assertEqual(LeadScraper._extract_address(_soup("<html></html>")), "")


class TestScrapeRetry(unittest.TestCase):
    """_get should retry transient errors with backoff."""

    def setUp(self):
        self.scraper = LeadScraper(delay=0, timeout=5)

    @patch("lead_scraper.time.sleep")
    def test_retries_on_timeout(self, mock_sleep):
        import requests as req
        with patch.object(self.scraper.session, "get", side_effect=req.Timeout):
            with patch.object(self.scraper, "_allowed", return_value=True):
                result = self.scraper._get("https://example.com", retries=3)
        self.assertIsNone(result)
        self.assertEqual(mock_sleep.call_count, 2)  # sleep between attempt 1→2 and 2→3

    @patch("lead_scraper.time.sleep")
    def test_no_retry_on_404(self, mock_sleep):
        import requests as req
        resp = MagicMock()
        resp.status_code = 404
        err = req.HTTPError(response=resp)
        with patch.object(self.scraper.session, "get", side_effect=err):
            with patch.object(self.scraper, "_allowed", return_value=True):
                result = self.scraper._get("https://example.com", retries=3)
        self.assertIsNone(result)
        mock_sleep.assert_not_called()

    def test_robots_blocked_returns_none(self):
        with patch.object(self.scraper, "_allowed", return_value=False):
            result = self.scraper._get("https://example.com")
        self.assertIsNone(result)


class TestSaveCSV(unittest.TestCase):
    def test_writes_all_fields(self):
        import tempfile, csv, os
        leads = [Lead(url="https://a.com", company_name="A", emails="a@a.com", status="ok")]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            path = f.name
        try:
            save_csv(leads, path)
            with open(path, newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["url"], "https://a.com")
            self.assertEqual(rows[0]["emails"], "a@a.com")
        finally:
            os.unlink(path)

    def test_empty_list_no_file_written(self):
        import tempfile, os
        path = tempfile.mktemp(suffix=".csv")
        save_csv([], path)
        self.assertFalse(os.path.exists(path))


if __name__ == "__main__":
    unittest.main()
