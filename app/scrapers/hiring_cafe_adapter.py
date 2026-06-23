from __future__ import annotations

import asyncio
import csv
import json
import logging
import os
import random
import re
import sys
import time
import urllib.parse
from typing import Any, Dict, List, Optional

import yaml
from playwright.async_api import ProxySettings, ViewportSize, async_playwright

# Allow running this file directly (python app/scrapers/hiring_cafe_adapter.py)
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from app.scrapers import ScraperAdapter

try:
    from playwright_stealth import Stealth
    _STEALTH_AVAILABLE = True
except Exception:
    _STEALTH_AVAILABLE = False

logger = logging.getLogger("scraper.hiring_cafe")


class HiringCafeAdapter(ScraperAdapter):
    """
    Scrapes job postings from hiring.cafe by opening each card's side panel
    to extract the full description.

    Supports multiple URLs via a CSV file. The CSV path is specified in the
    config as ``urls_csv``, which is usually resolved from the ``${HIRING_CAFE_URLS_CSV}``
    environment variable. The number of pagination pages to scrape per URL is
    read from ``user_preferences.yaml`` under the key ``hiring_cafe_pages_per_url``
    (defaults to 2). If no CSV is configured, falls back to the default
    ``https://hiring.cafe`` homepage.
    """

    # Pool of realistic user agents for fingerprint randomization
    _USER_AGENTS: List[str] = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:125.0) Gecko/20100101 Firefox/125.0",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
    ]

    # Common viewport sizes to randomize
    _VIEWPORTS: List[ViewportSize] = [
        {"width": 1280, "height": 720},
        {"width": 1366, "height": 768},
        {"width": 1440, "height": 900},
        {"width": 1536, "height": 864},
        {"width": 1920, "height": 1080},
        {"width": 1600, "height": 900},
    ]

    # Common locales
    _LOCALES: List[str] = [
        "en-US",
        "en-GB",
        "en-CA",
        "en-AU",
    ]

    def __init__(self):
        super().__init__()
        self._max_pages: int = 5
        self._headless: bool = False
        self._viewport: ViewportSize = {"width": 1280, "height": 720}
        self._urls_csv: Optional[str] = None
        self._proxy_list: List[str] = []
        self._current_proxy_index: int = 0
        self._captcha_api_key: Optional[str] = None
        self._captcha_service: str = "2captcha"  # or "capsolver"

    # ── Required ScraperAdapter interface ──

    def get_name(self) -> str:
        return "hiring_cafe"

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the adapter from scrapers_config.yaml.

        Supported config keys:
            max_pages (int, default=5): Number of pagination pages to scrape.
                This is overridden by ``hiring_cafe_pages_per_url`` from
                ``user_preferences.yaml`` if that key is present.
            headless (bool, default=False): Run browser in headless mode.
            urls_csv (str, optional): Path to a CSV file with a ``url`` column
                containing hiring.cafe URLs to scrape.  If omitted, scrapes
                the default ``https://hiring.cafe`` homepage.
            proxy_list (list[str], optional): List of proxy URLs to rotate through.
                Format: "http://user:pass@host:port" or "http://host:port".
            captcha_api_key (str, optional): API key for 2Captcha or CapSolver.
            captcha_service (str, default="2captcha"): CAPTCHA service to use
                ("2captcha" or "capsolver").
        """
        self._max_pages = int(config.get("max_pages", 5))
        self._headless = bool(config.get("headless", False))
        self._name = config.get("name", self._name)

        # Resolve the CSV path (already resolved by adapter_loader if using ${VAR})
        csv_path = config.get("urls_csv", "")
        if csv_path:
            self._urls_csv = csv_path

        # Load proxy list from config or environment
        proxy_list = config.get("proxy_list", [])
        if not proxy_list:
            # Try loading from environment variable (comma-separated)
            env_proxies = os.getenv("HIRING_CAFE_PROXY_LIST", "")
            if env_proxies:
                proxy_list = [p.strip() for p in env_proxies.split(",") if p.strip()]
        self._proxy_list = proxy_list
        if self._proxy_list:
            self.logger.info(f"Loaded {len(self._proxy_list)} proxies for rotation.")

        # Load CAPTCHA solving configuration
        captcha_key = config.get("captcha_api_key", "")
        if not captcha_key:
            # Try environment variables for different services
            captcha_key = (
                os.getenv("TWOCAPTCHA_API_KEY") or
                os.getenv("CAPSOLVER_API_KEY") or
                os.getenv("CAPTCHA_API_KEY") or
                ""
            )
        self._captcha_api_key = captcha_key or None

        captcha_service = config.get("captcha_service", "")
        if not captcha_service:
            captcha_service = os.getenv("CAPTCHA_SERVICE", "2captcha")
        if captcha_service:
            self._captcha_service = captcha_service

        if self._captcha_api_key:
            self.logger.info(f"CAPTCHA solver configured: {self._captcha_service}")

        # Override max_pages from user_preferences.yaml if present
        self._load_pages_from_user_preferences()

        self.logger = logging.getLogger(f"scraper.{self._name}")
        self.logger.info(
            f"Configured: max_pages={self._max_pages}, headless={self._headless}, "
            f"urls_csv={self._urls_csv}"
        )

    # ── Fingerprint and proxy helpers ──

    def _get_random_user_agent(self) -> str:
        """Return a random user agent from the pool."""
        return random.choice(self._USER_AGENTS)

    def _get_random_viewport(self) -> ViewportSize:
        """Return a random viewport size."""
        return random.choice(self._VIEWPORTS)

    def _get_random_locale(self) -> str:
        """Return a random locale."""
        return random.choice(self._LOCALES)

    @staticmethod
    def _get_timezone_for_locale(locale: str) -> str:
        """Map locale to a realistic timezone."""
        tz_map = {
            "en-US": "America/New_York",
            "en-GB": "Europe/London",
            "en-CA": "America/Toronto",
            "en-AU": "Australia/Sydney",
        }
        return tz_map.get(locale, "America/New_York")

    def _get_next_proxy(self) -> Optional[ProxySettings]:
        """Rotate to the next proxy in the list. Returns Playwright proxy config."""
        if not self._proxy_list:
            return None

        proxy_url = self._proxy_list[self._current_proxy_index]
        self._current_proxy_index = (self._current_proxy_index + 1) % len(self._proxy_list)

        self.logger.debug(f"Using proxy: {proxy_url}")

        # Parse proxy URL - Playwright expects server, and optionally username/password
        proxy_config: ProxySettings = {"server": proxy_url}

        # If proxy contains credentials, extract them
        if "@" in proxy_url:
            # Format: http://user:pass@host:port
            parsed = urllib.parse.urlparse(proxy_url)
            proxy_config["server"] = f"{parsed.scheme}://{parsed.hostname}:{parsed.port}"
            if parsed.username:
                proxy_config["username"] = parsed.username
            if parsed.password:
                proxy_config["password"] = parsed.password

        return proxy_config

    # ── Stealth and anti-detection ──

    async def _inject_stealth_scripts(self, context) -> None:
        """Inject JavaScript patches to mask automation fingerprint."""
        # Patch navigator.webdriver
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {
                get: () => false
            });
        """)

        # Patch chrome object
        await context.add_init_script("""
            window.chrome = {
                runtime: {}
            };
        """)

        # Patch permissions
        await context.add_init_script("""
            const originalQuery = window.navigator.permissions.query;
            window.navigator.permissions.query = (parameters) => (
                parameters.name === 'notifications' ?
                    Promise.resolve({ state: Notification.permission }) :
                    originalQuery(parameters)
            );
        """)

        # Patch plugins
        await context.add_init_script("""
            Object.defineProperty(navigator, 'plugins', {
                get: () => [1, 2, 3, 4, 5]
            });
        """)

        # Patch languages
        await context.add_init_script("""
            Object.defineProperty(navigator, 'languages', {
                get: () => ['en-US', 'en']
            });
        """)

        # Mask automation in iframe
        await context.add_init_script("""
            // Patch for iframe contentWindow
            const originalContentWindow = HTMLIFrameElement.prototype.contentWindow;
            Object.defineProperty(HTMLIFrameElement.prototype, 'contentWindow', {
                get: function() {
                    const win = originalContentWindow.call(this);
                    if (win) {
                        Object.defineProperty(win.navigator, 'webdriver', {
                            get: () => false
                        });
                    }
                    return win;
                }
            });
        """)

        # Override automation-related properties
        await context.add_init_script("""
            // Override automation-related properties
            const puppeteerFuncNames = [
                'scroll', 'scrollTo', 'scrollBy', 'scrollIntoView',
                'getBoundingClientRect', 'getClientRects', 'focus'
            ];

            puppeteerFuncNames.forEach(name => {
                const original = Element.prototype[name];
                if (original) {
                    Element.prototype[name] = function(...args) {
                        const result = original.apply(this, args);
                        return result;
                    };
                }
            });
        """)

    # ── Cloudflare detection and handling ──

    async def _detect_cloudflare_challenge(self, page) -> bool:
        """Detect if Cloudflare challenge page is present.
        
        Only returns True for unmistakable Cloudflare challenge signatures.
        This is intentionally conservative to avoid false positives on legitimate pages.
        """
        try:
            page_title = (await page.title()).lower()
            page_content = (await page.content()).lower()

            # ONLY check for the most unmistakable Cloudflare challenge title
            # "just a moment" is the classic Cloudflare interstitial title
            if "just a moment" in page_title:
                self.logger.warning(f"Cloudflare detected: title='{page_title}'")
                return True

            # ONLY check for the most unmistakable Cloudflare content signature
            # "cloudflare ray id:" is unique to actual Cloudflare challenge pages
            if "cloudflare ray id:" in page_content:
                self.logger.warning("Cloudflare detected: ray ID in page content")
                return True

            # Check for specific Cloudflare challenge iframe domains
            challenge_iframes = [
                "iframe[src*='challenges.cloudflare.com']",
            ]

            for selector in challenge_iframes:
                iframe = await page.query_selector(selector)
                if iframe:
                    self.logger.warning(f"Cloudflare detected: iframe found ({selector})")
                    return True

            # No Cloudflare challenge detected - this is a normal page
            return False

        except Exception as e:
            self.logger.debug(f"Error detecting Cloudflare challenge: {e}")
            return False

    async def _handle_cloudflare_block(self, page, url: str, max_retries: int = 3) -> bool:
        """
        Attempt to bypass Cloudflare challenge.

        Returns True if challenge was bypassed, False otherwise.
        """
        for attempt in range(max_retries):
            self.logger.info(
                f"Cloudflare challenge detected (attempt {attempt + 1}/{max_retries}), "
                f"attempting bypass..."
            )

            # Wait for potential automatic bypass
            await page.wait_for_timeout(5000)

            # Check if challenge was solved automatically
            if not await self._detect_cloudflare_challenge(page):
                self.logger.info("Cloudflare challenge bypassed successfully.")
                return True

            # If CAPTCHA is present, try to solve it
            if await self._captcha_is_present(page):
                self.logger.info("CAPTCHA detected, attempting to solve...")
                solved = await self._solve_captcha_if_present(page)
                if solved:
                    await page.wait_for_timeout(3000)
                    if not await self._detect_cloudflare_challenge(page):
                        return True

            # Try clicking the checkbox if present (Turnstile)
            if await self._click_turnstile_checkbox(page):
                await page.wait_for_timeout(3000)
                if not await self._detect_cloudflare_challenge(page):
                    self.logger.info("Turnstile checkbox clicked successfully.")
                    return True

            # Reload the page as last resort
            if attempt < max_retries - 1:
                self.logger.info("Reloading page to retry challenge...")
                await page.reload(wait_until="networkidle")
                await page.wait_for_timeout(5000)

        self.logger.error("Failed to bypass Cloudflare challenge after all retries.")
        return False

    async def _captcha_is_present(self, page) -> bool:
        """Check if a CAPTCHA is present on the page."""
        captcha_selectors = [
            "iframe[src*='hcaptcha']",
            "iframe[src*='turnstile']",
            "iframe[src*='recaptcha']",
            "[data-site-key]",
            "[id*='captcha']",
            "[class*='captcha']",
        ]

        for selector in captcha_selectors:
            if await page.query_selector(selector):
                return True

        return False

    async def _solve_captcha_if_present(self, page) -> bool:
        """
        Solve CAPTCHA using configured service (2Captcha or CapSolver).

        Returns True if CAPTCHA was solved or not present.
        """
        if not self._captcha_api_key:
            self.logger.warning("CAPTCHA detected but no API key configured.")
            return False

        try:
            site_key = await self._extract_captcha_site_key(page)
            if not site_key:
                self.logger.warning("Could not extract CAPTCHA site key.")
                return False

            self.logger.info(f"Solving CAPTCHA with {self._captcha_service}...")

            if self._captcha_service == "2captcha":
                solution = await self._solve_2captcha(site_key, page.url)
            elif self._captcha_service == "capsolver":
                solution = await self._solve_capsolver(site_key, page.url)
            else:
                self.logger.error(f"Unknown CAPTCHA service: {self._captcha_service}")
                return False

            if solution:
                await self._inject_captcha_solution(page, solution)
                return True

            return False

        except Exception as e:
            self.logger.error(f"Error solving CAPTCHA: {e}")
            return False

    async def _extract_captcha_site_key(self, page) -> Optional[str]:
        """Extract CAPTCHA site key from page."""
        try:
            # Try data-site-key attribute
            site_key = await page.get_attribute("[data-site-key]", "data-site-key")
            if site_key:
                return site_key

            # Try to find in iframe src
            iframe = await page.query_selector("iframe[src*='hcaptcha'], iframe[src*='turnstile']")
            if iframe:
                src = await iframe.get_attribute("src")
                # Extract site key from URL
                match = re.search(r'[?&]sitekey=([^&]+)', src or '')
                if match:
                    return match.group(1)

            return None
        except Exception:
            return None

    async def _solve_2captcha(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve CAPTCHA using 2Captcha API."""
        try:
            api_url = "http://2captcha.com/in.php"
            params = {
                "key": self._captcha_api_key,
                "method": "turnstile",
                "sitekey": site_key,
                "pageurl": page_url,
                "json": 1,
            }

            # Submit CAPTCHA
            async with page.context.request.get(api_url, params=params) as response:
                result = await response.json()

            if result.get("status") != 1:
                self.logger.error(f"2Captcha submission failed: {result.get('request')}")
                return None

            captcha_id = result.get("request")
            self.logger.info(f"CAPTCHA submitted, ID: {captcha_id}")

            # Poll for solution
            for _ in range(30):  # Wait up to 2 minutes
                await asyncio.sleep(4)
                check_url = "http://2captcha.com/res.php"
                params = {
                    "key": self._captcha_api_key,
                    "action": "get",
                    "id": captcha_id,
                    "json": 1,
                }

                async with page.context.request.get(check_url, params=params) as response:
                    result = await response.json()

                if result.get("status") == 1:
                    self.logger.info("CAPTCHA solved successfully.")
                    return result.get("request")

            self.logger.error("CAPTCHA solving timeout.")
            return None

        except Exception as e:
            self.logger.error(f"2Captcha error: {e}")
            return None

    async def _solve_capsolver(self, site_key: str, page_url: str) -> Optional[str]:
        """Solve CAPTCHA using CapSolver API."""
        try:
            api_url = "https://api.capsolver.com/createTask"

            payload = {
                "clientKey": self._captcha_api_key,
                "task": {
                    "type": "antiTurnstileTaskProxyLess",
                    "websiteURL": page_url,
                    "websiteKey": site_key,
                }
            }

            # Create task
            async with page.context.request.post(api_url, json=payload) as response:
                result = await response.json()

            if result.get("errorId") != 0:
                self.logger.error(f"CapSolver submission failed: {result.get('errorDescription')}")
                return None

            task_id = result.get("taskId")
            self.logger.info(f"CapSolver task created, ID: {task_id}")

            # Poll for solution
            for _ in range(30):  # Wait up to 2 minutes
                await asyncio.sleep(4)
                check_url = "https://api.capsolver.com/getTaskResult"
                payload = {
                    "clientKey": self._captcha_api_key,
                    "taskId": task_id,
                }

                async with page.context.request.post(check_url, json=payload) as response:
                    result = await response.json()

                if result.get("status") == "ready":
                    solution = result.get("solution", {})
                    token = solution.get("token")
                    if token:
                        self.logger.info("CAPTCHA solved successfully.")
                        return token

            self.logger.error("CapSolver timeout.")
            return None

        except Exception as e:
            self.logger.error(f"CapSolver error: {e}")
            return None

    async def _inject_captcha_solution(self, page, solution: str) -> None:
        """Inject CAPTCHA solution token into the page."""
        try:
            await page.evaluate(f"""
                (solution) => {{
                    // Try Turnstile
                    const turnstileWidget = document.querySelector('[data-turnstile]');
                    if (turnstileWidget) {{
                        turnstileWidget.setAttribute('data-turnstile-response', solution);
                    }}

                    // Trigger callback if exists
                    if (window.turnstile && window.turnstile.render) {{
                        // The solution will be picked up on next validation
                    }}

                    // Inject into textarea (common pattern)
                    const textarea = document.querySelector('textarea[name="cf-turnstile-response"]');
                    if (textarea) {{
                        textarea.value = solution;
                        textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}

                    const textarea2 = document.querySelector('textarea[name="g-recaptcha-response"]');
                    if (textarea2) {{
                        textarea2.value = solution;
                        textarea2.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    }}
                }}
            """, solution)
        except Exception as e:
            self.logger.error(f"Error injecting CAPTCHA solution: {e}")

    async def _click_turnstile_checkbox(self, page) -> bool:
        """Click Turnstile checkbox if present."""
        try:
            # Look for the checkbox iframe
            iframe = await page.query_selector("iframe[src*='turnstile'], iframe[src*='challenges']")
            if not iframe:
                return False

            # Click within the iframe
            frame = await iframe.content_frame()
            if frame:
                checkbox = await frame.query_selector("input[type='checkbox'], .checkbox")
                if checkbox:
                    await checkbox.click()
                    return True

            return False
        except Exception:
            return False

    # ── Load page count from user_preferences.yaml ──

    def _load_pages_from_user_preferences(self) -> None:
        """
        Read ``hiring_cafe_pages_per_url`` from ``user_preferences.yaml`` and
        override ``self._max_pages`` if the key is present.
        """
        prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
        if not os.path.exists(prefs_path):
            return
        try:
            with open(prefs_path, "r") as f:
                prefs = yaml.safe_load(f) or {}
            pages = prefs.get("hiring_cafe_pages_per_url")
            if pages is not None:
                self._max_pages = int(pages)
        except Exception as e:
            self.logger.warning(
                f"Could not read 'hiring_cafe_pages_per_url' from {prefs_path}: {e}"
            )

    # ── Load URLs from CSV ──

    def _load_urls_from_csv(self) -> List[str]:
        """
        Read the CSV file at ``self._urls_csv`` and return all URLs from the
        ``url`` column.  Returns an empty list if the file is missing or empty.
        """
        urls: List[str] = []
        if not self._urls_csv or not os.path.exists(self._urls_csv):
            return urls

        try:
            with open(self._urls_csv, "r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    url = (row.get("url") or "").strip()
                    if url:
                        urls.append(url)
            self.logger.info(f"Loaded {len(urls)} URL(s) from {self._urls_csv}")
        except Exception as e:
            self.logger.error(f"Failed to read URLs from {self._urls_csv}: {e}")

        return urls

    # ── Main scrape entry point ──

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Execute the scraping operation.

        If a CSV file was configured, scrapes each URL in the CSV for up to
        ``self._max_pages`` pages each.  Otherwise scrapes the default
        ``https://hiring.cafe`` homepage.

        Returns:
            List of job dicts conforming to the JobData schema.
        """
        all_jobs: List[Dict[str, Any]] = []

        # Determine which URLs to scrape
        urls_to_scrape = self._load_urls_from_csv()
        if not urls_to_scrape:
            urls_to_scrape = ["https://hiring.cafe"]
            self.logger.info("No CSV configured or empty CSV — using default URL.")

        self.logger.info(
            f"Starting HiringCafe scraper for {len(urls_to_scrape)} URL(s) "
            f"({self._max_pages} page(s) each)..."
        )

        async with Stealth().use_async(async_playwright()) as p:
            # Rotate proxy per URL if proxy list is configured
            proxy = self._get_next_proxy() if self._proxy_list else None

            browser = await p.chromium.launch(
                headless=self._headless,
                proxy=proxy,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-setuid-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ]
            )

            # Randomize fingerprint for each scrape session
            viewport = self._get_random_viewport()
            user_agent = self._get_random_user_agent()
            locale = self._get_random_locale()

            context = await browser.new_context(
                user_agent=user_agent,
                viewport=viewport,
                locale=locale,
                timezone_id=self._get_timezone_for_locale(locale),
                permissions=["geolocation"],
                geolocation={"latitude": 40.7128, "longitude": -74.0060},
                extra_http_headers={
                    "Accept-Language": f"{locale},en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )

            # Inject stealth scripts to mask automation
            await self._inject_stealth_scripts(context)

            page = await context.new_page()
            page.on("response", self._intercept_response)

            for url_index, url in enumerate(urls_to_scrape, start=1):
                self.logger.info(
                    f"=== Processing URL {url_index}/{len(urls_to_scrape)}: {url} ==="
                )

                # Create new page with fresh fingerprint for each URL
                if url_index > 1:
                    await page.close()
                    page = await context.new_page()
                    page.on("response", self._intercept_response)

                page_jobs = await self._scrape_single_url(page, url)
                all_jobs.extend(page_jobs)
                self.logger.info(f"URL {url_index}: extracted {len(page_jobs)} jobs.")

            await browser.close()

        self.logger.info(
            f"Scraping complete across {len(urls_to_scrape)} URL(s). "
            f"Total jobs: {len(all_jobs)}"
        )
        return all_jobs

    # ── Scrape a single URL ──

    async def _scrape_single_url(
        self, page, url: str
    ) -> List[Dict[str, Any]]:
        """
        Navigate to *url* on hiring.cafe and scrape jobs across up to
        ``self._max_pages`` pages of results.

        Args:
            page: Playwright page object (already in a browser context).
            url: The hiring.cafe URL to scrape.

        Returns:
            List of job dicts scraped from this URL.
        """
        url_jobs: List[Dict[str, Any]] = []

        self.logger.info(f"Navigating to {url} ...")
        await page.goto(url, wait_until="networkidle")

        # Quick check for Cloudflare challenge
        if await self._detect_cloudflare_challenge(page):
            self.logger.warning("Cloudflare challenge detected on initial load!")
            bypassed = await self._handle_cloudflare_block(page, url)
            if not bypassed:
                self.logger.error("Could not bypass Cloudflare challenge. Skipping URL.")
                return url_jobs
            # Wait for page to fully load after bypass
            await page.wait_for_timeout(5000)

        self.logger.info("Waiting for session authorization...")
        await page.wait_for_timeout(7000)

        for page_num in range(1, self._max_pages + 1):
            self.logger.info(f"--- URL: {url} | Page {page_num} ---")

            page_jobs = await self._extract_jobs_from_page(page)
            url_jobs.extend(page_jobs)
            self.logger.info(f"Page {page_num}: extracted {len(page_jobs)} jobs.")

            self.logger.info("Checking for next page...")
            next_button = await page.query_selector("[aria-label='Next page']")
            if next_button and await next_button.is_visible():
                await next_button.click()
                self.logger.info("Navigating to next page...")
                await page.wait_for_timeout(5000)
            else:
                self.logger.info("No more pages found.")
                break

        return url_jobs

    # ── Internal helpers ──

    @staticmethod
    async def _intercept_response(response):
        """Capture API responses for debugging."""
        if "search" in response.url or "jobs" in response.url:
            try:
                if "application/json" in response.headers.get("content-type", ""):
                    data = await response.json()
                    with open("raw_intercepted_jobs.json", "w") as f:
                        json.dump(data, f, indent=4)
            except Exception:
                pass

    async def _extract_jobs_from_page(self, page) -> List[Dict[str, Any]]:
        """
        Extract all job cards from the current page by opening each card's
        side panel to retrieve the full description.
        """
        jobs: List[Dict[str, Any]] = []

        # Check for Cloudflare challenge before proceeding
        if await self._detect_cloudflare_challenge(page):
            self.logger.warning("Cloudflare challenge detected during scraping!")
            bypassed = await self._handle_cloudflare_block(page, page.url)
            if not bypassed:
                self.logger.error("Could not bypass Cloudflare challenge. Returning empty.")
                return jobs
            # Wait for page to settle after bypass
            await page.wait_for_timeout(3000)

        grid_selector = "div[class*='grid'][class*='grid-cols-1']"
        side_panel = "div.chakra-slide"

        # Wait for the grid container to be present
        try:
            await page.wait_for_selector(grid_selector, timeout=15000)
        except Exception as e:
            self.logger.error(f"No job grid found on page: {e}")
            return jobs

        card_index = 0
        max_iterations = 150

        while card_index < max_iterations:
            try:
                # Re-query cards fresh each iteration to avoid stale references
                container = await page.query_selector(grid_selector)
                if not container:
                    break

                current_cards = await container.query_selector_all(":scope > div")
                if card_index >= len(current_cards):
                    self.logger.info(
                        f"Processed all {len(current_cards)} cards on this page."
                    )
                    break

                card = current_cards[card_index]
                await card.scroll_into_view_if_needed()
                await page.wait_for_timeout(300)

                # ── Extract metadata from the card ──
                job_data = await self._extract_card_info(card)

                self.logger.info(f"Opening details for: {job_data['title']}")

                # ── Click the card to open the side panel ──
                if not await self._click_card_to_open_panel(card, page, side_panel):
                    jobs.append(job_data)
                    card_index += 1
                    continue

                # ── Wait for the side panel to render ──
                await page.wait_for_timeout(1500)

                # ── Extract description, with retry for "Loading..." ──
                job_data["description"] = await self._extract_description_with_retry(
                    page, side_panel
                )

                # ── Close the side panel ──
                await self._close_side_panel(page, side_panel)

                await page.wait_for_timeout(500)
                # Handle multi-listing cards (companies that group multiple
                # jobs under the same card). Each toggle produces an entirely
                # new job that needs to be scraped from scratch.
                try:
                    extra_jobs = await self._handle_multi_listing_card(
                        card, page, side_panel, grid_selector, card_index
                    )
                    jobs.extend(extra_jobs)
                except Exception as e:
                    self.logger.debug(f"Multi-list handling failed: {e}")

                jobs.append(job_data)
                card_index += 1
                self.logger.info(f"Card {card_index}: '{job_data['title']}' done.")

                # Random delay between cards
                await asyncio.sleep(random.uniform(0.3, 1.0))

            except Exception as e:
                self.logger.error(f"Error processing card {card_index}: {e}")
                try:
                    await page.keyboard.press("Escape")
                    await page.wait_for_timeout(1000)
                except Exception:
                    pass
                card_index += 1
                continue

        return jobs

    @staticmethod
    async def _extract_card_info(card) -> Dict[str, str]:
        """Extract title, company, flexibility, pay, location from a card element."""
        # Title: bold span
        title_el = await card.query_selector(
            "span[class*='w-full'][class*='font-bold'][class*='text-start']"
        )
        title_text = (await title_el.inner_text()).strip() if title_el else "N/A"

        # Company: nested bold span in the company info section
        company_el = await card.query_selector(
            "div[class*='flex'][class*='mb-4'][class*='mt-2']"
            " span span[class*='font-bold']"
        )
        company_text = (await company_el.inner_text()).strip() if company_el else "N/A"

        # Pay: green-bordered span
        pay_el = await card.query_selector("span[class*='border-green-600']")
        pay_text = (await pay_el.inner_text()).strip() if pay_el else ""

        # Location: gray badge div → child span
        loc_el = await card.query_selector(
            "div[class*='w-fit'][class*='bg-gray-50'] span"
        )
        loc_text = (await loc_el.inner_text()).strip() if loc_el else ""

        # Flexibility: 2nd span in the flex-wrap container
        flex_container = await card.query_selector("div[class*='flex-wrap']")
        flex_text = ""
        if flex_container:
            flex_spans = await flex_container.query_selector_all("span")
            if len(flex_spans) >= 2:
                flex_text = (await flex_spans[1].inner_text()).strip()
            elif len(flex_spans) >= 1:
                flex_text = (await flex_spans[0].inner_text()).strip()

        return {
            "title": title_text,
            "company": company_text,
            "flexibility": flex_text if flex_text else "NA",
            "pay": pay_text,
            "location": loc_text,
            "description": "",
        }

    async def _click_card_to_open_panel(
        self, card, page, side_panel: str
    ) -> bool:
        """
        Click a card to open the side panel. Returns True if the panel appeared.

        Tries three strategies:
            1. Hover + click the cursor-zoom-in overlay (most common).
            2. Click the card body directly.
            3. Click using JavaScript as a last resort.
        """
        strategies = [
            self._click_via_zoom_overlay,
            self._click_card_direct,
            self._click_via_javascript,
        ]

        for strategy in strategies:
            try:
                await strategy(card, page)
                await page.wait_for_selector(side_panel, timeout=8000)
                self.logger.debug("Side panel opened successfully.")
                return True
            except Exception:
                self.logger.debug(
                    f"Click strategy {strategy.__name__} failed, trying next."
                )
                await page.wait_for_timeout(300)
                continue

        self.logger.warning(
            "All click strategies failed -- could not open side panel."
        )
        return False

    @staticmethod
    async def _click_via_zoom_overlay(card, page):
        """
        Strategy 1: Hover to reveal the cursor-zoom-in div, then click it.
        """
        await card.hover()
        await page.wait_for_timeout(500)

        zoom_target = await card.query_selector("[class*='cursor-zoom-in']")
        if zoom_target:
            await zoom_target.click()
        else:
            await card.click()

    @staticmethod
    async def _click_card_direct(card, page):
        """Strategy 2: Click the card directly without hover."""
        await card.click(force=True)

    @staticmethod
    async def _click_via_javascript(card, page):
        """Strategy 3: Use JavaScript dispatchEvent as a last resort."""
        await page.evaluate(
            """(element) => {
                const rect = element.getBoundingClientRect();
                const clickEvent = new PointerEvent('click', {
                    bubbles: true,
                    cancelable: true,
                    clientX: rect.left + rect.width / 2,
                    clientY: rect.top + rect.height / 2,
                });
                element.dispatchEvent(clickEvent);
            }""",
            card,
        )

    async def _extract_description_with_retry(
        self, page, side_panel: str, max_retries: int = 3
    ) -> str:
        """
        Extract the job description from the side panel, retrying if the panel
        only shows "Loading job description..." placeholder text.
        """
        panel = page.locator(side_panel)

        for attempt in range(max_retries):
            desc_text = ""
            try:
                desc_container = panel.locator(
                    "article.prose.prose-h1\\:text-2xl.pt-4.pb-16"
                ).first
                if await desc_container.count() == 0:
                    desc_container = panel.locator(
                        "div[class*='pt-4'][class*='pb-16']"
                    ).first
                if await desc_container.count() == 0:
                    desc_container = panel.locator(
                        "div[class*='prose']"
                    ).first

                if await desc_container.count() > 0:
                    desc_text = await desc_container.inner_text()
                else:
                    self.logger.debug("No description container found.")
                    await page.wait_for_timeout(2000)
                    continue
            except Exception as e:
                self.logger.debug(
                    f"Description extraction attempt {attempt + 1} failed: {e}"
                )
                await page.wait_for_timeout(2000)
                continue

            # Check if the text is just the loading placeholder
            stripped = desc_text.strip()
            if re.match(
                r"^Loading\s+job\s+description", stripped, re.IGNORECASE
            ):
                self.logger.info(
                    f"Attempt {attempt + 1}: still loading, waiting and retrying..."
                )
                await page.wait_for_timeout(3000)
                continue

            if len(stripped) < 20:
                self.logger.debug(
                    f"Attempt {attempt + 1}: description too short "
                    f"({len(stripped)} chars), retrying..."
                )
                await page.wait_for_timeout(2000)
                continue

            self.logger.debug(
                f"Description extracted ({len(desc_text)} chars)."
            )
            return desc_text

        self.logger.warning(
            f"Could not extract description after {max_retries} attempts."
        )
        return ""

    async def _close_side_panel(self, page, side_panel: str) -> None:
        """Close the side panel using Escape key and verify it closed."""
        self.logger.debug("Closing side panel...")

        await page.keyboard.press("Escape")
        await page.wait_for_timeout(500)

        try:
            await page.wait_for_selector(
                side_panel, state="hidden", timeout=5000
            )
            self.logger.debug("Side panel closed.")
        except Exception as e:
            self.logger.warning(
                f"Side panel did not close cleanly: {e}"
            )
            await page.keyboard.press("Escape")
            await page.wait_for_timeout(1000)

    @staticmethod
    async def _get_card_by_index(page, grid_selector: str, index: int):
        """Re-query the grid and return the card ElementHandle at the given index."""
        container = await page.query_selector(grid_selector)
        if not container:
            return None
        cards = await container.query_selector_all(":scope > div")
        if index >= len(cards):
            return None
        return cards[index]

    async def _handle_multi_listing_card(
        self, card, page, side_panel: str,
        grid_selector: str, card_index: int
    ) -> List[Dict[str, Any]]:
        """
        Detect and iterate multi-listing controls on a card.

        Looks for a ``div.flex.items-center.space-x-2`` control under the card
        that has (button, element, button) as immediate children. If found,
        repeatedly clicks the second button to rotate through listings on the
        same card. Each new listing is treated as an entirely new job: the
        card metadata (title, company, etc.) is extracted fresh, the side
        panel is opened, the description is scraped, and the panel is closed.
        Stops when the button becomes disabled or after a reasonable guard.

        ``grid_selector`` and ``card_index`` are used to re-query the card
        element after DOM updates (since the old handle may become stale).

        Returns:
            List of additional job dicts for the extra listings found.
        """
        extra_jobs: List[Dict[str, Any]] = []

        try:
            # Helper to get a fresh card handle
            async def _fresh_card():
                return await self._get_card_by_index(page, grid_selector, card_index)

            # Re-scroll and hover the card so toggle controls are interactive
            await card.scroll_into_view_if_needed()
            await page.wait_for_timeout(200)
            await card.hover()
            await page.wait_for_timeout(500)

            # ── Locate the multi-listing control div (search within the card) ──
            control_info = await page.evaluate("""
                (element) => {
                    const candidates = element.querySelectorAll('div');
                    for (const candidate of candidates) {
                        const cls = candidate.className || '';
                        if (cls.includes('flex') && cls.includes('items-center') && cls.includes('space-x-2')) {
                            const kids = Array.from(candidate.children);
                            if (kids.length >= 3) {
                                const firstTag = kids[0].tagName.toLowerCase();
                                const lastTag = kids[2].tagName.toLowerCase();
                                if (firstTag === 'button' && lastTag === 'button') {
                                    return { found: true, html: candidate.outerHTML.substring(0, 500) };
                                }
                            }
                        }
                    }
                    return { found: false, html: element.outerHTML.substring(0, 500) };
                }
            """, card)

            if not control_info.get("found"):
                self.logger.info(
                    f"No multi-listing toggle control found on card. "
                    f"Card HTML (first 500 chars): {control_info.get('html', 'N/A')}"
                )
                return extra_jobs

            self.logger.info(
                f"Multi-listing toggle control found! HTML: {control_info.get('html', 'N/A')}"
            )

            # Get an element handle for the control div using page.evaluate_handle
            control_handle = await page.evaluate_handle("""
                (element) => {
                    const candidates = element.querySelectorAll('div');
                    for (const candidate of candidates) {
                        const cls = candidate.className || '';
                        if (cls.includes('flex') && cls.includes('items-center') && cls.includes('space-x-2')) {
                            const kids = Array.from(candidate.children);
                            if (kids.length >= 3) {
                                const firstTag = kids[0].tagName.toLowerCase();
                                const lastTag = kids[2].tagName.toLowerCase();
                                if (firstTag === 'button' && lastTag === 'button') {
                                    return candidate;
                                }
                            }
                        }
                    }
                    return null;
                }
            """, card)

            if not control_handle:
                self.logger.info("Could not get handle for multi-listing control (evaluate_handle returned null).")
                return extra_jobs

            # Get the two buttons from the control
            children_handles = await control_handle.query_selector_all(":scope > *")
            if len(children_handles) < 3:
                self.logger.info(f"Multi-listing control has {len(children_handles)} children.")
                return extra_jobs

            next_btn = children_handles[2]  # Second button (next) - the one we click to rotate forward

            # Verify tag
            tag = await page.evaluate("(el) => el.tagName.toLowerCase()", next_btn)
            self.logger.info(f"Multi-listing next button tag: {tag}")

            # Build a composite fingerprint of the card's metadata so we can
            # detect listing changes even when only the location (or other
            # fields) changes while the title stays the same.
            def _fingerprint(info: dict) -> str:
                return "|||".join([
                    info.get("title", ""),
                    info.get("company", ""),
                    info.get("location", ""),
                    info.get("pay", ""),
                    info.get("flexibility", ""),
                ])

            # Grab the initial info so we can detect listing changes
            initial_info = await self._extract_card_info(card)
            previous_fingerprint = _fingerprint(initial_info)
            self.logger.info(
                f"Starting multi-list toggle for card: "
                f"'{initial_info['title']}' @ '{initial_info['location']}'"
            )

            # max_iterations = 3 means we toggle at most 3 times.
            # Combined with the original job already scraped, that gives
            # up to 4 total jobs per card. If a 4th toggle were available,
            # we log "more than 4" and stop.
            iterations = 0
            max_iterations = 3

            while iterations < max_iterations + 1:  # Allow one extra check to log "more than 4"
                try:
                    # First, check if next button is disabled — if so, we're done
                    try:
                        disabled_attr = await next_btn.get_attribute("disabled")
                        aria_disabled = await next_btn.get_attribute("aria-disabled")
                        cls = await next_btn.get_attribute("class")
                        if disabled_attr or aria_disabled == "true" or (cls and ("disabled" in cls or "opacity-50" in cls)):
                            if iterations >= max_iterations:
                                self.logger.info(
                                    f"HiringCafe: company '{initial_info.get('company', 'unknown')}' "
                                    "had more than 4 listings in this search"
                                )
                            break
                    except Exception:
                        pass

                    if iterations >= max_iterations:
                        # We've already done 3 toggles, button is still active = more than 4 total
                        self.logger.info(
                            f"HiringCafe: company '{initial_info.get('company', 'unknown')}' "
                            "had more than 4 listings in this search"
                        )
                        break

                    self.logger.info(f"Clicking multi-listing next button (iteration {iterations + 1})...")
                    try:
                        await next_btn.click()
                    except Exception:
                        try:
                            await next_btn.click(force=True)
                        except Exception:
                            self.logger.info("Failed to click multi-listing next button.")
                            break

                    # Wait for DOM to settle after toggle
                    await page.wait_for_timeout(500)

                    # Re-query the card by index — the old handle may be stale after DOM update
                    current_card = await _fresh_card()
                    if not current_card:
                        self.logger.info("Card element went away after toggle — page may have re-rendered.")
                        break

                    # Poll for any metadata change (short 1.5s max)
                    listing_changed = False
                    total_wait = 0.0
                    while total_wait < 1.5:
                        await page.wait_for_timeout(250)
                        total_wait += 0.25
                        try:
                            current_info = await self._extract_card_info(current_card)
                            current_fingerprint = _fingerprint(current_info)
                        except Exception:
                            continue

                        if current_fingerprint and current_fingerprint != previous_fingerprint:
                            listing_changed = True
                            break

                    if not listing_changed:
                        self.logger.info(
                            f"Card metadata did not change after button click — "
                            f"no more listings to toggle through. "
                            f"Still: '{initial_info['title']}' @ '{initial_info['location']}'"
                        )
                        break

                    # The card now shows a different listing.
                    # Treat it as an entirely new job: extract all metadata fresh.
                    extra_job_data = current_info  # Already extracted in the poll loop

                    # Re-query the card again (may have been re-rendered during polling)
                    current_card = await _fresh_card()
                    if not current_card:
                        self.logger.info("Card disappeared after title change.")
                        break

                    # Hover and open the side panel for this new listing
                    await current_card.scroll_into_view_if_needed()
                    await page.wait_for_timeout(200)
                    await current_card.hover()
                    await page.wait_for_timeout(500)
                    opened = await self._click_card_to_open_panel(current_card, page, side_panel)
                    if not opened:
                        extra_jobs.append(extra_job_data)
                        self.logger.info(
                            f"Could not open side panel for multi-listing toggle; "
                            f"recording card metadata only for '{current_info.get('title', '?')}'."
                        )
                        break

                    await page.wait_for_timeout(800)

                    # Extract description for this new listing
                    extra_job_data["description"] = await self._extract_description_with_retry(
                        page, side_panel
                    )

                    # Close the side panel
                    await self._close_side_panel(page, side_panel)

                    extra_jobs.append(extra_job_data)

                    iterations += 1
                    self.logger.info(
                        f"Multi-listing toggle #{iterations}: "
                        f"'{previous_fingerprint}' -> '{current_fingerprint}'"
                    )

                    # Re-query the card and toggle control for the next iteration
                    current_card = await _fresh_card()
                    if not current_card:
                        self.logger.info("Card disappeared after toggle iteration.")
                        break

                    await current_card.scroll_into_view_if_needed()
                    await current_card.hover()
                    await page.wait_for_timeout(500)

                    # Re-find the control within the fresh card
                    try:
                        control_handle = await page.evaluate_handle("""
                            (element) => {
                                const candidates = element.querySelectorAll('div');
                                for (const candidate of candidates) {
                                    const cls = candidate.className || '';
                                    if (cls.includes('flex') && cls.includes('items-center') && cls.includes('space-x-2')) {
                                        const kids = Array.from(candidate.children);
                                        if (kids.length >= 3) {
                                            const firstTag = kids[0].tagName.toLowerCase();
                                            const lastTag = kids[2].tagName.toLowerCase();
                                            if (firstTag === 'button' && lastTag === 'button') {
                                                return candidate;
                                            }
                                        }
                                    }
                                }
                                return null;
                            }
                        """, current_card)
                        if not control_handle:
                            self.logger.info("Multi-listing control disappeared after toggle.")
                            break
                        children_handles = await control_handle.query_selector_all(":scope > *")
                        if len(children_handles) < 3:
                            self.logger.info("Multi-listing control no longer has enough children.")
                            break
                        next_btn = children_handles[2]
                        tag = await page.evaluate("(el) => el.tagName.toLowerCase()", next_btn)
                        if tag != "button":
                            self.logger.info("Next child is no longer a button.")
                            break
                    except Exception as e:
                        self.logger.info(f"Could not re-query multi-listing control: {e}")
                        break

                    # Check if next button is now disabled
                    try:
                        disabled_attr = await next_btn.get_attribute("disabled")
                        aria_disabled = await next_btn.get_attribute("aria-disabled")
                        cls = await next_btn.get_attribute("class")
                        if disabled_attr or aria_disabled == "true" or (cls and ("disabled" in cls or "opacity-50" in cls)):
                            self.logger.info("Multi-listing next button is disabled — end of listings.")
                            break
                    except Exception:
                        break

                    # Prepare for next iteration
                    previous_fingerprint = current_fingerprint
                    await page.wait_for_timeout(400)

                except Exception as e:
                    self.logger.info(f"Error during multi-list iteration: {e}")
                    break

            if iterations > 0:
                self.logger.info(
                    f"Finished multi-listing: found {len(extra_jobs)} extra job(s) "
                    f"for this card."
                )
            return extra_jobs
        except Exception as e:
            self.logger.info(f"_handle_multi_listing_card failed: {e}")
            return extra_jobs


# ─────────────────────────────────────────────────────────
# Test / Debug Entry Point
# ─────────────────────────────────────────────────────────

def _setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Configure logging to stdout and optionally to a file."""
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # 1. Setup StreamHandler (Terminal output)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(level)
    root.addHandler(stream_handler)

    # 2. Setup FileHandler (File output)
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
            print(f"Logging to file: {log_file}")
        except Exception as e:
            print(f"Failed to setup file logging: {e}. Continuing with terminal only.")

async def _scrape_site(max_pages: int = 1, headless: bool = True) -> List[Dict[str, Any]]:
    """
    Create a HiringCafeAdapter, configure it, scrape, and return results.

    Args:
        max_pages: Number of pages to scrape (default 1 for quick testing).
        headless: Whether to run the browser headlessly.

    Returns:
        List of job dicts scraped from hiring.cafe.
    """
    adapter = HiringCafeAdapter()
    adapter.configure({
        "max_pages": max_pages,
        "headless": headless,
    })
    results = await adapter.scrape()
    return results


def _print_summary(jobs: List[Dict[str, Any]]) -> None:
    """Print a human-readable summary of scraped jobs."""
    print(f"\n{'='*70}")
    print(f"  SCRAPING COMPLETE — {len(jobs)} job(s) found")
    print(f"{'='*70}")

    if not jobs:
        print("  (no results)")
        print()

    for i, job in enumerate(jobs, start=1):
        title = job.get("title", "N/A")
        company = job.get("company", "N/A")
        location = job.get("location", "")
        flexibility = job.get("flexibility", "")
        pay = job.get("pay", "")
        source = job.get("source", "hiring_cafe")
        desc_len = len(job.get("description", "") or "")

        print(f"\n  [{i}] {title}")
        print(f"      Company:     {company}")
        print(f"      Location:    {location}")
        print(f"      Flexibility: {flexibility}")
        print(f"      Pay:         {pay}")
        print(f"      Source:      {source}")
        print(f"      Description: {desc_len} chars")

    print(f"\n{'='*70}\n")


def _print_json(jobs: List[Dict[str, Any]]) -> None:
    """Print scraped jobs as formatted JSON."""
    print(json.dumps(jobs, indent=2, ensure_ascii=False, default=str))


def _print_flat_text(jobs: List[Dict[str, Any]]) -> None:
    """Print scraped jobs as a compact flat-text dump (good for grepping)."""
    for i, job in enumerate(jobs, start=1):
        desc = (job.get("description") or "")[:200].replace("\n", " ").strip()
        print(f"[{i:>3}] {job.get('company','?'):35s} | {job.get('title','?'):60s} | {job.get('location',''):25s} | {desc}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Test the HiringCafe scraper adapter standalone.",
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=1,
        help="Number of pagination pages to scrape (default: 1).",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Run browser in headless mode (default: True).",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        dest="visible",
        help="Shortcut for --no-headless (show the browser window).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a file where logs should be written.",
    )
    parser.add_argument(
        "--format",
        choices=["summary", "json", "flat"],
        default="summary",
        help="Output format (default: summary).",
    )

    args = parser.parse_args()

    # If --visible is passed, override headless to False
    headless = args.headless
    if args.visible:
        headless = False

    log_level = logging.DEBUG if args.debug else logging.INFO
    _setup_logging(level=log_level, log_file=args.log_file)

    logger.info(
        f"Starting standalone test: pages={args.pages}, "
        f"headless={headless}, format={args.format}"
    )

    jobs = asyncio.run(_scrape_site(max_pages=args.pages, headless=headless))

    if args.format == "json":
        _print_json(jobs)
    elif args.format == "flat":
        _print_flat_text(jobs)
    else:
        _print_summary(jobs)