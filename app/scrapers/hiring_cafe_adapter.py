from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional

from playwright.async_api import ViewportSize, async_playwright
from playwright_stealth import Stealth

from app.scrapers import ScraperAdapter

logger = logging.getLogger("scraper.hiring_cafe")


class HiringCafeAdapter(ScraperAdapter):
    """
    Scrapes job postings from hiring.cafe by opening each card's side panel
    to extract the full description.
    """

    def __init__(self):
        super().__init__()
        self._max_pages: int = 5
        self._headless: bool = False
        self._viewport: ViewportSize = {"width": 1280, "height": 720}

    # ── Required ScraperAdapter interface ──

    def get_name(self) -> str:
        return "hiring_cafe"

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the adapter from scrapers_config.yaml.

        Supported config keys:
            max_pages (int, default=5): Number of pagination pages to scrape.
            headless (bool, default=False): Run browser in headless mode.
        """
        self._max_pages = int(config.get("max_pages", 5))
        self._headless = bool(config.get("headless", False))
        self._name = config.get("name", self._name)
        self.logger = logging.getLogger(f"scraper.{self._name}")
        self.logger.info(
            f"Configured: max_pages={self._max_pages}, headless={self._headless}"
        )

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Execute the scraping operation.

        Returns:
            List of job dicts conforming to the JobData schema.
        """
        all_jobs: List[Dict[str, Any]] = []

        self.logger.info("Starting HiringCafe scraper...")

        async with Stealth().use_async(async_playwright()) as p:
            browser = await p.chromium.launch(headless=self._headless)
            context = await browser.new_context(
                user_agent=(
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
                ),
                viewport=self._viewport,
            )

            page = await context.new_page()

            page.on("response", self._intercept_response)

            self.logger.info("Navigating to https://hiring.cafe ...")
            await page.goto("https://hiring.cafe", wait_until="networkidle")

            self.logger.info("Waiting for session authorization...")
            await page.wait_for_timeout(7000)

            for page_num in range(1, self._max_pages + 1):
                self.logger.info(f"--- Processing Page {page_num} ---")

                page_jobs = await self._extract_jobs_from_page(page)
                all_jobs.extend(page_jobs)
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

            await browser.close()

        self.logger.info(f"Scraping complete. Total jobs: {len(all_jobs)}")
        return all_jobs

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