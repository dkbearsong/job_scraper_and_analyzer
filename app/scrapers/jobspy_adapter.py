"""
JobSpy Adapter for the Scraper Adapter System.

Wraps the `jobspy` library to scrape from Indeed, LinkedIn, ZipRecruiter,
and Google. This adapter migrates Part B of the original
pipeline_stage_scrape() logic.
"""

from __future__ import annotations

import itertools
import random
import time
from typing import Any, Dict, List, Optional

from jobspy import scrape_jobs

from app.scrapers import ScraperAdapter


class JobSpyAdapter(ScraperAdapter):
    """
    Adapter that scrapes job boards using the jobspy library.

    Config options:
        boards: list          — Job board names to scrape
                                  (default: ["indeed", "linkedin", "zip_recruiter", "google"])
        search_terms: list    — Search terms to use
                                  (default: ["Software Engineer"])
        target_cities: list   — Target locations
                                  (default: ["Remote"])
        requests_wanted: int  — Results wanted per search (default: 200)
        hours_old: int        — How old (in hours) results can be (default: 24)
        delay_min: float      — Min delay between requests in seconds (default: 1)
        delay_max: float      — Max delay between requests in seconds (default: 4)
        country_indeed: str   — Country for Indeed searches (default: "USA")
    """

    def __init__(self):
        super().__init__()
        self._boards: List[str] = []
        self._search_terms: List[str] = []
        self._target_cities: List[str] = []
        self._requests_wanted: int = 200
        self._hours_old: int = 24
        self._delay_min: float = 1.0
        self._delay_max: float = 4.0
        self._country_indeed: str = "USA"

    def get_name(self) -> str:
        return self._config.get("name", "jobspy_adapter")

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the JobSpy adapter from config dict."""
        self._config = config
        self._boards = config.get("boards", ["indeed", "linkedin", "zip_recruiter", "google"])
        self._search_terms = config.get("search_terms", ["Software Engineer"])
        self._target_cities = config.get("target_cities", ["Remote"])
        self._requests_wanted = config.get("requests_wanted", 200)
        self._hours_old = config.get("hours_old", 24)
        self._delay_min = config.get("delay_min", 1.0)
        self._delay_max = config.get("delay_max", 4.0)
        self._country_indeed = config.get("country_indeed", "USA")

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Scrape job boards using jobspy and return standardized job dicts.

        Returns:
            List of job dicts conforming to the JobData schema.
        """
        all_jobs: List[Dict[str, Any]] = []

        for board, search_term, location in itertools.product(
            self._boards, self._search_terms, self._target_cities
        ):
            # Build kwargs for this board/term combo
            kwargs = self._build_board_kwargs(board, search_term)

            try:
                raw_jobs = self._scrape_board(
                    board, location, self._requests_wanted, self._hours_old, **kwargs
                )
                for raw_job in raw_jobs:
                    job = self._normalize_job(raw_job)
                    if job:
                        all_jobs.append(job)
            except Exception as e:
                self.logger.error(
                    f"JobSpy scrape failed for {board}/{search_term}/{location}: {e}"
                )

            # Rate limiting
            time.sleep(random.uniform(self._delay_min, self._delay_max))

        self.logger.info(
            f"JobSpyAdapter '{self.get_name()}': collected {len(all_jobs)} jobs "
            f"from {len(self._boards)} boards × {len(self._search_terms)} terms "
            f"× {len(self._target_cities)} locations."
        )
        return all_jobs

    @staticmethod
    def _build_board_kwargs(board: str, search_term: str) -> Dict[str, Any]:
        """
        Build keyword arguments specific to each job board.

        Returns a dict of kwargs to pass to jobspy's scrape_jobs().
        """
        kwargs: Dict[str, Any] = {}

        if board == "google":
            kwargs["google_search_term"] = search_term
        elif board == "indeed":
            kwargs["search_term"] = search_term
            kwargs["country_indeed"] = "USA"
        elif board == "linkedin":
            kwargs["search_term"] = search_term
            kwargs["linkedin_fetch_description"] = True
        else:
            kwargs["search_term"] = search_term

        return kwargs

    @staticmethod
    def _scrape_board(
        site_name: str,
        location: str,
        results_wanted: int,
        hours_old: int,
        **kwargs: Any,
    ) -> list:
        """
        Call jobspy's scrape_jobs for a single board/location combo.

        Returns:
            List of raw job dicts from jobspy.
        """
        # Filter kwargs to only allowed keys
        allowed_keys = {
            "linkedin_fetch_description",
            "country_indeed",
            "google_search_term",
            "search_term",
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        result = scrape_jobs(
            site_name=[site_name],
            location=location,
            results_wanted=results_wanted,
            hours_old=hours_old,
            **filtered_kwargs,
        )
        # jobspy may return a DataFrame or a list
        if result is None:
            return []
        try:
            # DataFrame case
            if hasattr(result, "to_dict"):
                return result.to_dict("records")
        except Exception:
            pass
        if isinstance(result, list):
            return result
        return []

    @staticmethod
    def _normalize_job(raw: Any) -> Optional[Dict[str, Any]]:
        """
        Normalize a single raw jobspy result into the standardized schema.

        Returns None if the raw data cannot be normalized.
        """
        if not isinstance(raw, dict):
            return None

        # Build pay string
        min_amount = raw.get("min_amount", "")
        max_amount = raw.get("max_amount", "")
        interval = raw.get("interval", "")
        pay_parts = [str(min_amount) if min_amount else "", str(max_amount) if max_amount else "", str(interval) if interval else ""]
        pay = " - ".join(p.strip() for p in pay_parts[:2] if p.strip())
        if interval and pay:
            pay = f"{pay} {interval}".strip()

        return {
            "source": raw.get("site", "jobspy"),
            "title": raw.get("title", ""),
            "url": raw.get("job_url", ""),
            "link": raw.get("job_url", ""),
            "company": raw.get("company", ""),
            "pay": pay,
            "description": raw.get("description", ""),
            "city": raw.get("city", ""),
            "state": raw.get("state", ""),
            "location": raw.get("location", ""),
            "flexibility": raw.get("work_type", "NA") or "NA",
        }