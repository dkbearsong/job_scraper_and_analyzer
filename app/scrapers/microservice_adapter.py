"""
Microservice Adapter for the Scraper Adapter System.

Wraps the existing local microservice (localhost:5052) scraping approach.
Loads site strategies from JSON files and sends them to the microservice
for extraction. This adapter migrates Part A of the original
pipeline_stage_scrape() logic.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from random import random
from typing import Any, Dict, List, Optional

import aiohttp

from app.scrapers import ScraperAdapter


class MicroserviceAdapter(ScraperAdapter):
    """
    Adapter that scrapes company career pages via a local microservice.

    Config options:
        sites_file: str       — Path to CSV file listing company names and URLs
        strategies_dir: str   — Directory containing site strategy JSON files
        microservice_host: str — Microservice host (default http://localhost)
        microservice_port: str — Microservice port (default 5052)
        timeout: int          — Request timeout in seconds (default 120)
        delay_min: float      — Min delay between requests (default 1)
        delay_max: float      — Max delay between requests (default 3)
    """

    def __init__(self):
        super().__init__()
        self._sites_file: str = ""
        self._strategies_dir: str = "./site_strategies"
        self._microservice_host: str = "http://localhost"
        self._microservice_port: str = "5052"
        self._timeout: int = 120
        self._delay_min: float = 1.0
        self._delay_max: float = 3.0

    def get_name(self) -> str:
        return self._config.get("name", "microservice_adapter")

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the microservice adapter from config dict."""
        self._config = config
        self._sites_file = config.get("sites_file", "")
        self._strategies_dir = config.get("strategies_dir", "./site_strategies")
        self._microservice_host = config.get("microservice_host", "http://localhost")
        self._microservice_port = config.get("microservice_port", "5052")
        self._timeout = config.get("timeout", 120)
        self._delay_min = config.get("delay_min", 1.0)
        self._delay_max = config.get("delay_max", 3.0)

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Load site strategies, call the microservice, and return job data.

        Returns:
            List of job dicts in the standardized schema.
        """
        # Load sites list
        sites = self._load_sites_list()
        if not sites or not sites.get("name"):
            self.logger.warning("No sites to scrape (sites_file not configured or empty).")
            return []

        all_jobs: List[Dict[str, Any]] = []

        for i in range(len(sites["name"])):
            company_name = sites["name"][i]
            company_url = sites["site"][i]
            strategy_path = os.path.join(self._strategies_dir, f"{company_name}.json")

            if not os.path.exists(strategy_path):
                self.logger.warning(f"Strategy file not found: {strategy_path}")
                continue

            try:
                with open(strategy_path, "r") as f:
                    strategy = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load strategy {strategy_path}: {e}")
                continue

            # Determine API method based on strategy contents
            api_method = (
                "extract-paginated" if strategy.get("pagination") is not None
                else ("extract-js" if strategy.get("js_config") is not None else "extract")
            )

            # Build payload
            payload = {
                "company": company_name,
                "site": company_url,
                "strategy": strategy,
                "api_method": api_method,
            }

            # Handle string vs list URLs
            urls = strategy.get("url", [])
            if isinstance(urls, str):
                urls = [urls]

            for url in urls:
                page_payload = dict(payload)
                page_payload["strategy"] = dict(strategy)
                page_payload["strategy"]["url"] = url
                page_payload["strategy"]["company_url"] = company_url

                try:
                    raw_data = await self._call_microservice(page_payload, api_method)
                    jobs = self._parse_response(raw_data, company_url, company_name)
                    all_jobs.extend(jobs)
                except Exception as e:
                    self.logger.error(
                        f"Failed to scrape {company_name} ({url}): {e}"
                    )

                # Rate limiting
                time.sleep(random() * (self._delay_max - self._delay_min) + self._delay_min)

        self.logger.info(
            f"MicroserviceAdapter '{self.get_name()}': "
            f"collected {len(all_jobs)} jobs from {len(sites['name'])} companies."
        )
        return all_jobs

    def _load_sites_list(self) -> Dict[str, List[str]]:
        """Load the CSV sites list file."""
        from collections import defaultdict
        import csv

        if not self._sites_file or not os.path.exists(self._sites_file):
            return {}

        sites: Dict[str, List[str]] = defaultdict(list)
        with open(self._sites_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                for key, value in row.items():
                    sites[key].append(value)
        return dict(sites)

    async def _call_microservice(
        self, payload: Dict[str, Any], api_method: str
    ) -> Dict[str, Any]:
        """Call the local microservice with the given payload."""
        url = f"{self._microservice_host}:{self._microservice_port}/{api_method}"
        timeout = aiohttp.ClientTimeout(total=self._timeout)

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload) as response:
                    status = response.status
                    try:
                        resp_json = await response.json()
                    except Exception:
                        text = await response.text()
                        resp_json = {"raw": text}

                    if isinstance(resp_json, dict):
                        resp_json.setdefault("status_code", str(status))
                        if "data" not in resp_json:
                            if resp_json and not any(
                                k in resp_json for k in ("data", "error", "status_code")
                            ):
                                resp_json = {"data": resp_json, "status_code": status}
                    else:
                        resp_json = {"data": resp_json, "status_code": status}

                    return resp_json
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            return {"status_code": 408, "data": [], "error": "Request timed out"}
        except aiohttp.ClientError as e:
            return {"status_code": 503, "data": [], "error": f"Client error: {e}"}
        except Exception as e:
            return {"status_code": 500, "data": [], "error": str(e)}

    def _parse_response(
        self,
        raw_data: Dict[str, Any],
        company_url: str,
        company_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse microservice response into standardized job dicts.

        Handles both single-page and paginated responses.
        """
        status_code = raw_data.get("status_code", raw_data.get("status"))
        if str(status_code) != "200" and status_code != 200:
            error = raw_data.get("error", "No error message")
            self.logger.warning(
                f"Microservice returned status {status_code} for {company_name}: {error}"
            )
            return []

        data = raw_data.get("data", [])
        if not data:
            return []

        # Check if paginated (first item has 'jobs' key)
        if isinstance(data, list) and data and isinstance(data[0], dict) and "jobs" in data[0]:
            jobs = []
            for page in data:
                page_jobs = page.get("jobs", [])
                if isinstance(page_jobs, list):
                    jobs.extend(page_jobs)
            data = jobs

        return self._normalize_jobs(data, company_url, company_name)

    def _normalize_jobs(
        self,
        data: List[Any],
        company_url: str,
        company_name: str,
    ) -> List[Dict[str, Any]]:
        """
        Normalize microservice response items into the standardized job schema.

        This handles the existing microservice response format:
            {"title": ..., "link": ..., "company": ..., "location": ..., "flexibility": ...}
        """
        jobs: List[Dict[str, Any]] = []

        for item in data:
            if not isinstance(item, dict):
                continue

            title = item.get("title")
            if not title or title == [None] or title == "":
                continue

            # Build link URL
            link = item.get("link")
            if link and not link.startswith(("http", "https")):
                link = f"{company_url}{link}"

            # Clean location (remove "location" prefix if present)
            location = item.get("location")
            if location and isinstance(location, str):
                if re.search(r"location", location, re.IGNORECASE):
                    location = re.sub(r"location", "", location, flags=re.IGNORECASE).strip()

            job = {
                "source": "company_board",
                "title": title,
                "company": item.get("company") or company_name,
                "url": link,
                "link": link,
                "company_url": company_url,
                "flexibility": item.get("flexibility", "NA") or "NA",
                "location": location or "",
                "pay": item.get("pay", ""),
                "description": item.get("description", ""),
                "city": item.get("city", ""),
                "state": item.get("state", ""),
            }

            jobs.append(job)

        return jobs