"""
HTTP Endpoint Adapter for the Scraper Adapter System.

Calls a remote HTTP/API endpoint and expects the response to be in the
standardized job data format. If the data is not correctly formatted,
the validator will catch it and log diagnostics.

For users who need to transform raw API responses, they should write
a custom module adapter instead.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from random import uniform
from typing import Any, Dict, List, Optional

import aiohttp

from app.scrapers import ScraperAdapter


class HttpAdapter(ScraperAdapter):
    """
    Adapter that calls an HTTP endpoint and returns job data.

    Config options:
        url: str              — The endpoint URL to call
        method: str           — HTTP method (GET or POST, default POST)
        headers: dict         — Optional HTTP headers
        timeout: int          — Request timeout in seconds (default 120)
        payloads: list        — Optional list of payloads to send (for batch endpoints)
        payloads_file: str    — Optional path to a JSON file containing payloads
        auth_header: str      — Optional Authorization header value
        delay_min: float      — Min delay between requests in seconds (default 1)
        delay_max: float      — Max delay between requests in seconds (default 3)
    """

    def __init__(self):
        super().__init__()
        self._url: str = ""
        self._method: str = "POST"
        self._headers: Dict[str, str] = {}
        self._timeout: int = 120
        self._payloads: List[Dict[str, Any]] = []
        self._delay_min: float = 1.0
        self._delay_max: float = 3.0

    def get_name(self) -> str:
        return self._config.get("name", "http_adapter")

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the HTTP adapter from config dict."""
        self._config = config
        self._url = config.get("url", "")
        self._method = config.get("method", "POST").upper()
        self._headers = config.get("headers", {})
        self._timeout = config.get("timeout", 120)
        self._delay_min = config.get("delay_min", 1.0)
        self._delay_max = config.get("delay_max", 3.0)

        if not self._url:
            raise ValueError("HttpAdapter requires a 'url' in config.")

        # Add auth header if provided
        auth_header = config.get("auth_header", "")
        if auth_header:
            self._headers["Authorization"] = auth_header

        # Load payloads from config or file
        self._payloads = config.get("payloads", [])
        payloads_file = config.get("payloads_file", "")
        if payloads_file and os.path.exists(payloads_file):
            try:
                with open(payloads_file, "r") as f:
                    file_payloads = json.load(f)
                if isinstance(file_payloads, list):
                    self._payloads.extend(file_payloads)
                elif isinstance(file_payloads, dict):
                    self._payloads.append(file_payloads)
            except Exception as e:
                self.logger.error(f"Failed to load payloads from {payloads_file}: {e}")

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Call the HTTP endpoint(s) and return collected job data.

        If payloads are provided, each payload is sent as a separate request.
        If no payloads are provided, a single request with empty body is made.
        """
        all_jobs: List[Dict[str, Any]] = []

        if not self._payloads:
            # Single request with no payload
            result = await self._make_request({})
            all_jobs.extend(self._extract_jobs(result))
        else:
            for i, payload in enumerate(self._payloads):
                result = await self._make_request(payload)
                jobs = self._extract_jobs(result)
                all_jobs.extend(jobs)

                # Rate limiting between requests
                if i < len(self._payloads) - 1:
                    delay = uniform(self._delay_min, self._delay_max)
                    time.sleep(delay)

        self.logger.info(
            f"HttpAdapter '{self.get_name()}': collected {len(all_jobs)} jobs "
            f"from {len(self._payloads) or 1} request(s)."
        )
        return all_jobs

    async def _make_request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a single HTTP request to the configured endpoint.

        Returns the parsed JSON response, or an error dict.
        """
        timeout = aiohttp.ClientTimeout(total=self._timeout)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                kwargs: Dict[str, Any] = {"headers": self._headers}

                if self._method == "POST":
                    kwargs["json"] = payload
                else:
                    # For GET, append payload as query params if present
                    if payload:
                        kwargs["params"] = payload

                async with session.request(self._method, self._url, **kwargs) as response:
                    try:
                        resp_json = await response.json()
                    except Exception:
                        text = await response.text()
                        resp_json = {"raw": text}

                    if isinstance(resp_json, dict):
                        resp_json.setdefault("status_code", str(response.status))
                    return resp_json

        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError as e:
            self.logger.error(f"HTTP request to {self._url} timed out: {e}")
            return {"status_code": 408, "data": [], "error": "Request timed out"}
        except aiohttp.ClientError as e:
            self.logger.error(f"HTTP client error for {self._url}: {e}")
            return {"status_code": 503, "data": [], "error": f"Client error: {e}"}
        except Exception as e:
            self.logger.error(f"Unexpected error calling {self._url}: {e}")
            return {"status_code": 500, "data": [], "error": str(e)}

    @staticmethod
    def _extract_jobs(response: Any) -> List[Dict[str, Any]]:
        """
        Extract job list from an HTTP response.

        Handles common response shapes:
            - {"data": [...]}
            - {"data": {"data": [...]}}  (nested)
            - [{"title": ...}, ...]  (raw list)
            - List of page objects with "jobs" key
        """
        data = response

        # Unwrap nested "data" keys (up to 2 levels)
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        if isinstance(data, dict) and "data" in data:
            data = data["data"]

        # If data is a list of page-like objects with "jobs" key
        if isinstance(data, list) and data and isinstance(data[0], dict) and "jobs" in data[0]:
            jobs = []
            for page in data:
                page_jobs = page.get("jobs", [])
                if isinstance(page_jobs, list):
                    jobs.extend(page_jobs)
            return jobs

        if isinstance(data, list):
            return data

        if isinstance(data, dict):
            # Try common keys
            for key in ("jobs", "results", "listings", "positions"):
                if key in data and isinstance(data[key], list):
                    return data[key]

        return []