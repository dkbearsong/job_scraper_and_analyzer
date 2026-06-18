"""
Pluggable Scraper Adapter System for Pipeline Stage 1.

Provides a standardized interface for scraping job data from various sources:
- Python modules/libraries (primary extension mechanism)
- HTTP/API endpoints (assumes correctly formatted responses)

Usage:
    Define a config in scrapers_config.yaml, and the AdapterLoader will
    dynamically import, configure, and run each enabled adapter.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


# ─────────────────────────────────────────────────────────
# Standardized Job Data Schema
# ─────────────────────────────────────────────────────────

class JobData(TypedDict, total=False):
    """
    Standardized raw output schema for all scraper adapters.

    Required fields:
        source: Origin identifier (e.g. "indeed", "company_board", "custom_api")
        title: Job title (must be a non-empty string)

    Recommended fields (defaults auto-filled if missing):
        url: Job listing URL
        company: Company name
        flexibility: Work type ("remote", "hybrid", "onsite", "NA")
        pay: Pay range string (e.g. "$80k-$120k")
        location: Full location string (e.g. "Austin, TX")

    Optional fields:
        link: Alias for url (for backward compatibility)
        description: Job description text
        city: City name
        state: State abbreviation
        company_url: Company career page URL
    """
    # Required
    source: str
    title: str
    # Recommended
    url: Optional[str]
    company: Optional[str]
    flexibility: Optional[str]
    pay: Optional[str]
    location: Optional[str]
    # Optional
    link: Optional[str]
    description: Optional[str]
    city: Optional[str]
    state: Optional[str]
    company_url: Optional[str]


# Field classifications for validation
REQUIRED_FIELDS: List[str] = ["title", "company"]

RECOMMENDED_FIELDS: List[str] = ["source", "url", "description", "flexibility", "pay", "location"]

RECOMMENDED_DEFAULTS: Dict[str, Any] = {
    "source": "unknown",
    "url": None,
    "description": None,
    "flexibility": "NA",
    "pay": "",
    "location": "",
}


# ─────────────────────────────────────────────────────────
# Scraper Adapter Abstract Base Class
# ─────────────────────────────────────────────────────────

class ScraperAdapter(ABC):
    """
    Abstract base class for all scraper adapters.

    Every adapter must implement:
        - get_name(): Return a human-readable name for logging
        - configure(config): Accept a dict of adapter-specific configuration
        - scrape(): Execute the scraping and return List[JobData]
    """

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._name: str = self.__class__.__name__
        self.logger = logging.getLogger(f"scraper.{self._name}")

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for this adapter (used in logs)."""
        ...

    @abstractmethod
    def configure(self, config: Dict[str, Any]) -> None:
        """
        Configure the adapter with source-specific settings.

        Args:
            config: Dict of adapter-specific configuration from scrapers_config.yaml
        """
        ...

    @abstractmethod
    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Execute the scraping operation.

        Returns:
            List of job dicts conforming to the JobData schema.
            Each dict must at minimum contain 'title' and 'company' keys.
        """
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.get_name()}')>"