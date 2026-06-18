"""
Fallback Scraping Instructions
===============================
Legacy scraping paths (Part A + Part B) used when scrapers_config.yaml
is not found or contains no enabled adapters.

Part A: Scrape from company career pages using site_strategies/ JSON files.
Part B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google) via JobSpy.

This module is imported by main.py as a fallback path.
"""

import csv
import itertools
import logging
import os
import random
import re
import time
from typing import Any, Dict, List

import pandas as pd
from jobspy import scrape_jobs as _scrape_jobs

from app.pull_data import DataPuller


def error_logger_continue(error_msg: str) -> None:
    """Log an error and continue execution (non-fatal)."""
    print(error_msg)
    logging.error(error_msg)


def scrape_single_job_board(
    new_data: dict, company_url: str, company: str = ""
) -> list:
    """
    Extract individual job entries from a single job board response.

    Args:
        new_data: Response dict from DataPuller with 'data' key.
        company_url: Base URL for resolving relative links.
        company: Fallback company name if not present in each entry.

    Returns:
        List of normalized job dicts.
    """
    company_list: list = []
    for item in new_data["data"]:
        if not isinstance(item, dict):
            continue
        if not item.get("title") or item["title"] in ([None], ""):
            continue
        link = item.get("link")
        if link and not link.startswith(("http", "https")):
            link = f"{company_url}{link}"
        maker = {
            "company": item["company"]
            if item.get("company") is not None
            else company,
            "company_url": company_url,
            "title": item["title"],
            "flexibility": (
                item["flexibility"]
                if item.get("flexibility") is not None
                else "NA"
            ),
            "url": link,
            "source": new_data.get("source", ""),
        }
        if item.get("location") is not None:
            if re.search(r"location", item["location"], re.IGNORECASE):
                item["location"] = re.sub(
                    r"location", "", item["location"], flags=re.IGNORECASE
                )
            maker["location"] = item["location"]
        company_list.append(maker)
    return company_list


def scrape_multi_job_board(
    new_data: dict, company_url: str, company: str
) -> list:
    """
    Extract jobs from a multi-page job board response.

    Args:
        new_data: Response dict where each element in 'data' has a 'jobs' key.
        company_url: Base URL for resolving relative links.
        company: Company name to assign.

    Returns:
        List of normalized job dicts.
    """
    full_list: list = []
    adjusted_nd: dict = {
        "data": [],
        "status": 200,
        "success": True,
        "source": new_data["source"],
    }
    for item in new_data["data"]:
        adjusted_nd["data"].append(item["jobs"])
    full_list += scrape_single_job_board(adjusted_nd, company_url, company)
    return full_list


async def scrape_sites(
    i: dict, company_url: str, dp: DataPuller
) -> list | None:
    """
    Scrape a single site strategy using DataPuller.

    Args:
        i: Strategy payload dict with 'strategy', 'api_method', etc.
        company_url: Base company URL.
        dp: DataPuller instance.

    Returns:
        List of scraped job dicts, or None on failure.
    """
    new_data = await dp.scrape_data(
        i["strategy"], api_method=i["api_method"]
    )
    new_data["source"] = i["strategy"]["source"]
    if new_data["status_code"] != 200:
        print(
            f"Scraping data failed. Error code {new_data['status_code']}. "
            f"Error: {new_data.get('error', 'No error message provided.')}"
        )
        return None
    if "data" not in new_data:
        print(
            f"Warning: No 'data' key in response from {i['company']}. "
            f"Response: {new_data}"
        )
        return None
    if not new_data["data"]:
        return None

    try:
        first_item = new_data["data"][0]
        if isinstance(first_item, dict) and first_item.get("jobs") is not None:
            maker = scrape_multi_job_board(
                new_data, company_url, i["company"]
            )
        else:
            maker = scrape_single_job_board(
                new_data, company_url, i["company"]
            )
    except (KeyError, IndexError, TypeError) as e:
        print(f"Error: {e}\nNew Data: {new_data}")
        return None

    return maker


def scrape_jb(
    sn: str, l: str, rw: int, ho: int, **kwargs
) -> pd.DataFrame:
    """
    Scrape job board using the JobSpy library.

    Args:
        sn: Site name (e.g. "indeed", "linkedin").
        l: Location string.
        rw: Number of results wanted.
        ho: Hours old filter.
        **kwargs: Additional JobSpy keyword arguments.

    Returns:
        List of scraped job dicts from JobSpy.
    """
    allowed_keys = {
        "linkedin_fetch_description",
        "country_indeed",
        "google_search_term",
        "search_term",
    }
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

    jobs = _scrape_jobs(
        site_name=[sn],
        location=l,
        results_wanted=rw,
        hours_old=ho,
        **filtered_kwargs,
    )
    return jobs


async def _pipeline_stage_scrape_legacy(
    dp: DataPuller, user_preferences: dict, sites: dict, skip_db: bool, verbose: bool
) -> List[Dict]:
    """
    Legacy (fallback) scraping path: hardcoded Part A + Part B.
    Used when scrapers_config.yaml is not found or has no adapters.

    Part A: Scrape from company career pages using site_strategies/ JSON files.
    Part B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google) via JobSpy.

    Args:
        dp: DataPuller instance for scraping and DB persistence.
        user_preferences: Dict with user's target_cities and other preferences.
        sites: Dict with 'name' and 'site' keys loaded from JOB_SITES file.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.

    Returns:
        Combined list of raw scraped job dicts from both parts.
    """
    print("Using legacy scraping path (no scrapers_config.yaml found).")

    # ==========================================
    # PART A: Scrape from company career pages
    # ==========================================
    print("--- Company Board Scraping ---")
    site_strategies: list = []
    data: list = []

    for i in range(len(sites.get("name", []))):
        strategy_path = f"./site_strategies/{sites['name'][i]}.json"
        if not os.path.exists(strategy_path):
            print(f"Warning: strategy file not found: {strategy_path}")
            continue
        strategy = {
            "company": sites["name"][i],
            "site": sites["site"][i],
            "strategy": dp.load_site_strategies(strategy_path),
            "api_method": "",
        }
        strat = strategy["strategy"]
        strategy["api_method"] = (
            "extract-paginated"
            if strat.get("pagination") is not None
            else (
                "extract-js"
                if strat.get("js_config") is not None
                else "extract"
            )
        )
        if verbose:
            print(
                f"Company: {strategy['company']} | "
                f"API method: {strategy['api_method']}"
            )
        site_strategies.append(strategy)
    print(f"Loaded {len(site_strategies)} site strategies.")

    for i in site_strategies:
        company_url = i["strategy"].pop("company_url", None)
        print(f"Scraping {i['company']}...")
        if isinstance(i["strategy"]["url"], str):
            d = await scrape_sites(i, company_url, dp)
            if not d:
                continue
            if isinstance(d, list):
                data += d
            else:
                data.append(d)
        elif isinstance(i["strategy"]["url"], list):
            for j in i["strategy"]["url"]:
                new_payload = i
                new_payload["strategy"]["url"] = j
                d = await scrape_sites(new_payload, company_url, dp)
                if not d:
                    continue
                if isinstance(d, list):
                    data += d
                else:
                    data.append(d)
    print(f"Total jobs scraped from company boards: {len(data)}")

    if not skip_db:
        dp.load_scraped_data_to_db(data)

    # ==========================================
    # PART B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google)
    # ==========================================
    print("--- Job Board Scraping ---")
    job_board_list = ["indeed", "linkedin", "zip_recruiter", "google"]

    search_terms_file = os.getenv("SEARCH_TERMS", "")
    search_terms: list = []
    if search_terms_file and os.path.exists(search_terms_file):
        try:
            with open(search_terms_file, mode="r", encoding="utf-8") as f:
                reader = csv.reader(f)
                for row in reader:
                    if row:
                        search_terms.append(row[0])
        except Exception as e:
            error_logger_continue(f"Failed to load search terms: {e}")
    if not search_terms:
        print("Warning: No search terms found. Using default.")
        search_terms = ["Software Engineer"]

    target_cities = user_preferences.get("target_cities", [])
    if not target_cities:
        target_cities = ["Remote"]

    jobs: list = []
    requests_wanted = 200
    delay = {"min": 1, "max": 4}

    for board, st, location in itertools.product(
        job_board_list, search_terms, target_cities
    ):
        kwa: dict = {}
        if board == "google":
            kwa["google_search_term"] = st
        elif board == "indeed":
            kwa["search_term"] = st
            kwa["country_indeed"] = "USA"
        elif board == "linkedin":
            kwa["search_term"] = st
            kwa["linkedin_fetch_description"] = True
        else:
            kwa["search_term"] = st

        try:
            for i in scrape_jb(board, location, requests_wanted, 24, **kwa):
                if isinstance(i, dict):
                    job = {
                        "source": i.get("site"),
                        "title": i.get("title"),
                        "url": i.get("job_url"),
                        "link": i.get("job_url"),
                        "company": i.get("company"),
                        "pay": (
                            f"{i.get('min_amount', '')} - "
                            f"{i.get('max_amount', '')} "
                            f"{i.get('interval', '')}"
                        ).strip(),
                        "description": i.get("description"),
                        "city": i.get("city"),
                        "state": i.get("state"),
                        "flexibility": i.get("work_type", "NA"),
                        "location": i.get("location", ""),
                    }
                else:
                    job = {
                        "source": None,
                        "title": None,
                        "url": None,
                        "link": None,
                        "company": None,
                        "pay": "",
                        "description": None,
                        "city": None,
                        "state": None,
                        "flexibility": "NA",
                        "location": "",
                    }
                jobs.append(job)
        except Exception as e:
            error_logger_continue(
                f"Job board scrape failed for {board}/{st}/{location}: {e}"
            )

        time.sleep(random.uniform(delay["min"], delay["max"]))

    print(f"Total jobs scraped from job boards: {len(jobs)}")

    if not skip_db:
        dp.load_scraped_data_to_db(jobs)

    return data + jobs