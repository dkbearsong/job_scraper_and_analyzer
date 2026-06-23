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
import json
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


async def scrape_job_descriptions(
    jobs: list, dp: DataPuller, verbose: bool = False
) -> list:
    """
    Scrape individual job pages to extract descriptions for each job.

    For each job that has a URL, this function calls the microservice
    to scrape the job page and extract the description text. It tries
    common CSS selectors used by popular ATS platforms (Greenhouse,
    Lever, Ashby, etc.) and falls back to extracting the page body text.

    Args:
        jobs: List of job dicts with 'url' keys.
        dp: DataPuller instance for microservice calls.
        verbose: If True, print detailed debug output.

    Returns:
        The same list of job dicts with 'description' fields populated.
    """
    # Common CSS selectors for job descriptions across popular ATS platforms
    # Ordered from most specific/common to most generic
    description_selectors = [
        ".description",                          # Generic
        ".job-description",                      # Generic
        ".job_description",                      # Generic
        "#job-description",                      # Generic
        "[data-testid='job-description']",       # Greenhouse
        ".posting-description",                  # Greenhouse
        ".posting-description-content",          # Greenhouse variant
        ".content",                              # Generic
        ".job-details",                          # Generic
        ".job-body",                             # Generic
        ".job-posting-description",              # Generic
        "[class*='description']",                # Broad match
        "[class*='Description']",                # Broad match (case-sensitive)
        "main",                                  # Fallback to main content
        "article",                               # Fallback to article
        "[role='main']",                         # Fallback to ARIA main
    ]

    jobs_with_descriptions = 0
    for idx, job in enumerate(jobs):
        job_url = job.get("url")
        if not job_url:
            if verbose:
                print(f"  Job {idx + 1}/{len(jobs)}: No URL, skipping description scrape.")
            continue

        if verbose:
            print(f"  Scraping description for job {idx + 1}/{len(jobs)}: {job.get('title', 'Unknown')}")

        description = ""
        for selector in description_selectors:
            payload = {
                "url": job_url,
                "strategy": "selector",
                "selectors": {
                    "description": selector
                }
            }
            try:
                result = await dp.scrape_data(payload, api_method="extract")
                if result.get("status_code") == 200 and result.get("data"):
                    data = result["data"]
                    if isinstance(data, list) and len(data) > 0:
                        desc_text = data[0].get("description", "")
                        if desc_text and len(desc_text) > 50:
                            description = desc_text
                            if verbose:
                                print(f"    Found description with selector: '{selector}' ({len(desc_text)} chars)")
                            break
            except Exception as e:
                if verbose:
                    print(f"    Selector '{selector}' failed: {e}")
                continue

        if description:
            job["description"] = description
            jobs_with_descriptions += 1
        elif verbose:
            print(f"    No description found for {job_url}")

        # Small delay between requests to avoid overwhelming the microservice
        time.sleep(random.uniform(0.5, 1.5))

    print(f"Scraped descriptions for {jobs_with_descriptions}/{len(jobs)} jobs.")
    return jobs


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


async def scrape_job_descriptions_from_db(
    data: list, dp: DataPuller, verbose: bool = False
) -> list:
    """
    Query the database for jobs missing descriptions (job_summary IS NULL),
    then for each one check if a job_page_strategy file exists for the source.
    If it does, scrape the description using that strategy and update the DB record.

    Args:
        data: The in-memory list of scraped job dicts (to enrich with descriptions).
        dp: DataPuller instance for database queries and microservice calls.
        verbose: If True, print detailed debug output.

    Returns:
        The enriched data list with descriptions populated where possible.
    """
    # Build a lookup: source -> (id, link) from DB for jobs missing descriptions
    query = """
        SELECT j.id, j.link, j.source
        FROM job j
        WHERE j.job_summary IS NULL
    """
    try:
        rows = dp.pull_data_db(query)
    except Exception as e:
        print(f"Error querying DB for jobs without descriptions: {e}")
        return data

    if not rows:
        print("No jobs found in DB missing descriptions.")
        return data

    jobs_without_desc = []
    for row in rows:
        # rows returned as list of tuples or dicts; handle both
        if isinstance(row, dict):
            jobs_without_desc.append({
                "db_id": row.get("id"),
                "url": row.get("link"),
                "source": row.get("source"),
            })
        else:
            # tuple: (id, link, source) in order of SELECT
            jobs_without_desc.append({
                "db_id": row[0],
                "url": row[1],
                "source": row[2],
            })

    print(f"Found {len(jobs_without_desc)} jobs in DB missing descriptions.")

    # Build a mapping from source name -> normalized source name (for file lookup)
    # Source values in DB come from site_strategies/*.json "source" field.
    # We check for a matching strategy file in job_page_strategy/ by source name.
    job_page_strategy_dir = "./job_page_strategy"
    if not os.path.isdir(job_page_strategy_dir):
        print(f"Warning: job_page_strategy directory not found at {job_page_strategy_dir}. Skipping DB description scraping.")
        return data

    updated_count = 0
    for job in jobs_without_desc:
        db_id = job["db_id"]
        url = job["url"]
        source = job.get("source", "")

        if not url:
            if verbose:
                print(f"  DB job {db_id}: No URL, skipping.")
            continue

        if not source:
            if verbose:
                print(f"  DB job {db_id}: No source, skipping.")
            continue

        # Check if a job_page_strategy file exists for this source
        strategy_path = os.path.join(job_page_strategy_dir, f"{source}.json")
        if not os.path.exists(strategy_path):
            if verbose:
                print(f"  DB job {db_id}: No job_page_strategy for source '{source}', skipping.")
            continue

        try:
            with open(strategy_path, "r") as f:
                strategy = json.load(f)
        except Exception as e:
            print(f"  Error loading strategy {strategy_path}: {e}")
            continue

        if verbose:
            print(f"  Scraping description for DB job {db_id} ({source}) via {strategy_path}")

        # Prepare the payload using the DB record's URL
        payload = dict(strategy)
        # Replace the URL template placeholder with the actual job URL if needed
        if payload.get("url") == "{url}":
            payload["url"] = url

        api_method = (
            "extract-js"
            if payload.get("js_config") is not None
            else "extract"
        )

        description = ""
        try:
            result = await dp.scrape_data(payload, api_method=api_method)
            if result.get("status_code") == 200 and result.get("data"):
                data_result = result["data"]
                if isinstance(data_result, list) and len(data_result) > 0:
                    # Strategy may use 'description' or other custom selector field
                    desc_key = next(
                        (k for k in ("description", "summary", "job-summary")
                         if data_result[0].get(k)),
                        None
                    )
                    if desc_key:
                        desc_text = data_result[0].get(desc_key, "")
                    else:
                        # If no known key, take the first non-empty string value
                        desc_text = next(
                            (v for v in data_result[0].values()
                             if isinstance(v, str) and len(v) > 50),
                            ""
                        )
                    if desc_text and len(desc_text) > 50:
                        description = desc_text
        except Exception as e:
            print(f"  Error scraping description for DB job {db_id}: {e}")
            continue

        if description:
            # Update the DB record
            try:
                dp.conn.update("job", {"job_summary": description}, {"id": db_id}, dbname=dp.dbname)
                updated_count += 1
                if verbose:
                    print(f"    Updated DB job {db_id} with description ({len(description)} chars)")
            except Exception as e:
                print(f"    Failed to update DB job {db_id}: {e}")

            # Enrich the in-memory data list: match by URL
            for item in data:
                if item.get("url") == url:
                    item["description"] = description
                    break
        else:
            if verbose:
                print(f"    No description found for DB job {db_id} ({url})")

        # Small delay between requests
        time.sleep(random.uniform(0.5, 1.5))

    print(f"Scraped and saved descriptions for {updated_count}/{len(jobs_without_desc)} DB jobs.")
    return data


async def _pipeline_stage_scrape_legacy(
    dp: DataPuller, user_preferences: dict, sites: dict, skip_db: bool, verbose: bool,
    enable_part_b: bool = False
) -> List[Dict]:
    """
    Legacy (fallback) scraping path: hardcoded Part A + optional Part B.
    Used when scrapers_config.yaml is not found or has no adapters.

    Part A: Scrape from company career pages using site_strategies/ JSON files.
            After scraping job listings, each job's individual page is also
            scraped to extract the full description.
    Part B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google) via JobSpy.
            OFF by default; toggle on via ``enable_fallback_part_b: true`` in
            ``user_preferences.yaml``.

    Args:
        dp: DataPuller instance for scraping and DB persistence.
        user_preferences: Dict with user's target_cities and other preferences.
        sites: Dict with 'name' and 'site' keys loaded from JOB_SITES file.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
        enable_part_b: If True, run Part B (job board scraping via JobSpy).

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

    # ── Load jobs into the database first (before scraping descriptions) ──
    if data and not skip_db:
        print("--- Loading Jobs into Database ---")
        dp.load_scraped_data_to_db(data)
    elif not skip_db:
        print("No company board jobs to load into DB.")

    # ── Then scrape descriptions from DB for jobs that are missing them ──
    if not skip_db:
        print("--- Scraping Missing Job Descriptions from DB (via job_page_strategy/) ---")
        data = await scrape_job_descriptions_from_db(data, dp, verbose=verbose)
    else:
        print("--- Skipping DB description scraping (skip_db=True) ---")
        # Fall back to the old in-memory description scraping if no DB
        if data:
            print("--- Scraping Job Descriptions (in-memory fallback) ---")
            data = await scrape_job_descriptions(data, dp, verbose=verbose)
        else:
            print("No company board jobs to scrape descriptions for.")

    # ==========================================
    # PART B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google)
    # ==========================================
    jobs: list = []
    if enable_part_b:
        print("--- Job Board Scraping (Part B enabled) ---")
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
    else:
        print("--- Job Board Scraping (Part B disabled — skipping) ---")

    return data + jobs
