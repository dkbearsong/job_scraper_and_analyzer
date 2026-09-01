"""
Fallback Scraping Instructions
===============================
Legacy scraping paths (Part A + Part B) used when scrapers_config.yaml
is not found or contains no enabled adapters.

Part A: Scrape from company career pages using site_strategies/ JSON files.
Part B: Scrape from job boards (Indeed, LinkedIn, ZipRecruiter, Google) via JobSpy.

This module is imported by main.py as a fallback path.
"""

import asyncio
import csv
import itertools
import json
import logging
import os
import random
import re
import time as time_module
import glob
from urllib.parse import urlparse
from typing import Any, Callable, Dict, List, Optional
from datetime import datetime
from app.prompt_injection_defender import sanitize_untrusted_text

import pandas as pd
from jobspy import scrape_jobs as _scrape_jobs

from app.pull_data import DataPuller


def parse_pay_range(val: str) -> str:
    """
    Intelligently parses a salary/pay range string, extracting the minimum,
    maximum, and unit (hourly or annual).
    E.g.
      '270K - 290k Annually' -> '$270k-290k Annual'
      '$100-$125k'           -> '$100k-125k Annual'
      '$85k-96k yearly'      -> '$85k-96k Annual'
      '$60-75 hourly'        -> '$60-75 hour'
      '24-28 hour'           -> '$24-28 hour'
      '$25/hour'             -> '$25-25 hour'
    Anything else gets dropped (returns empty string).
    """
    if not val or not isinstance(val, str):
        return ""
    
    val_lower = val.lower().strip()
    
    # 1. Determine unit: hour vs Annual
    unit = None
    if any(x in val_lower for x in ["hour", "hr", "hourly", "/h"]):
        unit = "hour"
    elif any(x in val_lower for x in ["annual", "annually", "year", "yearly", "yr"]):
        unit = "Annual"
        
    # 2. Extract numbers
    cleaned = val_lower.replace(",", "")
    
    # Matches numbers optionally followed by 'k'
    pattern = r'(\d+(?:\.\d+)?)\s*(k)?'
    matches = re.findall(pattern, cleaned)
    
    if not matches:
        return ""
        
    numbers = []
    has_k = False
    
    for num_str, k_suffix in matches:
        try:
            num = float(num_str)
            if num.is_integer():
                num = int(num)
                
            if num >= 1000:
                num = num / 1000
                if num.is_integer():
                    num = int(num)
                has_k = True
                
            if k_suffix:
                has_k = True
                
            numbers.append(num)
        except ValueError:
            continue
            
    if not numbers:
        return ""
        
    # Determine unit if not explicitly found
    if not unit:
        if has_k or any(n >= 100 for n in numbers):
            unit = "Annual"
        else:
            unit = "hour"
            
    # Format min and max
    if len(numbers) >= 2:
        min_val, max_val = numbers[0], numbers[1]
    else:
        min_val = max_val = numbers[0]
        
    if unit == "Annual":
        return f"${min_val}k-{max_val}k Annual"
    else:
        return f"${min_val}-{max_val} hour"



# ── Per-domain exponential backoff state ──
# {domain: {"next_allowed": float, "current_delay": float, "retry_count": int, "success_streak": int}}
_domain_backoff: Dict[str, Dict] = {}

# Default backoff parameters
_BACKOFF_MAX_RETRIES: int = 5
_BACKOFF_BASE_DELAY: float = 1.0
_BACKOFF_MAX_DELAY: float = 120.0
_BACKOFF_FACTOR: float = 2.0
_BACKOFF_SUCCESS_RESET: int = 3
# ── Global scraping statistics tracking ──
_global_scraped_by_source: Dict[str, int] = {}
_global_failed_by_source: Dict[str, int] = {}
_processed_job_keys: set = set()


def _print_and_log_stats() -> None:
    """Print cumulative scraping statistics by source and save them to a log file."""
    if not _global_scraped_by_source:
        return

    print("\n--- Scraping Statistics by Source ---")
    for src in sorted(_global_scraped_by_source.keys()):
        total = _global_scraped_by_source[src]
        failed = _global_failed_by_source.get(src, 0)
        pct = (failed / total * 100) if total > 0 else 0.0
        print(f"  {src}: {failed} / {total} failed to scrape descriptions ({pct:.1f}%)")
    
    overall_total = sum(_global_scraped_by_source.values())
    overall_failed = sum(_global_failed_by_source.values())
    overall_pct = (overall_failed / overall_total * 100) if overall_total > 0 else 0.0
    print(f"  Total: {overall_failed} / {overall_total} failed to scrape descriptions ({overall_pct:.1f}%)")
    print("------------------------------------")

    # Save to a separate log file
    stats_filename = f"scraping_stats_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    stats_filepath = os.path.join("logs", "stats", stats_filename)
    try:
        os.makedirs(os.path.join("logs", "stats"), exist_ok=True)
        stats_data = {
            "timestamp": datetime.now().isoformat(),
            "stats_by_source": {
                src: {
                    "failed": _global_failed_by_source.get(src, 0),
                    "total": total,
                    "failure_rate_pct": round((_global_failed_by_source.get(src, 0) / total * 100), 2) if total > 0 else 0.0
                }
                for src, total in _global_scraped_by_source.items()
            },
            "overall": {
                "failed": overall_failed,
                "total": overall_total,
                "failure_rate_pct": round(overall_pct, 2)
            }
        }
        with open(stats_filepath, "w", encoding="utf-8") as f:
            json.dump(stats_data, f, indent=2)
        print(f"  - Saved scraping statistics to {stats_filepath}")
    except Exception as e:
        print(f"Warning: failed to write statistics log file '{stats_filepath}': {e}")


_db_lock = asyncio.Lock()


async def _safe_db_call(fn: Callable, *args, **kwargs):
    """
    Execute a blocking psycopg2 database operation in a thread pool
    while holding a global asyncio lock to prevent concurrent access
    to the same psycopg2 connection.
    """
    async with _db_lock:
        return await asyncio.to_thread(fn, *args, **kwargs)


def _extract_domain(url: str) -> str:
    """Extract the domain (hostname) from a URL for rate-limit tracking."""
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return "unknown"


_js_scrape_lock = asyncio.Lock()
_host_server_errors: List[Dict[str, Any]] = []


def record_host_server_error(
    company: str = "Unknown",
    endpoint: str = "",
    target_url: str = "",
    status_code: Any = "UNKNOWN",
    error: str = "",
    exception: str = "",
    raw_response: str = "",
    stage: str = "Fallback Scraping"
) -> None:
    """Record an error response or network failure encountered from the scraping host server."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "stage": stage,
        "company": company or "Unknown",
        "endpoint": endpoint or "",
        "target_url": target_url or "",
        "status_code": status_code,
        "error": error or "",
        "exception": exception or "",
        "raw_response": str(raw_response)[:1000] if raw_response else "",
    }
    _host_server_errors.append(entry)


def get_host_server_errors() -> List[Dict[str, Any]]:
    """Return all recorded host server errors."""
    return list(_host_server_errors)


def clear_host_server_errors() -> None:
    """Clear recorded host server errors for a new run."""
    _host_server_errors.clear()


def format_host_server_error_report() -> str:
    """Format a detailed diagnostic report of all host server errors."""
    if not _host_server_errors:
        return "No host server error messages recorded."

    host = os.getenv("SCRAPER_HOST", "localhost")
    port = os.getenv("SCRAPER_PORT", "5052")
    lines = [
        "=" * 80,
        "HOST SERVER FAILURE REPORT",
        f"Host Server Target: http://{host}:{port}",
        f"Total Host Server Errors Recorded: {len(_host_server_errors)}",
        "=" * 80,
    ]
    for idx, err in enumerate(_host_server_errors, 1):
        lines.append(f"\n[{idx}] Source / Company: {err['company']} ({err['stage']})")
        if err.get("endpoint"):
            lines.append(f"    Host Endpoint:    {err['endpoint']}")
        if err.get("target_url"):
            lines.append(f"    Target Scrape URL:{err['target_url']}")
        lines.append(f"    HTTP Status Code: {err['status_code']}")
        if err.get("error"):
            lines.append(f"    Error Message:    {err['error']}")
        if err.get("exception"):
            lines.append(f"    Exception:        {err['exception']}")
        if err.get("raw_response"):
            lines.append(f"    Response Snippet: {err['raw_response']}")
    lines.append("\n" + "=" * 80)
    return "\n".join(lines)


def _is_js_request(args, kwargs) -> bool:
    api_method = kwargs.get("api_method", "")
    if api_method in ("extract-js", "extract-paginated"):
        return True
    
    # Check payload if present
    if args and isinstance(args[0], dict):
        payload = args[0]
        if payload.get("js_config") is not None:
            return True
        if payload.get("pagination") is not None:
            return True
            
    return False


def classify_strategy_type(strategy: dict, api_method: str = "") -> str:
    """
    Verify and classify a scraping strategy profile into one of:
    - 'non-javascript': Single static HTML page (no JS, no pagination, single URL)
    - 'javascript': Single JS-rendered page (uses JS, no pagination, single URL)
    - 'paginated': Paginated search/board (multi-page, single starting URL)
    - 'multi-link': Multiple starting URLs (URL list with > 1 URLs)

    Queue priority order: [non-javascript, javascript, paginated, multi-link]
    """
    urls = strategy.get("url")
    if isinstance(urls, list) and len(urls) > 1:
        return "multi-link"

    if strategy.get("pagination") is not None or api_method == "extract-paginated":
        return "paginated"

    is_js = (
        api_method == "extract-js"
        or strategy.get("js_config") is not None
        or (strategy.get("pagination") and strategy.get("pagination", {}).get("use_js"))
    )
    if is_js:
        return "javascript"

    return "non-javascript"


async def _request_with_domain_backoff(
    request_fn: Callable,
    *args,
    url: str = "",
    max_retries: int = _BACKOFF_MAX_RETRIES,
    base_delay: float = _BACKOFF_BASE_DELAY,
    max_delay: float = _BACKOFF_MAX_DELAY,
    backoff_factor: float = _BACKOFF_FACTOR,
    success_reset_threshold: int = _BACKOFF_SUCCESS_RESET,
    **kwargs
) -> dict:
    """
    Make an async request with per-domain exponential backoff that persists
    across all subsequent requests to the same domain.

    Before making the request, checks if the domain is in a cooldown period
    and waits if needed.  On a 429 response, retries with exponential backoff
    and updates the domain's cooldown state so that *all* subsequent requests
    to the same domain will wait.

    Args:
        request_fn: Async callable that returns a dict with a 'status_code' key.
        url: The URL being requested (used to extract the domain).
        max_retries: Max retry attempts per request.
        base_delay: Initial backoff delay in seconds.
        max_delay: Maximum backoff delay in seconds.
        backoff_factor: Multiplier for delay after each 429.
        success_reset_threshold: Number of consecutive successes before
                                 resetting the domain's backoff state.
        *args, **kwargs: Passed through to request_fn.

    Returns:
        Response dict from request_fn (may still be a 429 or error dict if retries exhausted).
    """
    domain = _extract_domain(url)
    state = _domain_backoff.setdefault(domain, {
        "next_allowed": 0.0,
        "current_delay": base_delay,
        "retry_count": 0,
        "success_streak": 0,
    })

    # ── Pre-request cooldown check ──
    now = time_module.time()
    if now < state["next_allowed"]:
        wait = state["next_allowed"] - now
        print(f"  [Rate Limit] Domain '{domain}' in cooldown. Waiting {wait:.1f}s...")
        await asyncio.sleep(wait)

    result = {}
    for attempt in range(max_retries + 1):
        try:
            if _is_js_request(args, kwargs):
                async with _js_scrape_lock:
                    result = await request_fn(*args, **kwargs)
            else:
                result = await request_fn(*args, **kwargs)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            result = {
                "status_code": "500",
                "data": [],
                "error": "Request exception during execution",
                "exception": repr(e),
                "url": url,
            }

        if not isinstance(result, dict):
            result = {
                "status_code": "500",
                "data": [],
                "error": f"Invalid return type from request: {type(result)}",
                "raw": repr(result),
                "url": url,
            }

        status_raw = result.get("status_code", 200)
        try:
            status = int(status_raw)
        except (ValueError, TypeError):
            status = 500

        if status == 429:
            state["retry_count"] += 1
            state["success_streak"] = 0

            # Use Retry-After header if present, else exponential backoff
            retry_after = result.get("retry_after") or result.get("headers", {}).get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = min(state["current_delay"] * (backoff_factor ** attempt), max_delay)
            else:
                delay = min(state["current_delay"] * (backoff_factor ** attempt), max_delay)

            state["current_delay"] = min(delay * backoff_factor, max_delay)
            state["next_allowed"] = time_module.time() + delay

            if attempt < max_retries:
                print(f"  [Rate Limit] 429 on '{domain}' (attempt {attempt + 1}/{max_retries}). "
                      f"Backing off {delay:.1f}s...")
                await asyncio.sleep(delay)
            else:
                print(f"  [Rate Limit] 429 on '{domain}' — exhausted {max_retries} retries. "
                      f"Giving up on this request.")
        else:
            # Success or non-429 error
            if status == 200:
                state["success_streak"] += 1
                if state["success_streak"] >= success_reset_threshold:
                    state["current_delay"] = base_delay
                    state["retry_count"] = 0
            return result

    return result


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
            "description": item.get("description"),
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
        
        # Parse pay range if present
        pay_raw = item.get("pay_range") or item.get("pay") or item.get("pay_rate")
        pay_parsed = parse_pay_range(pay_raw) if pay_raw else ""
        maker["pay"] = pay_parsed
        maker["pay_rate"] = pay_parsed

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
    strategy_url = i["strategy"].get("url", company_url)
    company_name = i.get("company", "Unknown")
    api_method = i.get("api_method", "extract")
    host = os.getenv("SCRAPER_HOST", "localhost")
    port = os.getenv("SCRAPER_PORT", "5052")
    endpoint = f"http://{host}:{port}/{api_method}"
    try:
        new_data = await _request_with_domain_backoff(
            dp.scrape_data,
            i["strategy"],
            url=strategy_url,
            api_method=api_method,
        )
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=strategy_url,
            status_code="408",
            error=f"Scraping request timed out after {micro_timeout}s",
            exception="asyncio.TimeoutError",
            stage="Part A (Company Board)"
        )
        print(f"Scraping timed out for {company_name} after {micro_timeout}s. URL: {strategy_url} (Continuing with remaining strategies...)")
        return None
    except Exception as e:
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=strategy_url,
            status_code="EXCEPTION",
            error="Exception during scraping request",
            exception=repr(e),
            stage="Part A (Company Board)"
        )
        print(f"Scraping exception for {company_name}: {e}. Continuing to next strategy...")
        return None

    if not isinstance(new_data, dict):
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=strategy_url,
            status_code="INVALID_RESPONSE",
            error=f"Expected response dict, got {type(new_data)}",
            raw_response=repr(new_data),
            stage="Part A (Company Board)"
        )
        return None

    new_data["source"] = i["strategy"].get("source", company_name)
    status_code = str(new_data.get("status_code", "200"))
    if status_code != "200":
        error_msg = new_data.get("error", "")
        exception_msg = new_data.get("exception", "")
        raw_text = new_data.get("raw", "")
        target_url = new_data.get("url", strategy_url)
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=target_url,
            status_code=status_code,
            error=error_msg,
            exception=exception_msg,
            raw_response=raw_text,
            stage="Part A (Company Board)"
        )
        details_parts = []
        if target_url:
            details_parts.append(f"URL: {target_url}")
        if error_msg:
            details_parts.append(f"Error: {error_msg}")
        if exception_msg:
            details_parts.append(f"Exception: {exception_msg}")
        if raw_text:
            details_parts.append(f"Response body: {raw_text[:500]}")
        if not details_parts:
            details_parts.append("No error message provided.")
        print(
            f"Scraping data failed for {company_name}. Status code {status_code}. "
            + " | ".join(details_parts)
            + " (Continuing with remaining strategies...)"
        )
        return None
    if "data" not in new_data:
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=strategy_url,
            status_code="NO_DATA_KEY",
            error="Response missing 'data' key",
            raw_response=repr(new_data),
            stage="Part A (Company Board)"
        )
        print(
            f"Warning: No 'data' key in response from {company_name}. "
            f"Response: {new_data}"
        )
        return None
    if not new_data["data"]:
        return None

    try:
        first_item = new_data["data"][0]
        if isinstance(first_item, dict) and first_item.get("jobs") is not None:
            maker = scrape_multi_job_board(
                new_data, company_url, company_name
            )
        else:
            maker = scrape_single_job_board(
                new_data, company_url, company_name
            )
    except (KeyError, IndexError, TypeError) as e:
        record_host_server_error(
            company=company_name,
            endpoint=endpoint,
            target_url=strategy_url,
            status_code="PARSE_ERROR",
            error=f"Error parsing job data structure: {e}",
            exception=repr(e),
            raw_response=repr(new_data.get("data")),
            stage="Part A (Company Board)"
        )
        print(f"Error parsing data for {company_name}: {e}\nNew Data: {new_data}")
        return None

    return maker


async def scrape_job_descriptions(
    jobs: list, dp: DataPuller, verbose: bool = False, concurrency: int = 5
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
        concurrency: Concurrency limit for scraping.

    Returns:
        The same list of job dicts with 'description' fields populated.
    """
    from collections import defaultdict

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

    concurrency_limit = int(os.getenv("SCRAPER_CONCURRENCY", concurrency))
    global_semaphore = asyncio.Semaphore(concurrency_limit)

    jobs_with_descriptions = 0
    skipped_jobs = []

    skipped_jobs_lock = asyncio.Lock()
    counter_lock = asyncio.Lock()

    # Pre-filter jobs that have no URL
    jobs_with_url = []
    for idx, job in enumerate(jobs):
        job_url = job.get("url")
        if not job_url:
            if verbose:
                print(f"  Job {idx + 1}/{len(jobs)}: No URL, skipping description scrape.")
            skipped_jobs.append({
                "id": job.get("id"),
                "job_name": job.get("title") or "",
                "company_name": job.get("company") or "",
                "source": job.get("source") or "",
                "date": datetime.now().isoformat(),
            })
        else:
            jobs_with_url.append(job)

    # Group jobs by domain
    domain_to_jobs = defaultdict(list)
    for job in jobs_with_url:
        domain = _extract_domain(job["url"])
        domain_to_jobs[domain].append(job)

    async def scrape_single_job(job: dict, idx: int, total: int):
        nonlocal jobs_with_descriptions
        job_url = job["url"]
        if verbose:
            print(f"  Scraping description for job {idx + 1}/{total}: {job.get('title', 'Unknown')}")

        # Construct selectors payload to fetch ALL selectors in a single request
        selectors_dict = {f"desc_{i}": selector for i, selector in enumerate(description_selectors)}
        payload = {
            "url": job_url,
            "strategy": "selector",
            "selectors": selectors_dict
        }

        description = ""
        async with global_semaphore:
            try:
                result = await _request_with_domain_backoff(
                    dp.scrape_data,
                    payload,
                    url=job_url,
                    api_method="extract",
                )
                if result.get("status_code") == 200 and result.get("data"):
                    data_list = result["data"]
                    if isinstance(data_list, list) and len(data_list) > 0:
                        first_item = data_list[0]
                        # Find the first matching selector description text > 50 chars
                        for i, selector in enumerate(description_selectors):
                            desc_text = first_item.get(f"desc_{i}", "")
                            if desc_text and len(desc_text) > 50:
                                description = desc_text
                                if verbose:
                                    print(f"    Found description with selector: '{selector}' ({len(desc_text)} chars)")
                                break
                        
                        # Parse pay range if present in scraped page
                        scraped_pay = first_item.get("pay_range") or first_item.get("pay") or first_item.get("pay_rate")
                        if scraped_pay:
                            pay_parsed = parse_pay_range(scraped_pay)
                            job["pay"] = pay_parsed
                            job["pay_rate"] = pay_parsed
            except Exception as e:
                if verbose:
                    print(f"    Request failed for {job_url}: {e}")

        if description:
            job["description"] = sanitize_untrusted_text(description)
            async with counter_lock:
                jobs_with_descriptions += 1
        else:
            if verbose:
                print(f"    No description found for {job_url}")
            log_entry = {
                "id": job.get("id"),
                "job_name": job.get("title") or "",
                "company_name": job.get("company") or "",
                "source": job.get("source") or "",
                "date": datetime.now().isoformat(),
            }
            async with skipped_jobs_lock:
                skipped_jobs.append(log_entry)

    async def process_domain_jobs(domain: str, jobs_list: list):
        for idx, job in enumerate(jobs_list):
            await scrape_single_job(job, idx, len(jobs_list))
            # Polite delay between requests to the same domain (async-friendly)
            await asyncio.sleep(random.uniform(0.5, 1.5))

    # Process all domains concurrently
    if domain_to_jobs:
        tasks = [process_domain_jobs(domain, jobs_list) for domain, jobs_list in domain_to_jobs.items()]
        await asyncio.gather(*tasks)

    # Log skipped jobs if any
    if skipped_jobs:
        log_filename = f"skipped_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join("logs", "skipped", log_filename)
        try:
            os.makedirs(os.path.join("logs", "skipped"), exist_ok=True)
            with open(log_filepath, "w", encoding="utf-8") as f:
                json.dump(skipped_jobs, f, indent=2)
            print(f"Logged {len(skipped_jobs)} skipped job(s) to {log_filepath}")
        except Exception as e:
            print(f"Warning: failed to write skip log file '{log_filepath}': {e}")

    # Remove jobs that don't have descriptions
    jobs = [j for j in jobs if j.get("description")]

    print(f"Scraped descriptions for {jobs_with_descriptions} jobs.")
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
    data: list, dp: DataPuller, verbose: bool = False, concurrency: int = 5
) -> list:
    """
    Query the database for jobs missing descriptions (job_summary IS NULL),
    then for each one check if a job_page_strategy file exists for the source.
    If not, dynamically generates and tests one.
    Scrapes the description using that strategy and updates the DB record.

    Args:
        data: The in-memory list of scraped job dicts (to enrich with descriptions).
        dp: DataPuller instance for database queries and microservice calls.
        verbose: If True, print detailed debug output.
        concurrency: Concurrency limit for scraping.

    Returns:
        The enriched data list with descriptions populated where possible.
    """
    data = await _scrape_job_descriptions_from_db_impl(data, dp, verbose, concurrency)
    return data


async def _scrape_job_descriptions_from_db_impl(data: list, dp: DataPuller, verbose: bool = False, concurrency: int = 5) -> list:
    from collections import defaultdict
    query = """
        SELECT j.id, j.link, j.source, j.job_name, c.company_name, j.date_added
        FROM job j
        LEFT JOIN company c ON j.company_id = c.id
        WHERE j.job_summary IS NULL AND (j.skip IS NULL OR j.skip = FALSE)
    """
    try:
        rows = await _safe_db_call(dp.pull_data_db, query)
    except Exception as e:
        print(f"Error querying DB for jobs without descriptions: {e}")
        return data

    if not rows:
        print("No jobs found in DB missing descriptions.")
        return data

    jobs_without_desc = []
    for row in rows:
        if isinstance(row, dict):
            jobs_without_desc.append({
                "db_id": row.get("id"),
                "url": row.get("link"),
                "source": row.get("source"),
                "title": row.get("job_name"),
                "company": row.get("company_name"),
                "date": row.get("date_added"),
            })
        else:
            jobs_without_desc.append({
                "db_id": row[0],
                "url": row[1],
                "source": row[2],
                "title": row[3] if len(row) > 3 else "",
                "company": row[4] if len(row) > 4 else "",
                "date": row[5] if len(row) > 5 else None,
            })

    print(f"Found {len(jobs_without_desc)} active job(s) in DB missing descriptions (combining newly scraped and pre-existing DB records).")

    job_page_strategy_dir = "./job_page_strategy"
    if not os.path.isdir(job_page_strategy_dir):
        print(f"Warning: job_page_strategy directory not found at {job_page_strategy_dir}. Skipping DB description scraping.")
        return data

    def find_strategy_file(source_name: str) -> str | None:
        if not source_name:
            return None
        source_lower = source_name.lower()
        for filepath in glob.glob(os.path.join(job_page_strategy_dir, "*.json")):
            filename = os.path.basename(filepath)
            name = filename[:-5].lower() # remove .json
            # Exact match, e.g. "Dice" -> "Dice.json"
            if name == source_lower:
                return filepath
            # Domain match, e.g. "Dice" -> "dice.com.json"
            if "." in name:
                domain_part = name.split('.')[0]
                if domain_part == source_lower:
                    return filepath
            # Inverse domain match, e.g. "dice.com" -> "Dice.json"
            if "." in source_lower:
                source_domain_part = source_lower.split('.')[0]
                if source_domain_part == name:
                    return filepath
        return None

    concurrency_limit = int(os.getenv("SCRAPER_CONCURRENCY", concurrency))
    global_semaphore = asyncio.Semaphore(concurrency_limit)

    updated_count = 0
    skipped_jobs = []
    job_ids_to_skip = []
    enriched_descriptions = {} # url -> description
    enriched_pays = {} # url -> pay
    failed_strategy_domains = set() # domains that failed strategy generation (avoid re-generating)
    processed_count = 0
    total_db_jobs = len(jobs_without_desc)
 
    skipped_jobs_lock = asyncio.Lock()
    job_ids_to_skip_lock = asyncio.Lock()
    updated_count_lock = asyncio.Lock()
    enriched_descriptions_lock = asyncio.Lock()
    enriched_pays_lock = asyncio.Lock()
    failed_strategy_domains_lock = asyncio.Lock()
    processed_count_lock = asyncio.Lock()

    # Group jobs by domain
    domain_to_jobs = defaultdict(list)
    for job in jobs_without_desc:
        url = job["url"]
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.lower()
            if domain.startswith("www."):
                domain = domain[4:]
        except Exception:
            domain = job.get("source", "").lower() or "unknown"
        domain_to_jobs[domain].append(job)

    async def scrape_single_db_job(job: dict):
        nonlocal updated_count, processed_count
        db_id = job["db_id"]
        url = job["url"]
        source = job.get("source", "")
        job_name = job.get("title", "")
        company_name = job.get("company", "")
        job_date = job.get("date")

        if not url or not source:
            if verbose:
                print(f"  DB job {db_id}: Missing URL or source, skipping.")
            log_entry = {
                "id": db_id,
                "job_name": job_name,
                "company_name": company_name,
                "source": source,
                "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
                "url": url,
            }
            async with skipped_jobs_lock:
                skipped_jobs.append(log_entry)
            async with job_ids_to_skip_lock:
                job_ids_to_skip.append(db_id)
            return

        # Look up strategy file matching the source
        strategy_path = find_strategy_file(source)
        if not strategy_path:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = source.lower()
            
            # If strategy generation previously failed for this domain, skip quickly
            if domain in failed_strategy_domains:
                log_entry = {
                    "id": db_id,
                    "job_name": job_name,
                    "company_name": company_name,
                    "source": source,
                    "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
                    "url": url,
                }
                async with skipped_jobs_lock:
                    skipped_jobs.append(log_entry)
                async with job_ids_to_skip_lock:
                    job_ids_to_skip.append(db_id)
                async with processed_count_lock:
                    processed_count += 1
                    if processed_count % 25 == 0 or processed_count == total_db_jobs:
                        print(f"  [DB Descriptions] Progress: {processed_count}/{total_db_jobs} jobs processed ({updated_count} descriptions updated, {len(skipped_jobs)} skipped)...")
                return

            print(f"  No strategy file found for source '{source}'. Generating strategy for domain '{domain}' using {url}...")
            # Call generation function
            success = await dp.generate_and_test_strategy(destination_link=url, job_id=db_id, domain_name=domain)
            if success:
                strategy_path = find_strategy_file(source)
                if not strategy_path:
                    strategy_path = os.path.join(job_page_strategy_dir, f"{domain}.json")
            else:
                async with failed_strategy_domains_lock:
                    failed_strategy_domains.add(domain)
                print(f"  Failed to generate a working strategy for DB job {db_id} ({url}). Cached failure for domain '{domain}'.")
                log_entry = {
                    "id": db_id,
                    "job_name": job_name,
                    "company_name": company_name,
                    "source": source,
                    "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
                    "url": url,
                }
                async with skipped_jobs_lock:
                    skipped_jobs.append(log_entry)
                async with job_ids_to_skip_lock:
                    job_ids_to_skip.append(db_id)
                async with processed_count_lock:
                    processed_count += 1
                    if processed_count % 25 == 0 or processed_count == total_db_jobs:
                        print(f"  [DB Descriptions] Progress: {processed_count}/{total_db_jobs} jobs processed ({updated_count} descriptions updated, {len(skipped_jobs)} skipped)...")
                return

        try:
            with open(strategy_path, "r") as f:
                strategy = json.load(f)
        except Exception as e:
            print(f"  Error loading strategy {strategy_path}: {e}")
            log_entry = {
                "id": db_id,
                "job_name": job_name,
                "company_name": company_name,
                "source": source,
                "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
                "url": url,
            }
            async with skipped_jobs_lock:
                skipped_jobs.append(log_entry)
            async with job_ids_to_skip_lock:
                job_ids_to_skip.append(db_id)
            return

        if verbose:
            print(f"  Scraping description for DB job {db_id} ({source}) via {strategy_path}")

        # Prepare the payload using the DB record's URL
        payload = dict(strategy)
        if payload.get("url") == "{url}":
            payload["url"] = url

        api_method = (
            "extract-js"
            if payload.get("js_config") is not None
            else "extract"
        )

        description = ""
        parsed_pay = ""
        micro_timeout = int(os.getenv("MICROSERVICE_TIMEOUT", "500"))
        async with global_semaphore:
            try:
                result = await asyncio.wait_for(
                    _request_with_domain_backoff(
                        dp.scrape_data,
                        payload,
                        url=url,
                        api_method=api_method,
                    ),
                    timeout=micro_timeout,
                )
                if result.get("status_code") == 200 and result.get("data"):
                    data_result = result["data"]
                    if isinstance(data_result, list) and len(data_result) > 0:
                        desc_key = next(
                            (k for k in ("description", "summary", "job-summary", "job_description", "job-description")
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
                            description = sanitize_untrusted_text(desc_text)

                        # Parse pay range if present in scraped page
                        scraped_pay = data_result[0].get("pay_range") or data_result[0].get("pay") or data_result[0].get("pay_rate")
                        if scraped_pay:
                            parsed_pay = parse_pay_range(scraped_pay)
                else:
                    status_code = str(result.get("status_code", "UNKNOWN"))
                    error_msg = result.get("error", "")
                    exception_msg = result.get("exception", "")
                    raw_text = result.get("raw", "")
                    host = os.getenv("SCRAPER_HOST", "localhost")
                    port = os.getenv("SCRAPER_PORT", "5052")
                    target_endpoint = result.get("url", f"http://{host}:{port}/{api_method}")
                    record_host_server_error(
                        company=company_name or source,
                        endpoint=target_endpoint,
                        target_url=url,
                        status_code=status_code,
                        error=error_msg,
                        exception=exception_msg,
                        raw_response=raw_text,
                        stage="DB Descriptions"
                    )
            except Exception as e:
                host = os.getenv("SCRAPER_HOST", "localhost")
                port = os.getenv("SCRAPER_PORT", "5052")
                record_host_server_error(
                    company=company_name or source,
                    endpoint=f"http://{host}:{port}/{api_method}",
                    target_url=url,
                    status_code="EXCEPTION",
                    error="Exception scraping description",
                    exception=repr(e),
                    stage="DB Descriptions"
                )
                print(f"  Error scraping description for DB job {db_id}: {e}")

        if description:
            try:
                update_fields = {"job_summary": description}
                if parsed_pay:
                    update_fields["pay_range"] = parsed_pay
                await _safe_db_call(dp.conn.update, "job", update_fields, {"id": db_id}, dbname=dp.dbname)
                async with updated_count_lock:
                    updated_count += 1
                if verbose:
                    pay_log = f", pay_range={parsed_pay}" if parsed_pay else ""
                    print(f"    Updated DB job {db_id} with description ({len(description)} chars){pay_log}")
            except Exception as e:
                print(f"    Failed to update DB record for job {db_id}: {e}")

            async with enriched_descriptions_lock:
                enriched_descriptions[url] = description
            if parsed_pay:
                async with enriched_pays_lock:
                    enriched_pays[url] = parsed_pay
        else:
            if verbose:
                print(f"    No description found for DB job {db_id} ({url})")
            log_entry = {
                "id": db_id,
                "job_name": job_name,
                "company_name": company_name,
                "source": source,
                "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
                "url": url,
            }
            async with skipped_jobs_lock:
                skipped_jobs.append(log_entry)
            async with job_ids_to_skip_lock:
                job_ids_to_skip.append(db_id)

        async with processed_count_lock:
            processed_count += 1
            if processed_count % 25 == 0 or processed_count == total_db_jobs:
                print(f"  [DB Descriptions] Progress: {processed_count}/{total_db_jobs} jobs processed ({updated_count} descriptions updated, {len(skipped_jobs)} skipped)...")

    async def process_domain_db_jobs(domain: str, jobs_list: list):
        for job in jobs_list:
            await scrape_single_db_job(job)
            # Polite delay between requests to the same domain (async-friendly)
            await asyncio.sleep(random.uniform(0.5, 1.5))

    # Process all domains concurrently
    if domain_to_jobs:
        tasks = [process_domain_db_jobs(domain, jobs_list) for domain, jobs_list in domain_to_jobs.items()]
        await asyncio.gather(*tasks)

    # Post-process in-memory data: Enrich descriptions and pay
    for item in data:
        item_url = item.get("url") or item.get("link")
        if item_url in enriched_descriptions:
            item["description"] = enriched_descriptions[item_url]
        if item_url in enriched_pays:
            item["pay"] = enriched_pays[item_url]
            item["pay_rate"] = enriched_pays[item_url]
    
    # Filter out skipped jobs from the list
    skipped_urls = {item.get("url") or item.get("link") for item in skipped_jobs if item.get("url") or item.get("link")}
    if skipped_urls:
        data = [item for item in data if item.get("url") not in skipped_urls and item.get("link") not in skipped_urls]

    if job_ids_to_skip:
        try:
            await _safe_db_call(dp.bulk_update_skip_status, job_ids_to_skip)
            print(f"Set skip=True for {len(job_ids_to_skip)} job(s) in the database.")
        except Exception as e:
            print(f"Warning: failed to update skip status for jobs: {e}")

    if skipped_jobs:
        log_filename = f"skipped_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        log_filepath = os.path.join("logs", "skipped", log_filename)
        try:
            os.makedirs(os.path.join("logs", "skipped"), exist_ok=True)
            with open(log_filepath, "w", encoding="utf-8") as f:
                json.dump(skipped_jobs, f, indent=2)
            print(f"Logged {len(skipped_jobs)} skipped job(s) to {log_filepath}")
        except Exception as e:
            print(f"Warning: failed to write skip log file '{log_filepath}': {e}")

    print(f"Scraped and saved descriptions for {updated_count}/{len(jobs_without_desc)} DB jobs. Marked {len(skipped_jobs)} jobs as skip=True in DB (due to missing strategy files, HTTP errors, or missing URLs).")
    return data


async def _load_jobs_without_embeddings(dp: DataPuller, limit: int = 50) -> list:
    """
    Load jobs from the database that have descriptions but no embeddings yet.
    Used as a fallback when scraping returns no jobs with descriptions.

    Args:
        dp: DataPuller instance for database access.
        limit: Maximum number of jobs to load.

    Returns:
        List of job dicts in raw scraped format with descriptions populated.
    """
    query = """
        SELECT j.id, j.job_name, c.company_name, j.link, j.job_summary,
               j.source, j.date_added, j.flexibility,
               o.city, o.state, o.location
        FROM job j
        JOIN company c ON j.company_id = c.id
        LEFT JOIN job_embeddings je ON j.id = je.job_id
        LEFT JOIN office o ON j.office_id = o.id
        WHERE j.job_summary IS NOT NULL
          AND j.skip IS NOT TRUE
          AND je.job_id IS NULL
        ORDER BY j.date_added DESC
        LIMIT %s
    """
    try:
        rows = await _safe_db_call(dp.conn.execute_sql, query, (limit,), fetch=True)
    except Exception as e:
        print(f"Error querying DB for jobs without embeddings: {e}")
        return []

    if not rows:
        print("No jobs found in DB missing embeddings.")
        return []

    jobs = []
    for row in rows:
        if isinstance(row, dict):
            job = {
                "id": row.get("id"),
                "title": row.get("job_name"),
                "company": row.get("company_name"),
                "link": row.get("link"),
                "url": row.get("link"),
                "description": row.get("job_summary"),
                "source": row.get("source"),
                "date_added": row.get("date_added"),
                "flexibility": row.get("flexibility", "NA"),
                "city": row.get("city"),
                "state": row.get("state"),
                "location": row.get("location"),
            }
        else:
            job = {
                "id": row[0],
                "title": row[1],
                "company": row[2],
                "link": row[3],
                "url": row[3],
                "description": row[4],
                "source": row[5],
                "date_added": row[6],
                "flexibility": row[7] if len(row) > 7 else "NA",
                "city": row[8] if len(row) > 8 else None,
                "state": row[9] if len(row) > 9 else None,
                "location": row[10] if len(row) > 10 else None,
            }
        jobs.append(job)

    print(f"Loaded {len(jobs)} jobs from DB with descriptions but no embeddings.")
    return jobs


async def _process_jobs_without_descriptions(results: list, dp: DataPuller) -> list:
    """
    Filter in-memory results list to keep only jobs that have descriptions.
    If an in-memory job lacks a description, attempt to populate it from DB first.
    If still missing, remove it from the active processing pool.

    Args:
        results: List of scraped job dicts.
        dp: DataPuller instance for database access.

    Returns:
        Filtered list of jobs that have descriptions.
    """
    if not results:
        _print_and_log_stats()
        return results

    def clean_source(s: str) -> str:
        if not s:
            return "Unknown"
        s_str = str(s).strip()
        return s_str if s_str else "Unknown"

    # Pre-count jobs that already have descriptions in memory for statistics tracking
    for idx, job in enumerate(results):
        if job.get("description"):
            job_key = job.get("link") or job.get("url") or job.get("id") or f"temp_key_{job.get('title')}_{job.get('company')}_{idx}"
            if job_key not in _processed_job_keys:
                _processed_job_keys.add(job_key)
                link_url = job.get("link") or job.get("url")
                src = clean_source(job.get("source") or job.get("site") or job.get("source_name") or (_extract_domain(link_url) if link_url else ""))
                _global_scraped_by_source[src] = _global_scraped_by_source.get(src, 0) + 1

    missing_indices = [idx for idx, job in enumerate(results) if not job.get("description")]
    if not missing_indices:
        _print_and_log_stats()
        return results

    total_missing = len(missing_indices)
    retained_from_db = 0
    indices_to_remove = []

    # Batch DB lookup for missing description jobs by ID and link/url
    job_ids = [results[i].get("id") for i in missing_indices if results[i].get("id") is not None]
    links = [results[i].get("link") or results[i].get("url") for i in missing_indices if results[i].get("link") or results[i].get("url")]

    db_desc_by_id = {}
    db_desc_by_link = {}
    if dp and hasattr(dp, "conn") and dp.conn:
        try:
            if job_ids:
                id_query = "SELECT id, job_summary FROM job WHERE id IN %s AND job_summary IS NOT NULL AND job_summary != ''"
                id_rows = await _safe_db_call(dp.conn.execute_sql, id_query, (tuple(job_ids),), fetch=True) or []
                for r in id_rows:
                    rid = r.get("id") if (isinstance(r, dict) or hasattr(r, 'get')) else r[0]
                    rsum = r.get("job_summary") if (isinstance(r, dict) or hasattr(r, 'get')) else r[1]
                    if rid is not None and rsum:
                        db_desc_by_id[rid] = rsum
            if links:
                link_query = "SELECT link, job_summary FROM job WHERE link IN %s AND job_summary IS NOT NULL AND job_summary != ''"
                link_rows = await _safe_db_call(dp.conn.execute_sql, link_query, (tuple(links),), fetch=True) or []
                for r in link_rows:
                    rlink = r.get("link") if (isinstance(r, dict) or hasattr(r, 'get')) else r[0]
                    rsum = r.get("job_summary") if (isinstance(r, dict) or hasattr(r, 'get')) else r[1]
                    if rlink and rsum:
                        db_desc_by_link[rlink] = rsum
        except Exception as e:
            print(f"Warning: failed bulk lookup in _process_jobs_without_descriptions: {e}")

    for idx in missing_indices:
        job = results[idx]
        jid = job.get("id")
        jlink = job.get("link") or job.get("url")
        summary = db_desc_by_id.get(jid) if jid in db_desc_by_id else db_desc_by_link.get(jlink)

        if summary and len(summary.strip()) > 50:
            job["description"] = summary
            retained_from_db += 1
            job_key = jlink or jid or f"temp_key_{idx}"
            if job_key not in _processed_job_keys:
                _processed_job_keys.add(job_key)
                src = clean_source(job.get("source") or job.get("site") or (_extract_domain(jlink) if jlink else ""))
                _global_scraped_by_source[src] = _global_scraped_by_source.get(src, 0) + 1
        else:
            indices_to_remove.append(idx)
            src = clean_source(job.get("source") or job.get("site") or (_extract_domain(jlink) if jlink else ""))
            _global_scraped_by_source[src] = _global_scraped_by_source.get(src, 0) + 1
            _global_failed_by_source[src] = _global_failed_by_source.get(src, 0) + 1

    print(f"Post-processing description filter across {total_missing} in-memory job(s) lacking descriptions:")
    if retained_from_db:
        print(f"  - Retained {retained_from_db} job(s) with valid descriptions retrieved from DB.")

    for idx in sorted(indices_to_remove, reverse=True):
        results.pop(idx)

    removed_count = len(indices_to_remove)
    print(f"  - Removed {removed_count} job(s) lacking descriptions from the active processing pool.")
    
    # Print and log stats at the end of the operation
    _print_and_log_stats()
    
    return results


async def _pipeline_stage_scrape_legacy(
    dp: DataPuller, user_preferences: dict, sites: dict, skip_db: bool, verbose: bool,
    enable_part_b: bool = False, skip_part_a: bool = False,
    db_limit: int = 50,
    reason: Optional[str] = None,
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
        skip_part_a: If True, skip the company career-page scraping (site_strategies/)
                     and jump straight to the description scraping step.
        reason: Optional string describing why the legacy path is being run.

    Returns:
        Combined list of raw scraped job dicts from both parts.
    """
    if reason:
        print(f"Using legacy scraping path ({reason}).")
    else:
        scrapers_config_path = os.getenv("SCRAPERS_CONFIG", "scrapers_config.yaml")
        if os.path.exists(scrapers_config_path):
            print("Using legacy scraping path (triggered via configuration).")
        else:
            print(f"Using legacy scraping path (no {scrapers_config_path} found).")

    # ==========================================
    # PART A: Scrape from company career pages
    # ==========================================
    data: list = []

    if not skip_part_a:
        print("--- Company Board Scraping ---")
        site_strategies_raw: list = []

        for i in range(len(sites.get("name", []))):
            strategy_path = f"./site_strategies/{sites['name'][i]}.json"
            if not os.path.exists(strategy_path):
                print(f"Warning: strategy file not found: {strategy_path}")
                continue
            strat = dp.load_site_strategies(strategy_path)
            api_method = (
                "extract-paginated"
                if strat.get("pagination") is not None
                else (
                    "extract-js"
                    if strat.get("js_config") is not None
                    else "extract"
                )
            )
            cat = classify_strategy_type(strat, api_method)
            strategy = {
                "company": sites["name"][i],
                "site": sites["site"][i],
                "strategy": strat,
                "api_method": api_method,
                "category": cat,
            }
            if verbose:
                print(
                    f"Company: {strategy['company']} | "
                    f"API method: {strategy['api_method']} | "
                    f"Category: {strategy['category']}"
                )
            site_strategies_raw.append(strategy)

        # Organize into queue priority: [non-javascript, javascript, paginated, multi-link]
        ordered_types = ["non-javascript", "javascript", "paginated", "multi-link"]
        categories = {cat: [] for cat in ordered_types}
        for s in site_strategies_raw:
            cat = s.get("category", "non-javascript")
            categories[cat].append(s)

        site_strategies = []
        for cat in ordered_types:
            site_strategies.extend(categories[cat])

        print(f"Loaded {len(site_strategies)} site strategies organized by queue priority:")
        for cat in ordered_types:
            companies = [s["company"] for s in categories[cat]]
            if companies:
                print(f"  - [{cat.upper()}] ({len(companies)}): {', '.join(companies)}")

        concurrency = user_preferences.get("scraper_concurrency", 5)
        static_semaphore = asyncio.Semaphore(int(os.getenv("SCRAPER_CONCURRENCY", concurrency)))

        async def scrape_strategy(strat_item):
            company_name = strat_item.get("company", "Unknown")
            queue_start_time = time_module.time()
            try:
                company_url = strat_item["strategy"].pop("company_url", None)
                results_local = []
                urls = strat_item["strategy"]["url"]
                
                # Check if this strategy is JS-based or static HTML
                is_js = _is_js_request([strat_item.get("strategy", {})], {"api_method": strat_item.get("api_method", "")})

                # If strategy url is a list, process them sequentially for this company to avoid hitting them too fast
                if isinstance(urls, list):
                    print(f"Scraping {company_name} ({len(urls)} URLs)...")
                    consecutive_failures = 0
                    first_scrape_start = None
                    for idx, url_val in enumerate(urls):
                        new_payload = dict(strat_item)
                        new_payload["strategy"] = dict(strat_item["strategy"])
                        new_payload["strategy"]["url"] = url_val
                        
                        # Only static requests use static_semaphore; JS requests serialize via _js_scrape_lock
                        if is_js:
                            if first_scrape_start is None:
                                first_scrape_start = time_module.time()
                            d = await scrape_sites(new_payload, company_url, dp)
                        else:
                            async with static_semaphore:
                                if first_scrape_start is None:
                                    first_scrape_start = time_module.time()
                                d = await scrape_sites(new_payload, company_url, dp)
                        if d:
                            consecutive_failures = 0
                            if isinstance(d, list):
                                results_local.extend(d)
                            else:
                                results_local.append(d)
                        else:
                            consecutive_failures += 1
                            if consecutive_failures >= 2 and (idx + 1) < len(urls):
                                print(f"  [{company_name}] Encountered {consecutive_failures} consecutive failures/timeouts. Skipping remaining {len(urls) - (idx + 1)} URLs to prevent stalling.")
                                break
                    scrape_start = first_scrape_start or queue_start_time
                else:
                    print(f"Scraping {company_name}...")
                    if is_js:
                        scrape_start = time_module.time()
                        d = await scrape_sites(strat_item, company_url, dp)
                    else:
                        async with static_semaphore:
                            scrape_start = time_module.time()
                            d = await scrape_sites(strat_item, company_url, dp)
                    if d:
                        if isinstance(d, list):
                            results_local.extend(d)
                        else:
                            results_local.append(d)
                scrape_duration = time_module.time() - scrape_start
                queue_wait = scrape_start - queue_start_time
                print(f"Finished {company_name} in {scrape_duration:.1f}s (queued: {queue_wait:.1f}s, {len(results_local)} jobs scraped).")
                return results_local
            except Exception as e:
                total_elapsed = time_module.time() - queue_start_time
                print(f"Error scraping strategy for {company_name} after {total_elapsed:.1f}s: {e}. Continuing with remaining strategies...")
                host = os.getenv("SCRAPER_HOST", "localhost")
                port = os.getenv("SCRAPER_PORT", "5052")
                record_host_server_error(
                    company=company_name,
                    endpoint=f"http://{host}:{port}/{strat_item.get('api_method', 'extract')}",
                    target_url=str(strat_item.get("strategy", {}).get("url", "")),
                    status_code="EXCEPTION",
                    error=f"Uncaught exception in scrape_strategy for {company_name}",
                    exception=repr(e),
                    stage="Part A (Company Board)"
                )
                return []

        # Execute queue category by category: [non-javascript, javascript, paginated, multi-link]
        for cat in ordered_types:
            cat_strategies = categories[cat]
            if not cat_strategies:
                continue

            print(f"\n--- Scraping [{cat.upper()}] Strategies ({len(cat_strategies)}) ---")
            if cat == "non-javascript":
                # Static HTML profiles are fast and safe to run concurrently in parallel
                cat_tasks = [scrape_strategy(strat) for strat in cat_strategies]
                cat_results = await asyncio.gather(*cat_tasks, return_exceptions=True)
                for sublist in cat_results:
                    if isinstance(sublist, Exception):
                        print(f"Strategy task raised an exception: {sublist}. Continuing...")
                    elif isinstance(sublist, list):
                        data.extend(sublist)
            else:
                # Browser-driven profiles (javascript, paginated, multi-link) require headless browser
                # resources on the crawler microservice; execute them sequentially to eliminate lock pile-up,
                # distorted timer accumulation, and socket contention.
                for strat in cat_strategies:
                    try:
                        res = await scrape_strategy(strat)
                        if isinstance(res, list):
                            data.extend(res)
                    except Exception as e:
                        print(f"Strategy task for {strat.get('company')} raised an exception: {e}. Continuing...")

        print(f"\nTotal jobs scraped from company boards: {len(data)}")

        # ── Load jobs into the database first (before scraping descriptions) ──
        if data and not skip_db:
            print("--- Loading Jobs into Database ---")
            await _safe_db_call(dp.load_scraped_data_to_db, data)
        elif not skip_db:
            print("No company board jobs to load into DB.")
    else:
        print("--- Skipping Part A (company career-page scraping) ---")

    # ── Then scrape descriptions from DB for jobs that are missing them ──
    concurrency = user_preferences.get("scraper_concurrency", 5)
    if not skip_db:
        print("--- Scraping Missing Job Descriptions from DB (via job_page_strategy/) ---")
        data = await scrape_job_descriptions_from_db(data, dp, verbose=verbose, concurrency=concurrency)
    else:
        print("--- Skipping DB description scraping (skip_db=True) ---")
        # Fall back to the old in-memory description scraping if no DB
        if data:
            print("--- Scraping Job Descriptions (in-memory fallback) ---")
            data = await scrape_job_descriptions(data, dp, verbose=verbose, concurrency=concurrency)
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
                        pay_raw = (
                            f"{i.get('min_amount', '')} - "
                            f"{i.get('max_amount', '')} "
                            f"{i.get('interval', '')}"
                        ).strip()
                        if not pay_raw or pay_raw == "-":
                            pay_raw = i.get("pay_range") or i.get("pay") or i.get("pay_rate") or ""
                        
                        pay_parsed = parse_pay_range(pay_raw) if pay_raw else ""

                        job = {
                            "source": i.get("site"),
                            "title": i.get("title"),
                            "url": i.get("job_url"),
                            "link": i.get("job_url"),
                            "company": i.get("company"),
                            "pay": pay_parsed,
                            "pay_rate": pay_parsed,
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
                            "pay_rate": "",
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

            time_module.sleep(random.uniform(delay["min"], delay["max"]))

        print(f"Total jobs scraped from job boards: {len(jobs)}")

        if not skip_db:
            await _safe_db_call(dp.load_scraped_data_to_db, jobs)
    else:
        print("--- Job Board Scraping (Part B disabled — skipping) ---")

    # ── Fallback: if no jobs have descriptions (or no jobs scraped at all), load from DB jobs missing embeddings ──
    combined = data + jobs
    if not skip_db and (not combined or not any(job.get("description") for job in combined)):
        print("--- Fallback: no scraped jobs have descriptions; loading from DB ---")
        db_jobs = await _load_jobs_without_embeddings(dp, limit=db_limit)
        if db_jobs:
            print(f"Loaded {len(db_jobs)} jobs from DB with descriptions but no embeddings.")
            data = db_jobs
            jobs = []

    # ── Post-process: remove jobs without descriptions and log them ──
    print("--- Post-processing: removing jobs without descriptions ---")
    data = await _process_jobs_without_descriptions(data, dp)
    jobs = await _process_jobs_without_descriptions(jobs, dp)

    return data + jobs
