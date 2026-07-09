"""
Test harness for the fallback (legacy) scraping pipeline defined in
app.fallback_scraping_instructions.

Prompts the user for the key parameters that _pipeline_stage_scrape_legacy
accepts, and then runs it, printing results to the console.

Usage:
    python -m app.test_fallback_scraping
"""

import asyncio
import json
import os
import sys
from datetime import datetime

from dotenv import load_dotenv

from app.pull_data import DataPuller
from app.fallback_scraping_instructions import _pipeline_stage_scrape_legacy


def _bool_input(prompt: str, default: bool = False) -> bool:
    """Prompt the user for a yes / no answer and return a bool."""
    default_str = "y" if default else "n"
    while True:
        raw = input(f"{prompt} (y/n) [{default_str}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  Please enter 'y' or 'n'.")


def _json_input(prompt: str) -> dict:
    """Prompt the user for a JSON / Python-dict literal and return a dict."""
    print(prompt)
    print("  (Enter a blank line to use an empty dict {})")
    lines: list[str] = []
    while True:
        line = input("  ")
        if not line.strip():
            break
        lines.append(line)
    raw = " ".join(lines).strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try evaluating as a Python literal (e.g. {"key": "val"})
        try:
            return eval(raw, {"__builtins__": {}}, {})
        except Exception:
            print("  Could not parse input as JSON. Using empty dict.")
            return {}


def _get_sites_input() -> dict:
    """
    Prompt the user for a 'sites' dict interactively.
    Returns something like::

        {"name": ["CompanyA", "CompanyB"], "site": ["https://a.com/careers", "https://b.com/jobs"]}
    """
    print("--- Sites / Companies to scrape ---")
    print("Enter company names one per line (blank line to finish):")
    names: list[str] = []
    while True:
        raw = input("  name: ").strip()
        if not raw:
            break
        names.append(raw)

    if not names:
        print("No companies entered. Using empty sites dict.")
        return {"name": [], "site": []}

    print("Now enter the career-page URL for each company (same order):")
    sites: list[str] = []
    for name in names:
        url = input(f"  URL for '{name}': ").strip()
        sites.append(url or "")

    return {"name": names, "site": sites}


def _process_jobs_without_descriptions(results: list, dp: DataPuller) -> list:
    """
    After scraping, remove jobs that did not get a description, set skip=True
    in the database for those jobs, and log them to a timestamped log file.

    Args:
        results: List of scraped job dicts.
        dp: DataPuller instance for database access.

    Returns:
        Filtered list of jobs that have descriptions.
    """
    missing_description_indices = [
        idx for idx, job in enumerate(results)
        if not job.get("description")
    ]

    if not missing_description_indices:
        return results

    skipped_jobs = []
    # Collect skip flags for bulk update
    job_ids_to_skip = []

    # Query lockup: join job with company to resolve company name
    lookup_query = """
        SELECT j.id, j.job_name, c.company_name, j.source, j.date_added, j.link
        FROM job j
        JOIN company c ON j.company_id = c.id
        WHERE j.link = %s
    """

    for idx in missing_description_indices:
        job = results[idx]
        link = job.get("link") or job.get("url") or ""
        job_name = job.get("title") or ""
        source = job.get("source") or ""
        company_name = job.get("company") or ""

        db_id = None
        job_date = None
        try:
            rows = dp.conn.execute_sql(lookup_query, (link,), fetch=True)
            if rows:
                row = rows[0]
                db_id = row.get("id") if isinstance(row, dict) else (row[0] if len(row) > 0 else None)
                if not company_name:
                    company_name = row.get("company_name") if isinstance(row, dict) else (row[2] if len(row) > 2 else company_name)
                if not job_name:
                    job_name = row.get("job_name") if isinstance(row, dict) else (row[1] if len(row) > 1 else job_name)
                if not source:
                    source = row.get("source") if isinstance(row, dict) else (row[3] if len(row) > 3 else source)
                if isinstance(row, dict):
                    job_date = row.get("date_added")
                elif len(row) > 4:
                    job_date = row[4]
        except Exception as e:
            print(f"Warning: failed to look up job in DB for link '{link}': {e}")

        log_entry = {
            "id": db_id,
            "job_name": job_name,
            "company_name": company_name,
            "source": source,
            "date": (job_date.isoformat() if hasattr(job_date, "isoformat") else str(job_date)) if job_date else datetime.now().isoformat(),
        }
        skipped_jobs.append(log_entry)

        if db_id is not None:
            job_ids_to_skip.append(db_id)

    # Bulk update skip flag in DB
    if job_ids_to_skip:
        try:
            dp.bulk_update_skip_status(job_ids_to_skip)
            print(f"Set skip=True for {len(job_ids_to_skip)} job(s) in the database.")
        except Exception as e:
            print(f"Warning: failed to update skip status for jobs: {e}")

    # Write log file
    log_filename = f"skipped_jobs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(log_filename, "w", encoding="utf-8") as f:
            json.dump(skipped_jobs, f, indent=2)
        print(f"Logged {len(skipped_jobs)} skipped job(s) to {log_filename}")
    except Exception as e:
        print(f"Warning: failed to write skip log file '{log_filename}': {e}")

    # Remove jobs without descriptions from results (reverse order to preserve indices)
    for idx in sorted(missing_description_indices, reverse=True):
        results.pop(idx)

    return results


async def main() -> None:
    load_dotenv()

    print("=" * 60)
    print(" Fallback Scraping Instructions — Test Harness")
    print("=" * 60)

    # ── User preferences ──
    user_preferences = _json_input(
        "Enter user_preferences as a JSON / dict literal"
        " (e.g. {\"target_cities\": [\"Remote\", \"New York\"],"
        " \"enable_fallback_part_b\": false}):"
    )

    # ── Sites / companies ──
    sites = _get_sites_input()
    print(f"Sites: {sites}")

    # ── Boolean flags ──
    skip_db = _bool_input("Skip database persistence (skip_db)?", default=True)
    verbose = _bool_input("Verbose output (verbose)?", default=False)
    enable_part_b = _bool_input(
        "Enable Part B (job-board scraping via JobSpy)", default=False
    )

    print("\n" + "=" * 60)
    print(" Configuration summary")
    print("=" * 60)
    print(f"  user_preferences : {json.dumps(user_preferences, indent=2)}")
    print(f"  sites            : {json.dumps(sites, indent=2)}")
    print(f"  skip_db          : {skip_db}")
    print(f"  verbose          : {verbose}")
    print(f"  enable_part_b    : {enable_part_b}")
    print()

    # ── Create a minimal DataPuller (won't connect to DB if skip_db=True) ──
    dp = DataPuller(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD", ""),
        dbname=os.getenv("DB_NAME", "postgres"),
    )

    print("Starting legacy scrape pipeline...\n")
    try:
        results = await _pipeline_stage_scrape_legacy(
            dp=dp,
            user_preferences=user_preferences,
            sites=sites,
            skip_db=skip_db,
            verbose=verbose,
            enable_part_b=enable_part_b,
        )
    except Exception as exc:
        print(f"\nPipeline failed with exception: {exc}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f" Pipeline finished — {len(results)} job(s) returned")
    print("=" * 60)

    # ── Post-process: remove jobs without descriptions and log them ──
    if results and not skip_db:
        results = _process_jobs_without_descriptions(results, dp)

    if results:
        print(json.dumps(results, indent=2, default=str))
    else:
        print("(No jobs scraped.)")


if __name__ == "__main__":
    asyncio.run(main())