import os
import asyncio
import json
import datetime
from app.pull_data import DataPuller
from app.scrapers.adapter_loader import AdapterLoader
from app.logger import error_logger_continue
from app.pipeline.pipeline_utils import _row_to_job_dict
from typing import List, Dict
from app.fallback_scraping_instructions import (
    _global_scraped_by_source,
    _global_failed_by_source,
    _processed_job_keys,
    _print_and_log_stats,
    _extract_domain
)

async def scrape_missing_descriptions_24h(dp: DataPuller, verbose: bool = False) -> List[Dict]:
    """
    Perform the special scraping workflow for jobs in the database from the last 24 hours that do not have descriptions.
    """
    # 1. Update matching jobs: skip = NULL
    update_query = """
        UPDATE job
        SET skip = NULL
        WHERE (job_summary IS NULL OR LENGTH(job_summary) < 600) AND date_added >= CURRENT_DATE - INTERVAL '1 day';
    """
    try:
        dp.conn.execute_sql(update_query)
        print("Updated jobs from the previous 24 hours missing or having short snippet descriptions to skip = NULL.")
    except Exception as e:
        print(f"Error resetting skip status for jobs missing descriptions: {e}")

    # 2. Pull all jobs from the previous 24 hours where the description (job_summary) is null or short API snippet
    select_query = """
        SELECT j.id, j.link, j.source, j.job_name, c.company_name, j.date_added
        FROM job j
        LEFT JOIN company c ON j.company_id = c.id
        WHERE (j.job_summary IS NULL OR LENGTH(j.job_summary) < 600) AND j.date_added >= CURRENT_DATE - INTERVAL '1 day';
    """
    try:
        rows = dp.conn.execute_sql(select_query, fetch=True)
    except Exception as e:
        print(f"Error querying jobs missing descriptions: {e}")
        return []

    if not rows:
        print("No jobs found in the previous 24 hours missing descriptions.")
        return []

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

    print(f"Found {len(jobs_without_desc)} jobs from the last 24 hours missing descriptions to scrape.")

    import datetime
    import glob
    from urllib.parse import urlparse
    import time
    import random
    
    os.makedirs(os.path.join("logs", "skipped"), exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join("logs", f"missing_desc_scrape_{timestamp}.log")
    
    job_page_strategy_dir = "./job_page_strategy"
    
    def find_strategy_file(source_name: str) -> str | None:
        if not source_name:
            return None
        source_lower = source_name.lower()
        for filepath in glob.glob(os.path.join(job_page_strategy_dir, "*.json")):
            filename = os.path.basename(filepath)
            name = filename[:-5].lower() # remove .json
            if name == source_lower:
                return filepath
            if "." in name:
                domain_part = name.split('.')[0]
                if domain_part == source_lower:
                    return filepath
            if "." in source_lower:
                source_domain_part = source_lower.split('.')[0]
                if source_domain_part == name:
                    return filepath
        return None

    successful_ids = []
    log_entries = []

    # ── Per-domain rate-limit tracking ──
    # Tracks consecutive 429s per domain so we can back off more aggressively.
    _domain_429_count: Dict[str, int] = {}
    MAX_CONSECUTIVE_429 = 3      # threshold after which we apply extra backoff
    EXTRA_BACKOFF_SECONDS = 30   # extra cooldown after hitting MAX_CONSECUTIVE_429 on a domain

    for job in jobs_without_desc:
        db_id = job["db_id"]
        url = job["url"]
        source = job.get("source", "")
        job_name = job.get("title", "")
        company_name = job.get("company", "")

        # Extract domain for rate-limit tracking
        try:
            parsed_domain = urlparse(url).netloc.lower()
            if parsed_domain.startswith("www."):
                parsed_domain = parsed_domain[4:]
        except Exception:
            parsed_domain = source.lower()

        if not url or not source:
            log_msg = f"Job ID {db_id} ({job_name} | {company_name}): Failed to scrape. Reason: Missing URL or source."
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")
            continue

        strategy_path = find_strategy_file(source)
        if not strategy_path:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = source.lower()
            
            print(f"  No strategy file found for source '{source}'. Generating strategy for domain '{domain}' using {url}...")
            success = await dp.generate_and_test_strategy(destination_link=url, job_id=db_id, domain_name=domain)
            if success:
                strategy_path = find_strategy_file(source)
                if not strategy_path:
                    strategy_path = os.path.join(job_page_strategy_dir, f"{domain}.json")
            else:
                log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Reason: Could not generate a working strategy."
                print(log_msg)
                log_entries.append(log_msg)
                try:
                    dp.bulk_update_skip_status([db_id])
                except Exception as e:
                    print(f"Warning: failed to update skip status for job {db_id}: {e}")
                continue

        try:
            with open(strategy_path, "r") as f:
                strategy = json.load(f)
        except Exception as e:
            log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Reason: Error loading strategy file: {e}"
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")
            continue

        payload = dict(strategy)
        if payload.get("url") == "{url}":
            payload["url"] = url

        api_method = "extract-js" if payload.get("js_config") is not None else "extract"

        # ── Attempt scrape with retry + exponential backoff on 429 ──
        max_retries = 4
        base_delay = 2.0
        description = ""
        scraper_response = None
        did_skip_due_to_rate_limit = False

        for attempt in range(1, max_retries + 1):
            # Domain-level cooldown: if this domain has been hammered with 429s,
            # insert an extra forced delay before we even try.
            domain_strikes = _domain_429_count.get(parsed_domain, 0)
            if domain_strikes >= MAX_CONSECUTIVE_429:
                extra_sleep = EXTRA_BACKOFF_SECONDS * (1 + random.random() * 0.5)
                print(f"  [Rate Limit] Domain '{parsed_domain}' has {domain_strikes} consecutive 429s. "
                      f"Cooling off for {extra_sleep:.0f}s before retry (attempt {attempt}/{max_retries})...")
                await asyncio.sleep(extra_sleep)
                # Reduce the counter so we don't loop forever on extra backoff;
                # it will only re-trigger if we get another 429.
                _domain_429_count[parsed_domain] = max(0, domain_strikes - 1)

            try:
                scraper_response = await dp.scrape_data(payload, api_method=api_method)
            except Exception as e:
                scraper_response = {"error": f"Exception raised during scrape: {e}"}

            status = scraper_response.get("status_code")
            if status == 429:
                # Track the 429 per domain
                _domain_429_count[parsed_domain] = _domain_429_count.get(parsed_domain, 0) + 1
                domain_strikes = _domain_429_count[parsed_domain]

                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s + jitter
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"  [429] Job ID {db_id}: Rate limited (domain={parsed_domain}, "
                          f"strike={domain_strikes}). Retrying in {delay:.1f}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # All retries exhausted — mark as skip but log the 429
                    log_msg = (f"Job ID {db_id} (URL: {url}): Failed to scrape after {max_retries} retries. "
                               f"All attempts returned 429. Scraper response: {json.dumps(scraper_response)}")
                    print(log_msg)
                    log_entries.append(log_msg)
                    try:
                        dp.bulk_update_skip_status([db_id])
                    except Exception as e:
                        print(f"Warning: failed to update skip status for job {db_id}: {e}")
                    did_skip_due_to_rate_limit = True
                    break

            elif scraper_response.get("data"):
                data_result = scraper_response["data"]
                if isinstance(data_result, list) and len(data_result) > 0:
                    desc_key = next(
                        (k for k in ("description", "summary", "job-summary", "job_description", "job-description")
                         if data_result[0].get(k)),
                        None
                    )
                    if desc_key:
                        desc_text = data_result[0].get(desc_key, "")
                    else:
                        desc_text = next(
                            (v for v in data_result[0].values() if isinstance(v, str) and len(v) > 50),
                            ""
                        )
                    if desc_text and len(desc_text) > 50:
                        description = desc_text
                        _domain_429_count[parsed_domain] = 0
                        break  # success — exit retry loop

                if status in (200, "200"):
                    _domain_429_count[parsed_domain] = 0
                    break
                else:
                    break
            else:
                # Non-429 error (4xx, 5xx, etc.) — do not retry, just mark as skip
                break

        if did_skip_due_to_rate_limit:
            continue

        if description:
            try:
                dp.conn.update("job", {"job_summary": description}, {"id": db_id}, dbname=dp.dbname)
                successful_ids.append(db_id)
                job["description"] = description
                log_msg = f"Job ID {db_id} (URL: {url}): Scraped successfully."
                print(log_msg)
                log_entries.append(log_msg)
            except Exception as e:
                log_msg = f"Job ID {db_id} (URL: {url}): Failed to update DB with description: {e}"
                print(log_msg)
                log_entries.append(log_msg)
        else:
            resp_str = json.dumps(scraper_response) if scraper_response else "No response returned"
            log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Scraper response: {resp_str}"
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")

        # Base delay between jobs (0.5-1.5s) — only if we didn't already sleep for a 429 retry
        time.sleep(random.uniform(0.5, 1.5))

    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_entries) + "\n")
        print(f"Detailed run status saved to: {log_file_path}")
    except Exception as e:
        print(f"Warning: failed to write run log file '{log_file_path}': {e}")

    # Track statistics globally
    for idx, job in enumerate(jobs_without_desc):
        db_id = job.get("db_id")
        url = job.get("url")
        source = job.get("source")
        
        # Resolve clean source name
        def clean_source(s: str) -> str:
            if not s:
                return "Unknown"
            s_str = str(s).strip()
            return s_str if s_str else "Unknown"

        src_name = source
        if not src_name:
            src_name = _extract_domain(url) if url else "Unknown"
        src_name = clean_source(src_name)

        job_key = url or f"db_id_{db_id}_{idx}"
        if job_key not in _processed_job_keys:
            _processed_job_keys.add(job_key)
            _global_scraped_by_source[src_name] = _global_scraped_by_source.get(src_name, 0) + 1
            if not job.get("description"):
                _global_failed_by_source[src_name] = _global_failed_by_source.get(src_name, 0) + 1

    # Print and log statistics
    _print_and_log_stats()

    if not successful_ids:
        return []

    # Fetch fully populated records from DB
    query_cols = [
        "j.id", "j.job_name", "c.company_name", "j.link", "j.job_summary", "j.description AS extracted_summary",
        "j.requirements", "j.responsibilities", "j.pay_range", "j.seniority", "j.work_type", "j.timezone",
        "j.source", "j.date_added", "o.city", "o.state", "o.location",
        "j.flexibility",
        "je.title_embedding", "je.requirements_embedding", "je.responsibilities_embedding", "je.description_embedding"
    ]
    joins = [
        "JOIN company c ON j.company_id = c.id",
        "LEFT JOIN office o ON j.office_id = o.id",
        "LEFT JOIN job_embeddings je ON j.id = je.job_id"
    ]
    ids_placeholder = ", ".join(["%s"] * len(successful_ids))
    query = f"""
        SELECT {', '.join(query_cols)}
        FROM job j
        {' '.join(joins)}
        WHERE j.id IN ({ids_placeholder})
    """
    try:
        rows = dp.conn.execute_sql(query, tuple(successful_ids), fetch=True)
        if rows:
            return [_row_to_job_dict(row) for row in rows]
    except Exception as e:
        print(f"Error fetching successfully scraped jobs from DB: {e}")

    return []


async def recalculate_db_final_scores(dp: DataPuller):
    """
    Recalculate final score, priority, and apply recommendation for all processed jobs in DB
    using stored vector_scores, cheap_llm_results, and strong_llm_results, and update final_application_queue.
    """
    from app.llm_classifier import FinalApplicationQueue
    
    print("=" * 60)
    print("RECALCULATING FINAL SCORES FROM DATABASE")
    print("=" * 60)

    query_cols = [
        "j.id", "j.job_name", "c.company_name", "j.link", "j.job_summary", "j.description AS extracted_summary",
        "j.requirements", "j.responsibilities", "j.pay_range", "j.seniority", "j.work_type", "j.timezone",
        "j.source", "j.date_added", "o.city", "o.state", "o.location", "j.flexibility",
        "vs.semantic_score", "vs.title_similarity", "vs.requirements_similarity",
        "vs.responsibility_similarity", "vs.adjusted_score", "vs.archetype_name AS best_archetype",
        "clr.fit_score AS cheap_fit_score", "clr.decision AS cheap_decision",
        "clr.strengths AS cheap_strengths", "clr.concerns AS cheap_concerns",
        "clr.hard_requirements_and_tools AS cheap_hard_requirements_and_tools",
        "clr.core_responsibilities AS cheap_core_responsibilities",
        "clr.years_of_experience AS cheap_years_of_experience",
        "clr.domain_and_education AS cheap_domain_and_education",
        "clr.raw_response AS cheap_raw_response",
        "slr.final_score AS strong_final_score", "slr.priority AS strong_priority",
        "slr.apply_recommendation AS strong_apply_rec", "slr.red_flags AS strong_red_flags",
        "slr.tailoring_notes AS strong_tailoring_notes", "slr.recruiter_bait_likelihood AS strong_bait",
        "slr.detailed_fit_analysis AS strong_fit_analysis",
        "slr.company_scale_fit AS strong_company_scale_fit",
        "slr.career_trajectory AS strong_career_trajectory",
        "slr.seniority_scope_calibration AS strong_seniority_scope_calibration",
        "slr.hero_story_match AS strong_hero_story_match",
        "slr.project_complexity AS strong_project_complexity",
        "slr.shadow_work_friction AS strong_shadow_work_friction",
        "slr.domain_business_model_friction AS strong_domain_business_model_friction",
        "slr.recruiter_red_flags AS strong_recruiter_red_flags",
        "slr.driving_points AS strong_driving_points",
        "slr.raw_response AS strong_raw_response"
    ]
    joins = [
        "JOIN company c ON j.company_id = c.id",
        "LEFT JOIN office o ON j.office_id = o.id",
        "LEFT JOIN (SELECT DISTINCT ON (job_id) * FROM vector_scores ORDER BY job_id, id DESC) vs ON j.id = vs.job_id",
        "LEFT JOIN (SELECT DISTINCT ON (job_id) * FROM cheap_llm_results ORDER BY job_id, id DESC) clr ON j.id = clr.job_id",
        "LEFT JOIN (SELECT DISTINCT ON (job_id) * FROM strong_llm_results ORDER BY job_id, id DESC) slr ON j.id = slr.job_id"
    ]
    where_clauses = [
        "j.skip IS NOT TRUE",
        "(vs.job_id IS NOT NULL OR clr.job_id IS NOT NULL OR slr.job_id IS NOT NULL)"
    ]

    query = f"""
        SELECT {', '.join(query_cols)}
        FROM job j
        {' '.join(joins)}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY j.date_added DESC
    """

    try:
        rows = dp.conn.execute_sql(query, fetch=True) or []
    except Exception as e:
        print(f"Error querying database for recalculation: {e}")
        return

    if not rows:
        print("No processed jobs found in DB to recalculate.")
        return

    seen_ids = set()
    unique_rows = []
    for r in rows:
        rid = r.get("id") if isinstance(r, dict) else r[0]
        if rid not in seen_ids:
            seen_ids.add(rid)
            unique_rows.append(r)

    print(f"Loaded {len(unique_rows)} unique job(s) from database for score recalculation.")

    col_map = {col.split(' AS ')[-1].split('.')[-1]: idx for idx, col in enumerate(query_cols)}

    def _get_val(r, key, default=None):
        if hasattr(r, 'get'):
            v = r.get(key)
            if v is not None:
                return v
        if key in col_map and hasattr(r, '__getitem__'):
            idx = col_map[key]
            if len(r) > idx and r[idx] is not None:
                return r[idx]
        return default

    jobs = []
    for r in unique_rows:
        job = _row_to_job_dict(r)
        job['best_archetype'] = _get_val(r, "best_archetype")
        semantic_score = _get_val(r, "semantic_score")
        job['semantic_score'] = semantic_score if semantic_score is not None else 0.0
        job['semantic_score_percent'] = int(round(semantic_score * 100)) if semantic_score is not None else 0
        job['title_similarity'] = _get_val(r, "title_similarity")
        job['requirements_similarity'] = _get_val(r, "requirements_similarity")
        job['responsibility_similarity'] = _get_val(r, "responsibility_similarity")
        job['adjusted_score'] = _get_val(r, "adjusted_score")
        
        cheap_fit = _get_val(r, "cheap_fit_score")
        cheap_dec = _get_val(r, "cheap_decision")
        if cheap_fit is not None or cheap_dec:
            job['cheap_llm_result'] = {
                "fit_score": cheap_fit if cheap_fit is not None else 50,
                "decision": cheap_dec or "maybe"
            }

        strong_final = _get_val(r, "strong_final_score")
        strong_rec = _get_val(r, "strong_apply_rec")
        strong_prio = _get_val(r, "strong_priority")
        if strong_final is not None or strong_rec:
            def _parse_json(val):
                if isinstance(val, str):
                    try: return json.loads(val)
                    except Exception: return {}
                elif isinstance(val, dict): return val
                return {}
                
            job['strong_llm_result'] = {
                "final_score": strong_final if strong_final is not None else 50,
                "priority": strong_prio or "medium",
                "apply_recommendation": strong_rec or "maybe",
                "company_scale_fit": _parse_json(_get_val(r, "strong_company_scale_fit")),
                "career_trajectory": _parse_json(_get_val(r, "strong_career_trajectory")),
                "seniority_scope_calibration": _parse_json(_get_val(r, "strong_seniority_scope_calibration")),
                "hero_story_match": _parse_json(_get_val(r, "strong_hero_story_match")),
                "project_complexity": _parse_json(_get_val(r, "strong_project_complexity")),
                "shadow_work_friction": _parse_json(_get_val(r, "strong_shadow_work_friction")),
                "domain_business_model_friction": _parse_json(_get_val(r, "strong_domain_business_model_friction")),
                "recruiter_red_flags": _parse_json(_get_val(r, "strong_recruiter_red_flags")),
                "driving_points": _get_val(r, "strong_driving_points") or [],
                "raw_response": _parse_json(_get_val(r, "strong_raw_response"))
            }
        jobs.append(job)

    queue_generator = FinalApplicationQueue()
    ranked_jobs = queue_generator.rank_jobs(jobs)

    dp.save_final_queue(ranked_jobs)
    print(f"Successfully recalculated scores and updated {len(ranked_jobs)} job(s) in final_application_queue table.")

    active_jobs = [j for j in ranked_jobs if j.get('priority') != 'skip' and j.get('apply_recommendation') != 'skip']
    print("\n" + "=" * 60)
    print("RECALCULATED TOP JOBS SUMMARY")
    print("=" * 60)
    for i, job in enumerate(active_jobs[:10], 1):
        features = job.get('features', {})
        metadata = job.get('metadata', {})
        cheap_score = job.get('cheap_llm_result', {}).get('fit_score', 'N/A')
        strong_score = job.get('strong_llm_result', {}).get('final_score', 'N/A')
        sem_score = job.get('semantic_score_percent', int(round((job.get('semantic_score') or 0)*100)))

        print(f"{i}. {features.get('title', 'Unknown')} at {metadata.get('company_name', 'Unknown')}")
        print(f"   Job ID: {metadata.get('job_id')} | Final Score: {job.get('final_score', 0):.1f}/100 | Priority: {job.get('priority', '').upper()}")
        print(f"   Breakdown: Semantic={sem_score}% | Cheap LLM={cheap_score} | Strong LLM={strong_score}")


