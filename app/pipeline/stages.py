import os
import json
import time
import logging
import asyncio
from typing import Any, Dict, List, Optional, Tuple
from app.pull_data import DataPuller
from app.text_engine import TextProcessor
from app.ai_engine import AIEngine
from app.archetype_engine import ArchetypeManager, Archetype
from app.scrapers.adapter_loader import AdapterLoader
from app.scrapers.validator import ScrapedDataValidator
from app.fallback_scraping_instructions import _pipeline_stage_scrape_legacy
from app.llm_classifier import CheapLLMClassifier, StrongLLMReranker, process_stage_6, process_stage_7, process_stage_8
from app.config_utils import _load_user_config, load_resume_as_text
from app.logger import _log_pipeline_stats, error_logger_crash, error_logger_continue
from app.rule_filters import apply_rule_filters
from app.ai_utils import _load_model_if_lm_studio, _unload_model_if_lm_studio, extract_and_cache_profile, _provider_is_lm_studio, call_llm_for_extraction, generate_embeddings, generate_embeddings_batch
from app.pipeline.pipeline_utils import _check_run_fallback_flag, _normalize_to_pool_format, _load_jobs_for_stage
from app.pipeline.maintenance import scrape_missing_descriptions_24h
from app.rule_filters import is_missing_value
from app.llm_usage_tracker import usage_tracker
from app.prompt_injection_defender import sanitize_untrusted_text, validate_and_clean_extracted_data
import yaml
import numpy as np

ARCHETYPES_CONFIG = {}
_arch_path = os.getenv('ARCHETYPES_CONFIG', '')
if _arch_path and os.path.exists(_arch_path):
    with open(_arch_path, 'r') as f:
        ARCHETYPES_CONFIG = json.load(f)

async def pipeline_stage_setup(skip_db: bool = False, verbose: bool = False,
                               resume_text: str = "", profile_text: str = "") -> dict:
    """
    Stage 0: Load environment, resume, profile, create AI engine, extract skills/titles.
    
    Args:
        skip_db: If True, skip database initialization.
        verbose: If True, print detailed debug output.
        resume_text: If provided, use this instead of loading from file.
        profile_text: If provided, use this instead of loading from file.
    
    Returns:
        dict with keys: resume, user_profile, skills, job_titles, dp, tp, ai, user_preferences
    """
    print("=" * 50)
    print("PIPELINE STAGE 0: SETUP")
    print("=" * 50)

    # Load .env variables
    if resume_text:
        resume = resume_text
    else:
        resume = load_resume_as_text("RESUME")
    
    if profile_text:
        user_profile = profile_text
    else:
        user_profile = load_resume_as_text("PROFILE")

    # Load configuration from user_preferences.yaml
    user_config = _load_user_config()

    # Create Data Puller Object
    dp = DataPuller(
        dbname=user_config.get("db_name", os.getenv("DB_NAME", "")),
        user=user_config.get("db_user", os.getenv("DB_USER", "")),
        password=user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
        host=user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
        port=str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
    )

    # Get sites file from .env
    sites_file = os.getenv("JOB_SITES", "")
    if sites_file and os.path.exists(sites_file):
        sites = dp.load_sites_list(sites_file)
        print("Sites Retrieved")
    else:
        sites = {"name": [], "site": []}
        print("Warning: JOB_SITES file not found or not configured.")

    # Extract skills and job titles from profile
    tp = TextProcessor()
    ai = AIEngine(
        default_provider_name="lm_studio",
        extraction_model=user_config.get("extraction_model", os.getenv("EXTRACTION_MODEL", "local-model")),
        embeddings_model=user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "local-model"))
    )
    requirements_raw = tp.get_section_content(user_profile, "Requirements") or tp.get_section_content(user_profile, "Skills")
    titles_raw = tp.get_section_content(user_profile, "Job Titles")
    requirements = tp.clean_list_from_text(requirements_raw)
    job_titles = tp.clean_list_from_text(titles_raw)

    if verbose:
        print(f"--- Profile Extraction ---")
        print(f"Extracted {len(requirements)} requirements: {requirements}")
        print(f"Extracted {len(job_titles)} job titles: {job_titles}")
        print(f"---------------------------\n")

    # Configure logging if not already configured
    root = logging.getLogger()
    if not root.handlers:
        os.makedirs("logs", exist_ok=True)
        error_handler = logging.FileHandler(os.path.join("logs", "app_error.log"), mode="a", encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        error_handler.setFormatter(error_formatter)
        root.addHandler(error_handler)
        root.setLevel(logging.ERROR)

    # Load user preferences
    prefs_yaml_path = os.getenv("USER_PREFERENCES_YAML", "")
    if prefs_yaml_path and os.path.exists(prefs_yaml_path):
        with open(prefs_yaml_path, 'r') as f:
            user_preferences = yaml.safe_load(f)
    else:
        user_preferences = {}

    result = {
        "resume": resume,
        "user_profile": user_profile,
        "requirements": requirements,
        "job_titles": job_titles,
        "dp": dp,
        "tp": tp,
        "ai": ai,
        "user_preferences": user_preferences,
        "sites": sites,
        "sites_file": sites_file,
        "user_config": user_config,
        "db_config": {
            "dbname": user_config.get("db_name", os.getenv("DB_NAME", "")),
            "user": user_config.get("db_user", os.getenv("DB_USER", "")),
            "password": user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
            "host": user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
            "port": str(user_config.get("db_port", os.getenv("DB_PORT", "5432"))),
        },
    }

    print("Stage 0 complete: Setup and profile extraction done.")
    return result


async def pipeline_stage_scrape(setup_data: dict, skip_db: bool = False,
                                verbose: bool = False,
                                max_pages: Optional[int] = None,
                                headless: bool = True,
                                debug_logging: bool = False,
                                skip_part_a: Optional[bool] = None,
                                db_limit: int = 50,
                                scrape_missing_24h: bool = False) -> List[Dict]:
    """
    Stage 1: Scrape jobs using the pluggable adapter system.

    Attempts to load scrapers from scrapers_config.yaml. If the config file
    is not found or contains no enabled adapters, falls back to the legacy
    hardcoded scraping path for backward compatibility.

    Args:
        setup_data: Output from pipeline_stage_setup.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
        max_pages: Override max_pages for all adapters (e.g. CLI --pages).
        headless: Override headless mode for all adapters (False = visible browser).
        debug_logging: If True, enable debug-level logging.
        skip_part_a: If True, skip Part A in legacy fallback path.
        db_limit: Max records to pull from DB in bulk-load fallback operations.
        scrape_missing_24h: If True, only re-scrape descriptions for recent DB jobs.

    Returns:
        List[Dict] of scraped jobs in the 'processed_job_pool' format.
    """
    dp = setup_data["dp"]
    tp = setup_data["tp"]
    user_preferences = setup_data.get("user_preferences", {})
    sites = setup_data.get("sites", {"name": [], "site": []})

    print("=" * 50)
    print("PIPELINE STAGE 1: SCRAPING")
    print("=" * 50)

    if scrape_missing_24h:
        print("[Stage 1] scrape_missing_24h is True. Running dedicated DB description re-scraping workflow...")
        if skip_db:
            print("[Stage 1] WARNING: skip_db is True, cannot run DB description re-scraping. Returning empty pool.")
            return []
        processed_job_pool = await scrape_missing_descriptions_24h(dp, verbose)
        print(f"Stage 1 complete: {len(processed_job_pool)} jobs in pool.")
        return processed_job_pool

    # Determine config path (check env var first, then default)
    scrapers_config_path = os.getenv("SCRAPERS_CONFIG", "scrapers_config.yaml")

    # ── Attempt adapter-based scraping ──
    all_scraped: List[Dict] = []
    used_adapters = False

    if os.path.exists(scrapers_config_path):
        try:
            loader = AdapterLoader(config_path=scrapers_config_path, verbose=verbose)
            loader.load_config()

            # Apply CLI overrides to adapter configs before loading
            if max_pages is not None or not headless or debug_logging:
                for entry in loader._config.get("scrapers", []):
                    cfg = entry.setdefault("config", {})
                    if max_pages is not None:
                        cfg["max_pages"] = max_pages
                    if not headless:
                        cfg["headless"] = False
                    if debug_logging:
                        cfg["debug"] = True

            adapter_list = loader.load_adapters()

            if adapter_list:
                used_adapters = True
                print(f"Loaded {len(adapter_list)} adapter(s) from {scrapers_config_path}")
                all_scraped = await loader.run_all(dp=dp)

                # Persist raw scraped data to DB
                if not skip_db and all_scraped:
                    dp.load_scraped_data_to_db(all_scraped)
            else:
                print(f"Config found at {scrapers_config_path} but no enabled adapters loaded.")
        except Exception as e:
            print(f"Adapter system failed: {e}. Falling back to legacy scraping.")

    # Read enable_fallback_part_b and skip_part_a from user preferences
    enable_part_b = user_preferences.get("enable_fallback_part_b", False)
    # CLI --skip-part-a overrides yaml value; if neither is set, default to False
    resolved_skip_part_a = (
        True
        if skip_part_a is True
        else bool(user_preferences.get("skip_part_a", False))
    )

    # ── Fallback to legacy path if adapters didn't run ──
    if not used_adapters:
        if not os.path.exists(scrapers_config_path):
            reason = f"no {scrapers_config_path} found"
        else:
            reason = f"no enabled adapters loaded from {scrapers_config_path} or adapter run failed"
        all_scraped = await _pipeline_stage_scrape_legacy(
            dp, user_preferences, sites, skip_db, verbose,
            enable_part_b=enable_part_b, skip_part_a=resolved_skip_part_a,
            db_limit=db_limit,
            reason=reason,
        )

    # ── Also run fallback if explicitly configured to do so ──
    if used_adapters and _check_run_fallback_flag(scrapers_config_path):
        print("run_fallback_after_adapters is True — also running legacy fallback scraping...")
        fallback_jobs = await _pipeline_stage_scrape_legacy(
            dp, user_preferences, sites, skip_db, verbose,
            enable_part_b=enable_part_b, skip_part_a=resolved_skip_part_a,
            db_limit=db_limit,
            reason="run_fallback_after_adapters is enabled",
        )
        if fallback_jobs:
            orig_adapter_count = len(all_scraped)
            orig_fallback_count = len(fallback_jobs)
            merged = {}
            for job in all_scraped:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            for job in fallback_jobs:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            all_scraped = list(merged.values())
            print(f"Merged {orig_fallback_count} fallback job(s) with {orig_adapter_count} adapter job(s) (deduplicated by link into {len(all_scraped)} total unique job(s)).")
        else:
            print("Fallback scraping returned no jobs.")

    # ── Fallback: if no jobs have descriptions, load from DB jobs missing embeddings ──
    if all_scraped and not skip_db and not any(job.get("description") for job in all_scraped):
        print("--- Fallback: no scraped jobs have descriptions; loading from DB ---")
        from app.fallback_scraping_instructions import _load_jobs_without_embeddings
        db_jobs = await _load_jobs_without_embeddings(dp, limit=db_limit)
        if db_jobs:
            print(f"Loaded {len(db_jobs)} jobs from DB with descriptions but no embeddings (deduplicating by link)...")
            merged = {}
            for job in all_scraped:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            for job in db_jobs:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            all_scraped = list(merged.values())

    # ── Sync in-memory descriptions from DB for any jobs missing them ──
    if all_scraped and not skip_db:
        jobs_needing_sync = [j for j in all_scraped if not j.get("description")]
        if jobs_needing_sync:
            print(f"Syncing in-memory descriptions from database for {len(jobs_needing_sync)} jobs missing them...")
            job_ids = [j.get("id") for j in jobs_needing_sync if j.get("id") is not None]
            links = [j.get("link") or j.get("url") for j in jobs_needing_sync if j.get("link") or j.get("url")]
            
            db_desc_by_id = {}
            db_desc_by_link = {}
            try:
                if job_ids:
                    id_query = "SELECT id, job_summary FROM job WHERE id IN %s AND job_summary IS NOT NULL AND job_summary != ''"
                    id_rows = dp.conn.execute_sql(id_query, (tuple(job_ids),), fetch=True) or []
                    for r in id_rows:
                        rid = r.get("id") if (isinstance(r, dict) or hasattr(r, 'get')) else r[0]
                        rsum = r.get("job_summary") if (isinstance(r, dict) or hasattr(r, 'get')) else r[1]
                        if rid is not None and rsum:
                            db_desc_by_id[rid] = rsum
                if links:
                    link_query = "SELECT link, job_summary FROM job WHERE link IN %s AND job_summary IS NOT NULL AND job_summary != ''"
                    link_rows = dp.conn.execute_sql(link_query, (tuple(links),), fetch=True) or []
                    for r in link_rows:
                        rlink = r.get("link") if (isinstance(r, dict) or hasattr(r, 'get')) else r[0]
                        rsum = r.get("job_summary") if (isinstance(r, dict) or hasattr(r, 'get')) else r[1]
                        if rlink and rsum:
                            db_desc_by_link[rlink] = rsum
            except Exception as e:
                print(f"Warning: failed bulk description sync from DB: {e}")
                
            sync_count = 0
            for job in jobs_needing_sync:
                jid = job.get("id")
                jlink = job.get("link") or job.get("url")
                summary = db_desc_by_id.get(jid) if jid in db_desc_by_id else db_desc_by_link.get(jlink)
                if summary and len(summary.strip()) > 50:
                    job["description"] = summary
                    sync_count += 1
            if sync_count > 0:
                print(f"Synced {sync_count} job description(s) from database to memory.")

    # ── Post-process: remove jobs without descriptions and log them ──
    if all_scraped:
        print("--- Post-processing: removing jobs without descriptions ---")
        from app.fallback_scraping_instructions import _process_jobs_without_descriptions
        all_scraped = await _process_jobs_without_descriptions(all_scraped, dp)

    # ── Convert raw scraped data to processed_job_pool format ──
    processed_job_pool = _normalize_to_pool_format(all_scraped)

    # If no jobs were scraped (e.g., in test mode), use dummy data as fallback
    if not processed_job_pool:
        from tests.test_data import make_dummy_stage1_output
        print("No real jobs scraped. Using dummy fallback data.")
        processed_job_pool = make_dummy_stage1_output(15)

    print(f"Stage 1 complete: {len(processed_job_pool)} jobs in pool.")
    return processed_job_pool


async def pipeline_stage_preliminary_filter(jobs: List[Dict], user_preferences: Optional[dict] = None,
                                             text_processor: Optional[TextProcessor] = None,
                                             dp: Optional[DataPuller] = None,
                                             skip_db: bool = False,
                                             verbose: bool = False) -> List[Dict]:
    """
    Stage 1.5: Early disqualification via hard rule filters before AI extraction & embedding pass.
    
    Filters jobs based on deterministic fields (work_type, seniority, pay, timezone, location).
    Jobs that fail are marked skip=True and excluded from Stage 2 AI processing.
    """
    print("=" * 50)
    print("PIPELINE STAGE 1.5: PRELIMINARY RULE FILTERING (EARLY DISQUALIFICATION)")
    print("=" * 50)

    if user_preferences is None:
        user_preferences = {}
    if text_processor is None:
        text_processor = TextProcessor()

    print(f"Running preliminary rule evaluation on {len(jobs)} jobs...")
    
    skipped_count = 0
    active_jobs = []

    for job in jobs:
        if job is None:
            continue

        # Extract features deterministically if missing
        features = job.get('features', {})
        if not features:
            desc = sanitize_untrusted_text(job.get('description', job.get('job_summary', '')))
            title = sanitize_untrusted_text(job.get('title', job.get('job_name', '')))
            flexibility = job.get('flexibility', 'NA')
            location = job.get('location', '')
            pay = job.get('pay', job.get('pay_range', ''))
            
            features = {
                'title': title,
                'description': desc,
                'pay': pay if not is_missing_value(pay) else text_processor.extract_salary(desc),
                'seniority': text_processor.detect_seniority(desc, title=title),
                'work_type': text_processor.detect_work_type_from_fields(flexibility, location, title, desc),
                'timezone': text_processor.detect_timezone(desc),
            }
            job['features'] = features

        is_skipped = apply_rule_filters(job, user_preferences)
        job['skip'] = is_skipped

        if is_skipped:
            skipped_count += 1
        else:
            active_jobs.append(job)

    print(f"Preliminary filtering complete: {skipped_count} jobs disqualified, {len(active_jobs)} active jobs proceed to Stage 2.")

    if dp and not skip_db:
        skipped_ids = [job.get('metadata', {}).get('job_id') or job.get('id') for job in jobs if job.get('skip') and (job.get('metadata', {}).get('job_id') or job.get('id'))]
        if skipped_ids:
            try:
                dp.bulk_update_skip_status(skipped_ids)
                print(f"Synced preliminary skip status for {len(skipped_ids)} jobs to database.")
            except Exception as e:
                print(f"Warning: Failed to sync preliminary skip status to DB: {e}")

    return active_jobs


async def pipeline_stage_embed_and_extract(jobs: List[Dict], ai_engine: Optional[AIEngine] = None,
                                           text_processor: Optional[TextProcessor] = None,
                                           dp: Optional[DataPuller] = None,
                                           skip_db: bool = False,
                                           verbose: bool = False) -> List[Dict]:
    """
    Stage 2: Run deterministic extraction, LLM extraction, and embedding generation on jobs.
    
    Args:
        jobs: List of job dicts from Stage 1 (or dummy data).
        ai_engine: AIEngine instance for LLM calls and embeddings.
        text_processor: TextProcessor instance for deterministic extraction.
        dp: DataPuller instance for DB operations.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
    
    Returns:
        List[Dict] of jobs with features enriched (skills, requirements, summary, embeddings).
    """
    print("=" * 50)
    print("PIPELINE STAGE 2: EMBEDDING GENERATION + LLM EXTRACTION")
    print("=" * 50)

    if ai_engine is None:
        ai_engine = AIEngine(default_provider_name="lm_studio")
    if text_processor is None:
        text_processor = TextProcessor()

    # Load LLM provider config from user_preferences.yaml
    _user_config = _load_user_config()
    _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))
    _embeddings_llm = _user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))

    # Check DB status counts for missing extractions and embeddings if DB is connected
    if dp and not skip_db:
        try:
            missing_ext_count_rows = dp.conn.execute_sql(
                "SELECT COUNT(*) FROM job j WHERE j.skip IS NOT TRUE AND j.job_summary IS NOT NULL AND (j.description IS NULL OR j.requirements IS NULL OR j.responsibilities IS NULL)",
                fetch=True
            )
            missing_ext_count = missing_ext_count_rows[0][0] if missing_ext_count_rows else 0

            missing_emb_count_rows = dp.conn.execute_sql(
                "SELECT COUNT(*) FROM job j LEFT JOIN job_embeddings je ON j.id = je.job_id WHERE j.skip IS NOT TRUE AND j.job_summary IS NOT NULL AND (je.job_id IS NULL OR je.title_embedding IS NULL)",
                fetch=True
            )
            missing_emb_count = missing_emb_count_rows[0][0] if missing_emb_count_rows else 0

            print(f"[Stage 2] Database status check: {missing_ext_count} active jobs are missing extracted data, {missing_emb_count} active jobs are missing embeddings.")
        except Exception as e:
            print(f"Warning: could not check DB job status counts: {e}")

    # First pass: deterministic extraction for jobs that don't have features yet
    print(f"Starting deterministic extraction on {len(jobs)} jobs...")
    processed_job_pool = []
    for idx, raw_job in enumerate(jobs):
        if raw_job is None:
            continue
        
        # Build or normalize job dict structure
        job_id = raw_job.get('id') or raw_job.get('metadata', {}).get('job_id', idx + 1)
        company_name = raw_job.get('company', raw_job.get('company_name', raw_job.get('metadata', {}).get('company_name', 'Unknown')))
        link = raw_job.get('link', raw_job.get('url', raw_job.get('metadata', {}).get('link', '')))

        # If the job is already in the processed format (from previous stage), use it
        if 'features' in raw_job and 'metadata' in raw_job:
            features = raw_job['features']
            desc = features.get('description', '')
            if desc:
                desc = sanitize_untrusted_text(desc)
                features['description'] = desc
                if is_missing_value(features.get('pay')):
                    features['pay'] = text_processor.extract_salary(desc)
                if is_missing_value(features.get('seniority')):
                    features['seniority'] = text_processor.detect_seniority(desc, title=features.get('title', ''))
                if is_missing_value(features.get('work_type')):
                    features['work_type'] = text_processor.detect_work_type_from_fields(
                        features.get('flexibility'), features.get('location'), features.get('title'), desc
                    )
                if is_missing_value(features.get('timezone')):
                    features['timezone'] = text_processor.detect_timezone(desc)
                if not features.get('requirements') or not features.get('responsibilities'):
                    r_reqs = text_processor.extract_requirements(desc)
                    r_resps = text_processor.extract_responsibilities(desc)
                    if r_reqs and not features.get('requirements'):
                        features['requirements'] = r_reqs
                    if r_resps and not features.get('responsibilities'):
                        features['responsibilities'] = r_resps
                    if not features.get('summary'):
                        features['summary'] = text_processor.extract_summary_from_description(desc, title=features.get('title', ''))
            # Normalize work_type capitalization (Remote/Hybrid/Onsite/Unknown)
            if features.get('work_type'):
                wt_lower = str(features['work_type']).lower()
                if 'remote' in wt_lower:
                    features['work_type'] = 'Remote'
                elif 'hybrid' in wt_lower:
                    features['work_type'] = 'Hybrid'
                elif 'onsite' in wt_lower or 'on-site' in wt_lower or 'in office' in wt_lower or 'in-office' in wt_lower:
                    features['work_type'] = 'Onsite'
                else:
                    features['work_type'] = 'Unknown'
            processed_job_pool.append(raw_job)
            continue

        # Check if job was loaded from DB with pre-existing requirements/responsibilities
        existing_db_reqs = raw_job.get('requirements')
        existing_db_resps = raw_job.get('responsibilities')
        
        # Build from raw format
        description = sanitize_untrusted_text(raw_job.get('description', raw_job.get('job_summary', '')))
        title = sanitize_untrusted_text(raw_job.get('title', raw_job.get('job_name', '')))
        flexibility = raw_job.get('flexibility', 'NA')
        location = raw_job.get('location', '')
        pay = raw_job.get('pay', raw_job.get('pay_range', ''))

        if not description or len(description) < 50:
            if verbose:
                error_logger_continue(f"Warning: insufficient description for job ID {job_id}")
            continue

        # Check DB deduplication cache by exact description text
        cached_data = None
        if dp and not skip_db:
            cached_data = dp.get_cached_job_features(description)

        if cached_data:
            extracted_data = {
                "metadata": {
                    "job_id": job_id,
                    "source": "cached_db",
                    "company_name": company_name,
                    "link": link,
                    "in_db": True,
                },
                "features": {
                    "title": title,
                    "description": description,
                    "pay": cached_data["features"].get("pay") or (pay if not is_missing_value(pay) else text_processor.extract_salary(description)),
                    "seniority": cached_data["features"].get("seniority") or text_processor.detect_seniority(description, title=title),
                    "work_type": cached_data["features"].get("work_type") or text_processor.detect_work_type_from_fields(flexibility, location, title, description),
                    "timezone": text_processor.detect_timezone(description),
                    "requirements": cached_data["features"].get("requirements") or [],
                    "responsibilities": cached_data["features"].get("responsibilities") or [],
                    "summary": cached_data["features"].get("summary") or text_processor.extract_summary_from_description(description, title=title),
                },
                "embeddings": cached_data.get("embeddings", {}),
            }
        else:
            detected_work_type = text_processor.detect_work_type_from_fields(flexibility, location, title, description)
            regex_reqs = existing_db_reqs or text_processor.extract_requirements(description)
            regex_resps = existing_db_resps or text_processor.extract_responsibilities(description)
            regex_summary = raw_job.get('extracted_summary') or text_processor.extract_summary_from_description(description, title=title)

            # If no structured section headers exist in description, fallback to sentence extraction
            if not regex_reqs and not regex_resps and len(description) > 0:
                clean_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', description) if len(s.strip()) > 15]
                regex_reqs = clean_sentences

            extracted_data = {
                "metadata": {
                    "job_id": job_id,
                    "source": "scraped",
                    "company_name": company_name,
                    "link": link,
                    "in_db": True,
                },
                "features": {
                    "title": title,
                    "description": description,
                    "pay": pay if not is_missing_value(pay) else text_processor.extract_salary(description),
                    "seniority": text_processor.detect_seniority(description, title=title),
                    "work_type": detected_work_type,
                    "timezone": text_processor.detect_timezone(description),
                    "requirements": regex_reqs or [],
                    "responsibilities": regex_resps or [],
                    "summary": regex_summary,
                },
                "embeddings": {
                    "title_vector": None,
                    "requirements_vector": None,
                    "responsibilities_vector": None,
                    "description_vector": None,
                    "pay_vector": None,
                    "location_vector": None,
                },
            }
        processed_job_pool.append(extracted_data)


    print(f"Successfully extracted data for {len(processed_job_pool)} jobs.")

    # Identify jobs needing LLM extraction and/or embeddings
    jobs_needing_extraction = []
    jobs_missing_embeddings = []

    for job in processed_job_pool:
        features = job.get('features', {})
        # A job only needs LLM extraction if BOTH requirements and responsibilities are empty
        needs_ext = (
            not features.get('requirements') and
            not features.get('responsibilities')
        )
        if needs_ext:
            jobs_needing_extraction.append(job)

        emb = job.get('embeddings', {})
        needs_emb = (
            not emb.get('title_vector') or
            not emb.get('requirements_vector') or
            not emb.get('responsibilities_vector') or
            not emb.get('pay_vector') or
            not emb.get('location_vector')
        )
        if needs_emb:
            jobs_missing_embeddings.append(job)

    print(f"[Stage 2] Status Check: {len(jobs_needing_extraction)} jobs need extraction, {len(jobs_missing_embeddings)} jobs need embeddings.")

    # Second pass: LLM extraction
    if len(jobs_needing_extraction) > 0:
        is_lm_studio_extraction = _provider_is_lm_studio(ai_engine, _extraction_llm)
        if is_lm_studio_extraction:
            extraction_model = ai_engine.extraction_model
            print(f"[Model Mgmt] Loading LM Studio extraction model: {extraction_model} ...")
            ai_engine.load_model(provider_name=_extraction_llm, model_name=extraction_model)
            loaded = ai_engine.wait_for_model_loaded(provider_name=_extraction_llm, model_name=extraction_model)
            if not loaded:
                print("[Model Mgmt] WARNING: extraction model may not be fully loaded yet.")

        print(f"Starting AI/LLM Extraction Pass on {len(jobs_needing_extraction)} jobs...")
        from app.ai_limiter import AILimiter, run_in_thread, ProgressTracker
        extraction_limiter = AILimiter("stage_2_extraction", _extraction_llm)
        tracker = ProgressTracker(len(jobs_needing_extraction), prefix="Extraction Pass")
        await tracker.start()

        async def process_extraction(index, job):
            try:
                if verbose:
                    print(f"Extraction job {index + 1}/{len(jobs_needing_extraction)}: {job['features']['title']}")

                if not job or 'features' not in job:
                    error_logger_continue(f"Warning: job at index {index} has invalid structure")
                    return

                description = job['features'].get('description', '')
                if not description:
                    error_logger_continue(f"Warning: job at index {index} has no description")
                    return

                # Extract skills, requirements and summary via LLM using AILimiter
                est_tokens = (len(description) // 4) + 1000
                ai_data = await extraction_limiter.execute(
                    call_llm_for_extraction, ai_engine, description, _extraction_llm,
                    est_tokens=est_tokens
                )

                requirements = []
                responsibilities = []
                summary = ""
                llm_pay = None
                llm_work_type = None
                llm_seniority = None

                if isinstance(ai_data, dict):
                    requirements = ai_data.get('requirements', [])
                    responsibilities = ai_data.get('responsibilities', [])
                    summary = ai_data.get('summary', "")
                    llm_pay = ai_data.get('pay_range')
                    llm_work_type = ai_data.get('work_type')
                    llm_seniority = ai_data.get('seniority')
                elif isinstance(ai_data, str):
                    try:
                        parsed = json.loads(ai_data)
                        if isinstance(parsed, dict):
                            requirements = parsed.get('requirements', [])
                            responsibilities = parsed.get('responsibilities', [])
                            summary = parsed.get('summary', "")
                            llm_pay = parsed.get('pay_range')
                            llm_work_type = parsed.get('work_type')
                            llm_seniority = parsed.get('seniority')
                    except Exception:
                        pass

                # Normalize extracted values to prevent type mismatches
                def to_string(v) -> str:
                    if v is None:
                        return ""
                    if isinstance(v, list):
                        return " ".join(to_string(x) for x in v if x is not None)
                    if isinstance(v, dict):
                        return " ".join(to_string(x) for x in v.values() if x is not None)
                    return str(v).strip()

                if isinstance(requirements, list):
                    requirements = [to_string(r) for r in requirements if r]
                else:
                    requirements = [to_string(requirements)] if requirements else []

                if isinstance(responsibilities, list):
                    responsibilities = [to_string(r) for r in responsibilities if r]
                else:
                    responsibilities = [to_string(responsibilities)] if responsibilities else []

                summary = to_string(summary)

                if isinstance(llm_pay, list):
                    llm_pay = " - ".join(to_string(p) for p in llm_pay if p)
                else:
                    llm_pay = to_string(llm_pay)

                if isinstance(llm_work_type, list):
                    llm_work_type = llm_work_type[0] if llm_work_type else ""
                llm_work_type = to_string(llm_work_type)

                if isinstance(llm_seniority, list):
                    llm_seniority = " ".join(to_string(s) for s in llm_seniority if s)
                llm_seniority = to_string(llm_seniority)

                features = job['features']
                features['requirements'] = requirements or []
                features['responsibilities'] = responsibilities or []
                features['summary'] = summary or ""

                # Update pay and work_type using LLM fallback if deterministic extraction missed them
                if llm_pay and is_missing_value(features.get('pay')):
                    features['pay'] = llm_pay
                if llm_work_type and is_missing_value(features.get('work_type')):
                    # Normalize work_type from LLM to match capitalization
                    wt = llm_work_type.strip().lower()
                    if 'remote' in wt:
                        features['work_type'] = 'Remote'
                    elif 'hybrid' in wt:
                        features['work_type'] = 'Hybrid'
                    elif 'onsite' in wt or 'on-site' in wt:
                        features['work_type'] = 'Onsite'
                    else:
                        features['work_type'] = 'Unknown'

                # Prefer LLM seniority if extracted successfully
                if llm_seniority and not is_missing_value(llm_seniority) and llm_seniority.lower() != "unknown":
                    s_lower = llm_seniority.strip().lower()
                    if "c-suite" in s_lower or "executive" in s_lower or "vp" in s_lower:
                        features['seniority'] = 'C-Suite'
                    elif "manager" in s_lower or "director" in s_lower or "head" in s_lower or "management" in s_lower:
                        features['seniority'] = 'Management'
                    elif "senior" in s_lower or "sr" in s_lower or "principal" in s_lower or "staff" in s_lower:
                        features['seniority'] = 'Senior'
                    elif "mid-level" in s_lower or "intermediate" in s_lower or "mid" in s_lower:
                        features['seniority'] = 'Mid-Level'
                    elif "junior" in s_lower or "jr" in s_lower or "entry" in s_lower or "associate" in s_lower or "intern" in s_lower:
                        features['seniority'] = 'Junior'
                    elif "lead" in s_lower:
                        features['seniority'] = 'Lead'
                    else:
                        features['seniority'] = llm_seniority.strip()

                # Post-validate extracted features before saving to DB
                job['features'] = validate_and_clean_extracted_data(features)
            finally:
                await tracker.increment()

        # Process all extraction tasks in parallel/sequential according to limiter
        extraction_tasks = [process_extraction(idx, job) for idx, job in enumerate(jobs_needing_extraction)]
        await asyncio.gather(*extraction_tasks)
        tracker.stop()

        if is_lm_studio_extraction:
            extraction_model = ai_engine.extraction_model
            print(f"[Model Mgmt] Unloading LM Studio extraction model: {extraction_model} ...")
            ai_engine.unload_model(provider_name=_extraction_llm, model_name=extraction_model)
            unloaded = ai_engine.wait_for_model_unloaded(provider_name=_extraction_llm, model_name=extraction_model)
            if not unloaded:
                print("[Model Mgmt] WARNING: extraction model may not be fully unloaded yet.")

    # Save extractions for ALL jobs in pool to database (regex + LLM extractions)
    if dp and not skip_db:
        print("Saving extractions (requirements, responsibilities, metadata) to database...")
        job_updates = []
        for job in processed_job_pool:
            meta = job.get('metadata', {})
            jid = meta.get('job_id') or job.get('id')
            if not jid:
                continue
            job_updates.append({
                "id": jid,
                "pay_range": job['features'].get('pay'),
                "seniority": job['features'].get('seniority'),
                "work_type": job['features'].get('work_type'),
                "timezone": job['features'].get('timezone'),
                "description": job['features'].get('summary'),
                "requirements": job['features'].get('requirements'),
                "responsibilities": job['features'].get('responsibilities'),
            })
        if job_updates:
            dp.update_job_metadata(job_updates)
            print(f"Saved extractions for {len(job_updates)} jobs to database.")

    # Combine jobs missing embeddings with those that were just extracted
    jobs_for_embeddings = list(jobs_missing_embeddings)
    seen_job_ids = {j['metadata']['job_id'] for j in jobs_for_embeddings}
    for job in jobs_needing_extraction:
        if job['metadata']['job_id'] not in seen_job_ids:
            jobs_for_embeddings.append(job)
            seen_job_ids.add(job['metadata']['job_id'])

    # Third pass: Embedding generation
    if len(jobs_for_embeddings) > 0:
        is_lm_studio_embeddings = _provider_is_lm_studio(ai_engine, _embeddings_llm)
        if is_lm_studio_embeddings:
            embeddings_model = ai_engine.embeddings_model
            print(f"[Model Mgmt] Loading LM Studio embeddings model: {embeddings_model} ...")
            ai_engine.load_model(provider_name=_embeddings_llm, model_name=embeddings_model)
            loaded = ai_engine.wait_for_model_loaded(provider_name=_embeddings_llm, model_name=embeddings_model)
            if not loaded:
                print("[Model Mgmt] WARNING: embeddings model may not be fully loaded yet.")

        print(f"Starting AI/LLM Embeddings Pass on {len(jobs_for_embeddings)} jobs...")
        from app.ai_limiter import AILimiter, run_in_thread, ProgressTracker
        embedding_limiter = AILimiter("stage_2_embedding", _embeddings_llm)
        embedding_tracker = ProgressTracker(len(jobs_for_embeddings), prefix="Embedding Pass")
        await embedding_tracker.start()

        async def process_embeddings(index, job):
            try:
                if verbose:
                    print(f"Embedding job {index + 1}/{len(jobs_for_embeddings)}: {job['features']['title']}")

                if not job or 'features' not in job:
                    return

                features = job['features']

                # Batch vector generation
                title_text = str(features.get('title') or '')
                reqs_list = features.get('requirements') or []
                requirements_text = ", ".join(str(r) for r in reqs_list if r) if isinstance(reqs_list, list) else str(reqs_list)
                resps_list = features.get('responsibilities') or []
                responsibilities_text = ", ".join(str(r) for r in resps_list if r) if isinstance(resps_list, list) else str(resps_list)
                summary_text = str(features.get('summary') or '')

                pay_val = features.get('pay') or features.get('pay_rate') or ''
                pay_text = f"Pay Range: {pay_val}" if pay_val else ''

                loc_parts = []
                if features.get('location'):
                    loc_parts.append(f"Location: {features.get('location')}")
                elif features.get('city') or features.get('state'):
                    city_state = ", ".join(str(x) for x in [features.get('city'), features.get('state')] if x)
                    loc_parts.append(f"Location: {city_state}")
                if features.get('work_type') and str(features.get('work_type')).lower() not in ('unknown', 'na'):
                    loc_parts.append(f"Work Arrangement: {features.get('work_type')}")
                if features.get('timezone') and str(features.get('timezone')).lower() != 'na':
                    loc_parts.append(f"Timezone: {features.get('timezone')}")
                location_text = ". ".join(loc_parts)

                raw_texts = [title_text, requirements_text, responsibilities_text, summary_text, pay_text, location_text]
                texts = [str(t) if t is not None else "" for t in raw_texts]

                async with embedding_limiter.semaphore:
                    est_tokens = (sum(len(t) for t in texts) // 4) + 64
                    vectors = await embedding_limiter.execute(
                        generate_embeddings_batch, ai_engine, texts, _embeddings_llm,
                        est_tokens=est_tokens, use_semaphore=False
                    )
                    if vectors and len(vectors) == 6:
                        job['embeddings']['title_vector'] = vectors[0]
                        job['embeddings']['requirements_vector'] = vectors[1]
                        job['embeddings']['responsibilities_vector'] = vectors[2]
                        job['embeddings']['description_vector'] = vectors[3]
                        job['embeddings']['pay_vector'] = vectors[4]
                        job['embeddings']['location_vector'] = vectors[5]
            finally:
                await embedding_tracker.increment()

        # Process all embedding tasks in parallel/sequential according to limiter
        embedding_tasks = [process_embeddings(idx, job) for idx, job in enumerate(jobs_for_embeddings)]
        await asyncio.gather(*embedding_tasks)
        embedding_tracker.stop()

        if is_lm_studio_embeddings:
            embeddings_model = ai_engine.embeddings_model
            print(f"[Model Mgmt] Unloading LM Studio embeddings model: {embeddings_model} ...")
            ai_engine.unload_model(provider_name=_embeddings_llm, model_name=embeddings_model)
            unloaded = ai_engine.wait_for_model_unloaded(provider_name=_embeddings_llm, model_name=embeddings_model)
            if not unloaded:
                print("[Model Mgmt] WARNING: embeddings model may not be fully unloaded yet.")

        # Save embeddings immediately to database
        if dp and not skip_db:
            print("Saving embeddings to database...")
            embedding_updates = []
            for job in jobs_for_embeddings:
                if not job.get('metadata', {}).get('in_db', False):
                    continue
                emb = job.get('embeddings', {})
                embedding_updates.append({
                    "job_id": job['metadata']['job_id'],
                    "title_embedding": emb.get('title_vector'),
                    "requirements_embedding": emb.get('requirements_vector'),
                    "responsibilities_embedding": emb.get('responsibilities_vector'),
                    "description_embedding": emb.get('description_vector'),
                    "pay_embedding": emb.get('pay_vector'),
                    "location_embedding": emb.get('location_vector'),
                })
            if embedding_updates:
                dp.save_job_embeddings(embedding_updates)
                print(f"Saved {len(embedding_updates)} jobs' embeddings.")
                try:
                    from app.rag_engine import RAGEngine
                    rag = RAGEngine(dp=dp, ai_engine=ai_engine)
                    for job in jobs_for_embeddings:
                        rag.index_job(job)
                    print(f"Indexed {len(jobs_for_embeddings)} jobs into RAG document store.")
                except Exception as e:
                    print(f"Warning: RAG indexing failed in Stage 2: {e}")

    print("AI/LLM Passes Complete.")
    print(f"Stage 2 complete: {len(processed_job_pool)} jobs processed.")
    return processed_job_pool


async def pipeline_stage_rule_filter(jobs: List[Dict], user_preferences: Optional[dict] = None,
                                     dp: Optional[DataPuller] = None,
                                     skip_db: bool = False,
                                     verbose: bool = False) -> List[Dict]:
    """
    Stage 3: Apply hard-constraint rule filtering to the job pool.
    
    Args:
        jobs: List of job dicts from Stage 2.
        user_preferences: Dict of user preferences (work types, seniority, pay, timezones).
        dp: DataPuller instance for DB sync.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
    
    Returns:
        List[Dict] of all jobs (processed_job_pool) with 'skip' boolean added,
        plus the 'active_job_pool' key if returning the full context.
    """
    print("=" * 50)
    print("PIPELINE STAGE 3: RULE FILTERING")
    print("=" * 50)

    if user_preferences is None:
        user_preferences = {}

    print(f"Starting Rule-Based Filtering on {len(jobs)} jobs...")

    for job in jobs:
        job['skip'] = apply_rule_filters(job, user_preferences)

    skipped_count = sum(1 for j in jobs if j['skip'])
    print(f"Filtering complete. {skipped_count} jobs skipped, {len(jobs) - skipped_count} active.")

    # Sync skip status to database
    if dp and not skip_db:
        print("Syncing skip status to database...")
        skipped_ids = [job['metadata']['job_id'] for job in jobs if job.get('skip')]
        if skipped_ids:
            dp.bulk_update_skip_status(skipped_ids)
            print(f"Updated {len(skipped_ids)} jobs as skipped.")
        else:
            print("No jobs to skip in database.")

    print(f"Stage 3 complete. Active jobs: {len(jobs) - skipped_count}")
    return jobs


async def pipeline_stage_archetype_integration(jobs: List[Dict], ai_engine: Optional[AIEngine] = None,
                                                dp: Optional[DataPuller] = None,
                                                setup_data: Optional[dict] = None,
                                                skip_db: bool = False,
                                                verbose: bool = False) -> Tuple[List[Dict], ArchetypeManager]:
    """
    Stage 4: Load/integrate archetypes (benchmarks + user profile/resume).
    
    Args:
        jobs: List of job dicts from Stage 3 (with skip flags).
        ai_engine: AIEngine for generating archetype embeddings.
        dp: DataPuller for DB operations.
        setup_data: Setup data containing resume/user_profile text.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        Tuple of (active_jobs, archetype_manager).
    """
    print("=" * 50)
    print("PIPELINE STAGE 4: ARCHETYPE ENGINE INTEGRATION")
    print("=" * 50)

    if ai_engine is None:
        ai_engine = AIEngine(default_provider_name="lm_studio")

    _user_config = _load_user_config()
    _embeddings_llm = _user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))

    from app.vector_engine import CallableEmbeddingProvider
    embed_fn = lambda text: generate_embeddings(ai_engine, text, provider_name=_embeddings_llm)
    embedding_provider = CallableEmbeddingProvider(embed_fn)
    archetype_manager = ArchetypeManager(embedding_provider=embedding_provider)

    # Get active (non-skipped) jobs
    active_jobs = [j for j in jobs if not j.get('skip')]
    print(f"Active jobs for archetype comparison: {len(active_jobs)}")

    print("Synchronizing candidate archetypes and benchmarks...")

    # Extract structured profile data using LLM with file-modification caching
    resume_text = ""
    profile_text = ""
    if setup_data:
        resume_text = setup_data.get("resume", "")
        profile_text = setup_data.get("user_profile", "")

    resume_path_env = os.getenv("RESUME", "")
    resume_path_resolved = resume_path_env if (resume_path_env and os.path.exists(resume_path_env)) else "archetype_profiles/resume_cache.json"

    profile_path_env = os.getenv("PROFILE", "")
    profile_path_resolved = profile_path_env if (profile_path_env and os.path.exists(profile_path_env)) else "archetype_profiles/user_profile_cache.json"

    resume_data = extract_and_cache_profile(
        ai_engine,
        resume_path_resolved,
        resume_text if resume_text else "No resume text provided.",
        "archetype_profiles/resume_cache.json"
    )
    profile_data = extract_and_cache_profile(
        ai_engine,
        profile_path_resolved,
        profile_text if profile_text else "No profile text provided.",
        "archetype_profiles/user_profile_cache.json"
    )

    resume_mtime = None
    if resume_path_resolved and os.path.exists(resume_path_resolved):
        resume_mtime = os.path.getmtime(resume_path_resolved)

    profile_mtime = None
    if profile_path_resolved and os.path.exists(profile_path_resolved):
        profile_mtime = os.path.getmtime(profile_path_resolved)

    all_archetypes = ARCHETYPES_CONFIG + [
        {
            "name": "Resume",
            "title": "",
            "requirements": "\n".join(resume_data.get("requirements", [])),
            "responsibilities": "\n".join(resume_data.get("responsibilities", [])),
            "summary": resume_data.get("summary", ""),
            "type": "resume",
            "source_mtime": resume_mtime
        },
        {
            "name": "User Profile",
            "title": "",
            "requirements": "\n".join(profile_data.get("requirements", [])),
            "responsibilities": "\n".join(profile_data.get("responsibilities", [])),
            "summary": profile_data.get("summary", ""),
            "type": "user_profile",
            "source_mtime": profile_mtime
        }
    ]

    # Determine the expected dimension of the current embedding provider
    test_emb = generate_embeddings(ai_engine, "test", provider_name=_embeddings_llm)
    expected_dim = len(test_emb) if test_emb else 0

    for arch_config in all_archetypes:
        cached_arch = None
        if dp:
            cached_arch = dp.get_archetype_embeddings(arch_config['name'])

        cache_valid = False
        if cached_arch:
            title_emb = cached_arch.get('title_embedding')
            if title_emb and len(title_emb) == expected_dim:
                source_mtime = arch_config.get("source_mtime")
                if source_mtime is not None:
                    db_meta = cached_arch.get('metadata') or {}
                    db_mtime = db_meta.get("source_mtime")
                    if db_mtime is not None:
                        if db_mtime >= source_mtime:
                            cache_valid = True
                        else:
                            print(f"Archetype '{arch_config['name']}' source file modified since embedding generation. Invalidating cache.")
                    else:
                        # Fallback to date_generated database timestamp comparison
                        db_dt = cached_arch.get('date_generated')
                        if db_dt:
                            import datetime
                            file_dt = datetime.datetime.fromtimestamp(source_mtime)
                            if db_dt.tzinfo is not None:
                                file_dt_tz = datetime.datetime.fromtimestamp(source_mtime, tz=datetime.timezone.utc)
                                if db_dt >= file_dt_tz:
                                    cache_valid = True
                                else:
                                    print(f"Archetype '{arch_config['name']}' source file modified since DB generation timestamp (with tz). Invalidating cache.")
                            else:
                                if db_dt >= file_dt:
                                    cache_valid = True
                                else:
                                    print(f"Archetype '{arch_config['name']}' source file modified since DB generation timestamp. Invalidating cache.")
                        else:
                            print(f"Archetype '{arch_config['name']}' has no modification timestamp metadata or database timestamp. Invalidating cache.")
                else:
                    # Config-driven benchmark archetypes do not map to files
                    cache_valid = True

        if cache_valid:
            archetype_manager.add_archetype(Archetype(
                name=arch_config['name'],
                type=cached_arch['archetype_type'],
                title_embedding=np.array(cached_arch['title_embedding']),
                requirements_embedding=np.array(cached_arch['requirements_embedding']),
                responsibilities_embedding=np.array(cached_arch['responsibilities_embedding']),
                metadata=cached_arch.get('metadata', {})
            ))
        else:
            if cached_arch:
                print(f"Cached archetype '{arch_config['name']}' is invalid (modified source or dimension mismatch). Regenerating...")
            else:
                print(f"Generating new embeddings for archetype: {arch_config['name']}")
            new_arch = archetype_manager.load_archetype(
                name=arch_config['name'],
                archetype_data=arch_config,
                archetype_type=arch_config.get("type", "benchmark"),
                metadata={"source_mtime": arch_config.get("source_mtime")} if arch_config.get("source_mtime") is not None else None
            )
            # Persist to DB for future runs
            if dp and not skip_db:
                dp.save_archetype_embeddings({
                    "archetype_name": new_arch.name,
                    "archetype_type": new_arch.type,
                    "title_embedding": new_arch.title_embedding.tolist() if new_arch.title_embedding is not None else None,
                    "requirements_embedding": new_arch.requirements_embedding.tolist() if new_arch.requirements_embedding is not None else None,
                    "responsibilities_embedding": new_arch.responsibilities_embedding.tolist() if new_arch.responsibilities_embedding is not None else None,
                    "metadata": json.dumps(new_arch.metadata)
                })

    # Load and synchronize negative archetypes (anti-profiles)
    neg_config = _user_config.get("negative_scoring", {})
    if neg_config.get("enabled", True):
        avoid_titles = neg_config.get("avoid_titles", [])
        avoid_functions = neg_config.get("avoid_functions", [])

        negative_archetypes_config = []
        for title in avoid_titles:
            if title:
                negative_archetypes_config.append({
                    "name": f"Avoid Title: {title}",
                    "title": title,
                    "requirements": title,
                    "responsibilities": title,
                    "type": "negative_title"
                })
        for func in avoid_functions:
            if func:
                negative_archetypes_config.append({
                    "name": f"Avoid Function: {func[:30]}",
                    "title": func,
                    "requirements": func,
                    "responsibilities": func,
                    "type": "negative_function"
                })

        for neg_arch in negative_archetypes_config:
            cached_arch = None
            if dp:
                cached_arch = dp.get_archetype_embeddings(neg_arch['name'])
            
            cache_valid = False
            if cached_arch:
                title_emb = cached_arch.get('title_embedding')
                if title_emb and len(title_emb) == expected_dim:
                    cache_valid = True

            if cache_valid:
                archetype_manager.add_negative_archetype(Archetype(
                    name=neg_arch['name'],
                    type=cached_arch['archetype_type'],
                    title_embedding=np.array(cached_arch['title_embedding']),
                    requirements_embedding=np.array(cached_arch['requirements_embedding']),
                    responsibilities_embedding=np.array(cached_arch['responsibilities_embedding']),
                    metadata=cached_arch.get('metadata', {})
                ))
            else:
                if cached_arch:
                    print(f"Cached negative archetype '{neg_arch['name']}' invalid. Regenerating...")
                else:
                    print(f"Generating new embeddings for negative archetype: {neg_arch['name']}")
                embeddings = archetype_manager.vector_engine.get_embeddings([
                    neg_arch['title'], neg_arch['requirements'], neg_arch['responsibilities']
                ])
                new_neg_arch = Archetype(
                    name=neg_arch['name'],
                    type=neg_arch['type'],
                    title_embedding=embeddings[0],
                    requirements_embedding=embeddings[1],
                    responsibilities_embedding=embeddings[2]
                )
                archetype_manager.add_negative_archetype(new_neg_arch)
                if dp and not skip_db:
                    dp.save_archetype_embeddings({
                        "archetype_name": new_neg_arch.name,
                        "archetype_type": new_neg_arch.type,
                        "title_embedding": new_neg_arch.title_embedding.tolist() if new_neg_arch.title_embedding is not None else None,
                        "requirements_embedding": new_neg_arch.requirements_embedding.tolist() if new_neg_arch.requirements_embedding is not None else None,
                        "responsibilities_embedding": new_neg_arch.responsibilities_embedding.tolist() if new_neg_arch.responsibilities_embedding is not None else None,
                        "metadata": json.dumps(new_neg_arch.metadata)
                    })

    # Index candidate profile and resume text into RAG store
    if setup_data and dp and not skip_db:
        try:
            from app.rag_engine import RAGEngine
            rag = RAGEngine(dp=dp, ai_engine=ai_engine)
            res_txt = setup_data.get("resume", "")
            prof_txt = setup_data.get("user_profile", "")
            if res_txt:
                rag.index_candidate_profile(res_txt, source_label="candidate_resume")
            if prof_txt:
                rag.index_candidate_profile(prof_txt, source_label="candidate_profile")
        except Exception as e:
            print(f"Warning: RAG candidate profile indexing failed in Stage 4: {e}")

    print(f"Stage 4 complete: Loaded {len(archetype_manager.archetypes)} positive and {len(archetype_manager.negative_archetypes)} negative archetypes.")
    return active_jobs, archetype_manager



async def pipeline_stage_vector_scoring(jobs: List[Dict], archetype_manager: Optional[ArchetypeManager] = None,
                                         dp: Optional[DataPuller] = None,
                                         skip_db: bool = False,
                                         verbose: bool = False) -> List[Dict]:
    """
    Stage 5: Compare jobs to archetypes, compute semantic scores, and filter by threshold.
    
    Args:
        jobs: List of active job dicts from Stage 4.
        archetype_manager: ArchetypeManager with loaded archetypes.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of filtered jobs with semantic scores and archetype matches.
    """
    print("=" * 50)
    print("PIPELINE STAGE 5: VECTOR SCORING WITH ARCHETYPES")
    print("=" * 50)

    # Check and load DB status
    from app.pipeline.pipeline_utils import ensure_db_status_loaded
    ensure_db_status_loaded(jobs, dp, skip_db)

    # Exclude jobs that already have all three results
    jobs_to_process = [j for j in jobs if not j.get('_db_has_all_three')]

    if archetype_manager is None:
        archetype_manager = ArchetypeManager()

    jobs_needing_vs = [j for j in jobs_to_process if not j.get('_db_has_vs')]

    if jobs_needing_vs:
        print(f"Comparing {len(jobs_needing_vs)} jobs to archetypes...")
        for index, job in enumerate(jobs_needing_vs):
            if verbose:
                print(f"Processing job {index + 1}/{len(jobs_needing_vs)}: {job['features']['title']}")

            matches = archetype_manager.compare_job_to_archetypes(job)
            neg_result = archetype_manager.compare_job_to_negative_archetypes(job)
            job['retrieval_metadata'] = archetype_manager.generate_retrieval_metadata(job, matches)
            job['archetype_matches'] = matches
            job['negative_match'] = neg_result
        print("Archetype comparison complete.")
    else:
        print("No jobs need archetype comparison (already scored in DB).")

    # Import adjustment functions from vector_engine
    from app.vector_engine import apply_keyword_adjustments, apply_metadata_adjustments

    print("Applying weighted semantic scoring...")
    for job in jobs_to_process:
        if job.get('_db_has_vs'):
            # Already has vector score in DB, so it is loaded and populated
            continue

        matches = job.get('archetype_matches', [])
        if not matches:
            continue

        best_match = matches[0]

        title_similarity = best_match.get('title_similarity', 0.0)
        requirements_similarity = best_match.get('requirements_similarity', 0.0)
        responsibility_similarity = best_match.get('responsibility_similarity', 0.0)

        # Weighted positive semantic score
        positive_score = (
            0.40 * title_similarity +
            0.35 * requirements_similarity +
            0.25 * responsibility_similarity
        )

        # Apply negative vector score penalty if configured
        _user_config = _load_user_config()
        neg_config = _user_config.get("negative_scoring", {})
        neg_enabled = neg_config.get("enabled", True)
        penalty_weight = float(neg_config.get("penalty_weight", 0.35))
        sim_threshold = float(neg_config.get("similarity_threshold", 0.50))

        neg_result = job.get('negative_match')
        if not neg_result and archetype_manager:
            try:
                neg_result = archetype_manager.compare_job_to_negative_archetypes(job)
                job['negative_match'] = neg_result
            except Exception:
                neg_result = {}

        max_neg_sim = 0.0
        worst_neg_name = ''
        if isinstance(neg_result, dict):
            raw_sim = neg_result.get('max_negative_similarity', 0.0)
            worst_neg_name = str(neg_result.get('worst_match_name', ''))
            if isinstance(raw_sim, (int, float)):
                max_neg_sim = float(raw_sim)

        negative_penalty = 0.0
        if neg_enabled and max_neg_sim > sim_threshold:
            excess_similarity = (max_neg_sim - sim_threshold) / max(0.001, (1.0 - sim_threshold))
            negative_penalty = min(1.0, excess_similarity) * penalty_weight


        semantic_score = positive_score - negative_penalty

        # Keyword adjustments
        job_requirements = job.get('features', {}).get('requirements', [])
        job_title = job.get('features', {}).get('title', '')
        semantic_score = apply_keyword_adjustments(semantic_score, job_requirements, job_title)

        # Metadata adjustments
        job_meta = {
            "is_remote": (job.get('features', {}).get('work_type') or '').lower() == 'remote',
            "salary": 0,
            "days_old": 30
        }
        pay_range = job.get('features', {}).get('pay', '')
        if pay_range:
            import re as _re
            numbers = _re.findall(r'\d+(?:,\d+)?', pay_range.replace(',', ''))
            if len(numbers) >= 2:
                job_meta["salary"] = int(numbers[1])
            elif len(numbers) == 1:
                job_meta["salary"] = int(numbers[0])

        semantic_score = apply_metadata_adjustments(semantic_score, job_meta)

        # Clamp and normalize
        semantic_score = max(0.0, min(1.0, semantic_score))
        score_percent = int(round(semantic_score * 100))

        job['semantic_score'] = semantic_score
        job['semantic_score_percent'] = score_percent
        job['title_similarity'] = title_similarity
        job['requirements_similarity'] = requirements_similarity
        job['responsibility_similarity'] = responsibility_similarity
        job['adjusted_score'] = semantic_score
        job['best_archetype'] = best_match.get('archetype_name', '')

    # Generate retrieval metadata
    for job in jobs_to_process:
        if 'retrieval_metadata' not in job:
            job['retrieval_metadata'] = {}
        if 'semantic_score' in job:
            job['retrieval_metadata']['semantic_score'] = job['semantic_score']
            job['retrieval_metadata']['semantic_score_percent'] = job['semantic_score_percent']
            job['retrieval_metadata']['best_archetype'] = job.get('best_archetype', '')
            neg_match = job.get('negative_match')
            if isinstance(neg_match, dict):
                max_neg = neg_match.get('max_negative_similarity', 0.0)
                if isinstance(max_neg, (int, float)) and max_neg > 0:
                    job['retrieval_metadata']['negative_penalty'] = round(float(max_neg), 4)
                    job['retrieval_metadata']['worst_negative_match'] = str(neg_match.get('worst_match_name', ''))



    # Filter by threshold (dynamic floor of 0.55, padded to target count)
    MIN_SCORE_THRESHOLD = 0.55
    TARGET_COUNT = 50

    filtered_job_pool = []
    for job in jobs_to_process:
        if job.get('semantic_score', 0) >= MIN_SCORE_THRESHOLD:
            filtered_job_pool.append(job)

    # Fallback: add top-X if not enough jobs meet threshold
    if len(filtered_job_pool) < TARGET_COUNT:
        sorted_jobs = sorted(jobs_to_process, key=lambda x: x.get('semantic_score', 0), reverse=True)
        top_n = max(TARGET_COUNT - len(filtered_job_pool), 1)
        for job in sorted_jobs[:top_n]:
            if job not in filtered_job_pool:
                filtered_job_pool.append(job)

    print(f"Filtered to {len(filtered_job_pool)} jobs (threshold >= {MIN_SCORE_THRESHOLD}).")

    # Persist vector scores
    if dp and not skip_db:
        try:
            # Only save newly calculated scores
            newly_scored_jobs = [j for j in jobs_to_process if not j.get('_db_has_vs')]
            if newly_scored_jobs:
                sorted_all_jobs = sorted(newly_scored_jobs, key=lambda x: x.get('semantic_score', 0), reverse=True)
                dp.save_vector_scores(sorted_all_jobs)
                print(f"Persisted {len(sorted_all_jobs)} new vector scores to database.")
            else:
                print("No new vector scores to persist.")
        except Exception as e:
            error_logger_continue(f"Vector score persistence failed: {e}")

    # Print top jobs
    sorted_filtered = sorted(filtered_job_pool, key=lambda x: x.get('semantic_score', 0), reverse=True)
    print("\n=== Semantic Score-Based Job Shortlisting ===")
    for i, job in enumerate(sorted_filtered[:5]):
        print(f"{i+1}. {job['features']['title']}")
        print(f"   Score: {job.get('semantic_score_percent', 'N/A')}% | Archetype: {job.get('best_archetype', 'None')}")

    print(f"Stage 5 complete: {len(filtered_job_pool)} jobs in filtered pool.")
    return filtered_job_pool


async def pipeline_stage_cheap_llm(jobs: List[Dict], setup_data: Optional[dict] = None,
                                    dp: Optional[DataPuller] = None,
                                    skip_db: bool = False,
                                    verbose: bool = False) -> List[Dict]:
    """
    Stage 6: Run cheap (fast) LLM classification on the filtered job pool.
    
    Args:
        jobs: List of filtered job dicts from Stage 5.
        setup_data: Setup data containing user_profile and requirements.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of shortlisted jobs with 'cheap_llm_result' added.
    """
    print("=" * 50)
    print("PIPELINE STAGE 6: CHEAP LLM CLASSIFICATION")
    print("=" * 50)

    # Check and load DB status
    from app.pipeline.pipeline_utils import ensure_db_status_loaded
    ensure_db_status_loaded(jobs, dp, skip_db)

    # Exclude jobs that already have all three results
    jobs_to_process = [j for j in jobs if not j.get('_db_has_all_three')]

    user_profile = setup_data.get("user_profile", "") if setup_data else ""
    requirements = setup_data.get("requirements", []) if setup_data else []

    if not user_profile:
        print("Warning: No user profile available for classification.")

    # Initialize cheap LLM classifier
    _user_config = _load_user_config()
    cheap_llm_provider = _user_config.get("cheap_llm_provider", os.getenv("CHEAP_LLM_PROVIDER", "gemini"))
    cheap_llm_model = _user_config.get("cheap_llm_model", os.getenv("CHEAP_LLM_MODEL"))
    cheap_classifier = CheapLLMClassifier(provider=cheap_llm_provider, model=cheap_llm_model)

    # Separate into already classified and needs classification
    jobs_to_classify = [j for j in jobs_to_process if not j.get('_db_has_clr')]
    jobs_already_classified = [j for j in jobs_to_process if j.get('_db_has_clr')]

    newly_shortlisted = []
    if jobs_to_classify:
        # Run Stage 6 on filtered job pool that needs classification
        newly_shortlisted = await process_stage_6(
            jobs=jobs_to_classify,
            classifier=cheap_classifier,
            candidate_profile=user_profile,
            candidate_requirements=requirements,
            batch_size=5
        )
    else:
        print("No jobs need cheap LLM classification (already processed in DB).")

    # Check decision for already classified jobs
    already_shortlisted = [
        j for j in jobs_already_classified
        if j.get('cheap_llm_result', {}).get('decision') in ('apply', 'maybe')
    ]

    shortlisted_jobs = newly_shortlisted + already_shortlisted

    # Persist results to database
    if dp and not skip_db:
        print("Persisting Stage 6 results to database...")
        try:
            # 1. Save cheap LLM results for ALL jobs that were newly classified
            classified_jobs = [j for j in jobs_to_classify if 'cheap_llm_result' in j]
            if classified_jobs:
                dp.save_cheap_llm_results(classified_jobs)
                print(f"Persisted {len(classified_jobs)} cheap LLM results.")
            else:
                print("No new cheap LLM results to persist.")
            
            # 2. Update job table skip status for newly classified skipped jobs
            skipped_ids = [j['metadata']['job_id'] for j in jobs_to_classify if j.get('cheap_llm_result', {}).get('decision') == 'skip']
            if skipped_ids:
                dp.bulk_update_skip_status(skipped_ids)
                print(f"Marked {len(skipped_ids)} skipped jobs as skip=True in the database.")
        except Exception as e:
            error_logger_continue(f"Stage 6 persistence failed: {e}")

    # 3. Clean up/remove skipped jobs from the in-memory pool in-place to free memory
    shortlisted_ids = {j['metadata']['job_id'] for j in shortlisted_jobs}
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        job_id = job.get('metadata', {}).get('job_id')
        if job_id not in shortlisted_ids:
            jobs.pop(i)
        i -= 1

    print(f"Stage 6 complete: {len(jobs)} active jobs remaining in memory pool.")
    return jobs


async def pipeline_stage_strong_llm(jobs: List[Dict], setup_data: Optional[dict] = None,
                                     dp: Optional[DataPuller] = None,
                                     skip_db: bool = False,
                                     verbose: bool = False) -> List[Dict]:
    """
    Stage 7: Run strong (deep) LLM reranking on top candidates from Stage 6.
    
    Args:
        jobs: List of shortlisted job dicts from Stage 6.
        setup_data: Setup data containing user_profile and requirements.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of deeply analyzed jobs with 'strong_llm_result' added.
    """
    print("=" * 50)
    print("PIPELINE STAGE 7: STRONG LLM RERANKING")
    print("=" * 50)

    # Check and load DB status
    from app.pipeline.pipeline_utils import ensure_db_status_loaded
    ensure_db_status_loaded(jobs, dp, skip_db)

    # Exclude jobs that already have all three results
    jobs_to_process = [j for j in jobs if not j.get('_db_has_all_three')]

    user_profile = setup_data.get("user_profile", "") if setup_data else ""
    requirements = setup_data.get("requirements", []) if setup_data else []

    # Initialize strong LLM reranker
    _user_config = _load_user_config()
    strong_llm_provider = _user_config.get("strong_llm_provider", os.getenv("STRONG_LLM_PROVIDER", "claude"))
    strong_llm_model = _user_config.get("strong_llm_model", os.getenv("STRONG_LLM_MODEL"))
    strong_reranker = StrongLLMReranker(provider=strong_llm_provider, model=strong_llm_model)

    # Configure how many jobs to deeply analyze
    top_n_for_deep_analysis = int(_user_config.get("top_n_deep_analysis", os.getenv("TOP_N_DEEP_ANALYSIS", "15")))

    # Sort all processable jobs by cheap fit score descending
    jobs_sorted = sorted(jobs_to_process, key=lambda x: x.get('cheap_llm_result', {}).get('fit_score', 0), reverse=True)
    top_jobs = jobs_sorted[:top_n_for_deep_analysis]

    # Separate into already reranked and needs reranking
    already_reranked = [j for j in top_jobs if j.get('_db_has_slr')]
    needs_rerank = [j for j in top_jobs if not j.get('_db_has_slr')]

    newly_reranked = []
    if needs_rerank:
        # Run Stage 7 on top candidates that need deep analysis
        newly_reranked = await process_stage_7(
            jobs=needs_rerank,
            reranker=strong_reranker,
            candidate_profile=user_profile,
            candidate_requirements=requirements,
            top_n=len(needs_rerank)
        )
    else:
        print("No jobs need strong LLM reranking (already reranked in DB).")

    deeply_analyzed_jobs = already_reranked + newly_reranked

    # Persist results
    if dp and not skip_db:
        if newly_reranked:
            print("Persisting Stage 7 results to database...")
            try:
                dp.save_strong_llm_results(newly_reranked)
                print(f"Persisted {len(newly_reranked)} strong LLM results.")
                try:
                    from app.rag_engine import RAGEngine
                    from app.ai_engine import AIEngine
                    _embeddings_llm = _user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))
                    embeddings_model = _user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "local-model"))
                    ai_engine_instance = AIEngine(default_provider_name=_embeddings_llm)
                    
                    is_lm_studio_embeddings = _provider_is_lm_studio(ai_engine_instance, _embeddings_llm)
                    if is_lm_studio_embeddings:
                        print(f"[Model Mgmt] Loading LM Studio embeddings model for Stage 7 RAG update: {embeddings_model} ...")
                        ai_engine_instance.load_model(provider_name=_embeddings_llm, model_name=embeddings_model)
                        loaded = ai_engine_instance.wait_for_model_loaded(provider_name=_embeddings_llm, model_name=embeddings_model)
                        if not loaded:
                            print("[Model Mgmt] WARNING: embeddings model may not be fully loaded.")

                    rag = RAGEngine(dp=dp, ai_engine=ai_engine_instance)
                    for job in newly_reranked:
                        rag.index_job(job)

                    if is_lm_studio_embeddings:
                        print(f"[Model Mgmt] Unloading LM Studio embeddings model: {embeddings_model} ...")
                        ai_engine_instance.unload_model(provider_name=_embeddings_llm, model_name=embeddings_model)
                        ai_engine_instance.wait_for_model_unloaded(provider_name=_embeddings_llm, model_name=embeddings_model)

                    print(f"Updated RAG document store for {len(newly_reranked)} deeply analyzed jobs.")
                except Exception as e:
                    print(f"Warning: RAG document update failed in Stage 7: {e}")
            except Exception as e:
                error_logger_continue(f"Stage 7 persistence failed: {e}")
        else:
            print("No new strong LLM results to persist.")

    # Remove jobs from memory pool that did not proceed to deep analysis in-place
    analyzed_ids = {j['metadata']['job_id'] for j in deeply_analyzed_jobs}
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        job_id = job.get('metadata', {}).get('job_id')
        if job_id not in analyzed_ids:
            jobs.pop(i)
        i -= 1

    print(f"Stage 7 complete: {len(jobs)} jobs deeply analyzed and remaining in memory pool.")
    return jobs


async def pipeline_stage_final_queue(jobs: List[Dict], dp: Optional[DataPuller] = None,
                                      skip_db: bool = False,
                                      verbose: bool = False) -> List[Dict]:
    """
    Stage 8: Generate the final ranked application queue from deeply analyzed jobs.
    
    Args:
        jobs: List of deeply analyzed job dicts from Stage 7.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of ranked jobs with final_score, priority, apply_recommendation.
    """
    print("=" * 50)
    print("PIPELINE STAGE 8: FINAL APPLICATION QUEUE")
    print("=" * 50)

    # Check and load DB status
    from app.pipeline.pipeline_utils import ensure_db_status_loaded
    ensure_db_status_loaded(jobs, dp, skip_db)

    if not jobs:
        print("No jobs remaining in memory to process for Stage 8.")
        return []

    # Run Stage 8 to generate final ranked queue
    final_queue = await process_stage_8(jobs=jobs)

    # Persist results
    if dp and not skip_db:
        print("Persisting final application queue to database...")
        try:
            dp.save_final_queue(final_queue)
            print(f"Persisted {len(final_queue)} jobs to final queue.")
            
            # Update job table skip status for jobs recommended to skip
            skipped_ids = [j['metadata']['job_id'] for j in final_queue if j.get('priority') == 'skip' or j.get('apply_recommendation') == 'skip']
            if skipped_ids:
                dp.bulk_update_skip_status(skipped_ids)
                print(f"Marked {len(skipped_ids)} skipped jobs as skip=True in the database.")
        except Exception as e:
            error_logger_continue(f"Final queue persistence failed: {e}")

    # Remove skipped jobs from the memory pool in-place to free memory
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        if job.get('priority') == 'skip' or job.get('apply_recommendation') == 'skip':
            jobs.pop(i)
        i -= 1

    # Filter final_queue returned list to only contain active non-skipped jobs
    active_final_queue = [j for j in final_queue if j.get('priority') != 'skip' and j.get('apply_recommendation') != 'skip']

    # Print detailed final queue (only for non-skipped jobs)
    print("\n" + "=" * 60)
    print("DETAILED FINAL APPLICATION QUEUE")
    print("=" * 60)

    for i, job in enumerate(active_final_queue[:10], 1):
        features = job.get('features', {})
        metadata = job.get('metadata', {})
        cheap_result = job.get('cheap_llm_result', {})
        strong_result = job.get('strong_llm_result', {})

        print(f"\n{i}. {features.get('title', 'Unknown')}")
        print(f"   Job ID: {metadata.get('job_id', 'Unknown')} | Company: {metadata.get('company_name', 'Unknown')}")
        print(f"   Link: {metadata.get('link', '')}")
        print(f"   Priority: {job.get('priority', 'unknown').upper()}")
        print(f"   Final Score: {job.get('final_score', 0):.1f}/100")
        print(f"   Recommendation: {job.get('apply_recommendation', 'maybe').upper()}")
        print(f"   Cheap LLM Fit: {cheap_result.get('fit_score', 0)}/100")
        print(f"   Strong LLM Score: {strong_result.get('final_score', 0)}/100")

    print(f"\nStage 8 complete: {len(active_final_queue)} active jobs in final queue.")
    return active_final_queue


