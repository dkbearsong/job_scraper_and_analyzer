import os
import sys
import argparse
import logging
import asyncio

# Ensure stdout and stderr flush immediately (line-buffered) even when redirected to a log file
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

# Disable tokenizers parallelism to prevent semaphore leaks and fork crashes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from typing import Optional, List, Dict, Tuple, Any, Union

from app.pipeline.stages import (
    pipeline_stage_setup,
    pipeline_stage_scrape,
    pipeline_stage_preliminary_filter,
    pipeline_stage_embed_and_extract,
    pipeline_stage_rule_filter,
    pipeline_stage_archetype_integration,
    pipeline_stage_vector_scoring,
    pipeline_stage_cheap_llm,
    pipeline_stage_strong_llm,
    pipeline_stage_final_queue
)
from app.pipeline.pipeline_utils import (
    _load_jobs_for_stage,
    _parse_stage_range,
    _print_stage_header,
    _move_existing_logs
)
from app.pipeline.maintenance import scrape_missing_descriptions_24h, recalculate_db_final_scores
from app.logger import _log_pipeline_stats
from app.ai_utils import _load_model_if_lm_studio, _unload_model_if_lm_studio
from app.config_utils import _load_user_config
from app.archetype_engine import ArchetypeManager
from app.llm_usage_tracker import usage_tracker
from dotenv import load_dotenv

load_dotenv()

def _merge_job_pools(existing_jobs: List[Dict], db_jobs: List[Dict]) -> List[Dict]:
    if not existing_jobs:
        return db_jobs or []
    if not db_jobs:
        return existing_jobs or []

    merged = {}
    for job in existing_jobs:
        if not job:
            continue
        key = job.get("metadata", {}).get("job_id") or job.get("id") or job.get("metadata", {}).get("link") or job.get("link")
        if key:
            merged[key] = job
    for job in db_jobs:
        if not job:
            continue
        key = job.get("metadata", {}).get("job_id") or job.get("id") or job.get("metadata", {}).get("link") or job.get("link")
        if key and key not in merged:
            merged[key] = job
    return list(merged.values())

async def main(scrape_pages: Optional[int] = None,
               scrape_visible: bool = False,
               scrape_debug: bool = False,
               skip_part_a: Optional[bool] = None,
               skip_job_boards: bool = False,
               stage_range: str = "0-8",
               db_limit: int = 50,
               skip_db: bool = False,
               scrape_missing_24h: bool = False,
               reprocess: Optional[Any] = 0,
               recalculate_final_scores: bool = False,
               rag_query: Optional[str] = None,
               rag_tailor: Optional[int] = None,
               verbose: bool = False):
    """
    Run selected pipeline stages.

    Args:
        scrape_pages: Override max_pages for scraping.
        scrape_visible: Show browser window (disable headless).
        scrape_debug: Enable debug logging during scraping.
        skip_part_a: Skip Part A of legacy fallback scraping.
        skip_job_boards: Skip all adapter and company board scraping and start directly from scraping job descriptions.
        stage_range: Inclusive stage range, e.g. "0-8", "2-4", "2".
        db_limit: Max records to pull from DB in any bulk-load operation.
        skip_db: Skip database operations for all stages.
        scrape_missing_24h: Only scrape descriptions for jobs from the last 24h.
        reprocess: Stage to start reprocessing at or number of days (e.g. 7, 6, 2, "7", "stage=7,days=2", True).
        recalculate_final_scores: Recalculate final scores from DB scores and exit.
        verbose: Enable verbose logging across pipeline stages.

    Returns:
        Pipeline result from the last executed stage.
    """
    # ── Parse stage range & reprocess settings ──
    stage_start, stage_end = _parse_stage_range(stage_range)
    reprocess_stage, reprocess_days = _parse_reprocess_arg(reprocess, stage_start=stage_start)

    # If the user specified a reprocess stage without restricting -s (default "0-8"), align pipeline start
    if reprocess_stage is not None and stage_range == "0-8":
        stage_start = reprocess_stage

    print("=" * 60)
    _print_stage_header("JOB SCRAPING AND ANALYSIS PIPELINE", stage_start, stage_end)
    print(f"DB limit: {db_limit}")
    print("=" * 60)

    # ── Relocate existing log files ──
    _move_existing_logs()

    # Stage 0 is always run (provides essential infrastructure)
    setup_data = await pipeline_stage_setup(skip_db=skip_db, verbose=verbose)
    dp = setup_data["dp"]
    ai = setup_data["ai"]
    tp = setup_data["tp"]

    # Store db_limit in setup_data for downstream stages
    setup_data["db_limit"] = db_limit

    _log_pipeline_stats("0: Setup", 0, source_hint="profile_extraction")

    # ── Handle --reprocess clearing in DB ──
    reprocessed_job_ids: List[int] = []
    if reprocess_stage is not None and dp and not skip_db:
        print("\n" + "=" * 60)
        print(f"[REPROCESS] CLEARING STAGE {reprocess_stage}+ DATA FOR THE LAST {reprocess_days} DAY(S)")
        print("=" * 60)
        cleared_result = dp.clear_jobs_for_reprocessing(stage=reprocess_stage, days=reprocess_days)
        if isinstance(cleared_result, (list, tuple, set)):
            reprocessed_job_ids = list(cleared_result)
        elif hasattr(dp, "last_cleared_job_ids") and dp.last_cleared_job_ids:
            reprocessed_job_ids = list(dp.last_cleared_job_ids)

    if recalculate_final_scores:
        if skip_db or not dp:
            print("[ERROR] Cannot recalculate final scores when skip_db is True.")
            return []
        await recalculate_db_final_scores(dp)
        return []

    if rag_query:
        print("\n=== RAG CORPUS NATURAL LANGUAGE QUERY ===")
        from app.rag_engine import RAGEngine
        rag = RAGEngine(dp=dp, ai_engine=ai)
        result = await rag.query_job_corpus(rag_query, top_k=5)
        print(f"\nQUERY: {result.get('query')}")
        print(f"\nANSWER:\n{result.get('answer')}")
        print("\nSOURCES:")
        for s in result.get('sources', []):
            print(f" - {s.get('job_name')} at {s.get('company_name')} (Sim: {s.get('similarity')}) | {s.get('link')}")
        return result

    if rag_tailor:
        print(f"\n=== RAG APPLICATION TAILORING FOR JOB ID {rag_tailor} ===")
        from app.rag_engine import RAGEngine
        rag = RAGEngine(dp=dp, ai_engine=ai)
        result = await rag.generate_tailored_application(rag_tailor)
        print(json.dumps(result, indent=2))
        return result

    # ── Per-stage execution ──
    processed_job_pool: List[Dict] = []
    active_jobs: List[Dict] = []
    archetype_manager: Optional[ArchetypeManager] = None
    filtered_job_pool: List[Dict] = []
    shortlisted_jobs: List[Dict] = []
    deeply_analyzed_jobs: List[Dict] = []
    final_queue: List[Dict] = []

    # Stage 1: Scrape
    if 1 <= stage_end and stage_start <= 1:
        processed_job_pool = await pipeline_stage_scrape(
            setup_data=setup_data,
            skip_db=skip_db,
            verbose=verbose,
            max_pages=scrape_pages,
            headless=not scrape_visible,
            debug_logging=scrape_debug,
            skip_part_a=skip_part_a,
            db_limit=db_limit,
            scrape_missing_24h=scrape_missing_24h,
            skip_job_boards=skip_job_boards,
        )
        _log_pipeline_stats("1: Scrape", len(processed_job_pool), source_hint="adapters+fallback")
    elif stage_start > 1:
        print(f"[SKIP] Stage 1 (Scrape) — outside range {stage_start}-{stage_end}")

    # Stage 1.5: Preliminary Rule Filter (Early Disqualification)
    if processed_job_pool:
        processed_job_pool = await pipeline_stage_preliminary_filter(
            jobs=processed_job_pool,
            user_preferences=setup_data.get("user_preferences", {}),
            text_processor=tp,
            dp=dp,
            skip_db=skip_db,
            verbose=verbose
        )
        _log_pipeline_stats("1.5: Preliminary Filter", len(processed_job_pool), source_hint="hard_rules")

    # Stage 2: Embedding Generation + LLM Extraction
    if 2 <= stage_end and stage_start <= 2:
        is_reprocess = bool(reprocess_stage and reprocess_stage <= 2)

        if is_reprocess and reprocessed_job_ids and not skip_db and dp:
            print(f"[REPROCESS] Loading all {len(reprocessed_job_ids)} cleared job(s) from DB to restart Stage 2...")
            reprocess_jobs = await _load_jobs_for_stage(
                dp, stage=2, limit=max(len(reprocessed_job_ids), db_limit),
                force_reprocess=True, job_ids=reprocessed_job_ids
            )
            print(f"[REPROCESS] Loaded {len(reprocess_jobs)} jobs from DB for Stage 2.")
            if processed_job_pool:
                processed_job_pool = _merge_job_pools(processed_job_pool, reprocess_jobs)
            else:
                processed_job_pool = reprocess_jobs
        elif not skip_db and dp:
            # Load any jobs from DB that are missing embeddings (but have descriptions) and merge them
            print("Checking DB for any active jobs missing embeddings...")
            db_jobs = await _load_jobs_for_stage(dp, stage=2, limit=db_limit, force_reprocess=False)
            if db_jobs:
                print(f"Found {len(db_jobs)} active jobs in DB missing embeddings. Merging them into the processing pool...")
                processed_job_pool = _merge_job_pools(processed_job_pool, db_jobs)

        if not processed_job_pool and not skip_db and dp:
            print("[INFO] No scraped jobs available. Attempting to load from DB for Stage 2 (Embed+Extract)...")
            processed_job_pool = await _load_jobs_for_stage(dp, stage=2, limit=db_limit, force_reprocess=is_reprocess)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        if processed_job_pool:
            # Check user config for Stage 2 extraction LLM
            _user_config = _load_user_config()
            _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))
            _extraction_model = _user_config.get("extraction_model", os.getenv("EXTRACTION_MODEL", "local-model"))
            
            if _extraction_llm == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 2", model_name=_extraction_model)

            processed_job_pool = await pipeline_stage_embed_and_extract(
                jobs=processed_job_pool,
                ai_engine=ai,
                text_processor=tp,
                dp=dp,
                skip_db=skip_db,
                verbose=verbose,
            )

            if _extraction_llm == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 2", model_name=_extraction_model)
        _log_pipeline_stats("2: Embed+Extract", len(processed_job_pool), source_hint="llm_extraction")
    elif stage_start > 2:
        print(f"[SKIP] Stage 2 (Embed+Extract) — outside range {stage_start}-{stage_end}")

    # Stage 3: Rule Filtering
    if 3 <= stage_end and stage_start <= 3:
        if not processed_job_pool:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 3 (Rule Filter)...")
            processed_job_pool = await _load_jobs_for_stage(dp, stage=3, limit=db_limit)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        if processed_job_pool:
            processed_job_pool = await pipeline_stage_rule_filter(
                jobs=processed_job_pool,
                user_preferences=setup_data.get("user_preferences", {}),
                dp=dp,
                skip_db=skip_db,
            )
        skipped_count = sum(1 for j in processed_job_pool if j.get('skip'))
        _log_pipeline_stats("3: Rule Filter", len(processed_job_pool), skipped_count=skipped_count, source_hint="hard_constraints")
    elif stage_start > 3:
        print(f"[SKIP] Stage 3 (Rule Filter) — outside range {stage_start}-{stage_end}")

    # Stage 4: Archetype Engine Integration
    if 4 <= stage_end and stage_start <= 4:
        if not processed_job_pool:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 4 (Archetype)...")
            processed_job_pool = await _load_jobs_for_stage(dp, stage=4, limit=db_limit)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        _load_model_if_lm_studio(ai, stage_label="Stage 4")
        active_jobs, archetype_manager = await pipeline_stage_archetype_integration(
            jobs=processed_job_pool,
            ai_engine=ai,
            dp=dp,
            setup_data=setup_data,
            skip_db=skip_db,
        )
        _unload_model_if_lm_studio(ai, stage_label="Stage 4")
        _log_pipeline_stats("4: Archetypes", len(active_jobs), source_hint="archetype_comparison")
    elif stage_start > 4:
        print(f"[SKIP] Stage 4 (Archetype) — outside range {stage_start}-{stage_end}")
        # Automatically initialize archetype manager if running Stage 5 next
        if stage_start == 5:
            print("[INFO] Initializing archetype manager for Stage 5...")
            _, archetype_manager = await pipeline_stage_archetype_integration(
                jobs=[],
                ai_engine=ai,
                dp=dp,
                setup_data=setup_data,
                skip_db=skip_db,
            )

    # Stage 5: Vector Scoring WITH ARCHETYPES
    if 5 <= stage_end and stage_start <= 5:
        if not active_jobs:
            if not processed_job_pool:
                print("[INFO] No jobs in memory. Attempting to load from DB for Stage 5 (Vector Scoring)...")
                processed_job_pool = await _load_jobs_for_stage(dp, stage=5, limit=db_limit)
                print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")
            active_jobs = [j for j in processed_job_pool if not j.get('skip')]

        if active_jobs and archetype_manager:
            filtered_job_pool = await pipeline_stage_vector_scoring(
                jobs=active_jobs,
                archetype_manager=archetype_manager,
                dp=dp,
                skip_db=skip_db,
            )
        _log_pipeline_stats("5: Vector Scoring", len(filtered_job_pool), source_hint="semantic_threshold")
    elif stage_start > 5:
        print(f"[SKIP] Stage 5 (Vector Score) — outside range {stage_start}-{stage_end}")


    # Stage 6: Cheap LLM Classification
    if 6 <= stage_end and stage_start <= 6:
        if not skip_db and dp:
            db_jobs = await _load_jobs_for_stage(dp, stage=6, limit=db_limit)
            if db_jobs:
                print(f"[INFO] Merging {len(db_jobs)} active jobs from DB waiting for Stage 6 (Cheap LLM)...")
                filtered_job_pool = _merge_job_pools(filtered_job_pool, db_jobs)

        if filtered_job_pool:
            _user_config = _load_user_config()
            cheap_llm_provider = _user_config.get("cheap_llm_provider", os.getenv("CHEAP_LLM_PROVIDER", "gemini"))
            cheap_llm_model = _user_config.get("cheap_llm_model", os.getenv("CHEAP_LLM_MODEL"))

            if cheap_llm_provider == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 6", model_name=cheap_llm_model)

            shortlisted_jobs = await pipeline_stage_cheap_llm(
                jobs=filtered_job_pool,
                setup_data=setup_data,
                dp=dp,
                skip_db=skip_db,
            )

            if cheap_llm_provider == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 6", model_name=cheap_llm_model)
        _log_pipeline_stats("6: Cheap LLM", len(shortlisted_jobs), source_hint="cheap_llm_classifier")
    elif stage_start > 6:
        print(f"[SKIP] Stage 6 (Cheap LLM) — outside range {stage_start}-{stage_end}")

    # Stage 7: Strong LLM Reranking
    if 7 <= stage_end and stage_start <= 7:
        if not skip_db and dp:
            db_jobs = await _load_jobs_for_stage(dp, stage=7, limit=db_limit)
            if db_jobs:
                print(f"[INFO] Merging {len(db_jobs)} active jobs from DB waiting for Stage 7 (Strong LLM)...")
                shortlisted_jobs = _merge_job_pools(shortlisted_jobs, db_jobs)

        if shortlisted_jobs:
            _user_config = _load_user_config()
            strong_llm_provider = _user_config.get("strong_llm_provider", os.getenv("STRONG_LLM_PROVIDER", "gemini"))
            strong_llm_model = _user_config.get("strong_llm_model", os.getenv("STRONG_LLM_MODEL"))

            if strong_llm_provider == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 7", model_name=strong_llm_model)

            deeply_analyzed_jobs = await pipeline_stage_strong_llm(
                jobs=shortlisted_jobs,
                setup_data=setup_data,
                dp=dp,
                skip_db=skip_db,
            )

            if strong_llm_provider == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 7", model_name=strong_llm_model)
        _log_pipeline_stats("7: Strong LLM", len(deeply_analyzed_jobs), source_hint="strong_llm_reranker")
    elif stage_start > 7:
        print(f"[SKIP] Stage 7 (Strong LLM) — outside range {stage_start}-{stage_end}")

    # Stage 8: Final Application Queue
    if 8 <= stage_end and stage_start <= 8:
        if not skip_db and dp:
            db_jobs = await _load_jobs_for_stage(dp, stage=8, limit=db_limit)
            if db_jobs:
                print(f"[INFO] Merging {len(db_jobs)} active jobs from DB waiting for Stage 8 (Final Queue)...")
                deeply_analyzed_jobs = _merge_job_pools(deeply_analyzed_jobs, db_jobs)

        if deeply_analyzed_jobs:
            final_queue = await pipeline_stage_final_queue(
                jobs=deeply_analyzed_jobs,
                dp=dp,
                skip_db=skip_db,
            )
        _log_pipeline_stats("8: Final Queue", len(final_queue), source_hint="final_ranking")
    elif stage_start > 8:
        print(f"[SKIP] Stage 8 (Final Queue) — outside range {stage_start}-{stage_end}")

    # ── Summary and Return logic ──
    if not skip_db and dp:
        try:
            import datetime
            import time
            run_id = f"cli_{int(time.time())}"
            dp.save_token_usage(run_id, datetime.datetime.now(), usage_tracker.all_records)
        except Exception as e:
            print(f"[WARNING] Could not save token usage: {e}")

    if stage_end == 8 and final_queue:
        # Pipeline Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total jobs processed: {len(processed_job_pool) or len(active_jobs)}")
        print(f"Jobs after semantic filtering: {len(filtered_job_pool)}")
        print(f"Jobs after cheap LLM classification: {len(shortlisted_jobs)}")
        print(f"Jobs after strong LLM reranking: {len(deeply_analyzed_jobs)}")
        apply_jobs_count = sum(1 for j in final_queue if j.get('apply_recommendation', '').lower() == 'apply')
        print(f"Final application queue: {apply_jobs_count} jobs")

        # LLM Usage Summary
        usage_tracker.print_summary()
        return final_queue

    # Summary for partial runs
    print("\n" + "=" * 60)
    print("PIPELINE RUN COMPLETE")
    print(f"Stages executed: {stage_start}-{stage_end}")
    print("=" * 60)

    # LLM Usage Summary
    usage_tracker.print_summary()

    # Return the most relevant result
    if stage_end == 7:
        return deeply_analyzed_jobs
    elif stage_end == 6:
        return shortlisted_jobs
    elif stage_end == 5:
        return filtered_job_pool
    elif stage_end == 4:
        return active_jobs
    else:
        return processed_job_pool


def _parse_stage_range(raw: Optional[str]) -> Tuple[int, int]:
    """
    Parse a stage range string into an inclusive (start, end) tuple.

    Accepts:
        "0-8"  -> (0, 8)
        "2-4"  -> (2, 4)
        "2"    -> (2, 2)
        "all"  -> (0, 8)
        None   -> (0, 8)

    Returns:
        Tuple of (start_stage, end_stage), both inclusive.
    """
    if not raw or str(raw).lower() in ("all", "full"):
        return (0, 8)

    raw_str = str(raw).strip()
    if "-" in raw_str:
        parts = raw_str.split("-", 1)
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
        except (ValueError, IndexError):
            print(f"Warning: invalid stage range '{raw}'. Defaulting to 0-8.")
            return (0, 8)
    else:
        try:
            n = int(raw_str)
            start = n
            end = n
        except ValueError:
            print(f"Warning: invalid stage '{raw}'. Defaulting to 0-8.")
            return (0, 8)

    # Clamp to valid range
    start = max(0, min(start, 8))
    end = max(0, min(end, 8))
    if start > end:
        start, end = end, start

    return (start, end)

def _parse_reprocess_arg(reprocess_val: Any, stage_start: int = 1) -> Tuple[Optional[int], int]:
    """
    Parses the reprocess CLI argument into (reprocess_stage, reprocess_days).
    Returns (None, 0) if reprocessing is disabled.
    """
    if reprocess_val is None or reprocess_val is False or reprocess_val == 0 or reprocess_val == "0":
        return None, 0

    reprocess_stage = 2
    reprocess_days = 1

    if reprocess_val is True or (isinstance(reprocess_val, str) and reprocess_val.lower() in ("true", "yes")):
        # If flag is given without value (e.g. --reprocess), use stage_start if >= 2 else 2
        reprocess_stage = stage_start if stage_start >= 2 else 2
        reprocess_days = 1
    elif isinstance(reprocess_val, int):
        if 2 <= reprocess_val <= 8:
            reprocess_stage = reprocess_val
            reprocess_days = 1
        elif reprocess_val == 1:
            reprocess_stage = stage_start if stage_start >= 2 else 2
            reprocess_days = 1
        else:
            # If > 8, assume it was days for stage 2
            reprocess_stage = 2
            reprocess_days = reprocess_val
    elif isinstance(reprocess_val, str):
        val = reprocess_val.strip()
        import re
        stage_match = re.search(r"stage\s*=\s*(\d+)", val, re.IGNORECASE)
        days_match = re.search(r"days?\s*=\s*(\d+)", val, re.IGNORECASE)

        if stage_match or days_match:
            if stage_match:
                reprocess_stage = int(stage_match.group(1))
            else:
                reprocess_stage = stage_start if stage_start >= 2 else 2
            if days_match:
                reprocess_days = int(days_match.group(1))
            else:
                reprocess_days = 1
        elif ":" in val:
            parts = val.split(":")
            reprocess_stage = int(parts[0]) if parts[0].isdigit() else 2
            reprocess_days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        elif "," in val:
            parts = val.split(",")
            reprocess_stage = int(parts[0]) if parts[0].isdigit() else 2
            reprocess_days = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 1
        elif val.isdigit():
            num = int(val)
            if 2 <= num <= 8:
                reprocess_stage = num
                reprocess_days = 1
            else:
                reprocess_stage = 2
                reprocess_days = num
        else:
            reprocess_stage = stage_start if stage_start >= 2 else 2
            reprocess_days = 1

    return reprocess_stage, reprocess_days


def _print_stage_header(title: str, stage_start: int, stage_end: int) -> None:
    """Print a centered pipeline header with stage info."""
    print(f"{title:^60}")
    if stage_start == 0 and stage_end == 8:
        print("Running full pipeline (stages 0-8)")
    elif stage_start == stage_end:
        print(f"Running stage {stage_start} only")
    else:
        print(f"Running stages {stage_start}-{stage_end}")


# Entry point
if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Job Scraping and Analysis Pipeline"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Max pages per URL for Stage 1 scraping (overrides config).",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window during scraping (disables headless mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a file where logs should be written.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database persistence for all stages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--skip-part-a",
        action="store_true",
        dest="skip_part_a",
        help="Skip Part A (company career-page scraping) in the legacy fallback path and jump straight to description scraping.",
    )
    parser.add_argument(
        "--skip-job-boards",
        "--scrape-descriptions-only",
        action="store_true",
        dest="skip_job_boards",
        help="Skip adapter and company board scraping in Stage 1, starting directly from scraping job descriptions from DB, and continue through the pipeline.",
    )
    parser.add_argument(
        "-s", "--stage",
        type=str,
        default="0-8",
        dest="stage_range",
        help="Stage range to run, e.g. '0-8' (all), '2-4', '2' (single stage). Stages: 0=Setup, 1=Scrape, 2=Embed, 3=Filter, 4=Archetype, 5=Vector, 6=CheapLLM, 7=StrongLLM, 8=FinalQueue",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=50,
        dest="db_limit",
        help="Max records to pull from database in any bulk-load operation (default: 50).",
    )
    parser.add_argument(
        "--scrape-missing-24h",
        action="store_true",
        dest="scrape_missing_24h",
        help="At Step 1, ONLY try to scrape descriptions for jobs from the previous 24h that do not have them.",
    )
    parser.add_argument(
        "--reprocess",
        type=str,
        nargs="?",
        const="true",
        default=None,
        dest="reprocess",
        help="Specify stage to start reprocessing at (e.g. --reprocess 7, --reprocess 6, --reprocess 2, or 'stage=7,days=2'). Clears downstream data in records pulled for that day so stages can be re-run.",
    )
    parser.add_argument(
        "--recalculate-final-scores",
        action="store_true",
        dest="recalculate_final_scores",
        help="Recalculate final scores and application queue from stored database scores without running pipeline stages.",
    )
    parser.add_argument(
        "--rag-query",
        type=str,
        default=None,
        dest="rag_query",
        help="Run a natural language query over the RAG document corpus.",
    )
    parser.add_argument(
        "--rag-tailor",
        type=int,
        default=None,
        dest="rag_tailor",
        help="Generate a RAG-based application tailoring strategy for a specific Job ID.",
    )

    args = parser.parse_args()

    # Always set root logger level and add logs/app_error.log
    root = logging.getLogger()
    log_level = logging.DEBUG if args.debug else logging.INFO
    root.setLevel(log_level)

    import os
    os.makedirs("logs", exist_ok=True)
    
    error_handler = logging.FileHandler(os.path.join("logs", "app_error.log"), mode="a", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(message)s"
    )
    error_handler.setFormatter(error_formatter)
    root.addHandler(error_handler)

    # Apply custom log file if specified
    if args.log_file:
        log_file_path = args.log_file
        if not os.path.isabs(log_file_path) and not log_file_path.startswith("logs/"):
            log_file_path = os.path.join("logs", log_file_path)
            
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    import asyncio
    asyncio.run(main(
        scrape_pages=args.pages,
        scrape_visible=args.visible,
        scrape_debug=args.debug,
        skip_part_a=args.skip_part_a,
        skip_job_boards=args.skip_job_boards,
        stage_range=args.stage_range,
        db_limit=args.db_limit,
        skip_db=args.skip_db,
        scrape_missing_24h=args.scrape_missing_24h,
        reprocess=args.reprocess,
        recalculate_final_scores=args.recalculate_final_scores,
        rag_query=args.rag_query,
        rag_tailor=args.rag_tailor,
        verbose=args.verbose,
    ))
