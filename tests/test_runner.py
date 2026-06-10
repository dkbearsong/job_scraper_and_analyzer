#!/usr/bin/env python3
"""
Pipeline Test Runner

Allows running individual stages or the full pipeline using dummy data.

Usage:
    python tests/test_runner.py [OPTIONS]
    
Options:
    --stage STAGE_NUM     Run a specific stage (0-8). Can be specified multiple times.
                          Examples:
                            --stage 2    Run only Stage 2
                            --stage 2 3  Run Stage 2 then Stage 3
    --all                 Run all stages sequentially (default if no --stage given)
    --from STAGE_NUM      Run from this stage to the end (inclusive)
    --to STAGE_NUM        Run stages up to this one (inclusive, used with --from)
    --list-stages         Print available stages and their descriptions
    --chain               Chain stages: feed each stage's output as the next's input
    --skip-db             Skip database persistence steps
    --verbose             Show detailed output

Examples:
    # Run all stages
    python tests/test_runner.py --all
    
    # Run a single stage with dummy data
    python tests/test_runner.py --stage 5
    
    # Run stages 2, 3, 4 sequentially
    python tests/test_runner.py --stage 2 3 4 --chain
    
    # Run from stage 3 to the end
    python tests/test_runner.py --from 3
    
    # Run stages 4 through 6
    python tests/test_runner.py --from 4 --to 6

Stage Mapping:
    0: Setup (profile extraction, DB init)
    1: Scraping (company boards + job boards)
    2: Embedding Generation + LLM Extraction
    3: Rule Filtering
    4: Archetype Engine Integration
    5: Vector Scoring with Archetypes
    6: Cheap LLM Classification
    7: Strong LLM Reranking
    8: Final Application Queue
"""

import argparse
import asyncio
import json
import sys
import os
import time
from typing import Any, Dict, List, Optional

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.test_data import get_dummy_data, STAGE_INPUT_SHAPES, STAGE_OUTPUT_SHAPES

# We import functions from main (which will be refactored below)
from main import (
    pipeline_stage_setup,
    pipeline_stage_scrape,
    pipeline_stage_embed_and_extract,
    pipeline_stage_rule_filter,
    pipeline_stage_archetype_integration,
    pipeline_stage_vector_scoring,
    pipeline_stage_cheap_llm,
    pipeline_stage_strong_llm,
    pipeline_stage_final_queue,
)


# =====================================================
# STAGE FUNCTIONS (wrappers that accept dummy data)
# =====================================================

STAGE_NAMES = {
    0: "Setup",
    1: "Scraping",
    2: "Embedding Generation + LLM Extraction",
    3: "Rule Filtering",
    4: "Archetype Engine Integration",
    5: "Vector Scoring with Archetypes",
    6: "Cheap LLM Classification",
    7: "Strong LLM Reranking",
    8: "Final Application Queue",
}

STAGE_INPUT_DESC = {
    0: "None (uses env vars and files on disk)",
    1: "Setup output dict: resume, profile, skills, job_titles, db_config",
    2: "List[dict] of scraped jobs with features (title, description, pay)",
    3: "List[dict] of jobs with embeddings and LLM-extracted data",
    4: "List[dict] of jobs with skip flags + user_preferences",
    5: "List[dict] of active (non-skipped) jobs",
    6: "List[dict] of jobs with archetype scores + semantic scores",
    7: "List[dict] of shortlisted jobs with cheap_llm_result",
    8: "List[dict] of deeply analyzed jobs with strong_llm_result",
}


async def run_stage_0(input_data: dict, **kwargs) -> dict:
    """Run Stage 0: Setup (uses real env + files)."""
    use_dummy = kwargs.get("use_dummy", False)
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)

    if use_dummy:
        print("[TEST] Stage 0: Using dummy setup data (env/file access skipped)")
        return input_data

    print("[TEST] Stage 0: Running real setup pipeline...")
    result = await pipeline_stage_setup(skip_db=skip_db, verbose=verbose)
    return result


async def run_stage_1(input_data: dict, **kwargs) -> List[Dict]:
    """Run Stage 1: Scraping."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)
    use_dummy = kwargs.get("use_dummy", False)

    print(f"[TEST] Stage 1: {'Using dummy scraped jobs' if use_dummy else 'Running real scraper'}...")

    if use_dummy:
        # Input data is already the dummy scraped jobs list
        return input_data

    result = await pipeline_stage_scrape(
        setup_data=input_data,
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_2(input_data: List[Dict], **kwargs) -> List[Dict]:
    """Run Stage 2: Embedding Generation + LLM Extraction."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)
    ai = kwargs.get("ai_engine")
    use_dummy = kwargs.get("use_dummy", False)

    print(f"[TEST] Stage 2: {'Using dummy embeddings data' if use_dummy else 'Running real embedding pipeline'}...")

    if use_dummy:
        return input_data

    result = await pipeline_stage_embed_and_extract(
        jobs=input_data,
        ai_engine=ai,
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_3(input_data: List[Dict], **kwargs) -> List[Dict]:
    """Run Stage 3: Rule Filtering."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)
    user_preferences = kwargs.get("user_preferences", {})

    print("[TEST] Stage 3: Running rule-based filtering...")
    result = await pipeline_stage_rule_filter(
        jobs=input_data,
        user_preferences=user_preferences,
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_4(input_data: List[Dict], **kwargs) -> tuple:
    """Run Stage 4: Archetype Engine Integration."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)
    ai = kwargs.get("ai_engine")

    print("[TEST] Stage 4: Running archetype integration...")
    active_jobs, archetype_mgr = await pipeline_stage_archetype_integration(
        jobs=input_data,
        ai_engine=ai,
        setup_data=kwargs.get("setup_data"),
        skip_db=skip_db,
        verbose=verbose,
    )
    return active_jobs, archetype_mgr


async def run_stage_5(input_data: tuple, **kwargs) -> List[Dict]:
    """Run Stage 5: Vector Scoring."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)

    # Input is (active_jobs, archetype_mgr)
    active_jobs, archetype_mgr = input_data if isinstance(input_data, tuple) else (input_data, None)

    print("[TEST] Stage 5: Running vector scoring...")
    result = await pipeline_stage_vector_scoring(
        jobs=active_jobs,
        archetype_manager=archetype_mgr,
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_6(input_data: List[Dict], **kwargs) -> List[Dict]:
    """Run Stage 6: Cheap LLM Classification."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)
    ai = kwargs.get("ai_engine")

    print("[TEST] Stage 6: Running cheap LLM classification...")
    result = await pipeline_stage_cheap_llm(
        jobs=input_data,
        setup_data=kwargs.get("setup_data"),
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_7(input_data: List[Dict], **kwargs) -> List[Dict]:
    """Run Stage 7: Strong LLM Reranking."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)

    print("[TEST] Stage 7: Running strong LLM reranking...")
    result = await pipeline_stage_strong_llm(
        jobs=input_data,
        setup_data=kwargs.get("setup_data"),
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


async def run_stage_8(input_data: List[Dict], **kwargs) -> List[Dict]:
    """Run Stage 8: Final Application Queue."""
    skip_db = kwargs.get("skip_db", False)
    verbose = kwargs.get("verbose", False)

    print("[TEST] Stage 8: Running final queue generation...")
    result = await pipeline_stage_final_queue(
        jobs=input_data,
        skip_db=skip_db,
        verbose=verbose,
    )
    return result


STAGE_RUNNERS = {
    0: run_stage_0,
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
    4: run_stage_4,
    5: run_stage_5,
    6: run_stage_6,
    7: run_stage_7,
    8: run_stage_8,
}


# =====================================================
# CHAINING: Map each stage's output to the next's input
# =====================================================

def get_input_for_stage(stage: int, previous_output: Any, dummy: bool) -> Any:
    """
    Given a stage number and the output from the previous stage,
    return the appropriate input for this stage.
    
    If dummy=True, returns pre-generated dummy data instead.
    """
    if dummy:
        return get_dummy_data(stage)

    # Chain outputs: determine what shape the previous stage returned
    if stage == 0:
        return None  # Setup takes no args
    if stage == 1:
        return previous_output  # Setup output dict
    if stage == 2:
        return previous_output  # List[Dict] scraped jobs
    if stage == 3:
        return previous_output  # List[Dict] after embeddings
    if stage == 4:
        return previous_output  # List[Dict] after rule filtering
    if stage == 5:
        # Stage 4 returns (active_jobs, archetype_mgr)
        return previous_output
    if stage == 6:
        # Stage 5 returns List[Dict] (filtered_job_pool)
        return previous_output
    if stage == 7:
        return previous_output  # List[Dict] shortlisted
    if stage == 8:
        return previous_output  # List[Dict] deeply analyzed
    return previous_output


def build_argument_pack(stage: int, setup_data: dict, use_dummy: bool,
                        skip_db: bool, verbose: bool, ai_engine=None) -> dict:
    """Build the kwargs dict for a stage runner."""
    args = {
        "use_dummy": use_dummy,
        "skip_db": skip_db,
        "verbose": verbose,
        "ai_engine": ai_engine,
        "setup_data": setup_data,
        "user_preferences": setup_data.get("user_preferences", {}) if setup_data else {},
    }
    return args


# =====================================================
# MAIN TEST RUNNER
# =====================================================

async def run_test_pipeline(stages: List[int], use_dummy: bool = True,
                            chain: bool = False, skip_db: bool = True,
                            verbose: bool = False):
    """
    Execute a test run of the specified stages.
    
    Args:
        stages: Ordered list of stage numbers to run.
        use_dummy: If True, use pre-generated dummy data for each stage.
        chain: If True, feed each stage's output as the next's input.
        skip_db: If True, skip database persistence operations.
        verbose: If True, show detailed output.
    """
    print("=" * 60)
    print("PIPELINE TEST RUNNER")
    print("=" * 60)
    print(f"Stages: {', '.join(str(s) for s in stages)}")
    print(f"Mode: {'Chained' if chain else 'Isolated'}")
    print(f"Data: {'Dummy' if use_dummy else 'Real (from env/files)'}")
    print(f"DB Persistence: {'Skipped' if skip_db else 'Enabled'}")
    print("=" * 60)

    if not stages:
        print("No stages specified. Use --all or --stage N.")
        return

    setup_data = None
    ai_engine = None
    previous_output = None
    all_outputs = {}

    for idx, stage_num in enumerate(stages):
        print(f"\n{'=' * 50}")
        print(f"STAGE {stage_num}: {STAGE_NAMES.get(stage_num, 'Unknown')}")
        print(f"{'=' * 50}")
        
        # Build input
        if stage_num == 0:
            # Stage 0: no input needed
            stage_input = {}
        elif not chain:
            # Isolated: use pre-generated dummy for this stage's input
            stage_input = get_dummy_data(stage_num)
        else:
            # Chained: use output from previous stage
            stage_input = get_input_for_stage(stage_num, previous_output, use_dummy)

        # Build runner args
        if stage_num == 0 and use_dummy:
            stage_input = get_dummy_data(0)
            setup_data = stage_input

        if stage_num > 0 and setup_data is None:
            # Ensure setup_data exists (get dummy if not chained from 0)
            setup_data = get_dummy_data(0)

        runner_args = build_argument_pack(
            stage=stage_num,
            setup_data=setup_data,
            use_dummy=use_dummy,
            skip_db=skip_db,
            verbose=verbose,
            ai_engine=ai_engine,
        )

        # Run the stage
        start_time = time.time()
        try:
            output = await STAGE_RUNNERS[stage_num](stage_input, **runner_args)
            elapsed = time.time() - start_time
            print(f"[✓] Stage {stage_num} completed in {elapsed:.2f}s")
        except Exception as e:
            elapsed = time.time() - start_time
            print(f"[✗] Stage {stage_num} FAILED after {elapsed:.2f}s: {e}")
            import traceback
            traceback.print_exc()
            print(f"\nAborting pipeline at stage {stage_num}.")
            break

        # Store output for chaining
        previous_output = output
        all_outputs[stage_num] = output

        # Brief validation
        if verbose:
            print(f"\n  Output type: {type(output).__name__}")
            if isinstance(output, dict):
                print(f"  Keys: {list(output.keys())[:10]}")
            elif isinstance(output, list):
                print(f"  Length: {len(output)}")
                if output:
                    print(f"  First item keys: {list(output[0].keys()) if isinstance(output[0], dict) else 'N/A'}")
            elif isinstance(output, tuple):
                print(f"  Tuple with {len(output)} elements")

    # Final summary
    print("\n" + "=" * 60)
    print("TEST RUN SUMMARY")
    print("=" * 60)
    completed = [s for s in stages if s in all_outputs]
    print(f"Completed stages: {', '.join(str(s) for s in completed)}")
    print(f"Total: {len(completed)}/{len(stages)} stages completed successfully")


def list_stages():
    """Print all available stages with descriptions."""
    print("\nAvailable Pipeline Stages:\n")
    print(f"{'Stage':<8} {'Name':<40} {'Input':<50} {'Output':<50}")
    print("-" * 148)
    for s in sorted(STAGE_NAMES.keys()):
        name = STAGE_NAMES[s]
        inp = STAGE_INPUT_DESC.get(s, "N/A")
        out = STAGE_OUTPUT_SHAPES.get(s, "N/A")
        print(f"{s:<8} {name:<40} {inp:<50} {out:<50}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Job Pipeline Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/test_runner.py --all              # Run all stages with dummy data
  python tests/test_runner.py --stage 5           # Run stage 5 only
  python tests/test_runner.py --stage 2 3 4 --chain  # Chain stages 2-4
  python tests/test_runner.py --from 3            # Run stages 3 through 8
  python tests/test_runner.py --from 3 --to 6     # Run stages 3-6
  python tests/test_runner.py --stage 8 --skip-db # Skip DB persistence
  python tests/test_runner.py --stage 0 --verbose # See all details
        """
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--all", action="store_true",
        help="Run all stages sequentially"
    )
    mode_group.add_argument(
        "--stage", nargs="+", type=int,
        help="Run specific stage(s) by number"
    )
    mode_group.add_argument(
        "--from", dest="from_stage", type=int,
        help="Run from this stage to the end"
    )

    parser.add_argument(
        "--to", type=int, default=8,
        help="End stage (used with --from, default: 8)"
    )
    parser.add_argument(
        "--chain", action="store_true",
        help="Chain stages: feed each stage's output as the next's input"
    )
    parser.add_argument(
        "--skip-db", action="store_true", default=True,
        help="Skip database persistence steps (default: True)"
    )
    parser.add_argument(
        "--no-skip-db", action="store_false", dest="skip_db",
        help="Enable database persistence (default: skip)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Show detailed output"
    )
    parser.add_argument(
        "--list-stages", action="store_true",
        help="Print available stages and exit"
    )

    return parser.parse_args()


async def main():
    args = parse_args()

    if args.list_stages:
        list_stages()
        return

    # Determine which stages to run
    stages = []
    if args.all:
        stages = list(range(0, 9))
    elif args.stage:
        stages = args.stage
    elif args.from_stage is not None:
        stages = list(range(args.from_stage, args.to + 1))
    else:
        # Default: run all stages
        stages = list(range(0, 9))

    # Validate stage numbers
    invalid = [s for s in stages if s < 0 or s > 8]
    if invalid:
        print(f"Error: Invalid stage number(s): {invalid}. Valid range: 0-8.")
        sys.exit(1)

    # Remove duplicates while preserving order
    seen = set()
    stages = [s for s in stages if not (s in seen or seen.add(s))]

    # For isolated testing, default to use_dummy=True
    use_dummy = not args.chain

    await run_test_pipeline(
        stages=stages,
        use_dummy=use_dummy,
        chain=args.chain,
        skip_db=args.skip_db,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())