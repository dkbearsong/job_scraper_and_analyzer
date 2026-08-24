#!/usr/bin/env python3
"""
Regenerate Embeddings Script
Re-generates vector embeddings for jobs and archetypes in the database.
Allows specifying how many days back to process via CLI flag (--days-back) or interactive prompt.
"""

import os
import sys
import argparse
import asyncio
import datetime
from typing import List, Dict

from dotenv import load_dotenv
from app.config_utils import _load_user_config
from app.pull_data import DataPuller
from app.ai_engine import AIEngine
from app.ai_utils import generate_embeddings_batch, generate_embeddings
from app.archetype_engine import ArchetypeManager, Archetype
import numpy as np

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Regenerate vector embeddings for jobs and archetypes.")
    parser.add_argument("--days-back", type=int, default=None, help="Number of days back to regenerate job embeddings for (e.g. 7, 30, 90).")
    parser.add_argument("--all-jobs", action="store_true", help="Regenerate embeddings for all jobs in database.")
    parser.add_argument("--skip-archetypes", action="store_true", help="Skip re-generating archetype embeddings.")
    args = parser.parse_args()

    days_back = args.days_back
    if days_back is None and not args.all_jobs:
        print("=" * 60)
        print("EMBEDDING REGENERATION UTILITY")
        print("=" * 60)
        user_input = input("Enter number of days back to regenerate embeddings for (or 'all' for all jobs): ").strip()
        if user_input.lower() == 'all':
            args.all_jobs = True
        else:
            try:
                days_back = int(user_input)
            except ValueError:
                print("Invalid input. Defaulting to 30 days back.")
                days_back = 30

    user_config = _load_user_config()
    dp = DataPuller(
        dbname=user_config.get("db_name", os.getenv("DB_NAME", "")),
        user=user_config.get("db_user", os.getenv("DB_USER", "")),
        password=user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
        host=user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
        port=str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
    )

    _embeddings_llm = user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "fastembed"))
    _embeddings_model = user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "BAAI/bge-small-en-v1.5"))

    print(f"\nInitializing AI Engine with embedding provider: '{_embeddings_llm}' (Model: {_embeddings_model})...")
    ai = AIEngine(
        default_provider_name=_embeddings_llm,
        embeddings_model=_embeddings_model
    )

    # 1. Regenerate Archetype Embeddings
    if not args.skip_archetypes:
        print("\n--- Regenerating Archetype Embeddings ---")
        from app.pipeline.stages import ARCHETYPES_CONFIG
        arch_config_list = ARCHETYPES_CONFIG or []
        for arch in arch_config_list:
            name = arch.get("name")
            if not name:
                continue
            title_txt = arch.get("title", "")
            req_txt = arch.get("requirements", "")
            resp_txt = arch.get("responsibilities", "")
            
            print(f"Generating embeddings for archetype: {name}")
            vecs = ai.embed_batch([title_txt, req_txt, resp_txt], provider_name=_embeddings_llm)
            if len(vecs) == 3:
                dp.save_archetype_embeddings({
                    "archetype_name": name,
                    "archetype_type": arch.get("type", "benchmark"),
                    "title_embedding": vecs[0],
                    "requirements_embedding": vecs[1],
                    "responsibilities_embedding": vecs[2],
                    "metadata": "{}"
                })
                print(f"  [OK] Saved embeddings for archetype '{name}'")
            else:
                print(f"  [FAIL] Could not generate embeddings for archetype '{name}'")

    # 2. Query jobs based on days_back filter
    print("\n--- Fetching Jobs for Embedding Regeneration ---")
    if args.all_jobs:
        sql = "SELECT id, job_name, job_summary, requirements, responsibilities FROM job WHERE skip IS NOT TRUE AND job_summary IS NOT NULL"
        params = ()
        print("Querying ALL active jobs in database...")
    else:
        sql = """
            SELECT id, job_name, job_summary, requirements, responsibilities 
            FROM job 
            WHERE skip IS NOT TRUE 
              AND job_summary IS NOT NULL
              AND date_added >= CURRENT_DATE - (%s || ' days')::INTERVAL
        """
        params = (days_back,)
        print(f"Querying active jobs from the last {days_back} days...")

    try:
        rows = dp.conn.execute_sql(sql, params, fetch=True) or []
    except Exception as e:
        print(f"Error querying jobs from DB: {e}")
        return

    print(f"Found {len(rows)} job(s) matching criteria.")
    if not rows:
        print("No jobs to re-embed. Done.")
        return

    # 3. Batch regenerate job embeddings
    print("\n--- Generating & Saving Vector Embeddings ---")
    batch_updates = []
    for idx, r in enumerate(rows, 1):
        if isinstance(r, dict):
            jid = r.get("id")
            title = r.get("job_name", "") or ""
            summary = r.get("job_summary", "") or ""
            reqs = r.get("requirements") or []
            resps = r.get("responsibilities") or []
        else:
            jid, title, summary, reqs, resps = r[0], r[1], r[2], r[3], r[4]

        req_text = ", ".join(reqs) if isinstance(reqs, list) else str(reqs or "")
        resp_text = ", ".join(resps) if isinstance(resps, list) else str(resps or "")

        vecs = ai.embed_batch([title, req_text, resp_text], provider_name=_embeddings_llm)
        if len(vecs) == 3 and vecs[0]:
            batch_updates.append({
                "job_id": jid,
                "title_embedding": vecs[0],
                "requirements_embedding": vecs[1],
                "responsibilities_embedding": vecs[2],
                "description_embedding": None  # Deferred
            })
            if idx % 10 == 0 or idx == len(rows):
                print(f"Processed {idx}/{len(rows)} jobs...")

    if batch_updates:
        print(f"\nPersisting embeddings for {len(batch_updates)} jobs to database...")
        dp.save_job_embeddings(batch_updates)
        print("Database save complete!")

    print("\n" + "=" * 60)
    print("EMBEDDING REGENERATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
