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
import re
from typing import List, Dict

from dotenv import load_dotenv
from app.config_utils import _load_user_config
from app.pull_data import DataPuller
from app.ai_engine import AIEngine
from app.ai_utils import generate_embeddings_batch, generate_embeddings
from app.archetype_engine import ArchetypeManager, Archetype
from app.vector_engine import apply_keyword_adjustments, apply_metadata_adjustments
import numpy as np

load_dotenv()

async def main():
    parser = argparse.ArgumentParser(description="Regenerate vector embeddings for jobs and archetypes.")
    parser.add_argument("--days-back", type=int, default=None, help="Number of days back to regenerate job embeddings for (e.g. 7, 30, 90).")
    parser.add_argument("--all-jobs", action="store_true", help="Regenerate embeddings for all jobs in database.")
    parser.add_argument("--skip-archetypes", action="store_true", help="Skip re-generating archetype embeddings.")
    parser.add_argument("--refresh-scores", action="store_true", help="Recompute and refresh Stage 5 vector scores in database for regenerated jobs.")
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

        if not args.refresh_scores:
            refresh_input = input("Also refresh cached vector scores in DB for these jobs? (y/N): ").strip()
            if refresh_input.lower() in ('y', 'yes'):
                args.refresh_scores = True

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
        sql = """
            SELECT id, job_name, job_summary, requirements, responsibilities, pay_range, work_type, date_added 
            FROM job 
            WHERE skip IS NOT TRUE AND job_summary IS NOT NULL
        """
        params = ()
        print("Querying ALL active jobs in database...")
    else:
        sql = """
            SELECT id, job_name, job_summary, requirements, responsibilities, pay_range, work_type, date_added 
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

        # Ensure non-empty fallbacks for requirements and responsibilities
        req_text = ", ".join(str(x) for x in reqs if x) if isinstance(reqs, list) and reqs else str(reqs or "")
        resp_text = ", ".join(str(x) for x in resps if x) if isinstance(resps, list) and resps else str(resps or "")

        # Fallback if empty
        if not req_text.strip():
            req_text = resp_text if resp_text.strip() else summary
        if not resp_text.strip():
            resp_text = req_text if req_text.strip() else summary

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

    # 4. Optionally Refresh Cached Vector Scores
    if args.refresh_scores and batch_updates:
        print("\n--- Refreshing Cached Vector Scores ---")
        from app.pipeline.stages import ARCHETYPES_CONFIG

        archetype_manager = ArchetypeManager()
        
        # Load benchmark archetypes from DB
        arch_config_list = ARCHETYPES_CONFIG or []
        for arch in arch_config_list:
            cached = dp.get_archetype_embeddings(arch["name"])
            if cached and cached.get("title_embedding"):
                archetype_manager.add_archetype(Archetype(
                    name=arch["name"],
                    type=cached.get("archetype_type", "benchmark"),
                    title_embedding=np.array(cached["title_embedding"]),
                    requirements_embedding=np.array(cached["requirements_embedding"]),
                    responsibilities_embedding=np.array(cached["responsibilities_embedding"]),
                    metadata=cached.get("metadata") or {}
                ))

        # Load negative archetypes from DB / config
        neg_config = user_config.get("negative_scoring", {})
        neg_enabled = neg_config.get("enabled", True)
        penalty_weight = float(neg_config.get("penalty_weight", 0.35))
        sim_threshold = float(neg_config.get("similarity_threshold", 0.50))

        if neg_enabled:
            avoid_titles = neg_config.get("avoid_titles", [])
            avoid_functions = neg_config.get("avoid_functions", [])
            for t in avoid_titles:
                if t:
                    cached = dp.get_archetype_embeddings(f"Avoid Title: {t}")
                    if cached and cached.get("title_embedding"):
                        archetype_manager.add_negative_archetype(Archetype(
                            name=f"Avoid Title: {t}",
                            type="negative_title",
                            title_embedding=np.array(cached["title_embedding"]),
                            requirements_embedding=np.array(cached["requirements_embedding"]),
                            responsibilities_embedding=np.array(cached["responsibilities_embedding"]),
                        ))
            for f in avoid_functions:
                if f:
                    cached = dp.get_archetype_embeddings(f"Avoid Function: {f[:30]}")
                    if cached and cached.get("title_embedding"):
                        archetype_manager.add_negative_archetype(Archetype(
                            name=f"Avoid Function: {f[:30]}",
                            type="negative_function",
                            title_embedding=np.array(cached["title_embedding"]),
                            requirements_embedding=np.array(cached["requirements_embedding"]),
                            responsibilities_embedding=np.array(cached["responsibilities_embedding"]),
                        ))

        scored_jobs = []
        emb_by_id = {u["job_id"]: u for u in batch_updates}

        for r in rows:
            if isinstance(r, dict):
                jid = r.get("id")
                title = r.get("job_name", "") or ""
                summary = r.get("job_summary", "") or ""
                reqs = r.get("requirements") or []
                resps = r.get("responsibilities") or []
                pay_val = r.get("pay_range", "") or ""
                work_type = r.get("work_type", "") or ""
                date_added = r.get("date_added")
            else:
                jid = r[0]
                title = r[1] or ""
                summary = r[2] or ""
                reqs = r[3] or []
                resps = r[4] or []
                pay_val = r[5] if len(r) > 5 and r[5] else ""
                work_type = r[6] if len(r) > 6 and r[6] else ""
                date_added = r[7] if len(r) > 7 else None

            emb_item = emb_by_id.get(jid)
            if not emb_item:
                continue

            job_dict = {
                "metadata": {"job_id": jid, "in_db": True},
                "features": {
                    "title": title,
                    "summary": summary,
                    "description": summary,
                    "requirements": reqs,
                    "responsibilities": resps,
                    "pay": pay_val,
                    "work_type": work_type,
                    "date_added": str(date_added) if date_added else ""
                },
                "embeddings": {
                    "title_vector": emb_item["title_embedding"],
                    "requirements_vector": emb_item["requirements_embedding"],
                    "responsibilities_vector": emb_item["responsibilities_embedding"],
                }
            }

            matches = archetype_manager.compare_job_to_archetypes(job_dict)
            if not matches:
                continue
            best_match = matches[0]

            t_sim = best_match.get("title_similarity", 0.0)
            req_sim = best_match.get("requirements_similarity", 0.0)
            resp_sim = best_match.get("responsibility_similarity", 0.0)

            eff_req_sim = req_sim if req_sim > 0.0 else (resp_sim if resp_sim > 0.0 else t_sim)
            eff_resp_sim = resp_sim if resp_sim > 0.0 else (req_sim if req_sim > 0.0 else t_sim)

            positive_score = 0.40 * t_sim + 0.35 * eff_req_sim + 0.25 * eff_resp_sim

            neg_result = archetype_manager.compare_job_to_negative_archetypes(job_dict)
            max_neg_sim = neg_result.get("max_negative_similarity", 0.0) if isinstance(neg_result, dict) else 0.0

            negative_penalty = 0.0
            if neg_enabled and max_neg_sim > sim_threshold:
                excess = (max_neg_sim - sim_threshold) / max(0.001, (1.0 - sim_threshold))
                negative_penalty = min(1.0, excess) * penalty_weight

            semantic_score = positive_score - negative_penalty

            kw_reqs = reqs if reqs else (resps if resps else [summary])
            semantic_score = apply_keyword_adjustments(semantic_score, kw_reqs, title)

            days_old = 30
            if date_added:
                try:
                    if isinstance(date_added, datetime.date):
                        days_old = (datetime.date.today() - date_added).days
                    elif isinstance(date_added, str):
                        dt = datetime.datetime.strptime(date_added[:10], "%Y-%m-%d").date()
                        days_old = (datetime.date.today() - dt).days
                except Exception:
                    pass

            job_meta = {
                "is_remote": str(work_type).lower() == "remote",
                "salary": 0,
                "days_old": days_old
            }
            if pay_val:
                numbers = re.findall(r'\d+(?:,\d+)?', pay_val.replace(',', ''))
                if len(numbers) >= 2:
                    job_meta["salary"] = int(numbers[1])
                elif len(numbers) == 1:
                    job_meta["salary"] = int(numbers[0])

            semantic_score = apply_metadata_adjustments(semantic_score, job_meta)
            semantic_score = max(0.0, min(1.0, semantic_score))

            job_dict["semantic_score"] = semantic_score
            job_dict["title_similarity"] = t_sim
            job_dict["requirements_similarity"] = req_sim
            job_dict["responsibility_similarity"] = resp_sim
            job_dict["adjusted_score"] = semantic_score
            job_dict["best_archetype"] = best_match.get("archetype_name", "")

            scored_jobs.append(job_dict)

        if scored_jobs:
            sorted_scored = sorted(scored_jobs, key=lambda x: x.get("semantic_score", 0), reverse=True)
            dp.save_vector_scores(sorted_scored, overwrite=True)
            print(f"Refreshed vector scores for {len(sorted_scored)} jobs in database.")

    print("\n" + "=" * 60)
    print("EMBEDDING REGENERATION COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
