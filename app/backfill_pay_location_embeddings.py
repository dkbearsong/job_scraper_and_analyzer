#!/usr/bin/env python3
"""
Backfill script to build and load in-memory sentence_transformers embeddings (384-dim)
across database jobs, specifically targeting jobs with legacy LM Studio embeddings
or recent jobs missing 384-dim vectors.

Usage:
    python app/backfill_pay_location_embeddings.py [--force] [--limit N] [--recent]
"""

import os
import sys
import json
import asyncio
import argparse
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.pull_data import DataPuller
from app.ai_engine import AIEngine
from app.rag_engine import RAGEngine
from app.ai_utils import generate_embeddings_batch
from app.config_utils import _load_user_config
from app.migrate_db import run_migration
from app.migrate_rag_db import run_rag_migration

async def backfill_embeddings(force: bool = False, limit: int = 0, recent_only: bool = False):
    load_dotenv()
    user_config = _load_user_config()
    host = user_config.get("db_host") or os.getenv("DB_HOST") or "localhost"
    port = user_config.get("db_port") or os.getenv("DB_PORT") or "5432"
    user = user_config.get("db_user") or os.getenv("DB_USER") or "postgres"
    password = user_config.get("db_password") or os.getenv("DB_PASSWORD") or ""
    db_name = user_config.get("db_name") or os.getenv("DB_NAME") or "web_scraper_db"

    print("=" * 60)
    print("BACKFILLING & RE-EMBEDDING DATABASE JOBS (IN-MEMORY 384-DIM)")
    print("=" * 60)

    # 1. Run migrations to ensure columns & vector tables exist
    print("Running database migrations...")
    run_migration()
    run_rag_migration()

    dp = DataPuller(dbname=db_name, user=user, password=password, host=host, port=str(port))
    
    # Use sentence_transformers in-memory embedding model
    embeddings_llm = user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "sentence_transformers"))
    embeddings_model = user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "all-MiniLM-L6-v2"))
    if embeddings_llm == "lm_studio" or embeddings_model == "local-model":
        embeddings_llm = "sentence_transformers"
        embeddings_model = "all-MiniLM-L6-v2"

    ai = AIEngine(default_provider_name=embeddings_llm, embeddings_model=embeddings_model)
    rag = RAGEngine(dp=dp, ai_engine=ai)

    # 2. Query jobs needing embedding regeneration (using fast array_length checks)
    query_sql = """
        SELECT j.id, j.job_name, c.company_name, j.link, j.pay_range, j.work_type, j.timezone,
               j.job_summary, j.description AS summary, j.requirements, j.responsibilities,
               o.city, o.state, o.location, j.flexibility,
               array_length(je.title_embedding, 1) AS title_dim,
               array_length(je.description_embedding, 1) AS desc_dim,
               j.date_added
        FROM job j
        LEFT JOIN company c ON j.company_id = c.id
        LEFT JOIN office o ON j.office_id = o.id
        LEFT JOIN job_embeddings je ON j.id = je.job_id
        WHERE j.skip IS NOT TRUE;
    """

    rows = dp.conn.execute_sql(query_sql, fetch=True) or []
    print(f"Retrieved {len(rows)} total active jobs from database.")

    jobs_to_process = []
    for r in rows:
        if isinstance(r, dict):
            jid = r.get("id")
            pay_range = r.get("pay_range")
            loc = r.get("location")
            city = r.get("city")
            state = r.get("state")
            wt = r.get("work_type") or r.get("flexibility")
            tz = r.get("timezone")
            title_dim = r.get("title_dim")
            desc_dim = r.get("desc_dim")
            title = r.get("job_name")
            company = r.get("company_name")
            link = r.get("link")
            job_summary = r.get("job_summary")
            summary = r.get("summary")
            requirements = r.get("requirements")
            responsibilities = r.get("responsibilities")
            date_added = r.get("date_added")
        else:
            jid, title, company, link, pay_range, wt, tz, job_summary, summary, requirements, responsibilities, city, state, loc, flex, title_dim, desc_dim = r[:17]
            date_added = r[17] if len(r) > 17 else None
            if not wt:
                wt = flex

        # Check vector dimensions: 384 indicates sentence_transformers
        is_lm_studio = (title_dim is not None and title_dim != 384) or (desc_dim is not None and desc_dim != 384)
        is_missing = (title_dim is None) or (desc_dim is None)

        if recent_only:
            needs_update = is_missing or is_lm_studio
        else:
            needs_update = force or is_lm_studio or is_missing

        if needs_update:
            jobs_to_process.append({
                "job_id": jid,
                "title": title,
                "company_name": company,
                "link": link,
                "pay_range": pay_range,
                "location": loc,
                "city": city,
                "state": state,
                "work_type": wt,
                "timezone": tz,
                "job_summary": job_summary,
                "summary": summary,
                "requirements": requirements,
                "responsibilities": responsibilities,
            })

    if limit > 0:
        jobs_to_process = jobs_to_process[:limit]

    print(f"Found {len(jobs_to_process)} job(s) needing embedding generation / re-embedding.")
    if not jobs_to_process:
        print("All target database jobs already have valid in-memory 384-dim embeddings!")
        return

    # Process in-memory with batching
    success_count = 0
    batch_updates = []
    
    for idx, job in enumerate(jobs_to_process, 1):
        jid = job["job_id"]
        title = str(job["title"] or "Unknown Title")
        
        # 1. Parse requirements & responsibilities
        reqs = job["requirements"]
        if isinstance(reqs, str):
            try:
                reqs = json.loads(reqs)
            except Exception:
                reqs = [reqs]
        elif not isinstance(reqs, list):
            reqs = []

        resps = job["responsibilities"]
        if isinstance(resps, str):
            try:
                resps = json.loads(resps)
            except Exception:
                resps = [resps]
        elif not isinstance(resps, list):
            resps = []

        summary_text = str(job["summary"] or job["job_summary"] or "")
        reqs_text = ", ".join(str(r) for r in reqs if r)
        resps_text = ", ".join(str(r) for r in resps if r)

        # 2. Prepare Pay Text
        pay_text = job["pay_range"] or ""
        if pay_text:
            pay_text = f"Pay Range: {pay_text}"

        # 3. Prepare Location Text
        loc_parts = []
        if job["location"]:
            loc_parts.append(f"Location: {job['location']}")
        elif job["city"] or job["state"]:
            city_state = ", ".join(str(x) for x in [job["city"], job["state"]] if x)
            loc_parts.append(f"Location: {city_state}")
        if job["work_type"] and str(job["work_type"]).lower() not in ("unknown", "na"):
            loc_parts.append(f"Work Arrangement: {job['work_type']}")
        if job["timezone"] and str(job["timezone"]).lower() != "na":
            loc_parts.append(f"Timezone: {job['timezone']}")
        location_text = ". ".join(loc_parts)

        # 4. Batch generate 6 vectors in 1 single call per job
        raw_texts = [title, reqs_text, resps_text, summary_text, pay_text, location_text]
        texts = [str(t) if t is not None else "" for t in raw_texts]
        
        vectors = generate_embeddings_batch(ai, texts, provider_name=embeddings_llm)
        if not vectors or len(vectors) != 6:
            vectors = [[], [], [], [], [], []]

        title_vec, req_vec, resp_vec, desc_vec, pay_vec, loc_vec = vectors

        # Upsert into job_embeddings
        embedding_update = {
            "job_id": jid,
            "title_embedding": title_vec,
            "requirements_embedding": req_vec,
            "responsibilities_embedding": resp_vec,
            "description_embedding": desc_vec,
            "pay_embedding": pay_vec,
            "location_embedding": loc_vec
        }
        batch_updates.append(embedding_update)

        # Re-index job into RAG documents store
        job_dict = {
            "metadata": {
                "job_id": jid,
                "company_name": job["company_name"],
                "link": job["link"]
            },
            "features": {
                "title": title,
                "summary": summary_text,
                "requirements": reqs,
                "responsibilities": resps,
                "pay": job["pay_range"],
                "work_type": job["work_type"],
                "location": job["location"],
                "city": job["city"],
                "state": job["state"],
                "timezone": job["timezone"]
            },
            "embeddings": {
                "title_vector": title_vec,
                "requirements_vector": req_vec,
                "responsibilities_vector": resp_vec,
                "description_vector": desc_vec,
                "pay_vector": pay_vec,
                "location_vector": loc_vec
            }
        }
        rag.index_job(job_dict)
        success_count += 1

        if len(batch_updates) >= 50:
            dp.save_job_embeddings(batch_updates)
            batch_updates = []
            print(f"[{idx}/{len(jobs_to_process)}] Persisted 50 job embeddings & RAG documents...")

    if batch_updates:
        dp.save_job_embeddings(batch_updates)
        print(f"[{len(jobs_to_process)}/{len(jobs_to_process)}] Persisted final batch of job embeddings...")

    print(f"\nSuccessfully re-embedded {success_count} job(s) into database & RAG store!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill / re-embed jobs with in-memory 384-dim embeddings.")
    parser.add_argument("--force", action="store_true", help="Force regeneration of embeddings for ALL active jobs.")
    parser.add_argument("--limit", type=int, default=0, help="Maximum number of jobs to re-embed (0 = all matching jobs).")
    parser.add_argument("--recent", action="store_true", help="Only process recent jobs or jobs missing/LM Studio embeddings.")
    args = parser.parse_args()

    asyncio.run(backfill_embeddings(force=args.force, limit=args.limit, recent_only=args.recent))
