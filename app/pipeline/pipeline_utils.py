import os
import shutil
import json
import yaml
from typing import Dict, List, Tuple, Any, Optional
from app.logger import error_logger_crash
from app.rule_filters import is_missing_value
from app.pull_data import DataPuller

def _row_to_job_dict(row) -> Dict:
    if hasattr(row, 'get'):
        d = row
    else:
        # Map tuple indices
        keys = [
            "id", "job_name", "company_name", "link", "job_summary", "extracted_summary",
            "requirements", "responsibilities", "pay_range", "seniority", "work_type", "timezone",
            "source", "date_added", "city", "state", "location",
            "flexibility",
            "title_embedding", "requirements_embedding", "responsibilities_embedding", "description_embedding",
            "pay_embedding", "location_embedding"
        ]
        d = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}

    # Parse requirements and responsibilities
    requirements = d.get("requirements")
    if isinstance(requirements, str):
        try:
            requirements = json.loads(requirements)
        except Exception:
            requirements = [s.strip() for s in requirements.split(",") if s.strip()]
    elif not isinstance(requirements, list):
        requirements = []

    responsibilities = d.get("responsibilities")
    if isinstance(responsibilities, str):
        try:
            responsibilities = json.loads(responsibilities)
        except Exception:
            responsibilities = [r.strip() for r in responsibilities.split("\n") if r.strip()]
    elif not isinstance(responsibilities, list):
        responsibilities = []

    return {
        "metadata": {
            "job_id": d.get("id"),
            "source": d.get("source", "scraped"),
            "in_db": True,
            "company_name": d.get("company_name", "Unknown"),
            "link": d.get("link", ""),
        },
        "features": {
            "title": d.get("job_name", ""),
            "description": d.get("job_summary", ""),  # Raw description is in job_summary in DB
            "summary": d.get("extracted_summary", ""), # Extracted summary is in description in DB
            "pay": d.get("pay_range") if not is_missing_value(d.get("pay_range")) else "",
            "pay_rate": d.get("pay_range") if not is_missing_value(d.get("pay_range")) else "",
            "seniority": d.get("seniority") if not is_missing_value(d.get("seniority")) else "NA",
            "work_type": d.get("work_type") if not is_missing_value(d.get("work_type")) else (d.get("flexibility") if not is_missing_value(d.get("flexibility")) else "NA"),
            "timezone": d.get("timezone") if not is_missing_value(d.get("timezone")) else "NA",
            "requirements": requirements,
            "responsibilities": responsibilities,
            "city": d.get("city"),
            "state": d.get("state"),
            "location": d.get("location"),
        },
        "embeddings": {
            "title_vector": d.get("title_embedding"),
            "requirements_vector": d.get("requirements_embedding"),
            "responsibilities_vector": d.get("responsibilities_embedding"),
            "description_vector": d.get("description_embedding"),
            "pay_vector": d.get("pay_embedding"),
            "location_vector": d.get("location_embedding"),
        }
    }


def _move_existing_logs() -> None:
    """Move legacy log files from the project root to the logs/ directory."""
    import shutil
    import glob
    os.makedirs(os.path.join("logs", "skipped"), exist_ok=True)
    
    root_to_logs = ["app_error.log", "pipeline_stats.log"]
    for file in root_to_logs:
        if os.path.exists(file):
            try:
                shutil.move(file, os.path.join("logs", file))
                print(f"[Startup] Relocated {file} to logs/")
            except Exception as e:
                print(f"[Startup] Warning: could not move {file}: {e}")

    for run_log in glob.glob("*_run.log"):
        # Don't move if it's already a full path or inside logs
        try:
            shutil.move(run_log, os.path.join("logs", run_log))
            print(f"[Startup] Relocated {run_log} to logs/")
        except Exception as e:
            print(f"[Startup] Warning: could not move {run_log}: {e}")

    for skip_file in glob.glob("skipped_jobs_*.json"):
        try:
            shutil.move(skip_file, os.path.join("logs", "skipped", skip_file))
            print(f"[Startup] Relocated {skip_file} to logs/skipped/")
        except Exception as e:
            print(f"[Startup] Warning: could not move {skip_file}: {e}")


def _parse_stage_range(raw: str) -> Tuple[int, int]:
    """
    Parse a stage range string into an inclusive (start, end) tuple.

    Accepts:
        "0-8"  -> (0, 8)
        "2-4"  -> (2, 4)
        "2"    -> (2, 2)
        "all"  -> (0, 8)

    Returns:
        Tuple of (start_stage, end_stage), both inclusive.
    """
    if raw.lower() in ("all", "full"):
        return (0, 8)

    if "-" in raw:
        parts = raw.split("-", 1)
        try:
            start = int(parts[0].strip())
            end = int(parts[1].strip())
        except (ValueError, IndexError):
            print(f"Warning: invalid stage range '{raw}'. Defaulting to 0-8.")
            return (0, 8)
    else:
        try:
            n = int(raw.strip())
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


def _print_stage_header(title: str, stage_start: int, stage_end: int) -> None:
    """Print a centered pipeline header with stage info."""
    print(f"{title:^60}")
    if stage_start == 0 and stage_end == 8:
        print("Running full pipeline (stages 0-8)")
    elif stage_start == stage_end:
        print(f"Running stage {stage_start} only")
    else:
        print(f"Running stages {stage_start}-{stage_end}")


def _normalize_to_pool_format(all_scraped: List[Dict]) -> List[Dict]:
    """
    Convert a list of raw scraped job dicts into the processed_job_pool format
    expected by downstream pipeline stages.
    
    Args:
        all_scraped: List of raw job dicts from adapters (must have 'title' key).
    
    Returns:
        List[Dict] in the processed_job_pool format.
    """
    processed_job_pool = []
    seen_ids = set()
    for idx, item in enumerate(all_scraped):
        if not item or not item.get('title'):
            continue
        job_id = item.get('id', idx + 1000)
        if job_id in seen_ids:
            job_id = max(seen_ids) + 1 + idx
        seen_ids.add(job_id)
        processed_job_pool.append({
            "metadata": {
                "job_id": job_id,
                "source": item.get('source', 'scraped'),
                "in_db": 'id' in item,
            },
            "features": {
                "title": item.get('title', ''),
                "description": item.get('description', ''),
                "pay": item.get('pay', '') or item.get('pay_rate', ''),
                "pay_rate": item.get('pay_rate', '') or item.get('pay', ''),
                "seniority": "NA",
                "work_type": item.get('flexibility', 'NA'),
                "timezone": "NA",
                "location": item.get('location', 'NA'),
                "city": item.get('city', 'NA'),
                "state": item.get('state', 'NA'),
            },
            "embeddings": {
                "description_vector": None,
                "requirements_vector": None,
            },
        })
    return processed_job_pool


def _check_run_fallback_flag(config_path: str) -> bool:
    """
    Check the top-level 'run_fallback_after_adapters' flag in the scrapers config YAML.
    
    Args:
        config_path: Path to the scrapers_config.yaml file.
    
    Returns:
        True if the flag is set to True, False otherwise.
    """
    if not os.path.exists(config_path):
        return False
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}
        return config.get("run_fallback_after_adapters", False)
    except Exception as e:
        print(f"Warning: Could not read 'run_fallback_after_adapters' from {config_path}: {e}")
        return False


async def _load_jobs_for_stage(dp: DataPuller, stage: int, limit: int = 50, force_reprocess: bool = False) -> List[Dict]:
    """
    Load jobs from the database for a specific stage, selecting only active jobs
    that have completed the prior stages but are missing the target stage's results.
    
    When force_reprocess=True and stage==2, selects jobs that already have embeddings
    (so they can be re-processed).
    """
    # Base query columns
    query_cols = [
        "j.id", "j.job_name", "c.company_name", "j.link", "j.job_summary", "j.description AS extracted_summary",
        "j.requirements", "j.responsibilities", "j.pay_range", "j.seniority", "j.work_type", "j.timezone",
        "j.source", "j.date_added", "o.city", "o.state", "o.location",
        "j.flexibility"
    ]
    
    joins = [
        "JOIN company c ON j.company_id = c.id",
        "LEFT JOIN office o ON j.office_id = o.id"
    ]
    
    # Conditional inclusion of job_embeddings table join to avoid pulling heavy vectors when not needed
    if stage > 2 or (stage == 2 and force_reprocess):
        query_cols.extend([
            "je.title_embedding", "je.requirements_embedding", "je.responsibilities_embedding", "je.description_embedding",
            "je.pay_embedding", "je.location_embedding"
        ])
        joins.append("LEFT JOIN job_embeddings je ON j.id = je.job_id")
    
    where_clauses = ["j.skip IS NOT TRUE"]
    
    if not force_reprocess:
        if stage < 8:
            where_clauses.append("NOT EXISTS (SELECT 1 FROM strong_llm_results WHERE job_id = j.id)")
        if stage < 6:
            where_clauses.append("NOT EXISTS (SELECT 1 FROM cheap_llm_results WHERE job_id = j.id)")
            
    if stage > 2:
        # For stages 3+, we need jobs that have completed Stage 2 (embeddings exist and are not NULL)
        where_clauses.append("je.job_id IS NOT NULL AND je.title_embedding IS NOT NULL")
    elif stage == 2:
        if force_reprocess:
            # Reprocess: select jobs that already have embeddings to re-run Stage 2
            where_clauses.append("je.job_id IS NOT NULL AND je.title_embedding IS NOT NULL")
        else:
            # Normal: handled via separate optimized query branches below
            pass
        # Ensure we have a job description to embed
        where_clauses.append("j.job_summary IS NOT NULL")
    
    # Add columns and joins depending on what stage we are starting at
    if stage >= 6:
        query_cols.extend([
            "vs.semantic_score", "vs.title_similarity", "vs.requirements_similarity",
            "vs.responsibility_similarity", "vs.adjusted_score", "vs.archetype_name AS best_archetype"
        ])
        joins.append("LEFT JOIN vector_scores vs ON j.id = vs.job_id")
        
    if stage >= 7:
        query_cols.extend([
            "clr.fit_score AS cheap_fit_score", "clr.decision AS cheap_decision",
            "clr.strengths AS cheap_strengths", "clr.concerns AS cheap_concerns",
            "clr.hard_requirements_and_tools AS cheap_hard_requirements_and_tools",
            "clr.core_responsibilities AS cheap_core_responsibilities",
            "clr.years_of_experience AS cheap_years_of_experience",
            "clr.domain_and_education AS cheap_domain_and_education",
            "clr.raw_response AS cheap_raw_response"
        ])
        joins.append("LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id")
        
    if stage >= 8:
        query_cols.extend([
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
        ])
        joins.append("LEFT JOIN strong_llm_results slr ON j.id = slr.job_id")
 
    # Add stage-specific filters (pull only if the current stage's result is missing)
    if stage == 3:
        if "LEFT JOIN vector_scores vs ON j.id = vs.job_id" not in joins:
            joins.append("LEFT JOIN vector_scores vs ON j.id = vs.job_id")
        where_clauses.append("vs.job_id IS NULL")
    elif stage == 4 or stage == 5:
        if "LEFT JOIN vector_scores vs ON j.id = vs.job_id" not in joins:
            joins.append("LEFT JOIN vector_scores vs ON j.id = vs.job_id")
        where_clauses.append("vs.job_id IS NULL")
    elif stage == 6:
        if "LEFT JOIN vector_scores vs ON j.id = vs.job_id" not in joins:
            joins.append("LEFT JOIN vector_scores vs ON j.id = vs.job_id")
        if "LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id" not in joins:
            joins.append("LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id")
        where_clauses.append("vs.job_id IS NOT NULL")
        where_clauses.append("vs.adjusted_score >= 0.72")
        where_clauses.append("clr.job_id IS NULL")
    elif stage == 7:
        if "LEFT JOIN strong_llm_results slr ON j.id = slr.job_id" not in joins:
            joins.append("LEFT JOIN strong_llm_results slr ON j.id = slr.job_id")
        where_clauses.append("clr.decision IN ('apply', 'maybe')")
        where_clauses.append("slr.job_id IS NULL")
    elif stage == 8:
        where_clauses.append("slr.job_id IS NOT NULL")
        joins.append("LEFT JOIN final_application_queue faq ON j.id = faq.job_id")
        where_clauses.append("faq.job_id IS NULL")
 
    if stage == 2 and not force_reprocess:
        # Optimized path: split queries to avoid slow cross-table OR scans causing timeouts
        where_ext = list(where_clauses)
        where_ext.append("(j.description IS NULL OR j.requirements IS NULL)")
        query_ext = f"""
            SELECT {', '.join(query_cols)}
            FROM job j
            {' '.join(joins)}
            WHERE {' AND '.join(where_ext)}
            ORDER BY j.date_added DESC
            LIMIT %s
        """
        
        where_emb = list(where_clauses)
        where_emb.append("NOT EXISTS (SELECT 1 FROM job_embeddings je WHERE je.job_id = j.id)")
        query_emb = f"""
            SELECT {', '.join(query_cols)}
            FROM job j
            {' '.join(joins)}
            WHERE {' AND '.join(where_emb)}
            ORDER BY j.date_added DESC
            LIMIT %s
        """
        
        try:
            dp.conn.execute_sql("SET statement_timeout = 60000")
            rows_ext = dp.conn.execute_sql(query_ext, (limit,), fetch=True) or []
            rows_emb = dp.conn.execute_sql(query_emb, (limit,), fetch=True) or []
            
            seen_ids = set()
            combined_rows = []
            for r in rows_ext + rows_emb:
                rid = r.get("id") if isinstance(r, dict) else r[0]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    combined_rows.append(r)
            rows = combined_rows[:limit]
        except Exception as e:
            print(f"Error querying DB for Stage {stage}: {e}")
            return []
    else:
        query = f"""
            SELECT {', '.join(query_cols)}
            FROM job j
            {' '.join(joins)}
            WHERE {' AND '.join(where_clauses)}
            ORDER BY j.date_added DESC
            LIMIT %s
        """
        
        try:
            dp.conn.execute_sql("SET statement_timeout = 60000")
            raw_rows = dp.conn.execute_sql(query, (limit,), fetch=True) or []
            seen_ids = set()
            deduped_rows = []
            for r in raw_rows:
                rid = r.get("id") if isinstance(r, dict) else r[0]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    deduped_rows.append(r)
            rows = deduped_rows
        except Exception as e:
            print(f"Error querying DB for Stage {stage}: {e}")
            return []
        
    if not rows:
        print(f"No jobs found in DB for Stage {stage} processing.")
        return []
        
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
    for row in rows:
        job = _row_to_job_dict(row)
        
        if stage >= 6:
            job['best_archetype'] = _get_val(row, "best_archetype")
            semantic_score = _get_val(row, "semantic_score")
            job['semantic_score'] = semantic_score if semantic_score is not None else 0.0
            job['semantic_score_percent'] = int(round(semantic_score * 100)) if semantic_score is not None else 0
            job['title_similarity'] = _get_val(row, "title_similarity")
            job['requirements_similarity'] = _get_val(row, "requirements_similarity")
            job['responsibility_similarity'] = _get_val(row, "responsibility_similarity")
            job['adjusted_score'] = _get_val(row, "adjusted_score")
            job['retrieval_metadata'] = {
                "semantic_score": job['semantic_score'],
                "semantic_score_percent": job['semantic_score_percent'],
                "best_archetype": _get_val(row, "best_archetype")
            }
                
        if stage >= 7:
            cheap_fit = _get_val(row, "cheap_fit_score")
            cheap_dec = _get_val(row, "cheap_decision")
            if cheap_dec or cheap_fit is not None:
                strengths = _get_val(row, "cheap_strengths")
                if isinstance(strengths, str):
                    try: strengths = json.loads(strengths)
                    except Exception: strengths = []
                concerns = _get_val(row, "cheap_concerns")
                if isinstance(concerns, str):
                    try: concerns = json.loads(concerns)
                    except Exception: concerns = []
                
                hard_requirements = _get_val(row, "cheap_hard_requirements_and_tools")
                if isinstance(hard_requirements, str):
                    try: hard_requirements = json.loads(hard_requirements)
                    except Exception: hard_requirements = []
                elif not isinstance(hard_requirements, list):
                    hard_requirements = []

                core_resp = _get_val(row, "cheap_core_responsibilities")
                if isinstance(core_resp, str):
                    try: core_resp = json.loads(core_resp)
                    except Exception: core_resp = []
                elif not isinstance(core_resp, list):
                    core_resp = []

                years_exp = _get_val(row, "cheap_years_of_experience")
                if isinstance(years_exp, str):
                    try: years_exp = json.loads(years_exp)
                    except Exception: years_exp = {}
                elif not isinstance(years_exp, dict):
                    years_exp = {}

                dom_edu = _get_val(row, "cheap_domain_and_education")
                if isinstance(dom_edu, str):
                    try: dom_edu = json.loads(dom_edu)
                    except Exception: dom_edu = {}
                elif not isinstance(dom_edu, dict):
                    dom_edu = {}

                raw_resp = _get_val(row, "cheap_raw_response")
                if isinstance(raw_resp, str):
                    try: raw_resp = json.loads(raw_resp)
                    except Exception: raw_resp = {}
                elif not isinstance(raw_resp, dict):
                    raw_resp = {}
                    
                job['cheap_llm_result'] = {
                    "fit_score": cheap_fit if cheap_fit is not None else 50,
                    "decision": cheap_dec or "maybe",
                    "strengths": strengths or [],
                    "concerns": concerns or [],
                    "hard_requirements_and_tools": hard_requirements,
                    "core_responsibilities": core_resp,
                    "years_of_experience": years_exp,
                    "domain_and_education": dom_edu,
                    "raw_response": raw_resp
                }
                
        if stage >= 8:
            strong_final = _get_val(row, "strong_final_score")
            strong_rec = _get_val(row, "strong_apply_rec")
            strong_prio = _get_val(row, "strong_priority")
            if strong_rec or strong_final is not None:
                red_flags = _get_val(row, "strong_red_flags")
                if isinstance(red_flags, str):
                    try: red_flags = json.loads(red_flags)
                    except Exception: red_flags = []
                tailoring_notes = _get_val(row, "strong_tailoring_notes")
                if isinstance(tailoring_notes, str):
                    try: tailoring_notes = json.loads(tailoring_notes)
                    except Exception: tailoring_notes = []

                comp_fit = _get_val(row, "strong_company_scale_fit")
                if isinstance(comp_fit, str):
                    try: comp_fit = json.loads(comp_fit)
                    except Exception: comp_fit = {}
                elif not isinstance(comp_fit, dict):
                    comp_fit = {}

                car_traj = _get_val(row, "strong_career_trajectory")
                if isinstance(car_traj, str):
                    try: car_traj = json.loads(car_traj)
                    except Exception: car_traj = {}
                elif not isinstance(car_traj, dict):
                    car_traj = {}

                sen_cal = _get_val(row, "strong_seniority_scope_calibration")
                if isinstance(sen_cal, str):
                    try: sen_cal = json.loads(sen_cal)
                    except Exception: sen_cal = {}
                elif not isinstance(sen_cal, dict):
                    sen_cal = {}

                hero_match = _get_val(row, "strong_hero_story_match")
                if isinstance(hero_match, str):
                    try: hero_match = json.loads(hero_match)
                    except Exception: hero_match = {}
                elif not isinstance(hero_match, dict):
                    hero_match = {}

                proj_comp = _get_val(row, "strong_project_complexity")
                if isinstance(proj_comp, str):
                    try: proj_comp = json.loads(proj_comp)
                    except Exception: proj_comp = {}
                elif not isinstance(proj_comp, dict):
                    proj_comp = {}

                shadow_fric = _get_val(row, "strong_shadow_work_friction")
                if isinstance(shadow_fric, str):
                    try: shadow_fric = json.loads(shadow_fric)
                    except Exception: shadow_fric = {}
                elif not isinstance(shadow_fric, dict):
                    shadow_fric = {}

                dom_fric = _get_val(row, "strong_domain_business_model_friction")
                if isinstance(dom_fric, str):
                    try: dom_fric = json.loads(dom_fric)
                    except Exception: dom_fric = {}
                elif not isinstance(dom_fric, dict):
                    dom_fric = {}

                recruiter_red_flags = _get_val(row, "strong_recruiter_red_flags")
                if isinstance(recruiter_red_flags, str):
                    try: recruiter_red_flags = json.loads(recruiter_red_flags)
                    except Exception: recruiter_red_flags = {}
                elif not isinstance(recruiter_red_flags, dict):
                    recruiter_red_flags = {}

                driving_pts = _get_val(row, "strong_driving_points")
                if isinstance(driving_pts, str):
                    try: driving_pts = json.loads(driving_pts)
                    except Exception: driving_pts = []
                elif not isinstance(driving_pts, list):
                    driving_pts = []

                raw_resp = _get_val(row, "strong_raw_response")
                if isinstance(raw_resp, str):
                    try: raw_resp = json.loads(raw_resp)
                    except Exception: raw_resp = {}
                elif not isinstance(raw_resp, dict):
                    raw_resp = {}
                    
                job['strong_llm_result'] = {
                    "final_score": strong_final if strong_final is not None else 50,
                    "priority": strong_prio or "medium",
                    "apply_recommendation": strong_rec or "maybe",
                    "red_flags": red_flags or [],
                    "tailoring_notes": tailoring_notes or [],
                    "recruiter_bait_likelihood": _get_val(row, "strong_bait") or "medium",
                    "detailed_fit_analysis": _get_val(row, "strong_fit_analysis") or "",
                    "company_scale_fit": comp_fit,
                    "career_trajectory": car_traj,
                    "seniority_scope_calibration": sen_cal,
                    "hero_story_match": hero_match,
                    "project_complexity": proj_comp,
                    "shadow_work_friction": shadow_fric,
                    "domain_business_model_friction": dom_fric,
                    "recruiter_red_flags": recruiter_red_flags,
                    "driving_points": driving_pts,
                    "raw_response": raw_resp
                }
                job['final_score'] = strong_final
                job['priority'] = strong_prio
                job['apply_recommendation'] = strong_rec
                
        jobs.append(job)
    return jobs


def ensure_db_status_loaded(jobs: List[Dict], dp: Optional[DataPuller], skip_db: bool):
    """
    Checks the database to see which jobs already have:
    - vector scores
    - cheap LLM results
    - strong LLM results
    
    Attaches the following keys to each job dict:
    - '_db_has_vs': bool
    - '_db_has_clr': bool
    - '_db_has_slr': bool
    - '_db_has_all_three': bool
    - '_db_checked': bool
    
    If the record exists, it also pulls the data and populates the job dict.
    """
    if not jobs:
        return
        
    # Check if we already checked these jobs
    unchecked_jobs = [j for j in jobs if not j.get('_db_checked')]
    if not unchecked_jobs:
        return
        
    if skip_db or dp is None:
        for j in unchecked_jobs:
            j['_db_has_vs'] = False
            j['_db_has_clr'] = False
            j['_db_has_slr'] = False
            j['_db_has_faq'] = False
            j['_db_has_all_three'] = False
            j['_db_checked'] = True
        return

    # Extract all valid job IDs
    job_ids = []
    job_map = {}
    for j in unchecked_jobs:
        job_id = j.get('metadata', {}).get('job_id')
        if job_id is not None:
            try:
                job_id = int(job_id)
                job_ids.append(job_id)
                job_map[job_id] = j
            except (ValueError, TypeError):
                pass
                
    # Initialize all unchecked jobs to False/None
    for j in unchecked_jobs:
        j['_db_has_vs'] = False
        j['_db_has_clr'] = False
        j['_db_has_slr'] = False
        j['_db_has_faq'] = False
        j['_db_has_all_three'] = False
        j['_db_checked'] = True

    if not job_ids:
        return

    # Query 1: Fetch vector scores
    try:
        vs_query = """
            SELECT DISTINCT ON (job_id) 
                job_id, archetype_name, semantic_score, title_similarity, 
                requirements_similarity, responsibility_similarity, adjusted_score
            FROM vector_scores
            WHERE job_id IN %s
            ORDER BY job_id, id DESC
        """
        vs_rows = dp.conn.execute_sql(vs_query, params=(tuple(job_ids),), fetch=True) or []
        for row in vs_rows:
            jid = row.get("job_id") if isinstance(row, dict) else row[0]
            if jid in job_map:
                job = job_map[jid]
                job['_db_has_vs'] = True
                
                # Pull vector score data
                best_archetype = row.get("archetype_name") if isinstance(row, dict) else row[1]
                semantic_score = row.get("semantic_score") if isinstance(row, dict) else row[2]
                title_sim = row.get("title_similarity") if isinstance(row, dict) else row[3]
                req_sim = row.get("requirements_similarity") if isinstance(row, dict) else row[4]
                resp_sim = row.get("responsibility_similarity") if isinstance(row, dict) else row[5]
                adj_score = row.get("adjusted_score") if isinstance(row, dict) else row[6]
                
                job['semantic_score'] = semantic_score if semantic_score is not None else 0.0
                job['semantic_score_percent'] = int(round(job['semantic_score'] * 100)) if job['semantic_score'] is not None else 0
                job['title_similarity'] = title_sim
                job['requirements_similarity'] = req_sim
                job['responsibility_similarity'] = resp_sim
                job['adjusted_score'] = adj_score
                job['best_archetype'] = best_archetype
                job['retrieval_metadata'] = {
                    "semantic_score": job['semantic_score'],
                    "semantic_score_percent": job['semantic_score_percent'],
                    "best_archetype": best_archetype
                }
    except Exception as e:
        print(f"Warning: Failed to fetch vector scores in ensure_db_status_loaded: {e}")

    # Query 2: Fetch cheap LLM results
    try:
        clr_query = """
            SELECT DISTINCT ON (job_id) 
                job_id, fit_score, decision, strengths, concerns, 
                hard_requirements_and_tools, core_responsibilities, 
                years_of_experience, domain_and_education, raw_response
            FROM cheap_llm_results
            WHERE job_id IN %s
            ORDER BY job_id, id DESC
        """
        clr_rows = dp.conn.execute_sql(clr_query, params=(tuple(job_ids),), fetch=True) or []
        for row in clr_rows:
            jid = row.get("job_id") if isinstance(row, dict) else row[0]
            if jid in job_map:
                job = job_map[jid]
                job['_db_has_clr'] = True
                
                # Pull cheap LLM data
                def _parse_json(val):
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except Exception:
                            return None
                    return val
                
                fit_score = row.get("fit_score") if isinstance(row, dict) else row[1]
                decision = row.get("decision") if isinstance(row, dict) else row[2]
                strengths = _parse_json(row.get("strengths") if isinstance(row, dict) else row[3])
                concerns = _parse_json(row.get("concerns") if isinstance(row, dict) else row[4])
                hard_reqs = _parse_json(row.get("hard_requirements_and_tools") if isinstance(row, dict) else row[5])
                core_resp = _parse_json(row.get("core_responsibilities") if isinstance(row, dict) else row[6])
                years_exp = _parse_json(row.get("years_of_experience") if isinstance(row, dict) else row[7])
                dom_edu = _parse_json(row.get("domain_and_education") if isinstance(row, dict) else row[8])
                raw_resp = _parse_json(row.get("raw_response") if isinstance(row, dict) else row[9])
                
                job['cheap_llm_result'] = {
                    "fit_score": fit_score if fit_score is not None else 50,
                    "decision": decision or "maybe",
                    "strengths": strengths or [],
                    "concerns": concerns or [],
                    "hard_requirements_and_tools": hard_reqs or [],
                    "core_responsibilities": core_resp or [],
                    "years_of_experience": years_exp or {},
                    "domain_and_education": dom_edu or {},
                    "raw_response": raw_resp or {}
                }
    except Exception as e:
        print(f"Warning: Failed to fetch cheap LLM results in ensure_db_status_loaded: {e}")

    # Query 3: Fetch strong LLM results
    try:
        slr_query = """
            SELECT DISTINCT ON (job_id) 
                job_id, final_score, priority, apply_recommendation, 
                red_flags, tailoring_notes, recruiter_bait_likelihood, 
                detailed_fit_analysis, company_scale_fit, career_trajectory, 
                seniority_scope_calibration, hero_story_match,
                project_complexity, shadow_work_friction, domain_business_model_friction,
                recruiter_red_flags, driving_points, raw_response
            FROM strong_llm_results
            WHERE job_id IN %s
            ORDER BY job_id, id DESC
        """
        slr_rows = dp.conn.execute_sql(slr_query, params=(tuple(job_ids),), fetch=True) or []
        for row in slr_rows:
            jid = row.get("job_id") if isinstance(row, dict) else row[0]
            if jid in job_map:
                job = job_map[jid]
                job['_db_has_slr'] = True
                
                # Pull strong LLM data
                def _parse_json(val):
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except Exception:
                            return None
                    return val
                
                final_score = row.get("final_score") if isinstance(row, dict) else row[1]
                priority = row.get("priority") if isinstance(row, dict) else row[2]
                apply_rec = row.get("apply_recommendation") if isinstance(row, dict) else row[3]
                red_flags = _parse_json(row.get("red_flags") if isinstance(row, dict) else row[4])
                tailoring_notes = _parse_json(row.get("tailoring_notes") if isinstance(row, dict) else row[5])
                bait = row.get("recruiter_bait_likelihood") if isinstance(row, dict) else row[6]
                fit_analysis = row.get("detailed_fit_analysis") if isinstance(row, dict) else row[7]
                comp_fit = _parse_json(row.get("company_scale_fit") if isinstance(row, dict) else row[8])
                car_traj = _parse_json(row.get("career_trajectory") if isinstance(row, dict) else row[9])
                sen_cal = _parse_json(row.get("seniority_scope_calibration") if isinstance(row, dict) else row[10])
                hero_match = _parse_json(row.get("hero_story_match") if isinstance(row, dict) else row[11])
                proj_comp = _parse_json(row.get("project_complexity") if isinstance(row, dict) else row[12])
                shadow_fric = _parse_json(row.get("shadow_work_friction") if isinstance(row, dict) else row[13])
                dom_fric = _parse_json(row.get("domain_business_model_friction") if isinstance(row, dict) else row[14])
                rec_red_flags = _parse_json(row.get("recruiter_red_flags") if isinstance(row, dict) else row[15])
                driving_pts = _parse_json(row.get("driving_points") if isinstance(row, dict) else row[16])
                raw_resp = _parse_json(row.get("raw_response") if isinstance(row, dict) else row[17])
                
                job['strong_llm_result'] = {
                    "final_score": final_score if final_score is not None else 50,
                    "priority": priority or "medium",
                    "apply_recommendation": apply_rec or "maybe",
                    "red_flags": red_flags or [],
                    "tailoring_notes": tailoring_notes or [],
                    "recruiter_bait_likelihood": bait or "medium",
                    "detailed_fit_analysis": fit_analysis or "",
                    "company_scale_fit": comp_fit or {},
                    "career_trajectory": car_traj or {},
                    "seniority_scope_calibration": sen_cal or {},
                    "hero_story_match": hero_match or {},
                    "project_complexity": proj_comp or {},
                    "shadow_work_friction": shadow_fric or {},
                    "domain_business_model_friction": dom_fric or {},
                    "recruiter_red_flags": rec_red_flags or {},
                    "driving_points": driving_pts or [],
                    "raw_response": raw_resp or {}
                }
                job['final_score'] = final_score
                job['priority'] = priority
                job['apply_recommendation'] = apply_rec
    except Exception as e:
        print(f"Warning: Failed to fetch strong LLM results in ensure_db_status_loaded: {e}")

    # Query 4: Fetch final queue status
    try:
        faq_query = """
            SELECT DISTINCT ON (job_id) 
                job_id, final_score, priority, apply_recommendation
            FROM final_application_queue
            WHERE job_id IN %s
            ORDER BY job_id, id DESC
        """
        faq_rows = dp.conn.execute_sql(faq_query, params=(tuple(job_ids),), fetch=True) or []
        for row in faq_rows:
            jid = row.get("job_id") if isinstance(row, dict) else row[0]
            if jid in job_map:
                job = job_map[jid]
                job['_db_has_faq'] = True
                
                # Load final queue data if present
                job['final_score'] = row.get("final_score") if isinstance(row, dict) else row[1]
                job['priority'] = row.get("priority") if isinstance(row, dict) else row[2]
                job['apply_recommendation'] = row.get("apply_recommendation") if isinstance(row, dict) else row[3]
    except Exception as e:
        print(f"Warning: Failed to fetch final application queue in ensure_db_status_loaded: {e}")

    # Set all_three flag
    for j in unchecked_jobs:
        if j.get('_db_has_vs') and j.get('_db_has_clr') and j.get('_db_has_slr'):
            j['_db_has_all_three'] = True

    # If cheap LLM, strong LLM, and final queue scores already exist, remove from memory
    i = len(jobs) - 1
    while i >= 0:
        j = jobs[i]
        if j.get('_db_has_clr') and j.get('_db_has_slr') and j.get('_db_has_faq'):
            print(f"Failsafe: Job ID {j.get('metadata', {}).get('job_id')} already has cheap LLM, strong LLM, and final queue scores. Removing from memory.")
            jobs.pop(i)
        i -= 1



