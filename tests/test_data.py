"""
Dummy data fixtures for testing each stage of the job pipeline.

Each stage's input/output data shapes are defined here so that:
- Any stage can be tested in isolation with realistic synthetic data
- Stages can be chained by feeding the output of one stage into the next
"""

import json
import os
from typing import Any, Dict, List


# =====================================================
# STAGE 0: SETUP OUTPUT
# =====================================================

def make_dummy_setup_output() -> dict:
    """Dummy output from Stage 0 (setup / profile extraction)."""
    return {
        "resume": "Experienced software engineer with 8 years in Python, Rust, and cloud infrastructure.",
        "user_profile": "Senior Python developer seeking remote roles in AI/ML and backend engineering.",
        "skills": ["Python", "Rust", "Kubernetes", "Docker", "PostgreSQL", "FastAPI", "AWS", "Machine Learning"],
        "job_titles": ["Senior Software Engineer", "Backend Engineer", "ML Engineer", "Platform Engineer"],
        "db_config": {
            "dbname": os.getenv("DB_NAME", "test_jobs"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", ""),
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
        },
        "user_preferences": {
            "target_cities": ["San Francisco, CA", "New York, NY", "Austin, TX"],
            "work_types": ["remote", "hybrid"],
            "seniority_levels": ["senior", "staff"],
            "pay_range": "80k-180k",
            "timezones": ["pst", "cst", "est"],
        },
    }


# =====================================================
# DUMMY RAW JOBS (as returned from scraping)
# =====================================================

def make_dummy_scraped_jobs(count: int = 10) -> List[Dict[str, Any]]:
    """
    Returns a list of raw job dicts that mimic the output of Stage 1 scraping.
    Each dict has the shape used across the pipeline.
    """
    titles = [
        "Senior Backend Engineer",
        "Python Developer",
        "Machine Learning Engineer",
        "Platform Engineer",
        "Full Stack Developer",
        "Data Engineer",
        "Site Reliability Engineer",
        "Staff Software Engineer",
        "AI Engineer",
        "Cloud Infrastructure Engineer",
    ]
    companies = [
        "TechCorp", "DataFlow", "AI Labs", "CloudNative Inc", "StartupX",
        "BigData Co", "InfraPro", "InnovateTech", "Neural Systems", "CloudBase",
    ]
    descriptions = [
        "We are looking for a senior backend engineer to design and build scalable microservices using Python, FastAPI, and PostgreSQL. "
        "You will work on distributed systems, REST APIs, and cloud-native architectures. Remote-first team with flexible hours. "
        "Required: 5+ years Python, experience with Docker/Kubernetes, strong SQL skills. Nice to have: Rust, AWS, Kafka.",
        "Join our data team to build ML pipelines and data processing systems. You'll work with Python, Pandas, and Spark to process "
        "large datasets and deploy models to production. Hybrid role in San Francisco. 3+ years experience required.",
        "Design and implement ML models for our recommendation systems. Work with TensorFlow, PyTorch, and custom algorithms. "
        "Senior-level role requiring deep understanding of ML fundamentals and production deployment experience.",
        "Build and maintain our cloud infrastructure platform using Kubernetes, Terraform, and CI/CD pipelines. "
        "Focus on reliability, scalability, and automation. Remote role with quarterly team meetups.",
        "Full stack developer needed for our growing SaaS platform. React frontend, Python backend, deployed on AWS. "
        "Experience with TypeScript, Python, and cloud services required. Hybrid in Austin.",
    ]
    # Pad descriptions if count > len(descriptions)
    while len(descriptions) < count:
        descriptions.extend(descriptions)

    jobs = []
    for i in range(count):
        idx = i % len(titles)
        jobs.append({
            "id": i + 1,
            "source": ["indeed", "linkedin", "zip_recruiter", "google"][i % 4],
            "title": titles[idx],
            "url": f"https://example.com/job/{i+1}",
            "company": companies[i % len(companies)],
            "pay": f"{80 + (i * 10)}k - {120 + (i * 10)}k",
            "description": descriptions[idx],
            "city": ["San Francisco", "New York", "Austin", "Remote"][i % 4],
            "state": ["CA", "NY", "TX", None][i % 4],
            "location": ["San Francisco, CA", "New York, NY", "Austin, TX", "Remote"][i % 4],
            "flexibility": ["remote", "hybrid", "onsite", "remote"][i % 4],
        })
    return jobs


# =====================================================
# STAGE 1 OUTPUT / STAGE 2 INPUT
# =====================================================

def make_dummy_stage1_output(count: int = 10) -> List[Dict[str, Any]]:
    """
    Dummy output of Stage 1 (scraped + merged from DB). 
    Shape matches 'processed_job_pool' before AI pass.
    """
    raw = make_dummy_scraped_jobs(count)
    jobs = []
    for i, r in enumerate(raw):
        jobs.append({
            "metadata": {
                "job_id": r["id"],
                "source": r["source"],
            },
            "features": {
                "title": r["title"],
                "description": r["description"],
                "pay": r["pay"],
                "seniority": "senior" if i % 3 != 0 else "mid",
                "work_type": r["flexibility"],
                "timezone": ["pst", "est", "cst", "pst"][i % 4],
            },
            "embeddings": {
                "description_vector": None,
                "skills_vector": None,
            },
        })
    return jobs


# =====================================================
# STAGE 2 OUTPUT
# =====================================================

def make_dummy_stage2_output(count: int = 10) -> List[Dict[str, Any]]:
    """Dummy output of Stage 2 (after embeddings + LLM extraction)."""
    jobs = make_dummy_stage1_output(count)
    dummy_embedding = [0.01 * (idx + 1) for idx in range(384)]
    for i, j in enumerate(jobs):
        j["features"]["skills"] = ["Python", "SQL", "Docker", "Kubernetes", "AWS"][:3 + (i % 3)]
        j["features"]["requirements"] = ["5+ years experience", "BS in CS", "Strong communication"]
        j["features"]["summary"] = f"Job summary for {j['features']['title']}: a great role."
        j["embeddings"]["title_vector"] = dummy_embedding
        j["embeddings"]["skills_vector"] = dummy_embedding
        j["embeddings"]["requirements_vector"] = dummy_embedding
        j["embeddings"]["description_vector"] = dummy_embedding
    return jobs


# =====================================================
# STAGE 3 OUTPUT (post rule-filtering)
# =====================================================

def make_dummy_stage3_output(count: int = 10, skip_ratio: float = 0.3) -> List[Dict[str, Any]]:
    """Dummy output of Stage 3. Some jobs marked as 'skip'."""
    jobs = make_dummy_stage2_output(count)
    for i, j in enumerate(jobs):
        j["skip"] = i < int(count * skip_ratio)
    return jobs


# =====================================================
# STAGE 4 OUTPUT (post archetype engine)
# =====================================================

def make_dummy_stage4_output(count: int = 7) -> List[Dict[str, Any]]:
    """Dummy output of Stage 4 (archetypes loaded, jobs not yet scored)."""
    jobs = make_dummy_stage3_output(count, skip_ratio=0.0)
    # Remove skipped
    jobs = [j for j in jobs if not j.get("skip")]
    # Archetype comparison data will be added in Stage 5
    return jobs


# =====================================================
# STAGE 5 OUTPUT (post vector scoring)
# =====================================================

def make_dummy_stage5_output(count: int = 7) -> List[Dict[str, Any]]:
    """Dummy output of Stage 5 (after archetype scoring + filtering)."""
    jobs = make_dummy_stage4_output(count)
    for i, j in enumerate(jobs):
        score = max(0.5, min(0.98, 0.6 + 0.05 * i))
        j["semantic_score"] = score
        j["semantic_score_percent"] = int(round(score * 100))
        j["title_similarity"] = score * 0.9
        j["skills_similarity"] = score * 0.85
        j["responsibility_similarity"] = score * 0.8
        j["adjusted_score"] = score
        j["best_archetype"] = ["Resume", "User Profile", "Senior Backend", "ML Engineer"][i % 4]
        j["retrieval_metadata"] = {
            "scores": [{"name": j["best_archetype"], "score": score}],
            "semantic_score": score,
            "semantic_score_percent": j["semantic_score_percent"],
            "best_archetype": j["best_archetype"],
        }
        j["archetype_matches"] = [{
            "archetype_name": j["best_archetype"],
            "score": score,
            "title_similarity": j["title_similarity"],
            "skills_similarity": j["skills_similarity"],
            "responsibility_similarity": j["responsibility_similarity"],
        }]
    return jobs


# =====================================================
# STAGE 6 OUTPUT (post cheap LLM)
# =====================================================

def make_dummy_stage6_output(count: int = 5) -> List[Dict[str, Any]]:
    """Dummy output of Stage 6 (cheap LLM classified + shortlisted)."""
    jobs = make_dummy_stage5_output(count)
    for i, j in enumerate(jobs):
        j["cheap_llm_result"] = {
            "fit_score": [85, 72, 60, 90, 78][i % 5],
            "decision": "apply" if i < 3 else "maybe",
            "strengths": [f"Strong match in skill set", f"Relevant experience"][i % 2],
            "concerns": [f"Missing domain expertise", ""][i % 2],
        }
    return [j for j in jobs if j.get("cheap_llm_result", {}).get("decision") in ("apply", "maybe")]


# =====================================================
# STAGE 7 OUTPUT (post strong LLM)
# =====================================================

def make_dummy_stage7_output(count: int = 5) -> List[Dict[str, Any]]:
    """Dummy output of Stage 7 (strong LLM reranked)."""
    jobs = make_dummy_stage6_output(count)
    for i, j in enumerate(jobs):
        j["strong_llm_result"] = {
            "final_score": [88, 75, 65, 92, 80][i % 5],
            "priority": ["high", "medium", "low", "high", "medium"][i % 5],
            "apply_recommendation": "apply" if i < 3 else "maybe",
            "red_flags": [f"Salary may be below market", ""][i % 2],
            "tailoring_notes": [f"Highlight experience with Python and Kubernetes"],
            "recruiter_bait_likelihood": ["low", "medium", "medium", "low", "low"][i % 5],
            "detailed_fit_analysis": f"Good fit for {j['features']['title']} role.",
        }
    return jobs


# =====================================================
# STAGE 8 OUTPUT (final queue)
# =====================================================

def make_dummy_stage8_output(count: int = 5) -> List[Dict[str, Any]]:
    """Dummy output of Stage 8 (final ranked queue)."""
    jobs = make_dummy_stage7_output(count)
    for i, j in enumerate(jobs):
        j["final_score"] = j.get("strong_llm_result", {}).get("final_score", 50) + 5
        j["priority"] = ["high", "high", "medium", "high", "medium"][i % 5]
        j["apply_recommendation"] = "apply" if i < 3 else "maybe"
        j["queue_position"] = i + 1
    return jobs


# =====================================================
# HELPER: LOAD DUMMY DATA FOR A SPECIFIC STAGE
# =====================================================

DUMMY_GENERATORS = {
    0: make_dummy_setup_output,
    1: lambda: make_dummy_scraped_jobs(20),
    2: lambda: make_dummy_stage1_output(15),
    "2_after_ai": lambda: make_dummy_stage2_output(15),
    3: lambda: make_dummy_stage3_output(15, skip_ratio=0.3),
    4: lambda: make_dummy_stage4_output(10),
    5: lambda: make_dummy_stage5_output(10),
    6: lambda: make_dummy_stage6_output(8),
    7: lambda: make_dummy_stage7_output(5),
    8: lambda: make_dummy_stage8_output(5),
}


def get_dummy_data(stage: int, variant: str = "") -> Any:
    """
    Return dummy data suitable for the given stage.
    
    Args:
        stage: Pipeline stage number (0-8)
        variant: Optional variant key (e.g., "after_ai" for stage 2)
    
    Returns:
        Dummy data appropriate for that stage's input.
    """
    key = stage
    if variant:
        key = f"{stage}_{variant}"
    gen = DUMMY_GENERATORS.get(key)
    if gen is None:
        gen = DUMMY_GENERATORS.get(stage)
    if gen is None:
        raise KeyError(f"No dummy data generator for stage {stage} (variant={variant!r})")
    return gen()


# =====================================================
# EXPORT: map stage numbers to expected data shapes
# =====================================================

STAGE_INPUT_SHAPES = {
    0: "Setup config (env vars, file paths)",
    1: "Setup output (resume, profile, skills, job_titles, db_config, user_preferences)",
    2: "List[dict]: scraped jobs with features (title, description, pay, etc.)",
    3: "List[dict]: jobs with embeddings + extracted skills/requirements/summary",
    4: "List[dict]: jobs with skip flags after rule filtering",
    5: "List[dict]: active jobs (non-skipped) + loaded archetypes",
    6: "List[dict]: jobs with archetype scores + semantic scores",
    7: "List[dict]: jobs with cheap LLM results, shortlisted",
    8: "List[dict]: jobs with strong LLM results, deeply analyzed",
}

STAGE_OUTPUT_SHAPES = {
    0: "dict: resume, user_profile, skills, job_titles, db_config, user_preferences",
    1: "List[dict]: scraped jobs with id, title, company, description, pay, etc.",
    2: "List[dict]: jobs with features, embeddings (title/skills/requirements vectors)",
    3: "List[dict]: jobs with skip=bool + DB sync'd skip status",
    4: "List[dict]: active jobs (non-skipped), archetype manager loaded",
    5: "List[dict]: jobs with semantic_score, archetype_matches, retrieval_metadata. filtered_pool saved to DB.",
    6: "List[dict]: shortlisted jobs (apply/maybe) with cheap_llm_result",
    7: "List[dict]: top-N deeply analyzed jobs with strong_llm_result",
    8: "List[dict]: final ranked queue with final_score, priority, apply_recommendation",
}