import os
import yaml
from dotenv import load_dotenv

# modules
from app.postgres_mgr import PostgresManager

def _load_user_config() -> dict:
    """Load the user_preferences.yaml file if it exists."""
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    config = {}
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    return config

def make_db():
    load_dotenv()
    # Load from user_preferences.yaml with .env fallback
    user_config = _load_user_config()
    host = user_config.get("db_host") or os.getenv("DB_HOST") or "localhost"
    port = user_config.get("db_port") or os.getenv("DB_PORT") or "5432"
    user = user_config.get("db_user") or os.getenv("DB_USER") or "postgres"
    password = user_config.get("db_password") or os.getenv("DB_PASSWORD") or ""
    db_name = user_config.get("db_name") or os.getenv("DB_NAME") or "web_scraper_db"

    pg_mgr = PostgresManager(host, int(port), user, password)
    pg_mgr.create_database(db_name)
    pg_mgr.connect(db_name)

    # connect to the new database and create tables

    has_vector_ext = False
    try:
        pg_mgr.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;", dbname=db_name)
        has_vector_ext = True
        print("pgvector extension enabled successfully.")
    except Exception as e:
        print(f"Warning: Could not create extension vector: {e}")

    tables = {
        "names": [
            "job",
            "company",
            "office",
            "job_embeddings",
            "archetype_embeddings",
            "token_usage_log",
            "cheap_llm_results",
            "strong_llm_results",
            "vector_scores",
            "final_application_queue",
            "rag_documents",
            "rag_metadata"
        ],
        "columns": [
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_name": "VARCHAR(255)",
                "company_id": "INT",
                "date_added": "DATE",
                "office_id": "INT",
                "link": "VARCHAR(1000)",
                "pay_range": "VARCHAR(1000)",
                "description": "TEXT",
                "job_summary": "TEXT",
                "title_rating": "INT",
                "summary_rating": "INT",
                "jsr_reasoning": "TEXT",
                "hiring_manager": "VARCHAR(255)",
                "recruiter": "VARCHAR(255)",
                "skills_to_work_on": "TEXT",
                "notes": "TEXT",
                "my_title_score": "INT",
                "my_summary_score": "INT",
                "skip": "BOOLEAN",
                "flexibility":"VARCHAR(100)",
                "source":"VARCHAR(255)",
                "requirements": "JSONB",
                "responsibilities": "JSONB"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "company_name": "VARCHAR(255)",
                "office_id": "INT",
                "primary_industry": "VARCHAR(255)",
                "company_url": "VARCHAR(255)"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "company_id": "INT",
                "city": "VARCHAR(255)",
                "state": "VARCHAR(255)",
                "country": "VARCHAR(255)",
                "location": "VARCHAR(1000)"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT UNIQUE",
                "title_embedding": "FLOAT8[]",
                "requirements_embedding": "FLOAT8[]",
                "responsibilities_embedding": "FLOAT8[]",
                "description_embedding": "FLOAT8[]",
                "pay_embedding": "FLOAT8[]",
                "location_embedding": "FLOAT8[]",
                "date_generated": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "archetype_name": "VARCHAR(255) UNIQUE",
                "archetype_type": "VARCHAR(100)",
                "title_embedding": "FLOAT8[]",
                "requirements_embedding": "FLOAT8[]",
                "responsibilities_embedding": "FLOAT8[]",
                "metadata": "JSONB",
                "date_generated": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "run_id": "VARCHAR(100) NOT NULL",
                "run_timestamp": "TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
                "provider": "VARCHAR(100) NOT NULL",
                "model": "VARCHAR(100) NOT NULL",
                "operation": "VARCHAR(100) NOT NULL",
                "input_tokens": "INTEGER NOT NULL",
                "output_tokens": "INTEGER NOT NULL",
                "total_tokens": "INTEGER NOT NULL"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT NOT NULL",
                "fit_score": "INT",
                "decision": "VARCHAR(20)",
                "strengths": "JSONB",
                "concerns": "JSONB",
                "hard_requirements_and_tools": "JSONB",
                "core_responsibilities": "JSONB",
                "years_of_experience": "JSONB",
                "domain_and_education": "JSONB",
                "raw_response": "JSONB",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT NOT NULL",
                "final_score": "INT",
                "priority": "VARCHAR(20)",
                "apply_recommendation": "VARCHAR(20)",
                "red_flags": "JSONB",
                "tailoring_notes": "JSONB",
                "recruiter_bait_likelihood": "VARCHAR(20)",
                "detailed_fit_analysis": "TEXT",
                "company_scale_fit": "JSONB",
                "career_trajectory": "JSONB",
                "project_complexity": "JSONB",
                "recruiter_red_flags": "JSONB",
                "driving_points": "JSONB",
                "raw_response": "JSONB",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT NOT NULL UNIQUE",
                "archetype_name": "VARCHAR(255) NOT NULL",
                "semantic_score": "REAL",
                "title_similarity": "REAL",
                "requirements_similarity": "REAL",
                "responsibility_similarity": "REAL",
                "adjusted_score": "REAL",
                "rank": "INT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT NOT NULL UNIQUE",
                "final_score": "REAL",
                "priority": "VARCHAR(20)",
                "apply_recommendation": "VARCHAR(20)",
                "queue_position": "INT",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "job_id": "INT",
                "chunk_type": "VARCHAR(50) NOT NULL",
                "content": "TEXT NOT NULL",
                "embedding": "vector" if has_vector_ext else "TEXT",
                "metadata": "JSONB",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "key_name": "VARCHAR(100) UNIQUE NOT NULL",
                "value_text": "TEXT",
                "updated_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        ]
    }
    for i in range(len(tables["names"])):
        pg_mgr.create_table(tables["names"][i], tables["columns"][i], True, db_name)

    fk_constraints = [
        ("job", ["company_id"], "company", ["id"], "fk_job_company", "", db_name),
        ("job", ["office_id"], "office", ["id"], "fk_job_office", "", db_name),
        ("company", ["office_id"], "office", ["id"], "fk_company_office", "", db_name),
        ("office", ["company_id"], "company", ["id"], "fk_office_company", "", db_name),
        ("job_embeddings", ["job_id"], "job", ["id"], "fk_job_embeddings_job", "CASCADE", db_name),
        ("cheap_llm_results", ["job_id"], "job", ["id"], "fk_cheap_llm_results_job", "CASCADE", db_name),
        ("strong_llm_results", ["job_id"], "job", ["id"], "fk_strong_llm_results_job", "CASCADE", db_name),
        ("vector_scores", ["job_id"], "job", ["id"], "fk_vector_scores_job", "CASCADE", db_name),
        ("final_application_queue", ["job_id"], "job", ["id"], "fk_final_application_queue_job", "CASCADE", db_name),
        ("rag_documents", ["job_id"], "job", ["id"], "fk_rag_documents_job", "CASCADE", db_name)
    ]
    for src_tbl, src_cols, ref_tbl, ref_cols, constraint_name, on_del, db in fk_constraints:
        try:
            pg_mgr.add_foreign_key(src_tbl, src_cols, ref_tbl, ref_cols, constraint_name, on_del, db)
        except Exception as fk_err:
            pass
    
    # Try creating HNSW index on rag_documents embedding
    try:
        pg_mgr.execute_sql(
            "CREATE INDEX IF NOT EXISTS idx_rag_documents_embedding ON rag_documents USING hnsw (embedding vector_cosine_ops);",
            dbname=db_name
        )
    except Exception as e:
        print(f"Warning: Could not create HNSW vector index: {e}")

    pg_mgr.close()
    
    return "Database and tables created successfully."
