import os
from dotenv import load_dotenv

# modules
from app.postgres_mgr import PostgresManager

def make_db():
    load_dotenv()
    # define initialization params and initialize database
    host = os.getenv("DB_HOST") or "localhost"
    port = os.getenv("DB_PORT") or "5432"
    user = os.getenv("DB_USER") or "postgres"
    password = os.getenv("DB_PASSWORD") or ""
    db_name = os.getenv("DB_NAME") or "web_scraper_db"

    pg_mgr = PostgresManager(host, int(port), user, password)
    pg_mgr.create_database(db_name)
    pg_mgr.connect(db_name)

    # connect to the new database and create tables

    

    tables = {
        "names": [
            "job",
            "company",
            "office",
            "job_embeddings",
            "archetype_embeddings"
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
                "source":"VARCHAR(255)"
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
                "skills_embedding": "FLOAT8[]",
                "responsibilities_embedding": "FLOAT8[]",
                "description_embedding": "FLOAT8[]",
                "date_generated": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            },
            {
                "id": "INT PRIMARY KEY GENERATED ALWAYS AS IDENTITY",
                "archetype_name": "VARCHAR(255) UNIQUE",
                "archetype_type": "VARCHAR(100)",
                "title_embedding": "FLOAT8[]",
                "skills_embedding": "FLOAT8[]",
                "responsibilities_embedding": "FLOAT8[]",
                "metadata": "JSONB",
                "date_generated": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            }
        ]
    }
    for i in range(len(tables["names"])):
        pg_mgr.create_table(tables["names"][i], tables["columns"][i], True, 'web_scraper_db')

    pg_mgr.add_foreign_key("job", ["company_id"], "company", ["id"], "fk_job_company", "", "web_scraper_db")
    pg_mgr.add_foreign_key("job", ["office_id"], "office", ["id"], "fk_job_office", "", "web_scraper_db")
    pg_mgr.add_foreign_key("company", ["office_id"], "office", ["id"], "fk_company_office", "", "web_scraper_db")
    pg_mgr.add_foreign_key("office", ["company_id"], "company", ["id"], "fk_office_company", "", "web_scraper_db")
    pg_mgr.add_foreign_key("job_embeddings", ["job_id"], "job", ["id"], "fk_job_embeddings_job", "CASCADE", "web_scraper_db")
    
    pg_mgr.close()
    
    return "Database and tables created successfully."
