import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Add the project root to python path to import app modules if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

db_name = os.getenv("DB_NAME", "job_searcher")
db_user = os.getenv("DB_USER", "postgres")
db_password = os.getenv("DB_PASSWORD", "")
db_host = os.getenv("DB_HOST", "localhost")
db_port = os.getenv("DB_PORT", "5432")

conn = psycopg2.connect(
    dbname=db_name,
    user=db_user,
    password=db_password,
    host=db_host,
    port=db_port
)

try:
    with conn.cursor(cursor_factory=RealDictCursor) as cur:
        # Run the user's exact query
        user_query = """
        SELECT 
            j.id AS "Job ID",
            j.skip AS Skip,
            j.date_added AS "Date Added",
            j.job_name AS "Job Name",
            c.company_name AS "Company Name",
            -- o.location AS Location,
            j.pay_range AS "Pay Range",
            j.source AS "Source",
            j.flexibility AS "Flexibility",
            j.link AS "Link",
            vs.semantic_score AS "Semantic Score",
            vs.title_similarity AS "Title Similarity",
            vs.skills_similarity AS "Skills Similarity",
            vs.responsibility_similarity AS "Responsibility Similarity",
            vs.adjusted_score AS "Adjusted Score",
            vs.rank AS Rank,
            vs.archetype_name AS "Best Archetype"
        FROM job j
        JOIN company c ON j.company_id = c.id
        JOIN office o ON j.office_id = o.id
        JOIN vector_scores vs ON j.id = vs.job_id
        LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id
        WHERE j.date_added >= CURRENT_DATE AND clr.job_id IS NULL AND vs.semantic_score >= 0.72
        ORDER BY "Adjusted Score" DESC;
        """
        cur.execute(user_query)
        rows = cur.fetchall()
        print(f"Found {len(rows)} matching jobs from user's SQL query.")
        
        # Check if there are any jobs in job table from today that have skip as NOT true and clr.job_id is NULL
        cur.execute("""
            SELECT COUNT(*) FROM job j
            LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id
            WHERE j.date_added >= CURRENT_DATE AND clr.job_id IS NULL AND j.skip IS NOT TRUE;
        """)
        unskipped_unprocessed = cur.fetchone()['count']
        print(f"Total unskipped and unprocessed jobs added today: {unskipped_unprocessed}")
        
finally:
    conn.close()
