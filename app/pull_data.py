import aiohttp
import json
import asyncio
import os
import csv
import time
import yaml
from dotenv import load_dotenv
from random import random
from datetime import date
from collections import defaultdict

# Modules
from app.postgres_mgr import PostgresManager
from app.make_db import make_db

load_dotenv()

ws_micro_host = os.getenv("SCRAPER_HOST")
ws_micro_port = os.getenv("SCRAPER_PORT")

class DataPuller:
    def __init__(self, host: str = "localhost", port: str = "5432", user: str = "postgres", password: str = "", dbname: str = "postgres"):
        self.host = host
        self.port = port
        self.conn = PostgresManager(host, int(port), user, password, dbname=dbname)
        self.dbname = dbname
        if self.conn.database_exists(self.dbname) == False:
            make_db()
            self.conn.connect(self.dbname)

    async def pull_data(self, source: str, payload: dict = {}) -> dict:
        rand_time = 3 * random()
        time.sleep(payload.get('seconds', rand_time))
        url = f"{ws_micro_host}:{ws_micro_port}/{source}/scrape"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=payload) as response:
                data = await response.json()
                return data

    # Load list of sites to scrape

    def load_sites_list(self,sites_file: str) -> dict:
        sites = defaultdict(list)
        with open(sites_file, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                for key, value in row.items():
                    sites[key].append(value)
        return sites

    # Pull request payload from .site_strategies/{site}.json

    def load_site_strategies(self, path: str) -> list:
        # print(path)
        with open(path, 'r') as f:
            site_strategies = json.load(f)
        return site_strategies

    # Load pulled data into database

    async def scrape_data(self, payload:dict, api_method:str="extract"):
        url = f"{ws_micro_host}:{ws_micro_port}/{api_method}"
        # Allow overriding the microservice request timeout via env var
        try:
            timeout_seconds = int(os.getenv("MICROSERVICE_TIMEOUT", "120"))
        except (TypeError, ValueError):
            timeout_seconds = 120
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                # print(f"url: {url} | payload: {payload}")
                async with session.post(url, json=payload) as response:
                    status = response.status
                    try:
                        resp_json = await response.json()
                    except Exception:
                        # fallback to text if JSON parsing fails
                        text = await response.text()
                        resp_json = {"raw": text}

                    # If the microservice already returns a dict with status_code, keep it,
                    # otherwise inject the HTTP status under 'status_code' and ensure 'data' exists.
                    if isinstance(resp_json, dict):
                        resp_json.setdefault('status_code', str(status))
                        if 'data' not in resp_json:
                            # If the body itself is the data (e.g., a list), wrap it
                            # but only if it's not empty dict
                            if resp_json and not any(k in resp_json for k in ('data', 'error', 'status_code')):
                                resp_json = {'data': resp_json, 'status_code': status}
                    else:
                        resp_json = {'data': resp_json, 'status_code': status}

                    return resp_json
        except asyncio.CancelledError:
            # propagate cancellation
            raise
        except asyncio.TimeoutError as e:
            return {'status_code': 408, 'data': [], 'error': 'Request timed out', 'exception': str(e)}
        except aiohttp.ClientError as e:
            return {'status_code': 503, 'data': [], 'error': 'Client error', 'exception': str(e)}
        except Exception as e:
            return {'status_code': 500, 'data': [], 'error': 'Unexpected error', 'exception': str(e)}

    def load_scraped_data_to_db(self, data: list):
        '''
        Load scraped data into PostgreSQL database
        input format:
            data = [
                {
                    "job_name": str,
                    "company": str,
                    "location": str,
                    "link": str,
                    "pay": str (optional),
                    "description": str (optional),
                    "city": str (optional),
                    "state": str (optional)
                }
            ]
        '''
        
        for item in data:
            # Get or insert company
            companies = self.conn.search("company", {"company_name": item['company']})
            if companies:
                company_id = companies[0][0]
            else:
                result = self.conn.insert("company", {"company_name": item.get('company'), "company_url": item.get('company_url')}, returning=["id"])
                # Ensure insert returned an id
                if result and isinstance(result, (list, tuple)) and len(result) > 0 and len(result[0]) > 0:
                    company_id = result[0][0]
                else:
                    # Skip this item if we couldn't obtain a company id
                    continue

            # Get or insert office
            if item.get('location') == None: # Presuming that if a company does not list locations on their career page they are remote focused
                item['location'] = 'Remote'
            offices = self.conn.search("office", {"company_id": company_id, "location": item['location']})
            if offices:
                office_id = offices[0][0]
            else:
                office_data = {"company_id": company_id, "location": item.get('location')}
                if item.get('city'):
                    office_data['city'] = item['city']
                if item.get('state'):
                    office_data['state'] = item['state']
                result = self.conn.insert("office", office_data, returning=["id"])
                if result and isinstance(result, (list, tuple)) and len(result) > 0 and len(result[0]) > 0:
                    office_id = result[0][0]
                else:
                    continue

            # Check for duplicate job within 3 months
            query = """
            SELECT * FROM job 
            WHERE job_name = %s AND company_id = %s AND office_id = %s 
            AND date_added >= CURRENT_DATE - INTERVAL '3 months'
            """
            rows = self.conn.execute_sql(query, (item['title'], company_id, office_id), fetch=True)
            if rows:
                # print(f"Rows found: {rows}")
                continue  # Skip duplicate
            
            # print(f"item source: {item['source']}")

            # Insert job
            insert_data = {
                "job_name": item['title'],
                "company_id": company_id,
                "office_id": office_id,
                "link": item['url'],
                "date_added": date.today(),
                "flexibility": item['flexibility'],
                "source":item['source']
            }
            
            if item.get('description'):
                insert_data['job_summary'] = item['description']

            if 'pay' in item:
                insert_data['pay'] = item['pay']

            self.conn.insert("job", insert_data)

        return

    def pull_data_db(self, query: str): # Need to modify this so it returns as a dict. Check 
        '''
        Allows running of database queries, specifically select statements to pull data

        input format:
            query = str (SQL select statement)
        returns: list of dicts
        '''
        rows = self.conn.execute_sql(query, fetch=True)
        # print(rows)
        return rows

    def insert_data_db(self, query: str, params: tuple = ()):
        self.conn.execute_sql(query, params)
        return

    def commit_data_db(self):
        # Note: PostgresManager commits automatically in execute_sql, but keeping for compatibility
        pass
        return

    def close_connection(self):
        self.conn.close()
        return
    
    def bulk_update_skip_status(self, job_ids: list):
        """Updates the 'is_skipped' column for a list of job IDs."""
        if not job_ids:
            return
        up = {
            "table": "job",
            "set_values": {'skip' : 'True'},
            "where": {"id": job_ids}
        }
        self.conn.update(up['table'], up['set_values'], up['where'], self.dbname)

        return
    
    def update_job_metadata(self, updates: list):
        """
        Updates job records with extracted metadata.
        
        Args:
            updates: List of dictionaries containing job_id and metadata fields to update
        """
        for update_data in updates:
            # Only update fields if they don't already exist (not None/empty)
            set_values = {}
            
            # Check pay_range
            if update_data.get('pay_range') is not None and update_data['pay_range'] != "":
                set_values["pay_range"] = update_data.get('pay_range')
                
            # Check seniority
            if update_data.get('seniority') is not None and update_data['seniority'] != "":
                set_values["seniority"] = update_data.get('seniority')
                
            # Check work_type
            if update_data.get('work_type') is not None and update_data['work_type'] != "":
                set_values["work_type"] = update_data.get('work_type')
                
            # Check timezone
            if update_data.get('timezone') is not None and update_data['timezone'] != "":
                set_values["timezone"] = update_data.get('timezone')
            
            # Only perform update if there are fields to set
            if set_values:
                where_clause = {"id": update_data.get('id')}
                
                # Use the existing update method from PostgresManager
                self.conn.update("job", set_values, where_clause, self.dbname)

    def save_job_embeddings(self, embedding_updates: list):
        """
        Saves generated job embeddings to the 'job_embeddings' table.
        
        Args:
            embedding_updates: List of dictionaries containing job_id and embedding data.
        """
        for data in embedding_updates:
            job_id = data.get("job_id")
            if not job_id:
                continue

            # Map data to table columns as specified in the Stage 4 requirements
            insert_data = {
                "job_id": job_id,
                "title_embedding": data.get("title_embedding"),
                "skills_embedding": data.get("skills_embedding"),
                "responsibilities_embedding": data.get("responsibilities_embedding"),
                "description_embedding": data.get("description_embedding")
            }

            # Filter out keys with None values to ensure clean insertion
            insert_data = {k: v for k, v in insert_data.items() if v is not None}

            # Insert into database using the established PostgresManager instance
            self.conn.insert("job_embeddings", insert_data, dbname=self.dbname)

    def bulk_create_table(self, create_sql: str, table_name: str = ""):
        """Creates a table if it doesn't exist using the established connection."""
        try:
            self.conn.execute_sql(create_sql, dbname=self.dbname)
        except Exception as e:
            print(f"Warning: Could not create table {table_name}: {e}")

    def save_vector_scores(self, filtered_job_pool: list):
        """Persists vector scoring results to the vector_scores table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS vector_scores (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL,
            archetype_name VARCHAR(255) NOT NULL,
            semantic_score REAL,
            title_similarity REAL,
            skills_similarity REAL,
            responsibility_similarity REAL,
            adjusted_score REAL,
            rank INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.bulk_create_table(create_table_sql, "vector_scores")
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_vs_job_id ON vector_scores(job_id)
        """
        self.bulk_create_table(create_index_sql, "idx_vs_job_id")

        for rank, job in enumerate(filtered_job_pool, start=1):
            insert_sql = """
                INSERT INTO vector_scores (job_id, archetype_name, semantic_score,
                                           title_similarity, skills_similarity,
                                           responsibility_similarity, adjusted_score, rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.conn.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                job.get('best_archetype', ''),
                job.get('semantic_score', 0),
                job.get('title_similarity', 0),
                job.get('skills_similarity', 0),
                job.get('responsibility_similarity', 0),
                job.get('adjusted_score', 0),
                rank
            ), dbname=self.dbname)

    def save_cheap_llm_results(self, shortlisted_jobs: list):
        """Persists Stage 6 cheap LLM results to the cheap_llm_results table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS cheap_llm_results (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL,
            fit_score INTEGER,
            decision VARCHAR(20),
            strengths JSONB,
            concerns JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.bulk_create_table(create_table_sql, "cheap_llm_results")

        for job in shortlisted_jobs:
            cheap_result = job.get('cheap_llm_result', {})
            insert_sql = """
                INSERT INTO cheap_llm_results (job_id, fit_score, decision, strengths, concerns)
                VALUES (%s, %s, %s, %s, %s)
            """
            self.conn.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                cheap_result.get('fit_score', 50),
                cheap_result.get('decision', 'maybe'),
                json.dumps(cheap_result.get('strengths', [])),
                json.dumps(cheap_result.get('concerns', []))
            ), dbname=self.dbname)

    def save_strong_llm_results(self, deeply_analyzed_jobs: list):
        """Persists Stage 7 strong LLM results to the strong_llm_results table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS strong_llm_results (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL,
            final_score INTEGER,
            priority VARCHAR(20),
            apply_recommendation VARCHAR(20),
            red_flags JSONB,
            tailoring_notes JSONB,
            recruiter_bait_likelihood VARCHAR(20),
            detailed_fit_analysis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.bulk_create_table(create_table_sql, "strong_llm_results")

        for job in deeply_analyzed_jobs:
            strong_result = job.get('strong_llm_result', {})
            insert_sql = """
                INSERT INTO strong_llm_results (job_id, final_score, priority,
                                               apply_recommendation, red_flags,
                                               tailoring_notes, recruiter_bait_likelihood,
                                               detailed_fit_analysis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            self.conn.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                strong_result.get('final_score', 50),
                strong_result.get('priority', 'medium'),
                strong_result.get('apply_recommendation', 'maybe'),
                json.dumps(strong_result.get('red_flags', [])),
                json.dumps(strong_result.get('tailoring_notes', [])),
                strong_result.get('recruiter_bait_likelihood', 'medium'),
                strong_result.get('detailed_fit_analysis', '')
            ), dbname=self.dbname)

    def save_final_queue(self, final_queue: list):
        """Persists Stage 8 final application queue to the final_application_queue table."""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS final_application_queue (
            id SERIAL PRIMARY KEY,
            job_id INTEGER NOT NULL UNIQUE,
            final_score REAL,
            priority VARCHAR(20),
            apply_recommendation VARCHAR(20),
            queue_position INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
        self.bulk_create_table(create_table_sql, "final_application_queue")

        for position, job in enumerate(final_queue, start=1):
            insert_sql = """
                INSERT INTO final_application_queue (job_id, final_score, priority,
                                                     apply_recommendation, queue_position)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (job_id) DO UPDATE SET
                    final_score = EXCLUDED.final_score,
                    priority = EXCLUDED.priority,
                    apply_recommendation = EXCLUDED.apply_recommendation,
                    queue_position = EXCLUDED.queue_position,
                    created_at = CURRENT_TIMESTAMP
            """
            self.conn.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                job.get('final_score', 0),
                job.get('priority', 'medium'),
                job.get('apply_recommendation', 'maybe'),
                position
            ), dbname=self.dbname)

    def get_archetype_embeddings(self, name: str):
        """Retrieves cached archetype embeddings from the database."""
        query = "SELECT title_embedding, skills_embedding, responsibilities_embedding, archetype_type, metadata FROM archetype_embeddings WHERE archetype_name = %s"
        rows = self.conn.execute_sql(query, (name,), fetch=True)
        if rows:
            # Convert row back to a clean dictionary
            res = dict(rows[0])
            return res
        return None

    def save_archetype_embeddings(self, archetype_data: dict):
        """Caches newly generated archetype embeddings."""
        self.conn.insert("archetype_embeddings", archetype_data, dbname=self.dbname)


def main():
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    # Load from user_preferences.yaml with .env fallback
    user_config = {}
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
    dp = DataPuller(
        dbname = user_config.get("db_name", os.getenv("DB_NAME", "")),
        user = user_config.get("db_user", os.getenv("DB_USER", "")),
        password = user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
        host = user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
        port = str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
    )
    query = f'''
        SELECT DISTINCT c.company_name AS company, c.company_url, j.source
        FROM company c
        JOIN job j ON j.company_id = c.id
        WHERE title_rating >= 80 AND skip IS NOT True AND job_summary IS NULL;
    '''
    data = dp.pull_data_db(query)
    # data is expected to be a list of dicts; convert to a mapping by company if desired,
    # otherwise just print the list directly.
    if isinstance(data, list):
        try:
            data2 = {row.get('company'): row for row in data}
        except Exception:
            data2 = data
    else:
        data2 = data
    print(f"Data: {data2}")

if __name__ == "__main__":
    main()
