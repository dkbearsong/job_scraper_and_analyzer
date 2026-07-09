import os
import yaml
from dotenv import load_dotenv
from app.postgres_mgr import PostgresManager

load_dotenv()
user_config = {}
prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
if os.path.exists(prefs_path):
    with open(prefs_path, 'r') as f:
        user_config = yaml.safe_load(f) or {}

host = user_config.get("db_host", os.getenv("DB_HOST", "localhost"))
port = int(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
user = user_config.get("db_user", os.getenv("DB_USER", "postgres"))
password = user_config.get("db_password", os.getenv("DB_PASSWORD", ""))
db_name = user_config.get("db_name", os.getenv("DB_NAME", "web_scraper_db"))

pg = PostgresManager(host, port, user, password)
pg.connect(db_name)

print("=== Checking Latest Archetype Embeddings ===")
rows = pg.execute_sql("SELECT archetype_name, array_length(title_embedding, 1) as title_dim, array_length(skills_embedding, 1) as skills_dim, array_length(responsibilities_embedding, 1) as resp_dim FROM archetype_embeddings", dbname=db_name, fetch=True)
for r in rows:
    print(r)

print("\n=== Checking Latest Vector Scores ===")
scores = pg.execute_sql("SELECT id, job_id, archetype_name, semantic_score, title_similarity, skills_similarity, responsibility_similarity FROM vector_scores ORDER BY id DESC LIMIT 10", dbname=db_name, fetch=True)
for s in scores:
    print(s)

pg.close()
