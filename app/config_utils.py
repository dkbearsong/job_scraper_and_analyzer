import os
import yaml
from docx import Document
from app.logger import error_logger_continue

def _load_user_config() -> dict:
    """Load the user_preferences.yaml file and return its contents."""
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    config = {}
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    return config


def load_resume_as_text(type):
    path = os.getenv(type)
    if not path:
        error_logger_continue(f"Environment variable '{type}' is not set.")
        return ""
    if path.lower().endswith('.txt'):
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    doc = Document(path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text.strip())
    return "\n".join(full_text)


