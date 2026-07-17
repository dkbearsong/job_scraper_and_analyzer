# Libraries
import os
import asyncio
from dotenv import load_dotenv
import json
from docx import Document
import re
import logging
import time
import numpy as np
import yaml
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime

# Modules
from app.pull_data import DataPuller
from app.text_engine import TextProcessor
from app.ai_engine import AIEngine
from app.archetype_engine import ArchetypeManager, Archetype
from app.llm_classifier import (
    CheapLLMClassifier,
    StrongLLMReranker,
    process_stage_6,
    process_stage_7,
    process_stage_8
)

# Scraper Adapter System
from app.scrapers.adapter_loader import AdapterLoader
from app.scrapers.validator import ScrapedDataValidator

# Fallback Scraping Instructions (legacy Part A + Part B path)
from app.fallback_scraping_instructions import (
    _pipeline_stage_scrape_legacy,
)

# LLM Usage Tracking
from app.llm_usage_tracker import usage_tracker


############################# Global Variables and Configs ############################
load_dotenv()

# Archetype Definitions
with open(os.getenv("ARCHETYPES_CONFIG", ""), 'r') as file:
    ARCHETYPES_CONFIG = json.load(file)

# Load user_preferences.yaml for config values
def _load_user_config() -> dict:
    """Load the user_preferences.yaml file and return its contents."""
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    config = {}
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    return config

############################## Error Handlers #########################################

def error_logger_crash(error_msg):
    print(error_msg)
    logging.error(error_msg)
    raise ValueError(error_msg)

def error_logger_continue(error_msg):
    print(error_msg)
    logging.error(error_msg)
    return


# ── Pipeline Stats Logging ──

def _setup_pipeline_stats_logger() -> logging.Logger:
    """Configure and return a logger that writes pipeline stats to ``logs/pipeline_stats.log``."""
    logger = logging.getLogger("pipeline_stats")
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers on re-initialization
    if logger.handlers:
        return logger

    os.makedirs("logs", exist_ok=True)
    handler = logging.FileHandler(os.path.join("logs", "pipeline_stats.log"), mode="a", encoding="utf-8")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Prevent propagating to the root logger (which goes to app_error.log)
    logger.propagate = False
    return logger


PIPELINE_STATS_LOGGER = _setup_pipeline_stats_logger()


def _log_pipeline_stats(
    stage_label: str,
    job_count: int,
    *,
    skipped_count: int = 0,
    source_hint: str = "",
    extra: dict | None = None,
) -> None:
    """
    Write a structured stats line to ``pipeline_stats.log``.

    Format:
        STAGE <label> | jobs=<N> | skipped=<M> | source=<hint> | extra=<json>

    Args:
        stage_label: Human-readable stage name (e.g. "1: Scrape").
        job_count: Number of jobs entering/exiting this stage.
        skipped_count: Number of jobs skipped (if applicable).
        source_hint: Short description of the data source (e.g. "hiring_cafe").
        extra: Optional dict of additional key=value pairs to log.
    """
    parts = [f"STAGE {stage_label}", f"jobs={job_count}"]
    if skipped_count:
        parts.append(f"skipped={skipped_count}")
    if source_hint:
        parts.append(f"source={source_hint}")
    if extra:
        for k, v in extra.items():
            parts.append(f"{k}={v}")
    message = " | ".join(parts)
    PIPELINE_STATS_LOGGER.info(message)

############################## Helper Functions #######################################
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

import math
import requests

GEO_CACHE_FILE = "app/geocoding_cache.json"

SEED_COORDINATES = {
    "austin, tx": (30.2672, -97.7431),
    "austin, texas": (30.2672, -97.7431),
    "austin": (30.2672, -97.7431),
    "austin area": (30.2672, -97.7431),
    "seattle, wa": (47.6062, -122.3321),
    "seattle, washington": (47.6062, -122.3321),
    "seattle": (47.6062, -122.3321),
    "seattle area": (47.6062, -122.3321),
    "boston, ma": (42.3601, -71.0589),
    "boston, massachusetts": (42.3601, -71.0589),
    "boston": (42.3601, -71.0589),
    "boston area": (42.3601, -71.0589),
    "san francisco, ca": (37.7749, -122.4194),
    "san francisco, california": (37.7749, -122.4194),
    "san francisco": (37.7749, -122.4194),
    "san francisco bay area": (37.7749, -122.4194),
    "sf bay area": (37.7749, -122.4194),
    "salt lake city, ut": (40.7608, -111.8910),
    "salt lake city, utah": (40.7608, -111.8910),
    "salt lake city": (40.7608, -111.8910),
    "salt lake county": (40.7608, -111.8910),
    "salt lake county, utah": (40.7608, -111.8910),
    "new york, ny": (40.7128, -74.0060),
    "new york, new york": (40.7128, -74.0060),
    "new york": (40.7128, -74.0060),
    "los angeles, ca": (34.0522, -118.2437),
    "los angeles, california": (34.0522, -118.2437),
    "los angeles": (34.0522, -118.2437),
    "chicago, il": (41.8781, -87.6298),
    "chicago, illinois": (41.8781, -87.6298),
    "chicago": (41.8781, -87.6298),
    "denver, co": (39.7392, -104.9903),
    "denver, colorado": (39.7392, -104.9903),
    "denver": (39.7392, -104.9903),
    "atlanta, ga": (33.7490, -84.3880),
    "atlanta, georgia": (33.7490, -84.3880),
    "atlanta": (33.7490, -84.3880),
    "remote": None,
    "na": None,
    "n/a": None
}

def load_geocoding_cache():
    if os.path.exists(GEO_CACHE_FILE):
        try:
            with open(GEO_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_geocoding_cache(cache):
    try:
        os.makedirs(os.path.dirname(GEO_CACHE_FILE), exist_ok=True)
        with open(GEO_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass

def geocode(location_str: str) -> Optional[Tuple[float, float]]:
    if not location_str:
        return None
    loc_clean = location_str.strip().lower()
    
    # Check seed coordinates
    if loc_clean in SEED_COORDINATES:
        return SEED_COORDINATES[loc_clean]
        
    cache = load_geocoding_cache()
    if loc_clean in cache:
        cached = cache[loc_clean]
        if cached:
            return cached[0], cached[1]
        return None

    # Call Nominatim OSM API
    headers = {"User-Agent": "JobScraperAnalyzer/1.0 (dbearsong@example.com)"}
    url = "https://nominatim.openstreetmap.org/search"
    params = {"q": location_str, "format": "json", "limit": 1}
    try:
        # Respect OpenStreetMap Nominatim request limit
        time.sleep(1.0)
        resp = requests.get(url, params=params, headers=headers, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            if data:
                lat = float(data[0]["lat"])
                lon = float(data[0]["lon"])
                cache[loc_clean] = [lat, lon]
                save_geocoding_cache(cache)
                return lat, lon
    except Exception as e:
        print(f"Geocoding error for {location_str}: {e}")
    
    # Cache None to avoid repeated slow failures
    cache[loc_clean] = None
    save_geocoding_cache(cache)
    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 3958.8  # Earth radius in miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def is_missing_value(val) -> bool:
    if val is None:
        return True
    if isinstance(val, (list, tuple, set)):
        if not val:
            return True
        return all(is_missing_value(x) for x in val)
    if isinstance(val, dict):
        if not val:
            return True
        return all(is_missing_value(x) for x in val.values())
    if isinstance(val, str):
        v = val.strip().lower()
        return v in ("", "na", "n/a", "not specified", "none", "unknown", "n/a (us or canada only)")
    return False

def term_matches(pref: str, job_val: str) -> bool:
    pref = pref.strip().lower()
    job_val = job_val.strip().lower()
    if not pref or not job_val:
        return False
        
    # Check for quotation marks
    is_quoted = (pref.startswith('"') and pref.endswith('"')) or (pref.startswith("'") and pref.endswith("'"))
    if is_quoted:
        clean_pref = pref[1:-1].strip()
        return clean_pref in job_val
    else:
        # Partial match
        if pref in job_val:
            return True
        # Word matches
        pref_words = [w for w in re.split(r'[^a-zA-Z0-9\-\']', pref) if w]
        job_words = [w for w in re.split(r'[^a-zA-Z0-9\-\']', job_val) if w]
        for pw in pref_words:
            if pw in job_words or any(pw in jw for jw in job_words):
                return True
        return False

def extract_job_locations(job: dict) -> List[str]:
    locs = []
    features = job.get('features', {})
    
    # Try city, state first
    city = features.get('city')
    state = features.get('state')
    if city and not is_missing_value(city):
        if state and not is_missing_value(state):
            locs.append(f"{city}, {state}")
        else:
            locs.append(city)
            
    # Try location
    location = features.get('location')
    if location and not is_missing_value(location):
        for part in re.split(r'[;|/|\|]', location):
            part_clean = part.strip()
            if part_clean and not is_missing_value(part_clean):
                locs.append(part_clean)
                
    # Deduplicate
    seen = set()
    deduped = []
    for l in locs:
        l_lower = l.lower()
        if l_lower not in seen:
            seen.add(l_lower)
            deduped.append(l)
    return deduped

def is_country_location(location_str: str) -> bool:
    if not location_str:
        return False
    loc_clean = location_str.strip().lower()
    country_names = {
        "united states", "usa", "u.s.a.", "united states of america", "us", "u.s.",
        "canada", "united kingdom", "uk", "u.k.", "great britain", "gb", "england",
        "germany", "france", "australia", "india", "singapore", "japan", "poland",
        "ireland", "spain", "italy", "brazil", "mexico", "switzerland", "netherlands",
        "sweden", "finland", "south korea", "philippines", "israel"
    }
    return loc_clean in country_names

def get_country_of_location(location_str: str) -> Optional[str]:
    """
    Tries to determine the country of a given location string without using geocoding.
    Returns the lowercase standard country name or None.
    """
    if not location_str:
        return None
    loc_clean = location_str.strip().lower()
    
    country_mappings = {
        "united states": "united states",
        "united states of america": "united states",
        "usa": "united states",
        "u.s.a.": "united states",
        "us": "united states",
        "u.s.": "united states",
        
        "canada": "canada",
        
        "united kingdom": "united kingdom",
        "uk": "united kingdom",
        "u.k.": "united kingdom",
        "great britain": "united kingdom",
        "gb": "united kingdom",
        "england": "united kingdom",
        "scotland": "united kingdom",
        "wales": "united kingdom",
        "northern ireland": "united kingdom",
        
        "india": "india",
        
        "germany": "germany",
        "deutschland": "germany",
        
        "france": "france",
        "australia": "australia",
        "singapore": "singapore",
        "japan": "japan",
        "ireland": "ireland",
    }
    
    if loc_clean in country_mappings:
        return country_mappings[loc_clean]
        
    words = re.findall(r'[a-zA-Z0-9]+', loc_clean)
    
    for word in words:
        if word in country_mappings:
            if word == "in" and len(words) > 1:
                if words[-1] == "in":
                    return "india"
                continue
            return country_mappings[word]
            
    # Check US states
    us_states = {
        "al", "ak", "az", "ar", "ca", "co", "ct", "de", "fl", "ga", "hi", "id", "il", "in", "ia", "ks", "ky", "la", "me", "md",
        "ma", "mi", "mn", "ms", "mo", "mt", "ne", "nv", "nh", "nj", "nm", "ny", "nc", "nd", "oh", "ok", "or", "pa", "ri", "sc",
        "sd", "tn", "tx", "ut", "vt", "va", "wa", "wv", "wi", "wy", "pr", "vi", "gu", "mp", "as",
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado", "connecticut", "delaware", "florida", "georgia",
        "hawaii", "idaho", "illinois", "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine", "maryland",
        "massachusetts", "michigan", "minnesota", "mississippi", "missouri", "montana", "nebraska", "nevada", "new hampshire",
        "new jersey", "new mexico", "new york", "north carolina", "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
        "rhode island", "south carolina", "south dakota", "tennessee", "texas", "utah", "vermont", "virginia", "washington",
        "west virginia", "wisconsin", "wyoming"
    }
    
    # Check Canada provinces
    ca_provinces = {
        "ab", "bc", "mb", "nb", "nl", "ns", "nt", "nu", "on", "pe", "qc", "sk", "yt",
        "alberta", "british columbia", "manitoba", "new brunswick", "newfoundland", "nova scotia", "ontario", "prince edward island", "quebec", "saskatchewan"
    }
    
    for word in words:
        if word in us_states:
            return "united states"
        if word in ca_provinces:
            return "canada"
            
    for name, canonical in country_mappings.items():
        if len(name.split()) > 1 and name in loc_clean:
            return canonical
            
    return None

def check_location_proximity(job: dict, user_preferences: dict) -> bool:
    """
    Returns True if the job should be SKIPPED because it is outside the target city range.
    Returns False if it is within range, or if location data is missing/invalid.
    """
    target_cities = user_preferences.get('target_cities', [])
    if not target_cities:
        return False
        
    # Check if work_type is remote
    work_type = job.get('features', {}).get('work_type', '')
    if isinstance(work_type, str) and 'remote' in work_type.lower():
        return False
        
    job_locations = extract_job_locations(job)
    if not job_locations:
        return False
        
    # Check if any location contains "remote" or is a country name
    for loc in job_locations:
        if 'remote' in loc.lower():
            return False
        if is_country_location(loc):
            return False
            
    max_range = user_preferences.get('target_city_range', 25)
    
    geocoded_any = False
    for target in target_cities:
        target_clean = target.strip().lower()
        
        # 1. Country target checks: if target is a country, match job locations in the same country
        if is_country_location(target):
            target_country = get_country_of_location(target)
            if target_country:
                for loc in job_locations:
                    loc_country = get_country_of_location(loc)
                    if loc_country == target_country:
                        return False  # Match! Do not skip this job.
        
        # 2. Exact/substring match first to avoid slow geocoding
        for loc in job_locations:
            loc_clean = loc.strip().lower()
            if target_clean in loc_clean or loc_clean in target_clean:
                return False
                
        target_coords = geocode(target)
        if not target_coords:
            continue
            
        for loc in job_locations:
            loc_coords = geocode(loc)
            if loc_coords:
                geocoded_any = True
                dist = haversine_distance(target_coords[0], target_coords[1], loc_coords[0], loc_coords[1])
                if dist <= max_range:
                    return False
                    
    if geocoded_any:
        return True
    return False

def apply_rule_filters(job: dict, user_preferences: dict) -> bool:
    """
    Evaluates a job against hard constraints.
    Returns True if the job should be SKIPPED, False if it passes.
    """
    features = job.get('features', {})

    # 1. Work Type Filter
    job_work_type = features.get('work_type')
    user_work_types = user_preferences.get('work_types', [])
    if user_work_types and not is_missing_value(job_work_type):
        matched = False
        for pref in user_work_types:
            if term_matches(pref, job_work_type):
                matched = True
                break
        if not matched:
            return True

    # 2. Seniority Filter
    job_seniority = features.get('seniority')
    user_seniority_levels = user_preferences.get('seniority_levels', [])
    if user_seniority_levels and not is_missing_value(job_seniority):
        matched = False
        for pref in user_seniority_levels:
            if term_matches(pref, job_seniority):
                matched = True
                break
        if not matched:
            return True

    # 3. Pay Filter
    job_pay = features.get('pay', "")
    target_pay_range = user_preferences.get('pay_range', '')
    if not is_missing_value(job_pay):
        if target_pay_range and job_pay:
            try:
                if filter_pay(target_pay_range, job_pay):
                    return True                            
            except (ValueError, IndexError):
                error_logger_continue(f"Error in parsing pay range for job {job['metadata']['job_id']}: {ValueError} at {IndexError}")
                pass

    # 4. Timezone Filter
    job_timezone = features.get('timezone')
    user_timezones = user_preferences.get('timezones', [])
    if user_timezones and not is_missing_value(job_timezone):
        matched = False
        for pref in user_timezones:
            if term_matches(pref, job_timezone):
                matched = True
                break
        if not matched:
            return True

    # 5. Cities/Proximity Filter
    if check_location_proximity(job, user_preferences):
        return True

    return False

def filter_pay(target_pay_range, job_pay):
    # Parse user's target pay range
    user_min_match = re.search(r'(\d+)(?:k|K)?', target_pay_range)
    user_max_match = re.search(r'-(\d+)(?:k|K)?', target_pay_range)
    if not (user_min_match and user_max_match):
        return None 

    user_min = convert_pay(user_min_match, target_pay_range)
    user_max = convert_pay(user_max_match, target_pay_range)

    job_min, job_max = None, None
    if isinstance(job_pay, str):
        # Remove commas and strip decimal digits (e.g. $111,259.26 -> $111259)
        job_pay_cleaned = re.sub(r'\.\d+', '', job_pay.replace(',', ''))
        matches = re.findall(r'(\d+)(?:k|K)?', job_pay_cleaned)
        if matches:
            job_min = int(matches[0]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[0])
        if len(matches) >= 2:
            job_max = int(matches[1]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[1])

    if job_max is not None:
        if job_min is None:
            job_min = job_max
        return True if job_max < user_min or job_min > user_max else False
    if job_min is not None:
        return False if user_min <= job_min <= user_max else True
    return None  # Fallback if no job values could be parsed

def convert_pay(match, context):
    val = int(match.group(1))
    return val * 1000 if 'k' in context.lower() else val

##################################### AI Functions #############################################################################

def _provider_is_lm_studio(ai: AIEngine, provider_name: str | None = None) -> bool:
    """
    Check whether the given (or default) provider is LM Studio.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional override; if None uses the engine's default.

    Returns:
        True if the resolved provider is LM Studio.
    """
    target = (provider_name or ai.default_provider_name).lower()
    return target == "lm_studio"


def _load_model_if_lm_studio(ai: AIEngine, provider_name: str | None = None,
                              stage_label: str = "", model_name: str | None = None) -> None:
    """
    If the resolved provider is LM Studio, send a model load request and wait for
    the model to become ready.  Does nothing for other providers.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional provider override.
        stage_label: Human-readable label for log messages.
        model_name: Optional model override.
    """
    if not _provider_is_lm_studio(ai, provider_name):
        return
    tag = f" [{stage_label}]" if stage_label else ""
    print(f"[Model Mgmt]{tag} Loading LM Studio model ...")
    ai.load_model(provider_name=provider_name, model_name=model_name)
    loaded = ai.wait_for_model_loaded(provider_name=provider_name, model_name=model_name)
    if not loaded:
        print(f"[Model Mgmt]{tag} WARNING: model may not be fully loaded yet.")


def _unload_model_if_lm_studio(ai: AIEngine, provider_name: str | None = None,
                                stage_label: str = "", model_name: str | None = None) -> None:
    """
    If the resolved provider is LM Studio, send a model unload request and wait for
    unloading to complete.  Does nothing for other providers.

    Args:
        ai: The AIEngine instance.
        provider_name: Optional provider override.
        stage_label: Human-readable label for log messages.
        model_name: Optional model override.
    """
    if not _provider_is_lm_studio(ai, provider_name):
        return
    tag = f" [{stage_label}]" if stage_label else ""
    print(f"[Model Mgmt]{tag} Unloading LM Studio model ...")
    ai.unload_model(provider_name=provider_name, model_name=model_name)
    unloaded = ai.wait_for_model_unloaded(provider_name=provider_name, model_name=model_name)
    if not unloaded:
        print(f"[Model Mgmt]{tag} WARNING: model may not be fully unloaded yet.")


def call_llm_for_extraction(ai: AIEngine, text: str, provider_name: str | None = None) -> dict:
    """Uses the provided AI engine to extract structured data."""
    if provider_name is None:
        return ai.extract(text)
    return ai.extract(text, provider_name=provider_name)

def generate_embeddings(ai: AIEngine, text: str, provider_name: str | None = None) -> list:
    """Uses the provided AI engine to generate a vector embedding."""
    if provider_name is None:
        return ai.embed(text)
    return ai.embed(text, provider_name=provider_name)

def extract_and_cache_profile(ai: AIEngine, source_path: str, raw_text: str, cache_path: str) -> dict:
    """
    Extracts structured profile data (skills, requirements, summary) from a resume
    or user profile using LLM extraction, with file-modification caching.
    """
    empty_result = {"skills": [], "requirements": [], "summary": ""}

    if not os.path.exists(source_path):
        print(f"Warning: profile source not found at {source_path}")
        return empty_result

    source_mtime = os.path.getmtime(source_path)

    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                cache = json.load(f)
            cache_time = cache.get('timestamp', 0)
            if cache_time >= source_mtime:
                print(f"Using cached profile from {cache_path}")
                return cache.get('data', empty_result)
        except Exception as e:
            print(f"Warning: failed to read profile cache {cache_path}: {e}")

    if not raw_text:
        print(f"Warning: empty text in {source_path}")
        return empty_result

    # Load extraction LLM from user_preferences.yaml
    _user_config = _load_user_config()
    _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))

    print(f"Extracting profile data from {source_path}...")
    ai_data = call_llm_for_extraction(ai, raw_text, provider_name=_extraction_llm)
    
    result = empty_result
    if isinstance(ai_data, dict):
        result = {
            "skills": ai_data.get('skills', []),
            "requirements": ai_data.get('requirements', []),
            "summary": ai_data.get('summary', ""),
        }
    elif isinstance(ai_data, str):
        try:
            parsed = json.loads(ai_data)
            if isinstance(parsed, dict):
                result = {
                    "skills": parsed.get('skills', []),
                    "requirements": parsed.get('requirements', []),
                    "summary": parsed.get('summary', ""),
                }
        except Exception:
            pass

    try:
        cache_dir = os.path.dirname(cache_path)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w') as f:
            json.dump({"timestamp": source_mtime, "data": result}, f, indent=2)
        print(f"Cached profile data to {cache_path}")
    except Exception as e:
        print(f"Warning: failed to write profile cache {cache_path}: {e}")

    return result


# =====================================================
# PIPELINE STAGE 0: SETUP AND PROFILE EXTRACTION
# =====================================================

async def pipeline_stage_setup(skip_db: bool = False, verbose: bool = False,
                               resume_text: str = "", profile_text: str = "") -> dict:
    """
    Stage 0: Load environment, resume, profile, create AI engine, extract skills/titles.
    
    Args:
        skip_db: If True, skip database initialization.
        verbose: If True, print detailed debug output.
        resume_text: If provided, use this instead of loading from file.
        profile_text: If provided, use this instead of loading from file.
    
    Returns:
        dict with keys: resume, user_profile, skills, job_titles, dp, tp, ai, user_preferences
    """
    print("=" * 50)
    print("PIPELINE STAGE 0: SETUP")
    print("=" * 50)

    # Load .env variables
    if resume_text:
        resume = resume_text
    else:
        resume = load_resume_as_text("RESUME")
    
    if profile_text:
        user_profile = profile_text
    else:
        user_profile = load_resume_as_text("PROFILE")

    # Load configuration from user_preferences.yaml
    user_config = _load_user_config()

    # Create Data Puller Object
    dp = DataPuller(
        dbname=user_config.get("db_name", os.getenv("DB_NAME", "")),
        user=user_config.get("db_user", os.getenv("DB_USER", "")),
        password=user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
        host=user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
        port=str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
    )

    # Get sites file from .env
    sites_file = os.getenv("JOB_SITES", "")
    if sites_file and os.path.exists(sites_file):
        sites = dp.load_sites_list(sites_file)
        print("Sites Retrieved")
    else:
        sites = {"name": [], "site": []}
        print("Warning: JOB_SITES file not found or not configured.")

    # Extract skills and job titles from profile
    tp = TextProcessor()
    ai = AIEngine(
        default_provider_name="lm_studio",
        extraction_model=user_config.get("extraction_model", os.getenv("EXTRACTION_MODEL", "local-model")),
        embeddings_model=user_config.get("embeddings_model", os.getenv("EMBEDDINGS_MODEL", "local-model"))
    )
    skills_raw = tp.get_section_content(user_profile, "Skills")
    titles_raw = tp.get_section_content(user_profile, "Job Titles")
    skills = tp.clean_list_from_text(skills_raw)
    job_titles = tp.clean_list_from_text(titles_raw)

    if verbose:
        print(f"--- Profile Extraction ---")
        print(f"Extracted {len(skills)} skills: {skills}")
        print(f"Extracted {len(job_titles)} job titles: {job_titles}")
        print(f"---------------------------\n")

    # Configure logging if not already configured
    root = logging.getLogger()
    if not root.handlers:
        os.makedirs("logs", exist_ok=True)
        error_handler = logging.FileHandler(os.path.join("logs", "app_error.log"), mode="a", encoding="utf-8")
        error_handler.setLevel(logging.ERROR)
        error_formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        error_handler.setFormatter(error_formatter)
        root.addHandler(error_handler)
        root.setLevel(logging.ERROR)

    # Load user preferences
    prefs_yaml_path = os.getenv("USER_PREFERENCES_YAML", "")
    if prefs_yaml_path and os.path.exists(prefs_yaml_path):
        with open(prefs_yaml_path, 'r') as f:
            user_preferences = yaml.safe_load(f)
    else:
        user_preferences = {}

    result = {
        "resume": resume,
        "user_profile": user_profile,
        "skills": skills,
        "job_titles": job_titles,
        "dp": dp,
        "tp": tp,
        "ai": ai,
        "user_preferences": user_preferences,
        "sites": sites,
        "sites_file": sites_file,
        "user_config": user_config,
        "db_config": {
            "dbname": user_config.get("db_name", os.getenv("DB_NAME", "")),
            "user": user_config.get("db_user", os.getenv("DB_USER", "")),
            "password": user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
            "host": user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
            "port": str(user_config.get("db_port", os.getenv("DB_PORT", "5432"))),
        },
    }

    print("Stage 0 complete: Setup and profile extraction done.")
    return result


# =====================================================
# PIPELINE STAGE 1: SCRAPING
# =====================================================

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
                "pay": item.get('pay', ''),
                "seniority": "NA",
                "work_type": item.get('flexibility', 'NA'),
                "timezone": "NA",
                "location": item.get('location', 'NA'),
                "city": item.get('city', 'NA'),
                "state": item.get('state', 'NA'),
            },
            "embeddings": {
                "description_vector": None,
                "skills_vector": None,
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


async def pipeline_stage_scrape(setup_data: dict, skip_db: bool = False,
                                verbose: bool = False,
                                max_pages: Optional[int] = None,
                                headless: bool = True,
                                debug_logging: bool = False,
                                skip_part_a: Optional[bool] = None,
                                db_limit: int = 50,
                                scrape_missing_24h: bool = False) -> List[Dict]:
    """
    Stage 1: Scrape jobs using the pluggable adapter system.

    Attempts to load scrapers from scrapers_config.yaml. If the config file
    is not found or contains no enabled adapters, falls back to the legacy
    hardcoded scraping path for backward compatibility.

    Args:
        setup_data: Output from pipeline_stage_setup.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
        max_pages: Override max_pages for all adapters (e.g. CLI --pages).
        headless: Override headless mode for all adapters (False = visible browser).
        debug_logging: If True, enable debug-level logging.
        skip_part_a: If True, skip Part A in legacy fallback path.
        db_limit: Max records to pull from DB in bulk-load fallback operations.
        scrape_missing_24h: If True, only re-scrape descriptions for recent DB jobs.

    Returns:
        List[Dict] of scraped jobs in the 'processed_job_pool' format.
    """
    dp = setup_data["dp"]
    tp = setup_data["tp"]
    user_preferences = setup_data.get("user_preferences", {})
    sites = setup_data.get("sites", {"name": [], "site": []})

    print("=" * 50)
    print("PIPELINE STAGE 1: SCRAPING")
    print("=" * 50)

    if scrape_missing_24h:
        print("[Stage 1] scrape_missing_24h is True. Running dedicated DB description re-scraping workflow...")
        if skip_db:
            print("[Stage 1] WARNING: skip_db is True, cannot run DB description re-scraping. Returning empty pool.")
            return []
        processed_job_pool = await scrape_missing_descriptions_24h(dp, verbose)
        print(f"Stage 1 complete: {len(processed_job_pool)} jobs in pool.")
        return processed_job_pool

    # Determine config path (check env var first, then default)
    scrapers_config_path = os.getenv("SCRAPERS_CONFIG", "scrapers_config.yaml")

    # ── Attempt adapter-based scraping ──
    all_scraped: List[Dict] = []
    used_adapters = False

    if os.path.exists(scrapers_config_path):
        try:
            loader = AdapterLoader(config_path=scrapers_config_path, verbose=verbose)
            loader.load_config()

            # Apply CLI overrides to adapter configs before loading
            if max_pages is not None or not headless or debug_logging:
                for entry in loader._config.get("scrapers", []):
                    cfg = entry.setdefault("config", {})
                    if max_pages is not None:
                        cfg["max_pages"] = max_pages
                    if not headless:
                        cfg["headless"] = False
                    if debug_logging:
                        cfg["debug"] = True

            adapter_list = loader.load_adapters()

            if adapter_list:
                used_adapters = True
                print(f"Loaded {len(adapter_list)} adapter(s) from {scrapers_config_path}")
                all_scraped = await loader.run_all(dp=dp)

                # Persist raw scraped data to DB
                if not skip_db and all_scraped:
                    dp.load_scraped_data_to_db(all_scraped)
            else:
                print(f"Config found at {scrapers_config_path} but no enabled adapters loaded.")
        except Exception as e:
            print(f"Adapter system failed: {e}. Falling back to legacy scraping.")

    # Read enable_fallback_part_b and skip_part_a from user preferences
    enable_part_b = user_preferences.get("enable_fallback_part_b", False)
    # CLI --skip-part-a overrides yaml value; if neither is set, default to False
    resolved_skip_part_a = (
        True
        if skip_part_a is True
        else bool(user_preferences.get("skip_part_a", False))
    )

    # ── Fallback to legacy path if adapters didn't run ──
    if not used_adapters:
        if not os.path.exists(scrapers_config_path):
            reason = f"no {scrapers_config_path} found"
        else:
            reason = f"no enabled adapters loaded from {scrapers_config_path} or adapter run failed"
        all_scraped = await _pipeline_stage_scrape_legacy(
            dp, user_preferences, sites, skip_db, verbose,
            enable_part_b=enable_part_b, skip_part_a=resolved_skip_part_a,
            db_limit=db_limit,
            reason=reason,
        )

    # ── Also run fallback if explicitly configured to do so ──
    if used_adapters and _check_run_fallback_flag(scrapers_config_path):
        print("run_fallback_after_adapters is True — also running legacy fallback scraping...")
        fallback_jobs = await _pipeline_stage_scrape_legacy(
            dp, user_preferences, sites, skip_db, verbose,
            enable_part_b=enable_part_b, skip_part_a=resolved_skip_part_a,
            db_limit=db_limit,
            reason="run_fallback_after_adapters is enabled",
        )
        if fallback_jobs:
            print(f"Merging {len(fallback_jobs)} fallback jobs with {len(all_scraped)} adapter jobs (deduplicating by link)...")
            merged = {}
            for job in all_scraped:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            for job in fallback_jobs:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            all_scraped = list(merged.values())
        else:
            print("Fallback scraping returned no jobs.")

    # ── Fallback: if no jobs have descriptions, load from DB jobs missing embeddings ──
    if all_scraped and not skip_db and not any(job.get("description") for job in all_scraped):
        print("--- Fallback: no scraped jobs have descriptions; loading from DB ---")
        from app.fallback_scraping_instructions import _load_jobs_without_embeddings
        db_jobs = await _load_jobs_without_embeddings(dp, limit=db_limit)
        if db_jobs:
            print(f"Loaded {len(db_jobs)} jobs from DB with descriptions but no embeddings (deduplicating by link)...")
            merged = {}
            for job in all_scraped:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            for job in db_jobs:
                link = job.get("link") or job.get("url") or ""
                if link:
                    merged[link] = job
            all_scraped = list(merged.values())

    # ── Sync in-memory descriptions from DB for any jobs missing them ──
    if all_scraped and not skip_db:
        print("Syncing in-memory descriptions from database for jobs missing them...")
        sync_count = 0
        for job in all_scraped:
            if not job.get("description"):
                job_id = job.get("id")
                link = job.get("link") or job.get("url")
                if job_id or link:
                    try:
                        if job_id:
                            query = "SELECT job_summary FROM job WHERE id = %s"
                            param = (job_id,)
                        else:
                            query = "SELECT job_summary FROM job WHERE link = %s"
                            param = (link,)
                        rows = dp.conn.execute_sql(query, param, fetch=True)
                        if rows:
                            row = rows[0]
                            summary = row.get("job_summary") if isinstance(row, dict) else row[0]
                            if summary:
                                job["description"] = summary
                                sync_count += 1
                    except Exception as e:
                        print(f"Warning: failed to fetch description from DB for link '{link}': {e}")
        if sync_count > 0:
            print(f"Synced {sync_count} job description(s) from database to memory.")

    # ── Post-process: remove jobs without descriptions and log them ──
    if all_scraped and not skip_db:
        print("--- Post-processing: removing jobs without descriptions ---")
        from app.fallback_scraping_instructions import _process_jobs_without_descriptions
        all_scraped = await _process_jobs_without_descriptions(all_scraped, dp)

    # ── Convert raw scraped data to processed_job_pool format ──
    processed_job_pool = _normalize_to_pool_format(all_scraped)

    # If no jobs were scraped (e.g., in test mode), use dummy data as fallback
    if not processed_job_pool:
        from tests.test_data import make_dummy_stage1_output
        print("No real jobs scraped. Using dummy fallback data.")
        processed_job_pool = make_dummy_stage1_output(15)

    print(f"Stage 1 complete: {len(processed_job_pool)} jobs in pool.")
    return processed_job_pool


# =====================================================
# PIPELINE STAGE 2: EMBEDDING GENERATION + LLM EXTRACTION
# =====================================================

async def pipeline_stage_embed_and_extract(jobs: List[Dict], ai_engine: Optional[AIEngine] = None,
                                           text_processor: Optional[TextProcessor] = None,
                                           dp: Optional[DataPuller] = None,
                                           skip_db: bool = False,
                                           verbose: bool = False) -> List[Dict]:
    """
    Stage 2: Run deterministic extraction, LLM extraction, and embedding generation on jobs.
    
    Args:
        jobs: List of job dicts from Stage 1 (or dummy data).
        ai_engine: AIEngine instance for LLM calls and embeddings.
        text_processor: TextProcessor instance for deterministic extraction.
        dp: DataPuller instance for DB operations.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
    
    Returns:
        List[Dict] of jobs with features enriched (skills, requirements, summary, embeddings).
    """
    print("=" * 50)
    print("PIPELINE STAGE 2: EMBEDDING GENERATION + LLM EXTRACTION")
    print("=" * 50)

    if ai_engine is None:
        ai_engine = AIEngine(default_provider_name="lm_studio")
    if text_processor is None:
        text_processor = TextProcessor()

    # Load LLM provider config from user_preferences.yaml
    _user_config = _load_user_config()
    _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))
    _embeddings_llm = _user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))

    # First pass: deterministic extraction for jobs that don't have features yet
    print(f"Starting deterministic extraction on {len(jobs)} jobs...")
    processed_job_pool = []
    for idx, raw_job in enumerate(jobs):
        if raw_job is None:
            continue
        
        # If the job is already in the processed format (from previous stage), use it
        if 'features' in raw_job and 'metadata' in raw_job:
            features = raw_job['features']
            desc = features.get('description', '')
            if desc:
                if is_missing_value(features.get('pay')):
                    features['pay'] = text_processor.extract_salary(desc)
                if is_missing_value(features.get('seniority')):
                    features['seniority'] = text_processor.detect_seniority(desc, title=features.get('title', ''))
                if is_missing_value(features.get('work_type')):
                    features['work_type'] = text_processor.detect_work_type(desc)
                if is_missing_value(features.get('timezone')):
                    features['timezone'] = text_processor.detect_timezone(desc)
            # Normalize work_type capitalization (Remote/Hybrid/Onsite)
            if features.get('work_type'):
                wt_lower = features['work_type'].lower()
                if 'remote' in wt_lower:
                    features['work_type'] = 'Remote'
                elif 'hybrid' in wt_lower:
                    features['work_type'] = 'Hybrid'
                elif 'onsite' in wt_lower or 'on-site' in wt_lower:
                    features['work_type'] = 'Onsite'
            processed_job_pool.append(raw_job)
            continue

        # Build from raw format
        job_id = raw_job.get('id') or raw_job.get('metadata', {}).get('job_id', idx + 1)
        description = raw_job.get('description', raw_job.get('job_summary', ''))
        title = raw_job.get('title', raw_job.get('job_name', ''))
        work_type = raw_job.get('flexibility', 'NA')
        pay = raw_job.get('pay', raw_job.get('pay_range', ''))
        company_name = raw_job.get('company', raw_job.get('company_name', raw_job.get('metadata', {}).get('company_name', 'Unknown'))        )
        link = raw_job.get('link', raw_job.get('url', raw_job.get('metadata', {}).get('link', '')))

        if not description or len(description) < 50:
            if verbose:
                error_logger_continue(f"Warning: insufficient description for job ID {job_id}")
            continue

        extracted_data = {
            "metadata": {
                "job_id": job_id,
                "source": "scraped",
                "company_name": company_name,
                "link": link,
            },
            "features": {
                "title": title,
                "description": description,
                "pay": pay if not is_missing_value(pay) else text_processor.extract_salary(description),
                "seniority": text_processor.detect_seniority(description, title=title),
                "work_type": work_type if not is_missing_value(work_type) else text_processor.detect_work_type(description),
                "timezone": text_processor.detect_timezone(description),
            },
            "embeddings": {
                "description_vector": None,
                "skills_vector": None,
            },
        }
        if extracted_data["features"].get('work_type'):
            wt_lower = extracted_data["features"]['work_type'].lower()
            if 'remote' in wt_lower:
                extracted_data["features"]['work_type'] = 'Remote'
            elif 'hybrid' in wt_lower:
                extracted_data["features"]['work_type'] = 'Hybrid'
            elif 'onsite' in wt_lower or 'on-site' in wt_lower:
                extracted_data["features"]['work_type'] = 'Onsite'
        processed_job_pool.append(extracted_data)

    print(f"Successfully extracted data for {len(processed_job_pool)} jobs.")

    # Second pass: LLM extraction
    is_lm_studio_extraction = _provider_is_lm_studio(ai_engine, _extraction_llm)
    if is_lm_studio_extraction:
        extraction_model = ai_engine.extraction_model
        print(f"[Model Mgmt] Loading LM Studio extraction model: {extraction_model} ...")
        ai_engine.load_model(provider_name=_extraction_llm, model_name=extraction_model)
        loaded = ai_engine.wait_for_model_loaded(provider_name=_extraction_llm, model_name=extraction_model)
        if not loaded:
            print("[Model Mgmt] WARNING: extraction model may not be fully loaded yet.")

    print(f"Starting AI/LLM Extraction Pass on {len(processed_job_pool)} jobs...")
    from app.ai_limiter import AILimiter, run_in_thread
    extraction_limiter = AILimiter("stage_2_extraction", _extraction_llm)

    async def process_extraction(index, job):
        if verbose:
            print(f"Extraction job {index + 1}/{len(processed_job_pool)}: {job['features']['title']}")

        if not job or 'features' not in job:
            error_logger_continue(f"Warning: job at index {index} has invalid structure")
            return

        description = job['features'].get('description', '')
        if not description:
            error_logger_continue(f"Warning: job at index {index} has no description")
            return

        # Extract skills, requirements and summary via LLM using AILimiter
        est_tokens = (len(description) // 4) + 1000
        async with extraction_limiter.semaphore:
            await extraction_limiter.wait(est_tokens)
            ai_data = await run_in_thread(call_llm_for_extraction, ai_engine, description, provider_name=_extraction_llm)

        skills = []
        requirements = []
        summary = ""
        llm_pay = None
        llm_work_type = None
        llm_seniority = None

        if isinstance(ai_data, dict):
            skills = ai_data.get('skills', [])
            requirements = ai_data.get('requirements', [])
            summary = ai_data.get('summary', "")
            llm_pay = ai_data.get('pay_range')
            llm_work_type = ai_data.get('work_type')
            llm_seniority = ai_data.get('seniority')
        elif isinstance(ai_data, str):
            try:
                parsed = json.loads(ai_data)
                if isinstance(parsed, dict):
                    skills = parsed.get('skills', [])
                    requirements = parsed.get('requirements', [])
                    summary = parsed.get('summary', "")
                    llm_pay = parsed.get('pay_range')
                    llm_work_type = parsed.get('work_type')
                    llm_seniority = parsed.get('seniority')
            except Exception:
                pass

        # Normalize extracted values to prevent type mismatches
        def to_string(v) -> str:
            if v is None:
                return ""
            if isinstance(v, list):
                return " ".join(to_string(x) for x in v if x is not None)
            if isinstance(v, dict):
                return " ".join(to_string(x) for x in v.values() if x is not None)
            return str(v).strip()

        if isinstance(skills, list):
            skills = [to_string(s) for s in skills if s]
        else:
            skills = [to_string(skills)] if skills else []

        if isinstance(requirements, list):
            requirements = [to_string(r) for r in requirements if r]
        else:
            requirements = [to_string(requirements)] if requirements else []

        summary = to_string(summary)

        if isinstance(llm_pay, list):
            llm_pay = " - ".join(to_string(p) for p in llm_pay if p)
        else:
            llm_pay = to_string(llm_pay)

        if isinstance(llm_work_type, list):
            llm_work_type = llm_work_type[0] if llm_work_type else ""
        llm_work_type = to_string(llm_work_type)

        if isinstance(llm_seniority, list):
            llm_seniority = " ".join(to_string(s) for s in llm_seniority if s)
        llm_seniority = to_string(llm_seniority)

        features = job['features']
        features['skills'] = skills or []
        features['requirements'] = requirements or []
        features['summary'] = summary or ""

        # Update pay and work_type using LLM fallback if deterministic extraction missed them
        if llm_pay and is_missing_value(features.get('pay')):
            features['pay'] = llm_pay
        if llm_work_type and is_missing_value(features.get('work_type')):
            # Normalize work_type from LLM to match capitalization
            wt = llm_work_type.strip().lower()
            if 'remote' in wt:
                features['work_type'] = 'Remote'
            elif 'hybrid' in wt:
                features['work_type'] = 'Hybrid'
            elif 'onsite' in wt or 'on-site' in wt:
                features['work_type'] = 'Onsite'
            else:
                features['work_type'] = 'Unknown'

        # Prefer LLM seniority if extracted successfully
        if llm_seniority and not is_missing_value(llm_seniority) and llm_seniority.lower() != "unknown":
            s_lower = llm_seniority.strip().lower()
            if "c-suite" in s_lower or "executive" in s_lower or "vp" in s_lower:
                features['seniority'] = 'C-Suite'
            elif "manager" in s_lower or "director" in s_lower or "head" in s_lower or "management" in s_lower:
                features['seniority'] = 'Management'
            elif "senior" in s_lower or "sr" in s_lower or "principal" in s_lower or "staff" in s_lower:
                features['seniority'] = 'Senior'
            elif "mid-level" in s_lower or "intermediate" in s_lower or "mid" in s_lower:
                features['seniority'] = 'Mid-Level'
            elif "junior" in s_lower or "jr" in s_lower or "entry" in s_lower or "associate" in s_lower or "intern" in s_lower:
                features['seniority'] = 'Junior'
            elif "lead" in s_lower:
                features['seniority'] = 'Lead'
            else:
                features['seniority'] = llm_seniority.strip()

    # Process all extraction tasks in parallel/sequential according to limiter
    extraction_tasks = [process_extraction(idx, job) for idx, job in enumerate(processed_job_pool)]
    await asyncio.gather(*extraction_tasks)

    if is_lm_studio_extraction:
        extraction_model = ai_engine.extraction_model
        print(f"[Model Mgmt] Unloading LM Studio extraction model: {extraction_model} ...")
        ai_engine.unload_model(provider_name=_extraction_llm, model_name=extraction_model)
        unloaded = ai_engine.wait_for_model_unloaded(provider_name=_extraction_llm, model_name=extraction_model)
        if not unloaded:
            print("[Model Mgmt] WARNING: extraction model may not be fully unloaded yet.")

    # Third pass: Embedding generation
    is_lm_studio_embeddings = _provider_is_lm_studio(ai_engine, _embeddings_llm)
    if is_lm_studio_embeddings:
        embeddings_model = ai_engine.embeddings_model
        print(f"[Model Mgmt] Loading LM Studio embeddings model: {embeddings_model} ...")
        ai_engine.load_model(provider_name=_embeddings_llm, model_name=embeddings_model)
        loaded = ai_engine.wait_for_model_loaded(provider_name=_embeddings_llm, model_name=embeddings_model)
        if not loaded:
            print("[Model Mgmt] WARNING: embeddings model may not be fully loaded yet.")

    print(f"Starting AI/LLM Embeddings Pass on {len(processed_job_pool)} jobs...")
    embedding_limiter = AILimiter("stage_2_embedding", _embeddings_llm)

    async def process_embeddings(index, job):
        if verbose:
            print(f"Embedding job {index + 1}/{len(processed_job_pool)}: {job['features']['title']}")

        if not job or 'features' not in job:
            return

        features = job['features']

        # Helper to generate embedding via limiter and run_in_thread
        async def get_embedding(text):
            if not text or not text.strip():
                return []
            est_tokens = (len(text) // 4) + 16
            async with embedding_limiter.semaphore:
                await embedding_limiter.wait(est_tokens)
                return await run_in_thread(generate_embeddings, ai_engine, text, provider_name=_embeddings_llm)

        # Vector generation
        title_text = features.get('title', '')
        job['embeddings']['title_vector'] = await get_embedding(title_text)

        skills_text = ", ".join(features.get('skills', []))
        job['embeddings']['skills_vector'] = await get_embedding(skills_text)

        requirements_text = ", ".join(features.get('requirements', []))
        job['embeddings']['requirements_vector'] = await get_embedding(requirements_text)

        summary_text = features.get('summary', '')
        job['embeddings']['description_vector'] = await get_embedding(summary_text)

    # Process all embedding tasks in parallel/sequential according to limiter
    embedding_tasks = [process_embeddings(idx, job) for idx, job in enumerate(processed_job_pool)]
    await asyncio.gather(*embedding_tasks)

    if is_lm_studio_embeddings:
        embeddings_model = ai_engine.embeddings_model
        print(f"[Model Mgmt] Unloading LM Studio embeddings model: {embeddings_model} ...")
        ai_engine.unload_model(provider_name=_embeddings_llm, model_name=embeddings_model)
        unloaded = ai_engine.wait_for_model_unloaded(provider_name=_embeddings_llm, model_name=embeddings_model)
        if not unloaded:
            print("[Model Mgmt] WARNING: embeddings model may not be fully unloaded yet.")

    print("AI/LLM Passes Complete.")

    # Persist to database if not skipping
    if dp and not skip_db:
        # Save embeddings
        print("Saving embeddings to database...")
        embedding_updates = []
        for job in processed_job_pool:
            if not job.get('metadata', {}).get('in_db', False):
                continue
            emb = job.get('embeddings', {})
            embedding_updates.append({
                "job_id": job['metadata']['job_id'],
                "title_embedding": emb.get('title_vector'),
                "skills_embedding": emb.get('skills_vector'),
                "responsibilities_embedding": emb.get('requirements_vector'),
                "description_embedding": emb.get('description_vector'),
            })
        if embedding_updates:
            dp.save_job_embeddings(embedding_updates)
            print(f"Saved {len(embedding_updates)} jobs' embeddings.")

        # Save metadata
        print("Updating job records in database with metadata...")
        job_updates = []
        for job in processed_job_pool:
            if not job.get('metadata', {}).get('in_db', False):
                continue
            job_updates.append({
                "id": job['metadata']['job_id'],
                "pay_range": job['features'].get('pay'),
                "seniority": job['features'].get('seniority'),
                "work_type": job['features'].get('work_type'),
                "timezone": job['features'].get('timezone'),
                "description": job['features'].get('summary'),
                "skills": job['features'].get('skills'),
                "responsibilities": job['features'].get('requirements'),
            })
        if job_updates:
            dp.update_job_metadata(job_updates)
            print(f"Updated {len(job_updates)} job records.")

    print(f"Stage 2 complete: {len(processed_job_pool)} jobs processed.")
    return processed_job_pool


# =====================================================
# PIPELINE STAGE 3: RULE FILTERING
# =====================================================

async def pipeline_stage_rule_filter(jobs: List[Dict], user_preferences: Optional[dict] = None,
                                     dp: Optional[DataPuller] = None,
                                     skip_db: bool = False,
                                     verbose: bool = False) -> List[Dict]:
    """
    Stage 3: Apply hard-constraint rule filtering to the job pool.
    
    Args:
        jobs: List of job dicts from Stage 2.
        user_preferences: Dict of user preferences (work types, seniority, pay, timezones).
        dp: DataPuller instance for DB sync.
        skip_db: If True, skip database persistence.
        verbose: If True, print detailed debug output.
    
    Returns:
        List[Dict] of all jobs (processed_job_pool) with 'skip' boolean added,
        plus the 'active_job_pool' key if returning the full context.
    """
    print("=" * 50)
    print("PIPELINE STAGE 3: RULE FILTERING")
    print("=" * 50)

    if user_preferences is None:
        user_preferences = {}

    print(f"Starting Rule-Based Filtering on {len(jobs)} jobs...")

    for job in jobs:
        job['skip'] = apply_rule_filters(job, user_preferences)

    skipped_count = sum(1 for j in jobs if j['skip'])
    print(f"Filtering complete. {skipped_count} jobs skipped, {len(jobs) - skipped_count} active.")

    # Sync skip status to database
    if dp and not skip_db:
        print("Syncing skip status to database...")
        skipped_ids = [job['metadata']['job_id'] for job in jobs if job.get('skip')]
        if skipped_ids:
            dp.bulk_update_skip_status(skipped_ids)
            print(f"Updated {len(skipped_ids)} jobs as skipped.")
        else:
            print("No jobs to skip in database.")

    print(f"Stage 3 complete. Active jobs: {len(jobs) - skipped_count}")
    return jobs


# =====================================================
# PIPELINE STAGE 4: ARCHETYPE ENGINE INTEGRATION
# =====================================================

async def pipeline_stage_archetype_integration(jobs: List[Dict], ai_engine: Optional[AIEngine] = None,
                                                dp: Optional[DataPuller] = None,
                                                setup_data: Optional[dict] = None,
                                                skip_db: bool = False,
                                                verbose: bool = False) -> Tuple[List[Dict], ArchetypeManager]:
    """
    Stage 4: Load/integrate archetypes (benchmarks + user profile/resume).
    
    Args:
        jobs: List of job dicts from Stage 3 (with skip flags).
        ai_engine: AIEngine for generating archetype embeddings.
        dp: DataPuller for DB operations.
        setup_data: Setup data containing resume/user_profile text.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        Tuple of (active_jobs, archetype_manager).
    """
    print("=" * 50)
    print("PIPELINE STAGE 4: ARCHETYPE ENGINE INTEGRATION")
    print("=" * 50)

    if ai_engine is None:
        ai_engine = AIEngine(default_provider_name="lm_studio")

    _user_config = _load_user_config()
    _embeddings_llm = _user_config.get("embeddings_llm", os.getenv("EMBEDDINGS_LLM", "lm_studio"))

    from app.vector_engine import CallableEmbeddingProvider
    embed_fn = lambda text: generate_embeddings(ai_engine, text, provider_name=_embeddings_llm)
    embedding_provider = CallableEmbeddingProvider(embed_fn)
    archetype_manager = ArchetypeManager(embedding_provider=embedding_provider)

    # Get active (non-skipped) jobs
    active_jobs = [j for j in jobs if not j.get('skip')]
    print(f"Active jobs for archetype comparison: {len(active_jobs)}")

    print("Synchronizing candidate archetypes and benchmarks...")

    # Extract structured profile data using LLM with file-modification caching
    resume_text = ""
    profile_text = ""
    if setup_data:
        resume_text = setup_data.get("resume", "")
        profile_text = setup_data.get("user_profile", "")

    resume_path_env = os.getenv("RESUME", "")
    resume_path_resolved = resume_path_env if (resume_path_env and os.path.exists(resume_path_env)) else "archetype_profiles/resume_cache.json"

    profile_path_env = os.getenv("PROFILE", "")
    profile_path_resolved = profile_path_env if (profile_path_env and os.path.exists(profile_path_env)) else "archetype_profiles/user_profile_cache.json"

    resume_data = extract_and_cache_profile(
        ai_engine,
        resume_path_resolved,
        resume_text if resume_text else "No resume text provided.",
        "archetype_profiles/resume_cache.json"
    )
    profile_data = extract_and_cache_profile(
        ai_engine,
        profile_path_resolved,
        profile_text if profile_text else "No profile text provided.",
        "archetype_profiles/user_profile_cache.json"
    )

    resume_mtime = None
    if resume_path_resolved and os.path.exists(resume_path_resolved):
        resume_mtime = os.path.getmtime(resume_path_resolved)

    profile_mtime = None
    if profile_path_resolved and os.path.exists(profile_path_resolved):
        profile_mtime = os.path.getmtime(profile_path_resolved)

    all_archetypes = ARCHETYPES_CONFIG + [
        {
            "name": "Resume",
            "title": ", ".join(resume_data.get("skills", [])),
            "skills": "\n".join(resume_data.get("skills", [])),
            "responsibilities": "\n".join(resume_data.get("requirements", [])),
            "summary": resume_data.get("summary", ""),
            "type": "resume",
            "source_mtime": resume_mtime
        },
        {
            "name": "User Profile",
            "title": ", ".join(profile_data.get("skills", [])),
            "skills": "\n".join(profile_data.get("skills", [])),
            "responsibilities": "\n".join(profile_data.get("requirements", [])),
            "summary": profile_data.get("summary", ""),
            "type": "user_profile",
            "source_mtime": profile_mtime
        }
    ]

    # Determine the expected dimension of the current embedding provider
    test_emb = generate_embeddings(ai_engine, "test", provider_name=_embeddings_llm)
    expected_dim = len(test_emb) if test_emb else 0

    for arch_config in all_archetypes:
        cached_arch = None
        if dp:
            cached_arch = dp.get_archetype_embeddings(arch_config['name'])

        cache_valid = False
        if cached_arch:
            title_emb = cached_arch.get('title_embedding')
            if title_emb and len(title_emb) == expected_dim:
                source_mtime = arch_config.get("source_mtime")
                if source_mtime is not None:
                    db_meta = cached_arch.get('metadata') or {}
                    db_mtime = db_meta.get("source_mtime")
                    if db_mtime is not None:
                        if db_mtime >= source_mtime:
                            cache_valid = True
                        else:
                            print(f"Archetype '{arch_config['name']}' source file modified since embedding generation. Invalidating cache.")
                    else:
                        # Fallback to date_generated database timestamp comparison
                        db_dt = cached_arch.get('date_generated')
                        if db_dt:
                            import datetime
                            file_dt = datetime.datetime.fromtimestamp(source_mtime)
                            if db_dt.tzinfo is not None:
                                file_dt_tz = datetime.datetime.fromtimestamp(source_mtime, tz=datetime.timezone.utc)
                                if db_dt >= file_dt_tz:
                                    cache_valid = True
                                else:
                                    print(f"Archetype '{arch_config['name']}' source file modified since DB generation timestamp (with tz). Invalidating cache.")
                            else:
                                if db_dt >= file_dt:
                                    cache_valid = True
                                else:
                                    print(f"Archetype '{arch_config['name']}' source file modified since DB generation timestamp. Invalidating cache.")
                        else:
                            print(f"Archetype '{arch_config['name']}' has no modification timestamp metadata or database timestamp. Invalidating cache.")
                else:
                    # Config-driven benchmark archetypes do not map to files
                    cache_valid = True

        if cache_valid:
            archetype_manager.add_archetype(Archetype(
                name=arch_config['name'],
                type=cached_arch['archetype_type'],
                title_embedding=np.array(cached_arch['title_embedding']),
                skills_embedding=np.array(cached_arch['skills_embedding']),
                responsibilities_embedding=np.array(cached_arch['responsibilities_embedding']),
                metadata=cached_arch.get('metadata', {})
            ))
        else:
            if cached_arch:
                print(f"Cached archetype '{arch_config['name']}' is invalid (modified source or dimension mismatch). Regenerating...")
            else:
                print(f"Generating new embeddings for archetype: {arch_config['name']}")
            new_arch = archetype_manager.load_archetype(
                name=arch_config['name'],
                archetype_data=arch_config,
                archetype_type=arch_config.get("type", "benchmark"),
                metadata={"source_mtime": arch_config.get("source_mtime")} if arch_config.get("source_mtime") is not None else None
            )
            # Persist to DB for future runs
            if dp and not skip_db:
                dp.save_archetype_embeddings({
                    "archetype_name": new_arch.name,
                    "archetype_type": new_arch.type,
                    "title_embedding": new_arch.title_embedding.tolist() if new_arch.title_embedding is not None else None,
                    "skills_embedding": new_arch.skills_embedding.tolist() if new_arch.skills_embedding is not None else None,
                    "responsibilities_embedding": new_arch.responsibilities_embedding.tolist() if new_arch.responsibilities_embedding is not None else None,
                    "metadata": json.dumps(new_arch.metadata)
                })

    print(f"Stage 4 complete: Loaded {len(archetype_manager.archetypes)} archetypes.")
    return active_jobs, archetype_manager


# =====================================================
# PIPELINE STAGE 5: VECTOR SCORING WITH ARCHETYPES
# =====================================================

async def pipeline_stage_vector_scoring(jobs: List[Dict], archetype_manager: Optional[ArchetypeManager] = None,
                                         dp: Optional[DataPuller] = None,
                                         skip_db: bool = False,
                                         verbose: bool = False) -> List[Dict]:
    """
    Stage 5: Compare jobs to archetypes, compute semantic scores, and filter by threshold.
    
    Args:
        jobs: List of active job dicts from Stage 4.
        archetype_manager: ArchetypeManager with loaded archetypes.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of filtered jobs with semantic scores and archetype matches.
    """
    print("=" * 50)
    print("PIPELINE STAGE 5: VECTOR SCORING WITH ARCHETYPES")
    print("=" * 50)

    if archetype_manager is None:
        archetype_manager = ArchetypeManager()

    print("Comparing jobs to archetypes...")
    for index, job in enumerate(jobs):
        if verbose:
            print(f"Processing job {index + 1}/{len(jobs)}: {job['features']['title']}")

        matches = archetype_manager.compare_job_to_archetypes(job)
        job['retrieval_metadata'] = archetype_manager.generate_retrieval_metadata(job, matches)
        job['archetype_matches'] = matches

    print("Archetype comparison complete.")

    # Import adjustment functions from vector_engine
    from app.vector_engine import apply_keyword_adjustments, apply_metadata_adjustments

    print("Applying weighted semantic scoring...")
    for job in jobs:
        matches = job.get('archetype_matches', [])
        if not matches:
            continue

        best_match = matches[0]

        title_similarity = best_match.get('title_similarity', 0.0)
        skills_similarity = best_match.get('skills_similarity', 0.0)
        responsibility_similarity = best_match.get('responsibility_similarity', 0.0)

        # Weighted semantic score
        semantic_score = (
            0.40 * title_similarity +
            0.35 * skills_similarity +
            0.25 * responsibility_similarity
        )

        # Keyword adjustments
        job_skills = job.get('features', {}).get('skills', [])
        job_title = job.get('features', {}).get('title', '')
        semantic_score = apply_keyword_adjustments(semantic_score, job_skills, job_title)

        # Metadata adjustments
        job_meta = {
            "is_remote": (job.get('features', {}).get('work_type') or '').lower() == 'remote',
            "salary": 0,
            "days_old": 30
        }
        pay_range = job.get('features', {}).get('pay', '')
        if pay_range:
            import re as _re
            numbers = _re.findall(r'\d+(?:,\d+)?', pay_range.replace(',', ''))
            if len(numbers) >= 2:
                job_meta["salary"] = int(numbers[1])
            elif len(numbers) == 1:
                job_meta["salary"] = int(numbers[0])

        semantic_score = apply_metadata_adjustments(semantic_score, job_meta)

        # Clamp and normalize
        semantic_score = max(0.0, min(1.0, semantic_score))
        score_percent = int(round(semantic_score * 100))

        job['semantic_score'] = semantic_score
        job['semantic_score_percent'] = score_percent
        job['title_similarity'] = title_similarity
        job['skills_similarity'] = skills_similarity
        job['responsibility_similarity'] = responsibility_similarity
        job['adjusted_score'] = semantic_score
        job['best_archetype'] = best_match.get('archetype_name', '')

    # Generate retrieval metadata
    for job in jobs:
        if 'retrieval_metadata' not in job:
            job['retrieval_metadata'] = {}
        if 'semantic_score' in job:
            job['retrieval_metadata']['semantic_score'] = job['semantic_score']
            job['retrieval_metadata']['semantic_score_percent'] = job['semantic_score_percent']
            job['retrieval_metadata']['best_archetype'] = job.get('best_archetype', '')

    # Filter by threshold
    MIN_SCORE_THRESHOLD = 0.72
    TARGET_COUNT = 20

    filtered_job_pool = []
    for job in jobs:
        if job.get('semantic_score', 0) >= MIN_SCORE_THRESHOLD:
            filtered_job_pool.append(job)

    # Fallback: add top-X if not enough jobs meet threshold
    if len(filtered_job_pool) < TARGET_COUNT:
        sorted_jobs = sorted(jobs, key=lambda x: x.get('semantic_score', 0), reverse=True)
        top_n = max(TARGET_COUNT - len(filtered_job_pool), 1)
        for job in sorted_jobs[:top_n]:
            if job not in filtered_job_pool:
                filtered_job_pool.append(job)

    print(f"Filtered to {len(filtered_job_pool)} jobs (threshold >= {MIN_SCORE_THRESHOLD}).")

    # Persist vector scores
    if dp and not skip_db:
        try:
            sorted_all_jobs = sorted(jobs, key=lambda x: x.get('semantic_score', 0), reverse=True)
            dp.save_vector_scores(sorted_all_jobs)
            print(f"Persisted {len(sorted_all_jobs)} vector scores to database.")
        except Exception as e:
            error_logger_continue(f"Vector score persistence failed: {e}")

    # Print top jobs
    sorted_filtered = sorted(filtered_job_pool, key=lambda x: x.get('semantic_score', 0), reverse=True)
    print("\n=== Semantic Score-Based Job Shortlisting ===")
    for i, job in enumerate(sorted_filtered[:5]):
        print(f"{i+1}. {job['features']['title']}")
        print(f"   Score: {job.get('semantic_score_percent', 'N/A')}% | Archetype: {job.get('best_archetype', 'None')}")

    print(f"Stage 5 complete: {len(filtered_job_pool)} jobs in filtered pool.")
    return filtered_job_pool


# =====================================================
# PIPELINE STAGE 6: CHEAP LLM CLASSIFICATION
# =====================================================

async def pipeline_stage_cheap_llm(jobs: List[Dict], setup_data: Optional[dict] = None,
                                    dp: Optional[DataPuller] = None,
                                    skip_db: bool = False,
                                    verbose: bool = False) -> List[Dict]:
    """
    Stage 6: Run cheap (fast) LLM classification on the filtered job pool.
    
    Args:
        jobs: List of filtered job dicts from Stage 5.
        setup_data: Setup data containing user_profile and skills.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of shortlisted jobs with 'cheap_llm_result' added.
    """
    print("=" * 50)
    print("PIPELINE STAGE 6: CHEAP LLM CLASSIFICATION")
    print("=" * 50)

    user_profile = setup_data.get("user_profile", "") if setup_data else ""
    skills = setup_data.get("skills", []) if setup_data else []

    if not user_profile:
        print("Warning: No user profile available for classification.")

    # Initialize cheap LLM classifier
    _user_config = _load_user_config()
    cheap_llm_provider = _user_config.get("cheap_llm_provider", os.getenv("CHEAP_LLM_PROVIDER", "gemini"))
    cheap_llm_model = _user_config.get("cheap_llm_model", os.getenv("CHEAP_LLM_MODEL"))
    cheap_classifier = CheapLLMClassifier(provider=cheap_llm_provider, model=cheap_llm_model)

    # Run Stage 6 on filtered job pool
    shortlisted_jobs = await process_stage_6(
        jobs=jobs,
        classifier=cheap_classifier,
        candidate_profile=user_profile,
        candidate_skills=skills,
        batch_size=5
    )

    # Persist results to database
    if dp and not skip_db:
        print("Persisting Stage 6 results to database...")
        try:
            # 1. Save cheap LLM results for ALL jobs that were classified
            classified_jobs = [j for j in jobs if 'cheap_llm_result' in j]
            dp.save_cheap_llm_results(classified_jobs)
            print(f"Persisted {len(classified_jobs)} cheap LLM results.")
            
            # 2. Update job table skip status for jobs classified as 'skip'
            skipped_ids = [j['metadata']['job_id'] for j in jobs if j.get('cheap_llm_result', {}).get('decision') == 'skip']
            if skipped_ids:
                dp.bulk_update_skip_status(skipped_ids)
                print(f"Marked {len(skipped_ids)} skipped jobs as skip=True in the database.")
        except Exception as e:
            error_logger_continue(f"Stage 6 persistence failed: {e}")

    # 3. Clean up/remove skipped jobs from the in-memory pool in-place to free memory
    shortlisted_ids = {j['metadata']['job_id'] for j in shortlisted_jobs}
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        job_id = job.get('metadata', {}).get('job_id')
        if job_id not in shortlisted_ids:
            jobs.pop(i)
        i -= 1

    print(f"Stage 6 complete: {len(jobs)} active jobs remaining in memory pool.")
    return jobs


# =====================================================
# PIPELINE STAGE 7: STRONG LLM RERANKING
# =====================================================

async def pipeline_stage_strong_llm(jobs: List[Dict], setup_data: Optional[dict] = None,
                                     dp: Optional[DataPuller] = None,
                                     skip_db: bool = False,
                                     verbose: bool = False) -> List[Dict]:
    """
    Stage 7: Run strong (deep) LLM reranking on top candidates from Stage 6.
    
    Args:
        jobs: List of shortlisted job dicts from Stage 6.
        setup_data: Setup data containing user_profile and skills.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of deeply analyzed jobs with 'strong_llm_result' added.
    """
    print("=" * 50)
    print("PIPELINE STAGE 7: STRONG LLM RERANKING")
    print("=" * 50)

    user_profile = setup_data.get("user_profile", "") if setup_data else ""
    skills = setup_data.get("skills", []) if setup_data else []

    # Initialize strong LLM reranker
    _user_config = _load_user_config()
    strong_llm_provider = _user_config.get("strong_llm_provider", os.getenv("STRONG_LLM_PROVIDER", "claude"))
    strong_llm_model = _user_config.get("strong_llm_model", os.getenv("STRONG_LLM_MODEL"))
    strong_reranker = StrongLLMReranker(provider=strong_llm_provider, model=strong_llm_model)

    # Configure how many jobs to deeply analyze
    top_n_for_deep_analysis = int(_user_config.get("top_n_deep_analysis", os.getenv("TOP_N_DEEP_ANALYSIS", "15")))

    # Run Stage 7 on top candidates from Stage 6
    deeply_analyzed_jobs = await process_stage_7(
        jobs=jobs,
        reranker=strong_reranker,
        candidate_profile=user_profile,
        candidate_skills=skills,
        top_n=top_n_for_deep_analysis
    )

    # Persist results
    if dp and not skip_db:
        print("Persisting Stage 7 results to database...")
        try:
            dp.save_strong_llm_results(deeply_analyzed_jobs)
            print(f"Persisted {len(deeply_analyzed_jobs)} strong LLM results.")
        except Exception as e:
            error_logger_continue(f"Stage 7 persistence failed: {e}")

    # Remove jobs from memory pool that did not proceed to deep analysis in-place
    analyzed_ids = {j['metadata']['job_id'] for j in deeply_analyzed_jobs}
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        job_id = job.get('metadata', {}).get('job_id')
        if job_id not in analyzed_ids:
            jobs.pop(i)
        i -= 1

    print(f"Stage 7 complete: {len(jobs)} jobs deeply analyzed and remaining in memory pool.")
    return jobs


# =====================================================
# PIPELINE STAGE 8: FINAL APPLICATION QUEUE
# =====================================================

async def pipeline_stage_final_queue(jobs: List[Dict], dp: Optional[DataPuller] = None,
                                      skip_db: bool = False,
                                      verbose: bool = False) -> List[Dict]:
    """
    Stage 8: Generate the final ranked application queue from deeply analyzed jobs.
    
    Args:
        jobs: List of deeply analyzed job dicts from Stage 7.
        dp: DataPuller for DB persistence.
        skip_db: If True, skip DB operations.
        verbose: If True, print detailed output.
    
    Returns:
        List[Dict] of ranked jobs with final_score, priority, apply_recommendation.
    """
    print("=" * 50)
    print("PIPELINE STAGE 8: FINAL APPLICATION QUEUE")
    print("=" * 50)

    # Run Stage 8 to generate final ranked queue
    final_queue = await process_stage_8(jobs=jobs)

    # Persist results
    if dp and not skip_db:
        print("Persisting final application queue to database...")
        try:
            dp.save_final_queue(final_queue)
            print(f"Persisted {len(final_queue)} jobs to final queue.")
            
            # Update job table skip status for jobs recommended to skip
            skipped_ids = [j['metadata']['job_id'] for j in final_queue if j.get('priority') == 'skip' or j.get('apply_recommendation') == 'skip']
            if skipped_ids:
                dp.bulk_update_skip_status(skipped_ids)
                print(f"Marked {len(skipped_ids)} skipped jobs as skip=True in the database.")
        except Exception as e:
            error_logger_continue(f"Final queue persistence failed: {e}")

    # Remove skipped jobs from the memory pool in-place to free memory
    i = len(jobs) - 1
    while i >= 0:
        job = jobs[i]
        if job.get('priority') == 'skip' or job.get('apply_recommendation') == 'skip':
            jobs.pop(i)
        i -= 1

    # Filter final_queue returned list to only contain active non-skipped jobs
    active_final_queue = [j for j in final_queue if j.get('priority') != 'skip' and j.get('apply_recommendation') != 'skip']

    # Print detailed final queue (only for non-skipped jobs)
    print("\n" + "=" * 60)
    print("DETAILED FINAL APPLICATION QUEUE")
    print("=" * 60)

    for i, job in enumerate(active_final_queue[:10], 1):
        features = job.get('features', {})
        metadata = job.get('metadata', {})
        cheap_result = job.get('cheap_llm_result', {})
        strong_result = job.get('strong_llm_result', {})

        print(f"\n{i}. {features.get('title', 'Unknown')}")
        print(f"   Job ID: {metadata.get('job_id', 'Unknown')} | Company: {metadata.get('company_name', 'Unknown')}")
        print(f"   Link: {metadata.get('link', '')}")
        print(f"   Priority: {job.get('priority', 'unknown').upper()}")
        print(f"   Final Score: {job.get('final_score', 0):.1f}/100")
        print(f"   Recommendation: {job.get('apply_recommendation', 'maybe').upper()}")
        print(f"   Cheap LLM Fit: {cheap_result.get('fit_score', 0)}/100")
        print(f"   Strong LLM Score: {strong_result.get('final_score', 0)}/100")

    print(f"\nStage 8 complete: {len(active_final_queue)} active jobs in final queue.")
    return active_final_queue


def _row_to_job_dict(row) -> Dict:
    if hasattr(row, 'get'):
        d = row
    else:
        # Map tuple indices
        keys = [
            "id", "job_name", "company_name", "link", "job_summary", "extracted_summary",
            "skills", "responsibilities", "pay_range", "seniority", "work_type", "timezone",
            "source", "date_added", "city", "state", "location",
            "flexibility",
            "title_embedding", "skills_embedding", "responsibilities_embedding", "description_embedding"
        ]
        d = {keys[i]: row[i] for i in range(min(len(keys), len(row)))}

    # Parse skills and responsibilities
    skills = d.get("skills")
    if isinstance(skills, str):
        try:
            skills = json.loads(skills)
        except Exception:
            skills = [s.strip() for s in skills.split(",") if s.strip()]
    elif not isinstance(skills, list):
        skills = []

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
            "seniority": d.get("seniority") if not is_missing_value(d.get("seniority")) else "NA",
            "work_type": d.get("work_type") if not is_missing_value(d.get("work_type")) else (d.get("flexibility") if not is_missing_value(d.get("flexibility")) else "NA"),
            "timezone": d.get("timezone") if not is_missing_value(d.get("timezone")) else "NA",
            "skills": skills,
            "requirements": responsibilities,
            "city": d.get("city"),
            "state": d.get("state"),
            "location": d.get("location"),
        },
        "embeddings": {
            "title_vector": d.get("title_embedding"),
            "skills_vector": d.get("skills_embedding"),
            "requirements_vector": d.get("responsibilities_embedding"),
            "description_vector": d.get("description_embedding"),
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


async def scrape_missing_descriptions_24h(dp: DataPuller, verbose: bool = False) -> List[Dict]:
    """
    Perform the special scraping workflow for jobs in the database from the last 24 hours that do not have descriptions.
    """
    # 1. Update matching jobs: skip = NULL
    update_query = """
        UPDATE job
        SET skip = NULL
        WHERE job_summary IS NULL AND date_added >= CURRENT_DATE - INTERVAL '1 day';
    """
    try:
        dp.conn.execute_sql(update_query)
        print("Updated jobs from the previous 24 hours missing descriptions to skip = NULL.")
    except Exception as e:
        print(f"Error resetting skip status for jobs missing descriptions: {e}")

    # 2. Pull all jobs from the previous 24 hours where the description (job_summary) is null
    select_query = """
        SELECT j.id, j.link, j.source, j.job_name, c.company_name, j.date_added
        FROM job j
        LEFT JOIN company c ON j.company_id = c.id
        WHERE j.job_summary IS NULL AND j.date_added >= CURRENT_DATE - INTERVAL '1 day';
    """
    try:
        rows = dp.conn.execute_sql(select_query, fetch=True)
    except Exception as e:
        print(f"Error querying jobs missing descriptions: {e}")
        return []

    if not rows:
        print("No jobs found in the previous 24 hours missing descriptions.")
        return []

    jobs_without_desc = []
    for row in rows:
        if isinstance(row, dict):
            jobs_without_desc.append({
                "db_id": row.get("id"),
                "url": row.get("link"),
                "source": row.get("source"),
                "title": row.get("job_name"),
                "company": row.get("company_name"),
                "date": row.get("date_added"),
            })
        else:
            jobs_without_desc.append({
                "db_id": row[0],
                "url": row[1],
                "source": row[2],
                "title": row[3] if len(row) > 3 else "",
                "company": row[4] if len(row) > 4 else "",
                "date": row[5] if len(row) > 5 else None,
            })

    print(f"Found {len(jobs_without_desc)} jobs from the last 24 hours missing descriptions to scrape.")

    import datetime
    import glob
    from urllib.parse import urlparse
    import time
    import random
    
    os.makedirs(os.path.join("logs", "skipped"), exist_ok=True)
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file_path = os.path.join("logs", f"missing_desc_scrape_{timestamp}.log")
    
    job_page_strategy_dir = "./job_page_strategy"
    
    def find_strategy_file(source_name: str) -> str | None:
        if not source_name:
            return None
        source_lower = source_name.lower()
        for filepath in glob.glob(os.path.join(job_page_strategy_dir, "*.json")):
            filename = os.path.basename(filepath)
            name = filename[:-5].lower() # remove .json
            if name == source_lower:
                return filepath
            if "." in name:
                domain_part = name.split('.')[0]
                if domain_part == source_lower:
                    return filepath
            if "." in source_lower:
                source_domain_part = source_lower.split('.')[0]
                if source_domain_part == name:
                    return filepath
        return None

    successful_ids = []
    log_entries = []

    # ── Per-domain rate-limit tracking ──
    # Tracks consecutive 429s per domain so we can back off more aggressively.
    _domain_429_count: Dict[str, int] = {}
    MAX_CONSECUTIVE_429 = 3      # threshold after which we apply extra backoff
    EXTRA_BACKOFF_SECONDS = 30   # extra cooldown after hitting MAX_CONSECUTIVE_429 on a domain

    for job in jobs_without_desc:
        db_id = job["db_id"]
        url = job["url"]
        source = job.get("source", "")
        job_name = job.get("title", "")
        company_name = job.get("company", "")

        # Extract domain for rate-limit tracking
        try:
            parsed_domain = urlparse(url).netloc.lower()
            if parsed_domain.startswith("www."):
                parsed_domain = parsed_domain[4:]
        except Exception:
            parsed_domain = source.lower()

        if not url or not source:
            log_msg = f"Job ID {db_id} ({job_name} | {company_name}): Failed to scrape. Reason: Missing URL or source."
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")
            continue

        strategy_path = find_strategy_file(source)
        if not strategy_path:
            try:
                parsed = urlparse(url)
                domain = parsed.netloc.lower()
                if domain.startswith("www."):
                    domain = domain[4:]
            except Exception:
                domain = source.lower()
            
            print(f"  No strategy file found for source '{source}'. Generating strategy for domain '{domain}' using {url}...")
            success = await dp.generate_and_test_strategy(destination_link=url, job_id=db_id, domain_name=domain)
            if success:
                strategy_path = find_strategy_file(source)
                if not strategy_path:
                    strategy_path = os.path.join(job_page_strategy_dir, f"{domain}.json")
            else:
                log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Reason: Could not generate a working strategy."
                print(log_msg)
                log_entries.append(log_msg)
                try:
                    dp.bulk_update_skip_status([db_id])
                except Exception as e:
                    print(f"Warning: failed to update skip status for job {db_id}: {e}")
                continue

        try:
            with open(strategy_path, "r") as f:
                strategy = json.load(f)
        except Exception as e:
            log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Reason: Error loading strategy file: {e}"
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")
            continue

        payload = dict(strategy)
        if payload.get("url") == "{url}":
            payload["url"] = url

        api_method = "extract-js" if payload.get("js_config") is not None else "extract"

        # ── Attempt scrape with retry + exponential backoff on 429 ──
        max_retries = 4
        base_delay = 2.0
        description = ""
        scraper_response = None
        did_skip_due_to_rate_limit = False

        for attempt in range(1, max_retries + 1):
            # Domain-level cooldown: if this domain has been hammered with 429s,
            # insert an extra forced delay before we even try.
            domain_strikes = _domain_429_count.get(parsed_domain, 0)
            if domain_strikes >= MAX_CONSECUTIVE_429:
                extra_sleep = EXTRA_BACKOFF_SECONDS * (1 + random.random() * 0.5)
                print(f"  [Rate Limit] Domain '{parsed_domain}' has {domain_strikes} consecutive 429s. "
                      f"Cooling off for {extra_sleep:.0f}s before retry (attempt {attempt}/{max_retries})...")
                await asyncio.sleep(extra_sleep)
                # Reduce the counter so we don't loop forever on extra backoff;
                # it will only re-trigger if we get another 429.
                _domain_429_count[parsed_domain] = max(0, domain_strikes - 1)

            try:
                scraper_response = await dp.scrape_data(payload, api_method=api_method)
            except Exception as e:
                scraper_response = {"error": f"Exception raised during scrape: {e}"}

            status = scraper_response.get("status_code")
            if status == 429:
                # Track the 429 per domain
                _domain_429_count[parsed_domain] = _domain_429_count.get(parsed_domain, 0) + 1
                domain_strikes = _domain_429_count[parsed_domain]

                if attempt < max_retries:
                    # Exponential backoff: 2s, 4s, 8s + jitter
                    delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 1)
                    print(f"  [429] Job ID {db_id}: Rate limited (domain={parsed_domain}, "
                          f"strike={domain_strikes}). Retrying in {delay:.1f}s "
                          f"(attempt {attempt+1}/{max_retries})...")
                    await asyncio.sleep(delay)
                    continue
                else:
                    # All retries exhausted — mark as skip but log the 429
                    log_msg = (f"Job ID {db_id} (URL: {url}): Failed to scrape after {max_retries} retries. "
                               f"All attempts returned 429. Scraper response: {json.dumps(scraper_response)}")
                    print(log_msg)
                    log_entries.append(log_msg)
                    try:
                        dp.bulk_update_skip_status([db_id])
                    except Exception as e:
                        print(f"Warning: failed to update skip status for job {db_id}: {e}")
                    did_skip_due_to_rate_limit = True
                    break

            elif status == 200 and scraper_response.get("data"):
                # Success — reset 429 count for this domain
                _domain_429_count[parsed_domain] = 0

                data_result = scraper_response["data"]
                if isinstance(data_result, list) and len(data_result) > 0:
                    desc_key = next(
                        (k for k in ("description", "summary", "job-summary", "job_description", "job-description")
                         if data_result[0].get(k)),
                        None
                    )
                    if desc_key:
                        desc_text = data_result[0].get(desc_key, "")
                    else:
                        desc_text = next(
                            (v for v in data_result[0].values() if isinstance(v, str) and len(v) > 50),
                            ""
                        )
                    if desc_text and len(desc_text) > 50:
                        description = desc_text
                break  # success — exit retry loop
            else:
                # Non-429 error (4xx, 5xx, etc.) — do not retry, just mark as skip
                break

        if did_skip_due_to_rate_limit:
            continue

        if description:
            try:
                dp.conn.update("job", {"job_summary": description}, {"id": db_id}, dbname=dp.dbname)
                successful_ids.append(db_id)
                log_msg = f"Job ID {db_id} (URL: {url}): Scraped successfully."
                print(log_msg)
                log_entries.append(log_msg)
            except Exception as e:
                log_msg = f"Job ID {db_id} (URL: {url}): Failed to update DB with description: {e}"
                print(log_msg)
                log_entries.append(log_msg)
        else:
            resp_str = json.dumps(scraper_response) if scraper_response else "No response returned"
            log_msg = f"Job ID {db_id} (URL: {url}): Failed to scrape. Scraper response: {resp_str}"
            print(log_msg)
            log_entries.append(log_msg)
            try:
                dp.bulk_update_skip_status([db_id])
            except Exception as e:
                print(f"Warning: failed to update skip status for job {db_id}: {e}")

        # Base delay between jobs (0.5-1.5s) — only if we didn't already sleep for a 429 retry
        time.sleep(random.uniform(0.5, 1.5))

    try:
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(log_entries) + "\n")
        print(f"Detailed run status saved to: {log_file_path}")
    except Exception as e:
        print(f"Warning: failed to write run log file '{log_file_path}': {e}")

    if not successful_ids:
        return []

    # Fetch fully populated records from DB
    query_cols = [
        "j.id", "j.job_name", "c.company_name", "j.link", "j.job_summary", "j.description AS extracted_summary",
        "j.skills", "j.responsibilities", "j.pay_range", "j.seniority", "j.work_type", "j.timezone",
        "j.source", "j.date_added", "o.city", "o.state", "o.location",
        "j.flexibility",
        "je.title_embedding", "je.skills_embedding", "je.responsibilities_embedding", "je.description_embedding"
    ]
    joins = [
        "JOIN company c ON j.company_id = c.id",
        "LEFT JOIN office o ON j.office_id = o.id",
        "LEFT JOIN job_embeddings je ON j.id = je.job_id"
    ]
    ids_placeholder = ", ".join(["%s"] * len(successful_ids))
    query = f"""
        SELECT {', '.join(query_cols)}
        FROM job j
        {' '.join(joins)}
        WHERE j.id IN ({ids_placeholder})
    """
    try:
        rows = dp.conn.execute_sql(query, tuple(successful_ids), fetch=True)
        if rows:
            return [_row_to_job_dict(row) for row in rows]
    except Exception as e:
        print(f"Error fetching successfully scraped jobs from DB: {e}")

    return []

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
        "j.skills", "j.responsibilities", "j.pay_range", "j.seniority", "j.work_type", "j.timezone",
        "j.source", "j.date_added", "o.city", "o.state", "o.location",
        "j.flexibility",
        "je.title_embedding", "je.skills_embedding", "je.responsibilities_embedding", "je.description_embedding"
    ]
    
    joins = [
        "JOIN company c ON j.company_id = c.id",
        "LEFT JOIN office o ON j.office_id = o.id",
        "LEFT JOIN job_embeddings je ON j.id = je.job_id"
    ]
    
    where_clauses = ["j.skip IS NOT TRUE"]
    
    if not force_reprocess:
        if stage < 8:
            where_clauses.append("j.id NOT IN (SELECT job_id FROM strong_llm_results)")
        if stage < 6:
            where_clauses.append("j.id NOT IN (SELECT job_id FROM cheap_llm_results)")
            
    if stage > 2:
        # For stages 3+, we need jobs that have completed Stage 2 (embeddings exist and are not NULL)
        where_clauses.append("je.job_id IS NOT NULL AND je.title_embedding IS NOT NULL")
    elif stage == 2:
        if force_reprocess:
            # Reprocess: select jobs that already have embeddings to re-run Stage 2
            where_clauses.append("je.job_id IS NOT NULL AND je.title_embedding IS NOT NULL")
        else:
            # Normal: select jobs that have been scraped but DON'T have embeddings yet,
            # or have an incomplete/NULL embedding record.
            where_clauses.append("(je.job_id IS NULL OR je.title_embedding IS NULL)")
        # Ensure we have a job description to embed
        where_clauses.append("j.job_summary IS NOT NULL")
    
    # Add columns and joins depending on what stage we are starting at
    if stage >= 6:
        query_cols.extend([
            "vs.semantic_score", "vs.title_similarity", "vs.skills_similarity",
            "vs.responsibility_similarity", "vs.adjusted_score", "vs.archetype_name AS best_archetype"
        ])
        joins.append("LEFT JOIN vector_scores vs ON j.id = vs.job_id")
        
    if stage >= 7:
        query_cols.extend([
            "clr.fit_score AS cheap_fit_score", "clr.decision AS cheap_decision",
            "clr.strengths AS cheap_strengths", "clr.concerns AS cheap_concerns"
        ])
        joins.append("LEFT JOIN cheap_llm_results clr ON j.id = clr.job_id")
        
    if stage >= 8:
        query_cols.extend([
            "slr.final_score AS strong_final_score", "slr.priority AS strong_priority",
            "slr.apply_recommendation AS strong_apply_rec", "slr.red_flags AS strong_red_flags",
            "slr.tailoring_notes AS strong_tailoring_notes", "slr.recruiter_bait_likelihood AS strong_bait",
            "slr.detailed_fit_analysis AS strong_fit_analysis"
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

    query = f"""
        SELECT {', '.join(query_cols)}
        FROM job j
        {' '.join(joins)}
        WHERE {' AND '.join(where_clauses)}
        ORDER BY j.date_added DESC
        LIMIT %s
    """
    
    try:
        # When Stage 1 is skipped, these staging queries may need more time
        # since they join several tables to load jobs that were previously
        # scraped. We double the default statement_timeout from 15s to 30s.
        dp.conn.execute_sql("SET statement_timeout = 60000")
        rows = dp.conn.execute_sql(query, (limit,), fetch=True)
    except Exception as e:
        print(f"Error querying DB for Stage {stage}: {e}")
        return []
        
    if not rows:
        print(f"No jobs found in DB for Stage {stage} processing.")
        return []
        
    jobs = []
    for row in rows:
        job = _row_to_job_dict(row)
        
        if stage >= 6 and isinstance(row, dict):
            job['best_archetype'] = row.get("best_archetype")
            semantic_score = row.get("semantic_score")
            job['semantic_score'] = semantic_score
            job['semantic_score_percent'] = int(round(semantic_score * 100)) if semantic_score is not None else 0
            job['title_similarity'] = row.get("title_similarity")
            job['skills_similarity'] = row.get("skills_similarity")
            job['responsibility_similarity'] = row.get("responsibility_similarity")
            job['adjusted_score'] = row.get("adjusted_score")
            job['retrieval_metadata'] = {
                "semantic_score": row.get("semantic_score"),
                "semantic_score_percent": job['semantic_score_percent'],
                "best_archetype": row.get("best_archetype")
            }
                
        if stage >= 7 and isinstance(row, dict):
            if row.get("cheap_decision"):
                strengths = row.get("cheap_strengths")
                if isinstance(strengths, str):
                    try: strengths = json.loads(strengths)
                    except Exception: strengths = []
                concerns = row.get("cheap_concerns")
                if isinstance(concerns, str):
                    try: concerns = json.loads(concerns)
                    except Exception: concerns = []
                    
                job['cheap_llm_result'] = {
                    "fit_score": row.get("cheap_fit_score"),
                    "decision": row.get("cheap_decision"),
                    "strengths": strengths or [],
                    "concerns": concerns or []
                }
                
        if stage >= 8 and isinstance(row, dict):
            if row.get("strong_apply_rec"):
                red_flags = row.get("strong_red_flags")
                if isinstance(red_flags, str):
                    try: red_flags = json.loads(red_flags)
                    except Exception: red_flags = []
                tailoring_notes = row.get("strong_tailoring_notes")
                if isinstance(tailoring_notes, str):
                    try: tailoring_notes = json.loads(tailoring_notes)
                    except Exception: tailoring_notes = []
                    
                job['strong_llm_result'] = {
                    "final_score": row.get("strong_final_score"),
                    "priority": row.get("strong_priority"),
                    "apply_recommendation": row.get("strong_apply_rec"),
                    "red_flags": red_flags or [],
                    "tailoring_notes": tailoring_notes or [],
                    "recruiter_bait_likelihood": row.get("strong_bait"),
                    "detailed_fit_analysis": row.get("strong_fit_analysis")
                }
                job['final_score'] = row.get("strong_final_score")
                job['priority'] = row.get("strong_priority")
                job['apply_recommendation'] = row.get("strong_apply_rec")
                
        jobs.append(job)
    return jobs


# =====================================================
# ORIGINAL MAIN (calls pipeline stages)
# =====================================================

async def main(scrape_pages: Optional[int] = None,
               scrape_visible: bool = False,
               scrape_debug: bool = False,
               skip_part_a: Optional[bool] = None,
               stage_range: str = "0-8",
               db_limit: int = 50,
               skip_db: bool = False,
               scrape_missing_24h: bool = False,
               reprocess: bool = False):
    """
    Run selected pipeline stages.

    Args:
        scrape_pages: Override max_pages for scraping.
        scrape_visible: Show browser window (disable headless).
        scrape_debug: Enable debug logging during scraping.
        skip_part_a: Skip Part A of legacy fallback scraping.
        stage_range: Inclusive stage range, e.g. "0-8", "2-4", "2".
        db_limit: Max records to pull from DB in any bulk-load operation.
        skip_db: Skip database operations for all stages.
        scrape_missing_24h: Only scrape descriptions for jobs from the last 24h.

    Returns:
        Pipeline result from the last executed stage.
    """
    # ── Parse stage range ──
    stage_start, stage_end = _parse_stage_range(stage_range)

    print("=" * 60)
    _print_stage_header("JOB SCRAPING AND ANALYSIS PIPELINE", stage_start, stage_end)
    print(f"DB limit: {db_limit}")
    print("=" * 60)

    # ── Relocate existing log files ──
    _move_existing_logs()

    # Stage 0 is always run (provides essential infrastructure)
    setup_data = await pipeline_stage_setup(skip_db=skip_db, verbose=False)
    dp = setup_data["dp"]
    ai = setup_data["ai"]
    tp = setup_data["tp"]

    # Store db_limit in setup_data for downstream stages
    setup_data["db_limit"] = db_limit

    _log_pipeline_stats("0: Setup", 0, source_hint="profile_extraction")

    # ── Per-stage execution ──
    processed_job_pool: List[Dict] = []
    active_jobs: List[Dict] = []
    archetype_manager: Optional[ArchetypeManager] = None
    filtered_job_pool: List[Dict] = []
    shortlisted_jobs: List[Dict] = []
    deeply_analyzed_jobs: List[Dict] = []
    final_queue: List[Dict] = []

    # Stage 1: Scrape
    if 1 <= stage_end and stage_start <= 1:
        processed_job_pool = await pipeline_stage_scrape(
            setup_data=setup_data,
            skip_db=skip_db,
            verbose=False,
            max_pages=scrape_pages,
            headless=not scrape_visible,
            debug_logging=scrape_debug,
            skip_part_a=skip_part_a,
            db_limit=db_limit,
            scrape_missing_24h=scrape_missing_24h,
        )
        _log_pipeline_stats("1: Scrape", len(processed_job_pool), source_hint="adapters+fallback")
    elif stage_start > 1:
        print(f"[SKIP] Stage 1 (Scrape) — outside range {stage_start}-{stage_end}")

    # Stage 2: Embedding Generation + LLM Extraction
    if 2 <= stage_end and stage_start <= 2:
        # Load any jobs from DB that are missing embeddings (but have descriptions) and merge them
        if not skip_db:
            print("Checking DB for any active jobs missing embeddings...")
            db_jobs = await _load_jobs_for_stage(dp, stage=2, limit=db_limit, force_reprocess=False)
            if db_jobs:
                print(f"Found {len(db_jobs)} active jobs in DB missing embeddings. Merging them into the processing pool...")
                merged = {}
                for job in processed_job_pool:
                    link = job.get("metadata", {}).get("link") or job.get("link") or ""
                    if link:
                        merged[link] = job
                for job in db_jobs:
                    link = job.get("metadata", {}).get("link") or job.get("link") or ""
                    if link:
                        if link not in merged:
                            merged[link] = job
                processed_job_pool = list(merged.values())

        if not processed_job_pool:
            print("[INFO] No scraped jobs available. Attempting to load from DB for Stage 2 (Embed+Extract)...")
            is_reprocess = reprocess
            processed_job_pool = await _load_jobs_for_stage(dp, stage=2, limit=db_limit, force_reprocess=is_reprocess)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        if processed_job_pool:
            # Check user config for Stage 2 extraction LLM
            _user_config = _load_user_config()
            _extraction_llm = _user_config.get("extraction_llm", os.getenv("EXTRACTION_LLM", "lm_studio"))
            _extraction_model = _user_config.get("extraction_model", os.getenv("EXTRACTION_MODEL", "local-model"))
            
            if _extraction_llm == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 2", model_name=_extraction_model)

            processed_job_pool = await pipeline_stage_embed_and_extract(
                jobs=processed_job_pool,
                ai_engine=ai,
                text_processor=tp,
                dp=dp,
                skip_db=skip_db,
                verbose=False,
            )

            if _extraction_llm == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 2", model_name=_extraction_model)
        _log_pipeline_stats("2: Embed+Extract", len(processed_job_pool), source_hint="llm_extraction")
    elif stage_start > 2:
        print(f"[SKIP] Stage 2 (Embed+Extract) — outside range {stage_start}-{stage_end}")

    # Stage 3: Rule Filtering
    if 3 <= stage_end and stage_start <= 3:
        if not processed_job_pool:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 3 (Rule Filter)...")
            processed_job_pool = await _load_jobs_for_stage(dp, stage=3, limit=db_limit)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        if processed_job_pool:
            processed_job_pool = await pipeline_stage_rule_filter(
                jobs=processed_job_pool,
                user_preferences=setup_data.get("user_preferences", {}),
                dp=dp,
                skip_db=skip_db,
            )
        skipped_count = sum(1 for j in processed_job_pool if j.get('skip'))
        _log_pipeline_stats("3: Rule Filter", len(processed_job_pool), skipped_count=skipped_count, source_hint="hard_constraints")
    elif stage_start > 3:
        print(f"[SKIP] Stage 3 (Rule Filter) — outside range {stage_start}-{stage_end}")

    # Stage 4: Archetype Engine Integration
    if 4 <= stage_end and stage_start <= 4:
        if not processed_job_pool:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 4 (Archetype)...")
            processed_job_pool = await _load_jobs_for_stage(dp, stage=4, limit=db_limit)
            print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")

        _load_model_if_lm_studio(ai, stage_label="Stage 4")
        active_jobs, archetype_manager = await pipeline_stage_archetype_integration(
            jobs=processed_job_pool,
            ai_engine=ai,
            dp=dp,
            setup_data=setup_data,
            skip_db=skip_db,
        )
        _unload_model_if_lm_studio(ai, stage_label="Stage 4")
        _log_pipeline_stats("4: Archetypes", len(active_jobs), source_hint="archetype_comparison")
    elif stage_start > 4:
        print(f"[SKIP] Stage 4 (Archetype) — outside range {stage_start}-{stage_end}")
        # Automatically initialize archetype manager if running Stage 5 next
        if stage_start == 5:
            print("[INFO] Initializing archetype manager for Stage 5...")
            _, archetype_manager = await pipeline_stage_archetype_integration(
                jobs=[],
                ai_engine=ai,
                dp=dp,
                setup_data=setup_data,
                skip_db=skip_db,
            )

    # Stage 5: Vector Scoring WITH ARCHETYPES
    if 5 <= stage_end and stage_start <= 5:
        if not active_jobs:
            if not processed_job_pool:
                print("[INFO] No jobs in memory. Attempting to load from DB for Stage 5 (Vector Scoring)...")
                processed_job_pool = await _load_jobs_for_stage(dp, stage=5, limit=db_limit)
                print(f"[INFO] Loaded {len(processed_job_pool)} jobs from DB.")
            active_jobs = [j for j in processed_job_pool if not j.get('skip')]

        if active_jobs and archetype_manager:
            filtered_job_pool = await pipeline_stage_vector_scoring(
                jobs=active_jobs,
                archetype_manager=archetype_manager,
                dp=dp,
                skip_db=skip_db,
            )
        _log_pipeline_stats("5: Vector Scoring", len(filtered_job_pool), source_hint="semantic_threshold")
    elif stage_start > 5:
        print(f"[SKIP] Stage 5 (Vector Score) — outside range {stage_start}-{stage_end}")

    # Stage 6: Cheap LLM Classification
    if 6 <= stage_end and stage_start <= 6:
        if not filtered_job_pool:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 6 (Cheap LLM)...")
            filtered_job_pool = await _load_jobs_for_stage(dp, stage=6, limit=db_limit)
            print(f"[INFO] Loaded {len(filtered_job_pool)} jobs from DB.")

        if filtered_job_pool:
            _user_config = _load_user_config()
            cheap_llm_provider = _user_config.get("cheap_llm_provider", os.getenv("CHEAP_LLM_PROVIDER", "gemini"))
            cheap_llm_model = _user_config.get("cheap_llm_model", os.getenv("CHEAP_LLM_MODEL"))

            if cheap_llm_provider == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 6", model_name=cheap_llm_model)

            shortlisted_jobs = await pipeline_stage_cheap_llm(
                jobs=filtered_job_pool,
                setup_data=setup_data,
                dp=dp,
                skip_db=skip_db,
            )

            if cheap_llm_provider == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 6", model_name=cheap_llm_model)
        _log_pipeline_stats("6: Cheap LLM", len(shortlisted_jobs), source_hint="cheap_llm_classifier")
    elif stage_start > 6:
        print(f"[SKIP] Stage 6 (Cheap LLM) — outside range {stage_start}-{stage_end}")

    # Stage 7: Strong LLM Reranking
    if 7 <= stage_end and stage_start <= 7:
        if not shortlisted_jobs:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 7 (Strong LLM)...")
            shortlisted_jobs = await _load_jobs_for_stage(dp, stage=7, limit=db_limit)
            print(f"[INFO] Loaded {len(shortlisted_jobs)} jobs from DB.")

        if shortlisted_jobs:
            _user_config = _load_user_config()
            strong_llm_provider = _user_config.get("strong_llm_provider", os.getenv("STRONG_LLM_PROVIDER", "gemini"))
            strong_llm_model = _user_config.get("strong_llm_model", os.getenv("STRONG_LLM_MODEL"))

            if strong_llm_provider == "lm_studio":
                _load_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 7", model_name=strong_llm_model)

            deeply_analyzed_jobs = await pipeline_stage_strong_llm(
                jobs=shortlisted_jobs,
                setup_data=setup_data,
                dp=dp,
                skip_db=skip_db,
            )

            if strong_llm_provider == "lm_studio":
                _unload_model_if_lm_studio(ai, provider_name="lm_studio", stage_label="Stage 7", model_name=strong_llm_model)
        _log_pipeline_stats("7: Strong LLM", len(deeply_analyzed_jobs), source_hint="strong_llm_reranker")
    elif stage_start > 7:
        print(f"[SKIP] Stage 7 (Strong LLM) — outside range {stage_start}-{stage_end}")

    # Stage 8: Final Application Queue
    if 8 <= stage_end and stage_start <= 8:
        if not deeply_analyzed_jobs:
            print("[INFO] No jobs in memory. Attempting to load from DB for Stage 8 (Final Queue)...")
            deeply_analyzed_jobs = await _load_jobs_for_stage(dp, stage=8, limit=db_limit)
            print(f"[INFO] Loaded {len(deeply_analyzed_jobs)} jobs from DB.")

        if deeply_analyzed_jobs:
            final_queue = await pipeline_stage_final_queue(
                jobs=deeply_analyzed_jobs,
                dp=dp,
                skip_db=skip_db,
            )
        _log_pipeline_stats("8: Final Queue", len(final_queue), source_hint="final_ranking")
    elif stage_start > 8:
        print(f"[SKIP] Stage 8 (Final Queue) — outside range {stage_start}-{stage_end}")

    # ── Summary and Return logic ──
    if not skip_db and dp:
        try:
            import datetime
            import time
            run_id = f"cli_{int(time.time())}"
            dp.save_token_usage(run_id, datetime.datetime.now(), usage_tracker.all_records)
        except Exception as e:
            print(f"[WARNING] Could not save token usage: {e}")

    if stage_end == 8 and final_queue:
        # Pipeline Summary
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
        print(f"Total jobs processed: {len(processed_job_pool) or len(active_jobs)}")
        print(f"Jobs after semantic filtering: {len(filtered_job_pool)}")
        print(f"Jobs after cheap LLM classification: {len(shortlisted_jobs)}")
        print(f"Jobs after strong LLM reranking: {len(deeply_analyzed_jobs)}")
        apply_jobs_count = sum(1 for j in final_queue if j.get('apply_recommendation', '').lower() == 'apply')
        print(f"Final application queue: {apply_jobs_count} jobs")

        # LLM Usage Summary
        usage_tracker.print_summary()
        return final_queue

    # Summary for partial runs
    print("\n" + "=" * 60)
    print("PIPELINE RUN COMPLETE")
    print(f"Stages executed: {stage_start}-{stage_end}")
    print("=" * 60)

    # LLM Usage Summary
    usage_tracker.print_summary()

    # Return the most relevant result
    if stage_end == 7:
        return deeply_analyzed_jobs
    elif stage_end == 6:
        return shortlisted_jobs
    elif stage_end == 5:
        return filtered_job_pool
    elif stage_end == 4:
        return active_jobs
    else:
        return processed_job_pool


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


# Entry point
if __name__ == "__main__":
    import sys
    import argparse
    import logging

    parser = argparse.ArgumentParser(
        description="Job Scraping and Analysis Pipeline"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=None,
        help="Max pages per URL for Stage 1 scraping (overrides config).",
    )
    parser.add_argument(
        "--visible",
        action="store_true",
        help="Show browser window during scraping (disables headless mode).",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug-level logging.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to a file where logs should be written.",
    )
    parser.add_argument(
        "--skip-db",
        action="store_true",
        help="Skip database persistence for all stages.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output.",
    )
    parser.add_argument(
        "--skip-part-a",
        action="store_true",
        dest="skip_part_a",
        help="Skip Part A (company career-page scraping) in the legacy fallback path and jump straight to description scraping.",
    )
    parser.add_argument(
        "-s", "--stage",
        type=str,
        default="0-8",
        dest="stage_range",
        help="Stage range to run, e.g. '0-8' (all), '2-4', '2' (single stage). Stages: 0=Setup, 1=Scrape, 2=Embed, 3=Filter, 4=Archetype, 5=Vector, 6=CheapLLM, 7=StrongLLM, 8=FinalQueue",
    )
    parser.add_argument(
        "-l", "--limit",
        type=int,
        default=50,
        dest="db_limit",
        help="Max records to pull from database in any bulk-load operation (default: 50).",
    )
    parser.add_argument(
        "--scrape-missing-24h",
        action="store_true",
        dest="scrape_missing_24h",
        help="At Step 1, ONLY try to scrape descriptions for jobs from the previous 24h that do not have them.",
    )
    parser.add_argument(
        "--reprocess",
        action="store_true",
        dest="reprocess",
        help="Force Stage 2 to reprocess/embed jobs that already have embeddings.",
    )

    args = parser.parse_args()

    # Always set root logger level and add logs/app_error.log
    root = logging.getLogger()
    log_level = logging.DEBUG if args.debug else logging.INFO
    root.setLevel(log_level)

    import os
    os.makedirs("logs", exist_ok=True)
    
    error_handler = logging.FileHandler(os.path.join("logs", "app_error.log"), mode="a", encoding="utf-8")
    error_handler.setLevel(logging.ERROR)
    error_formatter = logging.Formatter(
        "%(asctime)s:%(levelname)s:%(message)s"
    )
    error_handler.setFormatter(error_formatter)
    root.addHandler(error_handler)

    # Apply custom log file if specified
    if args.log_file:
        log_file_path = args.log_file
        if not os.path.isabs(log_file_path) and not log_file_path.startswith("logs/"):
            log_file_path = os.path.join("logs", log_file_path)
            
        file_handler = logging.FileHandler(log_file_path, mode="a", encoding="utf-8")
        file_handler.setLevel(log_level)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    import asyncio
    asyncio.run(main(
        scrape_pages=args.pages,
        scrape_visible=args.visible,
        scrape_debug=args.debug,
        skip_part_a=args.skip_part_a,
        stage_range=args.stage_range,
        db_limit=args.db_limit,
        skip_db=args.skip_db,
        scrape_missing_24h=args.scrape_missing_24h,
        reprocess=args.reprocess,
    ))
