# Libraries
import os
from dotenv import load_dotenv
import json
from docx import Document  # type: ignore
import re
import json
from jobspy import scrape_jobs
import csv
import logging
import random
import time
import numpy as np
import itertools
import yaml
from typing import Any

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


############################# Global Variables and Configs ############################
load_dotenv()

# Archetype Definitions
with open(os.getenv("ARCHETYPES_CONFIG", ""), 'r') as file:
    ARCHETYPES_CONFIG = json.load(file)

############################## Error Handlers #########################################

def error_logger_crash(error_msg):
    print(error_msg)
    logging.error(error_msg)
    raise ValueError(error_msg)

def error_logger_continue(error_msg):
    print(error_msg)
    logging.error(error_msg)
    return

############################## Helper Functions #######################################
def load_resume_as_text(type):
    # doc = Document(input("Provide the path for the resume file to use: "))
    doc = Document(os.getenv(type))
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text.strip())
    return "\n".join(full_text)

def scrape_single_job_board(new_data, company_url, company=""):
    company_list = [] # list for all jobs in new_data
    # print(f"new data Status code: {new_data['status_code']} | company: {company}")
    # print(f"Company URL: {company_url} | New Data: {new_data}")
    for item in new_data['data']:
        if type(item) is not dict:
            continue
        if not item.get('title') or item['title'] == [None] or item['title'] == "":
            continue
        link = item.get('link')
        if link and not link.startswith(("http","https")):
            link = f"{company_url}{link}"
        maker = {
            "company": item['company'] if item.get('company') is not None else company,
            "company_url": company_url,
            "title": item['title'],
            "flexibility": item['flexibility'] if item.get('flexibility') is not None else "NA",
            "url": link,
            "source":new_data['source'] if new_data.get('source') else ""
            }
        if item.get('location') is not None:
            if re.search(r"location", item['location'], re.IGNORECASE):
                item['location'] = re.sub(r'location', "", item['location'], flags=re.IGNORECASE)
            maker["location"] = item['location']
        company_list.append(maker)
    # print(f"Company List: {company_list}")
    return company_list
def scrape_multi_job_board(new_data, company_url, company):
    full_list = [] # list for all pages combined together
    # print(f"new data: {new_data}")
    adjusted_nd = {
        "data":[],
        "status":200,
        "success":True,
        "source":new_data['source']
    }
    for item in new_data['data']:
        adjusted_nd['data'].append(item['jobs'])
    # print(f"Adjusted Data: {adjusted_nd}")
    full_list += scrape_single_job_board(adjusted_nd, company_url, company)
        # print(f"page # {page}: {new_data['pages'][page]}")
        # print(f"page: {page} | current Full List: {full_list}\n\n")
    return full_list

async def scrape_sites(i, company_url, dp):
    # print(f"Payload: {i}")
    new_data = await dp.scrape_data(i['strategy'], api_method=i['api_method'])
    new_data['source'] = i['strategy']['source']
    # print(f"New Data source: {new_data['source']}. New Data: {str(new_data)}...")
    # if new_data['data'].get(0) is None:
    #     print(f"New Data: {new_data}")
    if new_data['status_code'] != 200:
        print(f"Scraping data failed. Error code {new_data['status_code']}. Error: {new_data['error'] if new_data.get('error') is not None else 'No error message provided.'}")
        return
    if 'data' not in new_data:
        print(f"Warning: No 'data' key in response from {i['company']}. Response: {new_data}")
        return
    if len(new_data['data']) != 0:
        try:
            
                maker = scrape_multi_job_board(new_data, company_url, i['company']) if new_data['data'][0].get('jobs') is not None else scrape_single_job_board(new_data, company_url, i['company'])
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error: {e}\nNew Data: {new_data}")
    else:
        return
    # print(f"Maker: {maker}")
    return maker

def scrape_jb(sn, l, rw, ho, **kwargs):
    allowed_keys = {
        "linkedin_fetch_description", 
        "country_indeed", 
        "google_search_term", 
        "search_term"
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}
    
    jobs = scrape_jobs(
        site_name=[sn], # "glassdoor", "bayt", "naukri", "bdjobs"
        location=l,
        results_wanted=rw,
        hours_old=ho,
        **filtered_kwargs
    )
    return jobs

def apply_rule_filters(job: dict, user_preferences: dict) -> bool:
    """
    Evaluates a job against hard constraints.
    Returns True if the job should be SKIPPED, False if it passes.
    """
    job_work_type = job['features'].get('work_type', None).lower()
    user_work_types = [t.lower() for t in user_preferences.get('work_types', [])]
    if user_work_types and job_work_type and job_work_type not in user_work_types:
        return True

    job_seniority = job['features'].get('seniority', None).lower()
    user_seniority_levels = [s.lower() for s in user_preferences.get('seniority_levels', [])]
    if user_seniority_levels and job_seniority and job_seniority not in user_seniority_levels:
        return True

    job_pay = job['features'].get('pay', "")
    target_pay_range = user_preferences.get('pay_range', '')
    if not job_pay or job_pay == "Not Specified":
        pass
    else:
        if target_pay_range and job_pay:
            try:
                if filter_pay(target_pay_range, job_pay):
                    return True                            
            except (ValueError, IndexError):
                # If parsing fails for some reason, allow the job to continue processing
                error_logger_continue(f"Error in parsing pay range for job {job['id']}: {ValueError} at {IndexError}")
                pass

    # 4. Timezone Filter
    job_timezone = job['features'].get('timezone', None).lower()
    user_timezones = [tz.lower() for tz in user_preferences.get('timezones', [])]
    
    if user_timezones and job_timezone and job_timezone not in user_timezones:
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
        matches = re.findall(r'(\d+)(?:k|K)?', job_pay)
        if matches:
            job_min = int(matches[0]) * 1000 if 'k' in job_pay.lower() else int(matches[0])
        if len(matches) >= 2:
            job_max = int(matches[1]) * 1000 if 'k' in job_pay.lower() else int(matches[1])

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

    The cache stores a JSON file containing the extracted data and a timestamp.
    If the source file has not been modified since the cache timestamp, the cached
    data is returned. Otherwise, extraction is re-run and the cache is updated.

    Args:
        ai: The AIEngine instance used for LLM extraction.
        source_path: Path to the .docx source file (used for mtime check).
        raw_text: The already-loaded text content of the file.
        cache_path: Path to the JSON cache file.

    Returns:
        dict with keys 'skills' (list), 'requirements' (list), 'summary' (str).
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

    print(f"Extracting profile data from {source_path}...")
    ai_data = call_llm_for_extraction(ai, raw_text, provider_name=os.getenv("EXTRACTION_LLM"))
    
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

##################################### Main ######################################################################################

async def main():
    # === Initial Setup ===
    # Load .env variables
    resume = load_resume_as_text("RESUME")
    user_profile = load_resume_as_text("PROFILE")

    # Create Data Puller Object
    dp = DataPuller(
        dbname = os.getenv("DB_NAME", ""),
        user =os.getenv("DB_USER", ""),
        password = os.getenv("DB_PASSWORD", ""),
        host = os.getenv("DB_HOST", "localhost"),
        port = os.getenv("DB_PORT", "5432")
    )

    # Get sites file from .env and pulls the sites in. Needs to be a csv set up with name and site columns
    """
    JOB_SITES should point to a csv file formatted as:
    name,site
    """
    
    sites_file = os.getenv("JOB_SITES","")
    sites = dp.load_sites_list(sites_file)
    print("Sites Retrieved")

    # Extract skills and job titles from profile
    tp = TextProcessor()
    ai = AIEngine(default_provider_name="lm_studio") 
    skills_raw = tp.get_section_content(user_profile, "Skills")
    titles_raw = tp.get_section_content(user_profile, "Job Titles")
    skills = tp.clean_list_from_text(skills_raw)
    job_titles = tp.clean_list_from_text(titles_raw)

    # Debugging output to verify extraction
    print(f"--- Profile Extraction ---")
    print(f"Extracted {len(skills)} skills: {skills}")
    print(f"Extracted {len(job_titles)} job titles: {job_titles}")
    print(f"---------------------------\n")

    # Configure logging
    logging.basicConfig(
        filename='app_error.log', 
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )
    
    # === Scrape from Company Boards ===

    # Pull in the site strategies based on the sites pulled from the sites file
    site_strategies = []
    data = []
    for i in range(len(sites['name'])):
        strategy = {
            "company": sites['name'][i],
            "site": sites['site'][i],
            "strategy": dp.load_site_strategies(f"./site_strategies/{sites['name'][i]}.json"),
            "api_method": "",
        }

        strategy['api_method'] = "extract-paginated" if strategy['strategy'].get('pagination') is not None else ("extract-js" if strategy['strategy'].get("js_config") is not None else "extract")  
        print(f"Company: {strategy['company']} | API method: {strategy['api_method']}")
        site_strategies.append(strategy)
    print("Site strategies loaded.")

    # Scrape the jobs from the sites using the link to the sites and the attached strategy. Fields should be set to return matching amount of records.
    for i in site_strategies:
        company_url = i['strategy'].pop('company_url', None)
        print(f"Scraping {i['company']}")
        if isinstance(i['strategy']['url'],str):
            d = await scrape_sites(i, company_url, dp)
            if not d:
                continue
            if isinstance(d, list):
                data += d
            else:
                data.append(d)
        elif isinstance(i['strategy']['url'],list):
            for j in i['strategy']['url']:
                new_payload = i
                new_payload['strategy']['url'] = j
                d = await scrape_sites(new_payload, company_url, dp)
                if not d:
                    continue
                if isinstance(d, list):
                    data += d
                else:
                    data.append(d)
    print(f"Total jobs scraped: {len(data)}")

    dp.load_scraped_data_to_db(data)

    del data, site_strategies, sites, sites_file, strategy

    # === Run Searches on Job Boards ===
   
    job_board_list = ["indeed", "linkedin", "zip_recruiter", "google"]

    # Build proxy list

    # Build search terms
    search_terms_file = os.getenv("SEARCH_TERMS", "")

    if not search_terms_file:
        error_logger_crash("Error: SEARCH_TERMS environment variable is empty or not provided.")

    try:
        search_terms = []
        with open(search_terms_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:
                    search_terms.append(row[0])
        
        if not search_terms:
            error_logger_crash("The provided CSV file is empty.")
            
    except Exception as e:
        error_logger_crash(f"Failed to load search terms from {search_terms_file}: {e}")
   
    # loop over site list
    prefs_yaml_path = os.getenv("USER_PREFERENCES_YAML", "")
    if prefs_yaml_path:
        with open(prefs_yaml_path, 'r') as f:
            user_preferences = yaml.safe_load(f)
    else:
        user_preferences = {}
    
    jobs = []
    requests_wanted = 200
    target_cities = user_preferences.get('target_cities', [])
    
    delay = {
        "min": 1,
        "max": 4
    }

    for board, st, location in itertools.product(job_board_list, search_terms, target_cities):
        # 1. Build kwargs based on board type
        kwa = {}
        if board == "google":
            kwa["google_search_term"] = st
        elif board == "indeed":
            kwa["search_terms"] = st
            kwa["country_indeed"] = "USA"
        elif board == "linkedin":
            kwa["search_terms"] = st
            kwa["linkedin_fetch_description"] = True
        else:
            kwa["search_terms"] = st

        # 2. Fetch and process jobs in a single loop
        for i in scrape_jb(board, location, requests_wanted, 24, **kwa):
            if isinstance(i, dict):
                job = {
                    "source": i.get('site'),
                    "title": i.get('title'),
                    "link": i.get('job_url'),
                    "company": i.get('company'),
                    "pay": f"{i.get('min_amount', '')} - {i.get('max_amount', '')} {i.get('interval', '')}".strip(),
                    "description": i.get('description'),
                    "city": i.get('city'),
                    "state": i.get('state')
                }
            else:
                job = {"source": None, "title": None, "link": None, "company": None, "pay": "", "description": None, "city": None, "state": None}
                
            jobs.append(job)

    time.sleep(random.uniform(delay['min'], delay['max']))

    # Load into db
    dp.load_scraped_data_to_db(jobs)

    # === EMBEDDING GENERATION (Constraint: Only new jobs should be embedded) ===
    pull_jobs_sql = """
    SELECT j.id, j.job_name, j.description, j.seniority, j.pay_range, j.timezone
    FROM job j
    LEFT JOIN job_embeddings je ON j.id = je.job_id
    WHERE je.job_id IS NULL;
    """

    todays_jobs = dp.pull_data_db(pull_jobs_sql)
    processed_job_pool: list[Any] = []
    if todays_jobs is None:
        todays_jobs = []

    print(f"Starting regex extraction on {len(todays_jobs)} jobs...")

    for row in todays_jobs:
        raw_job = {
            "id": row['id'],
            "title": row['job_name'],
            "description": row['description'],
            "seniority": row['seniority'],
            "pay_range": row['pay_range'],
            "timezone": row['timezone']
        }
        if not raw_job["description"] or len(raw_job["description"]) < 50:
            error_logger_continue(f"Warning: insufficient or missing description for job ID {raw_job['id']}")
            continue

        # Deterministic Extraction 
        work_type = tp.detect_work_type(raw_job["description"])
        seniority = tp.detect_seniority(raw_job["description"])
        salary = raw_job['pay_range'] if raw_job['pay_range'] else tp.extract_salary(raw_job["description"])
        timezone = raw_job['timezone'] if raw_job['timezone'] else tp.detect_timezone(raw_job["description"])

        extracted_data = {
            "metadata": {
                "job_id": raw_job["id"],
                "source": "sql_pull"
            },
            "features": {
                "title": raw_job["title"],
                "description": raw_job["description"],
                "pay": salary,
                "seniority": seniority,
                "work_type": work_type,
                "timezone": timezone
            },
            "embeddings": {
                "description_vector": None, 
                "skills_vector": None
            }
        }

        processed_job_pool.append(extracted_data)

    print(f"Successfully extracted data for {len(processed_job_pool)} jobs.")
    print(f"Sample Job: {processed_job_pool[0]['features'] if processed_job_pool else 'No jobs found'}")

    print(f"Starting AI/LLM Pass on {len(processed_job_pool)} jobs...")

    for index, job in enumerate(processed_job_pool):
        print(f"Processing job {index + 1}/{len(processed_job_pool)}...")
        
        if not job or 'features' not in job:
            error_logger_continue(f"Warning: job at index {index} has invalid structure")
            continue
            
        description = job['features'].get('description', '')
        if not description:
            error_logger_continue(f"Warning: job at index {index} has no description")
            continue

        # Extract skills, requirements and pass summary
        ai_data = call_llm_for_extraction(ai, description, provider_name=os.getenv("EXTRACTION_LLM"))
        
        skills = []
        requirements = []
        summary = ""

        if isinstance(ai_data, dict):
            skills = ai_data.get('skills', [])
            requirements = ai_data.get('requirements', [])
            summary = ai_data.get('summary', "")
        elif isinstance(ai_data, str):
            try:
                parsed = json.loads(ai_data)
                if isinstance(parsed, dict):
                    skills = parsed.get('skills', [])
                    requirements = parsed.get('requirements', [])
                    summary = parsed.get('summary', "")
            except Exception:
                # leave defaults
                pass

        features = job['features']
        features['skills'] = skills or []
        features['requirements'] = requirements or []
        features['summary'] = summary or ""

        # Vector Generation
        
        title_text = features['title']
        if title_text:
            job['embeddings']['title_vector'] = generate_embeddings(ai, title_text, provider_name=os.getenv("EMBEDDINGS_LLM"))
        else:
            job['embeddings']['title_vector'] = []

        skills_text = ", ".join(features['skills'])
        if skills_text:
            job['embeddings']['skills_vector'] = generate_embeddings(ai, skills_text, provider_name=os.getenv("EMBEDDINGS_LLM"))
        else:
            job['embeddings']['skills_vector'] = []

        requirements_text = ", ".join(features['requirements'])
        if requirements_text:
            job['embeddings']['requirements_vector'] = generate_embeddings(ai, requirements_text, provider_name=os.getenv("EMBEDDINGS_LLM"))
        else:
            job['embeddings']['requirements_vector'] = []

        summary_text = features['summary']
        if summary_text:
            job['embeddings']['description_vector'] = generate_embeddings(ai, summary_text, provider_name=os.getenv("EMBEDDINGS_LLM"))
        else:
            job['embeddings']['description_vector'] = []

        time.sleep(1) 

    print("AI/LLM Pass Complete.")

    # Persist embedding vectors to database
    print(f"Saving embeddings to database...")
    embedding_updates = []
    for job in processed_job_pool:
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
        print(f"Saved {len(embedding_updates)} jobs' embeddings to 'job_embeddings'.")
    else:
        print("No embeddings to save.")
    
    # Update job records in database with extracted metadata
    print("Updating job records in database with metadata...")
  
    job_updates = []
    for job in processed_job_pool:
        update_data = {
            "id": job['metadata']['job_id'],
            "pay_range": job['features'].get('pay'),
            "seniority": job['features'].get('seniority'),
            "work_type": job['features'].get('work_type'), 
            "timezone": job['features'].get('timezone')
        }
        job_updates.append(update_data)
    
    dp.update_job_metadata(job_updates)
    print(f"Successfully updated {len(job_updates)} job records with metadata.")

    # === Rule Filtering ===

    print(f"Starting Rule-Based Filtering on {len(processed_job_pool)} jobs...")

    for job in processed_job_pool:
        job['skip'] = apply_rule_filters(job, user_preferences)
    print(f"Filtering complete. {sum(1 for j in processed_job_pool if j['skip'])} jobs skipped.")

    # === DATABASE SYNC: Update SQL with Skip Status ===
    print("Syncing skip status to database...")
    
    skipped_ids = [job['metadata']['job_id'] for job in processed_job_pool if job.get('skip')]
    if skipped_ids:
        dp.bulk_update_skip_status(skipped_ids)
        print(f"Successfully updated {len(skipped_ids)} jobs in database as 'skipped'.")
    else:
        print("No jobs to skip in database.")
    active_job_pool = [j for j in processed_job_pool if not j.get('skip')]
    
    print(f"Proceeding to Vector Scoring with {len(active_job_pool)} active jobs.")

    # === Archetype Engine Integration ===
    
    archetype_manager = ArchetypeManager()
    print("Synchronizing candidate archetypes and benchmarks...")
    
    # Extract structured profile data using LLM with file-modification caching
    resume_data = extract_and_cache_profile(
        ai,
        os.getenv("RESUME", ""),
        resume,
        "archetype_profiles/resume_cache.json"
    )
    profile_data = extract_and_cache_profile(
        ai,
        os.getenv("PROFILE", ""),
        user_profile,
        "archetype_profiles/user_profile_cache.json"
    )

    all_archetypes = ARCHETYPES_CONFIG + [
        {
            "name": "Resume",
            "title": ", ".join(resume_data.get("skills", [])),
            "skills": "\n".join(resume_data.get("skills", [])),
            "responsibilities": "\n".join(resume_data.get("requirements", [])),
            "summary": resume_data.get("summary", ""),
            "type": "resume"
        },
        {
            "name": "User Profile",
            "title": ", ".join(profile_data.get("skills", [])),
            "skills": "\n".join(profile_data.get("skills", [])),
            "responsibilities": "\n".join(profile_data.get("requirements", [])),
            "summary": profile_data.get("summary", ""),
            "type": "user_profile"
        }
    ]

    for arch_config in all_archetypes:
        cached_arch = dp.get_archetype_embeddings(arch_config['name'])
        
        if cached_arch:
            archetype_manager.add_archetype(Archetype(
                name=arch_config['name'],
                type=cached_arch['archetype_type'],
                title_embedding=np.array(cached_arch['title_embedding']),
                skills_embedding=np.array(cached_arch['skills_embedding']),
                responsibilities_embedding=np.array(cached_arch['responsibilities_embedding']),
                metadata=cached_arch.get('metadata', {})
            ))
        else:
            print(f"Generating new embeddings for archetype: {arch_config['name']}")
            new_arch = archetype_manager.load_archetype(
                name=arch_config['name'],
                archetype_data=arch_config,
                archetype_type=arch_config.get("type", "benchmark")
            )
            # Persist to DB for future runs
            dp.save_archetype_embeddings({
                "archetype_name": new_arch.name,
                "archetype_type": new_arch.type,
                "title_embedding": new_arch.title_embedding.tolist() if new_arch.title_embedding is not None else None,
                "skills_embedding": new_arch.skills_embedding.tolist() if new_arch.skills_embedding is not None else None,
                "responsibilities_embedding": new_arch.responsibilities_embedding.tolist() if new_arch.responsibilities_embedding is not None else None,
                "metadata": json.dumps(new_arch.metadata)
            })
    
    print(f"Loaded {len(archetype_manager.archetypes)} archetypes for comparison.")

    # === Vector Scoring with Archetypes ===

    print("Comparing jobs to archetypes...")
    
    for index, job in enumerate(active_job_pool):
        print(f"Processing archetype comparison for job {index + 1}/{len(active_job_pool)}...")
        
        # Compare the current job against all archetypes
        matches = archetype_manager.compare_job_to_archetypes(job)
        
        # Generate retrieval metadata from archetype comparisons
        job['retrieval_metadata'] = archetype_manager.generate_retrieval_metadata(job, matches)
        
        # Add archetype comparison data to job for later use in ranking
        job['archetype_matches'] = matches

    print("Archetype comparison complete.")

    print("Ranking jobs with archetype-based scoring...")
    for job in active_job_pool:
        match_score = job['retrieval_metadata'].get('scores', [])
        if match_score:
            print(f"Job '{job['features']['title']}' has archetype matches: {[m['name'] + ':' + str(m['score']) for m in match_score]}")

    # Import adjustment functions from vector_engine
    from app.vector_engine import apply_keyword_adjustments, apply_metadata_adjustments
    
    print("Applying weighted semantic scoring...")
    for job in active_job_pool:
        matches = job.get('archetype_matches', [])
        if not matches:
            continue
            
        # Use the best matching archetype for scoring
        best_match = matches[0]  # Already sorted by highest score
        
        # Extract individual similarity scores from Stage 5C comparison
        title_similarity = best_match.get('title_similarity', 0.0)
        skills_similarity = best_match.get('skills_similarity', 0.0)
        responsibility_similarity = best_match.get('responsibility_similarity', 0.0)
        
        # Step 1.1: Compute weighted semantic score (Stage 5D)
        semantic_score = (
            0.40 * title_similarity +
            0.35 * skills_similarity +
            0.25 * responsibility_similarity
        )
        
        # Step 2.1: Apply keyword adjustments (Stage 5E)
        job_skills = job.get('features', {}).get('skills', [])
        job_title = job.get('features', {}).get('title', '')
        semantic_score = apply_keyword_adjustments(semantic_score, job_skills, job_title)
        
        # Step 2.2: Apply metadata adjustments (Stage 5F)
        # Build job metadata for adjustments
        job_meta = {
            "is_remote": job.get('features', {}).get('work_type', '').lower() == 'remote',
            "salary": 0,  # Will be parsed from pay_range if available
            "days_old": 30  # Default, would be calculated from job posting date
        }
        
        # Parse salary from pay_range if available
        pay_range = job.get('features', {}).get('pay', '')
        if pay_range:
            import re as _re
            numbers = _re.findall(r'\d+(?:,\d+)?', pay_range.replace(',', ''))
            if len(numbers) >= 2:
                job_meta["salary"] = int(numbers[1])  # Use max of range
            elif len(numbers) == 1:
                job_meta["salary"] = int(numbers[0])
        
        semantic_score = apply_metadata_adjustments(semantic_score, job_meta)
        
        # Step 1.2: Clamp and normalize (Stage 5G)
        semantic_score = max(0.0, min(1.0, semantic_score))
        score_percent = int(round(semantic_score * 100))
        
        # Store all scoring data in job object
        job['semantic_score'] = semantic_score
        job['semantic_score_percent'] = score_percent
        job['title_similarity'] = title_similarity
        job['skills_similarity'] = skills_similarity
        job['responsibility_similarity'] = responsibility_similarity
        job['adjusted_score'] = semantic_score  # After adjustments, before clamping was same
        job['best_archetype'] = best_match.get('archetype_name', '')

    print("Generating detailed ranking metadata...")
    for job in active_job_pool:
        if 'retrieval_metadata' not in job:
            job['retrieval_metadata'] = {}
        if 'semantic_score' in job:
            job['retrieval_metadata']['semantic_score'] = job['semantic_score']
            job['retrieval_metadata']['semantic_score_percent'] = job['semantic_score_percent']
            job['retrieval_metadata']['best_archetype'] = job.get('best_archetype', '')

    # Phase 3: Filtering & Persistence (Stage 5I)
    print("Applying semantic score threshold filtering (Stage 5I)...")
    MIN_SCORE_THRESHOLD = 0.72
    TARGET_COUNT = 20  # Optional fallback target
    
    filtered_job_pool = []
    for job in active_job_pool:
        if job.get('semantic_score', 0) >= MIN_SCORE_THRESHOLD:
            filtered_job_pool.append(job)
    
    # Optional: Add top-X% fallback if not enough jobs meet threshold
    if len(filtered_job_pool) < TARGET_COUNT:
        sorted_jobs = sorted(active_job_pool, key=lambda x: x.get('semantic_score', 0), reverse=True)
        top_n = max(TARGET_COUNT - len(filtered_job_pool), 1)
        # Add jobs that aren't already in filtered_pool
        for job in sorted_jobs[:top_n]:
            if job not in filtered_job_pool:
                filtered_job_pool.append(job)
    
    print(f"Filtered job pool from {len(active_job_pool)} to {len(filtered_job_pool)} jobs based on semantic score threshold (>= {MIN_SCORE_THRESHOLD}).")

    # Phase 3: Persist results to database (Stage 5J)
    print("Persisting vector scores to database...")
    try:
        # Use PostgreSQL (matching the existing database setup)
        from app.postgres_mgr import PostgresManager
        
        # Get DB connection params from env
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "")
        db_name = os.getenv("DB_NAME", "web_scraper_db")
        
        # Connect to database
        pg_mgr = PostgresManager(db_host, int(db_port), db_user, db_password)
        pg_mgr.connect(db_name)
        
        # Create vector_scores table if it doesn't exist
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
        pg_mgr.execute_sql(create_table_sql, dbname=db_name)
        
        # Create index if it doesn't exist (separate statement)
        create_index_sql = """
        CREATE INDEX IF NOT EXISTS idx_vs_job_id ON vector_scores(job_id)
        """
        pg_mgr.execute_sql(create_index_sql, dbname=db_name)
        
        # Insert ranked results
        for rank, job in enumerate(filtered_job_pool, start=1):
            insert_sql = """
                INSERT INTO vector_scores (job_id, archetype_name, semantic_score, 
                                           title_similarity, skills_similarity, 
                                           responsibility_similarity, adjusted_score, rank)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            pg_mgr.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                job.get('best_archetype', ''),
                job.get('semantic_score', 0),
                job.get('title_similarity', 0),
                job.get('skills_similarity', 0),
                job.get('responsibility_similarity', 0),
                job.get('adjusted_score', 0),
                rank
            ), dbname=db_name)
        
        pg_mgr.close()
        print(f"Successfully persisted {len(filtered_job_pool)} vector scores to database.")
        
    except Exception as e:
        print(f"Error persisting vector scores: {e}")
        logging.error(f"Vector score persistence failed: {e}")

    print("\n=== Semantic Score-Based Job Shortlisting ===")
    # Sort by semantic score for display
    sorted_filtered = sorted(filtered_job_pool, key=lambda x: x.get('semantic_score', 0), reverse=True)
    top_jobs = sorted_filtered[:5]
    for i, job in enumerate(top_jobs):
        print(f"{i+1}. {job['features']['title']}")
        print(f"   Semantic Score: {job.get('semantic_score_percent', 'N/A')}%")
        print(f"   Best archetype match: {job.get('best_archetype', 'None')}")
        print(f"   Title similarity: {job.get('title_similarity', 0):.3f}")
        print(f"   Skills similarity: {job.get('skills_similarity', 0):.3f}")
        print(f"   Responsibility similarity: {job.get('responsibility_similarity', 0):.3f}")
        print()

    # =====================================================
    # STAGE 6: CHEAP LLM CLASSIFICATION
    # =====================================================
    print("\n" + "="*60)
    print("STAGE 6: CHEAP LLM CLASSIFICATION")
    print("="*60)
    
    # Initialize cheap LLM classifier
    # Use environment variable to select provider: "gemini", "openai", or "lm_studio"
    cheap_llm_provider = os.getenv("CHEAP_LLM_PROVIDER", "gemini")
    cheap_classifier = CheapLLMClassifier(provider=cheap_llm_provider)
    
    # Run Stage 6 on filtered job pool
    shortlisted_jobs = await process_stage_6(
        jobs=filtered_job_pool,
        classifier=cheap_classifier,
        candidate_profile=user_profile,
        candidate_skills=skills,
        batch_size=5
    )
    
    # Persist Stage 6 results to database
    print("Persisting Stage 6 results to database...")
    try:
        from app.postgres_mgr import PostgresManager
        pg_mgr = PostgresManager(db_host, int(db_port), db_user, db_password)
        pg_mgr.connect(db_name)
        
        # Create cheap_llm_results table
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
        pg_mgr.execute_sql(create_table_sql, dbname=db_name)
        
        # Insert results
        for job in shortlisted_jobs:
            cheap_result = job.get('cheap_llm_result', {})
            insert_sql = """
                INSERT INTO cheap_llm_results (job_id, fit_score, decision, strengths, concerns)
                VALUES (%s, %s, %s, %s, %s)
            """
            pg_mgr.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                cheap_result.get('fit_score', 50),
                cheap_result.get('decision', 'maybe'),
                json.dumps(cheap_result.get('strengths', [])),
                json.dumps(cheap_result.get('concerns', []))
            ), dbname=db_name)
        
        pg_mgr.close()
        print(f"Persisted {len(shortlisted_jobs)} cheap LLM results to database.")
    except Exception as e:
        print(f"Error persisting Stage 6 results: {e}")
        logging.error(f"Stage 6 persistence failed: {e}")
    
    # =====================================================
    # STAGE 7: STRONG LLM RERANKING
    # =====================================================
    print("\n" + "="*60)
    print("STAGE 7: STRONG LLM RERANKING")
    print("="*60)
    
    # Initialize strong LLM reranker
    # Use environment variable to select provider: "claude", "openai", or "gemini"
    strong_llm_provider = os.getenv("STRONG_LLM_PROVIDER", "claude")
    strong_reranker = StrongLLMReranker(provider=strong_llm_provider)
    
    # Configure how many jobs to deeply analyze (top N from Stage 6)
    top_n_for_deep_analysis = int(os.getenv("TOP_N_DEEP_ANALYSIS", "15"))
    
    # Run Stage 7 on top candidates from Stage 6
    deeply_analyzed_jobs = await process_stage_7(
        jobs=shortlisted_jobs,
        reranker=strong_reranker,
        candidate_profile=user_profile,
        candidate_skills=skills,
        top_n=top_n_for_deep_analysis
    )
    
    # Persist Stage 7 results to database
    print("Persisting Stage 7 results to database...")
    try:
        pg_mgr = PostgresManager(db_host, int(db_port), db_user, db_password)
        pg_mgr.connect(db_name)
        
        # Create strong_llm_results table
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
        pg_mgr.execute_sql(create_table_sql, dbname=db_name)
        
        # Insert results
        for job in deeply_analyzed_jobs:
            strong_result = job.get('strong_llm_result', {})
            insert_sql = """
                INSERT INTO strong_llm_results (job_id, final_score, priority, 
                                               apply_recommendation, red_flags, 
                                               tailoring_notes, recruiter_bait_likelihood,
                                               detailed_fit_analysis)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            pg_mgr.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                strong_result.get('final_score', 50),
                strong_result.get('priority', 'medium'),
                strong_result.get('apply_recommendation', 'maybe'),
                json.dumps(strong_result.get('red_flags', [])),
                json.dumps(strong_result.get('tailoring_notes', [])),
                strong_result.get('recruiter_bait_likelihood', 'medium'),
                strong_result.get('detailed_fit_analysis', '')
            ), dbname=db_name)
        
        pg_mgr.close()
        print(f"Persisted {len(deeply_analyzed_jobs)} strong LLM results to database.")
    except Exception as e:
        print(f"Error persisting Stage 7 results: {e}")
        logging.error(f"Stage 7 persistence failed: {e}")
    
    # =====================================================
    # STAGE 8: FINAL APPLICATION QUEUE
    # =====================================================
    print("\n" + "="*60)
    print("STAGE 8: FINAL APPLICATION QUEUE")
    print("="*60)
    
    # Run Stage 8 to generate final ranked queue
    final_queue = await process_stage_8(jobs=deeply_analyzed_jobs)
    
    # Persist final queue to database
    print("Persisting final application queue to database...")
    try:
        pg_mgr = PostgresManager(db_host, int(db_port), db_user, db_password)
        pg_mgr.connect(db_name)
        
        # Create final_application_queue table
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
        pg_mgr.execute_sql(create_table_sql, dbname=db_name)
        
        # Insert final queue with positions
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
            pg_mgr.execute_sql(insert_sql, params=(
                job['metadata']['job_id'],
                job.get('final_score', 0),
                job.get('priority', 'medium'),
                job.get('apply_recommendation', 'maybe'),
                position
            ), dbname=db_name)
        
        pg_mgr.close()
        print(f"Persisted final queue with {len(final_queue)} jobs to database.")
    except Exception as e:
        print(f"Error persisting final queue: {e}")
        logging.error(f"Final queue persistence failed: {e}")
    
    # Print detailed final queue
    print("\n" + "="*60)
    print("DETAILED FINAL APPLICATION QUEUE")
    print("="*60)
    
    for i, job in enumerate(final_queue[:10], 1):  # Show top 10
        features = job.get('features', {})
        cheap_result = job.get('cheap_llm_result', {})
        strong_result = job.get('strong_llm_result', {})
        
        print(f"\n{i}. {features.get('title', 'Unknown')}")
        print(f"   Priority: {job.get('priority', 'unknown').upper()}")
        print(f"   Final Score: {job.get('final_score', 0):.1f}/100")
        print(f"   Recommendation: {job.get('apply_recommendation', 'maybe').upper()}")
        print(f"   Semantic Score: {job.get('semantic_score', 0)*100:.1f}%")
        print(f"   Cheap LLM Fit: {cheap_result.get('fit_score', 0)}/100")
        print(f"   Strong LLM Score: {strong_result.get('final_score', 0)}/100")
        print(f"   Salary: {features.get('pay', 'Not specified')}")
        print(f"   Work Type: {features.get('work_type', 'Unknown')}")
        
        if cheap_result.get('strengths'):
            print(f"   Strengths: {', '.join(cheap_result.get('strengths', [])[:3])}")
        if strong_result.get('tailoring_notes'):
            print(f"   Tailoring: {strong_result.get('tailoring_notes', ['None'])[0]}")
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Total jobs processed: {len(processed_job_pool)}")
    print(f"Jobs after semantic filtering: {len(filtered_job_pool)}")
    print(f"Jobs after cheap LLM classification: {len(shortlisted_jobs)}")
    print(f"Jobs after strong LLM reranking: {len(deeply_analyzed_jobs)}")
    print(f"Final application queue: {len(final_queue)} jobs")


# Entry point
if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
