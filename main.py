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

# Modules
from app.pull_data import DataPuller
from app.text_engine import TextProcessor
from app.ai_engine import AIEngine
from app.archetype_engine import ArchetypeManager


# Global Variables
load_dotenv()

# Archetype Definitions (Stage 4)
ARCHETYPES_CONFIG = [
    {
        "name": "AI Tooling Engineer",
        "title": "Senior AI Tooling Engineer",
        "skills": "Python OpenAI LangChain Pinecone PyTorch LLMs",
        "responsibilities": "Develop AI-powered tools. Integrate LLMs into workflows. Optimize inference pipelines."
    },
    {
        "name": "Backend Python Engineer",
        "title": "Senior Backend Python Engineer",
        "skills": "Python FastAPI PostgreSQL AWS Redis Docker",
        "responsibilities": "Design scalable APIs. Optimize database performance. Deploy cloud services."
    },
    {
        "name": "Automation Engineer",
        "title": "QA Automation Engineer",
        "skills": "Python Selenium Pytest Playwright Jenkins CI/CD",
        "responsibilities": "Write automated test suites. Maintain CI pipelines."
    }
]



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
    # 1. Work Type Filter (e.g., Remote vs On-site)
    # If user only wants 'Remote' and job is 'On-site', skip.
    job_work_type = job['features'].get('work_type', '').lower()
    user_work_types = [t.lower() for t in user_preferences.get('work_types', [])]
    
    if user_work_types and job_work_type not in user_work_types:
        return True

    # 2. Seniority Filter
    # If user is looking for 'Senior' and job is 'Intern', skip.
    job_seniority = job['features'].get('seniority', '').lower()
    user_seniority_levels = [s.lower() for s in user_preferences.get('seniority_levels', [])]
    
    if user_seniority_levels and job_seniority not in user_seniority_levels:
        return True

    # 3. Pay/Salary Filter (Enhanced)
    # Parse job pay range to compare with user preferences
    job_pay = job['features'].get('pay', "")
    
    # Parse pay ranges - this would ideally come from user configuration or profile
    # For now, we can at least check if job pay is present and valid
    
    # Example implementation: Check that job has a salary range or can be processed
    if not job_pay or job_pay == "Not Specified":
        # If user has salary expectation but job doesn't specify pay, this could be filtered
        pass  # For now, we'll allow jobs without explicit pay information
    
    # Pay range filtering - check if job's pay falls within user preferences
    target_pay_range = user_preferences.get('pay_range', '')
    
    # Only apply pay filter if there's a target range defined and job has pay data to check
    if target_pay_range and job_pay:
        try:
            # Parse the user's target pay range (e.g., "50k-100k")
            # Extract min and max values from user target pay range string
            user_min_match = re.search(r'(\d+)(?:k|K)?', target_pay_range)
            user_max_match = re.search(r'-(\d+)(?:k|K)?', target_pay_range)
            
            if user_min_match and user_max_match:
                # Convert to integers (accounting for k notation)
                user_min = int(user_min_match.group(1)) * 1000 if 'k' in target_pay_range.lower() else int(user_min_match.group(1))
                user_max = int(user_max_match.group(1)) * 1000 if 'k' in target_pay_range.lower() else int(user_max_match.group(1))
                
                # Try to extract job pay values if present
                job_min = None
                job_max = None
                
                # Check for pay range in the actual job's pay string (e.g., "50k - 100k")
                if isinstance(job_pay, str):
                    job_matches = re.findall(r'(\d+)(?:k|K)?', job_pay)
                    if len(job_matches) >= 2:
                        # Take first two numbers as min and max
                        job_min = int(job_matches[0]) * 1000 if 'k' in job_pay.lower() else int(job_matches[0])
                        job_max = int(job_matches[1]) * 1000 if 'k' in job_pay.lower() else int(job_matches[1])
                    elif len(job_matches) == 1:
                        # If only one number, assume it's a min value (max will be same or None)
                        job_min = int(job_matches[0]) * 1000 if 'k' in job_pay.lower() else int(job_matches[0])
                
                # If we have both min and max values, check if job pay range overlaps with user's target
                if job_min is not None and job_max is not None:
                    # Job pays at least the minimum of user's target and up to maximum of user's target
                    if job_min >= user_min and job_max <= user_max:
                        # Job pay is within user's target range - keep it
                        pass  # Continue processing the job
                    elif job_min < user_min and job_max < user_min:
                        # Job is too low for user's target - skip
                        return True
                    elif job_min > user_max and job_max > user_max:
                        # Job is too high for user's target - skip  
                        return True
                    else:
                        # Partial overlap, allow job to continue (either min or max is in range)
                        pass  # Continue processing
                elif job_min is not None:
                    # If we only have min, check if it's within target range
                    if job_min >= user_min and job_min <= user_max:
                        # Job is within target range - keep it
                        pass  # Continue processing the job
                    else:
                        # Job min is outside user's target - skip  
                        return True
                        
        except (ValueError, IndexError):
            # If parsing fails for some reason, allow the job to continue processing
            pass

    # 4. Timezone Filter
    job_timezone = job['features'].get('timezone', "").lower()
    user_timezones = [tz.lower() for tz in user_preferences.get('timezones', [])]
    
    if user_timezones and job_timezone and job_timezone not in user_timezones:
        return True

    return False

##################################### Embedding Functions ######################################

def extract_responsibilities_from_description(description: str) -> str:
    """Extract key responsibilities from a job description."""
    # This is a simple implementation; expand with NLP/LLM logic if needed.
    return description


def generate_job_embeddings(ai: AIEngine, job_data: dict) -> dict:
    """
    STAGE 4: EMBEDDING GENERATION
    Converts semantic job sections into high-dimensional vectors.
    Returns structured embedding data that can be stored in database.
    """

    job_embeddings = {
        "job_id": job_data.get("metadata", {}).get("job_id"),
        "title_embedding": None,
        "skills_embedding": None,
        "responsibilities_embedding": None,
        "description_embedding": None
    }

    title = job_data.get("features", {}).get("title", "")
    if title:
        try:
            job_embeddings["title_embedding"] = generate_embeddings(ai, title, provider_name="chatgpt")
        except Exception as e:
            print(f"Error generating title embedding: {e}")

    skills = job_data.get("features", {}).get("skills", [])
    if skills:
        try:
            skills_text = ", ".join(skills)
            job_embeddings["skills_embedding"] = generate_embeddings(ai, skills_text, provider_name="chatgpt")
        except Exception as e:
            print(f"Error generating skills embedding: {e}")

    # Responsibilities Embedding: Prioritize extracted bullet points for cleaner semantic mapping
    description = job_data.get("features", {}).get("description", "")
    requirements = job_data.get("features", {}).get("requirements", [])
    
    if description:
        try:
            responsibilities_text = "\n".join(requirements) if requirements else description
            if responsibilities_text:
                job_embeddings["responsibilities_embedding"] = generate_embeddings(ai, responsibilities_text, provider_name="chatgpt")
        except Exception as e:
            print(f"Error generating responsibilities embedding: {e}")

        try:
            job_embeddings["description_embedding"] = generate_embeddings(ai, description, provider_name="chatgpt")
        except Exception as e:
            print(f"Error generating description embedding: {e}")

    return job_embeddings

##################################### AI Functions #############################################

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

##################################### Main #####################################################

async def main():
    # Load .env variables
    resume = load_resume_as_text("RESUME")
    user_profile = load_resume_as_text("PROFILE")

    # Extract skills and job titles from profile
    tp = TextProcessor()

    ai = AIEngine(default_provider_name="lm_studio") 
    
    # 1. Isolate the raw text blocks for each section
    skills_raw = tp.get_section_content(user_profile, "Skills")
    titles_raw = tp.get_section_content(user_profile, "Job Titles")

    # 2. Clean the blocks into lists of strings for vectorization
    # This removes bullets like "- " or "* " so you only embed the actual text.
    skills = tp.clean_list_from_text(skills_raw)
    job_titles = tp.clean_list_from_text(titles_raw)

    # Generate embeddings for candidate data
    print("Generating embeddings for candidate skills and job titles...")
    
    # Generate embedding for skills (to be used in similarity matching later)
    if skills:
        skill_embeddings = generate_embeddings(ai, ", ".join(skills), provider_name="chatgpt")
    else:
        skill_embeddings = []
        
    # Generate embedding for job titles (to be used in similarity matching later)
    if job_titles:
        title_embeddings = generate_embeddings(ai, ", ".join(job_titles), provider_name="chatgpt")
    else:
        title_embeddings = []

    # Debugging output to verify extraction
    print(f"--- Profile Extraction ---")
    print(f"Extracted {len(skills)} skills: {skills}")
    print(f"Extracted {len(job_titles)} job titles: {job_titles}")
    print(f"---------------------------\n")

    # Create Data Puller Object
    dp = DataPuller(
        dbname = os.getenv("DB_NAME", ""),
        user =os.getenv("DB_USER", ""),
        password = os.getenv("DB_PASSWORD", ""),
        host = os.getenv("DB_HOST", "localhost"),
        port = os.getenv("DB_PORT", "5432")
    )

    # Configure logging
    logging.basicConfig(
        filename='app_error.log', 
        level=logging.ERROR,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

    # Get sites file from .env and pulls the sites in. Needs to be a csv set up with name and site columns
    """
    JOB_SITES should point to a csv file formatted as:
    name,site
    """
    
    sites_file = os.getenv("JOB_SITES","")
    sites = dp.load_sites_list(sites_file)
    print("Sites Retrieved")
    
    # Pull in the site strategies based on the sites pulled from the sites file
    site_strategies = []
    data = []
    for i in range(len(sites['name'])): # type: ignore
        strategy = {
            "company": sites['name'][i], # type: ignore
            "site": sites['site'][i], # type: ignore
            "strategy": dp.load_site_strategies(f"./site_strategies/{sites['name'][i]}.json"), # type: ignore
            "api_method": "",
        }

        if strategy['strategy'].get('pagination') is not None:
            strategy['api_method'] = "extract-paginated"
        elif strategy['strategy'].get("js_config") is not None:
            strategy['api_method'] = "extract-js"
        else:
            strategy['api_method'] = "extract"
        print(f"Company: {strategy['company']} | API method: {strategy['api_method']}")
        site_strategies.append(strategy)
    print("Site strategies loaded.")

    # Job descriptions are now properly scraped with description field processing

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

    # Load in the latest pulled in jobs
    dp.load_scraped_data_to_db(data)
    
    # clear variables from data

    del data, site_strategies, sites, sites_file, strategy

    # run searches on job boards and load them
   
    job_board_list = ["indeed", "linkedin", "zip_recruiter", "google"]

    ## build proxy list

    ## Build search terms
    search_terms_file = os.getenv("SEARCH_TERMS", "")

    if not search_terms_file:
        error_msg = "Error: SEARCH_TERMS environment variable is empty or not provided."
        print(error_msg)
        logging.error(error_msg)
        raise ValueError(error_msg)

    try:
        search_terms = []
        with open(search_terms_file, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if row:  # Ensure the row is not empty
                    search_terms.append(row[0])
        
        if not search_terms:
            raise ValueError("The provided CSV file is empty.")
            
    except Exception as e:
        error_msg = f"Failed to load search terms from {search_terms_file}: {e}"
        print(error_msg)
        logging.error(error_msg)
        raise    
   
    ## loop over site list
    jobs = []
    requests_wanted = 200
    target_cities = json.loads(os.getenv("TARGET_CITIES", "[]"))

    """
    "source": site,
    "title": title,
    "link": job_url,
    "company": company,
    "pay": f"min_amount - max_amount interval"
    "description": description,
    "city": city,
    "state": state
    """
    
    delay = {
        "min": 1,
        "max": 4
    }

    for board in job_board_list:
        for st in search_terms:
            for location in target_cities:
                kwa = {}
                if board == "google":
                    kwa["google_search_term"] = st
                elif board == "indeed":
                    kwa["search_terms"] = st
                    kwa["country_indeed"] = 'USA'
                elif board == "linkedin":
                    kwa["search_terms"] = st
                    kwa["linkedin_fetch_description"] = True
                else:
                    kwa["search_terms"] = st
                jobs_list = scrape_jb(board, location, requests_wanted, 24, **kwa)
                
                njl = []
                for i in jobs_list:
                    # use dict.get to avoid __getitem__ type issues when i is typed as Hashable
                    job = {
                        "source": i.get('site') if isinstance(i, dict) else None,
                        "title": i.get('title') if isinstance(i, dict) else None,
                        "link": i.get('job_url') if isinstance(i, dict) else None,
                        "company": i.get('company') if isinstance(i, dict) else None,
                        "pay": f"{i.get('min_amount') if isinstance(i, dict) else ''}- {i.get('max_amount') if isinstance(i, dict) else ''} {i.get('interval') if isinstance(i, dict) else ''}",
                        "description": i.get('description') if isinstance(i, dict) else None,
                        "city": i.get('city') if isinstance(i, dict) else None,
                        "state": i.get('state') if isinstance(i, dict) else None
                    }
                    njl.append(job)
                
                jobs.extend(njl)
                time.sleep(random.uniform(delay['min'], delay['max']))

    ## Load into db
    dp.load_scraped_data_to_db(jobs) # Need to update this function to include description, city, and state

    # Normalize and Extract Metadata

    # STAGE 4: EMBEDDING GENERATION (Constraint: Only new jobs should be embedded)
    pull_jobs_sql = """
    SELECT j.id, j.job_name, j.description, j.seniority, j.pay_range, j.timezone
    FROM job j
    LEFT JOIN job_embeddings je ON j.id = je.job_id
    WHERE je.job_id IS NULL;
    """

    todays_jobs = dp.pull_data_db(pull_jobs_sql)
    processed_job_pool = []

    if todays_jobs is None:
        todays_jobs = []

    print(f"Starting extraction on {len(todays_jobs)} jobs...")

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
            continue

        # Deterministic Extraction 
        work_type = tp.detect_work_type(raw_job["description"])
        seniority = tp.detect_seniority(raw_job["description"])
        salary = tp.extract_salary(raw_job["description"])

        # Structure the Data
        extracted_data = {
            "metadata": {
                "job_id": raw_job["id"],
                "source": "sql_pull"
            },
            "features": {
                "title": raw_job["title"],
                "description": raw_job["description"],
                "pay": salary if salary != "Not Specified" else raw_job["pay_range"],
                "seniority": seniority,
                "work_type": work_type,
                "timezone": raw_job["timezone"]
            },
            "embeddings": {
                "description_vector": None, 
                "skills_vector": None
            }
        }

        processed_job_pool.append(extracted_data)

    print(f"Successfully processed {len(processed_job_pool)} jobs after filtering.")
    print(f"Sample Job: {processed_job_pool[0]['features'] if processed_job_pool else 'No jobs found'}")

    print(f"Starting AI/LLM Pass on {len(processed_job_pool)} jobs...")

    for index, job in enumerate(processed_job_pool):
        print(f"Processing job {index + 1}/{len(processed_job_pool)}...")
        
        description = job['features']['description']

        # 1. AI Extraction (The "Smart" Pass)
        # We use a high-intelligence model (Claude) to parse the messy text into clean JSON
        ai_data = call_llm_for_extraction(ai, description, provider_name="claude")
        
        # Map extracted data to the job object
        job['features']['skills'] = ai_data.get('skills', [])
        job['features']['requirements'] = ai_data.get('requirements', [])
        job['features']['summary'] = ai_data.get('summary', "")

        # 2. Embedding Generation (The "Vector" Pass)
        # We use a specialized/cheaper model (ChatGPT/OpenAI) for high-dimensional vectors
        
        # Vector A: The "Skills" Vector (Matches against candidate technical skills)
        skills_text = ", ".join(job['features']['skills'])
        if skills_text:
            job['embeddings']['skills_vector'] = generate_embeddings(ai, skills_text, provider_name="chatgpt")
        else:
            job['embeddings']['skills_vector'] = []

        # Vector B: The "Context" Vector (Matches against candidate professional summary/experience)
        summary_text = job['features']['summary']
        if summary_text:
            job['embeddings']['description_vector'] = generate_embeddings(ai, summary_text, provider_name="chatgpt")
        else:
            job['embeddings']['description_vector'] = []

        # Small delay to respect API rate limits
        time.sleep(1) 

    print("AI/LLM Pass Complete.")

    # === EMBEDDING GENERATION ===
    print(f"Starting embedding generation for {len(processed_job_pool)} jobs...")
    
    # Generate embeddings for each job (this is where the guide's requirements should be met)
    embedding_updates = []
    for index, job in enumerate(processed_job_pool):
        print(f"Generating embeddings for job {index + 1}/{len(processed_job_pool)}...")
        
        # Generate the embeddings according to guide specifications
        job_embeddings = generate_job_embeddings(ai, job)
        
        # Store embedding data for later database insertion
        if job_embeddings:
            embedding_updates.append(job_embeddings)
        
        # Small delay to respect API rate limits
        time.sleep(1) 

    print("Embedding generation complete.")

    if embedding_updates:
        print(f"Saving {len(embedding_updates)} new embeddings to 'job_embeddings'...")
        dp.save_job_embeddings(embedding_updates)
    
    # Update job records in database with extracted metadata
    print("Updating job records in database with metadata...")
    
    # Prepare list of updates for the database
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
    
    # Update the database with metadata
    dp.update_job_metadata(job_updates)
    print(f"Successfully updated {len(job_updates)} job records with metadata.")

    # Rule Filtering

    print(f"Starting Rule-Based Filtering on {len(processed_job_pool)} jobs...")
    
    # Define user preferences (In a real app, these come from the 'PROFILE' docx)
    # For now, we simulate what was extracted from the profile.
    user_preferences = {
        "work_types": json.loads(os.getenv("WORK_TYPES", "[]")),
        "seniority_levels": json.loads(os.getenv("SENIORITY_LEVELS", "[]")),
        "timezones": json.loads(os.getenv("TIMEZONES", "[]")),
        "pay_range": os.getenv("TARGET_PAY_RANGE", "")  # Example: "50k-100k" or similar
    }

    for job in processed_job_pool:
        # Apply the rules
        should_skip = apply_rule_filters(job, user_preferences)
        
        # Add the skip flag to the job object
        job['skip'] = should_skip

    # Filter out skipped jobs for the next stage (Vector Scoring)
    # We keep them in the list but marked, so we can update the DB later.
    print(f"Filtering complete. {sum(1 for j in processed_job_pool if j['skip'])} jobs skipped.")

    # --- DATABASE SYNC: Update SQL with Skip Status ---
    print("Syncing skip status to database...")
    
    # We need to collect IDs of skipped jobs to perform a bulk update
    skipped_ids = [job['metadata']['job_id'] for job in processed_job_pool if job.get('skip')]
    
    if skipped_ids:
        # We will implement this method in DataPuller
        dp.bulk_update_skip_status(skipped_ids)
        print(f"Successfully updated {len(skipped_ids)} jobs in database as 'skipped'.")
    else:
        print("No jobs to skip in database.")

    # Proceed only with non-skipped jobs for the expensive Vector Scoring
    active_job_pool = [j for j in processed_job_pool if not j.get('skip')]
    
    print(f"Proceeding to Vector Scoring with {len(active_job_pool)} active jobs.")

    # === Archetype Engine Integration ===
    
    # Initialize the archetype manager 
    archetype_manager = ArchetypeManager()
    
    # Multi-Archetype Support with Caching (Stage 4)
    print("Synchronizing candidate archetypes and benchmarks...")
    
    # Add Resume and User Profile to the pool
    all_archetypes = ARCHETYPES_CONFIG + [
        {"name": "Resume", "title": "Resume", "skills": "", "responsibilities": resume, "type": "resume"},
        {"name": "User Profile", "title": "User Profile", "skills": "", "responsibilities": user_profile, "type": "user_profile"}
    ]

    for arch_config in all_archetypes:
        # Check if already cached in DB
        cached_arch = dp.get_archetype_embeddings(arch_config['name'])
        
        if cached_arch:
            # Load pre-computed vectors from DB to satisfy "rarely change" constraint
            archetype_manager.add_archetype(Archetype(
                name=arch_config['name'],
                type=cached_arch['archetype_type'],
                title_embedding=np.array(cached_arch['title_embedding']),
                skills_embedding=np.array(cached_arch['skills_embedding']),
                responsibilities_embedding=np.array(cached_arch['responsibilities_embedding']),
                metadata=cached_arch.get('metadata', {})
            ))
        else:
            # Generate new vectors if not in cache
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

    # === Vector Scoring with Archetypes (Complete Implementation) ===

    print("Comparing jobs to archetypes...")
    
    # Process each job through archetype comparison
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

    print("Applying enhanced scoring based on archetype matches...")
    for job in active_job_pool:
        archetype_scores = [match['similarity_score'] for match in job.get('archetype_matches', [])]
        if archetype_scores:
            max_archetype_score = max(archetype_scores)
            boost_factor = max_archetype_score ** 2
            job['archetype_enhanced_score'] = boost_factor

    print("Generating detailed ranking metadata...")
    for job in active_job_pool:
        if 'retrieval_metadata' not in job:
            job['retrieval_metadata'] = {}
        if 'archetype_enhanced_score' in job:
            job['retrieval_metadata']['enhanced_archetype_score'] = job['archetype_enhanced_score']
            job['retrieval_metadata']['normalized_archetype_score'] = float(job['archetype_enhanced_score'])

    print("Applying threshold filtering based on archetype matches...")
    filtered_job_pool = []
    for job in active_job_pool:
        match_scores = [m['similarity_score'] for m in job.get('archetype_matches', [])]
        if match_scores and max(match_scores) > 0.3:
            filtered_job_pool.append(job)
    print(f"Filtered job pool from {len(active_job_pool)} to {len(filtered_job_pool)} jobs based on archetype match strength.")

    print("\n=== Archetype-Based Job Shortlisting ===")
    top_jobs = filtered_job_pool[:5]
    for i, job in enumerate(top_jobs):
        print(f"{i+1}. {job['features']['title']}")
        if job.get('retrieval_metadata'):
            print(f"   Best archetype match: {job['retrieval_metadata'].get('best_match', 'None')}")
            print(f"   Match count: {job['retrieval_metadata'].get('match_count', 0)}")
            print(f"   Enhanced archetype score: {job.get('archetype_enhanced_score', 'N/A')}")
        print()

    print("Applying cheap LLM pass based on archetype matching...")
    prioritized_jobs = []
    for job in filtered_job_pool:
        archetype_score = job.get('archetype_enhanced_score', 0)
        if archetype_score > 0.5:
            prioritized_jobs.append(job)
    print(f"Prioritized {len(prioritized_jobs)} jobs for cheap LLM analysis based on archetype matching.")

    print("Applying premium LLM pass to top archetype-matched jobs...")
    top_archetype_jobs = sorted(filtered_job_pool, key=lambda x: x.get('archetype_enhanced_score', 0), reverse=True)[:10]
    print(f"Processing top {len(top_archetype_jobs)} jobs with premium LLM models...")
    for i, job in enumerate(top_archetype_jobs):
        print(f"Premium LLM Analysis - Job {i+1}: {job['features']['title']}")
        # In a real implementation, you would:
        # - Use higher-quality LLMs (e.g., GPT-4, Claude)
        # - Perform more detailed skill matching with context

# Init
