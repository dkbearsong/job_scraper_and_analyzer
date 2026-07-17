import os
import sys
import yaml
import json
import re
from datetime import datetime
from dotenv import load_dotenv

# Load env variables first to ensure config parameters like ARCHETYPES_CONFIG are available
load_dotenv()

# Ensure we can import from the main workspace
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from app.pull_data import DataPuller
from main import (
    is_missing_value,
    term_matches,
    _row_to_job_dict,
    geocode,
    extract_job_locations,
    convert_pay,
    haversine_distance
)

def _load_user_config() -> dict:
    """Load the default user config from .env or default path."""
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    config = {}
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            config = yaml.safe_load(f) or {}
    return config

def load_custom_preferences(filepath: str) -> dict:
    """Load custom preferences YAML file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preferences file not found at: {filepath}")
    with open(filepath, 'r') as f:
        return yaml.safe_load(f) or {}

def evaluate_step(step_name: str, passed: bool, job_val: str | list | None, pref_val: str | list | int | None, explanation: str = "") -> dict:
    return {
        "step_name": step_name,
        "passed": passed,  # True = Match/Pass, False = Mismatch/Skip
        "job_val": job_val,
        "pref_val": pref_val,
        "explanation": explanation
    }

def check_work_type(job: dict, user_preferences: dict) -> dict:
    features = job.get('features', {})
    job_work_type = features.get('work_type')
    user_work_types = user_preferences.get('work_types', [])
    
    if not user_work_types:
        return evaluate_step("Work Type", True, job_work_type, user_work_types, "No preference set, auto-pass.")
        
    if is_missing_value(job_work_type):
        return evaluate_step("Work Type", True, job_work_type, user_work_types, "Job work type is missing, auto-pass.")
        
    matched = False
    matched_pref = None
    for pref in user_work_types:
        if term_matches(pref, job_work_type):
            matched = True
            matched_pref = pref
            break
            
    if matched:
        return evaluate_step("Work Type", True, job_work_type, user_work_types, f"Match found: '{job_work_type}' matches preference '{matched_pref}'.")
    else:
        explanation = (
            f"INPUTS:\n"
            f"  - Job Work Type: {repr(job_work_type)}\n"
            f"  - User Allowed Work Types: {repr(user_work_types)}\n"
            f"EVALUATION:\n"
            f"  - Checked if any allowed preference matches job value '{job_work_type}' via substring/word matching.\n"
            f"  - Result: No match found. Marks record as True for skipping."
        )
        return evaluate_step("Work Type", False, job_work_type, user_work_types, explanation)

def check_seniority(job: dict, user_preferences: dict) -> dict:
    features = job.get('features', {})
    job_seniority = features.get('seniority')
    user_seniority_levels = user_preferences.get('seniority_levels', [])
    
    if not user_seniority_levels:
        return evaluate_step("Seniority", True, job_seniority, user_seniority_levels, "No preference set, auto-pass.")
        
    if is_missing_value(job_seniority):
        return evaluate_step("Seniority", True, job_seniority, user_seniority_levels, "Job seniority is missing, auto-pass.")
        
    matched = False
    matched_pref = None
    for pref in user_seniority_levels:
        if term_matches(pref, job_seniority):
            matched = True
            matched_pref = pref
            break
            
    if matched:
        return evaluate_step("Seniority", True, job_seniority, user_seniority_levels, f"Match found: '{job_seniority}' matches preference '{matched_pref}'.")
    else:
        explanation = (
            f"INPUTS:\n"
            f"  - Job Seniority: {repr(job_seniority)}\n"
            f"  - User Allowed Seniorities: {repr(user_seniority_levels)}\n"
            f"EVALUATION:\n"
            f"  - Checked if any allowed preference matches job value '{job_seniority}' via substring/word matching.\n"
            f"  - Result: No match found. Marks record as True for skipping."
        )
        return evaluate_step("Seniority", False, job_seniority, user_seniority_levels, explanation)

def check_pay(job: dict, user_preferences: dict) -> dict:
    features = job.get('features', {})
    job_pay = features.get('pay', "")
    target_pay_range = user_preferences.get('pay_range', '')
    
    if is_missing_value(job_pay) or not job_pay:
        return evaluate_step("Pay", True, job_pay, target_pay_range, "Job pay is missing/not specified, auto-pass.")
        
    if not target_pay_range:
        return evaluate_step("Pay", True, job_pay, target_pay_range, "No preference set, auto-pass.")
        
    user_min_match = re.search(r'(\d+)(?:k|K)?', target_pay_range)
    user_max_match = re.search(r'-(\d+)(?:k|K)?', target_pay_range)
    
    if not (user_min_match and user_max_match):
        return evaluate_step("Pay", True, job_pay, target_pay_range, "User pay range could not be parsed, auto-pass.")
        
    user_min = convert_pay(user_min_match, target_pay_range)
    user_max = convert_pay(user_max_match, target_pay_range)
    
    job_min, job_max = None, None
    if isinstance(job_pay, str):
        job_pay_cleaned = re.sub(r'\.\d+', '', job_pay.replace(',', ''))
        matches = re.findall(r'(\d+)(?:k|K)?', job_pay_cleaned)
        if matches:
            job_min = int(matches[0]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[0])
        if len(matches) >= 2:
            job_max = int(matches[1]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[1])
            
    is_skipped = None
    eval_detail = ""
    if job_max is not None:
        if job_min is None:
            job_min = job_max
        is_skipped = True if job_max < user_min or job_min > user_max else False
        eval_detail = f"Job pay range parsed as: {job_min} - {job_max}. Target range: {user_min} - {user_max}."
    elif job_min is not None:
        is_skipped = False if user_min <= job_min <= user_max else True
        eval_detail = f"Job pay parsed as single value: {job_min}. Target range: {user_min} - {user_max}."
    else:
        return evaluate_step("Pay", True, job_pay, target_pay_range, f"Job pay '{job_pay}' could not be parsed to numeric value, auto-pass.")
        
    if not is_skipped:
        return evaluate_step("Pay", True, job_pay, target_pay_range, f"Match: Job pay is within user range. {eval_detail}")
    else:
        explanation = (
            f"INPUTS:\n"
            f"  - Job Pay: {repr(job_pay)}\n"
            f"  - User Pay Range: {repr(target_pay_range)}\n"
            f"EVALUATION:\n"
            f"  - Parsed User Range: {user_min} to {user_max}\n"
            f"  - Parsed Job Pay: min={job_min}, max={job_max}\n"
            f"  - Check result: Job pay range {job_min}-{job_max} does not overlap with user range {user_min}-{user_max}.\n"
            f"  - Result: Marks record as True for skipping."
        )
        return evaluate_step("Pay", False, job_pay, target_pay_range, explanation)

def check_timezone(job: dict, user_preferences: dict) -> dict:
    features = job.get('features', {})
    job_timezone = features.get('timezone')
    user_timezones = user_preferences.get('timezones', [])
    
    if not user_timezones:
        return evaluate_step("Timezone", True, job_timezone, user_timezones, "No preference set, auto-pass.")
        
    if is_missing_value(job_timezone):
        return evaluate_step("Timezone", True, job_timezone, user_timezones, "Job timezone is missing, auto-pass.")
        
    matched = False
    matched_pref = None
    for pref in user_timezones:
        if term_matches(pref, job_timezone):
            matched = True
            matched_pref = pref
            break
            
    if matched:
        return evaluate_step("Timezone", True, job_timezone, user_timezones, f"Match found: '{job_timezone}' matches preference '{matched_pref}'.")
    else:
        explanation = (
            f"INPUTS:\n"
            f"  - Job Timezone: {repr(job_timezone)}\n"
            f"  - User Allowed Timezones: {repr(user_timezones)}\n"
            f"EVALUATION:\n"
            f"  - Checked if any allowed preference matches job value '{job_timezone}' via substring/word matching.\n"
            f"  - Result: No match found. Marks record as True for skipping."
        )
        return evaluate_step("Timezone", False, job_timezone, user_timezones, explanation)

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

def check_proximity(job: dict, user_preferences: dict) -> dict:
    target_cities = user_preferences.get('target_cities', [])
    if not target_cities:
        return evaluate_step("Proximity", True, None, target_cities, "No target cities set, auto-pass.")
        
    work_type = job.get('features', {}).get('work_type', '')
    if isinstance(work_type, str) and 'remote' in work_type.lower():
        return evaluate_step("Proximity", True, f"work_type: {work_type}", target_cities, "Job work_type contains 'remote', proximity filter bypassed (MATCH).")
        
    job_locations = extract_job_locations(job)
    if not job_locations:
        return evaluate_step("Proximity", True, None, target_cities, "No job locations extracted, auto-pass.")
        
    for loc in job_locations:
        if 'remote' in loc.lower():
            return evaluate_step("Proximity", True, f"locations: {job_locations}", target_cities, f"Job location '{loc}' contains 'remote', proximity filter bypassed (MATCH).")
        if is_country_location(loc):
            return evaluate_step("Proximity", True, f"locations: {job_locations}", target_cities, f"Job location '{loc}' is a country, proximity filter bypassed (MATCH).")
            
    max_range = user_preferences.get('target_city_range', 25)
    
    geocoded_any = False
    details = []
    
    for target in target_cities:
        target_clean = target.strip().lower()
        
        for loc in job_locations:
            loc_clean = loc.strip().lower()
            if target_clean in loc_clean or loc_clean in target_clean:
                return evaluate_step("Proximity", True, job_locations, target_cities, f"Exact/substring match found: Job location '{loc}' matches target '{target}'.")
                
        target_coords = geocode(target)
        if not target_coords:
            details.append(f"Could not geocode target city '{target}'")
            continue
            
        for loc in job_locations:
            loc_coords = geocode(loc)
            if loc_coords:
                geocoded_any = True
                dist = haversine_distance(target_coords[0], target_coords[1], loc_coords[0], loc_coords[1])
                if dist <= max_range:
                    return evaluate_step("Proximity", True, job_locations, target_cities, 
                                         f"Geocoded match found: Job location '{loc}' ({loc_coords}) is {dist:.2f} miles from '{target}' ({target_coords}), which is <= {max_range} miles.")
                else:
                    details.append(f"Job location '{loc}' ({loc_coords}) is {dist:.2f} miles from '{target}' ({target_coords}) (exceeds {max_range} miles).")
            else:
                details.append(f"Could not geocode job location '{loc}'")
                
    if geocoded_any:
        explanation = (
            f"INPUTS:\n"
            f"  - Job Locations: {repr(job_locations)}\n"
            f"  - Target Cities: {repr(target_cities)}\n"
            f"  - Allowed Range: {max_range} miles\n"
            f"EVALUATION:\n"
            f"  - Geocoding completed, but no job location was within {max_range} miles of any target city.\n"
            f"  - Details:\n    " + "\n    ".join(details) + "\n"
            f"  - Result: Marks record as True for skipping."
        )
        return evaluate_step("Proximity", False, job_locations, target_cities, explanation)
    else:
        return evaluate_step("Proximity", True, job_locations, target_cities, 
                             "Could not geocode or match any locations to target cities. Allowed to pass to avoid false negative.")

def evaluate_job_rules(job: dict, user_preferences: dict) -> list:
    """Evaluates all 5 rule steps for a job, returns list of step results."""
    return [
        check_work_type(job, user_preferences),
        check_seniority(job, user_preferences),
        check_pay(job, user_preferences),
        check_timezone(job, user_preferences),
        check_proximity(job, user_preferences)
    ]

def get_recent_dates(dp: DataPuller) -> list:
    """Get the 5 most recent dates that contain jobs in the database."""
    query = """
    SELECT date_added, COUNT(*) 
    FROM job 
    GROUP BY date_added 
    ORDER BY date_added DESC 
    LIMIT 5
    """
    rows = dp.conn.execute_sql(query, fetch=True)
    if not rows:
        return []
    # Extract dates
    dates = []
    for row in rows:
        if isinstance(row, dict):
            dates.append((row.get("date_added"), row.get("count")))
        else:
            dates.append((row[0], row[1]))
    return dates

def main():
    print("=" * 60)
    print("        STAGE 3 RULE FILTERING TEST RUNNER")
    print("=" * 60)

    # Load configuration
    user_config = _load_user_config()
    db_name = user_config.get("db_name", os.getenv("DB_NAME", ""))
    db_user = user_config.get("db_user", os.getenv("DB_USER", ""))
    db_password = user_config.get("db_password", os.getenv("DB_PASSWORD", ""))
    db_host = user_config.get("db_host", os.getenv("DB_HOST", "localhost"))
    db_port = str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))

    # Connect to DB
    print(f"Connecting to database '{db_name}' on {db_host}:{db_port}...")
    try:
        dp = DataPuller(dbname=db_name, user=db_user, password=db_password, host=db_host, port=db_port)
    except Exception as e:
        print(f"Error connecting to database: {e}")
        sys.exit(1)

    # Get recent dates
    recent_dates = get_recent_dates(dp)
    if recent_dates:
        print("\nRecent dates in database with jobs:")
        for idx, (dt, cnt) in enumerate(recent_dates, 1):
            dt_str = dt.strftime("%Y-%m-%d") if hasattr(dt, "strftime") else str(dt)
            print(f"  {idx}. {dt_str} ({cnt} jobs)")
    else:
        print("\nNo jobs found in the database yet.")

    # Prompt for date
    default_date_str = ""
    if recent_dates:
        first_date = recent_dates[0][0]
        default_date_str = first_date.strftime("%Y-%m-%d") if hasattr(first_date, "strftime") else str(first_date)

    print()
    date_prompt = f"Enter date to test (YYYY-MM-DD) [default: {default_date_str}]: " if default_date_str else "Enter date to test (YYYY-MM-DD): "
    user_input_date = input(date_prompt).strip()
    if not user_input_date:
        user_input_date = default_date_str

    if not user_input_date:
        print("Error: No date specified.")
        sys.exit(1)

    try:
        target_date = datetime.strptime(user_input_date, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Invalid date format. Please use YYYY-MM-DD.")
        sys.exit(1)

    # Prompt for preferences file
    pref_default = "test_preferences.yaml" if os.path.exists("test_preferences.yaml") else "user_preferences.yaml"
    pref_prompt = f"Enter preferences file path [default: {pref_default}]: "
    user_input_pref = input(pref_prompt).strip()
    if not user_input_pref:
        user_input_pref = pref_default

    print(f"\nLoading preferences from: {user_input_pref}")
    try:
        user_preferences = load_custom_preferences(user_input_pref)
    except Exception as e:
        print(f"Error loading preferences file: {e}")
        sys.exit(1)

    # Query all jobs on target date
    print(f"Retrieving all jobs added on {target_date}...")
    query = """
        SELECT j.id, j.job_name, c.company_name, j.link, j.job_summary, j.description AS extracted_summary,
               j.skills, j.responsibilities, j.pay_range, j.seniority, j.work_type, j.timezone,
               j.source, j.date_added, o.city, o.state, o.location,
               j.flexibility,
               je.title_embedding, je.skills_embedding, je.responsibilities_embedding, je.description_embedding
        FROM job j
        JOIN company c ON j.company_id = c.id
        LEFT JOIN office o ON j.office_id = o.id
        LEFT JOIN job_embeddings je ON j.id = je.job_id
        WHERE j.date_added = %s
        ORDER BY j.id ASC
    """
    
    dp.conn.execute_sql("SET statement_timeout = 60000")
    rows = dp.conn.execute_sql(query, (target_date,), fetch=True)
    if not rows:
        print(f"No jobs found added on date: {target_date}")
        sys.exit(0)

    print(f"Found {len(rows)} jobs. Evaluating filters...")

    # Log setup
    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/stage3_test_run_{target_date}_{timestamp}.log"
    log_file = open(log_filename, "w")

    def log_and_print(msg=""):
        print(msg)
        log_file.write(msg + "\n")

    log_and_print("=" * 80)
    log_and_print(f"STAGE 3 TEST RUN FOR DATE: {target_date}")
    log_and_print(f"PREFERENCES FILE: {user_input_pref}")
    log_and_print(f"RUN TIMESTAMP: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(f"PREFERENCES USED:\n{yaml.dump(user_preferences, default_flow_style=False)}")
    log_and_print("=" * 80)

    stats = {
        "total": 0,
        "matched": 0,
        "skipped": 0,
        "skip_by_step": {
            "Work Type": 0,
            "Seniority": 0,
            "Pay": 0,
            "Timezone": 0,
            "Proximity": 0
        }
    }

    for row in rows:
        stats["total"] += 1
        job = _row_to_job_dict(row)
        
        job_id = job['metadata']['job_id']
        title = job['features']['title']
        company = job['metadata']['company_name']
        
        step_results = evaluate_job_rules(job, user_preferences)
        
        # Check if any step failed (meaning passed is False)
        # Note: In Stage 3 logic, a job is SKIPPED if any step fails (passed=False).
        # So a job passes overall only if ALL steps are passed=True.
        failed_steps = [res for res in step_results if not res["passed"]]
        overall_match = len(failed_steps) == 0

        log_and_print(f"\n[{stats['total']}] JOB ID: {job_id} | Title: {title} | Company: {company}")
        log_and_print("-" * 60)

        for res in step_results:
            step_name = res["step_name"]
            passed = res["passed"]
            explanation = res["explanation"]
            
            status_str = "MATCH (Pass)" if passed else "NO MATCH (Skip)"
            log_and_print(f"  * {step_name:12}: {status_str}")
            
            if not passed:
                stats["skip_by_step"][step_name] += 1
                # Output exactly what inputs were and how it saw the values
                # Explanation contains the input/evaluation dump
                indented_explanation = "      " + explanation.replace("\n", "\n      ")
                log_and_print(indented_explanation)
            else:
                log_and_print(f"      Details: {explanation}")

        if overall_match:
            stats["matched"] += 1
            log_and_print("  => OVERALL RESULT: MATCH (Keep Job)")
        else:
            stats["skipped"] += 1
            failed_names = [res["step_name"] for res in failed_steps]
            log_and_print(f"  => OVERALL RESULT: NO MATCH (Skip Job) | Failed steps: {failed_names}")

    # Print summary statistics
    log_and_print("\n" + "=" * 80)
    log_and_print("SUMMARY STATISTICS")
    log_and_print("=" * 80)
    log_and_print(f"Total Jobs Evaluated: {stats['total']}")
    log_and_print(f"Matched (Kept):      {stats['matched']} ({stats['matched']/stats['total']*100:.1f}%)")
    log_and_print(f"Skipped (Filtered):  {stats['skipped']} ({stats['skipped']/stats['total']*100:.1f}%)")
    log_and_print("\nSkip Counts by Step (Jobs can fail multiple steps):")
    for step_name, count in stats["skip_by_step"].items():
        log_and_print(f"  - {step_name:12}: {count:4} jobs skipped ({count/stats['total']*100:.1f}%)")
    log_and_print("=" * 80)

    log_file.close()
    print(f"\nDetailed log saved to: {log_filename}")

if __name__ == "__main__":
    main()
