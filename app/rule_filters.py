import re
from app.logger import error_logger_continue

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


def role_matches_title(kw: str, title: str) -> bool:
    """
    Checks if a role keyword matches a job title.
    Requires all words of the role keyword phrase to be present in the job title.
    """
    kw_proc = kw.strip().lower()
    title_proc = title.strip().lower()
    if not kw_proc or not title_proc:
        return False
    if kw_proc in title_proc:
        return True
    kw_words = [w for w in re.split(r'[^a-zA-Z0-9\-\']', kw_proc) if w]
    title_words = [w for w in re.split(r'[^a-zA-Z0-9\-\']', title_proc) if w]
    if not kw_words:
        return False
    return all(w in title_words for w in kw_words)


def is_title_disqualified(title: str, disqualified_titles: list) -> bool:
    """
    Checks if a job title matches any disqualified word, phrase, or regex pattern.
    
    Supports:
    - Case-insensitive whole-word/bounded phrase matching (e.g. "sales" matches "Sales Director", but not "Wholesales" or "Salesforce")
    - Quoted strings (e.g. '"Exact Match"') for exact substring matching
    - Raw regex patterns (e.g. r'\b(sales|intern(ship)?)\b')
    """
    if not title or not disqualified_titles:
        return False
    if not isinstance(title, str):
        title = str(title)
    title_clean = title.strip()
    if not title_clean:
        return False

    for item in disqualified_titles:
        if not item:
            continue
        if not isinstance(item, str):
            item = str(item)
        item_clean = item.strip()
        if not item_clean:
            continue

        # Check for quotation marks for exact substring matching
        is_quoted = (item_clean.startswith('"') and item_clean.endswith('"')) or (item_clean.startswith("'") and item_clean.endswith("'"))
        if is_quoted:
            clean_item = item_clean[1:-1].strip().lower()
            if clean_item and clean_item in title_clean.lower():
                return True
            continue

        # If user explicitly provided a regex pattern containing special regex syntax
        has_regex_metachars = any(c in item_clean for c in r'.^$*+?{}[]\|()')
        if has_regex_metachars:
            try:
                if re.search(item_clean, title_clean, re.IGNORECASE):
                    return True
            except re.error:
                pass

        # Standard word/phrase bounded regex matching with flexible whitespace
        words = item_clean.split()
        if not words:
            continue
        escaped_words = [re.escape(w) for w in words]
        pattern_str = r'\s+'.join(escaped_words)

        left_b = r'\b' if re.match(r'^\w', item_clean) else r'(?:^|\s)'
        right_b = r'\b' if re.search(r'\w$', item_clean) else r'(?:$|\s)'
        bounded_regex = rf'{left_b}{pattern_str}{right_b}'

        if re.search(bounded_regex, title_clean, re.IGNORECASE):
            return True

    return False


def apply_rule_filters(job: dict, user_preferences: dict) -> bool:
    """
    Evaluates a job against hard constraints.
    Returns True if the job should be SKIPPED, False if it passes.
    """
    from app.location_utils import check_location_proximity
    features = job.get('features', {})

    # 0. Job Title Disqualification Filter (Stage 1.5 Deterministic Regex Filter)
    job_title = features.get('title') or job.get('title') or job.get('job_name', '')
    disqualified_titles = (
        user_preferences.get('disqualified_titles')
        or user_preferences.get('disqualified_job_titles')
        or user_preferences.get('disqualify_titles')
        or user_preferences.get('excluded_titles')
        or []
    )
    if disqualified_titles and job_title:
        if is_title_disqualified(job_title, disqualified_titles):
            return True

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

    # 2. Seniority Filter (with Role-Specific Overrides)
    job_seniority = features.get('seniority')
    job_title = features.get('title', '')
    
    # Check if a role-specific seniority override applies to the job title
    role_seniority_overrides = user_preferences.get('role_seniority_levels', [])
    user_seniority_levels = None

    if role_seniority_overrides and job_title:
        for override in role_seniority_overrides:
            keywords = override.get('role_keywords', [])
            allowed = override.get('allowed_seniority', [])
            for kw in keywords:
                if role_matches_title(kw, job_title):
                    user_seniority_levels = allowed
                    break
            if user_seniority_levels is not None:
                break

    # Fall back to global seniority_levels if no role override matched
    if user_seniority_levels is None:
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
        job_pay_lower = job_pay.lower()
        is_hourly = bool(re.search(r'(\bhr\b|\bhour\b|\bhourly\b|/\s*hr)', job_pay_lower))

        # Remove commas and strip decimal digits (e.g. $111,259.26 -> $111259)
        job_pay_cleaned = re.sub(r'\.\d+', '', job_pay.replace(',', ''))
        matches = re.findall(r'(\d+)(?:k|K)?', job_pay_cleaned)
        if matches:
            job_min = int(matches[0]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[0])
        if len(matches) >= 2:
            job_max = int(matches[1]) * 1000 if 'k' in job_pay_cleaned.lower() else int(matches[1])

        # Convert hourly pay to annual (assuming 2,000 work hours/year) if numbers are small
        if is_hourly:
            if job_min is not None and job_min < 1000:
                job_min *= 2000
            if job_max is not None and job_max < 1000:
                job_max *= 2000

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


