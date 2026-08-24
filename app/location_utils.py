import os
import json
import math
import requests
import re
import time
from typing import Optional, Tuple, List
from app.rule_filters import is_missing_value

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
        "poland": "poland",
        "spain": "spain",
        "italy": "italy",
        "brazil": "brazil",
        "mexico": "mexico",
        "switzerland": "switzerland",
        "netherlands": "netherlands",
        "sweden": "sweden",
        "finland": "finland",
        "south korea": "south korea",
        "korea": "south korea",
        "philippines": "philippines",
        "israel": "israel",
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
    Returns True if the job should be SKIPPED because it is outside the target city range / country.
    Returns False if it is within range, or if location data is missing/invalid.
    """
    target_cities = user_preferences.get('target_cities', [])
    if not target_cities:
        return False
        
    job_locations = extract_job_locations(job)
    work_type = job.get('features', {}).get('work_type', '')
    is_remote_job = isinstance(work_type, str) and 'remote' in work_type.lower()
    if not is_remote_job:
        for loc in job_locations:
            if 'remote' in loc.lower():
                is_remote_job = True
                break

    # Build target country set & target list
    target_countries = set()
    for target in target_cities:
        c = get_country_of_location(target)
        if c:
            target_countries.add(c)
        elif is_country_location(target):
            target_countries.add(target.strip().lower())

    # Extract countries for job locations
    job_countries = set()
    for loc in job_locations:
        c = get_country_of_location(loc)
        if c:
            job_countries.add(c)

    # 1. Country Level Filter:
    # If target specifies countries (e.g. "USA" -> "united states"), and job specifies countries
    if target_countries and job_countries:
        # If none of the job's countries match any target country
        if not (job_countries & target_countries):
            # Check if any exact substring match exists with target_cities (e.g. target="London, UK")
            has_substring_match = False
            for target in target_cities:
                target_clean = target.strip().lower()
                for loc in job_locations:
                    loc_clean = loc.strip().lower()
                    if target_clean in loc_clean or loc_clean in target_clean:
                        has_substring_match = True
                        break
                if has_substring_match:
                    break
            if not has_substring_match:
                return True  # Job is in a foreign country outside target_cities -> SKIP

    # 2. Remote Filter:
    # If job is remote and has no foreign country mismatch (or no location specified), allow pass
    if is_remote_job:
        # Only pass if job_countries doesn't conflict with target_countries
        if not target_countries or not job_countries or (job_countries & target_countries):
            return False

    if not job_locations:
        return False

    max_range = user_preferences.get('target_city_range', 25)
    
    geocoded_any = False
    for target in target_cities:
        target_clean = target.strip().lower()
        
        # Exact / substring match
        for loc in job_locations:
            loc_clean = loc.strip().lower()
            if target_clean in loc_clean or loc_clean in target_clean:
                return False
                
        # If target is a country and job location matches that country
        if is_country_location(target):
            t_country = get_country_of_location(target)
            if t_country:
                for loc in job_locations:
                    l_country = get_country_of_location(loc)
                    if l_country == t_country:
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

    # If target is a country (e.g. "USA") and job location is not in that country (e.g. "London, UK")
    if target_countries and job_countries and not (job_countries & target_countries):
        return True

    return False


