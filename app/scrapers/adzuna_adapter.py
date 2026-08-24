import aiohttp
import json
import ast
import os
import asyncio
from dotenv import load_dotenv
import csv
import logging
from furl import furl
from typing import Any, Dict, List
from app.scrapers import ScraperAdapter
from app.prompt_injection_defender import sanitize_untrusted_text

load_dotenv()

#================================================================================#
# Error logging and setup
#================================================================================#

def error_logger_crash(error_msg):
    print(error_msg)
    logging.error(error_msg)
    raise ValueError(error_msg)

def error_logger_continue(error_msg):
    print(error_msg)
    logging.error(error_msg)
    return

def get_data_puller():
    import yaml
    from app.pull_data import DataPuller
    user_config = {}
    prefs_path = os.getenv("USER_PREFERENCES_YAML", "user_preferences.yaml")
    if os.path.exists(prefs_path):
        with open(prefs_path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
    dp = DataPuller(
        dbname=user_config.get("db_name", os.getenv("DB_NAME", "")),
        user=user_config.get("db_user", os.getenv("DB_USER", "")),
        password=user_config.get("db_password", os.getenv("DB_PASSWORD", "")),
        host=user_config.get("db_host", os.getenv("DB_HOST", "localhost")),
        port=str(user_config.get("db_port", os.getenv("DB_PORT", "5432")))
    )
    return dp

#================================================================================#
# Adzuna Adapter Class
#================================================================================#

class AdzunaAdapter(ScraperAdapter):
    def __init__(self):
        super().__init__()
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        self.session = None
        self.logger = logging.getLogger(__name__)

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def get_name(self) -> str:
        return self._config.get("name", "my_custom_adapter")

    def configure(self, config: Dict[str, Any]) -> None:
        self._config = config

    # ================================================================================ #
    # Process and Conform Data
    # ================================================================================ #

    def pull_searches(self, searches_csv: str = ""):
        if not searches_csv:
            searches_csv = os.getenv("ADZUNDA_SEARCHES_CSV","")
        search_terms: list = []
        try:
            with open(searches_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if not row:
                        continue
                    search_terms.append(self.search_sanatizer(row))
        except Exception as e:
            error_logger_crash(f"Failed to load Adzunda search terms. Error: {e}")
        if not search_terms:
            error_logger_crash("No search terms found. Check the CSV header and data.")
        return search_terms

    def adzuna_api_adapter(
            self,
            page:int=1,
            country:str="us",
            rpp:int=0,
            what:str="",
            what_and:str="",
            what_phrase:str="",
            what_or:str="",
            what_exclude:str="",
            title_only:str="",
            where:str="",
            distance:int=5,
            max_days_old:int=7,
            salary_min:int=0,
            salary_max:int=999999,
            salary_include_unknown:bool=True,
            full_time:bool=False,
            part_time:bool=False,
            contract:bool=False,
            permanent:bool=False,
            company:str=""
            ):
        selected_flags = [(k, v) for (k, v) in [("full_time", full_time), ("part_time", part_time), ("contract", contract), ("permanent", permanent)] if v]
        if len(selected_flags) > 1:
            error_logger_crash("Adzuna Adapter Error: Multiple flags detected.")

        url = furl("https://api.adzuna.com/")
        url /= f"v1/api/jobs/{country}/search/{page}"
        url.args["app_id"] = self.app_id
        url.args["app_key"] = self.app_key

        params = {
            "results_per_page": rpp if rpp != 0 else None,
            "what": what if what != "" else None,
            "what_and": what_and if what_and != "" else None,
            "what_phrase": what_phrase if what_phrase != "" else None,
            "what_or": what_or if what_or != "" else None,
            "what_exclude": what_exclude if what_exclude != "" else None,
            "title_only": title_only if title_only != "" else None,
            "where": where if where != "" else None,
            "distance": distance if distance != 5 else None,
            "max_days_old": max_days_old if max_days_old != 7 else None,
            "salary_min": salary_min if salary_min != 0 else None,
            "salary_max": salary_max if salary_max != 999999 else None,
            "salary_include_unknown": "1" if salary_include_unknown else None,
            "company": company if company != "" else None
        }
        for flag_name, _ in selected_flags:
            params[flag_name] = "1"

        url.args.update({k: v for k, v in params.items() if v is not None})
        return url

    # ================================================================================ #
    # Data management
    # ================================================================================ #

    def search_sanatizer(self, search: dict = {}):
        converters = {
            "rpp": int,
            "distance": int,
            "max_days_old": int,
            "salary_min": int,
            "salary_max": int,
            "where": int,
        }
        bool_keys = {"salary_include_unknown", "full_time", "part_time", "contract", "permanent"}
        new_search = {}
        for key, value in search.items():
            if value in (None, "", [], {}):
                continue
            if key in converters:
                try:
                    new_search[key] = converters[key](value)
                except (ValueError, TypeError):
                    new_search[key] = value
            elif key in bool_keys:
                new_search[key] = self.str_to_bool(value)
            else:
                new_search[key] = value
        return new_search

    def str_to_bool(self, val):
        if isinstance(val, str):
            try:
                return ast.literal_eval(val.capitalize())
            except (ValueError, SyntaxError):
                return False
        return val

    def build_details_url(self, job_id: str) -> str:
        """Build the Adzuna details page URL from a job ID."""
        return f"https://www.adzuna.com/details/{job_id}?utm_medium=api&utm_source={self.app_id}"

    def process_data(self, response):
        """Process API response and build output with details URLs."""
        processed: list = []
        jobs_list = response.get('results', [])

        for job in jobs_list:
            job_id = job.get('id', '')
            details_url = self.build_details_url(job_id) if job_id else job.get('redirect_url', '')

            area = job['location'].get('area', [])
            city = area[3] if len(area) > 3 else (area[-1] if area else "")
            state = area[1] if len(area) > 1 else ""
            salary_min = job.get('salary_min')
            salary_max = job.get('salary_max')
            if salary_min is not None and salary_max is not None:
                pay_range = f"${salary_min:,} - ${salary_max:,}"
            elif salary_min is not None:
                pay_range = f"${salary_min:,}"
            elif salary_max is not None:
                pay_range = f"${salary_max:,}"
            else:
                pay_range = ""

            raw_title = job.get('title', '')
            sanitized_title = sanitize_untrusted_text(raw_title)

            new_job = {
                'title': sanitized_title,
                'job_name': sanitized_title,
                'description': "",  # Left empty so fallback page scraper fetches full job description page
                'company': job.get('company', {}).get('display_name', ''),
                'company_name': job.get('company', {}).get('display_name', ''),
                'pay': pay_range,
                'pay_range': pay_range,
                'url': details_url,
                'link': details_url,
                'location': f"{city}, {state}" if city and state else job['location'].get('display_name', ""),
                'source': "Adzuna",
                'flexibility': 'NA'
            }
            processed.append(new_job)

        return processed

    # ================================================================================ #
    # API Requests
    # ================================================================================ #

    DEFAULT_HEADERS = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async def _get_session(self) -> aiohttp.ClientSession:
        """Ensure an active aiohttp session with standard headers and timeout."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=60, connect=15)
            self.session = aiohttp.ClientSession(headers=self.DEFAULT_HEADERS, timeout=timeout)
        return self.session

    async def pull_jobs(self, page: int = 1, search: dict = {}, max_retries: int = 3, base_delay: float = 2.0):
        """
        Pull jobs from Adzuna with retry logic and exponential backoff.
        Handles transient 503, 502, 504, 429 errors and non-JSON responses gracefully.
        """
        url = self.adzuna_api_adapter(page, **search)
        session = await self._get_session()
        title = search.get("title_only") or search.get("what") or "all"

        for attempt in range(1, max_retries + 1):
            try:
                async with session.get(str(url)) as response:
                    status = response.status
                    if status == 200:
                        try:
                            data = await response.json()
                            processed = self.process_data(data)
                            return data.get('count', 0), processed
                        except Exception as json_err:
                            text = await response.text()
                            self.logger.warning(
                                f"[Adzuna] Failed to decode JSON for '{title}' (attempt {attempt}/{max_retries}): {json_err} | Body: {text[:200]}"
                            )
                    elif status in (429, 500, 502, 503, 504):
                        text = await response.text()
                        delay = base_delay * (2 ** (attempt - 1))
                        self.logger.warning(
                            f"[Adzuna] HTTP {status} for '{title}' (attempt {attempt}/{max_retries}). Retrying in {delay:.1f}s... | Response: {text[:150]}"
                        )
                        if attempt < max_retries:
                            await asyncio.sleep(delay)
                            continue
                        else:
                            error_logger_continue(f"[Adzuna] HTTP {status} for '{title}' after {max_retries} retries: {text[:200]}")
                            return 0, []
                    else:
                        text = await response.text()
                        error_logger_continue(f"[Adzuna] HTTP {status} unrecoverable error for '{title}': {text[:200]}")
                        return 0, []
            except (aiohttp.ClientError, asyncio.TimeoutError) as net_err:
                delay = base_delay * (2 ** (attempt - 1))
                self.logger.warning(
                    f"[Adzuna] Network error for '{title}' (attempt {attempt}/{max_retries}): {net_err}. Retrying in {delay:.1f}s..."
                )
                if attempt < max_retries:
                    await asyncio.sleep(delay)
                else:
                    error_logger_continue(f"[Adzuna] Network error for '{title}' after {max_retries} retries: {net_err}")
                    return 0, []
            except Exception as e:
                error_logger_continue(f"[Adzuna] Unexpected error pulling jobs for '{title}': {e}")
                return 0, []

        return 0, []

    async def full_run(self, searches_csv: str = ""):
        """
        Execute all searches defined in the searches CSV.
        Isolates errors per-search so one failing search does not abort the rest.
        """
        session = await self._get_session()
        all_jobs = []
        try:
            searches = self.pull_searches(searches_csv)
            for search in searches:
                search_title = search.get("title_only") or search.get("what") or "unnamed"
                page = 1
                try:
                    while True:
                        count, jobs = await self.pull_jobs(page, search)
                        if jobs:
                            all_jobs.append(jobs)
                        if search.get('rpp') and count > 0 and search['rpp'] * page < count:
                            page += 1
                        else:
                            break
                except Exception as search_err:
                    error_logger_continue(f"[Adzuna] Error during search '{search_title}': {search_err}")
            return all_jobs
        except Exception as e:
            error_logger_continue(f"Error pulling Adzuna data: {e}")
            return all_jobs
        finally:
            if self.session and not self.session.closed:
                await self.session.close()
                self.session = None

    async def scrape(self) -> List[Dict[str, Any]]:
        jobs = await self.full_run() or []

        flattened: List[Dict[str, Any]] = []
        for item in jobs:
            if isinstance(item, list):
                for j in item:
                    if isinstance(j, dict):
                        flattened.append(j)
            elif isinstance(item, dict):
                flattened.append(item)

        if flattened:
            try:
                dp = get_data_puller()
                dp.load_scraped_data_to_db(flattened)
                dp.close_connection()
                self.logger.info(f"Loaded {len(flattened)} jobs into the database.")
            except Exception as e:
                self.logger.error(f"Failed to load Adzuna jobs to database: {e}")

        return flattened


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run AdzunaAdapter standalone")
    parser.add_argument("--csv", default="", help="Path to searches CSV file")
    args = parser.parse_args()

    adapter = AdzunaAdapter()
    try:
        jobs = await adapter.full_run(args.csv)
        jobs = jobs or []
        print(f"Pulled {sum(len(p) if isinstance(p, list) else 1 for p in jobs)} total job entries")
        print(f"Job output:\n{json.dumps(jobs)}")
    finally:
        if adapter.session:
            await adapter.session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())