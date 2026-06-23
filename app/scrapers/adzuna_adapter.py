import aiohttp
import json
import ast
import os
from dotenv import load_dotenv
import csv
import logging
from furl import furl
from typing import Any, Dict, List
from app.scrapers import ScraperAdapter

load_dotenv()

#================================================================================#
# Error logging
#================================================================================#

def error_logger_crash(error_msg):
    print(error_msg)
    logging.error(error_msg)
    raise ValueError(error_msg)

def error_logger_continue(error_msg):
    print(error_msg)
    logging.error(error_msg)
    return

#================================================================================#
# Adzuna Adapter Class
#================================================================================#

class AdzunaAdapter(ScraperAdapter):
    def __init__(self):
        self.app_id = os.getenv("ADZUNA_APP_ID")
        self.app_key = os.getenv("ADZUNA_APP_KEY")
        self.session = None

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
        """
        Pulls env var for file that provides searches. Processes file and extracts different parameters
        for searches as dict
        
        Args:
            searches_csv: Optional file path to override the ADZUNDA_SEARCHES_CSV env var
            
        Returns:
            List of dicts mapping search parameter names to values for adzuna_api_adapter
        """
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
            "salary_min": salary_min if salary_min != 0 else  None,
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
    def search_sanatizer(self, search:dict={}):
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
            # Handles "True", "False", "true", "false" safely
            try:
                return ast.literal_eval(val.capitalize())
            except (ValueError, SyntaxError):
                return False # Default to False if invalid
        return val
    
    def process_data(self, response):
        processed:list = []
        for job in response['results']:
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
            new_job ={
                'job_name':job['title'],
                'company_name':job['company']['display_name'],
                'pay_range':pay_range,
                'link':job['redirect_url'],
                'location':f"{city}, {state}" if city and state else job['location'].get('display_name', ""),
                'source':'Adzuna'
            }
            processed.append(new_job)
        return processed

    # ================================================================================ #
    # API Requests
    # ================================================================================ #
    
    async def pull_jobs(self, page:int=1, search:dict={}):
        url = self.adzuna_api_adapter(page, **search)
        session = self.session
        session_created = False

        if session is None:
            session = aiohttp.ClientSession()
            self.session = session
            session_created = True

        try:
            async with session.get(str(url)) as response:
                data = await response.json()
                return data['count'], self.process_data(data)
        finally:
            if session_created and self.session:
                await self.session.close()
                self.session = None

    async def full_run(self, searches_csv: str = ""):
        if self.session is None:
            self.session = aiohttp.ClientSession()
        try:
            all_jobs = []
            searches = self.pull_searches(searches_csv)
            for search in searches:
                page = 1
                while True:
                    count, job = await self.pull_jobs(page, search)
                    if search.get('rpp') and search['rpp'] * page < count:
                        page += 1
                        all_jobs.append(job)
                    else:
                        all_jobs.append(job)
                        break
            return all_jobs
        except Exception as e:
            error_logger_crash(f"Error pulling Adzuna data: {e}")
        finally:
            if self.session:
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
        print(f"Job output:/n{json.dumps(jobs)}")
    finally:
        if adapter.session:
            await adapter.session.close()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())