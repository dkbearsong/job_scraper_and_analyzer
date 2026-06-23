# Job Scraper and Analyzer

A multi-stage pipeline that scrapes job listings from company career pages and job boards (LinkedIn, Indeed, ZipRecruiter, Google), analyzes them against your resume and user profile using LLM extraction + semantic embeddings, applies rule-based filtering, performs multi-layered scoring (vector similarity + cheap LLM + strong LLM), and produces a ranked final application queue.

## Pipeline Architecture

The system processes jobs through **9 stages**, each building on the previous:

| Stage | Name | What It Does |
|-------|------|--------------|
| 0 | **Setup** | Loads `.env`, reads resume + user profile from `documents/`, creates AI engine, extracts skills/job titles via text processing |
| 1 | **Scrape** | Scrapes jobs via the pluggable adapter system (configured by `scrapers_config.yaml`). Falls back to legacy Part A (company career pages via `site_strategies/`) + Part B (job boards via JobSpy) if no config file is found. |
| 2 | **Embed + Extract** | Runs deterministic extraction (salary, seniority, work type, timezone), LLM extraction (skills, requirements, summary), and generates embedding vectors for title, skills, requirements, and description |
| 3 | **Rule Filter** | Applies hard-constraint rules from `user_preferences.yaml` (work type, seniority level, pay range, timezone) — jobs that fail are skipped |
| 4 | **Archetype Integration** | Loads benchmark archetypes from `archetypes_config.json` plus your Resume and User Profile as archetypes, generating embeddings for each |
| 5 | **Vector Scoring** | Computes weighted semantic similarity (40% title, 35% skills, 25% responsibilities) between jobs and archetypes, applies keyword/metadata adjustments, and filters by a configurable threshold |
| 6 | **Cheap LLM** | Uses a fast/cheap LLM to classify top candidates with fit scores and brief rationale |
| 7 | **Strong LLM** | Uses a more capable LLM to deeply rerank the top-N candidates with detailed analysis |
| 8 | **Final Queue** | Generates a final ranked application queue with priority levels, final scores, and apply/don't-apply recommendations |

## Directory Structure

```
.
├── main.py                          # Pipeline orchestrator + entry point
├── scrapers_config.yaml             # Pluggable scraper adapter configuration
├── .env                             # Configuration (paths, API keys, LLM settings)
├── user_preferences.yaml            # Rule filtering preferences
├── app/
│   ├── ai_engine.py                 # AI/LLM interface (extraction, embeddings)
│   ├── archetype_engine.py          # Archetype management and comparison
│   ├── fallback_scraping_instructions.py  # Legacy Part A + Part B scraping (fallback path)
│   ├── llm_classifier.py            # Cheap LLM + Strong LLM classification stages
│   ├── make_db.py                   # Database schema creation
│   ├── postgres_mgr.py              # PostgreSQL database manager
│   ├── pull_data.py                 # Site scraping (DataPuller)
│   ├── text_engine.py               # Text processing and deterministic extraction
│   ├── tui.py                       # Textual-based Terminal UI
│   ├── vector_engine.py             # Vector operations, scoring adjustments
│   └── scrapers/
│       ├── __init__.py              # ScraperAdapter ABC, JobData schema, validator
│       ├── adapter_loader.py        # AdapterLoader: reads YAML, imports + runs adapters
│       ├── validator.py             # ScrapedDataValidator: validates job data schema
│       ├── jobspy_adapter.py        # Built-in adapter for JobSpy library
│       ├── microservice_adapter.py  # Built-in adapter for local scraping microservice
│       ├── http_adapter.py          # Built-in adapter for HTTP/API endpoints
│       └── hiring_cafe_adapter.py   # Built-in adapter for hiring.cafe
├── documents/
│   ├── Your Name Resume.docx        # Your resume (Word format)
│   ├── User Profile - YourName.txt  # Your skills/profile (plain text)
│   └── archetypes_config.json       # Benchmark archetype definitions
├── site_strategies/
│   └── <company_name>.json          # Per-company scraping strategies (legacy Part A)
└── archetype_profiles/
    ├── resume_cache.json            # Cached resume data (auto-generated)
    ├── user_profile_cache.json      # Cached profile data (auto-generated)
    └── user_profile_<name>.json     # Cached user profile (auto-generated)
```

## Initial Setup

### 1. Required Documents

Place these files in the `documents/` folder. The paths to these files are configured in `.env`.

| File | Purpose | Format |
|------|---------|--------|
| **Resume** (`.docx`) | Your resume — used as an archetype for semantic comparison | Word document (`.docx`) |
| **User Profile** (`.txt`) | Plain-text file listing your skills, job titles, and professional summary. This should include a "Skills:" section and a "Job Titles:" section for deterministic extraction | Plain text (`.txt`) |
| **`archetypes_config.json`** | Benchmark archetype definitions — roles/job families to compare scraped jobs against. Each entry defines a name, title, skills, and responsibilities | JSON array |

**Example `archetypes_config.json`:**

```json
[
    {
        "name": "AI Tooling Engineer",
        "title": "Senior AI Tooling Engineer",
        "skills": "Python OpenAI LangChain Pinecone PyTorch LLMs",
        "responsibilities": "Develop AI-powered tools. Integrate LLMs into workflows."
    },
    {
        "name": "Backend Python Engineer",
        "title": "Senior Backend Python Engineer",
        "skills": "Python FastAPI PostgreSQL AWS Redis Docker",
        "responsibilities": "Design scalable APIs. Optimize database performance. Deploy cloud services."
    }
]
```

**Example User Profile (`User Profile - YourName.txt`):**

```
Skills: Python, FastAPI, PostgreSQL, AWS, Docker, Kubernetes, CI/CD, Terraform
Job Titles: Senior Backend Engineer, DevOps Engineer, Platform Engineer

Summary: Experienced backend engineer with 8+ years building scalable distributed systems.
```

### 2. Configure `.env`

Create a `.env` file in the project root. Below is a complete reference of all supported variables:

```ini
# ──────────────────────────────────────────────
# FILE PATHS
# ──────────────────────────────────────────────

RESUME='./documents/Your Name Resume.docx'
# Path to your resume .docx file.

PROFILE='./documents/User Profile - YourName.txt'
# Path to your user profile .txt file (skills, job titles, summary).

SEARCH_TERMS=''
# Optional: path to a CSV file of search terms used for job board scraping.
# If left blank, defaults to ["Software Engineer"].
# Example CSV content:
#   Senior Python Engineer
#   AI/ML Engineer
#   Platform Engineer

ARCHETYPES_CONFIG='./documents/archetypes_config.json'
# Path to the JSON file defining benchmark archetypes for comparison.

USER_PREFERENCES_YAML='./user_preferences.yaml'
# Path to the YAML file containing rule-based filtering preferences.

SCRAPERS_CONFIG='scrapers_config.yaml'
# Path to the scraper adapter configuration YAML file.
# Controls which scrapers run during Stage 1.

# ──────────────────────────────────────────────
# API KEYS
# ──────────────────────────────────────────────

GEMINMI_API_KEY=''
# Google Gemini API key. Required only if using Gemini as an LLM provider.
# Get a key at: https://aistudio.google.com/apikey

OPENROUTER_API_KEY=''
# OpenRouter API key. Required only if using OpenRouter as an LLM provider.
# Get a key at: https://openrouter.ai/keys

# ──────────────────────────────────────────────
# LLM PROVIDERS — GENERAL CONFIGURATION
# ──────────────────────────────────────────────
#
# The pipeline uses LLMs at multiple stages via the AIEngine abstraction layer.
# Each stage accepts a `provider_name` parameter. Supported providers:
#
#   "lm_studio"   — Local inference via LM Studio (http://localhost:{LMS_PORT})
#                    Works with any model loaded in LM Studio. Set LMS_URL, LMS_PORT below.
#
#   "ollama"      — Local inference via Ollama (http://localhost:{OLLAMA_PORT})
#                    Works with any model pulled to your local Ollama instance.
#                    Set OLLAMA_URL, OLLAMA_PORT below.
#
#   "openrouter"  — Remote API via OpenRouter (https://openrouter.ai/api/v1)
#                    Provides access to hundreds of models from many providers.
#                    Requires OPENROUTER_API_KEY. Set OPENROUTER_BASE_URL below.
#
#   "gemini"      — Google Gemini API. Requires GEMINMI_API_KEY to be set.
#
#   "openai"      — Direct OpenAI API. Requires OPENAI_API_KEY (not currently exposed,
#                    but the engine can be extended to support it).
#
# You can mix providers across stages. For example, use ollama for fast local
# extraction/embeddings and openrouter for deep analysis with a stronger model.
#
# The MODEL variables are provider-specific model identifiers:
#   - For lm_studio:  model name as loaded in the LM Studio server
#   - For ollama:     model tag from `ollama list` (e.g. "llama3.1:8b")
#   - For openrouter: model slug from openrouter.ai/models (e.g. "openai/gpt-4o")
#   - For gemini:     model name (e.g. "gemini-2.0-flash")
# Leave blank to use the provider's default model.

# ─── Stage 2: Extraction (skills, requirements, summary from job descriptions) ───
EXTRACTION_LLM='lm_studio'
# Provider for LLM-based extraction:
#   "lm_studio", "ollama", "openrouter", or "gemini".

EXTRACTION_MODEL=''
# Model identifier for extraction. Examples:
#   "gemma-4-26b-a4b-it-mlx"   (lm_studio)
#   "llama3.1:8b"              (ollama)
#   "openai/gpt-4o-mini"       (openrouter)
#   "gemini-2.0-flash"         (gemini)
# Leave blank to use the provider's default model.

# ─── Stage 2: Embeddings (vector generation for titles, skills, descriptions) ───
EMBEDDINGS_LLM='lm_studio'
# Provider for embedding generation:
#   "lm_studio", "ollama", "openrouter", or "gemini".

EMBEDDINGS_MODEL=''
# Model identifier for embeddings. Examples:
#   "qwen3-embedding-8b-mxfp8"  (lm_studio)
#   "nomic-embed-text:latest"   (ollama)
#   "openai/text-embedding-3-small"  (openrouter)
#   "text-embedding-004"        (gemini)
# Leave blank to use the provider's default model.

# ─── Stage 6: Cheap LLM Classification ───
CHEAP_LLM_PROVIDER='lm_studio'
# Provider for fast/cheap classification:
#   "lm_studio", "ollama", "openrouter", or "gemini".

CHEAP_LLM_MODEL=''
# Model identifier for cheap classification. Leave blank for provider default.

# ─── Stage 7: Strong LLM Reranking ───
STRONG_LLM_PROVIDER='lm_studio'
# Provider for deep reranking:
#   "lm_studio", "ollama", "openrouter", or "gemini".

STRONG_LLM_MODEL=''
# Model identifier for strong reranking. If left blank, falls back to CHEAP_LLM_MODEL.

# ─── Stage 7: How many top candidates get deep analysis ───
TOP_N_DEEP_ANALYSIS='25'
# Number of top candidates from Stage 6 to pass through the strong LLM reranker.
# Higher values = more thorough analysis but more API calls/cost.

# ──────────────────────────────────────────────
# LM STUDIO (ONLY NEEDED IF USING lm_studio PROVIDER)
# ──────────────────────────────────────────────

LMS_URL='http://localhost'
# Base URL of your LM Studio server.

LMS_PORT='1234'
# Port of your LM Studio server.

LMS_API_KEY='lm-studio'
# API key for LM Studio (defaults to "lm-studio").

# ──────────────────────────────────────────────
# OLLAMA (ONLY NEEDED IF USING ollama PROVIDER)
# ──────────────────────────────────────────────
#
# Ollama runs locally and serves an OpenAI-compatible API.
# Install from https://ollama.com then pull models with:
#   ollama pull llama3.1:8b
#   ollama pull nomic-embed-text
#
# Make sure the Ollama service is running before using the pipeline
# (it runs as a background service on install, or start with `ollama serve`).

OLLAMA_URL='http://localhost'
# Base URL of your Ollama server.

OLLAMA_PORT='11434'
# Port of your Ollama server (default: 11434).

OLLAMA_API_KEY=''
# API key for Ollama (typically not needed for local use; leave blank).

# ──────────────────────────────────────────────
# OPENROUTER (ONLY NEEDED IF USING openrouter PROVIDER)
# ──────────────────────────────────────────────
#
# OpenRouter provides a unified API for hundreds of models from
# OpenAI, Anthropic, Google, Meta, Mistral, and many more.
# See available models at https://openrouter.ai/models
#
# Set the model identifier using the format "provider/model-name",
# for example:
#   - "openai/gpt-4o"
#   - "anthropic/claude-sonnet-4-20250514"
#   - "google/gemini-2.0-flash-001"
#   - "meta-llama/llama-3.1-70b-instruct"
#   - "mistralai/mixtral-8x22b-instruct"

OPENROUTER_BASE_URL='https://openrouter.ai/api/v1'
# Base URL for the OpenRouter API.
# Change only if you are using a self-hosted or alternative endpoint.

OPENROUTER_API_KEY=''
# OpenRouter API key. Required for openrouter provider.
# Get one at https://openrouter.ai/keys

# ──────────────────────────────────────────────
# SCRAPER API (OPTIONAL)
# ──────────────────────────────────────────────

SCRAPER_API=''
# URL for an external scraping API. If left blank, the DataPuller scrapes directly.
```

### 3. User Preferences (`user_preferences.yaml`)

This file controls hard-constraint rule filtering in **Stage 3**. Jobs that don't match are flagged as "skip" and excluded from further analysis.

```yaml
# User Preferences for Job Filtering
#
# work_types:       Accepted work arrangements. Options: remote, hybrid, on-site
# seniority_levels: Accepted seniority levels. Options: entry-level, mid-level, senior, lead, manager
# timezones:        Accepted timezones. Options: EST, CST, MST, PST, GMT, etc.
# pay_range:        Target annual pay range as a single string (e.g. "$95k-200k" or "95000-200000")
# target_cities:    Cities to search for jobs on job boards

target_cities:
  - "Austin, TX"
  - "Seattle, WA"
  - "Boston, MA"

work_types:
  - remote
  - hybrid

seniority_levels:
  - mid-level
  - senior

timezones:
  - EST
  - CST
  - MST

pay_range: "$95k-200k"
```

**How filtering works:** If a field is set, only jobs matching one of the listed values pass through. Leave a list empty or unset to allow all values for that field. If pay is "Not Specified" for a job, it passes through without comparison.

### 4. Configure `scrapers_config.yaml`

This file controls all scraping during **Stage 1**. It defines a list of adapter entries, each specifying a source to scrape from. If this file is not present, the pipeline falls back to the legacy hardcoded scraping path (Part A: `site_strategies/` + Part B: JobSpy job boards).

**Full config reference (`scrapers_config.yaml`):**

```yaml
# Scrapers Configuration
# =====================
# Each entry defines a scraper adapter to run during Pipeline Stage 1.
#
# Fields:
#   name:     Unique identifier for this scraper source
#   adapter:  Dotted Python module path OR built-in short name:
#               - "JobSpyAdapter"       Built-in: job boards via JobSpy
#               - "MicroserviceAdapter"  Built-in: company career page microservice
#               - "HttpAdapter"          Built-in: HTTP/API endpoint
#               - "my_module.MyClass"   Custom adapter (must subclass ScraperAdapter)
#   enabled:  true/false (default: true)
#   config:   Adapter-specific configuration (see each adapter's docs)
#
# Environment variables can be referenced as ${VAR_NAME} in config values.

scrapers:
  # ── Company Career Pages via Local Microservice ──
  - name: microservice_company_boards
    adapter: MicroserviceAdapter
    enabled: true
    config:
      sites_file: "${JOB_SITES}"
      strategies_dir: "./site_strategies"
      microservice_host: "http://localhost"
      microservice_port: "5052"
      timeout: 120
      delay_min: 1
      delay_max: 3

  # ── Job Boards via JobSpy (Indeed, LinkedIn, ZipRecruiter, Google) ──
  - name: jobspy_boards
    adapter: JobSpyAdapter
    enabled: true
    config:
      boards:
        - indeed
        - linkedin
        - zip_recruiter
        - google
      search_terms:
        - Software Engineer
      target_cities:
        - Remote
      requests_wanted: 200
      hours_old: 24
      delay_min: 1
      delay_max: 4
      country_indeed: "USA"

  # ── HTTP/API Endpoint (example) ──
  # - name: my_remote_scraper
  #   adapter: HttpAdapter
  #   enabled: false
  #   config:
  #     url: "https://my-scraper-service.com/api/scrape"
  #     method: POST
  #     timeout: 120
  #     headers:
  #       Authorization: "Bearer ${MY_API_TOKEN}"

  # ── HiringCafe (live browser-based scraper) ──
  - name: hiring_cafe
    adapter: app.scrapers.hiring_cafe_adapter.HiringCafeAdapter
    enabled: true
    config:
      max_pages: 5
      headless: false
```

**Adapter-specific `config` reference:**

| Adapter | Config Key | Type | Default | Description |
|---------|-----------|------|---------|-------------|
| **MicroserviceAdapter** | `sites_file` | str | `""` | Path to CSV file listing company names + URLs |
| | `strategies_dir` | str | `"./site_strategies"` | Directory containing site strategy JSON files |
| | `microservice_host` | str | `"http://localhost"` | Microservice host |
| | `microservice_port` | str | `"5052"` | Microservice port |
| | `timeout` | int | `120` | Request timeout (seconds) |
| | `delay_min` | float | `1.0` | Min delay between requests |
| | `delay_max` | float | `3.0` | Max delay between requests |
| **JobSpyAdapter** | `boards` | list | `["indeed"]` | Job boards to scrape (indeed, linkedin, zip_recruiter, google) |
| | `search_terms` | list | `["Software Engineer"]` | Search terms/queries |
| | `target_cities` | list | `["Remote"]` | Cities to search in |
| | `requests_wanted` | int | `200` | Results per board/search |
| | `hours_old` | int | `24` | Filter: listings within N hours |
| | `delay_min` / `delay_max` | float | `1` / `4` | Rate limiting delay range |
| | `country_indeed` | str | `"USA"` | Country for Indeed searches |
| **HttpAdapter** | `url` | str | *(required)* | Endpoint URL |
| | `method` | str | `"POST"` | HTTP method (GET or POST) |
| | `headers` | dict | `{}` | Custom HTTP headers |
| | `timeout` | int | `120` | Request timeout |
| | `payloads` | list | `[]` | Optional request payloads (one request each) |
| | `payloads_file` | str | `""` | JSON file containing payloads |
| | `auth_header` | str | `""` | Authorization header value |
| | `delay_min` / `delay_max` | float | `1` / `3` | Rate limiting delay |
| **HiringCafeAdapter** | `max_pages` | int | `5` | Max pages to scrape |
| | `headless` | bool | `false` | Run browser in headless mode |

### 5. Site Strategies (Company Career Page Scraping)

To scrape jobs from specific company career pages (legacy Part A or MicroserviceAdapter), create a `site_strategies/` directory and add a JSON file for each company. The filename (without `.json`) must match an entry in the `JOB_SITES` list.

**Example `site_strategies/example_company.json`:**

```json
{
    "company_url": "https://careers.example.com",
    "url": "https://careers.example.com/jobs",
    "fields": [
        {"selector": "h2.job-title", "name": "title"},
        {"selector": "span.location", "name": "location"},
        {"selector": ".description", "name": "description"}
    ],
    "pagination": {
        "parameter": "page",
        "start": 1,
        "max": 5
    }
}
```

**Strategy fields:**

| Field | Purpose |
|-------|---------|
| `url` | The URL to scrape (can be a string or list of strings for multiple pages) |
| `company_url` | Base company URL for resolving relative links |
| `fields` | CSS selectors mapping to title, location, description, etc. |
| `pagination` | (Optional) If present, the scraper uses `extract-paginated` method |
| `js_config` | (Optional) If present, the scraper uses `extract-js` method (browser rendering) |

**Loading strategies:**
1. Add company entries to `sites` list in the `DataPuller`'s JOB_SITES file (legacy) or configure `MicroserviceAdapter` in `scrapers_config.yaml`
2. Create a corresponding `site_strategies/<company_name>.json` file
3. The pipeline automatically loads and applies these strategies during Stage 1

### 6. Search Terms for Job Boards (Optional)

If you want to control which job titles/roles are searched on Indeed, LinkedIn, etc., create a CSV file (e.g., `search_terms.csv`) and set `SEARCH_TERMS` in `.env`:

```
Senior Python Engineer
AI/ML Engineer
Platform Engineer
DevOps Engineer
```

If left unset, the pipeline defaults to `["Software Engineer"]`.

## Writing a Custom Scraper Adapter

You can extend the pipeline by writing a custom scraper adapter and registering it in `scrapers_config.yaml`. The adapter system dynamically imports, configures, and runs any class that subclasses `ScraperAdapter`.

### Required Interface

Every adapter must subclass `ScraperAdapter` (defined in `app/scrapers/__init__.py`) and implement these three methods:

```python
from typing import Any, Dict, List
from app.scrapers import ScraperAdapter

class MyCustomAdapter(ScraperAdapter):
    def get_name(self) -> str:
        """
        Return a human-readable name for this adapter (used in logs).
        The name from scrapers_config.yaml is injected automatically.
        """
        return self._config.get("name", "my_custom_adapter")

    def configure(self, config: Dict[str, Any]) -> None:
        """
        Accept adapter-specific configuration from scrapers_config.yaml.
        Store whatever settings your adapter needs.
        """
        self._config = config
        # Example: self._api_key = config.get("api_key", "")
        #          self._max_results = config.get("max_results", 100)

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Execute the scraping operation.
        
        Returns:
            List of job dicts conforming to the JobData schema.
            Each dict must at minimum contain 'title' and 'company' keys.
        """
        # Your scraping logic here
        jobs = []
        # ...
        return jobs
```

### Job Data Schema

Your adapter's `scrape()` method must return a list of dicts conforming to this schema:

| Field | Required | Type | Description |
|-------|----------|------|-------------|
| `title` | **Yes** | str | Job title |
| `company` | **Yes** | str | Company name |
| `source` | No | str | Origin identifier (e.g. "indeed", "custom_api") |
| `url` | No | str or None | Job listing URL |
| `link` | No | str or None | Alias for url (backward compat) |
| `flexibility` | No | str or None | Work type: "remote", "hybrid", "onsite", "NA" |
| `pay` | No | str or None | Pay range (e.g. "$80k-$120k") |
| `location` | No | str or None | Full location (e.g. "Austin, TX") |
| `description` | No | str or None | Job description text |
| `city` | No | str or None | City name |
| `state` | No | str or None | State abbreviation |
| `company_url` | No | str or None | Company career page URL |

### Registering in `scrapers_config.yaml`

Add an entry under `scrapers:` with the dotted module path to your adapter class:

```yaml
scrapers:
  - name: my_custom_source
    adapter: my_package.scrapers.MyCustomAdapter
    enabled: true
    config:
      api_key: "${MY_API_KEY}"
      max_results: 100
      # any other adapter-specific settings
```

Configuration values support `${ENV_VAR}` syntax for referencing environment variables. The `name` field is automatically injected into the config dict so `get_name()` can access it.

### Reference Example

Here is the `HiringCafeAdapter` (from `app/scrapers/hiring_cafe_adapter.py`) as a complete working example:

```python
from typing import Any, Dict, List
from app.scrapers import ScraperAdapter

class HiringCafeAdapter(ScraperAdapter):
    def get_name(self) -> str:
        return self._config.get("name", "hiring_cafe")

    def configure(self, config: Dict[str, Any]) -> None:
        self._config = config
        self._max_pages = config.get("max_pages", 5)
        self._headless = config.get("headless", False)

    async def scrape(self) -> List[Dict[str, Any]]:
        # ... implementation returns List[Dict] with title, company, etc.
        pass
```

### Fallback Path

If `scrapers_config.yaml` is not found or contains no enabled adapters, the pipeline falls back to the legacy scraping logic in `app/fallback_scraping_instructions.py`. This legacy path runs:
- **Part A:** Company career pages using `site_strategies/` JSON files via a local microservice
- **Part B:** Job boards (Indeed, LinkedIn, ZipRecruiter, Google) via the JobSpy library

To migrate from the legacy path to the adapter system, simply create `scrapers_config.yaml` with the appropriate adapter entries.

## Running the Application

### Option A: Terminal UI (Recommended)

The TUI (Terminal User Interface) provides a visual dashboard for running and monitoring the pipeline:

```bash
python main.py --tui
# or
python main.py -t
```

**TUI Features:**
- **Status bar** — Shows pipeline status (idle/running/complete/error), current stage, elapsed time, and total job count
- **Sidebar** — Pipeline stage list with live status indicators (○ pending, ◉ running, ● complete, ⊗ error), job count breakdown per stage, and control buttons (Run / Stop / Reset)
- **Tabbed content** — Pipeline overview with live logs, Jobs data table, Job detail panel, and Full logs
- **Keyboard shortcuts:**
  - `F5` — Run pipeline
  - `F6` — Stop pipeline
  - `F7` — Reset pipeline
  - `Ctrl+P` — Focus Pipeline tab
  - `Ctrl+J` — Focus Jobs tab
  - `Ctrl+D` — Focus Detail tab
  - `Ctrl+L` — Focus Logs tab
  - `Ctrl+Q` / `Q` / `Escape` — Quit
- **Click a job row** in the Jobs tab to see full details (skills, scores, LLM analysis, strengths/concerns)

### Option B: Command Line (Headless)

Run the full pipeline end-to-end with console output only:

```bash
python main.py
```

This executes all 9 stages sequentially, printing progress summaries as it goes.

### Database Persistence

By default, the pipeline stores scraped jobs, embeddings, LLM results, and the final queue to a PostgreSQL database. The TUI runs with `skip_db=True` to avoid requiring a database connection during development or demos.

To configure PostgreSQL, set these in `.env` (used by `DataPuller`):

```ini
DB_NAME='job_scraper'
DB_USER='postgres'
DB_PASSWORD='your_password'
DB_HOST='localhost'
DB_PORT='5432'
```

The database schema is managed by `app/make_db.py`. Run it separately to initialize tables:

```bash
python -c "from app.make_db import init_db; init_db()"
```

## Understanding the Output

After a full pipeline run, each job in the final queue contains:

| Field | Source | Meaning |
|-------|--------|---------|
| `final_score` | Stage 8 | Overall score (0-100) combining all previous stages |
| `priority` | Stage 8 | Priority level: `high`, `medium`, or `low` |
| `apply_recommendation` | Stage 8 | Recommendation: `apply`, `maybe`, or `skip` |
| `semantic_score` / `semantic_score_percent` | Stage 5 | Vector similarity score (0.0-1.0 / 0-100%) against best-matching archetype |
| `best_archetype` | Stage 5 | Name of the archetype that most closely matches this job |
| `cheap_llm_result.fit_score` | Stage 6 | Fast LLM fit assessment (0-100) |
| `cheap_llm_result.decision` | Stage 6 | Fast LLM decision: `yes`, `maybe`, `no` |
| `strong_llm_result.final_score` | Stage 7 | Deep LLM analysis score (0-100) |
| `strong_llm_result.decision` | Stage 7 | Deep LLM decision with confidence level |