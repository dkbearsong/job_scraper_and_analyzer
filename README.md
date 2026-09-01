# Job Scraper and Analyzer

A multi-stage agentic pipeline that scrapes job listings from API services (Adzuna), job boards (Indeed, LinkedIn, ZipRecruiter, Google), and company career portals. It filters and evaluates them against your professional profile, resume, and benchmark archetypes using a hybrid architecture of deterministic extraction, vector similarity (pgvector), and multi-tiered LLM evaluation (fast cheap classification + deep qualitative reranking) to produce a ranked final application queue.

---

## Table of Contents

- [Pipeline Architecture](#pipeline-architecture)
- [Quick Start & Installation](#quick-start--installation)
- [Directory Structure](#directory-structure)
- [Key Features & Recent Advancements](#key-features--recent-advancements)
  - [Pluggable Scraper Adapters](#1-pluggable-scraper-adapters)
  - [Deterministic Extraction & Stage 1.5 Preliminary Filter](#2-deterministic-extraction--stage-15-preliminary-filter)
  - [Multi-Provider Hybrid LLM Architecture](#3-multi-provider-hybrid-llm-architecture)
  - [Rate Limiting & Tiered Throttling](#4-rate-limiting--tiered-throttling)
  - [Role-Specific Seniority & Geocoding Radius](#5-role-specific-seniority--geocoding-radius)
  - [Negative Vector Scoring](#6-negative-vector-scoring)
  - [RAG Retrieval & Application Tailoring](#7-rag-retrieval--application-tailoring)
  - [Model Context Protocol (MCP) Server](#8-model-context-protocol-mcp-server)
  - [Embedding Regeneration Utility](#9-embedding-regeneration-utility)
  - [Pipeline Test Runner](#10-pipeline-test-runner)
- [Writing Custom Scraper Adapters](#writing-custom-scraper-adapters)
- [Running the Application](#running-the-application)
  - [CLI Flags Reference](#cli-flags-reference)
  - [Common Usage Examples](#common-usage-examples)
- [Understanding Pipeline Output](#understanding-pipeline-output)

---

## Pipeline Architecture

The system processes jobs through a 10-step sequence designed for maximum cost-efficiency, filtering out unqualified listings early to conserve LLM tokens and API calls:

```
[Stage 0: Setup] ──> [Stage 1: Scrape] ──> [Stage 1.5: Preliminary Filter]
                                                         │ (disqualifies non-matches)
                                                         ▼
[Stage 4: Archetypes] ◄── [Stage 3: Rule Filter] ◄── [Stage 2: Embed & Extract]
         │
         ▼
[Stage 5: Vector Scoring] ──> [Stage 6: Cheap LLM] ──> [Stage 7: Strong LLM] ──> [Stage 8: Final Queue]
```

| Stage | Name | What It Does |
|-------|------|--------------|
| **0** | **Setup** | Loads `.env` and `user_preferences.yaml`, extracts skills and titles from resume (`.docx`) and user profile (`.txt`), initializes the AI engine, and configures the database connection. |
| **1** | **Scrape** | Executes pluggable scraper adapters configured in `scrapers_config.yaml` (e.g. Adzuna API, JobSpy boards, or local career microservice). Merges newly scraped jobs with deduplication. |
| **1.5** | **Preliminary Filter** | Fast deterministic pre-filter on title keywords, basic location, and work type *before* running expensive embeddings or LLM calls, immediately discarding obvious mismatches. |
| **2** | **Embed + Extract** | Runs deterministic regex extraction for salary, work flexibility, seniority, and timezone. Uses the extraction LLM for unstructured skills/requirements/summary and generates dense vector embeddings (title, requirements, responsibilities). |
| **3** | **Rule Filter** | Evaluates hard constraints from `user_preferences.yaml` (work types, seniority levels, role-specific seniority overrides, geocoded target city radius, pay range, and timezones). Failing jobs are flagged as `skip`. |
| **4** | **Archetype Integration** | Loads benchmark archetypes from `documents/archetypes_config.json` alongside your resume and profile archetypes, computing or loading their vector embeddings. Prepares negative scoring penalty criteria. |
| **5** | **Vector Scoring** | Calculates multi-vector cosine similarity (weighted: 40% title, 35% skills/requirements, 25% responsibilities) against archetypes. Applies negative penalties for unwanted titles/functions, and filters by similarity threshold. |
| **6** | **Cheap LLM** | Classifies candidate jobs using a fast, economical model (e.g. Gemini Flash Lite, local LM Studio / Ollama, or Groq), evaluating core fit, strengths, concerns, and outputting an initial `fit_score` (0-100) and `decision` (`yes`, `maybe`, `no`). |
| **7** | **Strong LLM** | Executes deep qualitative reranking on the top-N candidates using an advanced model (e.g. Gemini Pro, Claude 3.5 Sonnet, or GPT-4o), evaluating career trajectory, scale fit, recruiter red flags, and driving points. |
| **8** | **Final Queue** | Synthesizes scores from all stages into a final weighted score (0-100), assigning priority levels (`high`, `medium`, `low`) and actionable recommendations (`apply`, `maybe`, `skip`). Saves to database and logs results. |

---

## Quick Start & Installation

> [!IMPORTANT]
> All setup prerequisites, system requirements, database configuration (`pgvector`), environment variable references, and step-by-step guides have been centralized in **[INSTALL.md](INSTALL.md)**.

### Automated Setup

Run the interactive installation script from the project root:

```bash
./install.sh
```

Or in non-interactive mode:

```bash
./install.sh -y
```

### Manual Installation Summary

1. **Virtual Environment**: `python3 -m venv .venv && source .venv/bin/activate`
2. **Install Dependencies**: `pip install -r requirements.txt && playwright install chromium`
3. **Database**: PostgreSQL 14+ with `pgvector` extension enabled (`CREATE EXTENSION IF NOT EXISTS vector;`)
4. **Initialize Tables**: `python -c "from app.make_db import make_db; make_db()"`
5. **Configuration**: Copy `.env.example` to `.env` and customize `user_preferences.yaml`
6. **Self-Test**: `python tests/test_runner.py --stage 0 --skip-db`

For complete instructions, troubleshooting, and Docker setup, please refer to **[INSTALL.md](INSTALL.md)**.

---

## Directory Structure

```
.
├── INSTALL.md                       # Comprehensive installation & operational setup guide
├── README.md                        # Project overview, architecture, and operation reference
├── install.sh                       # Automated installation and dependency setup script
├── main.py                          # Pipeline orchestrator and CLI entry point
├── scrapers_config.yaml             # Scraper adapter configuration (Adzuna, JobSpy, etc.)
├── user_preferences.yaml            # Filtering rules, LLM providers/models, rate limits
├── .env.example                     # Sanitized environment configuration template
├── .env                             # Local secrets, database credentials, and file paths
├── job_sites.csv                    # Company career sites for scraping
├── diagnose_db.py                   # Database network and authentication connectivity tester
├── mcp_server.py                    # Model Context Protocol (MCP) server for AI assistants
├── regenerate_embeddings.py         # Utility for recomputing vector embeddings in PostgreSQL
├── requirements.txt                 # Python package dependencies
├── app/
│   ├── ai_engine.py                 # Multi-provider LLM interface (LM Studio, Gemini, OpenRouter, etc.)
│   ├── ai_limiter.py                # Concurrency and rate-limiting throttle (Free vs Paid tiers)
│   ├── ai_utils.py                  # Embedding generation and batching utilities
│   ├── archetype_engine.py          # Benchmark role archetypes and negative scoring criteria
│   ├── backfill_pay_location_embeddings.py  # Utility for backfilling legacy embeddings
│   ├── config_utils.py              # Configuration loader (.env + user_preferences.yaml)
│   ├── eval_harness.py              # Prompt evaluation harness for extraction accuracy
│   ├── fallback_scraping_instructions.py  # Fallback scraping (career sites + JobSpy)
│   ├── geocoding_cache.json         # Local persistent geocoding coordinate cache
│   ├── llm_classifier.py            # Cheap LLM (Stage 6) & Strong LLM (Stage 7) evaluators
│   ├── llm_usage_tracker.py         # Token consumption and cost auditing logger
│   ├── location_utils.py            # Geocoding and geographic radius calculations
│   ├── logger.py                    # Structured logging and pipeline statistics
│   ├── make_db.py                   # PostgreSQL schema creation and pgvector initialization
│   ├── postgres_mgr.py              # Low-level PostgreSQL connection and query manager
│   ├── prompt_injection_defender.py # Defense against prompt injections in job descriptions
│   ├── pull_data.py                 # DataPuller database persistence and retrieval layer
│   ├── rag_engine.py                # Hybrid vector + metadata retrieval and application tailoring
│   ├── rule_filters.py              # Deterministic filtering rules (salary, seniority, timezone)
│   ├── text_engine.py               # Deterministic text processing & regex extraction
│   ├── vector_engine.py             # Vector operations, similarity scoring, negative penalties
│   ├── pipeline/
│   │   ├── stages.py                # Implementations for Pipeline Stages 0 through 8
│   │   ├── pipeline_utils.py        # Stage routing, stage range parsing, pool merging
│   │   └── maintenance.py           # 24h description rescraping & final score recalculation
│   └── scrapers/
│       ├── __init__.py              # ScraperAdapter abstract base class & JobData schema
│       ├── adapter_loader.py        # Dynamic adapter loader driven by scrapers_config.yaml
│       ├── adzuna_adapter.py        # Adzuna job search API adapter with full description fetching
│       ├── jobspy_adapter.py        # Multi-board scraper adapter (Indeed, LinkedIn, Google, ZipRecruiter)
│       ├── microservice_adapter.py  # Local microservice adapter for company career pages
│       ├── http_adapter.py          # Generic HTTP REST API scraper adapter
│       └── validator.py             # Scraped job dictionary validator
├── documents/
│   ├── Your Resume.docx             # User resume used for archetype comparisons
│   ├── User Profile.txt             # Plain-text user skills, titles, and summary
│   ├── archetypes_config.json       # Benchmark target role definitions
│   ├── search_terms.csv             # JobSpy search queries
│   └── adzunda_searches.csv         # Adzuna search queries and parameters
├── logs/                            # Application logs, app_error.log, and pipeline_stats.log
└── tests/
    ├── test_runner.py               # Comprehensive pipeline stage runner with synthetic data
    └── test_*.py                    # Unit and integration test suites
```

---

## Key Features & Recent Advancements

### 1. Pluggable Scraper Adapters

Scraping in Stage 1 is fully decoupled using the `ScraperAdapter` architecture. Instead of hardcoding sources, adapters are declared in `scrapers_config.yaml`:
- **Adzuna Adapter (`AdzunaAdapter`)**: Connects to the Adzuna API, executing targeted searches defined in `documents/adzunda_searches.csv`, automatically deduplicating against the database, and scraping full descriptions.
- **JobSpy Adapter (`JobSpyAdapter`)**: Queries Indeed, LinkedIn, ZipRecruiter, and Google with configurable freshness (`hours_old`) and result limits.
- **Microservice Adapter (`MicroserviceAdapter`)**: Interfaces with a local scraping microservice using JSON strategy files from `site_strategies/`.
- **Custom HTTP Adapter (`HttpAdapter`)**: Allows querying arbitrary external endpoints.
- **Fallback Chaining**: Set `run_fallback_after_adapters: true` in `scrapers_config.yaml` to run legacy scrapers after modular adapters have finished.

### 2. Deterministic Extraction & Stage 1.5 Preliminary Filter

To drastically cut LLM API costs and execution time:
- **Preliminary Filtering (Stage 1.5)**: Disqualifies obvious non-matches immediately after scraping based on title keywords/regex disqualifiers (`disqualified_titles` in `user_preferences.yaml`), seniority, pay, and arrangement without making costly embedding or LLM calls. Failing jobs are marked `skip` in the database and purged from active pipeline memory.
- **Deterministic Text Engine**: Extracts salary numbers/ranges, work arrangements (remote/hybrid/onsite), seniority levels, and US timezones using optimized regex patterns in `app/text_engine.py`, reserving LLM calls strictly for ambiguous descriptions.
- **Prompt Injection Defense**: Untrusted job descriptions are sanitized by `app/prompt_injection_defender.py` before passing into LLM evaluation prompts.

### 3. Multi-Provider Hybrid LLM Architecture

The pipeline supports mixing and matching LLM providers for different stages in `user_preferences.yaml`:
- **Local Inference**: LM Studio, Ollama, native `sentence-transformers`, `fastembed`.
- **Cloud Providers**: Google Gemini, OpenRouter, OpenAI, Anthropic Claude, Groq, NVIDIA NIM, Cohere, SiliconFlow.
- **Example Strategy**: Use local `sentence-transformers` for embeddings (zero cost), local LM Studio or Groq for Stage 2 extraction, Gemini Flash Lite for Stage 6 cheap classification, and Claude 3.5 Sonnet or GPT-4o via OpenRouter for Stage 7 strong reranking.

### 4. Rate Limiting & Tiered Throttling

`app/ai_limiter.py` provides rate-limiting to prevent `429 Too Many Requests` errors:
- **Free vs. Paid Tiers**: Configurable in `user_preferences.yaml` with explicit `requests_per_minute`, `tokens_per_minute`, and `concurrency` limits.
- Set `STAGE_6_TIER='free'` or `STAGE_7_TIER='paid'` in `.env` to automatically throttle API requests to match your tier limits.

### 5. Role-Specific Seniority & Geocoding Radius

- **Role Seniority Overrides**: Allows fine-tuning allowed seniority levels per job title. For instance, requiring "senior" for Software Engineer, but accepting "mid-level" or "lead" for Support Engineer.
- **Geocoding & Radius Filtering**: Target cities configured in `user_preferences.yaml` (e.g. `target_city_range: 50`) calculate true geographic distance using cached coordinates in `app/geocoding_cache.json`.

### 6. Negative Vector Scoring

Jobs matching unwanted job functions or career paths are penalized during Stage 5 vector scoring:
- Configured under `negative_scoring` in `user_preferences.yaml`.
- Computes cosine similarity against `avoid_titles` (e.g. "Sales Representative", "Account Executive") and `avoid_functions` (e.g. "cold calling outbound quota prospecting").
- Jobs exceeding the threshold receive a proportional score reduction (up to `penalty_weight`, e.g. 35%).

### 7. RAG Retrieval & Application Tailoring

Built-in retrieval-augmented generation engine (`app/rag_engine.py`) backed by `pgvector` HNSW indexes:
- **Corpus Natural Language Query**:
  ```bash
  python main.py --rag-query "Find remote Python roles that mention Kubernetes and distributed systems"
  ```
- **Job Application Tailoring**:
  ```bash
  python main.py --rag-tailor 142
  ```
  Generates targeted resume bullet points, key strengths to highlight, and talking points tailored specifically for Job ID 142.

### 8. Model Context Protocol (MCP) Server

Expose your job search pipeline directly to AI assistants like Claude Desktop or Cursor via `mcp_server.py`:
- Tools exposed:
  - `search_job_corpus`: Hybrid vector + metadata semantic search across all scraped jobs.
  - `tailor_application`: Generates a custom tailoring strategy for a given Job ID.
  - `get_pipeline_status`: Reports stats on active jobs, extractions, and scores.
  - `list_top_jobs`: Returns the highest-ranked jobs from the final application queue.

### 9. Embedding Regeneration Utility

If you change your embedding model (e.g. switching from `all-MiniLM-L6-v2` to a larger model):
```bash
# Regenerate embeddings for jobs added in the last 14 days
python regenerate_embeddings.py --days-back 14

# Regenerate all jobs and archetypes
python regenerate_embeddings.py --all-jobs
```

### 10. Pipeline Test Runner

Test any stage or the entire pipeline using synthetic fixtures without requiring external scrapers or live database connections:
```bash
# List all stages and their input/output contracts
python tests/test_runner.py --list-stages

# Test Stage 3 (Rule Filtering) in isolation
python tests/test_runner.py --stage 3 --skip-db

# Test chained execution through Stage 5
python tests/test_runner.py --from 2 --to 5 --chain --skip-db
```

---

## Writing Custom Scraper Adapters

You can easily add new job sources by writing a subclass of `ScraperAdapter` (defined in `app/scrapers/__init__.py`) and registering it in `scrapers_config.yaml`.

### Required Adapter Interface

```python
from typing import Any, Dict, List
from app.scrapers import ScraperAdapter

class CustomJobBoardAdapter(ScraperAdapter):
    def get_name(self) -> str:
        """Return the adapter identifier used in logs."""
        return self._config.get("name", "custom_job_board")

    def configure(self, config: Dict[str, Any]) -> None:
        """Receive adapter-specific configuration from scrapers_config.yaml."""
        self._config = config
        self._api_key = config.get("api_key", "")
        self._max_results = config.get("max_results", 50)

    async def scrape(self) -> List[Dict[str, Any]]:
        """
        Execute scraping and return a list of job dictionaries.
        Must conform to the JobData schema (at minimum: 'title' and 'company').
        """
        jobs = []
        # ... perform API request or web scrape ...
        jobs.append({
            "title": "Senior Backend Engineer",
            "company": "Acme Corp",
            "source": "custom_job_board",
            "url": "https://example.com/jobs/123",
            "location": "Austin, TX",
            "flexibility": "remote",
            "pay": "$130k-$170k",
            "description": "Full job description text...",
        })
        return jobs
```

### Registering in `scrapers_config.yaml`

```yaml
scrapers:
  - name: my_job_source
    adapter: my_module.scrapers.CustomJobBoardAdapter
    enabled: true
    config:
      api_key: "${CUSTOM_API_KEY}"
      max_results: 100
```

---

## Running the Application

### Basic Command

Execute the complete end-to-end pipeline (Stages 0 through 8):

```bash
python main.py
```

### CLI Flags Reference

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `-s, --stage` | str | `"0-8"` | Stage range to run. Accepts a single stage (`3`), a range (`2-5`), or all (`0-8`). |
| `-l, --limit` | int | `50` | Maximum records to load from PostgreSQL in bulk-load operations when resuming. |
| `--skip-db` | flag | `False` | Disables database persistence across all stages (runs purely in-memory). |
| `--verbose` | flag | `False` | Enables detailed per-job debug logging across all pipeline stages. |
| `--debug` | flag | `False` | Enables debug-level logging output. |
| `--log-file` | str | `None` | Path to custom log file (stored under `logs/`). |
| `--pages` | int | `None` | Override max scraping pages per site for Stage 1. |
| `--visible` | flag | `False` | Shows browser window during browser-based scraping (disables headless). |
| `--skip-part-a` | flag | `False` | Skips company career-page scraping in the legacy fallback path. |
| `--scrape-missing-24h` | flag | `False` | At Stage 1, only re-scrapes descriptions for jobs from the last 24h that lack them. |
| `--reprocess` | int/flag | `0` | Clears database extractions and scores for the specified number of days (default: 1 day if flag is given without a value) to allow re-running stages. |
| `--recalculate-final-scores` | flag | `False` | Re-evaluates final scores and priority rankings from stored database scores without re-running models. |
| `--rag-query` | str | `None` | Executes a natural language query over the pgvector job corpus and prints an answer with citations. |
| `--rag-tailor` | int | `None` | Generates a tailored resume and application talking points for a specific Job ID. |

### Common Usage Examples

```bash
# Run the full pipeline
python main.py

# Run only rule filtering (Stage 3) with verbose output
python main.py -s 3 --verbose

# Run embedding generation through cheap classification (Stages 2-6)
python main.py -s 2-6

# Offline demo run without requiring PostgreSQL
python main.py --skip-db

# Scrape jobs with a visible browser window and debug logging
python main.py -s 1 --visible --debug

# Reprocess and re-evaluate jobs from the last 7 days
python main.py --reprocess 7 -s 2-8

# Recalculate final application queue scores after tuning weights in user_preferences.yaml
python main.py --recalculate-final-scores

# Query the job corpus using RAG
python main.py --rag-query "What companies are hiring remote engineers with FastAPI and Docker?"

# Generate tailored application strategy for Job ID 42
python main.py --rag-tailor 42
```

---

## Understanding Pipeline Output

Each job that completes the pipeline receives comprehensive evaluation data stored in PostgreSQL:

| Field | Source | Meaning |
|-------|--------|---------|
| `final_score` | Stage 8 | Normalized overall score (0-100) combining vector similarity, Cheap LLM, and Strong LLM evaluations. |
| `priority` | Stage 8 | Action priority: `high`, `medium`, or `low`. |
| `apply_recommendation` | Stage 8 | Final recommendation: `apply`, `maybe`, or `skip`. |
| `semantic_score` | Stage 5 | Vector similarity score (0.0 to 1.0) against the closest archetype. |
| `best_archetype` | Stage 5 | Name of the archetype that best matches the job. |
| `cheap_llm_result.fit_score` | Stage 6 | Fast LLM fit assessment (0-100). |
| `cheap_llm_result.decision` | Stage 6 | Fast LLM screening decision: `yes`, `maybe`, `no`. |
| `cheap_llm_result.strengths` | Stage 6 | Specific skills and qualifications that match your profile. |
| `cheap_llm_result.concerns` | Stage 6 | Gaps, missing requirements, or potential mismatches. |
| `strong_llm_result.final_score` | Stage 7 | In-depth LLM analysis score (0-100). |
| `strong_llm_result.driving_points` | Stage 7 | Key selling points to emphasize in your application and interview. |
| `strong_llm_result.red_flags` | Stage 7 | Warning signs identified in the job posting or company expectations. |