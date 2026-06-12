# Job Scraper and Analyzer

A multi-stage pipeline that scrapes job listings from company career pages and job boards (LinkedIn, Indeed, ZipRecruiter, Google), analyzes them against your resume and user profile using LLM extraction + semantic embeddings, applies rule-based filtering, performs multi-layered scoring (vector similarity + cheap LLM + strong LLM), and produces a ranked final application queue.

## Pipeline Architecture

The system processes jobs through **9 stages**, each building on the previous:

| Stage | Name | What It Does |
|-------|------|--------------|
| 0 | **Setup** | Loads `.env`, reads resume + user profile from `documents/`, creates AI engine, extracts skills/job titles via text processing |
| 1 | **Scrape** | Scrapes jobs from configured company career pages (using `site_strategies/` files) and major job boards (Indeed, LinkedIn, ZipRecruiter, Google) |
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
├── .env                             # Configuration (paths, API keys, LLM settings)
├── user_preferences.yaml            # Rule filtering preferences
├── app/
│   ├── ai_engine.py                 # AI/LLM interface (extraction, embeddings)
│   ├── archetype_engine.py          # Archetype management and comparison
│   ├── llm_classifier.py            # Cheap LLM + Strong LLM classification stages
│   ├── make_db.py                   # Database schema creation
│   ├── postgres_mgr.py              # PostgreSQL database manager
│   ├── pull_data.py                 # Site scraping (DataPuller)
│   ├── text_engine.py               # Text processing and deterministic extraction
│   ├── tui.py                       # Textual-based Terminal UI
│   └── vector_engine.py             # Vector operations, scoring adjustments
├── documents/
│   ├── Your Name Resume.docx        # Your resume (Word format)
│   ├── User Profile - YourName.txt  # Your skills/profile (plain text)
│   └── archetypes_config.json       # Benchmark archetype definitions
├── site_strategies/
│   └── <company_name>.json          # Per-company scraping strategies
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

# ──────────────────────────────────────────────
# API KEYS
# ──────────────────────────────────────────────

GEMINMI_API_KEY=''
# Google Gemini API key. Required only if using Gemini as an LLM provider.

# ──────────────────────────────────────────────
# LLM PROVIDERS — GENERAL CONFIGURATION
# ──────────────────────────────────────────────
#
# The pipeline uses LLMs at multiple stages via the AIEngine abstraction layer.
# Each stage accepts a `provider_name` parameter. Supported providers:
#
#   "lm_studio"  — Local inference via LM Studio (http://localhost:{LMS_PORT})
#                  Works with any model loaded in LM Studio. Set LMS_URL, LMS_PORT below.
#
#   "gemini"     — Google Gemini API. Requires GEMINMI_API_KEY to be set.
#
#   "openai"     — OpenAI API. Requires OPENAI_API_KEY (not currently exposed, but
#                  the engine can be extended to support it).
#
# You can mix providers across stages. For example, use lm_studio for fast local
# extraction/embeddings and a remote provider for deep analysis.
#
# The MODEL variables are provider-specific model identifiers. For lm_studio, this
# is the model name as loaded in the LM Studio server. For gemini, this would be
# something like "gemini-2.0-flash". Leave blank to use the provider's default.

# ─── Stage 2: Extraction (skills, requirements, summary from job descriptions) ───
EXTRACTION_LLM='lm_studio'
# Provider for LLM-based extraction: "lm_studio" or "gemini".

EXTRACTION_MODEL=''
# Model identifier for extraction. Examples:
#   "gemma-4-26b-a4b-it-mlx"  (lm_studio)
#   "gemini-2.0-flash"        (gemini)
# Leave blank to use the provider's default model.

# ─── Stage 2: Embeddings (vector generation for titles, skills, descriptions) ───
EMBEDDINGS_LLM='lm_studio'
# Provider for embedding generation: "lm_studio" or "gemini".

EMBEDDINGS_MODEL=''
# Model identifier for embeddings. Examples:
#   "qwen3-embedding-8b-mxfp8"  (lm_studio)
#   "text-embedding-004"        (gemini)
# Leave blank to use the provider's default model.

# ─── Stage 6: Cheap LLM Classification ───
CHEAP_LLM_PROVIDER='lm_studio'
# Provider for fast/cheap classification: "lm_studio" or "gemini".

CHEAP_LLM_MODEL=''
# Model identifier for cheap classification. Leave blank for provider default.

# ─── Stage 7: Strong LLM Reranking ───
STRONG_LLM_PROVIDER='lm_studio'
# Provider for deep reranking: "lm_studio" or "gemini".

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

### 4. Site Strategies (Company Career Page Scraping)

To scrape jobs from specific company career pages, create a `site_strategies/` directory and add a JSON file for each company. The filename (without `.json`) must match an entry in the `JOB_SITES` list.

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
1. Add company entries to `sites` list in the `DataPuller`'s JOB_SITES file
2. Create a corresponding `site_strategies/<company_name>.json` file
3. The pipeline automatically loads and applies these strategies during Stage 1

### 5. Search Terms for Job Boards (Optional)

If you want to control which job titles/roles are searched on Indeed, LinkedIn, etc., create a CSV file (e.g., `search_terms.csv`) and set `SEARCH_TERMS` in `.env`:

```
Senior Python Engineer
AI/ML Engineer
Platform Engineer
DevOps Engineer
```

If left unset, the pipeline defaults to `["Software Engineer"]`.

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