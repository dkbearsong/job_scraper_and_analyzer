# Installation & Setup Guide

This guide walks you through the system requirements, installation steps, database setup with `pgvector`, required documents, and configuration files needed to get the **Job Scraper and Analyzer** fully operational.

---

## Table of Contents

- [System Requirements](#system-requirements)
- [Quick Start: Automated Installation](#quick-start-automated-installation)
- [Manual Step-by-Step Installation](#manual-step-by-step-installation)
  - [1. Clone Repository & Python Environment](#1-clone-repository--python-environment)
  - [2. Install Python Dependencies](#2-install-python-dependencies)
  - [3. Install Browser Binaries](#3-install-browser-binaries)
  - [4. PostgreSQL & pgvector Database Setup](#4-postgresql--pgvector-database-setup)
  - [5. Prepare Required Documents](#5-prepare-required-documents)
  - [6. Configure Environment Variables (.env)](#6-configure-environment-variables-env)
  - [7. Configure User Preferences (user_preferences.yaml)](#7-configure-user-preferences-user_preferencesyaml)
  - [8. Configure Scraper Adapters (scrapers_config.yaml)](#8-configure-scraper-adapters-scrapers_configyaml)
- [Verification & Self-Testing](#verification--self-testing)
- [Troubleshooting & FAQ](#troubleshooting--faq)

---

## System Requirements

| Component | Minimum | Recommended | Notes |
|-----------|---------|-------------|-------|
| **Operating System** | macOS 12+, Ubuntu 20.04+, Debian 11+, or Windows WSL2 | macOS (Apple Silicon / Intel) or Ubuntu 22.04 LTS | Native Windows works via WSL2 (Ubuntu) |
| **Python** | Python 3.10 | Python 3.12 | PyTorch & tokenizers tested on Python 3.12 |
| **Database** | PostgreSQL 14+ | PostgreSQL 16+ with `pgvector` extension | Required for vector embeddings & RAG. Offline runs can use `--skip-db` |
| **RAM** | 8 GB | 16 GB+ | Needed if running local sentence-transformers or local LLMs |
| **Storage** | 2 GB free disk space | 10 GB+ | Model weights for sentence-transformers or local LLMs |
| **LLM Inference** | Remote API (e.g. Gemini / OpenRouter) | Local (LM Studio / Ollama) or Cloud API | System supports hybrid local + cloud models |

---

## Quick Start: Automated Installation

An automated installation script is provided in the root directory:

```bash
# Make the installation script executable (if needed)
chmod +x install.sh

# Run the interactive installer
./install.sh
```

### Non-Interactive & Custom Flags

You can pass flags to `./install.sh` for scripted or headless setups:

```bash
# Accept all defaults without prompting
./install.sh -y

# Skip pip installation (if dependencies are already installed)
./install.sh --skip-pip

# Skip database connectivity check & table initialization
./install.sh --skip-db

# Skip downloading Playwright browser binaries
./install.sh --skip-browsers

# View all flags
./install.sh --help
```

---

## Manual Step-by-Step Installation

If you prefer to configure the environment manually or need to tailor specific steps, follow the guide below.

### 1. Clone Repository & Python Environment

Clone the repository and create an isolated Python virtual environment:

```bash
git clone <repository-url>
cd job_scraper_and_analyzer

# Create virtual environment with Python 3.10+
python3 -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows (PowerShell):
# .venv\Scripts\Activate.ps1
```

> [!TIP]
> Always verify your active Python interpreter:
> ```bash
> which python   # Should point to ./job_scraper_and_analyzer/.venv/bin/python
> python --version  # Should be 3.10 or higher
> ```

### 2. Install Python Dependencies

Upgrade packaging tools and install required packages from `requirements.txt`:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

Key dependencies installed include:
- **PyTorch & Transformers**: `torch`, `transformers`, `sentence-transformers` for local embedding generation.
- **Database**: `psycopg2`, `pgvector` for PostgreSQL persistence and vector similarity search.
- **LLM SDKs**: `google-genai`, `openai`, `anthropic` for cloud model access.
- **Scraping & Web**: `python-jobspy`, `aiohttp`, `requests`, `furl`, `pandas`, `playwright`.
- **Parsing & Text**: `python-docx`, `PyYAML`, `python-dotenv`, `pydantic`.

### 3. Install Browser Binaries

For scrapers that render dynamic JavaScript or browser-based sites, install the Playwright Chromium browser:

```bash
playwright install chromium
```

---

### 4. PostgreSQL & pgvector Database Setup

The pipeline uses PostgreSQL with the `pgvector` extension for storing jobs, semantic vector embeddings, LLM evaluation results, and running hybrid RAG queries.

#### Option A: macOS via Homebrew

```bash
# 1. Install PostgreSQL 16 and pgvector
brew install postgresql@16 pgvector

# 2. Start PostgreSQL service
brew services start postgresql@16

# 3. Create default database user and database (if not already present)
createuser -s postgres
createdb -U postgres job_searcher

# 4. Enable the vector extension in the database
psql -U postgres -d job_searcher -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Option B: Ubuntu / Debian via APT

```bash
# 1. Install PostgreSQL and build tools
sudo apt update
sudo apt install -y postgresql postgresql-contrib postgresql-server-dev-all git build-essential

# 2. Install pgvector from source
cd /tmp
git clone --branch v0.7.0 https://github.com/pgvector/pgvector.git
cd pgvector
make
sudo make install

# 3. Start PostgreSQL service
sudo systemctl start postgresql

# 4. Create database and enable pgvector
sudo -u postgres psql -c "CREATE DATABASE job_searcher;"
sudo -u postgres psql -d job_searcher -c "CREATE EXTENSION IF NOT EXISTS vector;"
```

#### Option C: Docker (Fastest for Local Development)

You can run an official PostgreSQL image pre-configured with `pgvector`:

```bash
docker run -d \
  --name job_scraper_db \
  -e POSTGRES_DB=job_searcher \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=postgres \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

#### Initialize Database Tables

Once PostgreSQL is running and reachable, initialize the schema:

```bash
python -c "from app.make_db import make_db; print(make_db())"
```

This creates the following tables:
- `job`: Scraped job postings, salary, location, and metadata.
- `company` & `office`: Company career site profiles and geocoded locations.
- `job_embeddings`: Vector embeddings for title, requirements, responsibilities, pay, and location.
- `archetype_embeddings`: Vector representations of benchmark roles, your resume, and profile.
- `token_usage_log`: Historical token consumption, cost, and provider operations.
- `cheap_llm_results`: Fast initial fit assessment and criteria checks.
- `strong_llm_results`: Deep reranking and qualitative fit analysis.
- `vector_scores`: Multi-dimensional cosine similarity scores.
- `final_application_queue`: Prioritized queue with recommendations (`apply`, `maybe`, `skip`).
- `rag_documents` & `rag_metadata`: Text chunks indexed with HNSW vector cosine distance for natural language queries and tailored applications.

#### Verify Database Connectivity

Run the included diagnostic script:

```bash
python diagnose_db.py
```

---

### 5. Prepare Required Documents

The system compares scraped jobs against your professional background and target roles. Place your profile documents in the `documents/` folder:

```
documents/
├── Your Resume.docx             # Your resume in Word format
├── User Profile.txt             # Plain-text summary of skills & titles
├── archetypes_config.json       # Benchmark target roles
├── search_terms.csv             # Search queries for JobSpy (optional)
└── adzunda_searches.csv         # Queries for Adzuna API (optional)
```

#### A. Resume (`.docx`)
Place your resume saved as a Microsoft Word `.docx` file in `documents/`.

#### B. User Profile (`.txt`)
Create a plain-text file in `documents/` (e.g. `documents/User Profile.txt`). It should include `Skills:` and `Job Titles:` sections for deterministic parsing:

```text
Skills: Python, FastAPI, PostgreSQL, AWS, Docker, Kubernetes, CI/CD, Terraform, PyTorch
Job Titles: Senior Backend Engineer, DevOps Engineer, Platform Engineer

Summary: Experienced software engineer specializing in backend distributed systems, API architecture, and cloud infrastructure.
```

#### C. Benchmark Archetypes (`archetypes_config.json`)
This file defines benchmark roles to calculate semantic similarity against. An example file is located at `documents/archetypes_config.json`. Each entry defines target skills, responsibilities, and titles:

```json
[
  {
    "name": "Backend Python Engineer",
    "title": "Senior Backend Python Engineer",
    "skills": "Python FastAPI PostgreSQL AWS Redis Docker Distributed Systems",
    "responsibilities": "Design scalable APIs. Optimize database performance. Deploy cloud services."
  },
  {
    "name": "AI Tooling Engineer",
    "title": "Senior AI Tooling Engineer",
    "skills": "Python OpenAI LangChain Pinecone PyTorch LLMs",
    "responsibilities": "Develop AI-powered tools. Integrate LLMs into workflows."
  }
]
```

---

### 6. Configure Environment Variables (`.env`)

Copy the provided template to create your `.env` file:

```bash
cp .env.example .env
```

Edit `.env` and set your credentials:

```ini
# ──────────────────────────────────────────────
# File & Directory Paths
# ──────────────────────────────────────────────
RESUME='documents/Your Resume.docx'
PROFILE='documents/User Profile.txt'
ARCHETYPES_CONFIG='documents/archetypes_config.json'
SCRAPERS_CONFIG='scrapers_config.yaml'
USER_PREFERENCES_YAML='user_preferences.yaml'
JOB_SITES='job_sites.csv'
SEARCH_TERMS='documents/search_terms.csv'
ADZUNDA_SEARCHES_CSV='documents/adzunda_searches.csv'

# ──────────────────────────────────────────────
# PostgreSQL Database
# ──────────────────────────────────────────────
DB_NAME='job_searcher'
DB_USER='postgres'
DB_PASSWORD='your_password'
DB_HOST='localhost'
DB_PORT='5432'

# ──────────────────────────────────────────────
# Scrapers & API Keys
# ──────────────────────────────────────────────
# Free Adzuna API credentials: https://developer.adzuna.com/
ADZUNA_APP_ID='your_adzuna_app_id'
ADZUNA_APP_KEY='your_adzuna_app_key'

# ──────────────────────────────────────────────
# LLM Providers (Configure keys for the providers you use)
# ──────────────────────────────────────────────
GEMINI_API_KEY='your_gemini_api_key'
OPENROUTER_API_KEY='your_openrouter_api_key'
OPENAI_API_KEY='your_openai_api_key'
ANTHROPIC_API_KEY='your_anthropic_api_key'
GROQ_API_KEY='your_groq_api_key'

# Local Inference (LM Studio or Ollama)
LMS_URL='http://localhost'
LMS_PORT='1234'
LMS_API_KEY='lm-studio'

OLLAMA_URL='http://localhost'
OLLAMA_PORT='11434'

# ──────────────────────────────────────────────
# Concurrency & Tier Settings
# ──────────────────────────────────────────────
CONCURRENCY_MODE='concurrent'
STAGE_2_EXTRACTION_TIER='paid'
STAGE_2_EMBEDDING_TIER='paid'
STAGE_6_TIER='free'
STAGE_7_TIER='free'
```

---

### 7. Configure User Preferences (`user_preferences.yaml`)

This file governs deterministic rule filtering (Stage 3), LLM provider and model assignments, rate-limit throttling tiers, and negative vector scoring.

```yaml
# Target location and search radius (in miles)
target_cities:
  - "USA"
target_city_range: 50

# Work arrangement constraints: remote, hybrid, on-site
work_types:
  - remote
  - hybrid

# Seniority levels: entry-level, mid-level, senior, lead, manager
seniority_levels:
  - mid-level
  - senior

# Disqualified Job Titles (Stage 1.5 Deterministic Regex Filter)
# Immediately marks jobs matching these words, phrases, or regex patterns as skip in DB and removes from memory
disqualified_titles:
  - "Sales Representative"
  - "Account Executive"
  - "Call Center"
  - "Telemarketing"
  - "Insurance Agent"
  - "Real Estate Agent"

# Granular role-specific seniority overrides
role_seniority_levels:
  - role_keywords:
      - "support engineer"
      - "technical support engineer"
    allowed_seniority:
      - "mid-level"
      - "senior"
  - role_keywords:
      - "software engineer"
      - "backend engineer"
    allowed_seniority:
      - "mid-level"

# Accepted timezones
timezones:
  - PST
  - MST
  - CST
  - EST

# Desired annual pay range
pay_range: "$95k-300k"

# LLM Provider Configuration per Task
# Providers: lm_studio, gemini, openrouter, claude, chatgpt, sentence_transformers
extraction_llm: "lm_studio"
embeddings_llm: "sentence_transformers"
cheap_llm_provider: "gemini"
strong_llm_provider: "gemini"

# Specific Model Names
extraction_model: "liquid/lfm2-24b-a2b"
embeddings_model: "all-MiniLM-L6-v2"
cheap_llm_model: "gemini-3.1-flash-lite"
strong_llm_model: "gemini-3.5-flash-lite"

# Rate Limiting & Throttling Tiers (prevents API 429 errors)
stage_6:
  free:
    requests_per_minute: 15
    tokens_per_minute: 250000
    concurrency: 3
  paid:
    requests_per_minute: 500
    tokens_per_minute: 6000000
    concurrency: 15

stage_7:
  free:
    requests_per_minute: 15
    tokens_per_minute: 250000
    concurrency: 3
  paid:
    requests_per_minute: 500
    tokens_per_minute: 6000000
    concurrency: 15

# Negative Vector Scoring (Penalizes unwanted titles or functions)
negative_scoring:
  enabled: true
  penalty_weight: 0.35          # Up to 35% score penalty
  similarity_threshold: 0.50    # Minimum similarity trigger
  avoid_titles:
    - "Sales Representative"
    - "Account Executive"
    - "Call Center Agent"
    - "Manager"
  avoid_functions:
    - "cold calling outbound sales quota lead generation prospecting"
    - "door to door commission only high volume telemarketing"

top_n_deep_analysis: 25
```

---

### 8. Configure Scraper Adapters (`scrapers_config.yaml`)

Stage 1 scraping is modular. You can enable or disable adapters in `scrapers_config.yaml`:

```yaml
# Run legacy fallback (Part A career pages + Part B JobSpy) after adapters
run_fallback_after_adapters: true

scrapers:
  # Adzuna API adapter
  - name: Adzuna
    adapter: app.scrapers.adzuna_adapter.AdzunaAdapter
    enabled: true

  # Job boards via JobSpy (Indeed, LinkedIn, ZipRecruiter, Google)
  - name: jobspy_boards
    adapter: JobSpyAdapter
    enabled: false
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
      requests_wanted: 50
      hours_old: 24

  # Local microservice for company career pages
  - name: microservice_company_boards
    adapter: MicroserviceAdapter
    enabled: false
    config:
      sites_file: "${JOB_SITES}"
      strategies_dir: "./site_strategies"
      microservice_host: "http://localhost"
      microservice_port: "5052"
```

---

## Verification & Self-Testing

Once installed and configured, verify each layer of the application:

### 1. Test Database Connectivity
```bash
python diagnose_db.py
```
Expected output: `Port is reachable.` and `Connection successful!`.

### 2. Run Isolated Stage Tests (Dummy Data, No Database Needed)
You can test the entire pipeline logic without connecting to any external services:
```bash
# List all pipeline stages
python tests/test_runner.py --list-stages

# Run Stage 0 (Setup) smoke test
python tests/test_runner.py --stage 0 --skip-db

# Test Stage 2 & 3 (Extraction & Rule Filtering)
python tests/test_runner.py --stage 2 3 --chain --skip-db

# Test the entire pipeline with synthetic data
python tests/test_runner.py --all --skip-db
```

### 3. Run Pipeline Setup Stage (Real Documents, Live Engine)
```bash
python main.py -s 0 --skip-db
```
This tests loading your resume, profile, archetypes, and initializing the AI engine without writing to the database.

---

## Troubleshooting & FAQ

### 1. `could not open extension control file "vector.control"`
- **Cause**: PostgreSQL is installed, but the `pgvector` extension package is missing.
- **Fix**:
  - **macOS**: `brew install pgvector` then restart postgres: `brew services restart postgresql@16`.
  - **Ubuntu**: Build from source (`git clone https://github.com/pgvector/pgvector && cd pgvector && make && sudo make install`).
  - **Docker**: Switch to `pgvector/pgvector:pg16`.

### 2. `psycopg2.OperationalError: connection to server at "localhost" failed`
- **Cause**: PostgreSQL is not running or credentials in `.env` are incorrect.
- **Fix**: Run `python diagnose_db.py` to see the exact connection error. Check `DB_HOST`, `DB_PORT`, `DB_USER`, and `DB_PASSWORD` in `.env`.
- To run without a database in development/testing, pass `--skip-db`:
  ```bash
  python main.py --skip-db
  ```

### 3. API Error `429 Too Many Requests`
- **Cause**: The LLM provider rate limits have been exceeded.
- **Fix**:
  1. In `.env`, set `STAGE_6_TIER='free'` or `STAGE_7_TIER='free'`.
  2. In `user_preferences.yaml`, lower `requests_per_minute`, `tokens_per_minute`, or `concurrency` under the appropriate stage section.

### 4. LM Studio / Ollama Connection Refused
- **Cause**: Local inference server is not running on the specified host/port.
- **Fix**:
  - For **LM Studio**: Ensure the local server is started (Status: Running) on port `1234` with CORS enabled.
  - For **Ollama**: Ensure `ollama serve` is active on port `11434`. Pull models beforehand: `ollama pull llama3.1:8b`.

### 5. `ModuleNotFoundError: No module named 'app'`
- **Cause**: Python was run without the current working directory in `PYTHONPATH`.
- **Fix**: Ensure your virtual environment is active (`source .venv/bin/activate`) and run commands from the project root. For pytest: `PYTHONPATH=. pytest`.
