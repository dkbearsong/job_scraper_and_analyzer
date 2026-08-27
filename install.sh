#!/usr/bin/env bash
# ==============================================================================
# Job Scraper and Analyzer - Installation & Setup Script
# ==============================================================================
# Sets up the Python virtual environment, installs all required dependencies,
# initializes configuration files, verifies database readiness, and runs a smoke test.
#
# Usage:
#   ./install.sh [options]
#
# Options:
#   -h, --help           Show this help message
#   -y, --yes            Non-interactive mode (auto-accept all prompts)
#   --skip-pip           Skip installing Python dependencies
#   --skip-db            Skip database connectivity check and table initialization
#   --skip-browsers      Skip installing Playwright browser binaries
# ==============================================================================

set -eo pipefail

# ── ANSI Color Codes ─────────────────────────────────────────────────────────
BOLD="\033[1m"
GREEN="\033[0;32m"
YELLOW="\033[0;33m"
RED="\033[0;31m"
BLUE="\033[0;34m"
CYAN="\033[0;36m"
RESET="\033[0m"

log_info()    { echo -e "${BLUE}[INFO]${RESET} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${RESET} $*"; }
log_warn()    { echo -e "${YELLOW}[WARNING]${RESET} $*"; }
log_error()   { echo -e "${RED}[ERROR]${RESET} $*"; }
log_step()    { echo -e "\n${BOLD}${CYAN}==>${RESET} ${BOLD}$*${RESET}"; }

# ── Argument Parsing ──────────────────────────────────────────────────────────
AUTO_YES=false
SKIP_PIP=false
SKIP_DB=false
SKIP_BROWSERS=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            head -n 17 "$0" | tail -n 14
            exit 0
            ;;
        -y|--yes)
            AUTO_YES=true
            shift
            ;;
        --skip-pip)
            SKIP_PIP=true
            shift
            ;;
        --skip-db)
            SKIP_DB=true
            shift
            ;;
        --skip-browsers)
            SKIP_BROWSERS=true
            shift
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Run './install.sh --help' for usage."
            exit 1
            ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo -e "${BOLD}================================================================${RESET}"
echo -e "${BOLD}         Job Scraper and Analyzer - Setup & Installation        ${RESET}"
echo -e "${BOLD}================================================================${RESET}"

# ── Step 1: Check Python Version ─────────────────────────────────────────────
log_step "Step 1: Checking Python environment..."

PYTHON_BIN=""
for cmd in python3.12 python3.11 python3.10 python3; do
    if command -v "$cmd" >/dev/null 2>&1; then
        VERSION=$("$cmd" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
        MAJOR=$("$cmd" -c 'import sys; print(sys.version_info.major)' 2>/dev/null || true)
        MINOR=$("$cmd" -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || true)
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 10 ]]; then
            PYTHON_BIN="$cmd"
            log_success "Found Python $VERSION via '$cmd'"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    log_error "Python 3.10 or higher is required, but was not found."
    echo "Please install Python 3.10+ (e.g. 'brew install python@3.12' on macOS or 'apt install python3' on Ubuntu)."
    exit 1
fi

# ── Step 2: Virtual Environment Setup ─────────────────────────────────────────
log_step "Step 2: Setting up Python virtual environment (.venv)..."

if [[ ! -d ".venv" ]]; then
    log_info "Creating virtual environment at .venv using $PYTHON_BIN..."
    "$PYTHON_BIN" -m venv .venv
    log_success "Virtual environment created."
else
    log_info "Existing virtual environment found at .venv."
fi

VENV_PY=".venv/bin/python"
VENV_PIP=".venv/bin/pip"

if [[ ! -x "$VENV_PY" ]]; then
    log_error "Virtual environment python binary not found at $VENV_PY"
    exit 1
fi

# Upgrade pip
log_info "Upgrading pip, setuptools, and wheel..."
"$VENV_PIP" install --upgrade pip setuptools wheel >/dev/null 2>&1 || true

# ── Step 3: Install Python Dependencies ───────────────────────────────────────
log_step "Step 3: Installing dependencies from requirements.txt..."

if [[ "$SKIP_PIP" = true ]]; then
    log_info "Skipping pip install (--skip-pip flag provided)."
else
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing packages (this may take a few minutes for PyTorch and Transformers)..."
        "$VENV_PIP" install -r requirements.txt
        log_success "Python dependencies installed successfully."
    else
        log_warn "requirements.txt not found! Skipping pip installation."
    fi

    # Install Playwright browser if not skipped
    if [[ "$SKIP_BROWSERS" = false ]]; then
        if "$VENV_PY" -c "import playwright" >/dev/null 2>&1; then
            log_info "Ensuring Playwright browser binaries (Chromium) are installed..."
            "$VENV_PY" -m playwright install chromium || log_warn "Playwright browser install encountered a warning (can be re-run manually via: .venv/bin/playwright install chromium)"
        fi
    fi
fi

# ── Step 4: Configuration Files Initialization ────────────────────────────────
log_step "Step 4: Checking configuration files..."

# Check .env
if [[ ! -f ".env" ]]; then
    if [[ -f ".env.example" ]]; then
        log_info "No .env file found. Creating .env from .env.example..."
        cp .env.example .env
        log_success "Created .env template. Please remember to edit .env to add your database password and LLM API keys!"
    else
        log_warn ".env.example not found. Please create a .env file with your credentials."
    fi
else
    log_success ".env file is present."
fi

# Check user_preferences.yaml
if [[ -f "user_preferences.yaml" ]]; then
    log_success "user_preferences.yaml is present."
else
    log_warn "user_preferences.yaml is missing! This file is needed for Stage 3 rule filtering and LLM provider configuration."
fi

# Check scrapers_config.yaml
if [[ -f "scrapers_config.yaml" ]]; then
    log_success "scrapers_config.yaml is present."
else
    log_warn "scrapers_config.yaml is missing! The pipeline will fall back to legacy scraping mode."
fi

# ── Step 5: Documents Directory & Archetypes ──────────────────────────────────
log_step "Step 5: Checking documents directory..."

mkdir -p documents logs

if [[ -f "documents/archetypes_config.json" ]]; then
    log_success "Benchmark archetypes found at documents/archetypes_config.json"
else
    log_warn "documents/archetypes_config.json not found! Required for Stage 4 & 5 archetype matching."
fi

# Check resume and profile files configured in .env if readable
if [[ -f ".env" ]]; then
    RESUME_PATH=$("$VENV_PY" -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('RESUME', ''))" 2>/dev/null || true)
    PROFILE_PATH=$("$VENV_PY" -c "import os; from dotenv import load_dotenv; load_dotenv(); print(os.getenv('PROFILE', ''))" 2>/dev/null || true)

    if [[ -n "$RESUME_PATH" && -f "$RESUME_PATH" ]]; then
        log_success "Resume found at: $RESUME_PATH"
    else
        log_warn "Resume file not found at '$RESUME_PATH'. Place your .docx resume in documents/ and update RESUME in .env."
    fi

    if [[ -n "$PROFILE_PATH" && -f "$PROFILE_PATH" ]]; then
        log_success "User profile found at: $PROFILE_PATH"
    else
        log_warn "User profile not found at '$PROFILE_PATH'. Place your plain-text profile in documents/ and update PROFILE in .env."
    fi
fi

# ── Step 6: Database Connectivity & Initialization ────────────────────────────
log_step "Step 6: Checking PostgreSQL database connection..."

if [[ "$SKIP_DB" = true ]]; then
    log_info "Skipping database setup (--skip-db flag provided)."
else
    DB_STATUS=0
    "$VENV_PY" -c "
import os
from dotenv import load_dotenv
load_dotenv()
import psycopg2

host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
user = os.getenv('DB_USER', 'postgres')
password = os.getenv('DB_PASSWORD', '')
dbname = os.getenv('DB_NAME', 'job_searcher')

try:
    conn = psycopg2.connect(host=host, port=port, user=user, password=password, dbname='postgres', connect_timeout=4)
    conn.close()
    print('DB_SERVER_OK')
except Exception as e:
    print(f'DB_ERR: {e}')
    exit(1)
" > /tmp/db_check.out 2>&1 || DB_STATUS=$?

    if [[ $DB_STATUS -eq 0 ]]; then
        log_success "PostgreSQL server is reachable."

        INIT_DB=false
        if [[ "$AUTO_YES" = true ]]; then
            INIT_DB=true
        else
            echo ""
            read -p "Would you like to initialize / verify the database tables and pgvector extension now? (y/N): " -r RESPONSE
            if [[ "$RESPONSE" =~ ^[Yy]$ ]]; then
                INIT_DB=true
            fi
        fi

        if [[ "$INIT_DB" = true ]]; then
            log_info "Initializing database schema via app.make_db..."
            "$VENV_PY" -c "from app.make_db import make_db; res = make_db(); print(res)" || {
                log_warn "Database initialization returned a warning. Verify that 'CREATE EXTENSION vector;' is permitted for your DB user."
            }
            log_success "Database initialization complete."
        else
            log_info "Skipping table creation. You can run it later with: $VENV_PY -c 'from app.make_db import make_db; make_db()'"
        fi
    else
        log_warn "Could not connect to PostgreSQL server:"
        cat /tmp/db_check.out 2>/dev/null || true
        log_warn "You can still run the pipeline in offline/demo mode using the --skip-db flag."
        log_info "To diagnose connectivity, run: $VENV_PY diagnose_db.py"
    fi
    rm -f /tmp/db_check.out
fi

# ── Step 7: Verification Smoke Test ───────────────────────────────────────────
log_step "Step 7: Running verification smoke test..."

SMOKE_TEST_OK=true
"$VENV_PY" tests/test_runner.py --stage 0 --skip-db > /tmp/smoke_test.log 2>&1 || SMOKE_TEST_OK=false

if [[ "$SMOKE_TEST_OK" = true ]]; then
    log_success "Smoke test passed! Pipeline stage 0 (Setup) executed cleanly."
    rm -f /tmp/smoke_test.log
else
    log_warn "Smoke test encountered an issue. See details below:"
    tail -n 20 /tmp/smoke_test.log 2>/dev/null || true
    log_info "Full smoke test log saved at /tmp/smoke_test.log"
fi

# ── Summary & Next Steps ──────────────────────────────────────────────────────
echo -e "\n${BOLD}================================================================${RESET}"
echo -e "${BOLD}${GREEN}               Installation & Setup Completed!                  ${RESET}"
echo -e "${BOLD}================================================================${RESET}"
echo ""
echo -e "Next steps to get started:"
echo -e "  1. Activate your virtual environment:"
echo -e "       ${BOLD}${CYAN}source .venv/bin/activate${RESET}"
echo ""
echo -e "  2. Review and configure your environment:"
echo -e "       - Update ${BOLD}.env${RESET} with your DB credentials & LLM API keys"
echo -e "       - Customize ${BOLD}user_preferences.yaml${RESET} for your target roles, pay, and location"
echo -e "       - Add your resume to ${BOLD}documents/${RESET} and update RESUME in .env"
echo ""
echo -e "  3. Test the pipeline with dummy data (no API or DB needed):"
echo -e "       ${BOLD}${CYAN}python tests/test_runner.py --stage 0 1 2 3 --skip-db${RESET}"
echo ""
echo -e "  4. Run the full pipeline:"
echo -e "       ${BOLD}${CYAN}python main.py${RESET}"
echo ""
echo -e "  5. For full configuration reference and troubleshooting, consult:"
echo -e "       ${BOLD}INSTALL.md${RESET} and ${BOLD}README.md${RESET}"
echo -e "${BOLD}================================================================${RESET}\n"
