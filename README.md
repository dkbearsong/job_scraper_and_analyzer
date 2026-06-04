# Job Scraper and Analyzer

This project contains tools for scraping job listings, analyzing them, and comparing against user profiles using semantic embeddings.

## Architecture Overview

The system includes several key components:
- Job scraping and data processing
- Semantic analysis using embeddings
- Archetype engine for comparing jobs to user profiles/resumes
- Database management and storage

## Key Modules

1. **app/archetype_engine.py** - Handles archetype comparison using embeddings
2. **app/text_engine.py** - Text processing and semantic analysis  
3. **app/vector_engine.py** - Vector operations for embeddings
4. **app/postgres_mgr.py** - Database management

## Implementation Status

The archetype engine is currently a placeholder that needs to be fully implemented with:
- Loading of archetype profiles (resume, user profile)
- Semantic document creation for jobs
- Embedding-based comparison capabilities  
- Retrieval metadata generation

## Usage Instructions

1. Set up environment variables in .env file
2. Run main.py to start the job scraping and analysis pipeline