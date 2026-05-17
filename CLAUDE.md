# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

### Development & Infrastructure
- **Start API Server**: `python -m uvicorn src.api.main:app --reload`
- **Start Services (Postgres, Redis)**: `docker compose up -d`
- **Initialize Database**: `python scripts/init_db.py`
- **Download MS MARCO Dataset**: `python scripts/download_msmarco.py`

### Indexing & Data
- **Run Indexing Pipeline**: `python scripts/index_documents.py`
- **Index with custom collection**: `python scripts/index_documents.py --collection <path>`
- **Limit indexing docs**: `python scripts/index_documents.py --max-docs <count>`

### Testing & Evaluation
- **Run all tests**: `pytest tests/ -v`
- **Run specific test file**: `pytest tests/<test_file>.py -v`
- **Run with coverage**: `pytest tests/ --cov=src --cov-report=html`

## Architecture & Structure

### High-Level Design
The project implements a hybrid search pipeline: **BM25 (Keyword) $\to$ Vector Search (Semantic) $\to$ Reciprocal Rank Fusion (RRF) $\to$ Cross-Encoder Re-ranking $\to$ Learning-to-Rank (LTR)**.

### Core Directories
- `src/api/`: FastAPI application layer. Handles request/response models and routing.
- `src/search/`: Core search logic.
    - `bm25.py`: Keyword search using Tantivy.
    - `vector.py`: Semantic search using FAISS.
    - `fusion.py`: Combines BM25 and Vector results using RRF.
    - `cross_encoder_reranker.py`: Re-ranks top results using a Cross-Encoder model.
- `src/indexing/`: Pipeline for processing raw documents into searchable indexes.
    - `preprocessing.py`: Text cleaning, normalization, and validation.
    - `pipeline.py`: Orchestrates the end-to-end indexing flow.
- `src/database/`: Data persistence and caching.
    - `postgres.py`: SQLAlchemy models and session management.
    - `redis_client.py`: Redis-based caching for search results.
- `src/config.py`: Centralized configuration via environment variables (loaded via `.env`).
- `scripts/`: Standalone utilities for data acquisition, indexing, and evaluation.
- `tests/`: Pytest suite for unit and integration testing.
- `data/`: Stores MS MARCO datasets and generated indexes (Tantivy/FAISS).

### Dependency Direction
`api/` $\to$ `search/` $\to$ `database/` $\to$ `config.py` (No circular dependencies).
