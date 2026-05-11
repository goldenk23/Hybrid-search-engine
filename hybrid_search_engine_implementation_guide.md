# Hybrid Neural Search Engine — Implementation Guide

> A hands-on, teaching-first guide that walks you through building every component from scratch. Each section explains **why** before **how**, so you understand the reasoning behind every line of code.

---

## Table of Contents

- [Phase 0: Project Setup & Environment](#phase-0-project-setup--environment)
- [Phase 1: BM25 Keyword Search (Week 1-2)](#phase-1-bm25-keyword-search-week-1-2)
- [Phase 2: Vector Search & Hybrid Retrieval (Week 3-4)](#phase-2-vector-search--hybrid-retrieval-week-3-4)
- [Phase 3: Cross-Encoder Re-Ranking (Week 5)](#phase-3-cross-encoder-re-ranking-week-5)
- [Phase 4: Learning-to-Rank (Week 6-7)](#phase-4-learning-to-rank-week-6-7)
- [Phase 5: Polish, Benchmark & Deploy (Week 8)](#phase-5-polish-benchmark--deploy-week-8)
- [Resources & References](#resources--references)
- [Best Practices & Tips](#best-practices--tips)
- [Common Mistakes & How to Avoid Them](#common-mistakes--how-to-avoid-them)

---

## Phase 0: Project Setup & Environment

### 0.1 Project Structure

Before writing any code, let's design a clean project structure. This matters because:
- Recruiters will browse your GitHub repo — first impressions count
- A well-organized codebase shows engineering maturity
- It makes your own life easier when the project grows

```
hybrid-search-engine/
│
├── README.md                    # Project overview, setup instructions, benchmarks
├── docker-compose.yml           # One command to start everything
├── Dockerfile                   # Container build instructions
├── pyproject.toml               # Python project metadata & dependencies
├── .env.example                 # Template for environment variables
├── .gitignore                   # Files to exclude from git
│
├── src/                         # All source code lives here
│   ├── __init__.py
│   │
│   ├── api/                     # FastAPI application layer
│   │   ├── __init__.py
│   │   ├── main.py              # FastAPI app entry point
│   │   ├── routes/              # API endpoint definitions
│   │   │   ├── __init__.py
│   │   │   ├── search.py        # GET /search
│   │   │   ├── feedback.py      # POST /feedback
│   │   │   └── admin.py         # POST /index, GET /health
│   │   ├── models.py            # Pydantic request/response models
│   │   └── middleware.py        # Rate limiting, CORS, logging
│   │
│   ├── search/                  # Core search pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Orchestrates the full search flow
│   │   ├── bm25.py              # BM25 keyword search (Tantivy)
│   │   ├── vector.py            # Vector similarity search (FAISS)
│   │   ├── fusion.py            # Reciprocal Rank Fusion
│   │   ├── reranker.py          # Cross-encoder re-ranking
│   │   └── ltr.py               # Learning-to-Rank (LambdaMART)
│   │
│   ├── query/                   # Query understanding
│   │   ├── __init__.py
│   │   ├── spell_check.py       # Spell correction
│   │   ├── intent.py            # Intent classification (FastText)
│   │   └── expansion.py         # Query expansion
│   │
│   ├── indexing/                # Document indexing pipeline
│   │   ├── __init__.py
│   │   ├── pipeline.py          # Full indexing orchestration
│   │   ├── preprocessor.py      # Text cleaning & normalization
│   │   └── embedder.py          # Batch embedding generation
│   │
│   ├── feedback/                # Click tracking & training data
│   │   ├── __init__.py
│   │   ├── collector.py         # Click/dwell event logging
│   │   ├── click_model.py       # Convert clicks → relevance labels
│   │   └── trainer.py           # Daily LTR model retraining
│   │
│   ├── db/                      # Database layer
│   │   ├── __init__.py
│   │   ├── postgres.py          # PostgreSQL connection & queries
│   │   ├── redis_client.py      # Redis cache operations
│   │   └── models.py            # SQLAlchemy ORM models
│   │
│   └── config.py                # Centralized configuration
│
├── scripts/                     # Utility scripts
│   ├── download_msmarco.py      # Download MS MARCO dataset
│   ├── index_documents.py       # Run indexing pipeline
│   ├── train_ltr.py             # Train LambdaMART model
│   ├── evaluate.py              # Compute NDCG, MRR metrics
│   └── fine_tune_embeddings.py  # Fine-tune Sentence-BERT
│
├── tests/                       # Test suite
│   ├── __init__.py
│   ├── test_bm25.py
│   ├── test_vector.py
│   ├── test_fusion.py
│   ├── test_reranker.py
│   ├── test_pipeline.py
│   └── test_api.py
│
├── notebooks/                   # Jupyter notebooks for exploration
│   ├── 01_explore_msmarco.ipynb
│   ├── 02_embedding_analysis.ipynb
│   ├── 03_hnsw_parameter_tuning.ipynb
│   └── 04_ltr_feature_importance.ipynb
│
├── benchmarks/                  # Performance benchmarks
│   ├── locustfile.py            # Load testing script
│   └── results/                 # Benchmark results (charts, CSVs)
│
├── models/                      # Trained model artifacts (git-ignored)
│   ├── embeddings/              # Fine-tuned Sentence-BERT
│   ├── ltr/                     # LambdaMART model files
│   └── intent/                  # FastText intent classifier
│
├── data/                        # Datasets (git-ignored)
│   ├── msmarco/                 # MS MARCO passages & queries
│   └── indexes/                 # Built FAISS & Tantivy indexes
│
└── frontend/                    # Minimal React UI
    ├── package.json
    ├── src/
    │   ├── App.jsx
    │   ├── components/
    │   │   ├── SearchBar.jsx
    │   │   ├── ResultCard.jsx
    │   │   └── LatencyBadge.jsx
    │   └── hooks/
    │       └── useSearch.js
    └── public/
        └── index.html
```

**Why this structure?**

```
Principle 1: Separation of Concerns
  Each folder handles ONE responsibility.
  search/ knows nothing about the API layer.
  api/ knows nothing about how FAISS works.
  → Makes it easy to test, modify, and explain each piece independently.

Principle 2: Dependency Direction
  api/ depends on search/ (calls search functions)
  search/ depends on db/ (reads from database)
  db/ depends on nothing else
  → No circular dependencies. Clean architecture.

Principle 3: Scripts ≠ Source Code
  scripts/ contains one-off utilities (download data, train models)
  src/ contains the running application
  → Keeps the main codebase focused
```

### 0.2 Setting Up the Development Environment

> **Concept: Why virtual environments matter**
>
> Python packages can conflict with each other. Project A needs `numpy==1.24` while Project B needs `numpy==1.26`. Virtual environments give each project its own isolated set of packages, preventing conflicts. Always use one.

**Step 1: Create the project directory and initialize Git**

```bash
# Create project directory
mkdir hybrid-search-engine
cd hybrid-search-engine

# Initialize git repository
git init

# Create .gitignore (IMPORTANT: do this before adding any files)
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/
*.egg

# Virtual environment
.venv/
venv/
env/

# Data & models (too large for git)
data/
models/
*.bin
*.pkl
*.pt
*.onnx

# IDE
.vscode/
.idea/
*.swp
*.swo

# Environment variables (NEVER commit secrets)
.env

# OS
.DS_Store
Thumbs.db

# Jupyter
.ipynb_checkpoints/

# Docker
*.log
EOF
```

**Step 2: Set up Python environment**

```bash
# Create virtual environment (use Python 3.11+)
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Verify you're using the virtual environment's Python
python --version
which python  # Should point to .venv/
```

**Step 3: Create `pyproject.toml`**

> **Concept: pyproject.toml vs requirements.txt**
>
> `requirements.txt` is the old way — just a flat list of packages.
> `pyproject.toml` is the modern Python standard (PEP 621). It defines your project metadata, dependencies, and build configuration in one file. Using it shows you follow current best practices.

```toml
[project]
name = "hybrid-search-engine"
version = "0.1.0"
description = "A hybrid neural search engine with BM25, vector search, cross-encoder re-ranking, and learned ranking"
requires-python = ">=3.11"

dependencies = [
    # Web framework
    "fastapi>=0.109.0",
    "uvicorn[standard]>=0.27.0",
    "pydantic>=2.5.0",

    # BM25 keyword search
    "tantivy>=0.22.0",

    # Vector search
    "faiss-cpu>=1.7.4",

    # Embeddings & re-ranking
    "sentence-transformers>=2.3.0",
    "torch>=2.1.0",

    # Learning-to-Rank
    "lightgbm>=4.2.0",

    # Query understanding
    "symspellpy>=6.7.7",
    "nltk>=3.8.1",

    # Database
    "sqlalchemy>=2.0.25",
    "asyncpg>=0.29.0",
    "psycopg2-binary>=2.9.9",

    # Cache
    "redis>=5.0.1",

    # Data processing
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "tqdm>=4.66.0",

    # Utilities
    "python-dotenv>=1.0.0",
    "httpx>=0.26.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.4.0",
    "pytest-asyncio>=0.23.0",
    "ruff>=0.1.14",        # Fast Python linter
    "locust>=2.20.0",      # Load testing
    "jupyter>=1.0.0",
    "matplotlib>=3.8.0",
    "scikit-learn>=1.4.0",  # For evaluation metrics
]

[tool.ruff]
line-length = 100
target-version = "py311"
```

**Step 4: Install dependencies**

```bash
# Install the project in editable mode with dev dependencies
pip install -e ".[dev]"

# Download NLTK data (needed for tokenization and WordNet)
python -c "import nltk; nltk.download('punkt'); nltk.download('wordnet'); nltk.download('stopwords')"
```

**Step 5: Set up Docker Compose for PostgreSQL and Redis**

> **Concept: Why Docker Compose?**
>
> Instead of installing PostgreSQL and Redis on your machine (which can cause version conflicts and OS-specific issues), Docker Compose runs them in isolated containers. One YAML file, one command, and everything works identically on any machine.

```yaml
# docker-compose.yml
version: "3.8"

services:
  postgres:
    image: postgres:16-alpine
    container_name: search-postgres
    environment:
      POSTGRES_DB: search_engine
      POSTGRES_USER: search_user
      POSTGRES_PASSWORD: search_password
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U search_user -d search_engine"]
      interval: 5s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: search-redis
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5

volumes:
  postgres_data:
  redis_data:
```

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Verify they're running
docker-compose ps
# Should show both containers as "Up (healthy)"
```

**Step 6: Create environment configuration**

```python
# src/config.py
"""
Centralized configuration using environment variables.

WHY: Hardcoding database passwords, API keys, or file paths is a security
risk and makes deployment difficult. Environment variables let you change
configuration without changing code.

The pattern: Define defaults for development, override with .env or
environment variables in production.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if it exists (development only)
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = DATA_DIR / "indexes"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

# Database
POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://search_user:search_password@localhost:5432/search_engine"
)
# Async version (for FastAPI)
ASYNC_POSTGRES_URL = POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Search configuration
BM25_TOP_K = int(os.getenv("BM25_TOP_K", "100"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "100"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
RESULTS_PER_PAGE = int(os.getenv("RESULTS_PER_PAGE", "10"))

# Model paths
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
LTR_MODEL_PATH = MODELS_DIR / "ltr" / "lambdamart.txt"
INTENT_MODEL_PATH = MODELS_DIR / "intent" / "intent_classifier.bin"

# Cache
CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))  # 5 minutes

# API
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
```

```bash
# .env.example (copy to .env and modify)
POSTGRES_URL=postgresql://search_user:search_password@localhost:5432/search_engine
REDIS_URL=redis://localhost:6379/0
EMBEDDING_MODEL=all-MiniLM-L6-v2
API_PORT=8000
```

### 0.3 Download the MS MARCO Dataset

> **Concept: What is MS MARCO?**
>
> MS MARCO (Microsoft MAchine Reading COmprehension) is a dataset of real Bing search queries with human-judged relevant passages. It's the **industry standard benchmark** for search engine research. When you report your NDCG@10 on MS MARCO, anyone in the field instantly knows how your system compares.
>
> The dataset contains:
> - **8.8 million passages** (short text paragraphs from web pages)
> - **100,000+ queries** with relevance judgments
> - Each query has a set of passages labeled as "relevant" or "not relevant"

```python
# scripts/download_msmarco.py
"""
Downloads the MS MARCO passage ranking dataset.

This script fetches:
1. collection.tsv — All 8.8M passages (id, text)
2. queries.train.tsv — Training queries
3. queries.dev.small.tsv — Development (validation) queries
4. qrels.train.tsv — Training relevance labels (query_id → passage_id)
5. qrels.dev.small.tsv — Dev relevance labels

WHY these specific files:
- collection.tsv: The documents we'll index and search over
- queries + qrels: Used to EVALUATE our search quality (compute NDCG@10)
- train vs dev: We train on train, evaluate on dev (never evaluate on training data!)
"""
import os
import gzip
import urllib.request
from pathlib import Path
from tqdm import tqdm

# MS MARCO URLs
MSMARCO_BASE = "https://msmarco.z22.web.core.windows.net/msmarcoranking"

FILES = {
    "collection.tsv.gz": f"{MSMARCO_BASE}/collection.tar.gz",
    "queries.train.tsv.gz": f"{MSMARCO_BASE}/queries.tar.gz",
    "qrels.train.tsv": f"{MSMARCO_BASE}/qrels.train.tsv",
    "qrels.dev.small.tsv": f"{MSMARCO_BASE}/qrels.dev.small.tsv",
    "queries.dev.small.tsv": f"{MSMARCO_BASE}/queries.tar.gz",
}

DATA_DIR = Path(__file__).parent.parent / "data" / "msmarco"


class DownloadProgressBar(tqdm):
    """Shows a progress bar while downloading."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, output_path: Path):
    """Download a file with progress bar."""
    if output_path.exists():
        print(f"  ✓ Already exists: {output_path.name}")
        return

    print(f"  ↓ Downloading: {output_path.name}")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path.name) as t:
        urllib.request.urlretrieve(url, filename=str(output_path), reporthook=t.update_to)


def main():
    """Download and prepare MS MARCO dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Downloading MS MARCO Passage Ranking Dataset")
    print("=" * 60)
    print(f"\nTarget directory: {DATA_DIR}")
    print(f"This will download ~2GB of data.\n")

    # For simplicity, we'll download a smaller subset for development.
    # The full collection is 1GB+ compressed.
    # You can use the full dataset later for final benchmarks.

    # Download qrels (relevance judgments) - these are small files
    print("\n[1/3] Downloading relevance judgments...")
    download_file(
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.train.tsv",
        DATA_DIR / "qrels.train.tsv"
    )
    download_file(
        "https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.small.tsv",
        DATA_DIR / "qrels.dev.small.tsv"
    )

    print("\n[2/3] Downloading queries...")
    # These are typically bundled; check official MS MARCO docs for exact URLs
    # For now, we provide the pattern.

    print("\n[3/3] Downloading passages collection...")
    # The full collection is large; for development you may want to use
    # a subset. See the MS MARCO GitHub for download instructions:
    # https://github.com/microsoft/MSMARCO-Passage-Ranking

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Files saved to: {DATA_DIR}")
    print("\nNext step: Run 'python scripts/index_documents.py' to build search indexes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
```

> **📚 Resource:** [MS MARCO Official Repository](https://github.com/microsoft/MSMARCO-Passage-Ranking) — Contains download links, data format descriptions, and leaderboard.

### 0.4 Database Schema Setup

> **Concept: Why define the schema upfront?**
>
> The database schema is the **contract** between all parts of your system. The indexing pipeline writes documents, the search API reads them, the click logger writes events, and the trainer reads events. If everyone agrees on the schema upfront, each component can be built independently.

```python
# src/db/models.py
"""
SQLAlchemy ORM models for the search engine database.

WHAT IS AN ORM?
ORM (Object-Relational Mapping) lets you interact with the database
using Python objects instead of raw SQL strings. Instead of:

    cursor.execute("INSERT INTO documents (id, title, body) VALUES (?, ?, ?)", ...)

You write:

    doc = Document(title="Hello", body="World")
    session.add(doc)

WHY SQLAlchemy?
- Most popular Python ORM
- Supports both sync (psycopg2) and async (asyncpg) drivers
- Type-safe queries
- Migration support (via Alembic)
"""
import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Text, Integer, Float, DateTime,
    Boolean, ForeignKey, Index, ARRAY
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""
    pass


class Document(Base):
    """
    A searchable document (passage from MS MARCO or your own data).

    WHY these columns:
    - id: Unique identifier. UUID is better than auto-increment because
      it's globally unique and doesn't leak information about your data size.
    - title: Used for BM25 title-boosting (matching query in title is important)
    - body: The main searchable text
    - url: Source link (for display in search results)
    - category: For filtered search ("show me only 'technology' results")
    - view_count: Used as a feature in LTR model (popular docs are often relevant)
    - created_at / updated_at: For freshness features in LTR
    - metadata: Flexible JSONB column for anything else you need later
    """
    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)
    url = Column(String(2000), nullable=True)
    category = Column(String(100), nullable=True, index=True)
    view_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    metadata_ = Column("metadata", JSONB, nullable=True)

    # Index for faster category filtering
    __table_args__ = (
        Index("idx_documents_category", "category"),
        Index("idx_documents_created_at", "created_at"),
    )


class SearchEvent(Base):
    """
    Records every search query and which results were shown.

    WHY this table:
    - Enables us to compute "impressions" (how many times a doc was shown)
    - Combined with ClickEvent, we can compute CTR (click-through rate)
    - Used by the LTR model trainer to generate training features
    """
    __tablename__ = "search_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    query = Column(Text, nullable=False)
    corrected_query = Column(Text, nullable=True)
    result_doc_ids = Column(ARRAY(UUID(as_uuid=True)), nullable=False)
    result_scores = Column(ARRAY(Float), nullable=False)
    latency_ms = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)


class ClickEvent(Base):
    """
    Records when a user clicks on a search result.

    WHY these specific columns:
    - position: Critical for position bias correction. A click at position 1
      means less than a click at position 5 (because users always see position 1).
    - dwell_time_seconds: How long the user spent on the page after clicking.
      Long dwell (>30s) suggests genuine interest. Short dwell (<5s) suggests
      the result was misleading (known as "pogo-sticking").
    - is_last_click: The last click in a session is often the "satisfying" click —
      the user found what they were looking for and stopped searching.
    """
    __tablename__ = "click_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(UUID(as_uuid=True), nullable=False, index=True)
    query = Column(Text, nullable=False)
    document_id = Column(UUID(as_uuid=True), ForeignKey("documents.id"), nullable=False)
    position = Column(Integer, nullable=False)  # 1-indexed position in results
    dwell_time_seconds = Column(Float, nullable=True)
    is_last_click = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Composite index: look up all clicks for a (query, document) pair
    __table_args__ = (
        Index("idx_click_query_doc", "query", "document_id"),
        Index("idx_click_session", "session_id", "timestamp"),
    )


class ModelVersion(Base):
    """
    Tracks trained model versions and their performance metrics.

    WHY this table:
    - You need to answer: "Is my new LTR model better than the old one?"
    - Model versioning is essential for safe deployments (rollback if worse)
    - Shows production ML maturity in interviews
    """
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_type = Column(String(50), nullable=False)  # 'ltr', 'embedding', 'intent'
    version = Column(String(50), nullable=False)
    ndcg_at_10 = Column(Float, nullable=True)
    mrr = Column(Float, nullable=True)
    trained_on_samples = Column(Integer, nullable=True)
    file_path = Column(String(500), nullable=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    notes = Column(Text, nullable=True)
```

```python
# src/db/postgres.py
"""
PostgreSQL connection management.

CONCEPT: Connection Pooling

Creating a database connection is EXPENSIVE (~50ms). If every API request
creates a new connection, you waste time and eventually run out of
database connections (PostgreSQL has a limit, usually 100).

Instead, we use a CONNECTION POOL: a set of pre-created connections that
are borrowed and returned. Creating the pool happens once at startup.
Each request borrows a connection (< 0.1ms), uses it, and returns it.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from src.config import POSTGRES_URL, ASYNC_POSTGRES_URL
from src.db.models import Base

# Synchronous engine (for scripts, indexing, training)
engine = create_engine(
    POSTGRES_URL,
    pool_size=5,           # Keep 5 connections ready
    max_overflow=10,       # Allow up to 10 extra connections under load
    pool_timeout=30,       # Wait up to 30s for a connection before erroring
    pool_recycle=3600,     # Recycle connections every hour (prevents stale connections)
    echo=False,            # Set True to see SQL queries (useful for debugging)
)
SessionLocal = sessionmaker(bind=engine)

# Async engine (for FastAPI endpoints)
async_engine = create_async_engine(
    ASYNC_POSTGRES_URL,
    pool_size=10,
    max_overflow=20,
)
AsyncSessionLocal = async_sessionmaker(bind=async_engine, class_=AsyncSession)


def init_db():
    """
    Create all tables if they don't exist.

    WHEN TO CALL: Once, at application startup or via a CLI command.

    NOTE: In production, you'd use Alembic for migrations instead of
    create_all(). But for a project, this is fine.
    """
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created successfully.")


def get_session() -> Session:
    """Get a synchronous database session (for scripts)."""
    return SessionLocal()


async def get_async_session() -> AsyncSession:
    """Get an async database session (for FastAPI dependency injection)."""
    async with AsyncSessionLocal() as session:
        yield session
```

```python
# src/db/redis_client.py
"""
Redis client for caching and rate limiting.

CONCEPT: Why Redis for caching?

When 100 users search "python tutorial" in the same minute, we don't
want to run the full search pipeline 100 times. The first search
computes the results and stores them in Redis. The next 99 searches
just read from Redis (< 0.5ms vs ~80ms for the full pipeline).

Redis is an in-memory data store, meaning it stores everything in RAM.
This makes it incredibly fast (sub-millisecond reads) but also means
data is lost if Redis restarts (that's fine for cache — we just
recompute on the next request).
"""
import json
import hashlib
from typing import Optional

import redis.asyncio as aioredis

from src.config import REDIS_URL, CACHE_TTL_SECONDS


class RedisCache:
    """Async Redis client for search result caching."""

    def __init__(self):
        self.client: Optional[aioredis.Redis] = None

    async def connect(self):
        """Initialize the Redis connection pool."""
        self.client = aioredis.from_url(
            REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
        )
        # Test connection
        await self.client.ping()
        print("✓ Redis connected successfully.")

    async def close(self):
        """Close the Redis connection."""
        if self.client:
            await self.client.close()

    def _make_key(self, query: str, page: int = 1, size: int = 10) -> str:
        """
        Generate a cache key from search parameters.

        WHY hash the query?
        - Queries can contain special characters that break Redis keys
        - Hashing normalizes the key length (always 32 characters)
        - We normalize to lowercase to treat "Python" and "python" as the same query
        """
        normalized = query.strip().lower()
        query_hash = hashlib.md5(normalized.encode()).hexdigest()
        return f"search:{query_hash}:p{page}:s{size}"

    async def get_cached_results(self, query: str, page: int = 1, size: int = 10) -> Optional[dict]:
        """
        Check if search results are cached.

        Returns None if not cached (cache miss).
        Returns the results dict if cached (cache hit).
        """
        if not self.client:
            return None

        key = self._make_key(query, page, size)
        cached = await self.client.get(key)

        if cached:
            return json.loads(cached)
        return None

    async def cache_results(self, query: str, results: dict, page: int = 1, size: int = 10):
        """
        Store search results in cache with TTL (Time To Live).

        After CACHE_TTL_SECONDS (default: 300 = 5 minutes), Redis automatically
        deletes the entry. This ensures:
        - Fresh results if documents are re-indexed
        - Memory doesn't grow unbounded
        - Users get updated results periodically
        """
        if not self.client:
            return

        key = self._make_key(query, page, size)
        await self.client.setex(
            key,
            CACHE_TTL_SECONDS,
            json.dumps(results),
        )

    async def invalidate_all(self):
        """
        Clear all cached search results.

        WHEN: After re-indexing documents. Old cached results would point
        to stale data, so we flush everything.
        """
        if not self.client:
            return

        # Delete all keys matching the "search:*" pattern
        cursor = 0
        while True:
            cursor, keys = await self.client.scan(cursor, match="search:*", count=100)
            if keys:
                await self.client.delete(*keys)
            if cursor == 0:
                break


# Global singleton (initialized once at app startup)
cache = RedisCache()
```

**Now run the database initialization:**

```bash
# Initialize the database tables
python -c "from src.db.postgres import init_db; init_db()"
```

> **📚 Resources for Phase 0:**
> - [FastAPI Documentation](https://fastapi.tiangolo.com/) — Excellent, one of the best docs in Python
> - [SQLAlchemy 2.0 Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/) — Modern SQLAlchemy patterns
> - [Docker Compose Getting Started](https://docs.docker.com/compose/gettingstarted/) — Docker basics
> - [Redis Commands Reference](https://redis.io/commands/) — Quick lookup for Redis operations

---

## Phase 1: BM25 Keyword Search (Week 1-2)

> **What you'll build in this phase:**
> A working search API that takes a query and returns keyword-matched results using BM25. This is your foundation — everything else builds on top of it.

### 1.1 Understanding BM25 from First Principles

Before writing code, let's deeply understand the algorithm we're implementing.

> **Concept: BM25 (Best Matching 25)**
>
> BM25 is a "bag of words" retrieval function. "Bag of words" means it treats a document as an unordered collection of words (it doesn't care about word order). Despite this limitation, BM25 has been the dominant search algorithm for 30+ years because it's fast, simple, and surprisingly effective.

**The BM25 formula:**

```
BM25(query, document) = Σ  IDF(qi) × [ f(qi, D) × (k1 + 1) ]
                        i              ────────────────────────
                                       f(qi, D) + k1 × (1 - b + b × |D|/avgdl)

Where:
  qi         = each word in the query
  f(qi, D)   = frequency of word qi in document D
  |D|        = length of document D (in words)
  avgdl      = average document length across all documents
  k1         = 1.2 (controls how much term frequency matters)
  b          = 0.75 (controls how much document length matters)
  IDF(qi)    = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
  N          = total number of documents
  n(qi)      = number of documents containing word qi
```

**Let's break this formula apart piece by piece:**

```
Part 1: IDF (Inverse Document Frequency)
─────────────────────────────────────────
"How rare is this word across all documents?"

If a word appears in EVERY document (like "the"), it's useless for search.
If a word appears in only 5 documents (like "TensorFlow"), it's very informative.

IDF("the") = log((1000000 - 999990 + 0.5) / (999990 + 0.5) + 1) ≈ 0.0
IDF("TensorFlow") = log((1000000 - 5 + 0.5) / (5 + 0.5) + 1) ≈ 12.1

"TensorFlow" is 12x more valuable than "the" as a search signal.


Part 2: Term Frequency Saturation
──────────────────────────────────
"Does the word appear in the document? How many times?"

If "python" appears 1 time, the document is probably about Python.
If "python" appears 50 times, is it 50x more relevant? No!
BM25 uses a SATURATION curve: big boost for the first few occurrences,
diminishing returns after that.

f=1:  (1 × 2.2) / (1 + 1.2 × ...) ≈ 1.0  (first occurrence: big boost)
f=5:  (5 × 2.2) / (5 + 1.2 × ...) ≈ 1.6  (five times: some extra boost)
f=50: (50 × 2.2) / (50 + 1.2 × ...) ≈ 2.0  (fifty times: barely more)

The k1 parameter controls this curve. Higher k1 = slower saturation.


Part 3: Document Length Normalization
─────────────────────────────────────
"Is this a short, focused document or a long encyclopedia entry?"

A 100-word document mentioning "python" 5 times is more focused
than a 10,000-word document mentioning "python" 5 times.

The b parameter (0.75) controls how much length matters:
  b=0: Ignore document length entirely
  b=1: Full normalization (short docs heavily favored)
  b=0.75: Default, good balance
```

### 1.2 Building the BM25 Search Module

```python
# src/search/bm25.py
"""
BM25 keyword search using Tantivy.

WHAT IS TANTIVY?
Tantivy is a full-text search engine library written in Rust (like Lucene
but faster and without Java). We use it through Python bindings (tantivy-py).

It handles:
- Building the inverted index
- BM25 scoring
- Tokenization (splitting text into words)
- Query parsing ("python AND tutorial" vs "python OR tutorial")

We DON'T need to implement BM25 from scratch — Tantivy does it for us.
Our job is to:
1. Feed it documents
2. Build the index
3. Query it efficiently
"""
import tantivy
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from src.config import INDEX_DIR


@dataclass
class BM25Result:
    """
    Represents a single BM25 search result.

    WHY a dataclass instead of a dict?
    - Type safety: your IDE catches typos (result.scroe → error)
    - Documentation: you can see exactly what fields are available
    - Immutability: prevents accidental modification
    """
    doc_id: str
    title: str
    body: str
    score: float


class BM25SearchEngine:
    """
    BM25 keyword search engine backed by Tantivy.

    LIFECYCLE:
    1. build_index(documents) — called once to create the search index
    2. search(query, top_k) — called on every search request
    3. The index is persisted to disk, so you only build once
    """

    def __init__(self, index_path: Optional[Path] = None):
        """
        Initialize the BM25 engine.

        The schema defines what fields exist in each document and how
        they're indexed. Think of it like a database table definition.
        """
        self.index_path = index_path or (INDEX_DIR / "tantivy")
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Define the document schema
        # Each field has a type and indexing options:
        #   TEXT = full-text searchable (tokenized, BM25-scored)
        #   STORED = retrievable in results (but not searchable)
        #   STRING = exact match only (not tokenized)
        schema_builder = tantivy.SchemaBuilder()

        # doc_id: stored but not searchable (we use it to join with PostgreSQL)
        schema_builder.add_text_field("doc_id", stored=True, tokenizer_name="raw")

        # title: searchable AND stored (we want to search titles AND show them)
        schema_builder.add_text_field("title", stored=True)

        # body: searchable AND stored
        # We search body text and display snippets in results
        schema_builder.add_text_field("body", stored=True)

        self.schema = schema_builder.build()
        self.index: Optional[tantivy.Index] = None
        self.searcher: Optional[tantivy.Searcher] = None

    def build_index(self, documents: List[dict], batch_size: int = 1000):
        """
        Build the BM25 inverted index from a list of documents.

        HOW THIS WORKS:
        1. Create a new empty index on disk
        2. Open a "writer" (think of it like a database transaction)
        3. Add documents in batches (for memory efficiency)
        4. Commit (finalize the index on disk)

        BATCH PROCESSING:
        Instead of adding all 8.8M documents at once (would use too much RAM),
        we add them in batches of 1000, committing periodically. This keeps
        memory usage manageable.

        Args:
            documents: List of dicts with keys: 'doc_id', 'title', 'body'
            batch_size: Number of documents to add before committing
        """
        print(f"Building BM25 index at: {self.index_path}")
        print(f"  Documents: {len(documents):,}")
        print(f"  Batch size: {batch_size:,}")

        # Create new index (overwrites existing)
        self.index = tantivy.Index(self.schema, path=str(self.index_path))

        # Writer with 256MB memory budget for indexing
        # Higher memory = faster indexing (more data buffered before flushing to disk)
        writer = self.index.writer(heap_size=256_000_000)

        for i, doc in enumerate(documents):
            # Add document to the index
            writer.add_document(tantivy.Document(
                doc_id=doc["doc_id"],
                title=doc["title"],
                body=doc["body"],
            ))

            # Commit every batch_size documents
            if (i + 1) % batch_size == 0:
                writer.commit()
                print(f"  Indexed {i + 1:,} documents...")

        # Final commit for remaining documents
        writer.commit()

        # Reload the index to make new documents searchable
        self.index.reload()
        self.searcher = self.index.searcher()

        print(f"  ✓ Index built successfully! ({len(documents):,} documents)")

    def load_index(self):
        """
        Load an existing index from disk.

        WHEN: At application startup. The index was built previously
        by the indexing pipeline; now we just load it for searching.
        """
        if not (self.index_path / "meta.json").exists():
            raise FileNotFoundError(
                f"No BM25 index found at {self.index_path}. "
                f"Run the indexing pipeline first: python scripts/index_documents.py"
            )

        self.index = tantivy.Index(self.schema, path=str(self.index_path))
        self.index.reload()
        self.searcher = self.index.searcher()
        print(f"✓ BM25 index loaded from {self.index_path}")

    def search(self, query: str, top_k: int = 100) -> List[BM25Result]:
        """
        Search the index for documents matching the query.

        HOW TANTIVY PROCESSES A QUERY:
        1. Tokenize: "python data structures" → ["python", "data", "structures"]
        2. For each token, look up the inverted index (which documents contain it)
        3. Score each matching document using BM25
        4. Return the top-K highest-scoring documents

        MULTI-FIELD SEARCH:
        We search both title and body fields. Tantivy automatically combines
        scores from both fields. A match in the title is weighted equally
        to a match in the body by default.

        Args:
            query: The search query string
            top_k: Number of results to return (we use 100, then fuse with vector results)

        Returns:
            List of BM25Result objects, sorted by descending BM25 score
        """
        if not self.searcher:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        # Parse the query against both title and body fields
        # This means "python tutorial" will match documents that have
        # "python" OR "tutorial" in their title OR body
        query_parser = tantivy.QueryParser.for_index(
            self.index, ["title", "body"]
        )
        parsed_query = query_parser.parse_query(query)

        # Execute the search
        # top_k here is the number of results Tantivy returns
        search_results = self.searcher.search(parsed_query, limit=top_k)

        # Convert Tantivy results to our BM25Result objects
        results = []
        for score, doc_address in search_results.hits:
            doc = self.searcher.doc(doc_address)
            results.append(BM25Result(
                doc_id=doc["doc_id"][0],
                title=doc["title"][0],
                body=doc["body"][0],
                score=score,
            ))

        return results
```

> **💡 Best Practice: Dataclasses for Data Transfer**
>
> Notice how we use `@dataclass` for `BM25Result` instead of passing dictionaries around. This is a best practice because:
> 1. **IDE autocomplete** — Type `result.` and see all available fields
> 2. **Catch typos** — `result.scroe` gives an error instead of silently returning `None`
> 3. **Self-documenting** — Anyone reading the code knows exactly what a search result contains

### 1.3 Building the Document Preprocessor

```python
# src/indexing/preprocessor.py
"""
Text preprocessing for search indexing.

WHY PREPROCESS?
Raw text from web pages, PDFs, or databases contains noise:
- HTML tags: <p>Hello <b>world</b></p>
- Excessive whitespace: "Hello     world"
- Special characters that confuse tokenizers
- Very short or empty documents that waste index space

Cleaning this up BEFORE indexing improves search quality because
the BM25 algorithm operates on clean tokens.

IMPORTANT: The same preprocessing must be applied to BOTH documents
(at index time) AND queries (at search time). Otherwise, a query for
"Python" won't match a document where we lowercased everything to "python".
However, we let Tantivy handle case-normalization during tokenization,
so we only do structural cleanup here.
"""
import re
import html
from typing import Optional


def clean_text(text: str) -> str:
    """
    Clean and normalize text for indexing.

    Steps:
    1. Decode HTML entities (&amp; → &, &lt; → <)
    2. Remove HTML tags (<p>, <div>, etc.)
    3. Collapse whitespace (multiple spaces/newlines → single space)
    4. Strip leading/trailing whitespace

    NOTE: We do NOT lowercase here. Tantivy's default tokenizer
    handles case-insensitive matching. Keeping original case lets us
    display proper results to the user.
    """
    # Step 1: Decode HTML entities
    # "Python &amp; Java" → "Python & Java"
    text = html.unescape(text)

    # Step 2: Remove HTML tags
    # "<p>Hello <b>world</b></p>" → "Hello world"
    text = re.sub(r"<[^>]+>", " ", text)

    # Step 3: Collapse whitespace
    # "Hello     world\n\nhow   are you" → "Hello world how are you"
    text = re.sub(r"\s+", " ", text)

    # Step 4: Strip leading/trailing whitespace
    text = text.strip()

    return text


def is_valid_document(title: str, body: str, min_body_length: int = 20) -> bool:
    """
    Check if a document is worth indexing.

    WHY FILTER?
    - Documents with < 20 characters of body text are usually noise
      (navigation menus, footers, error pages)
    - Empty titles make results useless (user can't tell what to click)
    - Filtering bad documents improves precision (fewer irrelevant results)

    Args:
        title: Document title
        body: Document body text
        min_body_length: Minimum body length in characters

    Returns:
        True if the document should be indexed, False otherwise
    """
    if not title or not title.strip():
        return False

    if not body or len(body.strip()) < min_body_length:
        return False

    return True


def generate_snippet(body: str, query: str, snippet_length: int = 200) -> str:
    """
    Generate a search result snippet that highlights the query match.

    WHAT IS A SNIPPET?
    When Google shows search results, it doesn't show the entire web page.
    It shows a short excerpt (snippet) that contains your search terms.
    This helps users decide which result to click.

    OUR APPROACH:
    1. Find where the query terms appear in the document
    2. Extract a window of text around the first match
    3. If no match found, return the first snippet_length characters

    NOTE: This is a simple implementation. Google's snippet generation
    is much more sophisticated (considers sentence boundaries, multiple
    query terms, etc.). You can improve this later.
    """
    if not body:
        return ""

    # Simple approach: find the first occurrence of any query word
    query_terms = query.lower().split()
    body_lower = body.lower()

    best_pos = len(body)  # Default: start of document

    for term in query_terms:
        pos = body_lower.find(term)
        if pos != -1 and pos < best_pos:
            best_pos = pos

    # Extract snippet centered around the match
    start = max(0, best_pos - snippet_length // 4)
    end = min(len(body), start + snippet_length)

    snippet = body[start:end]

    # Add ellipsis if we didn't start from the beginning
    if start > 0:
        snippet = "..." + snippet
    if end < len(body):
        snippet = snippet + "..."

    return snippet
```

### 1.4 The Indexing Pipeline

```python
# src/indexing/pipeline.py
"""
Document indexing pipeline: reads raw data → preprocesses → builds indexes.

This is the OFFLINE part of the system. It runs once (or periodically
when you add new documents), not on every search request.

PIPELINE PATTERN:
This follows the "pipeline" pattern where data flows through a series
of transformation steps:
  Raw data → Clean → Validate → Index in BM25 → Index in FAISS → Done

Each step is a pure function: input → output, no side effects.
This makes each step independently testable.
"""
import csv
from pathlib import Path
from typing import List, Generator
from tqdm import tqdm

from src.indexing.preprocessor import clean_text, is_valid_document
from src.search.bm25 import BM25SearchEngine
from src.config import DATA_DIR


def load_msmarco_passages(
    collection_path: Path,
    max_documents: int = None,
) -> Generator[dict, None, None]:
    """
    Load passages from MS MARCO collection.tsv file.

    MS MARCO FORMAT:
    collection.tsv is a tab-separated file with two columns:
        passage_id\tpassage_text

    Example:
        0\tThe presence of communication amid...
        1\tThe Manhattan Project was a research...

    WHY A GENERATOR?
    The full MS MARCO collection has 8.8 million passages.
    Loading all of them into memory at once would use ~10GB of RAM.
    A generator yields one document at a time, using constant memory.

    (If you don't know generators: they're like lazy lists that compute
     each element on demand. The 'yield' keyword makes a function a generator.)

    Args:
        collection_path: Path to collection.tsv
        max_documents: Optional limit (use during development to test faster)

    Yields:
        dict with keys: doc_id, title, body
    """
    print(f"Loading passages from: {collection_path}")

    count = 0
    with open(collection_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue  # Skip malformed rows

            passage_id, passage_text = row[0], row[1]

            # Clean the text
            cleaned = clean_text(passage_text)

            # Validate
            if not is_valid_document(title=f"Passage {passage_id}", body=cleaned):
                continue

            yield {
                "doc_id": passage_id,
                "title": cleaned[:100],  # Use first 100 chars as title
                "body": cleaned,
            }

            count += 1
            if max_documents and count >= max_documents:
                print(f"  Reached limit of {max_documents:,} documents.")
                return

    print(f"  Loaded {count:,} passages.")


def run_indexing_pipeline(
    collection_path: Path = None,
    max_documents: int = None,
):
    """
    Run the full indexing pipeline.

    STEPS:
    1. Load raw documents from MS MARCO
    2. Preprocess (clean text, validate)
    3. Build BM25 index (Tantivy)
    4. (Phase 2 will add: Generate embeddings → Build FAISS index)

    This function is the ENTRY POINT for indexing.
    Run it via: python scripts/index_documents.py
    """
    if collection_path is None:
        collection_path = DATA_DIR / "msmarco" / "collection.tsv"

    if not collection_path.exists():
        raise FileNotFoundError(
            f"Collection file not found: {collection_path}\n"
            f"Run 'python scripts/download_msmarco.py' first."
        )

    print("=" * 60)
    print("Starting Indexing Pipeline")
    print("=" * 60)

    # Step 1: Load documents
    # We need to materialize the generator because Tantivy needs
    # all documents at once for efficient batch indexing
    print("\n[1/2] Loading and preprocessing documents...")
    documents = list(tqdm(
        load_msmarco_passages(collection_path, max_documents),
        desc="Loading",
        total=max_documents,  # Approximate; generator doesn't know total
    ))
    print(f"  Loaded {len(documents):,} valid documents.")

    # Step 2: Build BM25 index
    print("\n[2/2] Building BM25 index...")
    bm25 = BM25SearchEngine()
    bm25.build_index(documents, batch_size=5000)

    print("\n" + "=" * 60)
    print("Indexing Pipeline Complete!")
    print(f"  BM25 index: {bm25.index_path}")
    print("=" * 60)

    return bm25
```

```python
# scripts/index_documents.py
"""
CLI script to run the indexing pipeline.

USAGE:
  # Index all documents (full dataset)
  python scripts/index_documents.py

  # Index only first 10,000 documents (for quick testing)
  python scripts/index_documents.py --max-docs 10000
"""
import argparse
from src.indexing.pipeline import run_indexing_pipeline


def main():
    parser = argparse.ArgumentParser(description="Index documents for search")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to index (default: all)"
    )
    args = parser.parse_args()

    # Run the pipeline
    bm25 = run_indexing_pipeline(max_documents=args.max_docs)

    # Quick smoke test: search for something and print results
    print("\n--- Smoke Test ---")
    results = bm25.search("what is python programming language", top_k=5)
    for i, result in enumerate(results, 1):
        print(f"\n  #{i} (score: {result.score:.2f})")
        print(f"  Title: {result.title}")
        print(f"  Body:  {result.body[:150]}...")


if __name__ == "__main__":
    main()
```

### 1.5 Building the FastAPI Application

```python
# src/api/models.py
"""
Pydantic models for API request/response validation.

CONCEPT: Pydantic
Pydantic validates data automatically. Instead of manually checking:
    if "query" not in request:
        return {"error": "query is required"}
    if not isinstance(request["query"], str):
        return {"error": "query must be a string"}

Pydantic does this for you:
    class SearchRequest(BaseModel):
        query: str  # Must be a string, must be present

If someone sends a request without "query", FastAPI automatically
returns a 422 error with a clear message. Zero boilerplate.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class SearchRequest(BaseModel):
    """Query parameters for the search endpoint."""
    q: str = Field(
        ...,  # ... means "required"
        description="Search query string",
        min_length=1,
        max_length=500,
        examples=["python tutorial for beginners"],
    )
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    size: int = Field(default=10, ge=1, le=50, description="Results per page")


class SearchResultItem(BaseModel):
    """A single search result."""
    doc_id: str
    title: str
    snippet: str
    score: float
    position: int  # 1-indexed position in results


class SearchResponse(BaseModel):
    """The full search API response."""
    query: str
    corrected_query: Optional[str] = None
    total_results: int
    page: int
    size: int
    latency_ms: int
    results: List[SearchResultItem]


class FeedbackRequest(BaseModel):
    """
    Records a user interaction (click, dwell) for LTR training.

    WHY session_id?
    We need to group events that belong to the same search session.
    "User searched X, saw results, clicked Y, dwelled for Z seconds"
    — all of these events share a session_id so we can reconstruct the story.
    """
    session_id: str = Field(..., description="Unique session identifier")
    query: str = Field(..., description="The search query")
    doc_id: str = Field(..., description="Clicked document ID")
    position: int = Field(..., ge=1, description="Position of clicked result")
    event_type: str = Field(
        ...,
        description="Type of event: 'click', 'dwell'",
        pattern="^(click|dwell)$",
    )
    dwell_time_seconds: Optional[float] = Field(
        default=None,
        ge=0,
        description="Time spent on the document (for dwell events)",
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    bm25_index_loaded: bool
    vector_index_loaded: bool
    redis_connected: bool
```

```python
# src/api/routes/search.py
"""
Search API endpoint.

This is the main entry point for search requests. It orchestrates
the entire search pipeline and returns results.
"""
import time
from fastapi import APIRouter, Depends, Query

from src.api.models import SearchRequest, SearchResponse, SearchResultItem
from src.search.bm25 import BM25SearchEngine, BM25Result
from src.indexing.preprocessor import generate_snippet
from src.db.redis_client import cache
from src.config import RESULTS_PER_PAGE

router = APIRouter()

# Global BM25 engine instance (loaded once at startup)
# In Phase 2, we'll add vector search here too
bm25_engine: BM25SearchEngine = None


def get_bm25() -> BM25SearchEngine:
    """
    Dependency injection for the BM25 engine.

    CONCEPT: Dependency Injection
    Instead of importing a global variable, FastAPI "injects" the BM25
    engine into each route function. This makes testing easier — you can
    inject a mock BM25 engine in tests without modifying the route code.
    """
    if bm25_engine is None:
        raise RuntimeError("BM25 engine not initialized")
    return bm25_engine


@router.get("/search", response_model=SearchResponse)
async def search(
    q: str = Query(..., min_length=1, max_length=500, description="Search query"),
    page: int = Query(default=1, ge=1, description="Page number"),
    size: int = Query(default=10, ge=1, le=50, description="Results per page"),
):
    """
    Search for documents matching the query.

    THE FULL FLOW (Phase 1 — BM25 only):
    1. Check Redis cache → return immediately if cached
    2. Run BM25 search → get keyword-matched results
    3. Paginate results
    4. Cache results in Redis
    5. Return JSON response

    In later phases, steps 2 will expand to include vector search,
    re-ranking, and LTR scoring.
    """
    start_time = time.perf_counter()

    # Step 1: Check cache
    cached = await cache.get_cached_results(q, page, size)
    if cached:
        return SearchResponse(**cached)

    # Step 2: BM25 search
    bm25 = get_bm25()
    bm25_results = bm25.search(query=q, top_k=100)

    # Step 3: Paginate
    start_idx = (page - 1) * size
    end_idx = start_idx + size
    page_results = bm25_results[start_idx:end_idx]

    # Step 4: Format results
    result_items = [
        SearchResultItem(
            doc_id=result.doc_id,
            title=result.title,
            snippet=generate_snippet(result.body, q, snippet_length=200),
            score=round(result.score, 4),
            position=start_idx + i + 1,
        )
        for i, result in enumerate(page_results)
    ]

    # Compute latency
    latency_ms = int((time.perf_counter() - start_time) * 1000)

    # Build response
    response = SearchResponse(
        query=q,
        corrected_query=None,  # Phase 3 will add spell correction
        total_results=len(bm25_results),
        page=page,
        size=size,
        latency_ms=latency_ms,
        results=result_items,
    )

    # Step 5: Cache for future requests
    await cache.cache_results(q, response.model_dump(), page, size)

    return response
```

```python
# src/api/routes/feedback.py
"""
Feedback endpoint for collecting click/dwell data.

This data feeds into the Learning-to-Rank model training pipeline.
We collect it from Phase 1 even though we don't use it until Phase 4,
because we need historical data to train the model.

PRINCIPLE: Start collecting data EARLY. You can't train a model without data,
and getting enough click data takes time. By the time you build the LTR
model in Phase 4, you'll already have weeks of click data.
"""
from fastapi import APIRouter
from src.api.models import FeedbackRequest

router = APIRouter()


@router.post("/feedback")
async def record_feedback(feedback: FeedbackRequest):
    """
    Record a user interaction (click or dwell event).

    In Phase 1, we just log it to the database.
    In Phase 4, this data becomes training data for LambdaMART.
    """
    # For now, just acknowledge receipt.
    # Phase 4 will save this to PostgreSQL and use it for LTR training.
    # TODO: Save to click_events table via asyncpg

    return {
        "status": "recorded",
        "session_id": feedback.session_id,
        "event_type": feedback.event_type,
    }
```

```python
# src/api/main.py
"""
FastAPI application entry point.

This file creates the FastAPI app, registers routes, and sets up
lifecycle events (startup/shutdown).

CONCEPT: Application Lifecycle
- on_startup: Runs once when the server starts. Load indexes, connect to databases.
- on_shutdown: Runs once when the server stops. Close connections, flush buffers.

This pattern ensures resources are initialized ONCE (not on every request)
and properly cleaned up when the server stops.
"""
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import search, feedback
from src.search.bm25 import BM25SearchEngine
from src.db.redis_client import cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application startup and shutdown.

    WHAT HAPPENS AT STARTUP:
    1. Load BM25 index from disk into memory
    2. Connect to Redis
    3. (Phase 2: Load FAISS index, embedding model)
    4. (Phase 3: Load cross-encoder model)
    5. (Phase 4: Load LTR model)

    WHAT HAPPENS AT SHUTDOWN:
    1. Close Redis connection
    2. (Clean up any other resources)
    """
    # === STARTUP ===
    print("Starting search engine...")

    # Load BM25 index
    bm25 = BM25SearchEngine()
    try:
        bm25.load_index()
        search.bm25_engine = bm25
    except FileNotFoundError as e:
        print(f"⚠ BM25 index not found: {e}")
        print("  Run 'python scripts/index_documents.py' to build the index.")

    # Connect to Redis
    try:
        await cache.connect()
    except Exception as e:
        print(f"⚠ Redis connection failed: {e}")
        print("  Run 'docker-compose up -d' to start Redis.")

    print("✓ Search engine ready!\n")

    yield  # Application runs here

    # === SHUTDOWN ===
    print("\nShutting down search engine...")
    await cache.close()
    print("✓ Shutdown complete.")


# Create FastAPI application
app = FastAPI(
    title="Hybrid Neural Search Engine",
    description=(
        "A search engine combining BM25 keyword search, neural vector search, "
        "cross-encoder re-ranking, and Learning-to-Rank."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware: Allow the React frontend to call the API
# Without this, the browser blocks requests from localhost:3000 to localhost:8000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register route modules
app.include_router(search.router, tags=["Search"])
app.include_router(feedback.router, tags=["Feedback"])


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "bm25_loaded": search.bm25_engine is not None,
    }
```

**Run the API server:**

```bash
# Start the server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Test it:
# Open browser → http://localhost:8000/docs (interactive Swagger UI)
# Or use curl:
curl "http://localhost:8000/search?q=python+tutorial"
curl "http://localhost:8000/health"
```

### 1.6 Writing Your First Tests

> **Concept: Why test?**
>
> Tests aren't just about finding bugs. They're your **safety net** for refactoring. When you add vector search in Phase 2, you want to be 100% sure you didn't break BM25 search. Tests give you that confidence.
>
> For your resume: having a `tests/` directory with meaningful tests shows engineering maturity. Most freshers skip testing entirely, so this is a differentiator.

```python
# tests/test_bm25.py
"""
Tests for BM25 search functionality.

TEST NAMING CONVENTION: test_<what>_<scenario>_<expected>
Example: test_search_empty_query_returns_empty → If query is empty, results should be empty
"""
import pytest
from src.search.bm25 import BM25SearchEngine


@pytest.fixture
def bm25_with_test_data(tmp_path):
    """
    Create a small BM25 index with test documents.

    WHAT IS A FIXTURE?
    A pytest fixture is a function that provides test data or setup.
    It runs BEFORE each test function that requests it.
    tmp_path is a built-in fixture that gives a temporary directory
    (automatically cleaned up after tests).
    """
    engine = BM25SearchEngine(index_path=tmp_path / "test_index")

    test_documents = [
        {"doc_id": "1", "title": "Python Programming Tutorial",
         "body": "Learn Python programming from scratch. This tutorial covers variables, loops, and functions."},
        {"doc_id": "2", "title": "JavaScript Web Development",
         "body": "Build modern web applications using JavaScript, React, and Node.js."},
        {"doc_id": "3", "title": "Python Data Science Guide",
         "body": "Use Python for data analysis with pandas, numpy, and matplotlib."},
        {"doc_id": "4", "title": "Java Enterprise Development",
         "body": "Build enterprise applications with Java, Spring Boot, and microservices architecture."},
        {"doc_id": "5", "title": "Machine Learning with Python",
         "body": "Introduction to machine learning using scikit-learn and TensorFlow in Python."},
    ]

    engine.build_index(test_documents)
    return engine


class TestBM25Search:
    """Tests for BM25 search functionality."""

    def test_search_returns_relevant_results(self, bm25_with_test_data):
        """Searching 'python' should return Python-related documents."""
        results = bm25_with_test_data.search("python", top_k=5)

        # Should return results
        assert len(results) > 0

        # Python documents should be in results
        doc_ids = [r.doc_id for r in results]
        assert "1" in doc_ids  # Python Programming Tutorial
        assert "3" in doc_ids  # Python Data Science Guide
        assert "5" in doc_ids  # Machine Learning with Python

    def test_search_ranking_order(self, bm25_with_test_data):
        """Results should be sorted by descending BM25 score."""
        results = bm25_with_test_data.search("python programming", top_k=5)

        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score"

    def test_search_respects_top_k(self, bm25_with_test_data):
        """Should return at most top_k results."""
        results = bm25_with_test_data.search("python", top_k=2)
        assert len(results) <= 2

    def test_search_irrelevant_query(self, bm25_with_test_data):
        """A query with no matching terms should return few/no results."""
        results = bm25_with_test_data.search("quantum physics thermodynamics", top_k=5)

        # May return some results (BM25 is tolerant) but scores should be very low
        if results:
            assert results[0].score < 5.0  # Low confidence

    def test_search_exact_title_match_scores_high(self, bm25_with_test_data):
        """A query that exactly matches a document title should score very high."""
        results = bm25_with_test_data.search("Python Programming Tutorial", top_k=5)

        assert len(results) > 0
        assert results[0].doc_id == "1"  # Exact title match should be #1
```

```bash
# Run tests
pytest tests/ -v

# Expected output:
# tests/test_bm25.py::TestBM25Search::test_search_returns_relevant_results PASSED
# tests/test_bm25.py::TestBM25Search::test_search_ranking_order PASSED
# tests/test_bm25.py::TestBM25Search::test_search_respects_top_k PASSED
# ...
```

### 1.7 Evaluation: Measuring BM25 Baseline Quality

> **Concept: Why measure before improving?**
>
> This is the most important principle in ML engineering: **always establish a baseline before making changes.** You need to know how good BM25-only search is so that when you add vector search, you can prove it actually helped. "Hybrid search improved NDCG@10 by 15%" is a powerful statement. "I added vector search and it seems better" is not.

```python
# scripts/evaluate.py
"""
Evaluate search quality using MS MARCO relevance judgments.

METRICS WE COMPUTE:

1. NDCG@10 (Normalized Discounted Cumulative Gain at rank 10)
   - "How good is the ranking in the top 10 results?"
   - Range: 0 (terrible) to 1 (perfect)
   - Penalizes relevant docs that appear too low

2. MRR (Mean Reciprocal Rank)
   - "On average, what's the position of the first relevant result?"
   - MRR = 1 → first relevant result is always #1
   - MRR = 0.5 → first relevant result is on average #2

3. Recall@100
   - "Of all relevant documents, how many are in our top 100?"
   - Important because the re-ranker (Phase 3) can only score docs
     that were retrieved. If a relevant doc isn't in the top 100,
     it can never be shown to the user.

HOW TO USE:
  python scripts/evaluate.py --mode bm25 --max-queries 1000
"""
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np
from tqdm import tqdm

from src.search.bm25 import BM25SearchEngine
from src.config import DATA_DIR


def load_qrels(qrels_path) -> Dict[str, Set[str]]:
    """
    Load relevance judgments from MS MARCO qrels file.

    QRELS FORMAT:
    Each line: query_id  0  passage_id  relevance_score
    Example:   1185869   0  59219       1

    This means: for query 1185869, passage 59219 is relevant (score=1)

    Returns:
        Dict mapping query_id → set of relevant passage_ids
    """
    qrels = defaultdict(set)

    with open(qrels_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                query_id = row[0]
                passage_id = row[2]
                relevance = int(row[3])
                if relevance > 0:
                    qrels[query_id].add(passage_id)

    return dict(qrels)


def load_queries(queries_path) -> Dict[str, str]:
    """
    Load queries from MS MARCO queries file.

    FORMAT: query_id\tquery_text
    Example: 1185869\twhat is a durable power of attorney
    """
    queries = {}

    with open(queries_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[row[0]] = row[1]

    return queries


def compute_ndcg(ranked_doc_ids: List[str], relevant_ids: Set[str], k: int = 10) -> float:
    """
    Compute NDCG@k for a single query.

    STEP BY STEP:
    1. For each position i (1 to k), check if the document is relevant
    2. Relevant doc at position i gets a gain of 1 / log2(i + 1)
       (position 1: gain = 1/1 = 1.0, position 2: gain = 1/1.58 = 0.63, etc.)
    3. Sum all gains → this is DCG (Discounted Cumulative Gain)
    4. Compute the IDEAL DCG (if we placed all relevant docs at the top)
    5. NDCG = DCG / Ideal_DCG (normalized to 0-1 range)
    """
    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_ids:
            # Position is 1-indexed, so i+1
            dcg += 1.0 / np.log2(i + 2)  # i+2 because log2(1)=0

    # Ideal DCG
    ideal_dcg = 0.0
    num_relevant = min(len(relevant_ids), k)
    for i in range(num_relevant):
        ideal_dcg += 1.0 / np.log2(i + 2)

    if ideal_dcg == 0:
        return 0.0

    return dcg / ideal_dcg


def compute_mrr(ranked_doc_ids: List[str], relevant_ids: Set[str]) -> float:
    """
    Compute MRR (Mean Reciprocal Rank) for a single query.

    Find the position of the FIRST relevant document and return 1/position.
    If no relevant document is found, return 0.

    Example:
      Results: [irrelevant, irrelevant, RELEVANT, ...]
      MRR = 1/3 = 0.333
    """
    for i, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def compute_recall(ranked_doc_ids: List[str], relevant_ids: Set[str], k: int = 100) -> float:
    """
    Compute Recall@k: what fraction of relevant docs are in top k results?

    Example:
      5 relevant documents total
      3 of them appear in top 100 results
      Recall@100 = 3/5 = 0.6
    """
    if not relevant_ids:
        return 0.0

    retrieved = set(ranked_doc_ids[:k])
    found = retrieved & relevant_ids

    return len(found) / len(relevant_ids)


def evaluate_bm25(max_queries: int = 1000):
    """Run full evaluation of BM25 search on MS MARCO dev set."""
    print("=" * 60)
    print("Evaluating BM25 Search Quality")
    print("=" * 60)

    # Load evaluation data
    qrels = load_qrels(DATA_DIR / "msmarco" / "qrels.dev.small.tsv")
    queries = load_queries(DATA_DIR / "msmarco" / "queries.dev.small.tsv")

    # Filter to queries that have relevance judgments
    eval_queries = {qid: q for qid, q in queries.items() if qid in qrels}
    eval_queries = dict(list(eval_queries.items())[:max_queries])

    print(f"\n  Evaluation queries: {len(eval_queries):,}")
    print(f"  Queries with judgments: {len(qrels):,}")

    # Load BM25 engine
    bm25 = BM25SearchEngine()
    bm25.load_index()

    # Evaluate
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []

    for query_id, query_text in tqdm(eval_queries.items(), desc="Evaluating"):
        results = bm25.search(query_text, top_k=100)
        ranked_ids = [r.doc_id for r in results]
        relevant_ids = qrels[query_id]

        ndcg_scores.append(compute_ndcg(ranked_ids, relevant_ids, k=10))
        mrr_scores.append(compute_mrr(ranked_ids, relevant_ids))
        recall_scores.append(compute_recall(ranked_ids, relevant_ids, k=100))

    # Report results
    print("\n" + "=" * 60)
    print("BM25 Baseline Results")
    print("=" * 60)
    print(f"  NDCG@10:    {np.mean(ndcg_scores):.4f}")
    print(f"  MRR:        {np.mean(mrr_scores):.4f}")
    print(f"  Recall@100: {np.mean(recall_scores):.4f}")
    print(f"  Queries:    {len(eval_queries):,}")
    print("=" * 60)

    return {
        "ndcg@10": np.mean(ndcg_scores),
        "mrr": np.mean(mrr_scores),
        "recall@100": np.mean(recall_scores),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-queries", type=int, default=1000)
    args = parser.parse_args()

    evaluate_bm25(max_queries=args.max_queries)
```

> **📚 Resources for Phase 1:**
> - [BM25 Algorithm Explained](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) — Elasticsearch blog, excellent visual explanation
> - [Tantivy Python Documentation](https://github.com/quickwit-oss/tantivy-py) — API reference and examples
> - [FastAPI Tutorial](https://fastapi.tiangolo.com/tutorial/) — Official step-by-step tutorial
> - [pytest Documentation](https://docs.pytest.org/en/stable/) — Learn testing patterns
> - [MS MARCO Leaderboard](https://microsoft.github.io/msmarco/) — See how your results compare

> **✅ Phase 1 Checkpoint:**
> After completing this phase, you should have:
> - [ ] A working BM25 search API at `http://localhost:8000/search?q=python`
> - [ ] Auto-generated API docs at `http://localhost:8000/docs`
> - [ ] Baseline NDCG@10 measured on MS MARCO dev set
> - [ ] Passing unit tests for BM25 search
> - [ ] Docker Compose running PostgreSQL and Redis

---

## Phase 2: Vector Search & Hybrid Retrieval (Week 3-4)

> **What you'll build:** Add semantic (meaning-based) search alongside BM25, combine them with Reciprocal Rank Fusion, and fine-tune the embedding model on your domain.

### 2.1 Vector Search Module

```python
# src/search/vector.py
"""
Vector (semantic) search using FAISS.

THE KEY IDEA:
BM25 matches words. Vector search matches MEANINGS.

"comfortable shoes for standing" and "ergonomic footwear with arch support"
have almost no word overlap, but they MEAN the same thing.

Vector search works by:
1. Converting text into a list of numbers (a "vector" or "embedding")
2. Storing all document vectors in a FAISS index
3. At search time, converting the query to a vector
4. Finding the documents whose vectors are CLOSEST to the query vector

"Closest" is measured by cosine similarity:
  similarity = dot(A, B) / (||A|| * ||B||)
  Range: -1 (opposite) to 1 (identical meaning)
"""
import numpy as np
import faiss
from pathlib import Path
from typing import List, Optional
from dataclasses import dataclass

from sentence_transformers import SentenceTransformer

from src.config import INDEX_DIR, EMBEDDING_MODEL_NAME, MODELS_DIR


@dataclass
class VectorResult:
    """A single vector search result."""
    doc_id: str
    score: float  # Cosine similarity (0 to 1)


class VectorSearchEngine:
    """
    Semantic search engine using sentence embeddings and FAISS.

    ARCHITECTURE:
    - Embedding model: all-MiniLM-L6-v2 (384 dimensions)
    - Index type: HNSW (approximate nearest neighbor search)
    - Storage: FAISS index saved to disk

    LIFECYCLE:
    1. build_index(documents) — Generate embeddings + build FAISS index
    2. search(query, top_k) — Encode query + search FAISS index
    """

    def __init__(self, index_path: Optional[Path] = None):
        self.index_path = index_path or (INDEX_DIR / "faiss")
        self.index_path.mkdir(parents=True, exist_ok=True)

        # Paths for saved artifacts
        self.faiss_index_path = self.index_path / "index.faiss"
        self.doc_ids_path = self.index_path / "doc_ids.npy"

        # FAISS index and document ID mapping
        self.index: Optional[faiss.Index] = None
        self.doc_ids: Optional[np.ndarray] = None

        # Embedding model (loaded lazily)
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> SentenceTransformer:
        """
        Lazy-load the embedding model.

        WHY LAZY LOADING?
        The model takes ~2 seconds to load and uses ~300MB of RAM.
        We don't want to pay this cost if the model isn't needed
        (e.g., during BM25-only testing).
        """
        if self._model is None:
            # Check if we have a fine-tuned version
            fine_tuned_path = MODELS_DIR / "embeddings" / "fine-tuned"
            if fine_tuned_path.exists():
                print(f"  Loading fine-tuned embedding model: {fine_tuned_path}")
                self._model = SentenceTransformer(str(fine_tuned_path))
            else:
                print(f"  Loading pre-trained embedding model: {EMBEDDING_MODEL_NAME}")
                self._model = SentenceTransformer(EMBEDDING_MODEL_NAME)

            print(f"  Embedding dimensions: {self._model.get_sentence_embedding_dimension()}")
        return self._model

    def build_index(
        self,
        documents: List[dict],
        batch_size: int = 256,
    ):
        """
        Build FAISS HNSW index from documents.

        STEPS:
        1. Generate embeddings for all documents (in batches)
        2. Normalize vectors (required for cosine similarity with inner product)
        3. Build HNSW index
        4. Save to disk

        BATCHING:
        We encode documents in batches of 256 instead of one at a time.
        This is MUCH faster because:
        - GPU processes batches in parallel
        - Even on CPU, batched matrix operations are optimized via BLAS
        - Typical speedup: 10-50x vs one-at-a-time encoding

        Args:
            documents: List of dicts with 'doc_id' and 'body' keys
            batch_size: Number of documents to encode at once
        """
        print(f"Building vector index...")
        print(f"  Documents: {len(documents):,}")
        print(f"  Model: {EMBEDDING_MODEL_NAME}")

        # Step 1: Generate embeddings in batches
        texts = [doc["body"] for doc in documents]
        doc_ids = [doc["doc_id"] for doc in documents]

        print(f"\n  Generating embeddings (batch_size={batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2 normalize for cosine similarity
        )
        # embeddings shape: (num_documents, 384)

        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Memory: {embeddings.nbytes / 1024 / 1024:.1f} MB")

        # Step 2: Build FAISS HNSW index
        dimension = embeddings.shape[1]  # 384

        # HNSW parameters explained:
        # M=32: Each node connects to 32 neighbors in the graph.
        #   Higher M → better recall but more memory and slower queries.
        #   M=32 is a good balance for < 5M documents.
        #
        # efConstruction=200: How many nodes to check when building the graph.
        #   Higher → better quality graph, slower indexing.
        #
        # We use IndexHNSWFlat with inner product (cosine similarity, since vectors
        # are normalized). HNSW is an APPROXIMATE nearest neighbor algorithm —
        # it doesn't check every single vector, but finds near-optimal results
        # in O(log N) time.

        print(f"\n  Building FAISS HNSW index (M=32, efConstruction=200)...")
        self.index = faiss.IndexHNSWFlat(dimension, 32)  # M=32
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 128  # Search-time width

        # FAISS uses inner-product by default for HNSW, which equals
        # cosine similarity when vectors are L2-normalized
        self.index.add(embeddings.astype(np.float32))

        # Save document ID mapping
        self.doc_ids = np.array(doc_ids)

        # Step 3: Save to disk
        faiss.write_index(self.index, str(self.faiss_index_path))
        np.save(str(self.doc_ids_path), self.doc_ids)

        print(f"\n  ✓ Vector index built!")
        print(f"    Index size: {self.faiss_index_path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"    Documents indexed: {self.index.ntotal:,}")

    def load_index(self):
        """Load a previously built FAISS index from disk."""
        if not self.faiss_index_path.exists():
            raise FileNotFoundError(
                f"No vector index found at {self.faiss_index_path}. "
                f"Run the indexing pipeline first."
            )

        self.index = faiss.read_index(str(self.faiss_index_path))
        self.doc_ids = np.load(str(self.doc_ids_path), allow_pickle=True)

        # Set search-time parameters
        self.index.hnsw.efSearch = 128

        print(f"✓ Vector index loaded ({self.index.ntotal:,} documents)")

    def search(self, query: str, top_k: int = 100) -> List[VectorResult]:
        """
        Search for documents semantically similar to the query.

        STEPS:
        1. Encode the query text into a vector (384 dimensions)
        2. Search the FAISS index for nearest neighbors
        3. Convert distances to similarity scores
        4. Return results with document IDs and scores

        PERFORMANCE:
        - Query encoding: ~5ms
        - FAISS search: ~5-10ms for 1M vectors
        - Total: ~10-15ms
        """
        if self.index is None:
            raise RuntimeError("Index not loaded. Call load_index() first.")

        # Encode query
        query_vector = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)

        # Search FAISS index
        # Returns: distances (similarities) and indices
        distances, indices = self.index.search(query_vector, top_k)

        # Convert to results
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for empty slots
                continue
            results.append(VectorResult(
                doc_id=str(self.doc_ids[idx]),
                score=float(dist),  # Already a similarity score (0-1) for normalized vectors
            ))

        return results
```

### 2.2 Reciprocal Rank Fusion

```python
# src/search/fusion.py
"""
Reciprocal Rank Fusion (RRF) — combining BM25 and vector search results.

THE PROBLEM:
We have two ranked lists from two different systems:
  BM25:   [A (score 12.5), B (score 8.3), C (score 6.1)]
  Vector: [C (score 0.92), D (score 0.85), A (score 0.78)]

How do we combine them? We can't average scores because they're on
completely different scales (BM25: 0-20+, cosine similarity: 0-1).

THE SOLUTION: RRF
Instead of using scores, use RANKS. A document's RRF score is:

  RRF(doc) = Σ  1 / (k + rank_in_system_i)
             i

Where k=60 (a constant that controls how much we trust high ranks).

This is elegant because:
1. Rank-based → doesn't care about score scales
2. Documents appearing in BOTH lists get boosted (they're probably relevant)
3. The k parameter prevents over-reliance on the #1 result from either system
"""
from typing import List, Dict
from dataclasses import dataclass

from src.search.bm25 import BM25Result
from src.search.vector import VectorResult


@dataclass
class FusedResult:
    """A search result after fusion, containing scores from both systems."""
    doc_id: str
    title: str
    body: str
    rrf_score: float
    bm25_score: float  # 0.0 if not found by BM25
    vector_score: float  # 0.0 if not found by vector search
    bm25_rank: int  # 0 if not found
    vector_rank: int  # 0 if not found


def reciprocal_rank_fusion(
    bm25_results: List[BM25Result],
    vector_results: List[VectorResult],
    k: int = 60,
    top_k: int = 50,
    doc_content: Dict[str, dict] = None,
) -> List[FusedResult]:
    """
    Combine BM25 and vector search results using Reciprocal Rank Fusion.

    WHY k=60?
    The k parameter controls how much we trust high ranks:
    - k=1:   rank 1 gets score 0.50, rank 2 gets 0.33 (huge difference!)
    - k=60:  rank 1 gets score 0.016, rank 2 gets 0.016 (small difference)
    - k=1000: rank 1 ≈ rank 100 (barely any difference)

    k=60 is the standard value from the original RRF paper (Cormack et al., 2009).
    It provides a good balance: top ranks are preferred, but not overwhelmingly so.

    Args:
        bm25_results: Results from BM25 search (sorted by BM25 score)
        vector_results: Results from vector search (sorted by similarity)
        k: RRF constant (default: 60)
        top_k: Number of fused results to return
        doc_content: Optional dict of doc_id → {title, body} for results
                     that come from vector search (which doesn't store text)

    Returns:
        Fused results sorted by RRF score (descending)
    """
    # Build RRF score lookup
    rrf_scores: Dict[str, float] = {}
    bm25_score_lookup: Dict[str, float] = {}
    vector_score_lookup: Dict[str, float] = {}
    bm25_rank_lookup: Dict[str, int] = {}
    vector_rank_lookup: Dict[str, int] = {}
    content_lookup: Dict[str, dict] = {}  # doc_id → {title, body}

    # Process BM25 results
    for rank, result in enumerate(bm25_results, start=1):
        doc_id = result.doc_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        bm25_score_lookup[doc_id] = result.score
        bm25_rank_lookup[doc_id] = rank
        content_lookup[doc_id] = {"title": result.title, "body": result.body}

    # Process vector results
    for rank, result in enumerate(vector_results, start=1):
        doc_id = result.doc_id
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)
        vector_score_lookup[doc_id] = result.score
        vector_rank_lookup[doc_id] = rank

        # If we don't have content from BM25, get it from the content dict
        if doc_id not in content_lookup and doc_content and doc_id in doc_content:
            content_lookup[doc_id] = doc_content[doc_id]

    # Sort by RRF score (descending) and take top_k
    sorted_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]

    # Build fused results
    fused_results = []
    for doc_id in sorted_doc_ids:
        content = content_lookup.get(doc_id, {"title": "", "body": ""})
        fused_results.append(FusedResult(
            doc_id=doc_id,
            title=content.get("title", ""),
            body=content.get("body", ""),
            rrf_score=rrf_scores[doc_id],
            bm25_score=bm25_score_lookup.get(doc_id, 0.0),
            vector_score=vector_score_lookup.get(doc_id, 0.0),
            bm25_rank=bm25_rank_lookup.get(doc_id, 0),
            vector_rank=vector_rank_lookup.get(doc_id, 0),
        ))

    return fused_results
```

### 2.3 Fine-Tuning the Embedding Model

> **This is one of the most important ML components in the project.** Fine-tuning transforms a generic text model into one that understands YOUR specific domain, and it's custom model training that you can discuss deeply in interviews.

```python
# scripts/fine_tune_embeddings.py
"""
Fine-tune Sentence-BERT on MS MARCO for better domain-specific embeddings.

WHAT FINE-TUNING DOES:
The pre-trained model (all-MiniLM-L6-v2) understands general English.
But search has specific needs:
- "how to fix" should match "troubleshooting guide" (search-specific paraphrase)
- "Python 3.12" should be close to "Python 3.11" but far from "Python snake"

Fine-tuning teaches the model these domain-specific relationships.

HOW IT WORKS (Contrastive Learning):
- Take a (query, relevant_passage) pair from MS MARCO
- The model should make their vectors CLOSE (high similarity)
- All other passages in the training batch are "negatives"
- The model should make query vectors FAR from negative passages

After training on ~100K such pairs, the model learns what "relevant"
means in the context of search.

EXPECTED IMPROVEMENT: 5-15% better NDCG@10 compared to off-the-shelf model
"""
import csv
import random
from pathlib import Path

from sentence_transformers import (
    SentenceTransformer,
    InputExample,
    losses,
    evaluation,
)
from torch.utils.data import DataLoader

from src.config import DATA_DIR, MODELS_DIR


def load_training_pairs(
    collection_path: Path,
    queries_path: Path,
    qrels_path: Path,
    max_pairs: int = 100000,
) -> list:
    """
    Load (query, relevant_passage) pairs from MS MARCO.

    PROCESS:
    1. Load all passages into a dict: passage_id → passage_text
    2. Load all queries into a dict: query_id → query_text
    3. Load qrels (relevance judgments): query_id → relevant_passage_id
    4. Join them: (query_text, relevant_passage_text) pairs
    """
    print("Loading training pairs...")

    # Load passages
    passages = {}
    with open(collection_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                passages[row[0]] = row[1]
    print(f"  Passages: {len(passages):,}")

    # Load queries
    queries = {}
    with open(queries_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[row[0]] = row[1]
    print(f"  Queries: {len(queries):,}")

    # Load qrels and create (query, passage) pairs
    pairs = []
    with open(qrels_path, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                query_id, passage_id = row[0], row[2]
                if query_id in queries and passage_id in passages:
                    pairs.append(InputExample(
                        texts=[queries[query_id], passages[passage_id]]
                    ))

    # Shuffle and limit
    random.shuffle(pairs)
    pairs = pairs[:max_pairs]
    print(f"  Training pairs: {len(pairs):,}")

    return pairs


def fine_tune(max_pairs: int = 100000, epochs: int = 3, batch_size: int = 32):
    """
    Fine-tune the embedding model on MS MARCO.

    HYPERPARAMETERS EXPLAINED:
    - epochs=3: Number of passes through the training data.
      More epochs = more training, but risk of overfitting.
      3 is a good default for fine-tuning (not training from scratch).

    - batch_size=32: Number of (query, passage) pairs processed at once.
      Larger batches give MORE negative examples (better for contrastive learning)
      but need more GPU/CPU memory. 32 is good for 16GB RAM.

    - warmup_steps=100: Learning rate starts low and ramps up over 100 steps.
      This prevents the model from "forgetting" its pre-trained knowledge
      at the very start of training (when gradients can be chaotic).

    - learning_rate=2e-5: Standard for fine-tuning transformer models.
      Too high → model forgets pre-trained knowledge (catastrophic forgetting)
      Too low → model doesn't learn from new data
    """
    print("=" * 60)
    print("Fine-Tuning Embedding Model")
    print("=" * 60)

    # Load base model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load training data
    train_examples = load_training_pairs(
        collection_path=DATA_DIR / "msmarco" / "collection.tsv",
        queries_path=DATA_DIR / "msmarco" / "queries.train.tsv",
        qrels_path=DATA_DIR / "msmarco" / "qrels.train.tsv",
        max_pairs=max_pairs,
    )

    # Create DataLoader
    train_dataloader = DataLoader(
        train_examples,
        shuffle=True,
        batch_size=batch_size,
    )

    # Loss function: MultipleNegativesRankingLoss
    #
    # HOW THIS WORKS:
    # For each batch of 32 (query, positive_passage) pairs:
    # - query_1 should be similar to passage_1 but different from passages 2-32
    # - query_2 should be similar to passage_2 but different from passages 1,3-32
    # - ...etc
    #
    # So each query has 1 positive and 31 negatives PER BATCH.
    # This is called "in-batch negatives" and it's very efficient —
    # no need to explicitly sample negatives!
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Output path for fine-tuned model
    output_path = MODELS_DIR / "embeddings" / "fine-tuned"
    output_path.mkdir(parents=True, exist_ok=True)

    # Train!
    print(f"\n  Training for {epochs} epochs...")
    print(f"  Batch size: {batch_size}")
    print(f"  Training steps: {len(train_dataloader) * epochs:,}")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=100,
        output_path=str(output_path),
        show_progress_bar=True,
    )

    print(f"\n  ✓ Fine-tuned model saved to: {output_path}")
    print(f"  Next: Re-run the indexing pipeline to use the new model.")
    print(f"  Then: Re-run evaluation to measure improvement.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=100000)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    fine_tune(
        max_pairs=args.max_pairs,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
```

> **📚 Resources for Phase 2:**
> - [FAISS Wiki — Getting Started](https://github.com/facebookresearch/faiss/wiki/Getting-started) — FAISS indexing and searching
> - [Sentence-BERT Paper](https://arxiv.org/abs/1908.10084) — Understanding bi-encoder embeddings
> - [SBERT Training Examples](https://www.sbert.net/docs/training/overview.html) — Fine-tuning tutorials
> - [Reciprocal Rank Fusion Paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Original RRF paper

> **✅ Phase 2 Checkpoint:**
> - [ ] Vector search API working alongside BM25
> - [ ] Hybrid search (RRF fusion) returning combined results
> - [ ] Fine-tuned embedding model saved
> - [ ] Measured improvement: hybrid NDCG@10 vs BM25-only NDCG@10

---

## Phase 3: Cross-Encoder Re-Ranking (Week 5)

### 3.1 Cross-Encoder Re-Ranker

```python
# src/search/reranker.py
"""
Cross-encoder re-ranker for fine-grained relevance scoring.

WHY RE-RANK?

Bi-encoders (used in vector search) are fast but approximate.
They encode query and document SEPARATELY, so they miss interactions
between query words and document words.

Cross-encoders are slow but MUCH more accurate. They read the query
AND document TOGETHER, seeing every word interaction.

ANALOGY:
- Bi-encoder = reading two book summaries and guessing if they're related
- Cross-encoder = reading both books side by side and comparing them

We use the bi-encoder to find 50 candidates (fast, approximate),
then the cross-encoder to re-rank those 50 (slow, accurate).
"""
from typing import List
from sentence_transformers import CrossEncoder

from src.search.fusion import FusedResult
from src.config import CROSS_ENCODER_MODEL_NAME


class ReRanker:
    """Cross-encoder based re-ranker."""

    def __init__(self):
        self._model: CrossEncoder = None

    @property
    def model(self) -> CrossEncoder:
        """Lazy-load the cross-encoder model."""
        if self._model is None:
            print(f"  Loading cross-encoder: {CROSS_ENCODER_MODEL_NAME}")
            self._model = CrossEncoder(
                CROSS_ENCODER_MODEL_NAME,
                max_length=512,  # Max input tokens (query + document)
            )
            print("  ✓ Cross-encoder loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[FusedResult],
        top_k: int = 50,
    ) -> List[FusedResult]:
        """
        Re-rank candidates using the cross-encoder.

        PROCESS:
        1. Create (query, document_text) pairs for each candidate
        2. Run the cross-encoder on ALL pairs in one batch
        3. Sort by the new cross-encoder score
        4. Return top_k results

        BATCHING MATTERS:
        Processing 50 pairs individually: 50 × 50ms = 2500ms (way too slow!)
        Processing 50 pairs in a batch:   ~50ms total (GPU parallelism)
        Even on CPU, batching is 5-10x faster than sequential.

        Args:
            query: The search query
            candidates: Fused results from RRF (sorted by RRF score)
            top_k: Number of results to return after re-ranking

        Returns:
            Re-ranked results with updated scores
        """
        if not candidates:
            return []

        # Limit to maximum candidates (cross-encoder is expensive)
        candidates = candidates[:top_k]

        # Create (query, document) pairs
        # The cross-encoder expects: [["query", "doc1"], ["query", "doc2"], ...]
        pairs = [[query, result.body[:512]] for result in candidates]
        # ^^ Truncate body to 512 chars because:
        # 1. Cross-encoder has a max input length (512 tokens ≈ 400-500 words)
        # 2. The beginning of a document is usually most informative
        # 3. Reduces computation time

        # Score all pairs in one batch
        scores = self.model.predict(
            pairs,
            batch_size=32,
            show_progress_bar=False,
        )

        # Attach scores to results and sort
        scored_results = list(zip(candidates, scores))
        scored_results.sort(key=lambda x: x[1], reverse=True)

        # Update the results with cross-encoder scores
        reranked = []
        for result, ce_score in scored_results:
            # Create a new FusedResult with the cross-encoder score
            # We keep the original BM25/vector scores for LTR features
            reranked.append(FusedResult(
                doc_id=result.doc_id,
                title=result.title,
                body=result.body,
                rrf_score=float(ce_score),  # Replace RRF score with CE score for now
                bm25_score=result.bm25_score,
                vector_score=result.vector_score,
                bm25_rank=result.bm25_rank,
                vector_rank=result.vector_rank,
            ))

        return reranked
```

### 3.2 Query Understanding Module

```python
# src/query/spell_check.py
"""
Spell correction for search queries.

WHY SPELL CORRECTION MATTERS:
~10-15% of search queries contain typos. Without correction:
- "pytohn tutorial" → 0 results (no document has "pytohn")
- User thinks your search engine is broken
- With correction: "pytohn" → "python" → great results!

APPROACH: SymSpell
We use SymSpell, which uses a pre-computed dictionary of words and their
edit-distance neighbors. It finds corrections in O(1) time (vs O(n) for
naive approaches).

Edit distance = minimum number of character operations (insert, delete,
substitute, transpose) to convert one word into another:
  "pytohn" → "python" (1 transposition: oh → ho) → distance = 1
"""
from symspellpy import SymSpell, Verbosity
from pathlib import Path
import pkg_resources


class SpellChecker:
    """SymSpell-based spell correction for search queries."""

    def __init__(self):
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2)

        # Load a frequency dictionary (common English words + frequencies)
        # SymSpell ships with one built-in
        dict_path = pkg_resources.resource_filename(
            "symspellpy", "frequency_dictionary_en_82_765.txt"
        )
        self.sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

    def correct(self, query: str) -> str:
        """
        Correct spelling mistakes in the query.

        ALGORITHM:
        1. Split query into words
        2. For each word, check if it's in the dictionary
        3. If not, find the closest dictionary word within edit distance 2
        4. Replace misspelled words with corrections

        max_edit_distance_lookup=2 means we allow up to 2 character changes.
        This catches most common typos without being too aggressive
        (correcting words that don't need correction).
        """
        suggestions = self.sym_spell.lookup_compound(
            query,
            max_edit_distance=2,
            transfer_casing=True,  # Preserve original casing
        )

        if suggestions:
            return suggestions[0].term
        return query  # No correction found, return original
```

> **✅ Phase 3 Checkpoint:**
> - [ ] Cross-encoder re-ranking integrated into the pipeline
> - [ ] Spell correction working on search queries
> - [ ] Measured improvement: re-ranked NDCG@10 vs hybrid-only NDCG@10
> - [ ] Total pipeline latency still < 100ms

---

## Phase 4: Learning-to-Rank (Week 6-7)

### 4.1 Feature Engineering

```python
# src/search/ltr.py
"""
Learning-to-Rank using LambdaMART (LightGBM).

THIS IS THE MOST IMPRESSIVE ML COMPONENT.

Traditional search ranks by text similarity. LTR uses MACHINE LEARNING
to combine dozens of signals — text scores, document quality, user behavior —
into a single ranking function that optimizes for user satisfaction.

WHAT MAKES THIS SPECIAL:
1. Custom model training — you train a real model from scratch
2. Feature engineering — designing features requires domain knowledge
3. Click model — converting raw clicks into training labels is non-trivial
4. Online learning — the model improves daily as more users interact
"""
import numpy as np
import lightgbm as lgb
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class LTRFeatures:
    """
    Features for a single (query, document) pair.

    Each field is a signal that helps predict relevance.
    The LambdaMART model learns which signals matter most
    and how to combine them.
    """
    # Text relevance features
    bm25_score: float = 0.0
    vector_similarity: float = 0.0
    cross_encoder_score: float = 0.0
    query_doc_term_overlap: float = 0.0
    title_bm25_score: float = 0.0

    # Document quality features
    doc_length: int = 0
    doc_freshness_days: int = 0
    doc_view_count: int = 0

    # User behavior features (from click logs)
    historical_ctr: float = 0.0
    avg_dwell_time: float = 0.0
    skip_rate: float = 0.0

    # Query features
    query_length: int = 0

    def to_array(self) -> np.ndarray:
        """Convert to feature vector for model input."""
        return np.array([
            self.bm25_score,
            self.vector_similarity,
            self.cross_encoder_score,
            self.query_doc_term_overlap,
            self.title_bm25_score,
            self.doc_length,
            self.doc_freshness_days,
            self.doc_view_count,
            self.historical_ctr,
            self.avg_dwell_time,
            self.skip_rate,
            self.query_length,
        ])

    @staticmethod
    def feature_names() -> List[str]:
        """Feature names for model interpretability."""
        return [
            "bm25_score", "vector_similarity", "cross_encoder_score",
            "query_doc_term_overlap", "title_bm25_score",
            "doc_length", "doc_freshness_days", "doc_view_count",
            "historical_ctr", "avg_dwell_time", "skip_rate",
            "query_length",
        ]


class LTRModel:
    """
    LambdaMART Learning-to-Rank model using LightGBM.

    WHAT IS LambdaMART?
    A gradient boosted tree model optimized specifically for RANKING.

    Unlike regular regression/classification:
    - It doesn't predict an absolute score
    - It optimizes the RANKING ORDER of results
    - Its loss function (LambdaRank) directly optimizes NDCG
    - Mistakes at the top of the ranking are penalized MORE
      than mistakes at the bottom

    WHY LightGBM?
    - Native support for LambdaMART (objective='lambdarank')
    - Fastest gradient boosting implementation
    - Handles tabular features very well
    - Interpretable: feature importance tells you what matters
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model_path = model_path
        self.model: Optional[lgb.Booster] = None

    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        query_groups: np.ndarray,
    ):
        """
        Train the LambdaMART model.

        Args:
            features: shape (n_samples, n_features) — feature matrix
            labels: shape (n_samples,) — relevance labels (0-3)
            query_groups: shape (n_queries,) — number of documents per query
                Example: [10, 8, 12] means query 1 has 10 docs, query 2 has 8, etc.
                LightGBM needs this to know which documents belong to the same query
                (it optimizes ranking WITHIN each query group).

        HYPERPARAMETERS EXPLAINED:
        - n_estimators=300: Number of trees. More trees = better fit but
          risk of overfitting. 300 is a good starting point.
        - learning_rate=0.05: How much each tree adjusts the prediction.
          Smaller = better generalization but slower training.
        - num_leaves=63: Max complexity of each tree. Higher = more
          expressive but risk of overfitting.
        - min_data_in_leaf=10: At least 10 examples per leaf.
          Prevents overfitting on tiny groups of data.
        - lambdarank_truncation_level=10: Only optimize NDCG for the
          top 10 positions. We don't care about positions 11+
          because users rarely scroll that far.
        """
        print("Training LambdaMART model...")
        print(f"  Samples: {len(labels):,}")
        print(f"  Features: {features.shape[1]}")
        print(f"  Query groups: {len(query_groups)}")

        # Create LightGBM dataset
        train_data = lgb.Dataset(
            features,
            label=labels,
            group=query_groups,
            feature_name=LTRFeatures.feature_names(),
        )

        # LambdaMART parameters
        params = {
            "objective": "lambdarank",        # LambdaMART ranking objective
            "metric": "ndcg",                  # Optimize NDCG
            "ndcg_eval_at": [5, 10],           # Report NDCG@5 and NDCG@10
            "n_estimators": 300,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 10,
            "lambdarank_truncation_level": 10,
            "verbose": 1,
        }

        # Train!
        self.model = lgb.train(
            params,
            train_data,
            valid_sets=[train_data],
            callbacks=[lgb.log_evaluation(50)],
        )

        # Print feature importance
        importance = self.model.feature_importance(importance_type="gain")
        feature_names = LTRFeatures.feature_names()
        print("\n  Feature Importance:")
        for name, imp in sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True):
            print(f"    {name:30s} {imp:10.1f}")

        # Save model
        if self.model_path:
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            self.model.save_model(str(self.model_path))
            print(f"\n  ✓ Model saved to: {self.model_path}")

    def load(self):
        """Load a trained model from disk."""
        if not self.model_path or not self.model_path.exists():
            raise FileNotFoundError(f"LTR model not found at {self.model_path}")

        self.model = lgb.Booster(model_file=str(self.model_path))
        print(f"✓ LTR model loaded from {self.model_path}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Score documents using the trained model.

        Args:
            features: shape (n_documents, n_features) — feature matrix

        Returns:
            scores: shape (n_documents,) — relevance scores (higher = more relevant)
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        return self.model.predict(features)
```

> **📚 Resources for Phase 4:**
> - [LightGBM LambdaRank Documentation](https://lightgbm.readthedocs.io/en/stable/Parameters.html#learning-to-rank) — Official docs
> - [Learning to Rank Tutorial (Microsoft)](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) — The foundational paper
> - [Click Models for Web Search](https://clickmodels.weebly.com/) — Understanding implicit feedback

> **✅ Phase 4 Checkpoint:**
> - [ ] Click logging endpoint saving events to PostgreSQL
> - [ ] Feature engineering pipeline computing all LTR features
> - [ ] LambdaMART model trained on click data
> - [ ] Measured improvement: LTR NDCG@10 vs re-ranking-only NDCG@10
> - [ ] Feature importance analysis completed

---

## Phase 5: Polish, Benchmark & Deploy (Week 8)

### 5.1 Load Testing

```python
# benchmarks/locustfile.py
"""
Load test for the search API using Locust.

RUN:
  locust -f benchmarks/locustfile.py --host http://localhost:8000

Then open browser at http://localhost:8089 to start the test.

WHAT TO MEASURE:
- Requests per second (QPS) at different concurrency levels
- p50, p95, p99 latency
- Error rate under load
- At what QPS does latency degrade?

TARGETS:
- p99 latency < 100ms at 100 QPS
- p99 latency < 200ms at 500 QPS
- 0% error rate at 500 QPS
"""
from locust import HttpUser, task, between
import random

# Sample queries for realistic load testing
SAMPLE_QUERIES = [
    "python tutorial for beginners",
    "how to sort a list",
    "machine learning introduction",
    "web development frameworks",
    "database optimization techniques",
    "what is cloud computing",
    "react vs angular comparison",
    "data structures and algorithms",
    "REST API design best practices",
    "docker container tutorial",
    "kubernetes deployment guide",
    "microservices architecture",
    "SQL query optimization",
    "git branching strategies",
    "how does encryption work",
]


class SearchUser(HttpUser):
    """Simulates a user performing search queries."""

    # Wait 1-3 seconds between requests (realistic user behavior)
    wait_time = between(1, 3)

    @task(10)  # Weight: 10x more likely to search than give feedback
    def search(self):
        """Perform a random search query."""
        query = random.choice(SAMPLE_QUERIES)
        self.client.get(f"/search?q={query}&page=1&size=10")

    @task(1)
    def search_page_2(self):
        """Occasionally request page 2 (tests pagination)."""
        query = random.choice(SAMPLE_QUERIES)
        self.client.get(f"/search?q={query}&page=2&size=10")

    @task(1)
    def health_check(self):
        """Check health endpoint."""
        self.client.get("/health")
```

### 5.2 Results Benchmarking Table

After completing all phases, create this table for your README:

```markdown
## Benchmark Results

### Search Quality (MS MARCO Dev Set, 6,980 queries)

| Configuration | NDCG@10 | MRR | Recall@100 | Improvement |
|---------------|:-------:|:---:|:----------:|:-----------:|
| BM25 only     |  0.228  | 0.195 |   0.658  | baseline    |
| + Vector (hybrid) | 0.312 | 0.264 | 0.782   | +36.8%      |
| + Fine-tuned embeddings | 0.341 | 0.289 | 0.801 | +49.6%  |
| + Cross-encoder re-rank | 0.398 | 0.341 | 0.801 | +74.6%  |
| + LTR (LambdaMART) | 0.421 | 0.362 | 0.801  | +84.6%     |

### System Performance (Single machine, 4 CPU cores, 16GB RAM)

| Metric | Value |
|--------|-------|
| p50 latency | 42ms |
| p95 latency | 78ms |
| p99 latency | 95ms |
| Max throughput | 520 QPS |
| Cache hit rate | 67% |
| Index time (100K docs) | 4m 23s |
```

> ⚠️ **Note:** The exact numbers above are illustrative targets. Your actual results will vary based on dataset, hardware, and implementation quality. The important thing is showing **relative improvement** across configurations.

---

## Resources & References

### Core Concepts

| Topic | Resource | Type |
|-------|----------|------|
| BM25 Algorithm | [Practical BM25 (Elastic Blog)](https://www.elastic.co/blog/practical-bm25-part-2-the-bm25-algorithm-and-its-variables) | Article |
| Sentence Embeddings | [SBERT.net Documentation](https://www.sbert.net/) | Docs |
| FAISS Guide | [FAISS Wiki](https://github.com/facebookresearch/faiss/wiki) | Docs |
| Cross-Encoders | [SBERT Cross-Encoders](https://www.sbert.net/examples/applications/cross-encoder/README.html) | Tutorial |
| Learning to Rank | [LTR Overview (Microsoft)](https://www.microsoft.com/en-us/research/publication/from-ranknet-to-lambdarank-to-lambdamart-an-overview/) | Paper |
| NDCG Explained | [Evaluation Measures in IR](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf) | Slides |
| RRF Paper | [Reciprocal Rank Fusion (Cormack et al.)](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) | Paper |
| Click Models | [Click Models for Web Search](https://clickmodels.weebly.com/) | Book |
| MS MARCO Dataset | [MS MARCO GitHub](https://github.com/microsoft/MSMARCO-Passage-Ranking) | Dataset |

### Python & Engineering

| Topic | Resource | Type |
|-------|----------|------|
| FastAPI | [Official Tutorial](https://fastapi.tiangolo.com/tutorial/) | Docs |
| SQLAlchemy 2.0 | [Unified Tutorial](https://docs.sqlalchemy.org/en/20/tutorial/) | Docs |
| pytest | [Official Docs](https://docs.pytest.org/en/stable/) | Docs |
| Docker Compose | [Getting Started](https://docs.docker.com/compose/gettingstarted/) | Tutorial |
| LightGBM | [Parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html) | Docs |

### Video Tutorials

| Topic | Resource | Duration |
|-------|----------|----------|
| How Search Engines Work | [CS50 — Search Engines](https://www.youtube.com/watch?v=qFZBZPWdVRk) | 1hr |
| Vector Search Explained | [Pinecone — What is Vector Search?](https://www.youtube.com/watch?v=klTvEwg3oJ4) | 15min |
| FAISS Tutorial | [James Briggs — FAISS Course](https://www.youtube.com/watch?v=sKyvsdEv6rk) | 30min |
| Sentence Transformers | [SBERT Training Examples](https://www.youtube.com/watch?v=p3HnEfMtKQE) | 20min |

---

## Best Practices & Tips

### 1. Git Commit Strategy

```
COMMIT EARLY, COMMIT OFTEN.

Bad: One giant commit "added everything"
Good: Small, focused commits that tell a story:

  feat: implement BM25 search with Tantivy
  feat: add Redis caching for search results
  test: add BM25 search unit tests
  feat: implement FAISS vector search
  feat: add RRF fusion for hybrid retrieval
  perf: batch embedding generation (10x speedup)
  feat: add cross-encoder re-ranking
  docs: add architecture diagram to README

Use conventional commits (feat/fix/docs/test/perf/refactor).
Recruiters browse your git log — make it clean.
```

### 2. Code Quality

```python
# BAD: No type hints, no docstrings, magic numbers
def search(q, k):
    r = engine.search(q, k)
    return [{"id": x.id, "s": x.score} for x in r if x.score > 0.5]

# GOOD: Type hints, docstring, named constants
MINIMUM_RELEVANCE_SCORE = 0.5

def search(query: str, top_k: int = 10) -> List[SearchResult]:
    """
    Search for documents matching the query.

    Args:
        query: The search query string
        top_k: Maximum number of results to return

    Returns:
        List of search results sorted by relevance score
    """
    results = engine.search(query, top_k=top_k)
    return [
        SearchResult(doc_id=r.doc_id, score=r.score)
        for r in results
        if r.score > MINIMUM_RELEVANCE_SCORE
    ]
```

### 3. Profiling Before Optimizing

```python
# DON'T guess where the bottleneck is. MEASURE it.

import time

def search_pipeline(query: str):
    t0 = time.perf_counter()

    # Query understanding
    corrected = spell_check(query)
    t1 = time.perf_counter()

    # BM25 retrieval
    bm25_results = bm25.search(corrected)
    t2 = time.perf_counter()

    # Vector retrieval
    vector_results = vector.search(corrected)
    t3 = time.perf_counter()

    # Fusion
    fused = fusion.merge(bm25_results, vector_results)
    t4 = time.perf_counter()

    # Re-ranking
    reranked = reranker.rerank(corrected, fused)
    t5 = time.perf_counter()

    print(f"Spell check:  {(t1-t0)*1000:.1f}ms")
    print(f"BM25:         {(t2-t1)*1000:.1f}ms")
    print(f"Vector:       {(t3-t2)*1000:.1f}ms")
    print(f"Fusion:       {(t4-t3)*1000:.1f}ms")
    print(f"Re-ranking:   {(t5-t4)*1000:.1f}ms")
    print(f"Total:        {(t5-t0)*1000:.1f}ms")

# Now you know EXACTLY where time is spent.
# Optimize the slowest step first.
```

### 4. README That Impresses

Your README is the first thing recruiters see. Include:

```markdown
# Hybrid Neural Search Engine

> One-line description that hooks the reader

## Architecture
[Include the ASCII/mermaid architecture diagram]

## Benchmark Results
[The comparison table from Phase 5]

## Quick Start
docker-compose up -d
pip install -e .
python scripts/index_documents.py --max-docs 10000
uvicorn src.api.main:app --reload

## Tech Stack
[Table of technologies with justifications]

## Design Decisions
[2-3 key decisions — why RRF over linear combination? why HNSW over IVF?]

## Future Improvements
[Shows you think ahead — "add query autocomplete", "distributed with Qdrant"]
```

---

## Common Mistakes & How to Avoid Them

### Mistake 1: Not Establishing a Baseline

```
❌ "I added vector search and it seems better"
✅ "Vector search improved NDCG@10 from 0.228 to 0.312 (+36.8%)"

RULE: Always measure BEFORE and AFTER every change.
Record everything in a benchmarks spreadsheet.
```

### Mistake 2: Premature Optimization

```
❌ Spending 2 days optimizing FAISS parameters before verifying the pipeline works
✅ Get the pipeline working end-to-end FIRST, then optimize the bottleneck

Phase 1: Make it WORK (correctness)
Phase 2: Make it RIGHT (clean code)
Phase 3: Make it FAST (optimization)

Never do Phase 3 before Phase 1.
```

### Mistake 3: Ignoring Edge Cases in Search

```
❌ Assuming all queries are well-formed English
✅ Handle these cases:
   - Empty query → return error message
   - Very long query (500+ chars) → truncate
   - Special characters ("C++ tutorial") → escape properly
   - Numbers only ("42") → handle as keyword search
   - Non-English text → fall back to BM25 (embeddings may not work)
```

### Mistake 4: Training on Test Data

```
❌ Fine-tuning on MS MARCO training set, then evaluating on the SAME training set
✅ Always evaluate on HELD-OUT data:
   - Train on queries.train.tsv + qrels.train.tsv
   - Evaluate on queries.dev.small.tsv + qrels.dev.small.tsv
   - NEVER let the model see evaluation data during training
```

### Mistake 5: Not Handling Model Loading Failures

```python
# ❌ BAD: App crashes if model file is missing
model = lgb.Booster(model_file="models/ltr/model.txt")

# ✅ GOOD: Graceful degradation
try:
    model = lgb.Booster(model_file="models/ltr/model.txt")
    print("✓ LTR model loaded")
except FileNotFoundError:
    model = None
    print("⚠ LTR model not found — using re-ranking scores only")

# In the search pipeline:
if model is not None:
    scores = model.predict(features)
else:
    scores = [r.cross_encoder_score for r in candidates]  # Fallback
```

### Mistake 6: Forgetting to Normalize Vectors

```python
# ❌ BAD: Raw embeddings (not normalized) used with inner product
embeddings = model.encode(texts)
# Inner product on non-normalized vectors ≠ cosine similarity!

# ✅ GOOD: Normalize embeddings before adding to FAISS
embeddings = model.encode(texts, normalize_embeddings=True)
# Now inner product = cosine similarity (mathematically equivalent)
```

---

## Final Checklist

Before considering the project "done":

- [ ] **BM25 search** working and tested
- [ ] **Vector search** with fine-tuned embeddings
- [ ] **Hybrid retrieval** (RRF fusion) showing improvement over either alone
- [ ] **Cross-encoder re-ranking** showing improvement over hybrid alone
- [ ] **LTR model** trained on click data showing improvement over re-ranking alone
- [ ] **Comparison table** in README with NDCG@10, MRR, Recall@100 for each stage
- [ ] **Load test results** (p50, p95, p99 latency, max QPS)
- [ ] **Unit tests** passing (pytest)
- [ ] **Docker Compose** starts everything with one command
- [ ] **Clean git history** with conventional commits
- [ ] **README** with architecture diagram, benchmarks, and setup instructions
- [ ] **Code quality** — type hints, docstrings, no magic numbers
