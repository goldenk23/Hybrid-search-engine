# Hybrid Search Engine - Complete Usage Guide

This document contains complete, working code examples for all major components of the Hybrid Search Engine project.

---

## Table of Contents

1. [API Usage](#api-usage)
2. [Python Module Usage](#python-module-usage)
3. [CLI Scripts](#cli-scripts)
4. [Configuration](#configuration)
5. [Development Workflow](#development-workflow)
6. [Database Operations](#database-operations)
7. [Caching](#caching)
8. [Testing](#testing)

---

## API Usage

### Starting the API Server

```bash
# Basic startup
python -m uvicorn src.api.main:app --reload

# With custom host and port
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload

# Production mode (no reload)
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

### API Documentation

Once the server is running:
- **Interactive Documentation (Swagger UI)**: http://127.0.0.1:8000/docs
- **Alternative Documentation (ReDoc)**: http://127.0.0.1:8000/redoc

### API Endpoints

#### 1. Health Check

```bash
# Using curl
curl http://127.0.0.1:8000/health

# Response:
# {
#   "status": "ok",
#   "service": "hybrid-search-engine"
# }
```

#### 2. Search Endpoint

```bash
# Basic search with default parameters
curl "http://127.0.0.1:8000/search?q=python"

# Search with custom number of results
curl "http://127.0.0.1:8000/search?q=python&top_k=20"

# Search with multiple words
curl "http://127.0.0.1:8000/search?q=machine%20learning&top_k=10"

# URL-encoded query (spaces as %20)
curl "http://127.0.0.1:8000/search?q=data%20science%20tutorial&top_k=15"
```

#### Response Format

```json
{
  "query": "python",
  "total": 3,
  "latency_ms": 45,
  "results": [
    {
      "id": "doc_001",
      "title": "Python Programming Basics",
      "body": "Learn Python programming from scratch using variables loops and functions.",
      "category": "programming",
      "score": 12.5,
      "snippet": "Learn Python programming from scratch using variables loops and..."
    },
    {
      "id": "doc_002",
      "title": "Python Data Science Guide",
      "body": "Use Python for data analysis with pandas numpy and matplotlib.",
      "category": "data",
      "score": 10.3,
      "snippet": "Use Python for data analysis with pandas numpy and matplotlib."
    }
  ]
}
```

#### Using Python Requests Library

```python
import requests

# Basic search
response = requests.get(
    "http://127.0.0.1:8000/search",
    params={
        "q": "python programming",
        "top_k": 10
    }
)

results = response.json()
print(f"Query: {results['query']}")
print(f"Total results: {results['total']}")
print(f"Latency: {results['latency_ms']}ms")

for result in results['results']:
    print(f"  - {result['title']} (score: {result['score']})")

# Health check
health = requests.get("http://127.0.0.1:8000/health")
print(health.json())
```

---

## Python Module Usage

### BM25 Search Module

```python
from pathlib import Path
from src.search.bm25 import BM25Search

# Initialize BM25 search with default index location
bm25 = BM25Search()

# Initialize with custom index path
custom_index_path = Path("my_custom_index")
bm25 = BM25Search(index_path=custom_index_path)

# Add documents to index
documents = [
    {
        "id": "1",
        "title": "Python Programming Tutorial",
        "body": "Learn Python programming from scratch using variables loops and functions.",
        "category": "programming",
    },
    {
        "id": "2",
        "title": "JavaScript Web Development",
        "body": "Build modern web applications using JavaScript React and Node.",
        "category": "web",
    },
    {
        "id": "3",
        "title": "Python Data Science Guide",
        "body": "Use Python for data analysis with pandas numpy and matplotlib.",
        "category": "data",
    },
]

# Add all documents
bm25.add_documents(documents)

# Add documents with custom batch size
bm25.add_documents(documents, batch_size=5000)

# Search the index
results = bm25.search(query="python", top_k=10)

# Print results
for result in results:
    print(f"ID: {result['id']}")
    print(f"Title: {result['title']}")
    print(f"Body: {result['body']}")
    print(f"Category: {result['category']}")
    print(f"Score: {result['score']}")
    print("---")
```

### Text Preprocessing Module

```python
from src.indexing.preprocessing import clean_text, is_valid_document, generate_snippet

# Clean raw text
raw_text = "<p>Hello & goodbye</p>    Multiple   spaces"
cleaned = clean_text(raw_text)
print(cleaned)  # Output: "Hello & goodbye Multiple spaces"

# Handle None values
result = clean_text(None)
print(result)  # Output: ""

# Validate documents before indexing
title = "Machine Learning Basics"
body = "This is a comprehensive guide to machine learning concepts and algorithms."

is_valid = is_valid_document(title, body, min_body_length=20)
print(is_valid)  # Output: True

# Check invalid document
invalid_title = ""
is_valid = is_valid_document(invalid_title, body)
print(is_valid)  # Output: False

# Generate snippets for display
body_text = "Python is a versatile programming language. It's used for web development, data science, machine learning, and more. Python is known for its simplicity and readability."
query = "Python"
snippet = generate_snippet(body_text, query, snippet_length=200)
print(snippet)  
# Output: "Python is a versatile programming language. It's used for web development, data science, machine learning, and more..."
```

### Indexing Pipeline Module

```python
from pathlib import Path
from src.indexing.pipeline import load_msmarco_passages, run_indexing_pipeline

# Load passages from MS MARCO collection
collection_path = Path("data/msmarco/collection.tsv")

# Load first 1000 passages
documents = load_msmarco_passages(collection_path, max_documents=1000)

# Process loaded documents
for doc in documents:
    print(f"ID: {doc['id']}, Title: {doc['title']}")

# Run full indexing pipeline with default settings
# (uses data/msmarco/collection.tsv by default)
bm25 = run_indexing_pipeline()

# Run pipeline with custom collection and document limit
custom_collection = Path("data/msmarco/custom_collection.tsv")
bm25 = run_indexing_pipeline(
    collection_path=custom_collection,
    max_documents=50000
)

# Test the index with a search
results = bm25.search("python programming", top_k=5)
print(f"Found {len(results)} results")
```

### Configuration Module

```python
from src.config import (
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    INDEX_DIR,
    POSTGRES_URL,
    REDIS_URL,
    BM25_TOP_K,
    VECTOR_TOP_K,
    RERANK_TOP_K,
    RESULTS_PER_PAGE,
    EMBEDDING_MODEL_NAME,
    CROSS_ENCODER_MODEL_NAME,
    CACHE_TTL_SECONDS,
    API_HOST,
    API_PORT,
)

# Access configuration values
print(f"Project Root: {PROJECT_ROOT}")
print(f"Data Directory: {DATA_DIR}")
print(f"Index Directory: {INDEX_DIR}")
print(f"PostgreSQL URL: {POSTGRES_URL}")
print(f"Redis URL: {REDIS_URL}")
print(f"BM25 Top K: {BM25_TOP_K}")
print(f"Results Per Page: {RESULTS_PER_PAGE}")
print(f"Embedding Model: {EMBEDDING_MODEL_NAME}")
print(f"Cache TTL: {CACHE_TTL_SECONDS} seconds")
print(f"API Host: {API_HOST}")
print(f"API Port: {API_PORT}")

# All configuration values are read from environment variables
# If not set, they use sensible defaults (shown above)
```

---

## CLI Scripts

### Index Documents Script

```bash
# Index using default collection (data/msmarco/collection.tsv)
python scripts/index_documents.py

# Index with custom collection path
python scripts/index_documents.py --collection data/msmarco/custom_collection.tsv

# Index first 1000 documents
python scripts/index_documents.py --max-docs 1000

# Index custom collection with document limit
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 500000

# View script help
python scripts/index_documents.py --help
```

Expected output:
```
============================================================
starting indexing pipeline
============================================================

[1/2] loading and processing passages...
Processing passages: 100%|████████| 1000/1000 [00:05<00:00, 198.76it/s]
Loaded 1,000 valid documents.

[2/2] building BM25 index...
  Indexed 1,000 documents...

Indexing complete.
BM25 index path: /path/to/data/indexes/bm25

Smoke test:
1. Where is Patna? | score=8.45
2. Patna India History | score=7.23
3. Patna Tourism Guide | score=6.89
```

### Initialize Database Script

```bash
# Initialize database (create tables)
python scripts/init_db.py
```

Output:
```
Database tables created successfully.
```

### Download MS MARCO Script

```bash
# Download MS MARCO dataset
python scripts/download_msmarco.py
```

---

## Configuration

### Environment Variables

Create a `.env` file in the project root to override defaults:

```bash
# PostgreSQL Configuration
POSTGRES_URL=postgresql://search_user:search_password@localhost:5432/search_engine

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# Search Parameters
BM25_TOP_K=100
VECTOR_TOP_K=100
RERANK_TOP_K=50
RESULTS_PER_PAGE=10

# Model Configuration
EMBEDDING_MODEL=all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Cache Configuration
CACHE_TTL=300

# API Server Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### Using Configuration in Code

```python
import os
from dotenv import load_dotenv
from src.config import INDEX_DIR, CACHE_TTL_SECONDS

# Configuration is automatically loaded from .env
# Access configuration values from the config module
print(f"Using index directory: {INDEX_DIR}")
print(f"Cache TTL: {CACHE_TTL_SECONDS} seconds")

# Or override in code
os.environ["CACHE_TTL"] = "600"
# Note: You need to reload the config module for changes to take effect
```

---

## Development Workflow

### Project Structure Setup

```bash
# Navigate to project
cd hybrid-search-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate virtual environment (Linux/Mac)
source .venv/bin/activate

# Install project in development mode
pip install -e .

# Verify installation
pip list
```

### Starting Services

```bash
# Start PostgreSQL and Redis containers
docker compose up -d

# Wait for services to be ready
Start-Sleep -Seconds 5

# Initialize database
python scripts/init_db.py

# Check container status
docker compose ps
```

### Running the Application

```bash
# Terminal 1: Start the API server
python -m uvicorn src.api.main:app --reload

# Terminal 2: View PostgreSQL logs
docker compose logs -f postgres

# Terminal 3: View Redis logs
docker compose logs -f redis

# Access API documentation
# Open browser to: http://127.0.0.1:8000/docs
```

### Development with Hot Reload

```bash
# The --reload flag watches for file changes and restarts the server
python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000

# Changes to src/ files will automatically reload the server
```

---

## Database Operations

### PostgreSQL Connection

```python
from src.database.postgres import SessionLocal, get_session
from src.database.models import Base

# Synchronous session
session = SessionLocal()

try:
    # Perform database operations
    pass
finally:
    session.close()

# Using get_session helper
session = get_session()
try:
    # Perform database operations
    pass
finally:
    session.close()
```

### Async PostgreSQL Operations

```python
from src.database.postgres import AsyncSessionLocal, get_async_session
from sqlalchemy.ext.asyncio import AsyncSession

# For use in FastAPI dependency injection
async def my_route(session: AsyncSession = Depends(get_async_session)):
    # Perform async database operations
    pass

# Direct async session
from src.database.postgres import AsyncSessionLocal

async def my_async_function():
    async with AsyncSessionLocal() as session:
        # Perform async database operations
        pass
```

### Database Initialization

```python
from src.database.postgres import init_db

# Create all tables
init_db()
```

### Docker Database Access

```bash
# Access PostgreSQL directly
docker exec -it search-postgres psql -U search_user -d search_engine

# Common PostgreSQL commands
# \dt - List all tables
# \d table_name - Describe table
# SELECT * FROM table_name; - Query table
# \q - Quit

# Access Redis CLI
docker exec -it search-redis redis-cli

# Common Redis commands
# PING - Test connection
# KEYS * - List all keys
# GET key_name - Get value
# DEL key_name - Delete key
```

### Reset Database

```bash
# WARNING: This deletes all data

# Stop and remove containers with volumes
docker compose down -v

# Start fresh services
docker compose up -d

# Wait for services to be ready
Start-Sleep -Seconds 5

# Initialize database
python scripts/init_db.py
```

---

## Caching

### Redis Cache Usage

```python
import asyncio
from src.database.redis_client import cache

async def example_caching():
    # Initialize cache connection
    await cache.connect()
    
    try:
        # Try to get cached results
        cached = await cache.get_cached_results(
            query="python programming",
            page=1,
            size=10
        )
        
        if cached:
            print("Cache hit!")
            results = cached
        else:
            print("Cache miss, computing results...")
            results = {"query": "python programming", "results": [...]}
            
            # Cache the results with TTL
            await cache.cache_results(
                query="python programming",
                results=results,
                page=1,
                size=10
            )
        
        # Invalidate all cached results if needed
        await cache.invalidate_all()
        
    finally:
        # Close connection
        await cache.close()

# Run in asyncio event loop
asyncio.run(example_caching())
```

### Cache Configuration

```python
from src.config import CACHE_TTL_SECONDS

# Cache TTL is configurable via environment variable
# Default: 300 seconds (5 minutes)
print(f"Cache TTL: {CACHE_TTL_SECONDS} seconds")

# To change, set in .env:
# CACHE_TTL=600
```

---

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_bm25.py -v

# Run specific test
pytest tests/test_bm25.py::test_search_returns_relevant_results -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run with verbose output
pytest tests/ -vv

# Run with output capture disabled (see print statements)
pytest tests/ -s
```

### Test File Structure

```python
# tests/test_bm25.py
import pytest
from src.search.bm25 import BM25Search

@pytest.fixture
def bm25_with_test_data(tmp_path):
    """Fixture that provides a BM25 instance with test data."""
    engine = BM25Search(index_path=tmp_path / "bm25_test_index")
    
    documents = [
        {
            "id": "1",
            "title": "Python Programming Tutorial",
            "body": "Learn Python programming from scratch.",
            "category": "programming",
        },
        {
            "id": "2",
            "title": "JavaScript Web Development",
            "body": "Build modern web applications using JavaScript.",
            "category": "web",
        },
    ]
    engine.add_documents(documents)
    return engine

def test_search_returns_relevant_results(bm25_with_test_data):
    """Test that search returns relevant documents."""
    results = bm25_with_test_data.search("python", top_k=5)
    
    assert len(results) == 1
    assert results[0]["id"] == "1"

def test_search_results_are_sorted_by_score(bm25_with_test_data):
    """Test that results are sorted by score in descending order."""
    results = bm25_with_test_data.search("programming", top_k=5)
    scores = [result["score"] for result in results]
    
    assert scores == sorted(scores, reverse=True)

def test_search_respects_top_k(bm25_with_test_data):
    """Test that search respects the top_k parameter."""
    results = bm25_with_test_data.search("programming", top_k=1)
    
    assert len(results) <= 1
```

### Running Preprocessing Tests

```bash
# Test text cleaning
pytest tests/test_preprocessing.py::test_clean_text_removes_html_and_decodes_entities -v

# Test document validation
pytest tests/test_preprocessing.py::test_is_valid_document_rejects_empty_title -v

# Test snippet generation
pytest tests/test_preprocessing.py::test_generate_snippet_contains_query_term -v
```

### Running API Tests

```bash
# Test API endpoints
pytest tests/test_api.py -v

# Test health check endpoint
pytest tests/test_api.py::test_health_check -v

# Test search endpoint validation
pytest tests/test_api.py::test_search_rejects_too_short_query -v

# Test search response format
pytest tests/test_api.py::test_search_response_has_expected_shape -v
```

---

## Complete Workflow Example

Here's a complete example showing how all components work together:

```python
#!/usr/bin/env python
"""
Complete workflow example for the Hybrid Search Engine.
"""

from pathlib import Path
from src.search.bm25 import BM25Search
from src.indexing.preprocessing import clean_text, is_valid_document, generate_snippet
from src.config import INDEX_DIR

def main():
    print("=" * 60)
    print("Hybrid Search Engine - Complete Workflow Example")
    print("=" * 60)
    
    # Step 1: Initialize BM25 Search
    print("\n[1/4] Initializing BM25 Search...")
    bm25 = BM25Search(index_path=INDEX_DIR / "bm25")
    print("✓ BM25 initialized")
    
    # Step 2: Prepare and validate documents
    print("\n[2/4] Preparing documents...")
    raw_documents = [
        {
            "id": "1",
            "title": "Python Programming",
            "body": "<p>Learn Python programming basics</p>",
            "category": "programming",
        },
        {
            "id": "2",
            "title": "Web Development",
            "body": "Build modern web applications with JavaScript & React",
            "category": "web",
        },
    ]
    
    documents = []
    for doc in raw_documents:
        cleaned_body = clean_text(doc["body"])
        
        if is_valid_document(doc["title"], cleaned_body):
            documents.append({
                "id": doc["id"],
                "title": doc["title"],
                "body": cleaned_body,
                "category": doc["category"],
            })
    
    print(f"✓ Prepared {len(documents)} documents")
    
    # Step 3: Index documents
    print("\n[3/4] Indexing documents...")
    bm25.add_documents(documents)
    print("✓ Documents indexed")
    
    # Step 4: Search and display results
    print("\n[4/4] Searching...")
    query = "Python programming"
    results = bm25.search(query, top_k=10)
    
    print(f"\nSearch Results for '{query}':")
    print("-" * 60)
    
    for i, result in enumerate(results, 1):
        snippet = generate_snippet(result["body"], query)
        print(f"\n{i}. {result['title']}")
        print(f"   ID: {result['id']}")
        print(f"   Category: {result['category']}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Snippet: {snippet}")
    
    print("\n" + "=" * 60)
    print("Workflow Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
```

Run the example:
```bash
python example_workflow.py
```

---

## Troubleshooting

### Common Issues and Solutions

#### 1. BM25 Index Not Found

```python
# Error: FileNotFoundError: Index not found at path

# Solution: Initialize and index documents first
from src.indexing.pipeline import run_indexing_pipeline

bm25 = run_indexing_pipeline(max_documents=1000)
```

#### 2. PostgreSQL Connection Failed

```bash
# Check if Docker containers are running
docker compose ps

# Restart services if needed
docker compose down
docker compose up -d
docker compose ps

# Verify connection
docker exec -it search-postgres psql -U search_user -d search_engine -c "\dt"
```

#### 3. Redis Connection Failed

```bash
# Check Redis is running
docker compose ps

# Test Redis connection
docker exec -it search-redis redis-cli ping
# Should respond with PONG
```

#### 4. API Server Won't Start

```bash
# Check if port 8000 is already in use
# Option 1: Kill process on port 8000
# On Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Option 2: Use different port
python -m uvicorn src.api.main:app --port 8001
```

#### 5. Tests Failing

```bash
# Run tests with verbose output
pytest tests/ -vv -s

# Check for fixture issues
pytest --fixtures

# Run single test for debugging
pytest tests/test_bm25.py::test_search_returns_relevant_results -vv -s
```

---

## Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Tantivy Documentation](https://docs.rs/tantivy/)
- [SQLAlchemy Documentation](https://docs.sqlalchemy.org/)
- [Redis Documentation](https://redis.io/documentation)
- [Pytest Documentation](https://docs.pytest.org/)

---

**Last Updated**: May 12, 2026
**Version**: 0.1.0
