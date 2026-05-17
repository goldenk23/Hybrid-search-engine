"""
Centralized configuration using environment variables.

Hardcoding database passwords, API keys, or file paths makes deployment
difficult. Environment variables let us change configuration without
changing application code.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
INDEX_DIR = DATA_DIR / "indexes"
VECTOR_INDEX_PATH = Path(os.getenv("VECTOR_INDEX_PATH", INDEX_DIR / "vector.faiss"))
DOCSTORE_PATH = Path(os.getenv("DOCSTORE_PATH", DATA_DIR / "docstore.sqlite"))

DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

POSTGRES_URL = os.getenv(
    "POSTGRES_URL",
    "postgresql://search_user:search_password@localhost:5432/search_engine",
)
ASYNC_POSTGRES_URL = POSTGRES_URL.replace("postgresql://", "postgresql+asyncpg://")

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

BM25_TOP_K = int(os.getenv("BM25_TOP_K", "100"))
VECTOR_TOP_K = int(os.getenv("VECTOR_TOP_K", "100"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "50"))
RESULTS_PER_PAGE = int(os.getenv("RESULTS_PER_PAGE", "10"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
LTR_MODEL_PATH = MODELS_DIR / "ltr" / "lambdamart.txt"
INTENT_MODEL_PATH = MODELS_DIR / "intent" / "intent_classifier.bin"

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
