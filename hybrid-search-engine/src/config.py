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
BM25_INDEX_PATH = Path(os.getenv("BM25_INDEX_PATH", INDEX_DIR / "bm25_compact"))
VECTOR_INDEX_PATH = Path(os.getenv("VECTOR_INDEX_PATH", INDEX_DIR / "vector.faiss"))
DOCSTORE_PATH = Path(os.getenv("DOCSTORE_PATH", DATA_DIR / "docstore.sqlite"))

# Importing config should be a side-effect-free read of values.
# Creating directories here fires on every import — including during tests —
# and silently creates real data/ folders even when the test uses tmp_path.
# Call ensure_build_directories() explicitly from scripts that need it.


def ensure_build_directories() -> None:
    """Create data/, models/, and indexes/ directories if they don't exist.

    Call this at the top of indexing and download scripts.
    Never call it from the API or from tests.
    """
    for path in (DATA_DIR, MODELS_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)

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
# Pinned HuggingFace commit hash for the embedding model.  Empty string means
# "latest" — set this env var to lock a specific revision for reproducibility.
EMBEDDING_MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION", "")

CROSS_ENCODER_MODEL_NAME = os.getenv(
    "CROSS_ENCODER_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)
# Same as above for the cross-encoder reranker.
CROSS_ENCODER_MODEL_REVISION = os.getenv("CROSS_ENCODER_MODEL_REVISION", "")

# Bump this integer whenever you change text cleaning or validation logic in
# src/indexing/preprocessing.py.  A changed version makes the manifest
# detect that the index was built with different preprocessing and refuse
# to resume against mismatched artifacts.
PREPROCESSING_VERSION = "1"
LTR_MODEL_PATH = MODELS_DIR / "ltr" / "lambdamart.txt"
INTENT_MODEL_PATH = MODELS_DIR / "intent" / "intent_classifier.bin"

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL", "300"))

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
