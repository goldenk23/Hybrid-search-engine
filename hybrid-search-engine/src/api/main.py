"""
FastAPI application factory.

Key design decisions:
- create_app() builds the app; tests call it with a fake loader so no
  real indexes are needed.
- lifespan loads all heavy services once at startup — no lazy loading
  per request.
- /health/live  — is the process alive? (liveness probe)
- /health/ready — are search services loaded? (readiness probe)
- /metrics      — Prometheus counters and histograms (never labelled by
                  query text, which would leak user data and explode cardinality)

Run with:
    python -m uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from threading import BoundedSemaphore

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from src.api.models import HealthResponse
from src.api.routes.search import router as search_router
from src.config import (
    BM25_INDEX_PATH,
    CORS_ORIGINS,
    DOCSTORE_PATH,
    ENABLE_RERANKER,
    INDEX_DIR,
    VECTOR_INDEX_PATH,
)
from src.database.docstore import SQLiteDocstore
from src.indexing.artifact_state import load_json_required, sha256_path
from src.query.spell_check import SpellCorrector
from src.search.bm25 import BM25Search
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.hybrid_search import HybridSearchEngine
from src.search.vector import VectorSearch

# ------------------------------------------------------------------ Prometheus
# prometheus_client is optional — if it's not installed the metrics endpoint
# is simply disabled rather than crashing the whole app.
try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Histogram,
        generate_latest,
    )

    _REQUEST_COUNT = Counter(
        "http_requests_total",
        "Total HTTP requests",
        ["method", "route", "status"],
    )
    _REQUEST_LATENCY = Histogram(
        "http_request_duration_seconds",
        "HTTP request latency",
        ["method", "route"],
    )
    _PROMETHEUS_AVAILABLE = True
except ImportError:
    _PROMETHEUS_AVAILABLE = False


# ------------------------------------------------------------------ service container

@dataclass
class SearchServices:
    """All heavy objects loaded once at startup and shared across requests."""
    spell: SpellCorrector
    bm25: BM25Search
    hybrid: HybridSearchEngine
    reranker: CrossEncoderReranker | None
    # Semaphore: only one rerank runs at a time.  Extra requests get 429.
    rerank_slots: BoundedSemaphore


# ------------------------------------------------------------------ manifest validation

def verify_complete_manifest_and_hashes(manifest_path) -> dict:
    """Load the manifest and confirm it is complete with matching fingerprints.

    Raises RuntimeError if:
    - the file is missing or corrupt
    - status != 'complete'
    - any artifact fingerprint on disk differs from the stored value
    """
    manifest = load_json_required(manifest_path)

    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Artifact manifest is not complete (status={manifest.get('status')!r}). "
            "Finish indexing before starting the API."
        )

    stored = manifest.get("artifact_sha256", {})
    to_check = {
        "docstore": DOCSTORE_PATH,
        **({"bm25": BM25_INDEX_PATH} if BM25_INDEX_PATH.exists() else {}),
        **({"vector": VECTOR_INDEX_PATH} if VECTOR_INDEX_PATH.exists() else {}),
    }
    for name, path in to_check.items():
        if name not in stored:
            continue
        actual = sha256_path(path)
        if actual != stored[name]:
            raise RuntimeError(
                f"Artifact fingerprint mismatch for '{name}': "
                f"manifest={stored[name]!r}, disk={actual!r}. "
                "Re-index or run reset_all.py."
            )

    return manifest


# ------------------------------------------------------------------ service loader

def load_services() -> SearchServices:
    """Load and validate all search services. Called once at startup."""
    manifest_path = INDEX_DIR / "artifact_manifest.json"

    # If the manifest exists, validate it. If it doesn't (e.g. dev without
    # indexing), skip the check so the API can still start for development.
    if manifest_path.exists():
        manifest = verify_complete_manifest_and_hashes(manifest_path)
        expected_counts = manifest.get("counts", {})
    else:
        expected_counts = {}

    store = SQLiteDocstore(DOCSTORE_PATH, read_only=True)
    bm25 = BM25Search(docstore=store)
    vector = VectorSearch(docstore=store)
    vector.load()

    if expected_counts:
        actual = {
            "bm25":     bm25.committed_document_count(),
            "vector":   vector.index.ntotal,
            "docstore": store.count_documents(),
        }
        # Only compare keys that exist in both dicts.
        mismatches = {
            k: {"expected": expected_counts[k], "actual": actual[k]}
            for k in expected_counts
            if k in actual and expected_counts[k] != actual[k]
        }
        if mismatches:
            raise RuntimeError(f"Serving count mismatch: {mismatches}")

    spell = SpellCorrector()
    spell.load_default_dictionary()

    reranker: CrossEncoderReranker | None = None
    if ENABLE_RERANKER:
        reranker = CrossEncoderReranker()
        _ = reranker.model  # force-load before accepting traffic

    return SearchServices(
        spell=spell,
        bm25=bm25,
        hybrid=HybridSearchEngine(bm25_search=bm25, vector_search=vector),
        reranker=reranker,
        rerank_slots=BoundedSemaphore(1),
    )


# ------------------------------------------------------------------ app factory

def create_app(
    loader: Callable[[], SearchServices] = load_services,
) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        loader: callable that returns a SearchServices instance.
                Override in tests to inject fake services without real indexes.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        try:
            app.state.services = loader()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False

    application = FastAPI(
        title="Hybrid Search Engine",
        description="BM25 + vector + optional cross-encoder reranking.",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS: loaded from env so no code change is needed to allow a new origin.
    application.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    application.include_router(search_router)

    # ---- liveness probe — always 200 while the process is running ----
    @application.get("/health/live", tags=["health"])
    def live():
        return {"status": "alive"}

    # ---- readiness probe — 503 until services are loaded ----
    @application.get("/health/ready", tags=["health"])
    def ready(request: Request):
        if not getattr(request.app.state, "ready", False):
            raise HTTPException(503, detail="Search services are not ready")
        return {"status": "ready"}

    # ---- backward-compat health endpoint ----
    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health():
        return HealthResponse(status="ok", service="hybrid-search-engine")

    # ---- Prometheus metrics ----
    if _PROMETHEUS_AVAILABLE:
        @application.get("/metrics", include_in_schema=False)
        def metrics():
            return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

        @application.middleware("http")
        async def _prometheus_middleware(request: Request, call_next):
            route = request.url.path
            method = request.method
            start = time.perf_counter()
            response = await call_next(request)
            duration = time.perf_counter() - start
            # Never label by query text — that leaks user data and explodes cardinality.
            _REQUEST_COUNT.labels(method=method, route=route,
                                  status=response.status_code).inc()
            _REQUEST_LATENCY.labels(method=method, route=route).observe(duration)
            return response

    return application


# Module-level app instance used by uvicorn.
app = create_app()
