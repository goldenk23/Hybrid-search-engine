# src/api/main.py

"""
Main entry point of the FastAPI backend application.

This file is responsible for:
1. Creating the FastAPI app
2. Configuring middleware (CORS)
3. Connecting route files to the app
4. Defining global/simple endpoints like health checks
"""

# FastAPI is the main framework used to build the backend API
from fastapi import FastAPI

# CORS middleware allows frontend and backend
# running on different ports/domains to communicate
from fastapi.middleware.cors import CORSMiddleware

# Import response schema (Pydantic model)
# used by the /health endpoint
from src.api.models import HealthResponse

# Import search routes from another file
# "router" is renamed to "search_router" for readability
from src.api.routes.search import router as search_router


# -------------------------------------------------------------------
# Create the FastAPI application object
# -------------------------------------------------------------------
# Think of "app" as the main backend server instance.
#
# These fields are metadata and appear automatically
# in FastAPI Swagger documentation (/docs).
# -------------------------------------------------------------------
app = FastAPI(
    title="Hybrid Search Engine",
    description=(
        "Search API supporting BM25, vector search, "
        "reranking, and learning-to-rank."
    ),
    version="0.1.0",
)


# -------------------------------------------------------------------
# Configure CORS Middleware
# -------------------------------------------------------------------
# Browsers block requests between different origins by default.
#
# Example:
# Frontend -> localhost:3000
# Backend  -> localhost:8000
#
# Without CORS configuration, frontend requests would fail.
# -------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,

    # Frontend URLs allowed to access this backend
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],

    # Allows cookies/authentication headers to be sent
    allow_credentials=True,

    # Allow all HTTP methods:
    # GET, POST, PUT, DELETE, etc.
    allow_methods=["*"],

    # Allow all request headers
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Register/Search API Routes
# -------------------------------------------------------------------
# This connects all endpoints defined in:
#
# src/api/routes/search.py
#
# to the main FastAPI app.
#
# Example endpoints may include:
# /search
# /search/hybrid
# -------------------------------------------------------------------
app.include_router(search_router)


# -------------------------------------------------------------------
# Health Check Endpoint
# -------------------------------------------------------------------
# Used to verify that:
# - backend server is running
# - deployment is successful
# - container/service is healthy
#
# URL:
# GET /health
# -------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Simple health check endpoint.

    Returns:
        HealthResponse:
            JSON response showing server status.
    """

    # Return a Pydantic response object
    # FastAPI automatically converts it to JSON
    return HealthResponse(
        status="ok",
        service="hybrid-search-engine",
    )