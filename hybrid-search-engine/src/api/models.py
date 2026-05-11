"""
Pydantic models for API requests and responses.

These models define the JSON shape returned by the API.
"""
from pydantic import BaseModel
# BaseModel is a base class for creating data models in Pydantic. It provides validation and serialization capabilities for the defined fields.

class SearchResult(BaseModel):
    """
    Model representing a single search result.
    """
    id: str
    title: str
    body: str
    category: str
    score: float
    snippet: str | None=None

class SearchResponse(BaseModel):
    """
    Model representing the search response containing multiple results.
    """
    query: str
    total: int
    latency_ms: int
    results: list[SearchResult]

class HealthResponse(BaseModel):
    """
    Model representing the health check response.
    """
    status: str
    service: str