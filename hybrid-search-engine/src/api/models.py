"""
Pydantic models for API requests and responses.

These models define the JSON shape returned by the API.

The purpose of these models is to provide a clear contract for the API responses, ensuring that clients can reliably parse and understand the data returned by the search engine. They also help with validation and documentation of the API endpoints.
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