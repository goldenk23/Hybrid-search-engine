"""
Locust load test for the Hybrid Search Engine API.

USAGE
-----
Start the API first:
    python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

Run headless (CI / one-shot):
    locust -f locustfile.py \
        --host http://localhost:8000 \
        --headless \
        --users 10 \
        --spawn-rate 2 \
        --run-time 60s \
        --csv results/locust

Run with the web UI (interactive):
    locust -f locustfile.py --host http://localhost:8000
    Then open http://localhost:8089

WHAT IS TESTED
--------------
SearchUser simulates a realistic mix of API traffic:
  - 60 % BM25 /search
  - 30 % hybrid /hybrid-search
  - 10 % rerank  /hybrid-search/rerank

Sample queries are drawn from a fixed set so results are reproducible
across runs.  The rerank task uses a small candidates_k to keep latency
acceptable during load tests.

INTERPRETING RESULTS
--------------------
Save one run's output and commit it as evidence:
    --csv results/locust          → locust_stats.csv, locust_failures.csv
    --html results/locust.html    → full HTML report

Key numbers to record:
    users, duration, RPS (requests/s), failure %, p50, p95, p99 latency.
"""

from locust import HttpUser, between, task

_QUERIES = [
    "what causes rain",
    "how does photosynthesis work",
    "best programming languages for machine learning",
    "explain quantum entanglement",
    "history of the Roman Empire",
    "how to treat a sprained ankle",
    "difference between supervised and unsupervised learning",
    "what is the speed of light",
    "how does the immune system work",
    "causes of the French Revolution",
]


class SearchUser(HttpUser):
    """Simulates a user sending a realistic mix of search queries."""

    # Each simulated user waits 1–3 seconds between requests.
    # Adjust to model your actual traffic pattern.
    wait_time = between(1, 3)

    # Round-robin through the sample queries.
    _query_index = 0

    def _next_query(self) -> str:
        q = _QUERIES[SearchUser._query_index % len(_QUERIES)]
        SearchUser._query_index += 1
        return q

    @task(6)
    def bm25_search(self):
        """BM25 keyword search — 60 % of traffic."""
        self.client.get(
            "/search",
            params={"q": self._next_query(), "top_k": 10},
            name="/search",
        )

    @task(3)
    def hybrid_search(self):
        """Hybrid RRF search — 30 % of traffic."""
        self.client.get(
            "/hybrid-search",
            params={
                "q": self._next_query(),
                "top_k": 10,
                "bm25_weight": 0.25,
                "vector_weight": 1.0,
            },
            name="/hybrid-search",
        )

    @task(1)
    def rerank_search(self):
        """Cross-encoder rerank — 10 % of traffic (expensive, intentionally rare)."""
        self.client.get(
            "/hybrid-search/rerank",
            params={
                "q": self._next_query(),
                "top_k": 10,
                "candidates_k": 50,
            },
            name="/hybrid-search/rerank",
        )

    def on_start(self):
        """Confirm the API is ready before sending traffic."""
        with self.client.get("/health/ready", catch_response=True) as r:
            if r.status_code != 200:
                r.failure(f"API not ready: {r.status_code}")
