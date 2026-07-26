# Hybrid Search Engine — End-to-End Guide

This guide takes you from installation to a working UI, API checks, tests, indexing, and benchmarking.

## What runs

- **Backend:** FastAPI on `http://localhost:8000`
- **Frontend:** Next.js on `http://localhost:3000`
- **Storage:** SQLite docstore plus Tantivy BM25 and FAISS vector indexes

PostgreSQL and Redis are **not used**. If the existing indexes are complete, you do not need to rebuild them.

## 1. Prerequisites

Install:

- Python 3.11 or newer
- Node.js and [pnpm](https://pnpm.io/installation)
- Git

Open PowerShell in the repository root.

## 2. Install the backend

```powershell
Set-Location .\hybrid-search-engine
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,metrics]"
```

Verify the existing artifacts:

```powershell
.\.venv\Scripts\python.exe manage.py status
```

For the checked-in local artifacts, the manifest should be `complete` and BM25, vector, and docstore counts should each be **1,499,977**.

## 3. Start the platform

### Terminal 1 — backend

From `hybrid-search-engine`:

```powershell
.\.venv\Scripts\python.exe manage.py serve
```

Wait for `Application startup complete`. The first start can take longer while models load or download.
Check their status:

```powershell
Set-Location ..\hybrid-search-engine
.\.venv\Scripts\python.exe manage.py status
```

The current manifest should report `status=complete` and matching BM25, vector, and docstore counts. This repository's current generation contains **1,499,977 documents**.

If the artifacts already exist and match, skip to the next section. Otherwise, follow [Build or rebuild the indexes](#build-or-rebuild-the-indexes).

## 4. Start and test the platform

Use two PowerShell terminals. Initial backend startup can take time while indexes and models load.

### Terminal 1 — backend

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine'
.\.venv\Scripts\python.exe manage.py serve --reload
```

Wait until Uvicorn reports that application startup is complete. Then verify readiness from another terminal:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/health/ready'
```

Expected response: `status: ready`.

Useful backend URLs:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Liveness: <http://127.0.0.1:8000/health/live>
- Readiness: <http://127.0.0.1:8000/health/ready>
- Prometheus metrics, when installed: <http://127.0.0.1:8000/metrics>

### Terminal 2 — frontend

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine-frontend'
$env:NEXT_PUBLIC_API_BASE_URL='http://127.0.0.1:8000'
pnpm dev
```

Open <http://localhost:3000>.

### End-to-end UI check

1. Enter a query such as **what causes rain**.
2. Try **Keyword** and confirm results appear.
3. Try **Hybrid** and adjust BM25/vector weights.
4. Confirm result count, latency, snippets, and corrected query display correctly.
5. Try **Reranked** only after enabling the reranker as described below.

## 5. Test the API directly

All search endpoints are `GET` requests. Queries must contain 3–256 characters.

```powershell
# BM25 keyword search
Invoke-RestMethod 'http://127.0.0.1:8000/search?q=what%20causes%20rain&top_k=10'

# BM25 + vector search using Reciprocal Rank Fusion (RRF)
Invoke-RestMethod 'http://127.0.0.1:8000/hybrid-search?q=what%20causes%20rain&top_k=10&bm25_weight=1&vector_weight=1&rrf_k=60'

# Include full passage bodies (off by default)
Invoke-RestMethod 'http://127.0.0.1:8000/hybrid-search?q=what%20causes%20rain&top_k=3&include_body=true'
```

Every response includes `query`, `corrected_query`, `returned_count`, `latency_ms`, and `results`. Keyword results have `score`; hybrid results include RRF, BM25, vector scores, and source ranks.
To enable cross-encoder reranking, set the flag **before** starting the backend:

```powershell
$env:ENABLE_RERANKER = "true"
.\.venv\Scripts\python.exe manage.py serve
```

### Terminal 2 — frontend

From the repository root:

```powershell
Set-Location .\hybrid-search-engine-frontend
pnpm install --frozen-lockfile
pnpm dev
```

Open `http://localhost:3000`. API documentation is at `http://localhost:8000/docs`.

The frontend uses `http://localhost:8000` by default. To use another backend:

```powershell
$env:NEXT_PUBLIC_API_BASE_URL = "http://localhost:9000"
pnpm dev
```

## 4. Verify the API

Run these in another PowerShell terminal:

```powershell
# Process is alive
Invoke-RestMethod "http://localhost:8000/health/live"

# Search artifacts are loaded and ready
Invoke-RestMethod "http://localhost:8000/health/ready"

# BM25 keyword search
Invoke-RestMethod "http://localhost:8000/search?q=machine%20learning&top_k=3"

# BM25 + vector search with Reciprocal Rank Fusion (RRF)
Invoke-RestMethod "http://localhost:8000/hybrid-search?q=machine%20learning&top_k=3"

# Hybrid candidates rescored by the cross-encoder; requires ENABLE_RERANKER=true
Invoke-RestMethod "http://localhost:8000/hybrid-search/rerank?q=machine%20learning&top_k=3&candidates_k=100"
```

Add `| ConvertTo-Json -Depth 6` to any command for formatted JSON.

### Search endpoints

| Mode | Endpoint | Purpose |
| --- | --- | --- |
| Keyword | `GET /search` | Fast exact-term BM25 retrieval |
| Hybrid | `GET /hybrid-search` | Fuse BM25 and semantic vector candidates |
| Reranked | `GET /hybrid-search/rerank` | Hybrid retrieval followed by cross-encoder rescoring |

All endpoints use URL query parameters, not a JSON request body. Common parameters are:

- `q`: required query, 3–256 characters
- `top_k`: number of results; default 10
- `include_body=true`: include full passage bodies (omitted by default)
- Hybrid/reranked: `bm25_weight`, `vector_weight`, and `rrf_k`
- Reranked: `candidates_k` must be at least `top_k`

Responses contain `query`, `corrected_query`, `returned_count`, `latency_ms`, and `results`.
## 5. Test through the UI

At `http://localhost:3000`:

1. Search for `how does photosynthesis work`.
2. Try **Keyword**, **Hybrid**, and **Reranked** modes.
3. Confirm results, result count, and latency appear.
4. Change `top_k` and the BM25/vector weights, then search again.
5. If reranking returns `503`, restart the backend with `ENABLE_RERANKER=true`.

## 6. Build indexes only when needed

> Skip this section when `manage.py status` reports complete artifacts. Indexing 1.5 million passages is time- and disk-intensive.

To download MS MARCO and build BM25 followed by the vector index:

```powershell
Set-Location .\hybrid-search-engine
.\.venv\Scripts\python.exe manage.py setup --max-docs 1500000
```

If the corpus already exists, build or resume each stage explicitly:

```powershell
.\.venv\Scripts\python.exe manage.py index-bm25 --max-docs 1500000
.\.venv\Scripts\python.exe manage.py index-vector --max-docs 1500000
.\.venv\Scripts\python.exe manage.py status
```

Always run vector indexing after BM25; it finalizes the artifact manifest and verifies that all document counts agree. Interrupted indexing resumes when the same command is rerun.

## 7. Run checks

Backend checks, from `hybrid-search-engine`:

```powershell
.\.venv\Scripts\python.exe manage.py test
.\.venv\Scripts\python.exe -m ruff check src tests scripts Benchmark
```

Frontend checks, from `hybrid-search-engine-frontend`:

```powershell
pnpm lint
pnpm build
```

A quick check against an already-running backend is also available:

```powershell
.\.venv\Scripts\python.exe manage.py smoke --query "how does photosynthesis work"
```

## 8. Run the reproducible 1.5M benchmark

The fixed cohort and current 1,499,977-document artifacts are already present. From `hybrid-search-engine`, run:

```powershell
.\.venv\Scripts\python.exe Benchmark\benchmark_retrieval.py `
  --manifest data\indexes\artifact_manifest.json `
  --cohort Benchmark\cohorts\dev1000.json `
  --queries data\msmarco\queries.dev.small.tsv `
  --qrels data\msmarco\qrels.dev.small.tsv `
  --vector-index data\indexes\vector.faiss `
  --sq8-index data\indexes\vector.sq8.faiss `
  --corpus-label 1.5M `
  --corpus-size 1499977 `
  --repeats 5 `
  --rerank-queries 200 `
  --output-dir Benchmark\results
```

### Enable cross-encoder reranking

Reranking is disabled by default because it loads an additional model and is slower. Stop the backend, then restart it in the same terminal with:

```powershell
$env:ENABLE_RERANKER='true'
.\.venv\Scripts\python.exe manage.py serve --reload
```

Test it after readiness succeeds:

```powershell
Invoke-RestMethod 'http://127.0.0.1:8000/hybrid-search/rerank?q=what%20causes%20rain&top_k=10&candidates_k=100'
```

The reranker first retrieves `candidates_k` hybrid candidates, then returns the best `top_k`. `candidates_k` must be at least `top_k`. Only one rerank request runs at a time; concurrent requests can receive HTTP 429.

## 6. Build or rebuild the indexes

Only do this when artifacts are missing, stale, or intentionally being replaced. Building 1.5 million vector embeddings is a long-running, compute-heavy operation. Indexers save checkpoints, so rerunning the same command resumes interrupted work.

### One-command setup

This downloads MS MARCO and builds both indexes:

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine'
.\.venv\Scripts\python.exe manage.py setup --max-docs 1500000
```

The first run downloads the corpus (about 3 GB) and Hugging Face model files.

### Run stages separately

```powershell
# Download queries, qrels, and the passage collection
.\.venv\Scripts\python.exe manage.py download --include-collection

# Build/resume BM25, then vector indexing
.\.venv\Scripts\python.exe manage.py index-bm25 --collection data\msmarco\collection.tsv --max-docs 1500000
.\.venv\Scripts\python.exe manage.py index-vector --collection data\msmarco\collection.tsv --max-docs 1500000

# Inspect progress and final counts
.\.venv\Scripts\python.exe manage.py status
```

To delete generated indexes, manifest, and docstore while preserving raw MS MARCO files:

```powershell
# Destructive and not reversible without rebuilding
.\.venv\Scripts\python.exe manage.py reset --yes
```

## 7. Run checks

Backend tests use fake services and do not need the full indexes:

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine'
.\.venv\Scripts\python.exe manage.py test -v

# API tests only
.\.venv\Scripts\python.exe -m pytest tests\test_api.py -v

# Smoke test against a running backend
.\.venv\Scripts\python.exe manage.py smoke --query 'what causes rain'
```

Frontend production check:

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine-frontend'
pnpm build
```
This is a long-running benchmark. It evaluates BM25, vector, RRF variants, SQ8 (when supplied), and hybrid + cross-encoder reranking. Results are written atomically to:

- `Benchmark/results/1.5M.json` — machine-readable measurements and provenance
- `Benchmark/results/1.5M.md` — readable report
- `Benchmark/results/README.md` — cross-run summary

Only 268 queries in the fixed cohort have relevant documents inside this 1.5M subset, so quality metrics use those queries. The reranked run uses 200 of them. Its `Recall@100` is not directly comparable to candidate retrievers because reranking returns a shorter final list; use NDCG@10 and MRR@10 to judge reranking quality.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `/health/ready` returns `503` | Wait for startup. If it persists, inspect the backend error and run `manage.py status`. |
| Artifact hash or count mismatch | The indexes came from different builds. Rebuild BM25 and vector artifacts together; do not bypass validation. |
| Reranked search returns `503` | Set `$env:ENABLE_RERANKER = "true"` and restart the backend. |
| Reranked search returns `429` | One rerank is already running; retry after it finishes. |
| Search returns `422` | Use a query of at least 3 characters and ensure `candidates_k >= top_k`. |
| Frontend cannot reach the API | Confirm the backend is on port 8000 and `/health/ready` succeeds; otherwise set `NEXT_PUBLIC_API_BASE_URL`. |
| Model download fails | Check internet access and retry; Sentence Transformers models are downloaded on first use. |
| Port already in use | Backend: `manage.py serve --port 9000`; then set the frontend API URL to `http://localhost:9000`. |

For every available command and option:

```powershell
.\.venv\Scripts\python.exe manage.py --help
.\.venv\Scripts\python.exe manage.py <command> --help
```

## 8. Benchmark the current 1.5M generation

The checked-in cohort and current manifest can run the full reproducible benchmark below. It measures BM25, vectors, hybrid RRF variants, and cross-encoder reranking over five repetitions; it also validates the SQ8 index.

```powershell
Set-Location 'C:\Users\golde\Desktop\Projects\Hybrid_search_engine\hybrid-search-engine'
.\.venv\Scripts\python.exe Benchmark\benchmark_retrieval.py `
  --manifest data\indexes\artifact_manifest.json `
  --cohort Benchmark\cohorts\dev1000.json `
  --queries data\msmarco\queries.dev.small.tsv `
  --qrels data\msmarco\qrels.dev.small.tsv `
  --vector-index data\indexes\vector.faiss `
  --sq8-index data\indexes\vector.sq8.faiss `
  --corpus-label 1.5M `
  --corpus-size 1499977 `
  --repeats 5 `
  --rerank-queries 200 `
  --output-dir Benchmark\results
```

Outputs are written to:

```text
Benchmark\results\1.5M.json
Benchmark\results\1.5M.md
Benchmark\results\README.md
```

For a faster run without the cross-encoder:

```powershell
.\.venv\Scripts\python.exe manage.py benchmark --cohort Benchmark\cohorts\dev1000.json --corpus-label 1.5M --corpus-size 1499977 --skip-rerank
```

If you rebuild a different corpus, use the exact count from `artifact_manifest.json` for `--corpus-size` and create a fixed cohort with:

```powershell
.\.venv\Scripts\python.exe manage.py cohort --max-queries 500 --output Benchmark\cohorts\dev500.json
```

## Configuration

Optional environment variables can be set before backend startup or placed in `hybrid-search-engine\.env`:

```dotenv
ENABLE_RERANKER=false
RESULTS_PER_PAGE=10
BM25_TOP_K=100
VECTOR_TOP_K=100
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

Advanced path overrides are `BM25_INDEX_PATH`, `VECTOR_INDEX_PATH`, and `DOCSTORE_PATH`. Restart the backend after changing configuration.

## Troubleshooting

| Symptom | Check |
|---|---|
| Backend fails during startup | Run `manage.py status`; ensure the manifest is complete and BM25/vector/docstore counts and hashes match. |
| Frontend says it cannot reach the API | Confirm `/health/ready`, `NEXT_PUBLIC_API_BASE_URL`, port 8000, and backend CORS origins. |
| Reranked mode returns HTTP 503 | Set `ENABLE_RERANKER=true` before starting the backend, then wait for model loading. |
| Search returns HTTP 422 | Use a query of 3–256 characters and valid limits; for reranking, ensure `candidates_k >= top_k`. |
| Port 8000 is occupied | Find it with `Get-NetTCPConnection -LocalPort 8000`, or start with `manage.py serve --port 8001` and update the frontend API URL. |
| Indexing was interrupted | Rerun the same indexing command; it resumes from its checkpoint. |

For every backend command and option, run:

```powershell
.\.venv\Scripts\python.exe manage.py --help
.\.venv\Scripts\python.exe manage.py <command> --help
```
