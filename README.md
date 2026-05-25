# 🔍 Hybrid Search Engine

A production-ready hybrid search engine built on the **MS MARCO** passage corpus. It combines keyword-based BM25 retrieval, dense vector semantic search, and optional cross-encoder reranking into a single FastAPI service.

---

## ✨ Features

| Feature | Details |
|---|---|
| **BM25 Keyword Search** | Powered by [Tantivy](https://github.com/quickwit-oss/tantivy) for fast, exact keyword matching |
| **Dense Vector Search** | Sentence embeddings via `all-MiniLM-L6-v2` + FAISS inner-product index |
| **Reciprocal Rank Fusion** | Weighted RRF merges BM25 and vector results into a single ranked list |
| **Cross-Encoder Reranking** | Optional third-stage reranker using `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **Spell Correction** | SymSpell query pre-processing with compound and per-word correction |
| **REST API** | FastAPI with Swagger UI, CORS support, and health check endpoint |
| **Resumable Indexing** | Checkpoint-based streaming indexer — resume interrupted index builds safely |
| **Compressed Docstore** | zlib-compressed SQLite document store for fast ID → document lookup |

---

## 🏗️ Architecture

```
Query
  │
  ▼
Spell Correction (SymSpell)
  │
  ├──► BM25 Search (Tantivy)  ─────────────────────┐
  │                                                 │
  └──► Vector Search (FAISS + SentenceTransformers) ┘
                                                    │
                                         Reciprocal Rank Fusion (RRF)
                                                    │
                                      (Optional) Cross-Encoder Reranker
                                                    │
                                              Final Results
```

Both BM25 and Vector retrievers pull document bodies from a shared **SQLite docstore**, keeping the search indexes lean.

---

## 📊 Benchmark Results

Evaluated on the **MS MARCO Dev** query set. Metrics: NDCG@10, MRR@10, Recall@100, and latency percentiles.

### 1M Corpus — 315 Evaluation Queries

| System | NDCG@10 | MRR@10 | Recall@100 | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.2052 | 0.1624 | 0.6444 | 17.91 | 33.83 |
| Vector | 0.3794 | 0.3105 | 0.9048 | 116.70 | 132.49 |
| Hybrid RRF (1.0 / 1.0) | 0.3339 | 0.2710 | 0.8905 | 148.27 | 185.30 |
| Weighted RRF (0.50 / 1.00) | 0.3568 | 0.2874 | 0.9016 | 147.58 | 174.59 |
| Weighted RRF (0.25 / 1.00) | 0.3737 | 0.2992 | 0.9048 | 150.84 | 172.36 |
| **Hybrid + Cross-Encoder** | **0.4068** | **0.3319** | 0.6400 | 844.87 | 991.17 |

### 2M Corpus — 153 Evaluation Queries

| System | NDCG@10 | MRR@10 | Recall@100 | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|---:|---:|
| BM25 | 0.1753 | 0.1423 | 0.5621 | 32.81 | 64.45 |
| Vector | 0.3051 | 0.2423 | 0.8660 | 215.92 | 1218.71 |
| Hybrid RRF (1.0 / 1.0) | 0.2758 | 0.2225 | 0.8431 | 410.82 | 1355.02 |
| Weighted RRF (0.50 / 1.00) | 0.3028 | 0.2382 | 0.8693 | 241.70 | 280.12 |
| Weighted RRF (0.25 / 1.00) | 0.3139 | 0.2460 | 0.8660 | 1050.10 | 1207.97 |
| **Hybrid + Cross-Encoder** | **0.3607** | **0.2808** | 0.6200 | 1955.62 | 2196.24 |

> **Key takeaways**
> - Vector retrieval consistently beats BM25 alone on NDCG and Recall in partial-corpus benchmarks.
> - Down-weighting BM25 (0.25 / 1.00) often matches or exceeds equal-weight hybrid RRF.
> - Cross-encoder reranking achieves the highest NDCG/MRR at the cost of ~10× higher latency.
> - For latency-sensitive applications, **Weighted RRF (0.25 / 1.00)** offers the best quality-speed trade-off.

---

## 🚀 Quick Start

### 1. Install

```bash
cd hybrid-search-engine
python -m venv .venv

# Windows
.\.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

pip install -e .[dev]
```

### 2. Download MS MARCO Data

```bash
python scripts/download_msmarco.py --include-collection
```

### 3. Build Indexes

```bash
# BM25 index (1M passages)
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 1000000 --reset

# Vector index (1M passages)
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 1000000 --reset

# Optional: quantize vector index to 8-bit for smaller footprint
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

### 4. Start the API

```bash
python -m uvicorn src.api.main:app --reload
```

API runs at **http://127.0.0.1:8000**
- Swagger UI: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

---

## 🔌 API Endpoints

### `GET /search` — BM25 only

```
GET /search?q=what+causes+rain&top_k=10
```

### `GET /hybrid-search` — BM25 + Vector + RRF

```
GET /hybrid-search?q=what+causes+rain&bm25_weight=0.25&vector_weight=1.0&rrf_k=60
```

### `GET /hybrid-search/rerank` — Hybrid + Cross-Encoder

```
GET /hybrid-search/rerank?q=what+causes+rain&top_k=10&candidates_k=100
```

### `GET /health`

```json
{ "status": "ok" }
```

---

## 🗂️ Project Structure

```
hybrid-search-engine/
├── src/
│   ├── config.py                  # All environment/path/model config
│   ├── api/
│   │   ├── main.py                # FastAPI app entry point
│   │   ├── models.py              # Pydantic response models
│   │   └── routes/search.py       # Search route handlers
│   ├── search/
│   │   ├── bm25.py                # Tantivy BM25 search
│   │   ├── vector.py              # FAISS vector search
│   │   ├── fusion.py              # Reciprocal Rank Fusion
│   │   ├── hybrid_search.py       # BM25 + Vector orchestration
│   │   └── cross_encoder_reranker.py  # Cross-encoder reranker
│   ├── query/
│   │   └── spell_check.py         # SymSpell query correction
│   ├── indexing/
│   │   ├── pipeline.py            # MS MARCO streaming loader
│   │   ├── preprocessing.py       # Text cleaning & snippet generation
│   │   └── checkpoint.py          # Resumable indexing checkpoints
│   └── database/
│       └── docstore.py            # Compressed SQLite docstore
├── scripts/
│   ├── index_documents.py         # CLI: build BM25 index
│   ├── index_vectors.py           # CLI: build FAISS index
│   ├── quantize_vector_index.py   # CLI: quantize FAISS to SQ8/FP16
│   ├── evaluate.py                # BM25-only metric evaluation
│   └── download_msmarco.py        # Download MS MARCO dataset
├── Benchmark/
│   ├── benchmark_retrieval.py     # Full benchmark runner
│   ├── results/                   # Saved benchmark results (JSON + Markdown)
│   └── create_eval_subset.py      # Create smaller eval subsets
├── tests/                         # Pytest test suite
└── data/
    ├── msmarco/                   # Query / qrels / collection files
    ├── indexes/                   # BM25 and FAISS indexes
    └── docstore.sqlite            # Compressed document store
```

---

## ⚙️ Configuration

All settings live in `src/config.py` and can be overridden via environment variables.

| Variable | Default | Description |
|---|---|---|
| `BM25_INDEX_PATH` | `data/indexes/bm25_compact` | Tantivy BM25 index directory |
| `VECTOR_INDEX_PATH` | `data/indexes/vector.faiss` | FAISS vector index file |
| `DOCSTORE_PATH` | `data/docstore.sqlite` | SQLite docstore |
| `API_HOST` | `0.0.0.0` | API bind host |
| `API_PORT` | `8000` | API port |

**Default models:**
- Embedding: `all-MiniLM-L6-v2`
- Cross-encoder: `cross-encoder/ms-marco-MiniLM-L-6-v2`

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

> **Note:** `tests/test_api.py` requires a built BM25 index and docstore to exist. Run indexing first.

---

## 📈 Running Benchmarks

```bash
# Benchmark against 1M corpus, 500 eval queries, 100 rerank queries
python Benchmark/benchmark_retrieval.py \
  --corpus-label 1M \
  --corpus-size 1000000 \
  --max-queries 500 \
  --rerank-queries 100
```

Results are saved to `Benchmark/results/` as both `.json` and `.md` files and the summary `README.md` is updated automatically.

To skip the slow cross-encoder step:

```bash
python Benchmark/benchmark_retrieval.py --corpus-label 1M --corpus-size 1000000 --skip-rerank
```

---

## 📦 Tech Stack

| Layer | Library |
|---|---|
| API | FastAPI, Uvicorn, Pydantic |
| BM25 | Tantivy (Python bindings) |
| Vector index | FAISS (CPU) |
| Embeddings | sentence-transformers |
| Reranking | sentence-transformers CrossEncoder |
| Spell correction | symspellpy |
| Docstore | SQLite (zlib-compressed) |
| Data | numpy, pandas, tqdm |
| Testing | pytest, pytest-asyncio |
| Load testing | Locust |

---


