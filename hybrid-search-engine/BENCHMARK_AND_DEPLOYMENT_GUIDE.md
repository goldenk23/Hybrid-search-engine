# Benchmark and Deployment Guide

This guide is for building credible resume-grade benchmarks while keeping deployment costs low.

## Current Artifact Sizes

Measured on the current local 750K-document MS MARCO index:

| Artifact | Path | Size |
|---|---|---:|
| Quantized FAISS index | `data/indexes/vector.sq8.faiss` | 209.4 MiB |
| Compact BM25 index | `data/indexes/bm25_compact` | 268.2 MiB |
| SQLite docstore | `data/docstore.sqlite` | 174.8 MiB |
| Total search artifacts | vector + BM25 + docstore | 652.4 MiB |

The application code is small. The deployment challenge is the search artifacts plus Python ML dependencies.

## Improvements Before Benchmarking

Do these before publishing benchmark numbers. They keep the results credible and prevent avoidable re-runs.

### Must Do

1. Add a multi-system evaluator.
   The current `scripts/evaluate.py` evaluates only BM25. Before serious benchmarks, add an evaluator that can run:
   - BM25
   - vector search
   - hybrid RRF
   - hybrid + cross-encoder rerank

2. Filter qrels for partial corpus evaluation.
   For 750K, 1M, 2M, 3M, and 4M indexes, many MS MARCO qrel document IDs may not exist in the current index. Filter evaluation queries to those with at least one relevant document in the indexed corpus.

3. Save benchmark results as JSON.
   Do not rely on terminal output. Save one result file per corpus size.

4. Record artifact sizes with every benchmark.
   Index size is part of the engineering story. Include BM25, vector, docstore, and total size.

5. Record document counts.
   Confirm BM25 count, FAISS `ntotal`, and SQLite docstore count match or explain why they differ.

6. Use the same query set across all systems for a given corpus.
   BM25, vector, hybrid, and reranker should be evaluated on the same filtered query IDs.

7. Warm up before measuring latency.
   The first requests include model/index loading and OS file cache effects. Measure cold start separately if you want, but do not mix cold start into normal p50/p95.

### Strongly Recommended

1. Add vector-only evaluation.
   This lets you prove hybrid retrieval is actually better than either component alone.

2. Add an option to use the quantized FAISS index.
   For benchmark clarity, run one set with `vector.faiss` and one set with `vector.sq8.faiss`, or explicitly state which one is used.

3. Add missing tests for:
   - RRF ordering
   - hybrid search shape
   - evaluator metric formulas
   - qrel filtering

4. Add a README benchmark summary table.
   The guide is for you; the README is for recruiters and interviewers.

5. Add a small demo mode.
   Use 50K-100K docs for a free public demo, while keeping large benchmarks local.

### Optional Later

1. Add Learning to Rank only after benchmark infrastructure is stable.
   LTR is impressive only if you can prove it improves `NDCG@10` or `MRR@10`.

2. Add HTTP load testing after in-process quality benchmarks.
   First measure retrieval quality and engine latency. Then measure API overhead separately.

## Deployment Recommendation

For a student budget, separate the project into two modes:

| Mode | Purpose | Corpus | Components | Hosting Target |
|---|---|---:|---|---|
| Demo mode | Public live demo | 50K-100K | BM25 or quantized vector + BM25 | Hugging Face Spaces / Oracle Always Free |
| Benchmark mode | Resume metrics | 750K-4M | BM25 + vector + RRF + reranker | Local machine |
| Full research mode | Stretch goal | 4M+ | Same as benchmark mode | Local machine / paid VM |

Do not try to deploy the 4M-document system on a tiny free web service. Keep the large index local and publish measured results.

## Low-Cost Hosting Options

### Option 1: Hugging Face Spaces

Best for: public demo, recruiter-friendly link.

Recommended setup:

- Use Docker Space or Gradio/FastAPI wrapper.
- Use a small corpus, ideally 50K-100K docs.
- Disable cross-encoder reranking by default.
- Store artifacts in the repo only if they are small enough and public.
- Keep the full benchmark results in the README.

Pros:

- Easy public ML demo.
- Free CPU Basic hardware is useful for small demos.
- Good fit for search/ML projects.

Cons:

- Default disk is not persistent.
- Large index artifacts increase build/startup pain.
- Cross-encoder reranking may feel slow.

### Option 2: Oracle Cloud Always Free

Best for: realistic always-on student deployment if you can get capacity.

Recommended setup:

- Use a small VM with persistent block storage.
- Deploy the 750K compact bundle only if RAM and disk are comfortable.
- Use systemd or Docker Compose.
- Put artifacts on attached block volume.

Pros:

- Persistent VM-style deployment.
- More realistic backend hosting than many free PaaS options.
- Generous storage compared with many free platforms.

Cons:

- Signup/capacity can be frustrating.
- You must be careful to stay inside Always Free limits.
- More DevOps work.

### Option 3: Railway / Render / Fly.io

Best for: short demos or very small corpus deployments.

Recommended setup:

- Deploy API only with a small prebuilt index.
- Avoid cross-encoder in the live demo.
- Watch storage and memory limits closely.

Pros:

- Simple deployment flow.
- Good for web APIs.

Cons:

- Modern free tiers are usually credit-limited or sleep-prone.
- Persistent disk and RAM can become the blocker.
- Your 650 MiB artifact bundle plus ML dependencies is already heavy for free PaaS.

## Practical Deployment Strategy

Use this portfolio strategy:

1. Public demo: 50K-100K docs, cheap/free hosting.
2. README benchmark table: 750K, 1M, 2M, 3M, 4M local results.
3. Short video/GIF: show API docs and example queries.
4. Architecture diagram: show BM25, FAISS, RRF, reranker, docstore.

This is stronger than an unstable large free-tier deployment.

## Corpus Scaling Plan

Estimate based on the current 750K compact artifacts:

| Corpus Size | Estimated Artifacts | Purpose |
|---:|---:|---|
| 750K | 652 MiB | Current baseline |
| 1M | 870 MiB | First scale step |
| 2M | 1.7 GiB | Strong benchmark size |
| 3M | 2.6 GiB | Stress test |
| 4M | 3.5 GiB | Resume-grade large local benchmark |

These are rough estimates. Actual sizes vary with segment merges, compression, and average passage length.

## Benchmark Matrix

Run every retrieval stage at each corpus size:

| Corpus | BM25 | Vector | Hybrid RRF | Hybrid + Rerank |
|---:|---|---|---|---|
| 750K | yes | yes | yes | yes |
| 1M | yes | yes | yes | yes |
| 2M | yes | yes | yes | yes |
| 3M | yes | yes | yes | optional |
| 4M | yes | yes | yes | optional |

Reranking is expensive. For 3M and 4M, it is acceptable to report rerank metrics on a smaller query sample.

## Metrics to Report

For each corpus and retrieval stage:

- `NDCG@10`: ranking quality of the first page.
- `MRR@10`: how quickly the first relevant result appears.
- `Recall@100`: whether retrieval preserves relevant candidates.
- `p50 latency`: typical response time.
- `p95 latency`: high-percentile response time.
- `index size`: total artifact size.
- `indexed documents`: actual committed document count.

Recommended README table:

| Corpus | System | NDCG@10 | MRR@10 | Recall@100 | p50 ms | p95 ms | Artifact Size |
|---:|---|---:|---:|---:|---:|---:|---:|
| 750K | BM25 | TBD | TBD | TBD | TBD | TBD | TBD |
| 750K | Vector | TBD | TBD | TBD | TBD | TBD | TBD |
| 750K | Hybrid RRF | TBD | TBD | TBD | TBD | TBD | TBD |
| 750K | Hybrid + Rerank | TBD | TBD | TBD | TBD | TBD | TBD |

## Important Evaluation Rule

MS MARCO qrels reference relevant passage IDs from the full corpus. If you benchmark a partial corpus, some relevant documents may not exist in your index.

Use one of these approaches:

1. Filter eval queries to queries whose relevant documents are present in the indexed subset.
2. Report that partial-corpus metrics are measured only over available qrels.
3. Use Recall@100 carefully, because missing relevant docs can make partial-corpus recall look worse than the retrieval system really is.

Approach 1 is best.

## Metric Definitions and Formulas

### NDCG@10

`NDCG@10` measures how good the top 10 ordering is.

For binary qrels:

```text
DCG@10 = sum(relevance_i / log2(rank_i + 1))
IDCG@10 = best possible DCG@10 for the query
NDCG@10 = DCG@10 / IDCG@10
```

If a relevant document is at rank 1, it contributes `1 / log2(2) = 1.0`.
If it is at rank 5, it contributes `1 / log2(6)`.

### MRR@10

`MRR@10` measures how quickly the first relevant document appears.

```text
MRR@10 = 1 / rank_of_first_relevant_result
```

If no relevant document appears in the top 10, the score is `0`.

### Recall@100

`Recall@100` measures whether the retrieval stage found the relevant documents.

```text
Recall@100 = relevant_documents_found_in_top_100 / total_relevant_documents
```

This is especially important for reranking. A reranker cannot recover a relevant document that was missing from the first-stage candidates.

### p50 and p95 Latency

Latency should be measured per system:

- p50: median query latency.
- p95: 95th percentile query latency.

Measure in-process search latency first. HTTP latency can be measured separately later.

## Indexing Commands

Run commands from `hybrid-search-engine/`.

### 750K

```bash
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 750000 --reset
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 750000 --reset
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

### 1M

```bash
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 1000000 --reset
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 1000000 --reset
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

### 2M

```bash
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 2000000 --reset
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 2000000 --reset
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

### 3M

```bash
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 3000000 --reset
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 3000000 --reset
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

### 4M

```bash
python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 4000000 --reset
python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 4000000 --reset
python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
```

If indexing is interrupted, rerun the same command without `--reset` to resume.

## Artifact Size Measurement

Use PowerShell:

```powershell
Get-ChildItem data/indexes/bm25_compact -Recurse -File |
  Measure-Object -Property Length -Sum

Get-ChildItem data/indexes/vector.sq8.faiss, data/docstore.sqlite |
  Select-Object Name,Length
```

Record sizes in both bytes and MiB.

Python helper:

```python
from pathlib import Path


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def mib(num_bytes: int) -> float:
    return round(num_bytes / 1024 / 1024, 2)


artifact_sizes = {
    "bm25_compact_mib": mib(size_bytes(Path("data/indexes/bm25_compact"))),
    "vector_sq8_mib": mib(size_bytes(Path("data/indexes/vector.sq8.faiss"))),
    "docstore_mib": mib(size_bytes(Path("data/docstore.sqlite"))),
}
artifact_sizes["total_mib"] = round(sum(artifact_sizes.values()), 2)
print(artifact_sizes)
```

## Document Count Verification

Use this before every benchmark run:

```python
import sqlite3
from pathlib import Path

import faiss
import tantivy


def count_bm25(index_path: Path) -> int:
    index = tantivy.Index.open(str(index_path))
    return index.searcher().num_docs


def count_faiss(index_path: Path) -> int:
    index = faiss.read_index(str(index_path))
    return index.ntotal


def count_docstore(docstore_path: Path) -> int:
    with sqlite3.connect(docstore_path) as conn:
        return conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]


print("bm25:", count_bm25(Path("data/indexes/bm25_compact")))
print("faiss:", count_faiss(Path("data/indexes/vector.faiss")))
print("docstore:", count_docstore(Path("data/docstore.sqlite")))
```

If these counts differ, write down why before trusting the benchmark.

## Latency Benchmarking

Use a fixed query set for all corpus sizes. Start with 100-500 queries from `queries.dev.small.tsv`.

Recommended stages:

1. Warm up each engine with 5-10 queries.
2. Run each query once for quality metrics.
3. Run each query multiple times for latency.
4. Report p50 and p95.

Example query categories:

- factual: `what is quantum entanglement`
- medical/general: `symptoms of vitamin d deficiency`
- technical: `python list comprehension`
- historical: `who was alan turing`
- navigational style: `windows update error`

## Benchmark Procedure

For each corpus size:

1. Build BM25 index.
2. Build vector index.
3. Quantize vector index.
4. Confirm document counts.
5. Run BM25 evaluation.
6. Run vector evaluation.
7. Run hybrid RRF evaluation.
8. Run reranker evaluation on a smaller candidate/query sample.
9. Record latency.
10. Save results to `benchmarks/results/<corpus-size>.json`.

## Full Benchmark Script Snippet

Create a new script such as `scripts/benchmark_retrieval.py`. This snippet is intentionally self-contained so it is easy to adapt.

It evaluates:

- BM25
- vector search
- hybrid RRF
- hybrid + rerank

It calculates:

- `NDCG@10`
- `MRR@10`
- `Recall@100`
- `p50_ms`
- `p95_ms`
- artifact sizes
- document counts

```python
from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Callable

import faiss
import numpy as np
import tantivy
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.search.bm25 import BM25Search
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.hybrid_search import HybridSearchEngine
from src.search.vector import VectorSearch


SearchFn = Callable[[str, int], list[dict]]


def load_queries(path: Path) -> dict[str, str]:
    queries = {}
    with path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[row[0]] = row[1]
    return queries


def load_qrels(path: Path) -> dict[str, set[str]]:
    qrels = defaultdict(set)
    with path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) < 4:
                continue
            query_id, document_id, relevance = row[0], row[2], int(row[3])
            if relevance > 0:
                qrels[query_id].add(document_id)
    return dict(qrels)


def load_indexed_doc_ids(max_doc_id: int | None = None) -> set[str]:
    # MS MARCO passage IDs are numeric and the indexing pipeline indexes from
    # the start of collection.tsv. For max-docs=N, indexed IDs are usually in
    # the first N rows, but IDs can have gaps. Reading docstore is more robust.
    with sqlite3.connect(DATA_DIR / "docstore.sqlite") as conn:
        rows = conn.execute("SELECT id FROM documents").fetchall()
    ids = {str(row[0]) for row in rows}
    if max_doc_id is not None:
        ids = {doc_id for doc_id in ids if doc_id.isdigit() and int(doc_id) <= max_doc_id}
    return ids


def filter_eval_queries(
    queries: dict[str, str],
    qrels: dict[str, set[str]],
    indexed_doc_ids: set[str],
    max_queries: int | None,
) -> list[tuple[str, str, set[str]]]:
    filtered = []
    for query_id, query_text in queries.items():
        relevant_ids = qrels.get(query_id)
        if not relevant_ids:
            continue

        available_relevant_ids = relevant_ids.intersection(indexed_doc_ids)
        if not available_relevant_ids:
            continue

        filtered.append((query_id, query_text, available_relevant_ids))
        if max_queries is not None and len(filtered) >= max_queries:
            break
    return filtered


def ndcg_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    dcg = 0.0
    for index, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_ids:
            dcg += 1.0 / np.log2(index + 2)

    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    return 0.0 if idcg == 0 else float(dcg / idcg)


def mrr_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int = 10) -> float:
    for index, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_ids:
            return 1.0 / (index + 1)
    return 0.0


def recall_at_k(ranked_doc_ids: list[str], relevant_ids: set[str], k: int = 100) -> float:
    if not relevant_ids:
        return 0.0
    return len(set(ranked_doc_ids[:k]).intersection(relevant_ids)) / len(relevant_ids)


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, pct))


def evaluate_system(
    name: str,
    search_fn: SearchFn,
    eval_queries: list[tuple[str, str, set[str]]],
    top_k: int,
    warmup_queries: int = 10,
) -> dict:
    for _, query_text, _ in eval_queries[:warmup_queries]:
        search_fn(query_text, top_k)

    ndcg_scores = []
    mrr_scores = []
    recall_scores = []
    latencies_ms = []

    for _, query_text, relevant_ids in tqdm(eval_queries, desc=f"Evaluating {name}"):
        start = time.perf_counter()
        results = search_fn(query_text, top_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        ranked_doc_ids = [str(result["id"]) for result in results]
        ndcg_scores.append(ndcg_at_k(ranked_doc_ids, relevant_ids, k=10))
        mrr_scores.append(mrr_at_k(ranked_doc_ids, relevant_ids, k=10))
        recall_scores.append(recall_at_k(ranked_doc_ids, relevant_ids, k=100))

    return {
        "queries": len(eval_queries),
        "ndcg_at_10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "mrr_at_10": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "recall_at_100": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "p50_ms": round(median(latencies_ms), 2) if latencies_ms else 0.0,
        "p95_ms": round(percentile(latencies_ms, 95), 2),
    }


def size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def mib(num_bytes: int) -> float:
    return round(num_bytes / 1024 / 1024, 2)


def artifact_sizes() -> dict[str, float]:
    bm25 = mib(size_bytes(DATA_DIR / "indexes" / "bm25_compact"))
    vector = mib(size_bytes(DATA_DIR / "indexes" / "vector.faiss"))
    vector_sq8 = mib(size_bytes(DATA_DIR / "indexes" / "vector.sq8.faiss"))
    docstore = mib(size_bytes(DATA_DIR / "docstore.sqlite"))
    return {
        "bm25_compact_mib": bm25,
        "vector_faiss_mib": vector,
        "vector_sq8_mib": vector_sq8,
        "docstore_mib": docstore,
        "total_compact_with_sq8_mib": round(bm25 + vector_sq8 + docstore, 2),
    }


def document_counts() -> dict[str, int]:
    counts = {}

    bm25_path = DATA_DIR / "indexes" / "bm25_compact"
    if bm25_path.exists():
        counts["bm25"] = tantivy.Index.open(str(bm25_path)).searcher().num_docs

    vector_path = DATA_DIR / "indexes" / "vector.faiss"
    if vector_path.exists():
        counts["vector"] = faiss.read_index(str(vector_path)).ntotal

    docstore_path = DATA_DIR / "docstore.sqlite"
    if docstore_path.exists():
        with sqlite3.connect(docstore_path) as conn:
            counts["docstore"] = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval systems.")
    parser.add_argument("--corpus-label", required=True, help="Example: 750k, 1m, 2m")
    parser.add_argument("--corpus-size", type=int, required=True, help="Indexed document target")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "msmarco" / "queries.dev.small.tsv")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "msmarco" / "qrels.dev.small.tsv")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--rerank-queries", type=int, default=100)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)
    indexed_doc_ids = load_indexed_doc_ids()
    eval_queries = filter_eval_queries(queries, qrels, indexed_doc_ids, args.max_queries)

    print(f"Filtered eval queries: {len(eval_queries)}")
    if not eval_queries:
        raise RuntimeError("No eval queries have relevant docs in the current index.")

    bm25 = BM25Search()
    vector = VectorSearch()
    hybrid = HybridSearchEngine(bm25_search=bm25, vector_search=vector)

    systems = {
        "bm25": evaluate_system(
            "bm25",
            lambda query, top_k: bm25.search(query, top_k=top_k),
            eval_queries,
            top_k=100,
        ),
        "vector": evaluate_system(
            "vector",
            lambda query, top_k: vector.search(query, top_k=top_k),
            eval_queries,
            top_k=100,
        ),
        "hybrid_rrf": evaluate_system(
            "hybrid_rrf",
            lambda query, top_k: hybrid.search(query, top_k=top_k),
            eval_queries,
            top_k=100,
        ),
    }

    if not args.skip_rerank:
        reranker = CrossEncoderReranker()
        rerank_eval_queries = eval_queries[: args.rerank_queries]

        def rerank_search(query: str, top_k: int) -> list[dict]:
            candidates = hybrid.search(query, top_k=100)
            return reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=min(top_k, 10),
                max_candidates=50,
            )

        systems["hybrid_rerank"] = evaluate_system(
            "hybrid_rerank",
            rerank_search,
            rerank_eval_queries,
            top_k=10,
            warmup_queries=3,
        )

    result = {
        "corpus_label": args.corpus_label,
        "corpus_size": args.corpus_size,
        "eval_queries": len(eval_queries),
        "artifact_size_mib": artifact_sizes(),
        "document_counts": document_counts(),
        "systems": systems,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / f"{args.corpus_label}.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
```

Run it like this:

```bash
python scripts/benchmark_retrieval.py --corpus-label 750k --corpus-size 750000 --max-queries 500 --rerank-queries 100
python scripts/benchmark_retrieval.py --corpus-label 1m --corpus-size 1000000 --max-queries 500 --rerank-queries 100
python scripts/benchmark_retrieval.py --corpus-label 2m --corpus-size 2000000 --max-queries 500 --rerank-queries 100
python scripts/benchmark_retrieval.py --corpus-label 3m --corpus-size 3000000 --max-queries 500 --rerank-queries 50
python scripts/benchmark_retrieval.py --corpus-label 4m --corpus-size 4000000 --max-queries 500 --rerank-queries 50
```

For a faster first pass:

```bash
python scripts/benchmark_retrieval.py --corpus-label 750k --corpus-size 750000 --max-queries 100 --skip-rerank
```

## Summary Table Script Snippet

After generating JSON files, create a Markdown summary table:

```python
from __future__ import annotations

import json
from pathlib import Path


RESULTS_DIR = Path("benchmarks/results")
SYSTEM_NAMES = ["bm25", "vector", "hybrid_rrf", "hybrid_rerank"]


def fmt(value: float | int | None) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    return f"{value:.4f}" if value < 10 else f"{value:.2f}"


rows = [
    "| Corpus | System | Queries | NDCG@10 | MRR@10 | Recall@100 | p50 ms | p95 ms | Artifacts MiB |",
    "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
]

for path in sorted(RESULTS_DIR.glob("*.json")):
    data = json.loads(path.read_text(encoding="utf-8"))
    artifact_size = data["artifact_size_mib"]["total_compact_with_sq8_mib"]
    for system_name in SYSTEM_NAMES:
        metrics = data["systems"].get(system_name)
        if not metrics:
            continue
        rows.append(
            "| "
            + " | ".join(
                [
                    data["corpus_label"],
                    system_name,
                    str(metrics["queries"]),
                    fmt(metrics["ndcg_at_10"]),
                    fmt(metrics["mrr_at_10"]),
                    fmt(metrics["recall_at_100"]),
                    fmt(metrics["p50_ms"]),
                    fmt(metrics["p95_ms"]),
                    fmt(artifact_size),
                ]
            )
            + " |"
        )

summary = "\n".join(rows) + "\n"
(RESULTS_DIR / "summary.md").write_text(summary, encoding="utf-8")
print(summary)
```

Save it as `scripts/summarize_benchmarks.py` and run:

```bash
python scripts/summarize_benchmarks.py
```

## Interpreting Results

Expected patterns:

- BM25 should be strong on exact-match factual queries.
- Vector should help semantic/paraphrased queries.
- Hybrid RRF should usually improve recall and robustness.
- Cross-encoder reranking should improve `NDCG@10` and `MRR@10`, but latency will increase.
- Larger corpora usually improve recall opportunity, but latency and artifact size increase.

Red flags:

- Hybrid RRF worse than both BM25 and vector on most metrics.
- Vector search returns many missing docstore records.
- BM25, FAISS, and docstore document counts differ unexpectedly.
- Reranker improves nothing but adds large latency.
- Partial-corpus qrel filtering leaves too few eval queries.

## Suggested Result Files

Create:

```text
benchmarks/results/750k.json
benchmarks/results/1m.json
benchmarks/results/2m.json
benchmarks/results/3m.json
benchmarks/results/4m.json
benchmarks/results/summary.md
```

Each JSON should include:

```json
{
  "corpus_size": 750000,
  "artifact_size_mib": {
    "bm25_compact": 268.2,
    "vector_sq8": 209.4,
    "docstore": 174.8,
    "total": 652.4
  },
  "systems": {
    "bm25": {
      "ndcg_at_10": null,
      "mrr_at_10": null,
      "recall_at_100": null,
      "p50_ms": null,
      "p95_ms": null
    }
  }
}
```

## Resume Framing

Good bullet after benchmarks:

> Built and benchmarked a hybrid neural search engine over up to 4M MS MARCO passages using Tantivy BM25, FAISS vector retrieval, reciprocal rank fusion, and cross-encoder reranking; measured NDCG@10, MRR@10, Recall@100, p50/p95 latency, and index-size tradeoffs across multiple corpus scales.

That is stronger than simply saying the app is deployed.
