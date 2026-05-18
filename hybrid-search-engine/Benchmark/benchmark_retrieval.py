"""Benchmark retrieval quality and latency for the hybrid search engine.

WHAT THIS SCRIPT MEASURES
-------------------------

For the currently built indexes, this script evaluates:
    1. BM25 search
    2. Vector search
    3. Hybrid RRF search
    4. Hybrid RRF + cross-encoder reranking, unless --skip-rerank is used

It calculates and saves:
    - NDCG@10
    - MRR@10
    - Recall@100
    - p50 latency in milliseconds
    - p95 latency in milliseconds
    - BM25/vector/docstore artifact sizes
    - BM25/vector/docstore document counts
    - A presentation-ready Markdown benchmark table

IMPORTANT PRECONDITION
----------------------

Build the indexes first. This script does not index documents.

Example for a 750K corpus:
    python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 750000 --reset
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 750000 --reset
    python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss

The script filters evaluation queries so only queries with at least one relevant
document in the current docstore are used. This is important for partial-corpus
benchmarks such as 750K, 1M, 2M, 3M, or 4M.

USAGE COMMANDS
--------------

Fast smoke benchmark without reranking:
    python Benchmark/benchmark_retrieval.py --corpus-label 750k --corpus-size 750000 --max-queries 100 --skip-rerank

Standard 750K benchmark:
    python Benchmark/benchmark_retrieval.py --corpus-label 1M --corpus-size 1000000 --max-queries 500 --rerank-queries 100

Standard 1M benchmark:
    python Benchmark/benchmark_retrieval.py --corpus-label 1m --corpus-size 1000000 --max-queries 500 --rerank-queries 100

Standard 2M benchmark:
    python Benchmark/benchmark_retrieval.py --corpus-label 2m --corpus-size 2000000 --max-queries 500 --rerank-queries 100

Large 3M benchmark with fewer reranker queries:
    python Benchmark/benchmark_retrieval.py --corpus-label 3m --corpus-size 3000000 --max-queries 500 --rerank-queries 50

Large 4M benchmark with fewer reranker queries:
    python Benchmark/benchmark_retrieval.py --corpus-label 4m --corpus-size 4000000 --max-queries 500 --rerank-queries 50

Use custom evaluation subset files:
    python Benchmark/benchmark_retrieval.py --corpus-label 750k --corpus-size 750000 \
        --queries data/msmarco/queries.dev.small.tsv \
        --qrels data/msmarco/qrels.dev.small.tsv

Save results to a custom folder:
    python Benchmark/benchmark_retrieval.py --corpus-label 750k --corpus-size 750000 \
        --output-dir Benchmark/results

OUTPUT
------

By default, results are saved as:
    Benchmark/results/<corpus-label>.json
    Benchmark/results/<corpus-label>.md
    Benchmark/results/README.md

Example:
    Benchmark/results/750k.json
    Benchmark/results/750k.md

NOTES
-----

- Reranking is much slower than BM25/vector/hybrid search. Use --skip-rerank
  for quick iteration.
- The first run may be slower because FAISS/SentenceTransformer/CrossEncoder
  models and indexes need to load.
- For fair comparisons, use the same query subset for every system and every
  corpus size.
"""

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


def format_system_name(name: str) -> str:
    display_names = {
        "bm25": "BM25",
        "vector": "Vector",
        "hybrid_rrf": "Hybrid RRF",
        "hybrid_rrf_bm25_050_vector_100": "Weighted RRF (BM25 0.50, Vector 1.00)",
        "hybrid_rrf_bm25_025_vector_100": "Weighted RRF (BM25 0.25, Vector 1.00)",
        "hybrid_rerank": "Hybrid RRF + Cross-Encoder",
    }
    return display_names.get(name, name.replace("_", " ").title())


def format_number(value: object, decimals: int = 4) -> str:
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def build_markdown_table(headers: list[str], rows: list[list[str]], right_align_columns: bool = False) -> str:
    if not headers or not rows:
        return ""

    separator = ["---"]
    separator.extend("---:" if right_align_columns else "---" for _ in headers[1:])

    table_lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(separator) + " |",
    ]
    for row in rows:
        table_lines.append("| " + " | ".join(row) + " |")
    return "\n".join(table_lines)


def benchmark_markdown(result: dict) -> str:
    systems = result["systems"]
    artifact_sizes = result["artifact_size_mib"]
    document_counts_result = result["document_counts"]

    lines = [
        f"# Retrieval Benchmark: {result['corpus_label']}",
        "",
        "## Summary",
        "",
        f"- Corpus size: {result['corpus_size']:,} documents",
        f"- Evaluation queries: {result['eval_queries']:,}",
        f"- BM25 documents: {document_counts_result.get('bm25', 0):,}",
        f"- Vector documents: {document_counts_result.get('vector', 0):,}",
        f"- Docstore documents: {document_counts_result.get('docstore', 0):,}",
        f"- Compact index size with SQ8: {artifact_sizes.get('total_compact_with_sq8_mib', 0):,.2f} MiB",
        "",
        "## Benchmark Results",
        "",
    ]

    benchmark_headers = ["System", "Queries", "NDCG@10", "MRR@10", "Recall@100", "p50 Latency (ms)", "p95 Latency (ms)"]
    benchmark_rows = []
    for system_name, metrics in systems.items():
        benchmark_rows.append([
            format_system_name(system_name),
            format_number(metrics["queries"], 0),
            format_number(metrics["ndcg_at_10"]),
            format_number(metrics["mrr_at_10"]),
            format_number(metrics["recall_at_100"]),
            format_number(metrics["p50_ms"], 2),
            format_number(metrics["p95_ms"], 2),
        ])

    lines.append(build_markdown_table(benchmark_headers, benchmark_rows, right_align_columns=True))
    lines.extend(
        [
            "",
            "## Storage Footprint",
            "",
        ]
    )

    storage_headers = ["Artifact", "Size (MiB)"]
    storage_rows = [
        ["BM25 compact index", f"{artifact_sizes.get('bm25_compact_mib', 0):,.2f}"],
        ["Vector FAISS index", f"{artifact_sizes.get('vector_faiss_mib', 0):,.2f}"],
        ["Vector SQ8 FAISS index", f"{artifact_sizes.get('vector_sq8_mib', 0):,.2f}"],
        ["SQLite docstore", f"{artifact_sizes.get('docstore_mib', 0):,.2f}"],
        ["Total compact with SQ8", f"{artifact_sizes.get('total_compact_with_sq8_mib', 0):,.2f}"],
    ]

    lines.append(build_markdown_table(storage_headers, storage_rows, right_align_columns=True))
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Higher NDCG@10, MRR@10, and Recall@100 indicate better retrieval quality.",
            "- Lower p50 and p95 latency indicate faster query response times.",
            "- Hybrid reranking may use fewer queries when `--rerank-queries` is lower than `--max-queries`.",
            "",
        ]
    )
    return "\n".join(lines)


def load_saved_benchmark_results(output_dir: Path) -> list[dict]:
    results = []
    for json_path in sorted(output_dir.glob("*.json")):
        try:
            results.append(json.loads(json_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            print(f"Skipping invalid benchmark JSON: {json_path}")
    return sorted(results, key=lambda item: (item.get("corpus_size", 0), item.get("corpus_label", "")))


def benchmark_readme(results: list[dict]) -> str:
    lines = [
        "# Hybrid Search Benchmark Results",
        "",
        "This file is updated automatically after each benchmark run.",
        "",
    ]

    if not results:
        lines.extend(["No benchmark results are available yet.", ""])
        return "\n".join(lines)

    overview_headers = [
        "Corpus",
        "System",
        "Queries",
        "NDCG@10",
        "MRR@10",
        "Recall@100",
        "p50 Latency (ms)",
        "p95 Latency (ms)",
        "Compact Size (MiB)",
    ]
    overview_rows = []

    for result in results:
        compact_size = f"{result['artifact_size_mib'].get('total_compact_with_sq8_mib', 0):,.2f}"
        for system_name, metrics in result["systems"].items():
            overview_rows.append(
                [
                    str(result["corpus_label"]),
                    format_system_name(system_name),
                    format_number(metrics["queries"], 0),
                    format_number(metrics["ndcg_at_10"]),
                    format_number(metrics["mrr_at_10"]),
                    format_number(metrics["recall_at_100"]),
                    format_number(metrics["p50_ms"], 2),
                    format_number(metrics["p95_ms"], 2),
                    compact_size,
                ]
            )

    lines.extend(
        [
            "## Presentation Summary Table",
            "",
            build_markdown_table(overview_headers, overview_rows, right_align_columns=True),
            "",
            "## Result Files",
            "",
            "| Corpus | JSON | Report |",
            "|---|---|---|",
        ]
    )

    for result in results:
        corpus_label = result["corpus_label"]
        lines.append(f"| {corpus_label} | `{corpus_label}.json` | `{corpus_label}.md` |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Re-running a benchmark with the same `--corpus-label` updates that corpus JSON and report.",
            "- This README is rebuilt from all JSON files in this folder after every run.",
            "- Higher retrieval metrics are better; lower latency metrics are better.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval systems.")
    parser.add_argument("--corpus-label", required=True, help="Example: 750k, 1m, 2m")
    parser.add_argument("--corpus-size", type=int, required=True, help="Indexed document target")
    parser.add_argument("--queries", type=Path, default=DATA_DIR / "msmarco" / "queries.dev.small.tsv")
    parser.add_argument("--qrels", type=Path, default=DATA_DIR / "msmarco" / "qrels.dev.small.tsv")
    parser.add_argument("--max-queries", type=int, default=500)
    parser.add_argument("--rerank-queries", type=int, default=100)
    parser.add_argument("--skip-rerank", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=project_root / "Benchmark" / "results")
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
        "hybrid_rrf_bm25_050_vector_100": evaluate_system(
            "hybrid_rrf_bm25_050_vector_100",
            lambda query, top_k: hybrid.search(
                query,
                top_k=top_k,
                bm25_weight=0.50,
                vector_weight=1.00,
            ),
            eval_queries,
            top_k=100,
        ),
        "hybrid_rrf_bm25_025_vector_100": evaluate_system(
            "hybrid_rrf_bm25_025_vector_100",
            lambda query, top_k: hybrid.search(
                query,
                top_k=top_k,
                bm25_weight=0.25,
                vector_weight=1.00,
            ),
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
    markdown_path = args.output_dir / f"{args.corpus_label}.md"
    readme_path = args.output_dir / "README.md"
    result_exists = output_path.exists() or markdown_path.exists()
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    markdown_path.write_text(benchmark_markdown(result), encoding="utf-8")
    readme_path.write_text(benchmark_readme(load_saved_benchmark_results(args.output_dir)), encoding="utf-8")
    print(json.dumps(result, indent=2))
    action = "Updated" if result_exists else "Saved"
    print(f"{action}: {output_path}")
    print(f"{action} report: {markdown_path}")
    print(f"Updated results README: {readme_path}")


if __name__ == "__main__":
    main()
