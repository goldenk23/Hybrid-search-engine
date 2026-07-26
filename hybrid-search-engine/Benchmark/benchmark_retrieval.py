"""Benchmark retrieval quality and latency for the hybrid search engine.

PRECONDITIONS
-------------
1. Build both indexes and the docstore first:
       python scripts/index_documents.py ...
       python scripts/index_vectors.py   ...

2. Create a cohort (run once, then commit the JSON):
       python Benchmark/cohort.py create \\
           --queries data/msmarco/queries.dev.small.tsv \\
           --qrels   data/msmarco/qrels.dev.small.tsv  \\
           --max-queries 500 \\
           --output  Benchmark/cohorts/dev500.json

3. Run the benchmark with explicit arguments (no hidden defaults):
       python Benchmark/benchmark_retrieval.py \\
           --manifest  data/indexes/artifact_manifest.json \\
           --cohort    Benchmark/cohorts/dev500.json \\
           --queries   data/msmarco/queries.dev.small.tsv \\
           --qrels     data/msmarco/qrels.dev.small.tsv  \\
           --vector-index data/indexes/vector.faiss \\
           --corpus-label 1M \\
           --corpus-size  1000000 \\
           --repeats 5

OUTPUTS
-------
    Benchmark/results/<corpus-label>.json
    Benchmark/results/<corpus-label>.md
    Benchmark/results/README.md

All three are written atomically (temp-file → rename).

WHAT IS RECORDED IN EVERY RESULT
---------------------------------
    schema_version, UTC timestamp, Git commit + dirty flag,
    sys.argv, platform, CPU count, cohort / query / qrel fingerprints,
    generation_id from the manifest, model name + revision,
    RRF k, per-system repeats, package versions.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sqlite3
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from statistics import median

import faiss
import numpy as np
import tantivy
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Benchmark.cohort import load_cohort
from src.config import BM25_INDEX_PATH, DOCSTORE_PATH
from src.evaluation.metrics import mrr_at_k, ndcg_at_k, recall_at_k
from src.indexing.artifact_state import load_json_required, sha256_path, write_json_atomic
from src.search.bm25 import BM25Search
from src.search.cross_encoder_reranker import CrossEncoderReranker
from src.search.hybrid_search import HybridSearchEngine
from src.search.vector import VectorSearch

SearchFn = Callable[[str, int], list[dict]]


# ------------------------------------------------------------------ guards

def _verify_manifest(manifest_path: Path, corpus_size: int) -> dict:
    """Load the manifest and refuse to benchmark if it is not complete."""
    manifest = load_json_required(manifest_path)

    if manifest.get("status") != "complete":
        raise RuntimeError(
            f"Benchmark requires a complete generation "
            f"(manifest status = {manifest.get('status')!r}). "
            "Finish indexing first."
        )

    counts = manifest.get("counts", {})
    for name in ("bm25", "vector", "docstore"):
        if name in counts and counts[name] != corpus_size:
            raise RuntimeError(
                f"Artifact count mismatch: manifest says {name}={counts[name]:,} "
                f"but --corpus-size={corpus_size:,}. "
                "Re-index or use the correct --corpus-size."
            )

    return manifest


def _verify_artifact_fingerprints(manifest: dict, vector_index_path: Path) -> None:
    """Compare on-disk fingerprints to the manifest receipt."""
    stored = manifest.get("artifact_sha256", {})
    to_check = {
        "vector": vector_index_path,
        "docstore": DOCSTORE_PATH,
    }
    if BM25_INDEX_PATH.exists():
        to_check["bm25"] = BM25_INDEX_PATH

    for name, path in to_check.items():
        if name not in stored:
            continue
        actual = sha256_path(path)
        if actual != stored[name]:
            raise RuntimeError(
                f"Artifact fingerprint mismatch for '{name}'.\n"
                f"  manifest: {stored[name]}\n"
                f"  on disk:  {actual}\n"
                "The index may have been rebuilt without updating the manifest."
            )


def _verify_sq8(sq8_path: Path, corpus_size: int) -> None:
    """If a SQ8-compressed index is supplied, confirm it matches the corpus size."""
    sq8 = faiss.read_index(str(sq8_path))
    if sq8.ntotal != corpus_size:
        raise RuntimeError(
            f"SQ8 index is stale: ntotal={sq8.ntotal:,} "
            f"but --corpus-size={corpus_size:,}."
        )


# ------------------------------------------------------------------ document counts

def _document_counts(vector_index_path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if BM25_INDEX_PATH.exists():
        counts["bm25"] = tantivy.Index.open(str(BM25_INDEX_PATH)).searcher().num_docs
    if vector_index_path.exists():
        counts["vector"] = faiss.read_index(str(vector_index_path)).ntotal
    if DOCSTORE_PATH.exists():
        with sqlite3.connect(DOCSTORE_PATH) as conn:
            counts["docstore"] = conn.execute(
                "SELECT COUNT(*) FROM documents"
            ).fetchone()[0]
    return counts


# ------------------------------------------------------------------ artifact sizes

def _size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _mib(b: int) -> float:
    return round(b / 1024 / 1024, 2)


def _artifact_sizes(vector_index_path: Path, sq8_path: Path | None) -> dict[str, float]:
    bm25_mib    = _mib(_size_bytes(BM25_INDEX_PATH))
    vector_mib  = _mib(_size_bytes(vector_index_path))
    sq8_mib     = _mib(_size_bytes(sq8_path)) if sq8_path else 0.0
    store_mib   = _mib(_size_bytes(DOCSTORE_PATH))
    return {
        "bm25_compact_mib": bm25_mib,
        "vector_faiss_mib": vector_mib,
        "vector_sq8_mib":   sq8_mib,
        "docstore_mib":     store_mib,
        "total_compact_with_sq8_mib": round(bm25_mib + sq8_mib + store_mib, 2),
    }


# ------------------------------------------------------------------ provenance

def _git_info() -> dict:
    def _run(*args) -> str:
        try:
            return subprocess.check_output(args, text=True,
                                           stderr=subprocess.DEVNULL).strip()
        except Exception:  # noqa: BLE001 — subprocess may fail for any reason; safe default
            return "unknown"

    commit = _run("git", "rev-parse", "--short", "HEAD")
    dirty  = _run("git", "status", "--porcelain") != ""
    return {"commit": commit, "dirty": dirty}


def _package_versions() -> dict[str, str]:
    pkgs = ["faiss-cpu", "sentence-transformers", "tantivy", "numpy", "tqdm"]
    out: dict[str, str] = {}
    for pkg in pkgs:
        try:
            out[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def _build_provenance(args: argparse.Namespace, manifest: dict,
                       cohort_path: Path) -> dict:
    return {
        "schema_version":     1,
        "utc_timestamp":      time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git":                _git_info(),
        "argv":               sys.argv,
        "platform":           platform.platform(),
        "cpu_count":          os.cpu_count(),
        "generation_id":      manifest.get("generation_id"),
        "embedding_model":    manifest.get("embedding_model"),
        "embedding_revision": manifest.get("embedding_revision"),
        "cohort_sha256":      sha256_path(cohort_path),
        "rrf_k":              args.rrf_k,
        "repeats":            args.repeats,
        "packages":           _package_versions(),
    }


# ------------------------------------------------------------------ evaluation

def _percentile(values: list[float], pct: float) -> float:
    return float(np.percentile(values, pct)) if values else 0.0


def _evaluate_system(
    name: str,
    search_fn: SearchFn,
    eval_queries: list[tuple[str, str, set[str]]],
    top_k: int,
    repeats: int,
    warmup: int = 3,
) -> dict:
    """Run search_fn for each query `repeats` times; report median latencies."""
    # Warmup: discard first few queries to avoid cold-start timing.
    for _, qtext, _ in eval_queries[:warmup]:
        search_fn(qtext, top_k)

    ndcg_scores: list[float]  = []
    mrr_scores: list[float]   = []
    recall_scores: list[float] = []
    # One list per run so we can take the per-query median across repeats.
    run_latencies: list[list[float]] = [[] for _ in range(repeats)]

    for _, qtext, relevant in tqdm(eval_queries, desc=f"Evaluating {name}"):
        for run in range(repeats):
            t0 = time.perf_counter()
            results = search_fn(qtext, top_k)
            run_latencies[run].append((time.perf_counter() - t0) * 1000)

        # Quality is stable across repeats — compute once from the last run.
        ranked = [str(r["id"]) for r in results]
        ndcg_scores.append(ndcg_at_k(ranked, relevant, k=10))
        mrr_scores.append(mrr_at_k(ranked, relevant, k=10))
        recall_scores.append(recall_at_k(ranked, relevant, k=100))

    # Per-query median latency across repeats, then p50/p95 over queries.
    median_latencies = [median(run_latencies[r][q] for r in range(repeats))
                        for q in range(len(eval_queries))]

    return {
        "queries":      len(eval_queries),
        "ndcg_at_10":   float(np.mean(ndcg_scores))   if ndcg_scores else 0.0,
        "mrr_at_10":    float(np.mean(mrr_scores))    if mrr_scores  else 0.0,
        "recall_at_100": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "p50_ms":       round(median(median_latencies), 2) if median_latencies else 0.0,
        "p95_ms":       round(_percentile(median_latencies, 95), 2),
    }


# ------------------------------------------------------------------ filtering

def _filter_queries(
    queries: dict[str, str],
    qrels:   dict[str, set[str]],
    indexed_doc_ids: set[str],
) -> list[tuple[str, str, set[str]]]:
    """Keep only queries that have at least one relevant doc in the index."""
    out = []
    for qid, qtext in queries.items():
        relevant = qrels.get(qid, set())
        available = relevant & indexed_doc_ids
        if available:
            out.append((qid, qtext, available))
    return out


def _load_indexed_ids() -> set[str]:
    if not DOCSTORE_PATH.exists():
        return set()
    with sqlite3.connect(DOCSTORE_PATH) as conn:
        return {str(row[0]) for row in conn.execute("SELECT id FROM documents")}


# ------------------------------------------------------------------ markdown

def _format(v: object, d: int = 4) -> str:
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


_DISPLAY = {
    "bm25":                              "BM25",
    "vector":                            "Vector",
    "hybrid_rrf":                        "Hybrid RRF (1.0 / 1.0)",
    "hybrid_rrf_bm25_050_vector_100":    "Weighted RRF (0.50 / 1.00)",
    "hybrid_rrf_bm25_025_vector_100":    "Weighted RRF (0.25 / 1.00)",
    "hybrid_rerank":                     "Hybrid + Cross-Encoder",
}


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    sep = ["---"] + ["---:" for _ in headers[1:]]
    lines = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(sep)     + " |"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def _result_to_markdown(result: dict) -> str:
    d  = result["document_counts"]
    sz = result["artifact_size_mib"]
    lines = [
        f"# Retrieval Benchmark: {result['corpus_label']}",
        "",
        "## Summary",
        "",
        f"- Corpus size: {result['corpus_size']:,}",
        f"- Evaluation queries: {result['eval_queries']:,}",
        f"- BM25 documents: {d.get('bm25', 0):,}",
        f"- Vector documents: {d.get('vector', 0):,}",
        f"- Docstore documents: {d.get('docstore', 0):,}",
        f"- Generation ID: {result['provenance'].get('generation_id', 'n/a')}",
        f"- Git commit: {result['provenance']['git'].get('commit', 'n/a')}"
          + (" (dirty)" if result['provenance']['git'].get('dirty') else ""),
        f"- Repeats per query: {result['provenance']['repeats']}",
        "",
        "## Benchmark Results",
        "",
    ]
    hdrs = ["System", "Queries", "NDCG@10", "MRR@10", "Recall@100",
            "p50 (ms)", "p95 (ms)"]
    rows = []
    for sname, m in result["systems"].items():
        rows.append([
            _DISPLAY.get(sname, sname),
            _format(m["queries"], 0),
            _format(m["ndcg_at_10"]),
            _format(m["mrr_at_10"]),
            _format(m["recall_at_100"]),
            _format(m["p50_ms"], 2),
            _format(m["p95_ms"], 2),
        ])
    lines += [_md_table(hdrs, rows), "", "## Storage", ""]
    shdrs = ["Artifact", "MiB"]
    srows = [
        ["BM25 index",         f"{sz.get('bm25_compact_mib', 0):,.2f}"],
        ["Vector FAISS",       f"{sz.get('vector_faiss_mib', 0):,.2f}"],
        ["Vector SQ8",         f"{sz.get('vector_sq8_mib', 0):,.2f}"],
        ["Docstore",           f"{sz.get('docstore_mib', 0):,.2f}"],
        ["Total (BM25+SQ8+DS)",f"{sz.get('total_compact_with_sq8_mib',0):,.2f}"],
    ]
    lines += [_md_table(shdrs, srows), ""]
    return "\n".join(lines)


def _readme_markdown(results: list[dict]) -> str:
    lines = [
        "# Hybrid Search Benchmark Results",
        "",
        "Auto-generated. Re-run a benchmark to update.",
        "",
    ]
    if not results:
        return "\n".join(lines + ["No results yet.", ""])

    hdrs = ["Corpus", "System", "Queries", "NDCG@10", "MRR@10",
            "Recall@100", "p50 (ms)", "p95 (ms)"]
    rows = []
    for r in results:
        for sname, m in r["systems"].items():
            rows.append([
                str(r["corpus_label"]),
                _DISPLAY.get(sname, sname),
                _format(m["queries"], 0),
                _format(m["ndcg_at_10"]),
                _format(m["mrr_at_10"]),
                _format(m["recall_at_100"]),
                _format(m["p50_ms"], 2),
                _format(m["p95_ms"], 2),
            ])
    lines += [_md_table(hdrs, rows), ""]
    return "\n".join(lines)


def _load_saved_results(output_dir: Path) -> list[dict]:
    out = []
    for p in sorted(output_dir.glob("*.json")):
        if p.stem == "README":
            continue
        try:
            data = __import__("json").loads(p.read_text(encoding="utf-8"))
            # Reject results whose label doesn't match their filename.
            if str(data.get("corpus_label")) == p.stem:
                out.append(data)
            else:
                print(f"Warning: skipping {p.name} — label mismatch")
        except Exception:  # noqa: BLE001 — JSON may be malformed; skip and warn
            print(f"Warning: skipping invalid JSON: {p.name}")
    return sorted(out, key=lambda r: (r.get("corpus_size", 0), r.get("corpus_label", "")))


# ------------------------------------------------------------------ CLI

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark retrieval systems.")

    # Every input is explicit — no hidden defaults that silently use a wrong file.
    parser.add_argument("--manifest",      type=Path, required=True,
                        help="Path to artifact_manifest.json")
    parser.add_argument("--cohort",        type=Path, required=True,
                        help="Path to cohort JSON created by Benchmark/cohort.py")
    parser.add_argument("--queries",       type=Path, required=True,
                        help="Path to queries TSV (used to validate cohort fingerprint)")
    parser.add_argument("--qrels",         type=Path, required=True,
                        help="Path to qrels TSV (used to validate cohort fingerprint)")
    parser.add_argument("--vector-index",  type=Path, required=True,
                        help="Path to the FAISS index the API serves")
    parser.add_argument("--corpus-label",  required=True,
                        help="Short label for this run, e.g. '1M'")
    parser.add_argument("--corpus-size",   type=int, required=True,
                        help="Expected document count in each artifact")
    parser.add_argument("--sq8-index",     type=Path, default=None,
                        help="Optional SQ8-compressed FAISS index to validate")
    parser.add_argument("--rrf-k",         type=int, default=60)
    parser.add_argument("--repeats",       type=int, default=5,
                        help="Times to repeat each query for stable latency")
    parser.add_argument("--rerank-queries", type=int, default=100,
                        help="Number of cohort queries to use for reranking")
    parser.add_argument("--skip-rerank",   action="store_true")
    parser.add_argument("--output-dir",    type=Path,
                        default=project_root / "Benchmark" / "results")
    return parser.parse_args()


# ------------------------------------------------------------------ main

def main() -> None:
    args = _parse_args()

    # ---- guards ----
    manifest = _verify_manifest(args.manifest, args.corpus_size)
    _verify_artifact_fingerprints(manifest, args.vector_index)
    if args.sq8_index:
        _verify_sq8(args.sq8_index, args.corpus_size)

    # ---- cohort ----
    queries, qrels = load_cohort(args.cohort, args.queries, args.qrels)
    indexed_ids    = _load_indexed_ids()
    eval_queries   = _filter_queries(queries, qrels, indexed_ids)

    if not eval_queries:
        raise RuntimeError("No eval queries have relevant docs in the current index.")
    print(f"Cohort: {len(queries)} queries  →  {len(eval_queries)} with relevant docs indexed")

    # ---- load indexes ----
    bm25   = BM25Search()
    vector = VectorSearch(index_path=args.vector_index)
    hybrid = HybridSearchEngine(bm25_search=bm25, vector_search=vector)

    # ---- run systems ----
    systems: dict[str, dict] = {}

    systems["bm25"] = _evaluate_system(
        "bm25",
        lambda q, k: bm25.search(q, top_k=k),
        eval_queries, top_k=100, repeats=args.repeats,
    )
    systems["vector"] = _evaluate_system(
        "vector",
        lambda q, k: vector.search(q, top_k=k),
        eval_queries, top_k=100, repeats=args.repeats,
    )
    systems["hybrid_rrf"] = _evaluate_system(
        "hybrid_rrf",
        lambda q, k: hybrid.search(q, top_k=k, rrf_k=args.rrf_k),
        eval_queries, top_k=100, repeats=args.repeats,
    )
    systems["hybrid_rrf_bm25_050_vector_100"] = _evaluate_system(
        "hybrid_rrf_bm25_050_vector_100",
        lambda q, k: hybrid.search(q, top_k=k, bm25_weight=0.50,
                                   vector_weight=1.00, rrf_k=args.rrf_k),
        eval_queries, top_k=100, repeats=args.repeats,
    )
    systems["hybrid_rrf_bm25_025_vector_100"] = _evaluate_system(
        "hybrid_rrf_bm25_025_vector_100",
        lambda q, k: hybrid.search(q, top_k=k, bm25_weight=0.25,
                                   vector_weight=1.00, rrf_k=args.rrf_k),
        eval_queries, top_k=100, repeats=args.repeats,
    )

    if not args.skip_rerank:
        reranker       = CrossEncoderReranker()
        rerank_queries = eval_queries[:args.rerank_queries]

        def _rerank_fn(q: str, k: int) -> list[dict]:
            candidates = hybrid.search(q, top_k=100, rrf_k=args.rrf_k)
            return reranker.rerank(query=q, candidates=candidates,
                                   top_k=min(k, 10), max_candidates=50)

        systems["hybrid_rerank"] = _evaluate_system(
            "hybrid_rerank", _rerank_fn, rerank_queries,
            top_k=10, repeats=args.repeats, warmup=3,
        )

    # ---- assemble result ----
    result = {
        "schema_version": 1,
        "corpus_label":   args.corpus_label,
        "corpus_size":    args.corpus_size,
        "eval_queries":   len(eval_queries),
        "document_counts": _document_counts(args.vector_index),
        "artifact_size_mib": _artifact_sizes(args.vector_index, args.sq8_index),
        "systems":        systems,
        "provenance":     _build_provenance(args, manifest, args.cohort),
    }

    # ---- atomic output ----
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / f"{args.corpus_label}.json"
    md_path   = args.output_dir / f"{args.corpus_label}.md"
    readme    = args.output_dir / "README.md"

    write_json_atomic(json_path, result)
    # Write markdown atomically too.
    tmp_md = md_path.with_suffix(".md.tmp")
    tmp_md.write_text(_result_to_markdown(result), encoding="utf-8")
    os.replace(tmp_md, md_path)

    saved = _load_saved_results(args.output_dir)
    tmp_rm = readme.with_suffix(".md.tmp")
    tmp_rm.write_text(_readme_markdown(saved), encoding="utf-8")
    os.replace(tmp_rm, readme)

    print(f"Saved:  {json_path}")
    print(f"Report: {md_path}")
    print(f"README: {readme}")


if __name__ == "__main__":
    main()
