"""
ANN speed-vs-quality experiment: FlatIP (exact) vs HNSW at efSearch 32/64/128.

THE POINT
---------
"HNSW is faster" is not evidence — it's a claim. This script *measures* the
actual trade-off on your specific corpus and cohort so you can report honest
numbers instead of guessing.

PRECONDITIONS
-------------
1. You have a built FlatIP index:
       python scripts/index_vectors.py ...

2. You have built an HNSW index from it:
       python scripts/build_hnsw_index.py \
           --input  data/indexes/vector.faiss \
           --output data/indexes/vector.hnsw.faiss

3. You have a cohort:
       python Benchmark/cohort.py create \
           --queries data/msmarco/queries.dev.small.tsv \
           --qrels   data/msmarco/qrels.dev.small.tsv  \
           --max-queries 500 \
           --output  Benchmark/cohorts/dev500.json

USAGE
-----
    python Benchmark/benchmark_ann.py \
        --flat-index  data/indexes/vector.faiss \
        --hnsw-index  data/indexes/vector.hnsw.faiss \
        --cohort      Benchmark/cohorts/dev500.json \
        --queries     data/msmarco/queries.dev.small.tsv \
        --qrels       data/msmarco/qrels.dev.small.tsv \
        --output      Benchmark/results/ann_comparison.json

WHAT IS REPORTED
----------------
For each system (FlatIP, HNSW-32, HNSW-64, HNSW-128):
    NDCG@10, Recall@100, p50 latency (ms), p95 latency (ms),
    index size on disk (MiB), approximate RAM usage (MiB)

All systems are evaluated on the same cohort and the same queries.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path
from statistics import median

import faiss
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Benchmark.cohort import load_cohort
from src.config import DOCSTORE_PATH, EMBEDDING_MODEL_NAME
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.indexing.artifact_state import write_json_atomic

# ------------------------------------------------------------------ helpers

def _load_indexed_ids() -> set[str]:
    if not DOCSTORE_PATH.exists():
        return set()
    with sqlite3.connect(DOCSTORE_PATH) as conn:
        return {str(r[0]) for r in conn.execute("SELECT id FROM documents")}


def _filter_queries(queries, qrels, indexed_ids):
    out = []
    for qid, qtext in queries.items():
        relevant = qrels.get(qid, set())
        available = relevant & indexed_ids
        if available:
            out.append((qid, qtext, available))
    return out


def _size_mib(path: Path) -> float:
    return round(path.stat().st_size / 1024 / 1024, 2) if path.exists() else 0.0


def _ram_mib(index: faiss.Index) -> float:
    """Rough estimate: ntotal * dimension * 4 bytes for FlatIP;
    HNSW adds graph overhead but this gives a lower bound."""
    try:
        d = index.d
        return round(index.ntotal * d * 4 / 1024 / 1024, 2)
    except Exception:  # noqa: BLE001 — FAISS index may be unloaded; safe default
        return 0.0


def _percentile(values: list[float], p: float) -> float:
    return float(np.percentile(values, p)) if values else 0.0


# ------------------------------------------------------------------ search runner

def _run_system(
    name: str,
    index: faiss.Index,
    model,
    eval_queries: list,
    top_k: int,
    repeats: int,
) -> dict:
    ndcg_scores, recall_scores = [], []
    run_latencies = [[] for _ in range(repeats)]

    for _, qtext, relevant in tqdm(eval_queries, desc=name):
        qvec = model.encode(
            [qtext],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")

        for run in range(repeats):
            t0 = time.perf_counter()
            _scores, faiss_ids = index.search(qvec, top_k)
            run_latencies[run].append((time.perf_counter() - t0) * 1000)

        ranked = [
            str(int(fid))
            for fid in faiss_ids[0]
            if fid != -1
        ]
        ndcg_scores.append(ndcg_at_k(ranked, relevant, k=10))
        recall_scores.append(recall_at_k(ranked, relevant, k=100))

    median_latencies = [
        median(run_latencies[r][q] for r in range(repeats))
        for q in range(len(eval_queries))
    ]

    return {
        "queries":      len(eval_queries),
        "ndcg_at_10":   round(float(np.mean(ndcg_scores)), 4)   if ndcg_scores else 0.0,
        "recall_at_100": round(float(np.mean(recall_scores)), 4) if recall_scores else 0.0,
        "p50_ms":       round(median(median_latencies), 2)       if median_latencies else 0.0,
        "p95_ms":       round(_percentile(median_latencies, 95), 2),
    }


# ------------------------------------------------------------------ HNSW wrapper

def _set_ef_search(index: faiss.Index, ef: int) -> None:
    """Set efSearch on the HNSW sub-index inside an IDMap2 wrapper."""
    inner = faiss.downcast_index(index.index)
    inner.hnsw.efSearch = ef


# ------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare FlatIP (exact) vs HNSW at efSearch 32/64/128."
    )
    parser.add_argument("--flat-index",  type=Path, required=True,
                        help="Path to FlatIP+IDMap2 FAISS index")
    parser.add_argument("--hnsw-index",  type=Path, required=True,
                        help="Path to HNSW+IDMap2 FAISS index")
    parser.add_argument("--cohort",      type=Path, required=True)
    parser.add_argument("--queries",     type=Path, required=True)
    parser.add_argument("--qrels",       type=Path, required=True)
    parser.add_argument("--top-k",       type=int, default=100)
    parser.add_argument("--repeats",     type=int, default=5,
                        help="Query repetitions for stable latency")
    parser.add_argument("--output",      type=Path,
                        default=project_root / "Benchmark" / "results" / "ann_comparison.json")
    args = parser.parse_args()

    # ---- guards ----
    for p, label in [(args.flat_index, "flat"), (args.hnsw_index, "hnsw")]:
        if not p.exists():
            raise FileNotFoundError(f"{label} index not found: {p}")

    # ---- cohort ----
    queries, qrels = load_cohort(args.cohort, args.queries, args.qrels)
    indexed_ids    = _load_indexed_ids()
    eval_queries   = _filter_queries(queries, qrels, indexed_ids)
    if not eval_queries:
        raise RuntimeError("No eval queries have relevant docs in the current index.")
    print(f"Cohort: {len(queries)} queries → {len(eval_queries)} with relevant docs indexed")

    # ---- load model ----
    from sentence_transformers import SentenceTransformer
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    # ---- load indexes ----
    print("Loading indexes…")
    flat_index = faiss.read_index(str(args.flat_index))
    hnsw_index = faiss.read_index(str(args.hnsw_index))

    # ---- run experiments ----
    systems: dict[str, dict] = {}

    systems["flat_ip"] = _run_system(
        "FlatIP (exact)", flat_index, model, eval_queries,
        top_k=args.top_k, repeats=args.repeats,
    )

    for ef in (32, 64, 128):
        _set_ef_search(hnsw_index, ef)
        systems[f"hnsw_ef{ef}"] = _run_system(
            f"HNSW efSearch={ef}", hnsw_index, model, eval_queries,
            top_k=args.top_k, repeats=args.repeats,
        )

    # ---- artifact sizes ----
    sizes = {
        "flat_ip":   _size_mib(args.flat_index),
        "hnsw":      _size_mib(args.hnsw_index),
    }
    ram = {
        "flat_ip":   _ram_mib(flat_index),
        "hnsw":      _ram_mib(hnsw_index),
    }

    result = {
        "schema_version":  1,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "repeats":         args.repeats,
        "top_k":           args.top_k,
        "eval_queries":    len(eval_queries),
        "index_size_mib":  sizes,
        "ram_estimate_mib": ram,
        "systems":         systems,
    }

    # ---- print summary ----
    print("\n" + "=" * 72)
    print(f"{'System':<22} {'NDCG@10':>8} {'Recall@100':>11} {'p50 ms':>8} {'p95 ms':>8} {'MiB':>7}")
    print("-" * 72)
    labels = {
        "flat_ip":    "FlatIP (exact)",
        "hnsw_ef32":  "HNSW ef=32",
        "hnsw_ef64":  "HNSW ef=64",
        "hnsw_ef128": "HNSW ef=128",
    }
    for key, label in labels.items():
        m = systems[key]
        sz = sizes.get("hnsw" if "hnsw" in key else "flat_ip", 0)
        print(
            f"{label:<22} {m['ndcg_at_10']:>8.4f} {m['recall_at_100']:>11.4f} "
            f"{m['p50_ms']:>8.2f} {m['p95_ms']:>8.2f} {sz:>7.1f}"
        )
    print("=" * 72)

    write_json_atomic(args.output, result)
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
