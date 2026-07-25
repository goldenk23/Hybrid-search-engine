"""
Build an HNSW approximate-nearest-neighbour index from an existing FlatIP index.

WHY HNSW?
---------
Your current IndexFlatIP scans every vector for every query — exact results,
but O(n) per query. As the corpus grows, this gets slower linearly.

HNSW (Hierarchical Navigable Small World) builds a layered graph so search
can "hop" toward the answer instead of scanning everything. Typical speedup:
10-100x at efSearch=64, with 1-3% quality loss on NDCG@10.

This script reads the vectors out of your existing FlatIP+IDMap2 index and
adds them into a new HNSWFlat+IDMap2 index. You do NOT need to re-embed —
the vectors are already there.

USAGE
-----
    python scripts/build_hnsw_index.py \
        --input  data/indexes/vector.faiss \
        --output data/indexes/vector.hnsw.faiss

    # optional tuning (defaults are good starting points):
    python scripts/build_hnsw_index.py \
        --input  data/indexes/vector.faiss \
        --output data/indexes/vector.hnsw.faiss \
        --M 32 --ef-construction 200

PARAMETERS
----------
M               Controls the number of bi-directional links per node.
                Higher M = better recall + more RAM + slower build.
                16-64 is typical; 32 is a good default.

efConstruction  Controls how carefully the graph is built.
                Higher = better recall + slower build time.
                100-400 is typical; 200 is a good default.

efSearch is set at search time, not build time. See Benchmark/benchmark_ann.py.

OUTPUT
------
    data/indexes/vector.hnsw.faiss   — new HNSW index, same IDs as the source
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faiss
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import INDEX_DIR, VECTOR_INDEX_PATH


def build_hnsw(
    input_path: Path,
    output_path: Path,
    M: int = 32,
    ef_construction: int = 200,
) -> None:
    """Read vectors + IDs from a FlatIP+IDMap2 index and write an HNSW+IDMap2 index."""
    if not input_path.exists():
        raise FileNotFoundError(f"Source index not found: {input_path}")

    print(f"Loading source index: {input_path}")
    src = faiss.read_index(str(input_path))
    n = src.ntotal
    d = src.d
    print(f"  {n:,} vectors, dimension {d}")

    # ---- extract vectors and IDs from the source IndexIDMap2 ----
    # reconstruct_n pulls all n vectors out of the index in one shot.
    print("Extracting vectors…")
    vectors = np.zeros((n, d), dtype="float32")
    src.reconstruct_n(0, n, vectors)

    # Pull the stored IDs from the IDMap2 wrapper.
    ids = faiss.vector_to_array(src.id_map).copy()
    print(f"  Extracted {len(ids):,} IDs")

    # ---- build the HNSW index ----
    print(f"Building HNSW (M={M}, efConstruction={ef_construction})…")
    t0 = time.perf_counter()

    # IndexHNSWFlat with METRIC_INNER_PRODUCT — vectors are L2-normalised so
    # inner product == cosine similarity, matching the FlatIP index behaviour.
    base = faiss.IndexHNSWFlat(d, M, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = ef_construction

    # Wrap with IDMap2 to keep the same real MS MARCO passage IDs.
    hnsw_index = faiss.IndexIDMap2(base)
    hnsw_index.add_with_ids(vectors, ids)

    elapsed = time.perf_counter() - t0
    print(f"  Built in {elapsed:.1f}s  ({hnsw_index.ntotal:,} vectors indexed)")

    # ---- save ----
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Write atomically: temp file → rename so a crash never leaves a partial index.
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    faiss.write_index(hnsw_index, str(tmp))
    import os
    os.replace(tmp, output_path)
    size_mib = output_path.stat().st_size / 1024 / 1024
    print(f"Saved: {output_path}  ({size_mib:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an HNSW index from an existing FlatIP vector index."
    )
    parser.add_argument(
        "--input", type=Path,
        default=VECTOR_INDEX_PATH,
        help="Source FlatIP+IDMap2 FAISS index",
    )
    parser.add_argument(
        "--output", type=Path,
        default=INDEX_DIR / "vector.hnsw.faiss",
        help="Output HNSW+IDMap2 FAISS index",
    )
    parser.add_argument(
        "--M", type=int, default=32,
        help="HNSW M parameter (bi-directional links per node, default 32)",
    )
    parser.add_argument(
        "--ef-construction", type=int, default=200,
        help="HNSW efConstruction (build-time search effort, default 200)",
    )
    args = parser.parse_args()

    build_hnsw(
        input_path=args.input,
        output_path=args.output,
        M=args.M,
        ef_construction=args.ef_construction,
    )


if __name__ == "__main__":
    main()
