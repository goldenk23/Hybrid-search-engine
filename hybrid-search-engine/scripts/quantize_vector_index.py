"""
Create a smaller FAISS vector index without overwriting the original.

Usage:
    python scripts/quantize_vector_index.py
    python scripts/quantize_vector_index.py --method fp16
    python scripts/quantize_vector_index.py --input data/indexes/vector.faiss --output data/indexes/vector.sq8.faiss
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import faiss
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import INDEX_DIR


METHODS = {
    "sq8": faiss.ScalarQuantizer.QT_8bit,
    "fp16": faiss.ScalarQuantizer.QT_fp16,
}


def reconstruct_batch(index: faiss.Index, start: int, size: int) -> np.ndarray:
    vectors = np.empty((size, index.d), dtype="float32")
    for offset in range(size):
        index.reconstruct(start + offset, vectors[offset])
    return vectors


def sample_vectors(index: faiss.Index, sample_size: int) -> np.ndarray:
    sample_size = min(sample_size, index.ntotal)
    positions = np.linspace(0, index.ntotal - 1, sample_size, dtype=np.int64)
    vectors = np.empty((sample_size, index.d), dtype="float32")
    for out_pos, index_pos in enumerate(positions):
        index.reconstruct(int(index_pos), vectors[out_pos])
    return vectors


def quantize_index(
    input_path: Path,
    output_path: Path,
    method: str,
    train_size: int,
    batch_size: int,
) -> None:
    if output_path.exists():
        raise FileExistsError(f"Output already exists, refusing to overwrite: {output_path}")

    original = faiss.read_index(str(input_path))
    base = faiss.downcast_index(original.index) if hasattr(original, "index") else original

    print(f"Loaded: {input_path}")
    print(f"Type: {type(original).__name__}, vectors: {original.ntotal:,}, dim: {original.d}")

    quantizer = faiss.IndexScalarQuantizer(
        original.d,
        METHODS[method],
        original.metric_type,
    )
    quantized = faiss.IndexIDMap2(quantizer) if hasattr(original, "id_map") else quantizer

    print(f"Training {method} quantizer on {min(train_size, original.ntotal):,} vectors...")
    quantizer.train(sample_vectors(base, train_size))

    ids = faiss.vector_to_array(original.id_map) if hasattr(original, "id_map") else None

    print("Adding vectors to quantized index...")
    for start in range(0, original.ntotal, batch_size):
        size = min(batch_size, original.ntotal - start)
        vectors = reconstruct_batch(base, start, size)

        if ids is None:
            quantized.add(vectors)
        else:
            quantized.add_with_ids(vectors, ids[start : start + size])

        print(f"  added {start + size:,}/{original.ntotal:,}", end="\r")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(quantized, str(output_path))
    print(f"\nSaved: {output_path}")
    print(f"Original MB: {input_path.stat().st_size / 1024 / 1024:.1f}")
    print(f"Quantized MB: {output_path.stat().st_size / 1024 / 1024:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quantize a FAISS vector index.")
    parser.add_argument("--input", type=Path, default=INDEX_DIR / "vector.faiss")
    parser.add_argument("--output", type=Path, default=INDEX_DIR / "vector.sq8.faiss")
    parser.add_argument("--method", choices=METHODS, default="sq8")
    parser.add_argument("--train-size", type=int, default=100_000)
    parser.add_argument("--batch-size", type=int, default=20_000)
    args = parser.parse_args()

    quantize_index(
        input_path=args.input,
        output_path=args.output,
        method=args.method,
        train_size=args.train_size,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
