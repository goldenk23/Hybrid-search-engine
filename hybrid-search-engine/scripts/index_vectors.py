"""
Build a full-corpus FAISS vector index for MS MARCO passages.

This script is intentionally separate from scripts/index_documents.py because
embedding millions of passages is a heavier long-running job than BM25.

SAFE STAGED USAGE (RECOMMENDED):
================================

Stage 1: Test with small dataset and reset (10K documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 10000 --reset
    - Validates embedding pipeline and index creation
    - Clears any previous indexing state
    - Tests the full checkpoint/resume flow

Stage 2: Medium dataset with resume enabled (100K documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 100000
    - Resume automatically continues from last checkpoint
    - If interrupted, re-run same command to resume
    - Verifies checkpointing works correctly at scale

Stage 3: Large dataset with resume (1M documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 1000000
    - Resume continues from previous checkpoint
    - Can be interrupted and resumed safely

Full Corpus Indexing (all documents, no limit):
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv
    - Indexes entire collection.tsv
    - Resume automatic; safe to restart on interruption

MONITORING AND STATUS:
=======================

Check progress without indexing:
    python scripts/index_vectors.py --status
    - Shows: index file size, total documents indexed, last document ID, timestamp

ADVANCED OPTIONS:
==================

Reset and start over (delete index and checkpoint):
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --reset

Ignore checkpoint and continue from start (without deletion):
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --no-resume

Custom batch sizes (for memory tuning):
    python scripts/index_vectors.py --batch-size 256 --encode-batch-size 32

OUTPUT FILES:
==============

    data/indexes/vector.faiss           - FAISS index (persisted incrementally)
    data/indexes/vector_checkpoint.json - Resumption metadata with progress info
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, EMBEDDING_MODEL_NAME, INDEX_DIR
from src.indexing.pipeline import load_msmarco_passages


DEFAULT_INDEX_PATH = INDEX_DIR / "vector.faiss"
DEFAULT_CHECKPOINT_PATH = INDEX_DIR / "vector_checkpoint.json"


def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load vector indexing checkpoint metadata if it exists."""
    if not checkpoint_path.exists():
        return None

    with checkpoint_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_checkpoint(
    checkpoint_path: Path,
    *,
    total_documents_indexed: int,
    last_document_id: str | None,
    collection_path: Path,
    index_path: Path,
    model_name: str,
) -> None:
    """Save progress only after the FAISS index has been written to disk."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "total_documents_indexed": total_documents_indexed,
        "last_document_id": last_document_id,
        "collection_path": str(collection_path),
        "index_path": str(index_path),
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with checkpoint_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def clear_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint metadata if present."""
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def remove_index(index_path: Path) -> None:
    """Remove the existing vector index file if present."""
    if index_path.exists():
        index_path.unlink()


def create_faiss_index(dimension: int) -> faiss.Index:
    """
    Create an exact cosine-similarity index.

    Embeddings are normalized by SentenceTransformer, so inner product is
    equivalent to cosine similarity. IndexIDMap2 stores MS MARCO passage IDs
    directly inside FAISS, avoiding a huge Python doc_ids mapping.
    """
    base_index = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap2(base_index)


def encode_texts(
    model: Any,
    texts: list[str],
    encode_batch_size: int,
) -> np.ndarray:
    """Encode and normalize one batch of document texts."""
    embeddings = model.encode(
        texts,
        batch_size=encode_batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return embeddings.astype("float32")


def add_batch_to_index(
    index: faiss.Index,
    embeddings: np.ndarray,
    document_ids: list[str],
) -> None:
    """Add one embedding batch to FAISS using numeric MS MARCO passage IDs."""
    try:
        ids = np.array([int(document_id) for document_id in document_ids], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(
            "Full vector indexing expects numeric document IDs. "
            "MS MARCO collection.tsv uses numeric passage IDs."
        ) from exc

    index.add_with_ids(embeddings, ids)


def save_index(index: faiss.Index, index_path: Path) -> None:
    """Persist the FAISS index to disk."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))


def build_vector_index(
    *,
    collection_path: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    max_documents: int | None = None,
    batch_size: int = 800,
    encode_batch_size: int = 90,
    save_every: int = 50_000,
    reset: bool = False,
    resume: bool = True,
) -> faiss.Index:
    """Build or resume a full MS MARCO vector index."""
    if not collection_path.exists():
        raise FileNotFoundError(f"Collection file not found: {collection_path}")

    if reset:
        print("RESET requested: removing vector index and checkpoint")
        remove_index(index_path)
        clear_checkpoint(checkpoint_path)

    checkpoint = load_checkpoint(checkpoint_path) if resume and not reset else None
    start_count = 0
    skip_until_id = None
    index: faiss.Index | None = None

    if checkpoint:
        start_count = int(checkpoint["total_documents_indexed"])
        skip_until_id = checkpoint["last_document_id"]
        print("Checkpoint found:")
        print(f"  Previously indexed: {start_count:,}")
        print(f"  Last document ID: {skip_until_id}")

        if index_path.exists():
            print(f"Loading existing vector index: {index_path}")
            index = faiss.read_index(str(index_path))
            print(f"  FAISS ntotal: {index.ntotal:,}")
        else:
            raise FileNotFoundError(
                f"Checkpoint exists but vector index is missing: {index_path}. "
                "Use --reset to start over."
            )

    print("=" * 70)
    print("STARTING VECTOR INDEXING")
    print("=" * 70)
    print(f"Collection: {collection_path}")
    print(f"Index path: {index_path}")
    print(f"Model: {EMBEDDING_MODEL_NAME}")
    print(f"Batch size: {batch_size}")
    print(f"Encode batch size: {encode_batch_size}")
    print(f"Save every: {save_every:,} documents")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    remaining_documents = max_documents
    if max_documents is not None and start_count:
        remaining_documents = max(max_documents - start_count, 0)

    passages = load_msmarco_passages(
        collection_path=collection_path,
        max_documents=remaining_documents,
        skip_until_id=skip_until_id,
    )

    count = start_count
    last_document_id = skip_until_id
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    try:
        for document in tqdm(passages, desc="Vector indexing", total=remaining_documents):
            batch_ids.append(str(document["id"]))
            batch_texts.append(f"{document.get('title', '')} {document.get('body', '')}")

            if len(batch_texts) < batch_size:
                continue

            embeddings = encode_texts(model, batch_texts, encode_batch_size)

            if index is None:
                index = create_faiss_index(embeddings.shape[1])

            add_batch_to_index(index, embeddings, batch_ids)

            count += len(batch_texts)
            last_document_id = batch_ids[-1]
            batch_texts.clear()
            batch_ids.clear()

            if count % save_every == 0:
                print(f"\nSaving checkpoint at {count:,} documents...")
                save_index(index, index_path)
                save_checkpoint(
                    checkpoint_path,
                    total_documents_indexed=count,
                    last_document_id=last_document_id,
                    collection_path=collection_path,
                    index_path=index_path,
                    model_name=EMBEDDING_MODEL_NAME,
                )
                print(f"  Saved FAISS ntotal: {index.ntotal:,}")
                gc.collect()

        if batch_texts:
            embeddings = encode_texts(model, batch_texts, encode_batch_size)

            if index is None:
                index = create_faiss_index(embeddings.shape[1])

            add_batch_to_index(index, embeddings, batch_ids)

            count += len(batch_texts)
            last_document_id = batch_ids[-1]

        if index is None:
            raise RuntimeError("No documents were indexed.")

        print(f"\nFinal save at {count:,} documents...")
        save_index(index, index_path)
        save_checkpoint(
            checkpoint_path,
            total_documents_indexed=count,
            last_document_id=last_document_id,
            collection_path=collection_path,
            index_path=index_path,
            model_name=EMBEDDING_MODEL_NAME,
        )
        print(f"Vector indexing complete. FAISS ntotal: {index.ntotal:,}")

        return index

    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nInterrupted or failed: {exc}")
        if index is not None and last_document_id is not None:
            print(f"Saving recoverable checkpoint at {count:,} documents...")
            save_index(index, index_path)
            save_checkpoint(
                checkpoint_path,
                total_documents_indexed=count,
                last_document_id=last_document_id,
                collection_path=collection_path,
                index_path=index_path,
                model_name=EMBEDDING_MODEL_NAME,
            )
            print("Checkpoint saved. Re-run the same command to resume.")
        raise


def print_status(index_path: Path, checkpoint_path: Path) -> None:
    """Print current vector index/checkpoint status."""
    checkpoint = load_checkpoint(checkpoint_path)

    print("=" * 70)
    print("VECTOR INDEX STATUS")
    print("=" * 70)
    print(f"Index path: {index_path}")
    print(f"Index exists: {index_path.exists()}")

    if index_path.exists():
        index = faiss.read_index(str(index_path))
        print(f"FAISS ntotal: {index.ntotal:,}")

    if checkpoint:
        print(f"Checkpoint documents: {checkpoint['total_documents_indexed']:,}")
        print(f"Last document ID: {checkpoint['last_document_id']}")
        print(f"Model: {checkpoint['model_name']}")
        print(f"Timestamp: {checkpoint['timestamp']}")
    else:
        print("Checkpoint: none")

    print("=" * 70)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Build a FAISS vector index for MS MARCO.")
    parser.add_argument(
        "--collection",
        type=Path,
        default=DATA_DIR / "msmarco" / "collection.tsv",
        help="Path to collection.tsv",
    )
    parser.add_argument(
        "--index-path",
        type=Path,
        default=DEFAULT_INDEX_PATH,
        help="Output FAISS index path",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Vector checkpoint JSON path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to index for testing",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=800,
        help="Number of passages to encode before adding to FAISS",
    )
    parser.add_argument(
        "--encode-batch-size",
        type=int,
        default=90,
        help="Internal SentenceTransformer encode batch size",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50_000,
        help="Save FAISS index and checkpoint every N documents",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing vector index and checkpoint before indexing",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore checkpoint and continue from the start without deleting index",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show vector indexing status without indexing",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point."""
    args = parse_args()

    if args.status:
        print_status(args.index_path, args.checkpoint_path)
        return

    build_vector_index(
        collection_path=args.collection,
        index_path=args.index_path,
        checkpoint_path=args.checkpoint_path,
        max_documents=args.max_docs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        save_every=args.save_every,
        reset=args.reset,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
