"""
Build a full-corpus FAISS vector index for MS MARCO passages.

This script is intentionally separate from scripts/index_documents.py because
embedding millions of passages is a heavier long-running job than BM25.

SAFE STAGED USAGE (RECOMMENDED):
================================

Stage 1: Test with small dataset and reset (10K documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 10000 --reset

Stage 2: Medium dataset (100K documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 100000

Stage 3: Large dataset (1M documents)
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv --max-docs 1000000

Full corpus (no limit):
    python scripts/index_vectors.py --collection data/msmarco/collection.tsv

If interrupted, re-run the same command — it resumes safely from the last checkpoint.
To start completely over, add --reset.

MONITORING:
===========

    python scripts/index_vectors.py --status

OUTPUT FILES:
==============

    data/indexes/vector.faiss              — FAISS index
    data/indexes/vector_checkpoint.json    — resumption metadata
    data/indexes/artifact_manifest.json    — build receipt (generation ID, counts, fingerprints)
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import (
    DATA_DIR,
    DOCSTORE_PATH,
    EMBEDDING_MODEL_NAME,
    EMBEDDING_MODEL_REVISION,
    INDEX_DIR,
    PREPROCESSING_VERSION,
    VECTOR_INDEX_PATH,
    BM25_INDEX_PATH,
    ensure_build_directories,
)
from src.database.docstore import SQLiteDocstore
from src.indexing.artifact_state import (
    load_json_required,
    reconcile_count,
    sha256_path,
    write_json_atomic,
)
from src.indexing.pipeline import load_msmarco_passages


DEFAULT_INDEX_PATH = INDEX_DIR / "vector.faiss"
DEFAULT_CHECKPOINT_PATH = INDEX_DIR / "vector_checkpoint.json"
DEFAULT_MANIFEST_PATH = INDEX_DIR / "artifact_manifest.json"


# ------------------------------------------------------------------ checkpoint

def load_checkpoint(checkpoint_path: Path) -> dict[str, Any] | None:
    """Load vector indexing checkpoint metadata if it exists."""
    if not checkpoint_path.exists():
        return None
    return load_json_required(checkpoint_path)


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
    payload = {
        "total_documents_indexed": total_documents_indexed,
        "last_document_id": last_document_id,
        "collection_path": str(collection_path),
        "index_path": str(index_path),
        "model_name": model_name,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_json_atomic(checkpoint_path, payload)


def clear_checkpoint(checkpoint_path: Path) -> None:
    """Remove checkpoint metadata if present."""
    checkpoint_path.unlink(missing_ok=True)


# ------------------------------------------------------------------ index ops

def remove_index(index_path: Path) -> None:
    """Remove the vector index file and the legacy doc-IDs sidecar if present."""
    index_path.unlink(missing_ok=True)
    # Delete the old position→doc_id sidecar; IDs now live inside IndexIDMap2.
    (index_path.parent / "vector_doc_ids.npy").unlink(missing_ok=True)


def create_faiss_index(dimension: int) -> faiss.Index:
    """
    Create an exact cosine-similarity index.

    Embeddings are L2-normalised by SentenceTransformer, so inner product ==
    cosine similarity.  IndexIDMap2 stores MS MARCO passage IDs directly
    inside FAISS, so there is no separate doc_ids sidecar to go out of sync.
    """
    base_index = faiss.IndexFlatIP(dimension)
    return faiss.IndexIDMap2(base_index)


def encode_texts(
    model: Any,
    texts: list[str],
    encode_batch_size: int,
) -> np.ndarray:
    """Encode and L2-normalise one batch of document texts."""
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
        ids = np.array([int(did) for did in document_ids], dtype=np.int64)
    except ValueError as exc:
        raise ValueError(
            "Full vector indexing expects numeric document IDs. "
            "MS MARCO collection.tsv uses numeric passage IDs."
        ) from exc
    index.add_with_ids(embeddings, ids)


def save_index_atomic(index: faiss.Index, index_path: Path) -> None:
    """Persist the FAISS index to disk without risk of a partial write.

    Strategy: write to a sibling .tmp file, then os.replace() it into place.
    os.replace() is atomic — the live file is either the old complete version
    or the new complete version, never a half-written mix of both.
    If the process dies mid-write the .tmp is harmless and will be overwritten
    on the next attempt.
    """
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_suffix(index_path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)           # remove stale temp from prior crash
    faiss.write_index(index, str(temporary))
    os.replace(temporary, index_path)           # instant atomic swap


# ------------------------------------------------------------------ manifest helpers

def _manifest_identity(collection_path: Path) -> dict:
    """Return the fields that must stay constant across a resume."""
    return {
        "collection_sha256": sha256_path(collection_path),
        "preprocessing_version": PREPROCESSING_VERSION,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_revision": EMBEDDING_MODEL_REVISION,
        "vector_id_mode": "faiss_id_map_2",
    }


def _validate_resume_manifest(manifest: dict, collection_path: Path) -> None:
    """Raise RuntimeError if the manifest identity doesn't match current config."""
    current = _manifest_identity(collection_path)
    mismatches = {
        key: {"stored": manifest.get(key), "current": current[key]}
        for key in current
        if manifest.get(key) != current[key]
    }
    if mismatches:
        raise RuntimeError(
            f"Cannot resume: manifest identity mismatch: {mismatches}. "
            "Use --reset to start a new build."
        )


# ------------------------------------------------------------------ main build

def build_vector_index(
    *,
    collection_path: Path,
    index_path: Path = DEFAULT_INDEX_PATH,
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    max_documents: int | None = None,
    batch_size: int = 800,
    encode_batch_size: int = 90,
    save_every: int = 50_000,
    reset: bool = False,
) -> faiss.Index:
    """Build or resume a full MS MARCO vector index."""
    if not collection_path.exists():
        raise FileNotFoundError(f"Collection file not found: {collection_path}")

    if reset:
        print("RESET requested: removing vector index, checkpoint, and manifest")
        remove_index(index_path)
        clear_checkpoint(checkpoint_path)
        manifest_path.unlink(missing_ok=True)

    # ---- decide: fresh build or resume? ----

    manifest: dict | None = None
    start_count = 0
    skip_until_id: str | None = None
    index: faiss.Index | None = None

    if manifest_path.exists():
        # A manifest exists — this is a resume.  Validate identity before touching anything.
        manifest = load_json_required(manifest_path)
        _validate_resume_manifest(manifest, collection_path)

        checkpoint = load_checkpoint(checkpoint_path)
        checkpoint_count = (
            int(checkpoint["total_documents_indexed"]) if checkpoint else 0
        )

        if index_path.exists():
            index = faiss.read_index(str(index_path))
            durable_count = index.ntotal
        else:
            durable_count = 0

        # reconcile_count raises if checkpoint is ahead of the durable index.
        start_count = reconcile_count(checkpoint_count, durable_count)
        skip_until_id = checkpoint.get("last_document_id") if checkpoint else None

        print(f"Resuming build (generation {manifest['generation_id'][:8]}…)")
        print(f"  Durable vectors: {durable_count:,}  |  checkpoint: {checkpoint_count:,}")
        print(f"  Resuming from: {start_count:,}")

        # Already at or past the target — nothing to do.
        if max_documents is not None and start_count >= max_documents:
            print(f"Already have {start_count:,} documents (target: {max_documents:,}). Nothing to do.")
            return index

    else:
        # Fresh build — stamp the receipt before the first write.
        manifest = {
            "schema_version": 1,
            "generation_id": str(uuid.uuid4()),
            "status": "building",
            "target_documents": max_documents,
            **_manifest_identity(collection_path),
        }
        write_json_atomic(manifest_path, manifest)
        print(f"New build (generation {manifest['generation_id'][:8]}…)")

    print("=" * 70)
    print("STARTING VECTOR INDEXING")
    print("=" * 70)
    print(f"Collection:        {collection_path}")
    print(f"Index path:        {index_path}")
    print(f"Model:             {EMBEDDING_MODEL_NAME}")
    print(f"Preprocessing ver: {PREPROCESSING_VERSION}")
    print(f"Batch size:        {batch_size}")
    print(f"Encode batch:      {encode_batch_size}")
    print(f"Save every:        {save_every:,} documents")
    print("=" * 70)

    from sentence_transformers import SentenceTransformer
    # Pass the pinned revision so the exact model weights are reproducible.
    # Empty string in config means "latest" -> None lets huggingface resolve it.
    # The query path (src/search/vector.py) pins the same revision, so index-time
    # and query-time embeddings can never silently drift apart.
    model = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        revision=EMBEDDING_MODEL_REVISION or None,
    )

    remaining = None if max_documents is None else max_documents - start_count

    passages = load_msmarco_passages(
        collection_path=collection_path,
        max_documents=remaining,
        skip_until_id=skip_until_id,
    )

    count = start_count
    last_saved_count = start_count   # track how many docs were saved at last checkpoint
    last_document_id: str | None = skip_until_id
    batch_texts: list[str] = []
    batch_ids: list[str] = []

    try:
        for document in tqdm(passages, desc="Vector indexing", total=remaining):
            batch_ids.append(str(document["id"]))
            batch_texts.append(
                f"{document.get('title', '')} {document.get('body', '')}"
            )

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

            # Save when we've accumulated at least save_every *new* docs since
            # the last save.  Using a delta (count - last_saved_count) instead
            # of modulo (count % save_every) means we save reliably even when
            # the batch size doesn't divide evenly into save_every.
            if count - last_saved_count >= save_every:
                print(f"\nSaving checkpoint at {count:,} documents…")
                save_index_atomic(index, index_path)
                save_checkpoint(
                    checkpoint_path,
                    total_documents_indexed=count,
                    last_document_id=last_document_id,
                    collection_path=collection_path,
                    index_path=index_path,
                    model_name=EMBEDDING_MODEL_NAME,
                )
                last_saved_count = count
                print(f"  FAISS ntotal: {index.ntotal:,}")
                gc.collect()

        # flush the last partial batch
        if batch_texts:
            embeddings = encode_texts(model, batch_texts, encode_batch_size)
            if index is None:
                index = create_faiss_index(embeddings.shape[1])
            add_batch_to_index(index, embeddings, batch_ids)
            count += len(batch_texts)
            last_document_id = batch_ids[-1]

        if index is None:
            raise RuntimeError("No documents were indexed.")

        print(f"\nFinal save at {count:,} documents…")
        save_index_atomic(index, index_path)
        save_checkpoint(
            checkpoint_path,
            total_documents_indexed=count,
            last_document_id=last_document_id,
            collection_path=collection_path,
            index_path=index_path,
            model_name=EMBEDDING_MODEL_NAME,
        )
        print(f"Vector indexing complete. FAISS ntotal: {index.ntotal:,}")

        # ---- finalize manifest ----
        # Collect counts from all three artifacts and verify they agree.
        docstore = SQLiteDocstore(DOCSTORE_PATH, read_only=True)
        counts = {
            "vector": index.ntotal,
            "docstore": docstore.count_documents(),
        }
        # BM25 count is optional at this stage — only checked when the index exists.
        if BM25_INDEX_PATH.exists():
            try:
                import tantivy
                bm25_index = tantivy.Index.open(str(BM25_INDEX_PATH))
                counts["bm25"] = bm25_index.searcher().num_docs
            except Exception as exc:
                print(f"Warning: could not read BM25 count: {exc}")

        unique_counts = set(counts.values())
        if len(unique_counts) != 1:
            print(
                f"Warning: artifact counts disagree: {counts}. "
                "Run index_documents.py first, or rebuild both together."
            )
        else:
            print(f"Artifact counts agree: {counts}")

        manifest.update({
            "status": "complete",
            "counts": counts,
            "artifact_sha256": {
                "vector": sha256_path(index_path),
                "docstore": sha256_path(DOCSTORE_PATH),
                **({"bm25": sha256_path(BM25_INDEX_PATH)} if BM25_INDEX_PATH.exists() else {}),
            },
        })
        write_json_atomic(manifest_path, manifest)
        print(f"Manifest finalized: {manifest_path}")

        return index

    except (Exception, KeyboardInterrupt) as exc:
        print(f"\nInterrupted or failed: {exc}")
        if index is not None and last_document_id is not None:
            print(f"Saving recoverable checkpoint at {count:,} documents…")
            save_index_atomic(index, index_path)
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


# ------------------------------------------------------------------ status

def print_status(
    index_path: Path,
    checkpoint_path: Path,
    manifest_path: Path,
) -> None:
    """Print current vector index / checkpoint / manifest status."""
    print("=" * 70)
    print("VECTOR INDEX STATUS")
    print("=" * 70)
    print(f"Index path:    {index_path}  (exists: {index_path.exists()})")

    if index_path.exists():
        idx = faiss.read_index(str(index_path))
        print(f"FAISS ntotal:  {idx.ntotal:,}")

    if checkpoint_path.exists():
        cp = load_json_required(checkpoint_path)
        print(f"Checkpoint:    {cp['total_documents_indexed']:,} docs  |  last id: {cp['last_document_id']}  |  {cp['timestamp']}")
    else:
        print("Checkpoint:    none")

    if manifest_path.exists():
        mf = load_json_required(manifest_path)
        print(f"Manifest:      status={mf.get('status')}  generation={mf.get('generation_id', '')[:8]}…")
        if "counts" in mf:
            print(f"  Counts:      {mf['counts']}")
    else:
        print("Manifest:      none")

    print("=" * 70)


# ------------------------------------------------------------------ CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a FAISS vector index for MS MARCO."
    )
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
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Artifact manifest JSON path",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum documents to index",
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
        help="Delete existing index, checkpoint, and manifest before indexing",
    )
    # --no-resume is intentionally removed: resuming from row 1 while keeping
    # an existing index would double-index documents.  Use --reset to start over.
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show vector indexing status without indexing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_build_directories()

    if args.status:
        print_status(args.index_path, args.checkpoint_path, args.manifest_path)
        return

    build_vector_index(
        collection_path=args.collection,
        index_path=args.index_path,
        checkpoint_path=args.checkpoint_path,
        manifest_path=args.manifest_path,
        max_documents=args.max_docs,
        batch_size=args.batch_size,
        encode_batch_size=args.encode_batch_size,
        save_every=args.save_every,
        reset=args.reset,
    )


if __name__ == "__main__":
    main()
