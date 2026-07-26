"""
Document indexing pipeline with checkpoint and resume support.

This module loads raw documents, cleans them, validates them, and builds
the BM25 index used by the search engine. It supports:
- Checkpointing every 25,000 documents
- Resuming from checkpoints if indexing crashes
- Continuous progress saving
"""

from collections.abc import Generator
from pathlib import Path

from src.config import DATA_DIR, INDEX_DIR
from src.indexing.checkpoint import IndexCheckpoint
from src.indexing.preprocessing import clean_text, is_valid_document
from src.search.bm25 import BM25Search


def load_msmarco_passages(
    collection_path: Path,
    max_documents: int | None = None,
    skip_until_id: str | None = None,
    skip_documents: int = 0,
) -> Generator[dict, None, None]:
    # Generator that yields document dictionaries with keys: 'id', 'title', 'body', 'category'
    # Format: Generator[YieldType, SendType, ReturnType]
    #   - YieldType (dict): Each document as a dictionary
    #   - SendType (None): This generator doesn't accept sent values
    #   - ReturnType (None): Returns None when generator completes
    """
    Load passages from an MS MARCO collection.tsv file.

    Expected format:
        passage_id<TAB>passage_text

    Args:
        collection_path: Path to collection.tsv file
        max_documents: Maximum documents to yield. None = no limit.
                       0 = yield nothing (zero means zero, not unlimited).
                       Negative values raise ValueError.
        skip_until_id: Skip all documents until this ID is found (for resume)
        skip_documents: Number of valid documents to skip from the start
    """
    # --- explicit guards: check before touching the file ---

    if max_documents is not None and max_documents < 0:
        raise ValueError(f"max_documents cannot be negative (got {max_documents})")
    if skip_documents < 0:
        raise ValueError(f"skip_documents cannot be negative (got {skip_documents})")

    # Zero means zero: caller wants nothing indexed, return immediately.
    # The old code used `if max_documents:` which treated 0 as "no limit" —
    # that could silently index the entire 8M-passage corpus.
    if max_documents == 0:
        return

    count = 0
    skipping = skip_until_id is not None
    skipped_large = 0

    with collection_path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            line = line.rstrip("\n\r")

            if "\t" not in line:
                continue

            parts = line.split("\t", 1)
            if len(parts) < 2:
                continue

            passage_id, passage_text = parts[0], parts[1]

            # Resume logic: skip until we reach the last indexed document
            if skipping:
                if passage_id == skip_until_id:
                    skipping = False
                    print(f"  Resuming from document ID: {passage_id}")
                continue

            # Skip documents that are too large to avoid memory issues
            max_size_bytes = 5 * 1024 * 1024 * 1024  # 5 GB
            if len(passage_text) > max_size_bytes:
                skipped_large += 1
                size_mb = len(passage_text) / 1024 / 1024
                print(f"  Warning: Skipping oversized document {passage_id} ({size_mb:.1f} MB)")
                continue

            cleaned_body = clean_text(passage_text)

            # MS MARCO collection.tsv has no title column — two fields only:
            # passage_id and passage_text.  Use empty string rather than faking
            # a title from body[:100], which would store invented metadata.
            # require_title=False because this corpus is title-less; requiring a
            # title here would reject every single passage.
            if not is_valid_document(title="", body=cleaned_body, require_title=False):
                continue

            yield {
                "id": passage_id,
                "title": "",           # no title in MS MARCO collection.tsv
                "body": cleaned_body,
                "category": "msmarco",
            }
            count += 1

            # Explicit None check: `if max_documents:` would treat 0 as falsy
            # and skip the limit — we already handled 0 above, but being
            # explicit here makes every future reader's life easier.
            if max_documents is not None and count >= max_documents:
                if skipped_large > 0:
                    print(f"  Skipped {skipped_large} oversized documents")
                return


def run_indexing_pipeline(
    collection_path: Path | None = None,
    max_documents: int | None = None,
    reset: bool = False,
    resume: bool = True,
) -> BM25Search:
    """
    Run the full indexing pipeline with checkpoint and resume support.

    Features:
    - Checkpoints every 25,000 documents
    - Resumes from last checkpoint if indexing was interrupted
    - Saves progress continuously
    - Can clear index with --reset flag

    Steps:
    1. Load/create checkpoint status
    2. Load raw passages (resume from last checkpoint if applicable)
    3. Clean and validate them
    4. Build the BM25 index with periodic checkpoints

    Args:
        collection_path: Path to collection.tsv (defaults to data/msmarco/collection.tsv)
        max_documents: Target total to reach. None = index everything.
                       Treated as a final target, not "how many more to add" —
                       so if start_count already meets the target, return early.
        reset: Delete existing index before reindexing (default: False)
        resume: Resume from checkpoint if available (default: True)
    """
    if collection_path is None:
        collection_path = DATA_DIR / "msmarco" / "collection.tsv"

    if not collection_path.exists():
        raise FileNotFoundError(
            f"Collection file not found: {collection_path}\n"
            "Create a small test collection or download the full MS MARCO collection "
            "using the provided script."
        )

    # Initialize checkpoint manager
    checkpoint_manager = IndexCheckpoint(INDEX_DIR / "bm25")

    print("=" * 70)
    print("STARTING RESUMABLE INDEXING PIPELINE")
    print("=" * 70)

    checkpoint = None
    skip_until_id = None
    start_count = 0

    if resume and not reset:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint:
            print("\nCheckpoint found:")
            print(f"   Collection: {checkpoint['collection_path']}")
            print(f"   Previously indexed: {checkpoint['total_documents_indexed']:,} documents")
            print(f"   Last indexed ID: {checkpoint['last_document_id']}")
            skip_until_id = checkpoint['last_document_id']
            start_count = checkpoint['total_documents_indexed']
            print("\nRESUMING from checkpoint...\n")
        else:
            print("\nNo checkpoint found - starting fresh indexing\n")
    elif reset:
        print("\nRESET flag provided - clearing checkpoint and index\n")
        checkpoint_manager.clear_checkpoint()
    else:
        print("\nResume disabled - starting fresh indexing\n")

    # Early-return: if we already have at least as many documents as the target,
    # there is nothing to do — don't open the collection file or build a writer.
    # The old code computed max(max_documents - start_count, 0) and passed 0 as
    # remaining, but that fell through to load_msmarco_passages where 0 was
    # treated as "no limit" due to the truthiness bug now fixed above.
    bm25 = BM25Search(reset=reset)
    durable_count = bm25.committed_document_count()

    if max_documents is not None and durable_count >= max_documents:
        print(
            f"\nAlready have {durable_count:,} documents (target: {max_documents:,}). "
            "Nothing to do."
        )
        return bm25

    # remaining is how many more we need on top of what's durably indexed.
    remaining = None if max_documents is None else max_documents - durable_count

    print("[1/2] Streaming and indexing passages...")

    passages = load_msmarco_passages(
        collection_path,
        max_documents=remaining,
        skip_until_id=skip_until_id,
    )

    count = bm25.add_documents_stream_with_checkpoint(
        passages,
        checkpoint_manager,
        collection_path=collection_path,
        batch_size=1000,
        checkpoint_interval=25000,
        start_count=start_count,
    )

    print(f"\nIndexed {count:,} documents in total.")
    print(f"BM25 index path: {bm25.index_path}")

    # Refuse to declare success on an empty index. If every passage failed
    # validation (e.g. the title-required bug) or the collection was empty/
    # misformatted, the writer commits 0 docs and everything downstream silently
    # returns nothing. Fail loudly here so a broken build is never mistaken for
    # a finished one — the whole point of "index once, reliably".
    durable_after = bm25.committed_document_count()
    if durable_after == 0:
        raise RuntimeError(
            "Indexing finished with 0 documents in the BM25 index. "
            "Every passage was rejected by is_valid_document, or the collection "
            f"at {collection_path} is empty or misformatted. "
            "Refusing to leave an empty index. Checkpoint left intact for inspection."
        )

    checkpoint_manager.clear_checkpoint()

    print("\nIndexing complete!")
    print("=" * 70)

    return bm25
