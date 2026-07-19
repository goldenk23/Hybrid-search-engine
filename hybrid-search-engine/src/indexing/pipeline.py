"""
Document indexing pipeline with checkpoint and resume support.

This module loads raw documents, cleans them, validates them, and builds

    collection_path: Path,
    max_documents: int | None = None,
    skip_until_id: str | None = None,
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
        max_documents: Maximum documents to load
        skip_until_id: Skip all documents until this ID is found (for resume)
    """
    count = 0
    skipping = skip_until_id is not None
    skipped_large = 0
    skipped_errors = 0
    
    with collection_path.open("r", encoding="utf-8", newline="") as file:
        # Manual TSV parsing line-by-line (avoids CSV field size limit issues)
        for line in file:
            line = line.rstrip("\n\r")
            
            # Split on first tab to get ID and text
            if "\t" not in line:
                continue
            
            parts = line.split("\t", 1)  # Split on first tab only
            if len(parts) < 2:
                continue
            
            passage_id, passage_text = parts[0], parts[1]
            
            # Resume logic: skip until we reach the last indexed document
            if skipping:
                if passage_id == skip_until_id:
                    skipping = False
                    print(f"  Resuming from document ID: {passage_id}")
                continue
            
            # Skip documents that are too large (>5GB) to avoid memory issues
            # Most documents should be much smaller; this catches pathological cases
            max_size_bytes = 5 * 1024 * 1024 * 1024  # 5GB
            if len(passage_text) > max_size_bytes:
                skipped_large += 1
                size_mb = len(passage_text) / 1024 / 1024
                print(f"  Warning: Skipping oversized document {passage_id} ({size_mb:.1f}MB)")
                continue
            
            cleaned_body = clean_text(passage_text)
            title = cleaned_body[:100]  # Use the first 100 characters as a title
            if not is_valid_document(title=title, body=cleaned_body):
                continue
            yield {
                "id": passage_id,
                "title": title,
                "body": cleaned_body,
                "category": "msmarco",
            }
            count += 1
            if max_documents and count >= max_documents:
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
        max_documents: Maximum documents to index (None = all)
        reset: Delete existing index before reindexing (default: False)
        resume: Resume from checkpoint if available (default: True)
    """
    if collection_path is None:
        collection_path = DATA_DIR / "msmarco" / "collection.tsv"

    if not collection_path.exists():
        raise FileNotFoundError(
            f"Collection file not found: {collection_path}\n"
            "create a small test collection or download the full MS MARCO collection using the provided script."
        )
    
    # Initialize checkpoint manager
    checkpoint_manager = IndexCheckpoint(INDEX_DIR / "bm25")
    
    print("=" * 70)
    print("STARTING RESUMABLE INDEXING PIPELINE")
    print("=" * 70)

    # Check for existing checkpoint
    checkpoint = None
    skip_until_id = None
    start_count = 0
    is_resuming = False
    
    if resume and not reset:
        checkpoint = checkpoint_manager.load_checkpoint()
        if checkpoint:
            is_resuming = True
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

    print("[1/2] Streaming and indexing passages...")
    bm25 = BM25Search(reset=reset, is_resuming=is_resuming)  # Pass both flags

    remaining_documents = max_documents
    if max_documents is not None:
        remaining_documents = max(max_documents - start_count, 0)

    # Load documents - can resume from checkpoint
    passages = load_msmarco_passages(
        collection_path,
        remaining_documents,
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
    
    # Clear checkpoint after successful completion
    checkpoint_manager.clear_checkpoint()
    
    print("\nIndexing complete!")
    print("=" * 70)

    return bm25
