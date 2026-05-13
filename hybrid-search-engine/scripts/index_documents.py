r"""
Run the document indexing pipeline with checkpoint and resume support.

FEATURES:
---------
✓ Automatically saves checkpoint every 25,000 documents
✓ Resumes from checkpoint if indexing is interrupted
✓ Zero data loss on crashes - resumes from last checkpoint
✓ Progressive indexing - never loses more than one checkpoint interval of work

WORKING EXAMPLES:
-----------------

1. START/RESUME INDEXING (Auto-resumes from checkpoint if interrupted)
   ====================================================================
   python scripts/index_documents.py --collection data/msmarco/collection.tsv
   
   Behavior:
   - If checkpoint exists → resumes from checkpoint
   - If no checkpoint → starts fresh indexing
   - On crash/interrupt → saves checkpoint automatically

## Remove-Item -Path "data\indexes\bm25" -Recurse -Force (for deleting index directory on Windows PowerShell)
2. RESUME WITH DOCUMENT LIMIT
   ===========================
   python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 5000000
   
   Behavior:
   - If checkpoint exists → resumes from that checkpoint
   - Continues until 5M documents indexed
   - Saves new checkpoint every 25,000 documents
   - On crash → resumes from the latest successful checkpoint


3. START FRESH (IGNORE CHECKPOINT, KEEP INDEX)
   ============================================
   python scripts/index_documents.py --collection data/msmarco/collection.tsv --no-resume
   
   Behavior:
   - Ignores any existing checkpoint
   - Continues adding documents to existing index
   - Useful when you have partial index and want fresh start
   - Does NOT delete checkpoint or index


4. COMPLETE RESET (DELETE CHECKPOINT + INDEX + START FRESH)
   =========================================================
   python scripts/index_documents.py --collection data/msmarco/collection.tsv --reset
   
   Behavior:
   - Deletes existing checkpoint file
   - Deletes existing index directory
   - Starts completely fresh indexing
   - On crash → loses all previous work (no checkpoint saved yet)
   - Use only when you want absolute fresh start


5. CHECK CHECKPOINT STATUS (WITHOUT INDEXING)
   ==========================================
   python scripts/index_documents.py --status
   
   Output shows:
   - Collection file path
   - Total documents indexed so far
   - Last indexed document ID
   - When checkpoint was last saved
   - Useful to monitor progress on long-running jobs


6. INDEX ENTIRE COLLECTION (NO LIMIT)
   ==================================
   python scripts/index_documents.py --collection data/msmarco/collection.tsv
   
   Behavior:
   - No --max-docs limit
   - Indexes entire collection continuously
   - Saves checkpoint every 25,000 documents
   - On crash → resumes from last checkpoint

"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.pipeline import run_indexing_pipeline
from src.indexing.checkpoint import IndexCheckpoint
from src.config import INDEX_DIR


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the document indexing pipeline with checkpoint and resume support.",
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=None,
        help="Path to the collection.tsv file. If not provided, defaults to data/msmarco/collection.tsv",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Maximum number of documents to index",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete existing index and checkpoints before reindexing (start completely fresh)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Don't resume from checkpoint - start fresh but keep the index",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show checkpoint status without indexing",
    )
    args = parser.parse_args()

    # Show checkpoint status if requested
    if args.status:
        checkpoint_manager = IndexCheckpoint(INDEX_DIR / "bm25")
        status = checkpoint_manager.get_checkpoint_status()
        if status:
            print("\n" + "=" * 70)
            print("CHECKPOINT STATUS")
            print("=" * 70)
            print(f"Collection: {status['collection_path']}")
            print(f"Documents indexed: {status['total_documents_indexed']:,}")
            print(f"Last indexed ID: {status['last_document_id']}")
            print(f"Last checkpoint: {status['timestamp']}")
            print("=" * 70 + "\n")
        else:
            print("\nNo checkpoint found.")
        return

    resume = not args.no_resume  # Default to resuming unless --no-resume is passed

    bm25 = run_indexing_pipeline(
        collection_path=args.collection,
        max_documents=args.max_docs,
        reset=args.reset,
        resume=resume,
    )

    print("\n" + "=" * 70)
    print("SMOKE TEST")
    print("=" * 70)
    results = bm25.search("where is patna?", top_k=3)

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title'][:60]}... | score={result['score']:.2f}")
    print("=" * 70 + "\n")


# Run the main function when this script is executed not imported as a module
if __name__ == "__main__":
    main()
