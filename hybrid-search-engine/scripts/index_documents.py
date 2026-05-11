"""
Run the document indexing pipeline.

Usage:
    python scripts/index_documents.py --collection data/msmarco/collection.tsv
    python scripts/index_documents.py --max-docs 1000
    python scripts/index_documents.py --collection data/msmarco/collection.tsv --max-docs 500000
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.indexing.pipeline import run_indexing_pipeline

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the document indexing pipeline.",
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
    args = parser.parse_args()

    bm25 = run_indexing_pipeline(
        collection_path=args.collection,
        max_documents=args.max_docs,
    )

    print("\nSmoke test:")
    results = bm25.search("where is patna?", top_k=3)

    for index, result in enumerate(results, start=1):
        print(f"{index}. {result['title']} | score={result['score']}")


# Run the main function when this script is executed not imported as a module
if __name__ == "__main__":
    main()
