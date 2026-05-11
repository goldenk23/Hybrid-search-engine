"""
Document indexing pipeline.

This module loads raw documents, cleans them, validates them, and builds
the BM25 index used by the search engine.
"""

import csv
from collections.abc import Generator
from pathlib import Path
from tqdm import tqdm
from src.config import DATA_DIR
from src.indexing.preprocessing import clean_text, is_valid_document
from src.search.bm25 import BM25Search


def load_msmarco_passages(
    collection_path: Path,
    max_documents: int | None = None,
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
    """
    count = 0
    with collection_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")

        for row in reader:
            if len(row) < 2:
                continue
            passage_id, passage_text = row[0], row[1]
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
                return


def run_indexing_pipeline(
    collection_path: Path | None = None,
    max_documents: int | None = None,
) -> BM25Search:
    """
    Run the full indexing pipeline.

    Steps:
    1. Load raw passages.
    2. Clean and validate them.
    3. Build the BM25 index.
    """
    if collection_path is None:
        collection_path = DATA_DIR / "msmarco" / "collection.tsv"

    if not collection_path.exists():
        raise FileNotFoundError(
            f"Collection file not found: {collection_path}\n"
            "create a small test collection or download the full MS MARCO collection using the provided script."
        )
    print("=" * 60)
    print("starting indexing pipeline")
    print("=" * 60)

    print("\n[1/2] loading and processing passages...")

    documents = list(
        tqdm(
            load_msmarco_passages(collection_path, max_documents),
            desc="Processing passages",
            total=max_documents or None,
        )
    )
    print(f"Loaded {len(documents):,} valid documents.")
    print("\n[2/2] building BM25 index...")
    bm25 = BM25Search()
    bm25.add_documents(documents)
    print("\nIndexing complete.")
    print(f"BM25 index path: {bm25.index_path}")

    return bm25
