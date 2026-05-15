# Usage:
#   python scripts/test_vector.py "machine learning" --top-k 5
#   python scripts/test_vector.py "how to fix abdominal pain"

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.search.vector import VectorSearch


def read_docs(collection_path: Path, wanted_ids: set[str]) -> dict[str, dict[str, str]]:
    docs = {}
    with collection_path.open("r", encoding="utf-8", newline="") as file:
        for line in file:
            parts = line.rstrip("\n\r").split("\t", 2)
            if len(parts) < 2 or parts[0] not in wanted_ids:
                continue

            doc_id = parts[0]
            title = parts[1] if len(parts) == 3 else parts[1][:100]
            body = parts[2] if len(parts) == 3 else parts[1]
            docs[doc_id] = {"id": doc_id, "title": title, "body": body}

            if len(docs) == len(wanted_ids):
                break
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Search the vector index.")
    parser.add_argument("query", nargs="?", default="machine learning")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    collection_path = DATA_DIR / "msmarco" / "collection.tsv"
    engine = VectorSearch()
    results = engine.search(args.query, top_k=args.top_k)
    docs = read_docs(collection_path, {result["id"] for result in results})

    print(f"Query: {args.query}\n")
    for rank, result in enumerate(results, start=1):
        doc = docs.get(result["id"], {"title": "NOT FOUND", "body": "NOT FOUND"})
        print(f"{rank}. id: {result['id']}  score: {result['score']:.4f}")
        print(f"title: {doc['title']}")
        print(f"body: {doc['body']}\n")


if __name__ == "__main__":
    main()
