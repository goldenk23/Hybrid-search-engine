"""
Delete all generated artifacts and start fresh.

USAGE:
    python scripts/reset_all.py --yes

The --yes flag is required.  Without it the script exits immediately —
this prevents accidental runs from wiping hours of indexing work.

WHAT IT DELETES (generated artifacts only):
    data/indexes/bm25/
    data/indexes/bm25_compact/
    data/indexes/vector.faiss
    data/indexes/vector.sq8.faiss
    data/indexes/vector_doc_ids.npy       (legacy sidecar)
    data/indexes/vector_checkpoint.json
    data/indexes/artifact_manifest.json
    data/docstore.sqlite

WHAT IT NEVER TOUCHES:
    data/msmarco/collection.tsv           (raw corpus — cannot be cheaply regenerated)
    data/msmarco/queries.*.tsv
    data/msmarco/qrels.*.tsv
    anything outside data/

After running, rebuild with:
    python scripts/index_documents.py --collection data/msmarco/collection.tsv ...
    python scripts/index_vectors.py   --collection data/msmarco/collection.tsv ...
"""

import argparse
import shutil
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR, DOCSTORE_PATH, INDEX_DIR

# Every path here must be a generated artifact — never the raw corpus.
GENERATED = [
    INDEX_DIR / "bm25",
    INDEX_DIR / "bm25_compact",
    INDEX_DIR / "vector.faiss",
    INDEX_DIR / "vector.sq8.faiss",
    INDEX_DIR / "vector_doc_ids.npy",          # legacy sidecar from old ID system
    INDEX_DIR / "vector_checkpoint.json",
    INDEX_DIR / "artifact_manifest.json",
    DOCSTORE_PATH,
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Delete all generated search artifacts (indexes, docstore, manifest)."
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required: confirm you want to delete all generated artifacts.",
    )
    args = parser.parse_args()

    if not args.yes:
        raise SystemExit(
            "Refusing destructive reset without --yes.\n"
            "Re-run with --yes to confirm."
        )

    # Resolve once so every comparison is against an absolute path.
    data_root = DATA_DIR.resolve()

    deleted = []
    skipped = []

    for path in GENERATED:
        resolved = path.resolve()

        # Safety boundary: never delete anything outside data/.
        # This guard catches misconfigured DATA_DIR or a symlink escape.
        if not resolved.is_relative_to(data_root):
            raise RuntimeError(
                f"Unsafe reset path outside data/: {resolved}\n"
                "Aborting — nothing was deleted."
            )

        if resolved.is_dir():
            shutil.rmtree(resolved)
            deleted.append(resolved)
        elif resolved.exists():
            resolved.unlink()
            deleted.append(resolved)
        else:
            skipped.append(resolved)

    print("Reset complete.")
    if deleted:
        print("Deleted:")
        for p in deleted:
            print(f"  {p}")
    if skipped:
        print("Already absent (skipped):")
        for p in skipped:
            print(f"  {p}")


if __name__ == "__main__":
    main()
