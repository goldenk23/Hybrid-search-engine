"""
Build a evaluation-honest corpus subset for MS MARCO benchmarking.

WHY THIS EXISTS
---------------
Indexing the first N passages from collection.tsv means most dev-set relevant
passages are missing from the index — making MRR/Recall look artificially low
and incomparable to published results.

This script builds a corpus that:
  1. Contains ALL passages that are relevant to any dev query (gold passages).
  2. Fills the rest up to --target-size with uniformly random passages.

The result: every benchmark query has its answer in the index, so your
MRR@10 / Recall@1000 numbers are honest and comparable to papers — at the
same index size and deployment cost as a naive first-N slice.

USAGE
-----
    # Build a 1.5M eval-honest corpus (recommended):
    python scripts/build_eval_corpus.py \\
        --collection data/msmarco/collection.tsv \\
        --qrels      data/msmarco/qrels.dev.small.tsv \\
        --output     data/msmarco/collection.eval.tsv \\
        --target-size 1500000

    # Then index the output file instead of collection.tsv:
    python scripts/index_documents.py --collection data/msmarco/collection.eval.tsv --reset
    python scripts/index_vectors.py   --collection data/msmarco/collection.eval.tsv --reset

OUTPUT
------
    data/msmarco/collection.eval.tsv   — tab-separated: passage_id<TAB>passage_text
    Same format as collection.tsv, so every existing script accepts it unchanged.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR


def load_gold_ids(qrels_path: Path) -> set[str]:
    """Return the set of passage IDs that are relevant to any dev query."""
    gold: set[str] = set()
    with qrels_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.strip().split("\t")
            # qrels format: query_id  0  passage_id  relevance
            if len(parts) >= 3:
                gold.add(parts[2])
    return gold


def stream_collection(collection_path: Path):
    """Yield (passage_id, line) tuples from collection.tsv."""
    with collection_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.rstrip("\n\r")
            if "\t" not in line:
                continue
            passage_id = line.split("\t", 1)[0]
            yield passage_id, line


def build_eval_corpus(
    collection_path: Path,
    qrels_path: Path,
    output_path: Path,
    target_size: int,
    seed: int,
) -> None:
    if not collection_path.exists():
        raise FileNotFoundError(f"Collection not found: {collection_path}")
    if not qrels_path.exists():
        raise FileNotFoundError(f"Qrels not found: {qrels_path}")

    print("=" * 70)
    print("BUILDING EVALUATION-HONEST CORPUS SUBSET")
    print("=" * 70)
    print(f"Collection:  {collection_path}")
    print(f"Qrels:       {qrels_path}")
    print(f"Output:      {output_path}")
    print(f"Target size: {target_size:,} passages")
    print(f"Random seed: {seed}")
    print()

    # ── Step 1: load gold passage IDs ─────────────────────────────────────────
    print("Loading gold passage IDs from qrels…")
    gold_ids = load_gold_ids(qrels_path)
    print(f"  Gold passages (relevant to at least one query): {len(gold_ids):,}")

    if len(gold_ids) >= target_size:
        raise ValueError(
            f"target_size ({target_size:,}) is smaller than the number of gold "
            f"passages ({len(gold_ids):,}). Use a larger --target-size."
        )

    filler_slots = target_size - len(gold_ids)
    print(f"  Filler slots available: {filler_slots:,}")
    print()

    # ── Step 2: single pass — collect gold lines, reservoir-sample the rest ───
    # Reservoir sampling (Algorithm R) gives a uniform random sample in one
    # pass without knowing the total corpus size upfront.
    # ponytail: O(n) time, O(target_size) memory — correct and simple enough.
    print("Scanning collection (one pass)…")

    gold_lines: dict[str, str] = {}   # id → full line, for gold passages
    reservoir: list[str] = []         # random sample of non-gold lines
    rng = random.Random(seed)
    non_gold_seen = 0

    for passage_id, line in tqdm(
        stream_collection(collection_path), desc="Scanning", unit=" passages"
    ):
        if passage_id in gold_ids:
            gold_lines[passage_id] = line
        else:
            non_gold_seen += 1
            if len(reservoir) < filler_slots:
                reservoir.append(line)
            else:
                # Replace a random earlier entry with decreasing probability.
                j = rng.randint(0, non_gold_seen - 1)
                if j < filler_slots:
                    reservoir[j] = line

    found_gold = len(gold_lines)
    missing_gold = len(gold_ids) - found_gold
    print(f"\n  Gold passages found in collection: {found_gold:,}")
    if missing_gold:
        # Some gold IDs exist in full qrels but not in our collection slice —
        # this is normal if you downloaded a qrels file for a different split.
        print(f"  Gold passages NOT found (will not affect indexing): {missing_gold:,}")
    print(f"  Filler passages sampled: {len(reservoir):,}")

    # ── Step 3: shuffle and write ─────────────────────────────────────────────
    print("\nShuffling and writing output…")
    all_lines = list(gold_lines.values()) + reservoir
    rng.shuffle(all_lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for line in tqdm(all_lines, desc="Writing", unit=" passages"):
            fh.write(line + "\n")
    tmp.replace(output_path)  # atomic swap

    size_gb = output_path.stat().st_size / 1e9
    print(f"\nDone. Written {len(all_lines):,} passages to {output_path} ({size_gb:.2f} GB)")
    print("\nNext steps:")
    print(f"  python scripts/index_documents.py --collection {output_path} --reset")
    print(f"  python scripts/index_vectors.py   --collection {output_path} --reset")
    print("=" * 70)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build an evaluation-honest corpus subset for MS MARCO."
    )
    parser.add_argument(
        "--collection",
        type=Path,
        default=DATA_DIR / "msmarco" / "collection.tsv",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DATA_DIR / "msmarco" / "qrels.dev.small.tsv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "msmarco" / "collection.eval.tsv",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=1_500_000,
        help="Total passages to include (gold + random filler). Default: 1,500,000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling. Default: 42",
    )
    args = parser.parse_args()

    build_eval_corpus(
        collection_path=args.collection,
        qrels_path=args.qrels,
        output_path=args.output,
        target_size=args.target_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
