"""Create a query/qrels subset for repeatable evaluation runs.

USAGE COMMANDS
--------------

Create or replace subset files in data/msmarco/ with first 100 eligible queries:
    python Benchmark/create_eval_subset.py --num-queries 100

Create or replace subset files in data/msmarco/ with 500 random eligible queries:
    python Benchmark/create_eval_subset.py --num-queries 500 --strategy random --seed 42

Create or replace the evaluation subset inside data/msmarco/:
    python Benchmark/create_eval_subset.py --num-queries 1000 --output-dir data/msmarco

Use custom full-source files:
    python Benchmark/create_eval_subset.py --num-queries 100 \
        --queries data/msmarco/queries.dev.tsv \
        --qrels data/msmarco/qrels.dev.tsv

OUTPUT FILES
------------

The script writes these files, replacing previous versions in --output-dir:
    queries.dev.small.tsv
    qrels.dev.small.tsv
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "msmarco"
DEFAULT_OUTPUT_DIR = DEFAULT_DATA_DIR


def load_queries(path: Path) -> dict[str, str]:
    """Load query_id -> query_text from a MS MARCO queries TSV."""
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                queries[row[0]] = row[1]
    return queries


def load_qrels(path: Path) -> dict[str, list[list[str]]]:
    """Load query_id -> qrel rows from a MS MARCO qrels TSV."""
    qrels: dict[str, list[list[str]]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.reader(file, delimiter="\t")
        for row in reader:
            if len(row) >= 4:
                qrels[row[0]].append(row[:4])
    return dict(qrels)


def select_query_ids(
    query_ids: list[str],
    *,
    num_queries: int,
    strategy: str,
    seed: int,
) -> list[str]:
    """Select query IDs using either stable first-N or seeded random sampling."""
    if num_queries > len(query_ids):
        raise ValueError(
            f"Requested {num_queries:,} queries, but only {len(query_ids):,} "
            "queries have qrels and query text."
        )

    if strategy == "first":
        return query_ids[:num_queries]

    selected = query_ids[:]
    random.Random(seed).shuffle(selected)
    return selected[:num_queries]


def write_subset(
    *,
    selected_query_ids: list[str],
    queries: dict[str, str],
    qrels: dict[str, list[list[str]]],
    output_dir: Path,
) -> tuple[Path, Path, int]:
    """Write queries.dev.small.tsv and qrels.dev.small.tsv to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    queries_out = output_dir / "queries.dev.small.tsv"
    qrels_out = output_dir / "qrels.dev.small.tsv"

    with queries_out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        for query_id in selected_query_ids:
            writer.writerow([query_id, queries[query_id]])

    qrel_rows_written = 0
    selected_set = set(selected_query_ids)
    with qrels_out.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter="\t", lineterminator="\n")
        for query_id in selected_query_ids:
            for row in qrels.get(query_id, []):
                if row[0] in selected_set:
                    writer.writerow(row)
                    qrel_rows_written += 1

    return queries_out, qrels_out, qrel_rows_written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create replaceable queries.dev.small.tsv and qrels.dev.small.tsv "
            "subsets for evaluation."
        )
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        required=True,
        help="Number of dev queries to include in the subset.",
    )
    parser.add_argument(
        "--queries",
        type=Path,
        default=DEFAULT_DATA_DIR / "queries.dev.tsv",
        help="Source full dev queries TSV.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=DEFAULT_DATA_DIR / "qrels.dev.tsv",
        help="Source full dev qrels TSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where subset files are written.",
    )
    parser.add_argument(
        "--strategy",
        choices=["first", "random"],
        default="first",
        help="Use first N eligible queries or seeded random sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --strategy random is selected.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_queries <= 0:
        raise ValueError("--num-queries must be greater than 0.")
    if not args.queries.exists():
        raise FileNotFoundError(f"Queries file not found: {args.queries}")
    if not args.qrels.exists():
        raise FileNotFoundError(f"Qrels file not found: {args.qrels}")

    queries = load_queries(args.queries)
    qrels = load_qrels(args.qrels)

    eligible_query_ids = [
        query_id
        for query_id in queries
        if query_id in qrels
    ]

    selected_query_ids = select_query_ids(
        eligible_query_ids,
        num_queries=args.num_queries,
        strategy=args.strategy,
        seed=args.seed,
    )

    queries_out, qrels_out, qrel_rows_written = write_subset(
        selected_query_ids=selected_query_ids,
        queries=queries,
        qrels=qrels,
        output_dir=args.output_dir,
    )

    print("Evaluation subset created.")
    print(f"Queries requested: {args.num_queries:,}")
    print(f"Eligible source queries: {len(eligible_query_ids):,}")
    print(f"Queries written: {len(selected_query_ids):,}")
    print(f"Qrel rows written: {qrel_rows_written:,}")
    print(f"Queries file: {queries_out}")
    print(f"Qrels file: {qrels_out}")


if __name__ == "__main__":
    main()
