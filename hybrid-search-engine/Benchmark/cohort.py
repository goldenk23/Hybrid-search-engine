"""
Cohort helpers — create and load fixed evaluation question sets.

A cohort is a committed JSON file that locks in:
  - the exact query IDs to evaluate
  - SHA-256 fingerprints of the queries file and qrels file

Locking the question set means two benchmark runs are always comparable:
same questions, same answer key.  Without this, a run that happened to pick
easier questions looks better than one that picked harder ones.

Usage
-----
Create a cohort from the dev-small files (run once, commit the output):

    python Benchmark/cohort.py create \\
        --queries data/msmarco/queries.dev.small.tsv \\
        --qrels   data/msmarco/qrels.dev.small.tsv \\
        --max-queries 500 \\
        --output  Benchmark/cohorts/dev500.json

Load it in a benchmark script:

    from Benchmark.cohort import load_cohort
    queries, qrels = load_cohort(Path("Benchmark/cohorts/dev500.json"),
                                 queries_path, qrels_path)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------ hashing

def _sha256_file(path: Path) -> str:
    """Return the hex SHA-256 digest of a file's contents."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ------------------------------------------------------------------ I/O helpers

def _load_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 2:
                queries[row[0]] = row[1]
    return queries


def _load_qrels(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = defaultdict(set)
    with path.open("r", encoding="utf-8") as fh:
        for row in csv.reader(fh, delimiter="\t"):
            if len(row) >= 4 and int(row[3]) > 0:
                qrels[row[0]].add(row[2])
    return dict(qrels)


def _write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(tmp, path)


# ------------------------------------------------------------------ public API

def create_cohort(
    queries_path: Path,
    qrels_path: Path,
    output_path: Path,
    max_queries: int | None = None,
) -> dict:
    """Build a cohort JSON from queries + qrels and write it atomically.

    The cohort contains:
      - schema_version
      - fingerprints of the source files (so load_cohort can detect drift)
      - the list of query IDs (in stable order)
      - the query texts and relevant doc IDs (inline, so the cohort is
        self-contained and can be validated without the original files)

    Args:
        queries_path: TSV file with (query_id, query_text) rows
        qrels_path:   TSV file with (query_id, 0, doc_id, relevance) rows
        output_path:  where to write the cohort JSON
        max_queries:  cap the cohort size (None = all queries with qrels)

    Returns:
        The cohort dict that was written.
    """
    queries = _load_queries(queries_path)
    qrels   = _load_qrels(qrels_path)

    # Keep only queries that have at least one relevant document.
    selected = []
    for qid in queries:
        if qid in qrels:
            selected.append(qid)
        if max_queries is not None and len(selected) >= max_queries:
            break

    cohort = {
        "schema_version": 1,
        "queries_sha256": _sha256_file(queries_path),
        "qrels_sha256":   _sha256_file(qrels_path),
        "max_queries":    max_queries,
        "query_ids":      selected,
        # Inline the text and relevant docs so the cohort is self-contained.
        "queries":  {qid: queries[qid] for qid in selected},
        "qrels":    {qid: sorted(qrels[qid]) for qid in selected},
    }

    _write_atomic(output_path, cohort)
    print(f"Cohort written: {output_path}  ({len(selected)} queries)")
    return cohort


def load_cohort(
    cohort_path: Path,
    queries_path: Path | None = None,
    qrels_path: Path | None = None,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Load a cohort and return (queries, qrels) dicts.

    If queries_path and qrels_path are supplied, their SHA-256 fingerprints
    are compared against the values stored in the cohort.  A mismatch means
    the source files changed after the cohort was created — raise immediately
    rather than silently benchmark against a different question set.

    The function loads *every* query ID in the cohort or raises RuntimeError.
    Silently dropping questions would make the run incomparable to others.

    Args:
        cohort_path:  path to the cohort JSON
        queries_path: optional — validate fingerprint against cohort
        qrels_path:   optional — validate fingerprint against cohort

    Returns:
        (queries dict, qrels dict) — same shapes as _load_queries/_load_qrels
    """
    try:
        cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read cohort file: {cohort_path}") from exc

    if not isinstance(cohort, dict):
        raise TypeError(f"Cohort file is not a JSON object: {cohort_path}")

    # Fingerprint validation — only when source paths are supplied.
    if queries_path is not None:
        actual = _sha256_file(queries_path)
        if actual != cohort.get("queries_sha256"):
            raise RuntimeError(
                f"Queries file has changed since cohort was created.\n"
                f"  cohort fingerprint: {cohort.get('queries_sha256')}\n"
                f"  current file:       {actual}\n"
                f"Re-create the cohort with Benchmark/cohort.py create."
            )

    if qrels_path is not None:
        actual = _sha256_file(qrels_path)
        if actual != cohort.get("qrels_sha256"):
            raise RuntimeError(
                f"Qrels file has changed since cohort was created.\n"
                f"  cohort fingerprint: {cohort.get('qrels_sha256')}\n"
                f"  current file:       {actual}\n"
                f"Re-create the cohort with Benchmark/cohort.py create."
            )

    stored_queries: dict[str, str]       = cohort.get("queries", {})
    stored_qrels:   dict[str, list[str]] = cohort.get("qrels",   {})
    query_ids: list[str]                  = cohort.get("query_ids", [])

    # Load every ID or fail — never silently drop questions.
    missing = [qid for qid in query_ids if qid not in stored_queries]
    if missing:
        raise RuntimeError(
            f"Cohort is corrupt: {len(missing)} query IDs have no text. "
            f"First missing: {missing[0]}"
        )

    queries = {qid: stored_queries[qid] for qid in query_ids}
    qrels   = {qid: set(stored_qrels[qid]) for qid in query_ids
               if qid in stored_qrels}

    return queries, qrels


# ------------------------------------------------------------------ CLI

def _cmd_create(args: argparse.Namespace) -> None:
    create_cohort(
        queries_path=args.queries,
        qrels_path=args.qrels,
        output_path=args.output,
        max_queries=args.max_queries,
    )


def _cmd_info(args: argparse.Namespace) -> None:
    cohort = json.loads(args.cohort.read_text(encoding="utf-8"))
    print(f"Queries:          {len(cohort.get('query_ids', []))}")
    print(f"queries_sha256:   {cohort.get('queries_sha256', 'n/a')}")
    print(f"qrels_sha256:     {cohort.get('qrels_sha256',   'n/a')}")
    print(f"max_queries:      {cohort.get('max_queries',    'n/a')}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or inspect benchmark cohorts.")
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="Build a new cohort JSON.")
    p_create.add_argument("--queries",     type=Path, required=True)
    p_create.add_argument("--qrels",       type=Path, required=True)
    p_create.add_argument("--output",      type=Path, required=True)
    p_create.add_argument("--max-queries", type=int,  default=None)

    p_info = sub.add_parser("info", help="Show cohort metadata.")
    p_info.add_argument("cohort", type=Path)

    args = parser.parse_args()
    {"create": _cmd_create, "info": _cmd_info}[args.command](args)


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    main()
