"""
manage.py — single control unit for the Hybrid Search Engine.

USAGE
-----
    python manage.py <command> [options]

COMMANDS
--------
  Setup & data
    download        Download MS MARCO queries, qrels, and optionally the full corpus
    setup           Download data + build both indexes in one shot

  Indexing
    index-bm25      Build (or resume) the BM25 index
    index-vector    Build (or resume) the FAISS vector index
    index-hnsw      Build an HNSW approximate index from an existing FlatIP index
    quantize        Quantize the FAISS vector index to SQ8 or FP16
    reset           Wipe all generated artifacts (requires --yes)

  Serving
    serve           Start the FastAPI server

  Testing
    test            Run the pytest suite
    smoke           Run a quick manual smoke test against a running API

  Benchmarking
    cohort          Create a fixed evaluation cohort JSON
    benchmark       Run the full retrieval benchmark
    benchmark-ann   Compare FlatIP vs HNSW speed/quality trade-off

  Housekeeping
    status          Show index, checkpoint, and manifest status

EXAMPLES
--------
  Setup & data
    # Download queries, qrels only (fast)
    python manage.py download

    # Download queries, qrels AND the full 8M-passage corpus (~3 GB)
    python manage.py download --include-collection

    # Full first-time setup: download + BM25 + vector (1M passages)
    python manage.py setup --max-docs 1000000

    # Full setup from scratch (wipe existing indexes first)
    python manage.py setup --max-docs 1000000 --reset

  Indexing
    # Build BM25 index (1M passages, resume if interrupted)
    python manage.py index-bm25 --max-docs 1000000

    # Build BM25 index from scratch
    python manage.py index-bm25 --max-docs 1000000 --reset

    # Build vector index (1M passages)
    python manage.py index-vector --max-docs 1000000

    # Check vector index progress without indexing
    python manage.py index-vector --status

    # Build HNSW approximate index from existing FlatIP index
    python manage.py index-hnsw

    # Build HNSW with custom parameters
    python manage.py index-hnsw --M 64 --ef-construction 400

    # Compress vector index to SQ8 (smaller, faster)
    python manage.py quantize

    # Compress to FP16 instead
    python manage.py quantize --method fp16

    # Wipe all generated artifacts (indexes, docstore, manifest)
    python manage.py reset --yes

  Serving
    # Start the API (production mode)
    python manage.py serve

    # Start with auto-reload (development mode)
    python manage.py serve --reload

    # Bind to all interfaces on a custom port
    python manage.py serve --host 0.0.0.0 --port 9000

  Testing
    # Run the full pytest suite
    python manage.py test

    # Verbose output
    python manage.py test -v

    # Run only docstore tests
    python manage.py test -k "docstore"

    # Run only fusion and metrics tests
    python manage.py test -k "fusion or metrics"

    # Hit the live API with a quick smoke query (server must be running)
    python manage.py smoke

    # Smoke test with a custom query
    python manage.py smoke --query "how does photosynthesis work"

  Benchmarking
    # Step 1: create a fixed 500-query evaluation cohort (run once, commit it)
    python manage.py cohort --max-queries 500 --output Benchmark/cohorts/dev500.json

    # Step 2: run the full benchmark (all systems, 5 repeats per query)
    python manage.py benchmark \
        --cohort Benchmark/cohorts/dev500.json \
        --corpus-label 1M \
        --corpus-size 1000000

    # Fast benchmark (skip slow cross-encoder reranking)
    python manage.py benchmark \
        --cohort Benchmark/cohorts/dev500.json \
        --corpus-label 1M \
        --corpus-size 1000000 \
        --skip-rerank

    # Benchmark with a compressed SQ8 index
    python manage.py benchmark \
        --cohort Benchmark/cohorts/dev500.json \
        --corpus-label 1M \
        --corpus-size 1000000 \
        --sq8-index data/indexes/vector.sq8.faiss

    # ANN speed-vs-quality experiment (build HNSW index first)
    python manage.py index-hnsw
    python manage.py benchmark-ann \
        --cohort Benchmark/cohorts/dev500.json

  Housekeeping
    # Show index size, document counts, checkpoint, and manifest status
    python manage.py status
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# All paths are relative to this file's directory (the project root).
ROOT = Path(__file__).parent
DATA_DIR  = ROOT / "data"
INDEX_DIR = DATA_DIR / "indexes"
MSMARCO   = DATA_DIR / "msmarco"


# ------------------------------------------------------------------ helpers

def _run(cmd: list[str], **kwargs) -> int:
    """Run a subprocess and return its exit code."""
    print(f"\n▶  {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, **kwargs)
    return result.returncode


def _python(*args) -> list[str]:
    """Build a command list using the current interpreter."""
    return [sys.executable, *[str(a) for a in args]]


def _die(code: int) -> None:
    if code != 0:
        sys.exit(code)


# ------------------------------------------------------------------ commands

def cmd_download(args: argparse.Namespace) -> None:
    cmd = _python("scripts/download_msmarco.py")
    if args.include_collection:
        cmd.append("--include-collection")
    _die(_run(cmd, cwd=ROOT))


def cmd_setup(args: argparse.Namespace) -> None:
    """Download data then build BM25 + vector indexes in sequence."""
    print("=" * 60)
    print("SETUP: download → index-bm25 → index-vector")
    print("=" * 60)

    # 1. download
    dl_cmd = _python("scripts/download_msmarco.py", "--include-collection")
    _die(_run(dl_cmd, cwd=ROOT))

    # 2. BM25
    bm25_cmd = _python(
        "scripts/index_documents.py",
        "--collection", MSMARCO / "collection.tsv",
    )
    if args.max_docs:
        bm25_cmd += ["--max-docs", str(args.max_docs)]
    if args.reset:
        bm25_cmd.append("--reset")
    _die(_run(bm25_cmd, cwd=ROOT))

    # 3. vector
    vec_cmd = _python(
        "scripts/index_vectors.py",
        "--collection", MSMARCO / "collection.tsv",
    )
    if args.max_docs:
        vec_cmd += ["--max-docs", str(args.max_docs)]
    if args.reset:
        vec_cmd.append("--reset")
    _die(_run(vec_cmd, cwd=ROOT))

    print("\n✓ Setup complete. Run:  python manage.py serve")


def cmd_index_bm25(args: argparse.Namespace) -> None:
    cmd = _python("scripts/index_documents.py")
    if args.collection:
        cmd += ["--collection", args.collection]
    if args.max_docs:
        cmd += ["--max-docs", str(args.max_docs)]
    if args.reset:
        cmd.append("--reset")
    _die(_run(cmd, cwd=ROOT))


def cmd_index_vector(args: argparse.Namespace) -> None:
    cmd = _python("scripts/index_vectors.py")
    if args.collection:
        cmd += ["--collection", args.collection]
    if args.max_docs:
        cmd += ["--max-docs", str(args.max_docs)]
    if args.reset:
        cmd.append("--reset")
    if args.status:
        cmd.append("--status")
    _die(_run(cmd, cwd=ROOT))


def cmd_index_hnsw(args: argparse.Namespace) -> None:
    cmd = _python(
        "scripts/build_hnsw_index.py",
        "--input",  args.input  or INDEX_DIR / "vector.faiss",
        "--output", args.output or INDEX_DIR / "vector.hnsw.faiss",
        "--M", str(args.M),
        "--ef-construction", str(args.ef_construction),
    )
    _die(_run(cmd, cwd=ROOT))


def cmd_quantize(args: argparse.Namespace) -> None:
    cmd = _python("scripts/quantize_vector_index.py")
    if args.input:
        cmd += ["--input", args.input]
    if args.output:
        cmd += ["--output", args.output]
    if args.method:
        cmd += ["--method", args.method]
    _die(_run(cmd, cwd=ROOT))


def cmd_reset(args: argparse.Namespace) -> None:
    cmd = _python("scripts/reset_all.py")
    if args.yes:
        cmd.append("--yes")
    _die(_run(cmd, cwd=ROOT))


def cmd_serve(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable, "-m", "uvicorn", "src.api.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")
    _die(_run(cmd, cwd=ROOT))


def cmd_test(args: argparse.Namespace) -> None:
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    if args.verbose:
        cmd.append("-v")
    if args.k:
        cmd += ["-k", args.k]
    _die(_run(cmd, cwd=ROOT))


def cmd_smoke(args: argparse.Namespace) -> None:
    """Hit the live API with a real query and print the result."""
    import json
    import urllib.request

    base = f"http://{args.host}:{args.port}"
    query = args.query
    url   = f"{base}/search?q={urllib.parse.quote(query)}&top_k=3"

    import urllib.parse
    print(f"Query: {query!r}")
    print(f"URL:   {url}\n")

    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read())
        print(f"returned_count: {data['returned_count']}   latency: {data['latency_ms']} ms")
        for i, r in enumerate(data["results"], 1):
            print(f"  {i}. [{r['score']:.3f}] {r['title'] or r['snippet'] or '(no title)'}")
    except Exception as exc:
        print(f"ERROR: {exc}")
        print("Is the server running?  Try:  python manage.py serve")
        sys.exit(1)


def cmd_cohort(args: argparse.Namespace) -> None:
    output = args.output or ROOT / "Benchmark" / "cohorts" / "dev500.json"
    cmd = _python(
        "Benchmark/cohort.py", "create",
        "--queries", args.queries or MSMARCO / "queries.dev.small.tsv",
        "--qrels",   args.qrels   or MSMARCO / "qrels.dev.small.tsv",
        "--output",  output,
    )
    if args.max_queries:
        cmd += ["--max-queries", str(args.max_queries)]
    _die(_run(cmd, cwd=ROOT))


def cmd_benchmark(args: argparse.Namespace) -> None:
    if not args.cohort:
        print("ERROR: --cohort is required. Create one first:")
        print("  python manage.py cohort --max-queries 500")
        sys.exit(1)

    manifest = INDEX_DIR / "artifact_manifest.json"
    cmd = _python(
        "Benchmark/benchmark_retrieval.py",
        "--manifest",     manifest,
        "--cohort",       args.cohort,
        "--queries",      args.queries or MSMARCO / "queries.dev.small.tsv",
        "--qrels",        args.qrels   or MSMARCO / "qrels.dev.small.tsv",
        "--vector-index", args.vector_index or INDEX_DIR / "vector.faiss",
        "--corpus-label", args.corpus_label,
        "--corpus-size",  str(args.corpus_size),
        "--repeats",      str(args.repeats),
    )
    if args.skip_rerank:
        cmd.append("--skip-rerank")
    if args.sq8_index:
        cmd += ["--sq8-index", args.sq8_index]
    _die(_run(cmd, cwd=ROOT))


def cmd_benchmark_ann(args: argparse.Namespace) -> None:
    if not args.cohort:
        print("ERROR: --cohort is required. Create one first:")
        print("  python manage.py cohort --max-queries 500")
        sys.exit(1)

    cmd = _python(
        "Benchmark/benchmark_ann.py",
        "--flat-index",  args.flat_index  or INDEX_DIR / "vector.faiss",
        "--hnsw-index",  args.hnsw_index  or INDEX_DIR / "vector.hnsw.faiss",
        "--cohort",      args.cohort,
        "--queries",     args.queries or MSMARCO / "queries.dev.small.tsv",
        "--qrels",       args.qrels   or MSMARCO / "qrels.dev.small.tsv",
        "--repeats",     str(args.repeats),
    )
    _die(_run(cmd, cwd=ROOT))


def cmd_status(args: argparse.Namespace) -> None:
    _run(_python(
        "scripts/index_vectors.py",
        "--status",
        "--index-path",      INDEX_DIR / "vector.faiss",
        "--checkpoint-path", INDEX_DIR / "vector_checkpoint.json",
        "--manifest-path",   INDEX_DIR / "artifact_manifest.json",
    ), cwd=ROOT)

    from src.indexing.checkpoint import IndexCheckpoint
    sys.path.insert(0, str(ROOT))
    cp = IndexCheckpoint(INDEX_DIR / "bm25")
    status = cp.get_checkpoint_status()
    print("\nBM25 checkpoint:")
    if status:
        print(f"  documents: {status['total_documents_indexed']:,}")
        print(f"  last id:   {status['last_document_id']}")
        print(f"  saved at:  {status['timestamp']}")
    else:
        print("  no checkpoint found")


# ------------------------------------------------------------------ parser

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="manage.py",
        description="Hybrid Search Engine — main control unit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")

    # ---- download ----
    p = sub.add_parser("download", help="Download MS MARCO data")
    p.add_argument("--include-collection", action="store_true",
                   help="Also download the full collection.tsv (~3 GB)")

    # ---- setup ----
    p = sub.add_parser("setup", help="Download + build all indexes")
    p.add_argument("--max-docs",  type=int, default=None)
    p.add_argument("--reset",     action="store_true")

    # ---- index-bm25 ----
    p = sub.add_parser("index-bm25", help="Build or resume the BM25 index")
    p.add_argument("--collection", type=Path, default=None)
    p.add_argument("--max-docs",   type=int,  default=None)
    p.add_argument("--reset",      action="store_true")

    # ---- index-vector ----
    p = sub.add_parser("index-vector", help="Build or resume the FAISS vector index")
    p.add_argument("--collection", type=Path, default=None)
    p.add_argument("--max-docs",   type=int,  default=None)
    p.add_argument("--reset",      action="store_true")
    p.add_argument("--status",     action="store_true")

    # ---- index-hnsw ----
    p = sub.add_parser("index-hnsw", help="Build HNSW index from existing FlatIP index")
    p.add_argument("--input",          type=Path, default=None)
    p.add_argument("--output",         type=Path, default=None)
    p.add_argument("--M",              type=int,  default=32)
    p.add_argument("--ef-construction",type=int,  default=200)

    # ---- quantize ----
    p = sub.add_parser("quantize", help="Quantize vector index to SQ8 or FP16")
    p.add_argument("--input",  type=Path, default=None)
    p.add_argument("--output", type=Path, default=None)
    p.add_argument("--method", choices=["sq8", "fp16"], default="sq8")

    # ---- reset ----
    p = sub.add_parser("reset", help="Wipe all generated artifacts")
    p.add_argument("--yes", action="store_true",
                   help="Required: confirm destructive reset")

    # ---- serve ----
    p = sub.add_parser("serve", help="Start the FastAPI server")
    p.add_argument("--host",   default="127.0.0.1")
    p.add_argument("--port",   type=int, default=8000)
    p.add_argument("--reload", action="store_true",
                   help="Enable auto-reload (development only)")

    # ---- test ----
    p = sub.add_parser("test", help="Run the pytest suite")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("-k", help="pytest -k filter expression")

    # ---- smoke ----
    p = sub.add_parser("smoke", help="Send a query to the live API")
    p.add_argument("--query", default="what causes rain")
    p.add_argument("--host",  default="127.0.0.1")
    p.add_argument("--port",  type=int, default=8000)

    # ---- cohort ----
    p = sub.add_parser("cohort", help="Create a fixed evaluation cohort")
    p.add_argument("--queries",     type=Path, default=None)
    p.add_argument("--qrels",       type=Path, default=None)
    p.add_argument("--output",      type=Path, default=None)
    p.add_argument("--max-queries", type=int,  default=500)

    # ---- benchmark ----
    p = sub.add_parser("benchmark", help="Run full retrieval benchmark")
    p.add_argument("--cohort",        type=Path, required=False, default=None)
    p.add_argument("--corpus-label",  default="1M")
    p.add_argument("--corpus-size",   type=int, default=1_000_000)
    p.add_argument("--queries",       type=Path, default=None)
    p.add_argument("--qrels",         type=Path, default=None)
    p.add_argument("--vector-index",  type=Path, default=None)
    p.add_argument("--sq8-index",     type=Path, default=None)
    p.add_argument("--repeats",       type=int,  default=5)
    p.add_argument("--skip-rerank",   action="store_true")

    # ---- benchmark-ann ----
    p = sub.add_parser("benchmark-ann", help="FlatIP vs HNSW speed/quality experiment")
    p.add_argument("--cohort",      type=Path, required=False, default=None)
    p.add_argument("--queries",     type=Path, default=None)
    p.add_argument("--qrels",       type=Path, default=None)
    p.add_argument("--flat-index",  type=Path, default=None)
    p.add_argument("--hnsw-index",  type=Path, default=None)
    p.add_argument("--repeats",     type=int,  default=5)

    # ---- status ----
    sub.add_parser("status", help="Show index, checkpoint, and manifest status")

    return parser


# ------------------------------------------------------------------ entry

COMMANDS = {
    "download":      cmd_download,
    "setup":         cmd_setup,
    "index-bm25":    cmd_index_bm25,
    "index-vector":  cmd_index_vector,
    "index-hnsw":    cmd_index_hnsw,
    "quantize":      cmd_quantize,
    "reset":         cmd_reset,
    "serve":         cmd_serve,
    "test":          cmd_test,
    "smoke":         cmd_smoke,
    "cohort":        cmd_cohort,
    "benchmark":     cmd_benchmark,
    "benchmark-ann": cmd_benchmark_ann,
    "status":        cmd_status,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    fn = COMMANDS.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    fn(args)


if __name__ == "__main__":
    main()
