# Hybrid Search Engine — FAANG-Quality Upgrade Guide

Use this order: **correct data → hermetic tests → honest benchmarks → safe API → reproducible deployment → performance**. Complete and verify one phase before starting the next.

> Snippets show the changed block, with tutorial comments. Keep unrelated existing code. Phase 1 is one atomic migration: do not resume a real build halfway through it.

## Issue map
| Current problem | Required result |
|---|---|
| Tests and imports touch real `data/` | Inject temporary resources; remove import side effects |
| Docstore loses title/category | Persist and read all document metadata |
| Vector builder calls `.get()` on a list | Iterate over each document |
| FAISS uses two ID formats | `IndexIDMap2` everywhere; delete sidecar mapping |
| Zero remaining means unlimited | Explicit `is not None` checks |
| Crash can duplicate indexed rows | Manifest identity + durable-count reconciliation |
| Corrupt metadata/index fails open | Raise; never silently restart/recreate |
| FAISS/checkpoint can be partially written | Temporary file + atomic replace |
| Save cadence misses boundaries | Save by distance from last save |
| Same-size artifacts can be mixed | One generation ID, exact counts, artifact hashes |
| Tests are slow and ambient | Synthetic fakes; pytest only collects `tests/` |
| Benchmark cohorts/metric depths differ | Fixed IDs and correctly named metrics |
| Stale SQ8/counts can be published | Validate searched artifact against manifest |
| Results lack provenance/repeats | Record hashes, revisions, versions, command, hardware |
| Lazy globals duplicate models/engines | FastAPI lifespan and one service container |
| `/health` lies about readiness | Separate liveness/readiness |
| Expensive requests are unbounded | Shared validation, semaphore, gateway limits |
| Responses always contain full passages | Snippet by default; optional body |
| No operational evidence | Logs, Prometheus metrics, Locust run |
| Dependencies/models float | Lock packages; pin model/data revisions |
| CI/container are incomplete | Frozen CI; safe non-root image and `.dockerignore` |
| PostgreSQL/Redis/LTR are unused claims | Fully implement or remove; remove now |
| Flat FAISS scales linearly | Compare exact search with ANN on one cohort |

---

# Phase 0 — Protect current generated data

The current tests can modify the ignored SQLite file, which Git cannot restore.

```powershell
git switch -c harden-search-engine
if (Test-Path hybrid-search-engine\data\docstore.sqlite) {
    Copy-Item hybrid-search-engine\data\docstore.sqlite `
      hybrid-search-engine\data\docstore.backup.sqlite
}
if (Test-Path hybrid-search-engine\data\indexes\vector_checkpoint.json) {
    Copy-Item hybrid-search-engine\data\indexes\vector_checkpoint.json `
      hybrid-search-engine\data\indexes\vector_checkpoint.backup.json
}
```

Do not run the full current suite until Phase 1, Step 1 is complete.

---

# Phase 1 — Correct, isolated, crash-safe artifacts

## Step 1 — Inject storage and preserve metadata

### Issue
`BM25Search(index_path=temp)` still opens the default docstore, so tests overwrite real rows. The current table stores only body text and invents title/category on reads.

### Change
Replace the relevant parts of `src/database/docstore.py`:

```python
class SQLiteDocstore:
    def __init__(self, db_path: Path | None = None, *, read_only: bool = False):
        self.db_path = db_path or DOCSTORE_PATH
        self.read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        if self.read_only:
            # SQLite itself now blocks accidental API writes.
            uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            return sqlite3.connect(uri, uri=True)
        return sqlite3.connect(self.db_path)

    def init(self) -> None:
        if self.read_only:
            return
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    body_compressed BLOB NOT NULL
                )
            """)

    def count_documents(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])
```

Update upsert and read SQL:

```python
# Inside upsert_documents()
conn.executemany("""
    INSERT INTO documents (id, title, category, body_compressed)
    VALUES (?, ?, ?, ?)
    ON CONFLICT(id) DO UPDATE SET
        title=excluded.title,
        category=excluded.category,
        body_compressed=excluded.body_compressed
""", [
    (str(doc["id"]), doc.get("title", ""), doc.get("category", ""),
     self._compress_text(doc.get("body", "")))
    for doc in documents
])

# Inside get_documents_by_ids(): select stored metadata.
rows = conn.execute(f"""
    SELECT id, title, category, body_compressed
    FROM documents WHERE id IN ({placeholders})
""", ids).fetchall()

for row in rows:
    documents[row["id"]] = {
        "id": row["id"],
        "title": row["title"],
        "category": row["category"],
        "body": self._decompress_text(row["body_compressed"]),
    }
```

Inject it into both retrievers:

```python
# BM25Search.__init__
def __init__(self, index_path=None, *, docstore=None, create_if_missing=False):
    self.index_path = index_path or BM25_INDEX_PATH
    self.docstore = docstore or SQLiteDocstore()
    self.docstore.init()
    # Step 5 below supplies safe index opening.

# VectorSearch.__init__
def __init__(self, index_path=None, *, docstore=None, model=None):
    self.index_path = index_path or VECTOR_INDEX_PATH
    self.docstore = docstore or SQLiteDocstore()
    self.docstore.init()
    self.model = model or SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        revision=EMBEDDING_MODEL_REVISION or None,
    )
```

Tests pass temporary resources:

```python
store = SQLiteDocstore(tmp_path / "docstore.sqlite")
engine = BM25Search(
    tmp_path / "bm25",
    docstore=store,
    create_if_missing=True,
)
```

The old database lacks the new columns. Back it up, then rebuild once with Step 6's `reset_all.py`; do not guess old metadata.

### Verify
From `hybrid-search-engine/`:

```powershell
$before = (Get-FileHash data\docstore.sqlite).Hash
pytest tests\test_bm25.py -q
$after = (Get-FileHash data\docstore.sqlite).Hash
$before -eq $after  # Must be True.
```

## Step 2 — Fix vector construction and use one ID format

### Issue
`documents` is a list, so `documents.get()` crashes. A stale positional `.npy` mapping can also conflict with a full `IndexIDMap2` index.

### Change
Use numeric MS MARCO IDs inside FAISS everywhere:

```python
# VectorSearch.build_index()
texts = [
    f"{document.get('title', '')} {document.get('body', '')}"
    for document in documents  # Singular document = one dictionary.
]
embeddings = self._encode(texts)
ids = np.asarray([int(document["id"]) for document in documents], dtype=np.int64)

base = faiss.IndexFlatIP(embeddings.shape[1])
self.index = faiss.IndexIDMap2(base)
self.index.add_with_ids(embeddings, ids)
self.docstore.upsert_documents(documents)
self.save()
```

Search labels are now real passage IDs:

```python
scores, indices = self.index.search(query_embedding, top_k)
for similarity, faiss_id in zip(scores[0], indices[0]):
    if faiss_id == -1:
        continue
    doc_id = str(int(faiss_id))
    doc_ids.append(doc_id)
    scores_by_id[doc_id] = float(similarity)
```

Delete `self.doc_ids` and all `.npy` save/load branches. Update `scripts/index_vectors.py`:

```python
def remove_index(index_path: Path) -> None:
    index_path.unlink(missing_ok=True)
    (index_path.parent / "vector_doc_ids.npy").unlink(missing_ok=True)
```

### Verify
A fake-model test must verify IDs, not only count:

```python
stored_ids = faiss.vector_to_array(vector.index.id_map).tolist()
assert stored_ids == [10, 20]
```

## Step 3 — Make target limits edge-case correct

### Issue
Python treats zero as false. `if max_documents` therefore turns “zero remaining” into “no limit.”

### Change
In `load_msmarco_passages()`:

```python
if max_documents is not None and max_documents < 0:
    raise ValueError("max_documents cannot be negative")
if skip_documents < 0:
    raise ValueError("skip_documents cannot be negative")
if max_documents == 0:
    return

# Every later limit check uses None explicitly.
if max_documents is not None and returned >= max_documents:
    return
```

Treat `--max-docs` as the final target:

```python
if max_documents is not None and durable_count >= max_documents:
    return existing_index  # Do not open a writer or alter state.
remaining = None if max_documents is None else max_documents - durable_count
```

Tests: zero yields nothing; negative raises; rerunning a completed target leaves count unchanged.

## Step 4 — Add one generation manifest and fail-closed state

### Issue
A crash can save an index but not its checkpoint. Count equality alone cannot prove artifacts came from the same corpus/model/build.

### Change
Define identity before any build (`src/config.py`):

```python
EMBEDDING_MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION", "")
CROSS_ENCODER_MODEL_REVISION = os.getenv("CROSS_ENCODER_MODEL_REVISION", "")
PREPROCESSING_VERSION = "1"  # Increment whenever cleaning/validation changes.
```

Create `src/indexing/artifact_state.py`:

```python
import hashlib, json, os
from pathlib import Path


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        digest.update(str(file.relative_to(path.parent)).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_json_required(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable state file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def reconcile_count(checkpoint_count: int, durable_count: int) -> int:
    if checkpoint_count > durable_count:
        raise RuntimeError("Checkpoint is ahead of the durable index")
    return durable_count  # Index may be ahead after commit-before-checkpoint crash.
```

Before the first write, create `artifact_manifest.json`:

```python
manifest = {
    "schema_version": 1,
    "generation_id": str(uuid.uuid4()),
    "status": "building",
    "collection_sha256": sha256_path(collection_path),
    "preprocessing_version": PREPROCESSING_VERSION,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_revision": EMBEDDING_MODEL_REVISION,
    "vector_id_mode": "faiss_id_map_2",
    "target_documents": max_documents,
}
write_json_atomic(manifest_path, manifest)
```

On resume:

```python
manifest = load_json_required(manifest_path)  # Invalid JSON must stop.
# Compare corpus hash, preprocessing version, model, revision and ID mode here.
checkpoint_count = 0
if checkpoint_path.exists():
    checkpoint_count = int(load_json_required(checkpoint_path)["total_documents_indexed"])
durable_count = opened_index_count  # BM25 num_docs or FAISS ntotal.
start_count = reconcile_count(checkpoint_count, durable_count)
```

Replace last-ID skipping with valid-document count skipping inside the existing parse/clean/validate loop:

```python
valid_seen = returned = 0
for document in parsed_documents:
    if valid_seen < start_count:
        valid_seen += 1
        continue
    if remaining is not None and returned >= remaining:
        return
    returned += 1
    yield document
```

`parsed_documents` means documents after current TSV parsing, cleaning, and validation—not a list loaded into memory.

Remove `--no-resume`: starting at row one while retaining an index is unsafe. Allow only validated resume or explicit reset.

When both indexes finish, validate counts and finalize the same manifest:

```python
counts = {"bm25": bm25_count, "vector": vector_count, "docstore": docstore_count}
if len(set(counts.values())) != 1:
    raise RuntimeError(f"Artifact counts disagree: {counts}")

manifest.update({
    "status": "complete",
    "counts": counts,
    "artifact_sha256": {
        "bm25": sha256_path(bm25_path),
        "vector": sha256_path(vector_path),
        "docstore": sha256_path(docstore_path),
    },
})
write_json_atomic(manifest_path, manifest)
```

Keep completed checkpoint metadata. Generation ID + identities + hashes prevent a same-size foreign artifact from being accepted.

## Step 5 — Make saves/opening safe

```python
def save_index_atomic(index: faiss.Index, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    faiss.write_index(index, str(temporary))
    os.replace(temporary, path)

last_saved_count = start_count
if count - last_saved_count >= save_every:
    save_index_atomic(index, index_path)
    write_json_atomic(checkpoint_path, checkpoint_payload)
    last_saved_count = count
```

BM25 serving must never create/delete an index:

```python
if self.index_path.exists():
    try:
        self.index = tantivy.Index.open(str(self.index_path))
    except Exception as exc:
        raise RuntimeError(f"Unreadable BM25 index: {self.index_path}") from exc
elif create_if_missing:
    self.index_path.mkdir(parents=True, exist_ok=True)
    self.index = tantivy.Index(self.schema, path=str(self.index_path))
else:
    raise FileNotFoundError(f"BM25 index not found: {self.index_path}")
```

Only indexing passes `create_if_missing=True`. Never silently delete a corrupt index.

## Step 6 — Add one explicit reset command

Create `scripts/reset_all.py`; it deletes generated artifacts but protects the corpus:

```python
import argparse, shutil
from src.config import DATA_DIR, DOCSTORE_PATH, INDEX_DIR

GENERATED = [
    INDEX_DIR / "bm25", INDEX_DIR / "bm25_compact",
    INDEX_DIR / "vector.faiss", INDEX_DIR / "vector.sq8.faiss",
    INDEX_DIR / "vector_doc_ids.npy", INDEX_DIR / "vector_checkpoint.json",
    INDEX_DIR / "artifact_manifest.json", DOCSTORE_PATH,
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    if not args.yes:
        raise SystemExit("Refusing destructive reset without --yes")

    data_root = DATA_DIR.resolve()
    for path in GENERATED:
        resolved = path.resolve()
        if not resolved.is_relative_to(data_root):
            raise RuntimeError(f"Unsafe reset path: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

Run it once, then build BM25/docstore and vector under one new generation. Never include `data/msmarco/collection.tsv` in `GENERATED`.

### Phase 1 verification
- Crash before/after index replacement; resume creates no duplicates.
- Checkpoint ahead fails; index ahead resumes from durable count.
- Changed corpus/model/preprocessing fails.
- Complete manifest has one generation, equal counts, and matching hashes.
- Corrupt index/checkpoint fails without deletion.

---

# Phase 2 — Hermetic tests and CI

## Step 1 — Stop import/test side effects

Current `scripts/test_*.py` programs can execute real indexing. Rename them `smoke_*.py`, add `if __name__ == "__main__":`, or delete duplicates.

Add to `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

Remove `DATA_DIR.mkdir()`, `MODELS_DIR.mkdir()`, and `INDEX_DIR.mkdir()` from `src/config.py`. Replace them with a function called only by build/download CLIs:

```python
def ensure_build_directories() -> None:
    for path in (DATA_DIR, MODELS_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
```

Move spell-dictionary/model loading out of route import (Phase 4). Set `show_progress_bar=False` in request-time vector/reranker calls.

## Step 2 — Test with small fakes

```python
class FakeBM25:
    def search(self, query: str, top_k: int):
        return [{"id": "1", "title": "Python", "body": "Python builds APIs.",
                 "category": "test", "score": 1.0}][:top_k]


def test_search_uses_fake(monkeypatch, client):
    monkeypatch.setattr("src.api.routes.search.get_bm25", lambda: FakeBM25())
    response = client.get("/search?q=python")
    assert response.status_code == 200
    assert response.json()["results"][0]["id"] == "1"
```

Cover: RRF bounds/ties, real FAISS IDs, docstore round-trip/read-only mode, every resume failure, all routes, busy/disabled reranker, and hand-calculated metrics. Goal: offline, no `data/` access, under 15 seconds.

## Step 3 — Add CI only after local checks pass

```powershell
ruff check src tests scripts Benchmark
pytest -q
```

`.github/workflows/ci.yml`:

```yaml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: {python-version: "3.11", cache: pip}
      - run: python -m pip install -e ".[dev]"
      - run: python -m ruff check src tests scripts Benchmark
      - run: pytest -q
```

Phase 5 changes CI installation to a frozen lock.

---

# Phase 3 — Trustworthy benchmark evidence

## Step 1 — One tested metric module

Move NDCG/MRR/Recall to `src/evaluation/metrics.py`; both evaluation scripts import it. Delete duplicate implementations.

```python
def mrr_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
```

Test metrics with tiny rankings whose answers you calculate by hand.

## Step 2 — Fixed cohorts and correct depths

Commit cohort JSON containing query IDs plus query/qrels hashes. Load every ID or fail—never silently filter it. If reranking 500 queries is expensive, create a fixed 100-query rerank cohort and rerun **all baselines and reranking** on those 100.

- Retrieval table: NDCG@10, MRR@10, Recall@100.
- Ten-result rerank table: NDCG@10, MRR@10, Recall@10.
- Compare corpus sizes only when they share a cohort.

Add explicit CLI inputs:

```python
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--cohort", type=Path, required=True)
parser.add_argument("--vector-index", type=Path, required=True)
parser.add_argument("--sq8-index", type=Path)
parser.add_argument("--rrf-k", type=int, default=60)
parser.add_argument("--repeats", type=int, default=5)
```

## Step 3 — Refuse stale evidence

```python
manifest = load_json_required(args.manifest)
if manifest.get("status") != "complete":
    raise RuntimeError("Benchmark requires a complete generation")

counts = document_counts()
if any(counts.get(name) != args.corpus_size for name in ("bm25", "vector", "docstore")):
    raise RuntimeError(f"Artifact count mismatch: {counts}")

if args.sq8_index:
    sq8 = faiss.read_index(str(args.sq8_index))
    if sq8.ntotal != args.corpus_size:
        raise RuntimeError("SQ8 artifact is stale")
```

Also compare selected artifact hashes to the manifest. Search `args.vector_index`; do not headline SQ8 size unless the benchmark and API use that exact SQ8 file.

## Step 4 — Save provenance, repeat latency, publish atomically

Store: result schema version, UTC time, Git SHA/dirty state, `sys.argv`, platform/CPU count, cohort/query/qrels hashes, generation ID, model revisions, RRF parameters, package versions, and repeat count.

Run at least five repeats, rotate system order, and report median p50/p95. Write JSON/Markdown via temporary file + `os.replace()`.

Delete stale generated reports (`2MA` and broken links included), rerun, and reject any result whose `corpus_label` differs from its filename stem.

---

# Phase 4 — Production-shaped API

## Step 1 — App factory, startup validation, one shared engine set

Move spell loading out of module import. Define `ENABLE_RERANKER` in config. Use an app factory so tests pass fake services:

```python
@dataclass
class SearchServices:
    spell: SpellCorrector
    bm25: BM25Search
    hybrid: HybridSearchEngine
    reranker: CrossEncoderReranker | None
    rerank_slots: BoundedSemaphore


def load_services() -> SearchServices:
    manifest = verify_complete_manifest_and_hashes(INDEX_DIR / "artifact_manifest.json")
    store = SQLiteDocstore(DOCSTORE_PATH, read_only=True)
    bm25 = BM25Search(docstore=store, create_if_missing=False)
    vector = VectorSearch(docstore=store)
    vector.load()
    actual = {"bm25": bm25.committed_document_count(),
              "vector": vector.index.ntotal, "docstore": store.count_documents()}
    if actual != manifest["counts"]:
        raise RuntimeError(f"Serving count mismatch: {actual}")

    spell = SpellCorrector()
    spell.load_default_dictionary()
    reranker = CrossEncoderReranker() if ENABLE_RERANKER else None
    if reranker is not None:
        _ = reranker.model  # Enabled means ready before traffic.

    return SearchServices(spell, bm25,
        HybridSearchEngine(bm25_search=bm25, vector_search=vector),
        reranker, BoundedSemaphore(1))


def create_app(loader: Callable[[], SearchServices] = load_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        try:
            app.state.services = loader()
            app.state.ready = True
            yield
        finally:
            app.state.ready = False

    application = FastAPI(lifespan=lifespan)
    application.include_router(search_router)
    return application


app = create_app()
```

`verify_complete_manifest_and_hashes()` is a thin wrapper around Phase 1's loader/hash helper: require `status=complete`, ensure required identity/count/hash keys exist, and compare every deployed path's SHA-256.

Routes get services through `Request`/`Depends`; delete lazy module globals. Tests call `create_app(lambda: fake_services)`.

## Step 2 — Validate shared logic and request boundaries

Put RRF checks inside `reciprocal_rank_fusion()`, so direct callers are safe:

```python
if k <= 0 or top_k < 0:
    raise ValueError("Invalid RRF depth")
if not all(math.isfinite(w) and 0 <= w <= 100 for w in (bm25_weight, vector_weight)):
    raise ValueError("Weights must be finite and between 0 and 100")
if bm25_weight == vector_weight == 0:
    raise ValueError("At least one weight must be positive")
if top_k == 0:
    return []
```

Use `max_length=256` for `q`; require `candidates_k >= top_k`; retain result/candidate upper bounds. Apply a non-blocking semaphore around reranking and return 429 when busy. Return 503 when reranking is disabled.

Rename misleading response `total` to `returned_count`. In all three result models make `body: str | None = None`; all routes accept `include_body=False` and return snippets by default.

## Step 3 — Real health, logging, metrics, and gateway limits

```python
@app.get("/health/live")
def live():
    return {"status": "alive"}

@app.get("/health/ready")
def ready(request: Request):
    if not getattr(request.app.state, "ready", False):
        raise HTTPException(503, "Search services are not ready")
    return {"status": "ready"}
```

Add `prometheus-client` through the lock. Mount `/metrics`; count requests and histogram latency using method, route template, and status labels—never raw query text.

At the gateway, commit concrete config: request timeout, connection timeout, and per-IP rate/burst limit. Authentication is optional for a public read-only demo; rate limiting is not. Configure CORS origins from an environment list instead of hardcoding localhost.

Create a real `locustfile.py` for hybrid/rerank traffic. Save one reproducible run's users, duration, throughput, failures, p50/p95/p99, CPU, and memory.

---

# Phase 5 — Reproducible build and deployment

## Step 1 — Remove dead scope, then lock

Remove unused PostgreSQL/Redis/SQLAlchemy drivers, LightGBM/LTR, their Compose services, and README claims. Audit nltk, pandas, Jupyter, matplotlib, scikit-learn, direct torch, and pytest-asyncio before keeping them. Keep one downloader; delete duplicate HF scripts or deliberately declare/test `datasets`. Move test-only packages to dev dependencies.

Then create a cross-platform lock:

```powershell
python -m pip install "uv==<REVIEWED_FIXED_VERSION>"
uv lock
uv sync --frozen --extra dev
```

Commit `uv.lock`. CI uses `uv sync --frozen --extra dev`; production uses `uv sync --frozen --no-dev`. Build/index commands reject blank model revisions; unit tests inject fakes. Verify downloaded corpus archives against committed SHA-256 values.

## Step 2 — Safe container context and runtime

`.dockerignore`:

```text
.git
.venv
**/__pycache__
.pytest_cache
data
models
Benchmark/results
*.sqlite
*.faiss
```

Dockerfile (replace placeholders with reviewed immutable digests):

```dockerfile
FROM python:3.11.9-slim@sha256:<PINNED_DIGEST>
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:<PINNED_TAG>@sha256:<PINNED_DIGEST> /uv /bin/uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src

RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser
HEALTHCHECK CMD /app/.venv/bin/python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready')"
CMD ["/app/.venv/bin/python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

Mount `data/` and exact-revision model snapshots read-only. Set `EMBEDDING_MODEL`/`CROSS_ENCODER_MODEL` to those local paths and verify model-directory checksums in the deployment manifest. Do not download models at runtime. One worker avoids loading another multi-GB index; scale replicas behind a load balancer.

---

# Phase 6 — One measured ANN experiment

After Phases 1–5 pass, compare exact FlatIP with HNSW:

```python
base = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
base.hnsw.efConstruction = 200
index = faiss.IndexIDMap2(base)
index.add_with_ids(embeddings, ids)

# IDMap2 wraps HNSW; unwrap it before setting query effort.
inner = faiss.downcast_index(index.index)
inner.hnsw.efSearch = 64
```

On the same cohort/generation, report NDCG@10, Recall@100, p50, p95, RAM, and disk for FlatIP, HNSW `efSearch` 32/64/128, and the actual deployed SQ8/IVF-PQ candidate. Do not claim a performance/storage win unless the API loads that exact artifact.

---

# Phase 7 — Publish verified evidence

Until every check below passes, call it a **portfolio-grade hybrid search prototype**, not production-ready. README should contain a tiny synthetic demo, locked setup, architecture/failure behaviour, fixed-cohort reports, generation/count/index type, CI/test/load evidence, hardware/cost, limitations, ANN trade-off, and license.

## Final acceptance checklist
- [ ] Tests/imports never access `data/`, network, or production model cache.
- [ ] Pytest collects only `tests/`; smoke programs have main guards.
- [ ] Docstore preserves metadata and is read-only in serving.
- [ ] One FAISS ID format is used; IDs and counts are tested.
- [ ] Zero/negative/completed targets and crash boundaries are tested.
- [ ] Invalid index/checkpoint/manifest fails without deletion.
- [ ] Complete manifest has generation, identities, counts, and hashes.
- [ ] BM25/vector/docstore/deployed compressed index match that manifest.
- [ ] CI lint/tests pass from a clean clone.
- [ ] Metrics have one tested implementation and fixed cohorts/depths.
- [ ] Benchmark records command, hashes, revisions, versions, hardware, repeats.
- [ ] API preloads one shared validated service set and exposes readiness.
- [ ] Query, weights, candidates, body size, concurrency, timeout, and rate are bounded.
- [ ] Logs, Prometheus metrics, and a saved Locust procedure exist.
- [ ] Lock, model/data checksums, `.dockerignore`, and non-root image exist.
- [ ] Unused database/cache/LTR dependencies and claims are removed.
- [ ] README numbers come only from the corrected benchmark.

When these are green, your interview story is strong: **you found data-integrity risks, made artifacts crash-safe, rebuilt honest evidence, hardened serving, and measured a real quality/latency trade-off.**
