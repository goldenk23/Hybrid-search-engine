# Hybrid Search Engine — Plain-English Upgrade Guide

This is a simplified companion to `FAANG_UPGRADE_GUIDE.md`. Same plan, same steps, same
end result — but written so you can actually understand *why* each change matters for
*your* project. The original file stays as the precise reference; use this one to learn.

## How to read this guide

Each section follows the same shape:

- **The problem in plain words** — what's broken, using an everyday analogy.
- **Why it matters** — what goes wrong if you ignore it.
- **What you'll change** — the actual fix (code snippets are the real instructions; keep
  them as-is).
- **How you'll know it worked** — the check that proves it.

### A quick glossary you'll need throughout

These terms show up everywhere in the original guide. Learn them once here:

- **Artifact** — any file your build *produces*: the BM25 index, the FAISS vector index,
  the SQLite docstore. Think of them as the "baked goods" your indexing pipeline outputs.
  The raw MS MARCO corpus is the *ingredients*, not an artifact.
- **Docstore** (`data/docstore.sqlite`) — your local database that keeps the actual text
  of each passage so you can show it in results. FAISS and BM25 only store *math and
  IDs*; the readable text lives here.
- **BM25** — the keyword-matching search (matches literal words, like classic Google).
- **Vector / semantic search** — search by *meaning* using embeddings (numbers that
  capture what text means). Powered by FAISS.
- **FAISS** — Facebook's library for finding the closest vectors fast. Your semantic index.
- **Embedding** — a list of numbers representing a piece of text's meaning. Similar
  meanings produce similar numbers.
- **Manifest** — a small "receipt" file describing exactly which corpus, model, and
  settings produced a set of artifacts. Proof they belong together.
- **Checkpoint** — a "save point" written during a long indexing run so a crash doesn't
  force you to start from zero.
- **Hermetic test** — a test that runs in a sealed bubble: no touching real data, no
  network, no surprises. Same result every time.
- **Idempotent / crash-safe** — you can run it again after a crash and it heals itself
  instead of creating duplicates or corruption.

---

## The order matters — don't skip ahead

The plan runs in this exact order, and each phase depends on the one before it:

**correct data → sealed tests → honest benchmarks → safe API → reproducible deploy → performance**

The reason is simple: you can't measure how *good* your search is (benchmarks) if the
underlying data is *wrong* (correctness). Measuring on broken data just gives you
confident nonsense. So fix the foundation first.

Finish and verify one phase before starting the next.

---

# The problems, in one table

Before the details, here's every issue the guide fixes, translated into plain words:

| What's wrong now | What "fixed" looks like |
|---|---|
| Tests and imports touch your **real** `data/` folder | Tests use throwaway temp folders; importing code doesn't create folders |
| The docstore throws away each passage's title and category | Store title and category properly, read them back properly |
| The vector builder loops over a list wrong and crashes | Loop over each document correctly |
| FAISS uses two different ID systems that can clash | Use one ID system everywhere; delete the extra mapping file |
| "Zero documents" is treated as "unlimited documents" | Check for zero explicitly, so zero means zero |
| A crash mid-build can index the same rows twice | A "receipt" + count check make re-runs safe |
| Corrupt files silently get recreated, hiding the problem | Corrupt files raise a loud error instead |
| A file can be half-written when the power dies | Write to a temp file, then swap it in atomically |
| Saves happen at the wrong moments | Save based on how much work happened since last save |
| Two index files of the same size could get mixed up | Stamp each build with an ID + exact counts + fingerprints |
| Tests are slow and depend on your environment | Fake, tiny stand-ins; tests only run from `tests/` |
| Benchmarks compare apples to oranges | Fixed question sets and correctly-named scores |
| Old, stale index files can sneak into results | Check the file matches the receipt before trusting it |
| Benchmark results don't say how they were made | Record fingerprints, versions, the exact command, the hardware |
| The app loads models multiple times, wasting memory | Load everything once at startup, share it |
| `/health` says "OK" even when search isn't ready | Separate "am I alive" from "am I ready to serve" |
| Expensive requests have no limits | Shared validation + a concurrency cap + gateway limits |
| Every response ships the full passage text | Return a short snippet by default; full text only on request |
| No proof the system actually works under load | Logs, metrics, and a saved load-test run |
| Dependency and model versions drift over time | Pin exact versions of packages, models, and data |
| The build recipe (CI/Docker) is incomplete | Frozen versions, a safe non-root container |
| README claims PostgreSQL/Redis/LTR that aren't used | Remove the unused claims and code |
| Flat search gets slower as the corpus grows | Compare exact search vs an approximate (faster) index |

Now the details, phase by phase.

---

# Phase 0 — Protect what you already built (do this first)

**The problem in plain words.** Your `data/docstore.sqlite` is in `.gitignore`, which
means Git is *not* tracking it. If a test accidentally overwrites it, Git can't bring it
back — there's no undo. It's like editing a document that has no version history.

**Why it matters.** The current tests can write to that real file. One bad run and the
data you spent hours indexing is gone.

**What you'll do.** Make a branch and copy the important files to `.backup` versions
before touching anything:

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

**Rule:** don't run the full existing test suite until Phase 1 Step 1 is done, because
those tests are exactly what can corrupt your data.

---

# Phase 1 — Make the data correct, isolated, and crash-safe

This is the most important phase. Everything else is polish; this is the foundation.

## Step 1 — Stop tests from clobbering real data, and store metadata properly

**The problem in plain words.** Two bugs here:

1. Even when a test says "use this temporary folder for the index," your code *still*
   opens the real default docstore behind the scenes. So the test quietly writes to your
   production data. Imagine renting a practice kitchen but the oven is secretly wired to
   your home oven.
2. Your docstore only saves the passage *body*. It doesn't save the title or category.
   So when you read a document back, the code *fakes* them — I confirmed this in your
   `docstore.py`: `"title": body[:100]` (just the first 100 characters of the body) and
   `"category": "msmarco"` (hardcoded). Results look real but the metadata is invented.

**Why it matters.** Fake metadata means your search results lie, and your benchmarks
score against lies. And silent writes to real data make tests dangerous to run.

**What you'll change.** In `src/database/docstore.py`, let the docstore accept a path and
a read-only mode, and actually store title + category:

```python
class SQLiteDocstore:
    def __init__(self, db_path: Path | None = None, *, read_only: bool = False):
        self.db_path = db_path or DOCSTORE_PATH
        self.read_only = read_only
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        if self.read_only:
            # "mode=ro" tells SQLite itself to reject any write. A safety lock at
            # the database level, not just a promise in our code.
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

Then save and read the real title/category instead of faking them:

```python
# Inside upsert_documents(): now also stores title and category.
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

# Inside get_documents_by_ids(): read the stored metadata, don't invent it.
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

**Key idea: dependency injection.** Instead of each search class *creating* its own
docstore in secret, you *hand it* one from outside. That's called dependency injection —
"don't build your own tools, accept them as arguments." It lets tests hand over a fake or
temporary one. Update both retrievers:

```python
# BM25Search.__init__
def __init__(self, index_path=None, *, docstore=None, create_if_missing=False):
    self.index_path = index_path or BM25_INDEX_PATH
    self.docstore = docstore or SQLiteDocstore()
    self.docstore.init()

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

Now a test can pass its own throwaway docstore and index folder:

```python
store = SQLiteDocstore(tmp_path / "docstore.sqlite")
engine = BM25Search(tmp_path / "bm25", docstore=store, create_if_missing=True)
```

**Note on your old database:** it doesn't have the new title/category columns. Don't try
to guess them. Back it up (Phase 0), then rebuild fresh with the reset script from Step 6.

**How you'll know it worked.** Run a test and confirm the real docstore's fingerprint
(hash) is unchanged — proof the test didn't touch it:

```powershell
$before = (Get-FileHash data\docstore.sqlite).Hash
pytest tests\test_bm25.py -q
$after = (Get-FileHash data\docstore.sqlite).Hash
$before -eq $after  # Must print True.
```

## Step 2 — Fix the vector builder crash and use one ID system

**The problem in plain words.** Two issues in `src/search/vector.py`:

1. The build loop is written as `for documents in documents` — it reuses the same name
   for the list *and* each item. It works by luck right now but it's a trap waiting to
   break. (The original guide flags this as "`.get()` on a list.")
2. You have two ways of remembering which vector belongs to which passage:
   - a separate `vector_doc_ids.npy` file that maps "position 5 → passage 12345", and
   - the modern approach where FAISS stores the real passage ID *inside itself*.
   Keeping both is like having two clocks showing different times — eventually they
   disagree and you don't know which to trust.

**Why it matters.** A crash or a mismatch between those two ID systems gives you wrong
search results that are very hard to debug.

**What you'll change.** Pick one system: store the real MS MARCO passage ID *inside*
FAISS using `IndexIDMap2` (a FAISS wrapper that lets you attach your own IDs to vectors
instead of FAISS's internal 0,1,2… positions):

```python
# VectorSearch.build_index()
texts = [
    f"{document.get('title', '')} {document.get('body', '')}"
    for document in documents  # "document" (singular) = one dictionary. Clear names.
]
embeddings = self._encode(texts)
ids = np.asarray([int(document["id"]) for document in documents], dtype=np.int64)

base = faiss.IndexFlatIP(embeddings.shape[1])
self.index = faiss.IndexIDMap2(base)   # the wrapper that remembers YOUR ids
self.index.add_with_ids(embeddings, ids)
self.docstore.upsert_documents(documents)
self.save()
```

At search time, FAISS now hands back the real passage IDs directly:

```python
scores, indices = self.index.search(query_embedding, top_k)
for similarity, faiss_id in zip(scores[0], indices[0]):
    if faiss_id == -1:      # -1 means "no more results"
        continue
    doc_id = str(int(faiss_id))
    doc_ids.append(doc_id)
    scores_by_id[doc_id] = float(similarity)
```

Then **delete** `self.doc_ids` and every bit of `.npy` save/load code — the second clock
is gone. Update `scripts/index_vectors.py` to clean up the old file too:

```python
def remove_index(index_path: Path) -> None:
    index_path.unlink(missing_ok=True)
    (index_path.parent / "vector_doc_ids.npy").unlink(missing_ok=True)
```

**How you'll know it worked.** A test that checks the *actual IDs* stored, not just the
count:

```python
stored_ids = faiss.vector_to_array(vector.index.id_map).tolist()
assert stored_ids == [10, 20]
```

## Step 3 — Make "how many documents" limits handle edge cases

**The problem in plain words.** In Python, the number `0` counts as "false" in an `if`.
So `if max_documents:` is secretly `if max_documents is not zero`. That means asking for
*zero* documents accidentally means *no limit — index everything*. A dangerous surprise.

**Why it matters.** A `--max-docs 0` (or a calculation that lands on zero remaining)
could kick off indexing your entire 8-million-passage corpus by accident.

**What you'll change.** In `load_msmarco_passages()`, check for `None` and negatives
explicitly instead of relying on truthiness:

```python
if max_documents is not None and max_documents < 0:
    raise ValueError("max_documents cannot be negative")
if skip_documents < 0:
    raise ValueError("skip_documents cannot be negative")
if max_documents == 0:
    return   # zero means zero: index nothing

# Every later limit check is explicit about None:
if max_documents is not None and returned >= max_documents:
    return
```

Treat `--max-docs` as the *final total you want*, not "how many more to add":

```python
if max_documents is not None and durable_count >= max_documents:
    return existing_index  # already have enough; don't touch anything
remaining = None if max_documents is None else max_documents - durable_count
```

**How you'll know it worked.** Tests: zero yields nothing; negative raises an error;
re-running an already-complete target leaves the count unchanged.

## Step 4 — Add a "receipt" so artifacts can't get mixed up, and fail loudly

**The problem in plain words.** Indexing 8M passages takes a long time. Two things can go
wrong:

1. A crash might save the FAISS index but *not* its checkpoint (or vice versa), leaving
   the two out of sync.
2. Two index files can be the *same size* but come from *different corpora or models*.
   Size alone can't prove they belong together — like two identical-looking keys that
   open different doors.

**Why it matters.** If you can't prove your BM25 index, vector index, and docstore all
came from the same build, your search and your benchmarks are built on sand.

**What you'll change.** Introduce a **manifest** — a receipt written before the build,
finalized after. First, define what identifies a build (`src/config.py`):

```python
EMBEDDING_MODEL_REVISION = os.getenv("EMBEDDING_MODEL_REVISION", "")
CROSS_ENCODER_MODEL_REVISION = os.getenv("CROSS_ENCODER_MODEL_REVISION", "")
PREPROCESSING_VERSION = "1"  # bump this whenever you change text cleaning/validation
```

Then create `src/indexing/artifact_state.py` with four small helpers. Read the comments —
each one solves a specific failure:

```python
import hashlib, json, os
from pathlib import Path


def sha256_path(path: Path) -> str:
    # A fingerprint of a file (or folder). Same bytes -> same fingerprint.
    # Lets us prove a file hasn't changed or been swapped.
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    for file in files:
        digest.update(str(file.relative_to(path.parent)).encode())
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    # Write to a temp file first, then rename. A rename is instant and can't be
    # "half done", so you never get a half-written receipt if the power dies.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def load_json_required(path: Path) -> dict:
    # If the receipt is missing or corrupt, STOP. Don't guess, don't rebuild silently.
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable state file: {path}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def reconcile_count(checkpoint_count: int, durable_count: int) -> int:
    # "durable" = actually saved in the index on disk.
    # If the checkpoint claims MORE than the index actually has, something's wrong -> stop.
    # If the index has more (a crash after saving index but before checkpoint), trust the index.
    if checkpoint_count > durable_count:
        raise RuntimeError("Checkpoint is ahead of the durable index")
    return durable_count
```

Before the first write, stamp the receipt with everything that identifies this build:

```python
manifest = {
    "schema_version": 1,
    "generation_id": str(uuid.uuid4()),   # a unique ID for THIS build
    "status": "building",
    "collection_sha256": sha256_path(collection_path),  # fingerprint of the corpus
    "preprocessing_version": PREPROCESSING_VERSION,
    "embedding_model": EMBEDDING_MODEL_NAME,
    "embedding_revision": EMBEDDING_MODEL_REVISION,
    "vector_id_mode": "faiss_id_map_2",
    "target_documents": max_documents,
}
write_json_atomic(manifest_path, manifest)
```

When you resume after a crash, read the receipt and make sure nothing changed:

```python
manifest = load_json_required(manifest_path)   # corrupt receipt -> stop, don't guess
# Compare corpus fingerprint, preprocessing version, model, revision, ID mode here.
checkpoint_count = 0
if checkpoint_path.exists():
    checkpoint_count = int(load_json_required(checkpoint_path)["total_documents_indexed"])
durable_count = opened_index_count       # BM25 doc count or FAISS ntotal
start_count = reconcile_count(checkpoint_count, durable_count)
```

Instead of "skip until the last ID I saw," skip by *counting valid documents* — more
robust:

```python
valid_seen = returned = 0
for document in parsed_documents:   # documents AFTER parsing/cleaning/validation
    if valid_seen < start_count:
        valid_seen += 1
        continue
    if remaining is not None and returned >= remaining:
        return
    returned += 1
    yield document
```

**Remove `--no-resume`.** Starting from row one while keeping an existing index is unsafe
(you'd double-index). Only two options: safely resume, or fully reset (Step 6).

When both indexes finish, verify all three counts match and finalize the receipt:

```python
counts = {"bm25": bm25_count, "vector": vector_count, "docstore": docstore_count}
if len(set(counts.values())) != 1:      # all three must be equal
    raise RuntimeError(f"Artifact counts disagree: {counts}")

manifest.update({
    "status": "complete",
    "counts": counts,
    "artifact_sha256": {                # fingerprint each finished artifact
        "bm25": sha256_path(bm25_path),
        "vector": sha256_path(vector_path),
        "docstore": sha256_path(docstore_path),
    },
})
write_json_atomic(manifest_path, manifest)
```

Now the generation ID + fingerprints + matching counts make it impossible for a
same-size foreign file to sneak in unnoticed.

## Step 5 — Never leave a half-written file, never touch a healthy index while serving

**The problem in plain words.** Saving a big FAISS index isn't instant. If the process
dies mid-save, you get a corrupt half-file. And on the *serving* side, your BM25 code can
currently *create or delete* an index — a search server should only ever *read*.

**Why it matters.** A corrupt index that gets silently recreated hides a real problem and
can wipe hours of work.

**What you'll change.** Save atomically (temp file → swap), and save based on how much
new work happened:

```python
def save_index_atomic(index: faiss.Index, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    faiss.write_index(index, str(temporary))
    os.replace(temporary, path)         # instant swap; no half-written file

last_saved_count = start_count
if count - last_saved_count >= save_every:   # save every N new docs
    save_index_atomic(index, index_path)
    write_json_atomic(checkpoint_path, checkpoint_payload)
    last_saved_count = count
```

Make BM25 serving read-only and loud about corruption:

```python
if self.index_path.exists():
    try:
        self.index = tantivy.Index.open(str(self.index_path))
    except Exception as exc:
        raise RuntimeError(f"Unreadable BM25 index: {self.index_path}") from exc
elif create_if_missing:                          # only the indexer sets this True
    self.index_path.mkdir(parents=True, exist_ok=True)
    self.index = tantivy.Index(self.schema, path=str(self.index_path))
else:
    raise FileNotFoundError(f"BM25 index not found: {self.index_path}")
```

**Rule:** only the indexing script passes `create_if_missing=True`. Serving never
creates or deletes. Never silently delete a corrupt index — raise instead.

## Step 6 — One clear "reset everything" command

**The problem in plain words.** Right now there's no safe, obvious way to wipe generated
files and start fresh. People end up deleting things by hand and risk nuking the corpus.

**What you'll change.** Create `scripts/reset_all.py` that deletes only *generated*
artifacts, refuses to run without an explicit `--yes`, and guards against deleting
anything outside `data/`:

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
        if not resolved.is_relative_to(data_root):     # never delete outside data/
            raise RuntimeError(f"Unsafe reset path: {resolved}")
        if resolved.is_dir():
            shutil.rmtree(resolved)
        else:
            resolved.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

**Critical:** never put `data/msmarco/collection.tsv` (your raw corpus) in that list.
That's the one thing you can't regenerate cheaply. Run reset once, then rebuild BM25 +
docstore + vector together under one new receipt.

### Phase 1 finish line — prove all of this

- Kill the process before *and* after an index swap; resuming makes no duplicates.
- A checkpoint that's ahead of the index fails; an index ahead of the checkpoint resumes.
- Changing the corpus/model/preprocessing is detected and fails.
- The finished receipt has one generation ID, three equal counts, and matching fingerprints.
- A corrupt index or checkpoint fails loudly *without* being deleted.

---

# Phase 2 — Tests that run in a sealed bubble

**The big idea.** A good test is *hermetic*: it doesn't touch your real data, doesn't hit
the network, doesn't depend on your machine's mood. It gives the same answer every time,
in seconds. Right now your tests do the opposite — some can trigger real indexing.

## Step 1 — Stop code from doing work just by being imported

**The problem in plain words.** Your `scripts/test_*.py` files can *run real indexing*
when imported, and `src/config.py` *creates folders* the moment it's imported (I confirmed
the `DATA_DIR.mkdir()` / `MODELS_DIR.mkdir()` / `INDEX_DIR.mkdir()` calls at the top).
Importing a module should be quiet — like opening a book shouldn't rearrange your room.

**What you'll change.**

Rename the script-style tests to `smoke_*.py` and wrap their action in a
`if __name__ == "__main__":` guard so importing them does nothing. Then tell pytest to
only look in the real `tests/` folder (add to `pyproject.toml`):

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

Remove the three `mkdir()` calls from `config.py`. Replace them with a function that only
the build/download scripts call on purpose:

```python
def ensure_build_directories() -> None:
    for path in (DATA_DIR, MODELS_DIR, INDEX_DIR):
        path.mkdir(parents=True, exist_ok=True)
```

Also set `show_progress_bar=False` for request-time model calls (progress bars belong in
batch jobs, not inside a web request).

## Step 2 — Test with tiny fakes instead of real models

**The problem in plain words.** Loading the real embedding model in a test is slow and
depends on downloads. Instead, hand the code a *fake* that returns a fixed answer — you're
testing *your logic*, not the model.

**What you'll change.** A fake retriever is just a class with a `search` method:

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

**Cover these cases:** RRF fusion (bounds and ties), real FAISS IDs, docstore round-trip
and read-only mode, every crash/resume failure, all API routes, reranker busy/disabled,
and metrics you calculated by hand. **Goal:** fully offline, no `data/` access, under 15
seconds total.

## Step 3 — Add CI (automatic checks on every push) only after local passes

**The idea.** CI = Continuous Integration: a robot that runs your linter and tests on
every push, so broken code gets caught before it spreads. Get it green locally first:

```powershell
ruff check src tests scripts Benchmark
pytest -q
```

Then add `.github/workflows/ci.yml`:

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

(Phase 5 will switch this to a frozen, exact-version install.)

---

# Phase 3 — Benchmarks you can actually trust

**The big idea.** A benchmark is only meaningful if it's *fair* and *reproducible*. Same
questions, same scoring, recorded so anyone can repeat it. Otherwise it's just a number
you made up.

Quick definitions of the scores you'll report:

- **NDCG@10** — "of the top 10 results, how well-ranked are the relevant ones?" Rewards
  putting good answers near the top. (1.0 = perfect.)
- **MRR@10** — "how high up is the *first* correct answer?" (1/rank of first hit.)
- **Recall@100** — "of all the correct answers, how many did we find in the top 100?"

## Step 1 — One tested copy of the scoring math

**The problem in plain words.** The scoring formulas are duplicated across scripts. Two
copies drift apart and you can't trust either.

**What you'll change.** Move them to one file, `src/evaluation/metrics.py`, and have every
script import from it:

```python
def mrr_at_k(ids: list[str], relevant: set[str], k: int) -> float:
    for rank, doc_id in enumerate(ids[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0
```

Test each metric with a tiny ranking whose answer you worked out by hand.

## Step 2 — Fixed question sets and correct depths

**The problem in plain words.** If different runs use different questions, comparisons are
meaningless. Lock the exact list of query IDs into a committed file (a "cohort").

**What you'll do.** Commit a cohort JSON with the query IDs plus fingerprints of the
queries and answer-key. Load *every* ID or fail — never silently drop questions. Reranking
is expensive, so if needed make a fixed 100-query rerank cohort and run *all* systems on
those same 100.

- Retrieval table: NDCG@10, MRR@10, Recall@100.
- Reranking table (10 results): NDCG@10, MRR@10, Recall@10.
- Only compare corpus sizes when they share the same cohort.

Make every input an explicit command-line argument (no hidden defaults):

```python
parser.add_argument("--manifest", type=Path, required=True)
parser.add_argument("--cohort", type=Path, required=True)
parser.add_argument("--vector-index", type=Path, required=True)
parser.add_argument("--sq8-index", type=Path)
parser.add_argument("--rrf-k", type=int, default=60)
parser.add_argument("--repeats", type=int, default=5)
```

## Step 3 — Refuse to benchmark stale or mismatched data

**The problem in plain words.** It's easy to accidentally benchmark an old index. Guard
against it: check the receipt says "complete" and the counts match before running.

```python
manifest = load_json_required(args.manifest)
if manifest.get("status") != "complete":
    raise RuntimeError("Benchmark requires a complete generation")

counts = document_counts()
if any(counts.get(name) != args.corpus_size for name in ("bm25", "vector", "docstore")):
    raise RuntimeError(f"Artifact count mismatch: {counts}")

if args.sq8_index:                       # SQ8 = a compressed vector index (Phase 6)
    sq8 = faiss.read_index(str(args.sq8_index))
    if sq8.ntotal != args.corpus_size:
        raise RuntimeError("SQ8 artifact is stale")
```

Also compare the artifact fingerprints to the receipt. Benchmark the *exact* index your
API serves — don't quote a compressed index's size unless that's what you actually use.

## Step 4 — Record how the benchmark was run, and repeat it

**The idea.** A trustworthy result carries its own paperwork so anyone can reproduce it.

**What to save with every result:** schema version, UTC timestamp, Git commit (and whether
the repo had uncommitted changes), the exact command (`sys.argv`), platform + CPU count,
cohort/query/answer-key fingerprints, generation ID, model revisions, RRF settings,
package versions, and how many repeats.

**On timing:** run at least 5 times, rotate the order of systems, and report the *median*
p50 and p95 latency (median smooths out random spikes). Write the JSON/Markdown output
atomically (temp file → swap). Delete old stale reports and reject any result whose label
doesn't match its filename.

---

# Phase 4 — An API shaped like a real service

**The big idea.** A production API loads its heavy models *once* at startup, shares them,
checks it's actually ready before accepting traffic, and puts limits on expensive
requests. Right now models can load lazily and repeatedly, and `/health` can lie.

## Step 1 — Load everything once, share it, validate at startup

**The problem in plain words.** If each request lazily loads a model, you waste memory and
time, and the first user hits a slow cold start. Load once when the app boots; hand the
same objects to every request.

**Key concepts:**

- **App factory** — a function that *builds* your app. Tests can call it with fakes
  instead of real models. ("Don't hardcode the app; make a function that assembles it.")
- **Lifespan** — FastAPI's startup/shutdown hook. Load models in startup, mark "ready."
- **Service container** — one bundle (`SearchServices`) holding everything, passed around
  instead of scattered global variables.
- **Semaphore** — a limited set of "tickets." A `BoundedSemaphore(1)` means only one
  rerank runs at a time; others get told "busy" instead of piling up.

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
    store = SQLiteDocstore(DOCSTORE_PATH, read_only=True)   # serving is read-only
    bm25 = BM25Search(docstore=store, create_if_missing=False)
    vector = VectorSearch(docstore=store)
    vector.load()
    actual = {"bm25": bm25.committed_document_count(),
              "vector": vector.index.ntotal, "docstore": store.count_documents()}
    if actual != manifest["counts"]:            # serving must match the receipt
        raise RuntimeError(f"Serving count mismatch: {actual}")

    spell = SpellCorrector()
    spell.load_default_dictionary()
    reranker = CrossEncoderReranker() if ENABLE_RERANKER else None
    if reranker is not None:
        _ = reranker.model    # force-load now, so "enabled" means "ready before traffic"

    return SearchServices(spell, bm25,
        HybridSearchEngine(bm25_search=bm25, vector_search=vector),
        reranker, BoundedSemaphore(1))


def create_app(loader: Callable[[], SearchServices] = load_services) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.ready = False
        try:
            app.state.services = loader()     # load once at startup
            app.state.ready = True
            yield
        finally:
            app.state.ready = False

    application = FastAPI(lifespan=lifespan)
    application.include_router(search_router)
    return application


app = create_app()
```

`verify_complete_manifest_and_hashes()` is just a small wrapper around the Phase 1 helpers:
require `status == complete`, check the required keys exist, and confirm every served
file's fingerprint matches the receipt.

Routes receive services via FastAPI's `Request`/`Depends` — delete the lazy globals. Tests
call `create_app(lambda: fake_services)`.

## Step 2 — Put limits and validation in shared code

**The problem in plain words.** If validation lives only in the route, anything calling
the function directly skips it. Put the checks *inside* the shared function so nobody can
bypass them.

```python
# Inside reciprocal_rank_fusion() itself:
if k <= 0 or top_k < 0:
    raise ValueError("Invalid RRF depth")
if not all(math.isfinite(w) and 0 <= w <= 100 for w in (bm25_weight, vector_weight)):
    raise ValueError("Weights must be finite and between 0 and 100")
if bm25_weight == vector_weight == 0:
    raise ValueError("At least one weight must be positive")
if top_k == 0:
    return []
```

Other boundaries: cap the query length (`max_length=256`); require `candidates_k >= top_k`
(can't return more than you fetched); keep upper bounds on result counts. Wrap reranking in
the semaphore and return **429 (Too Many Requests)** when busy, **503 (Service
Unavailable)** when reranking is disabled.

**Response cleanups:** rename the misleading `total` field to `returned_count`. Make
`body: str | None = None` in all result models, and let every route accept
`include_body=False` so it returns a short *snippet* by default (full text only on
request). Smaller responses = faster, cheaper.

## Step 3 — Real health checks, logging, metrics, gateway limits

**The problem in plain words.** `/health` returning "OK" while search isn't loaded is a
lie that breaks load balancers. Split it in two:

- **Liveness** — "is the process running at all?" (restart me if not)
- **Readiness** — "am I actually ready to serve search?" (don't send traffic yet if not)

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

Add `prometheus-client` (Prometheus = a standard metrics system). Expose `/metrics`; count
requests and measure latency, labeled by method/route/status — **never** by raw query text
(that leaks user data and explodes the metric count).

At the **gateway** (the entry point in front of your app), set concrete limits: request
timeout, connection timeout, and per-IP rate limit. For a public read-only demo, auth is
optional but *rate limiting is not* — otherwise one user can hammer you. Load CORS allowed
origins from an environment variable, not hardcoded localhost.

Finally, write a real `locustfile.py` (Locust = a load-testing tool that simulates many
users) and save one run's numbers: users, duration, throughput, failures, p50/p95/p99,
CPU, memory.

---

# Phase 5 — A build anyone can reproduce

**The big idea.** "Works on my machine" isn't good enough. Pin exact versions of
everything so the build is identical everywhere, and ship a safe container.

## Step 1 — Delete unused stuff, then lock versions

**The problem in plain words.** Your `config.py` declares PostgreSQL, Redis, and an LTR
(Learning-To-Rank) model path that nothing actually uses. Dead code and fake claims hurt
credibility — an interviewer *will* ask "where's the Postgres?" and "nowhere" is a bad
answer.

**What you'll do.** Remove the unused PostgreSQL/Redis/SQLAlchemy drivers, LightGBM/LTR,
their Compose services, and the README claims about them. Audit nltk, pandas, Jupyter,
matplotlib, scikit-learn, torch, pytest-asyncio — keep only what's used. Keep one corpus
downloader; delete duplicates. Move test-only packages into dev dependencies.

Then create a lock file (exact versions of every package, direct and indirect):

```powershell
python -m pip install "uv==<REVIEWED_FIXED_VERSION>"
uv lock
uv sync --frozen --extra dev
```

Commit `uv.lock`. CI installs with `uv sync --frozen --extra dev`; production uses
`uv sync --frozen --no-dev`. Build commands must reject blank model revisions (so you
always know exactly which model built the index). Verify downloaded corpus archives
against committed fingerprints.

## Step 2 — A safe container

**`.dockerignore`** (keep huge/secret stuff *out* of the image):

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

**Dockerfile** — pin exact image digests, run as a non-root user, and check readiness:

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

Mount `data/` and the exact model files read-only, point the model env vars at those local
paths, and verify their checksums. Don't download models at runtime. One worker keeps
memory sane (each worker would load its own multi-GB index); to scale, run more *replicas*
behind a load balancer, not more workers.

---

# Phase 6 — One honest speed-vs-quality experiment

**The big idea.** Your current FAISS index (`IndexFlatIP`) checks the query against
*every* vector — exact, but gets slower as the corpus grows. An **ANN** (Approximate
Nearest Neighbor) index like **HNSW** is much faster but may miss a few results. The
grown-up move is to *measure* that trade, not guess.

**HNSW** builds a layered graph of vectors so search can "hop" toward the answer instead of
scanning everything. `efSearch` controls effort: higher = more accurate but slower.

```python
base = faiss.IndexHNSWFlat(dimension, 32, faiss.METRIC_INNER_PRODUCT)
base.hnsw.efConstruction = 200
index = faiss.IndexIDMap2(base)
index.add_with_ids(embeddings, ids)

# IDMap2 wraps HNSW, so reach inside it to set search effort:
inner = faiss.downcast_index(index.index)
inner.hnsw.efSearch = 64
```

On the **same cohort and generation**, report NDCG@10, Recall@100, p50, p95, RAM, and disk
for: exact FlatIP, HNSW at efSearch 32/64/128, and whatever compressed index (SQ8/IVF-PQ)
you actually deploy. **Rule:** don't claim a speed/storage win unless your API loads that
exact index.

---

# Phase 7 — Publish honest evidence

Until every box below is checked, call it a **"portfolio-grade hybrid search prototype,"**
not "production-ready." Honesty here is a *strength*, not a weakness.

Your README should include: a tiny synthetic demo, the locked setup steps, the
architecture and how it behaves on failure, the fixed-cohort benchmark tables, the
generation ID / counts / index type, CI and load-test evidence, hardware and cost,
limitations, the ANN trade-off, and a license.

## Final checklist (plain-English)

- [ ] Tests never touch real `data/`, the network, or your model cache.
- [ ] Pytest only runs `tests/`; script-style tests have a `__main__` guard.
- [ ] Docstore keeps real title/category and is read-only when serving.
- [ ] One FAISS ID system; IDs and counts are tested.
- [ ] Zero/negative/already-done targets and crash points are all tested.
- [ ] Corrupt index/checkpoint/receipt fails loudly, nothing gets deleted.
- [ ] The finished receipt has a generation ID, matching identities, counts, and fingerprints.
- [ ] BM25 + vector + docstore + deployed compressed index all match that receipt.
- [ ] CI lint + tests pass from a fresh clone.
- [ ] Scoring math has one tested copy and fixed question sets.
- [ ] Benchmarks record the command, fingerprints, revisions, versions, hardware, repeats.
- [ ] API loads one shared, validated set of services and reports real readiness.
- [ ] Query, weights, candidates, body size, concurrency, timeout, and rate are all bounded.
- [ ] Logs, Prometheus metrics, and a saved load-test procedure exist.
- [ ] Lock file, model/data checksums, `.dockerignore`, and non-root image exist.
- [ ] Unused database/cache/LTR code and claims are gone.
- [ ] Every README number comes from the corrected benchmark.

When these are green, your interview story writes itself: *you found data-integrity risks,
made the artifacts crash-safe, rebuilt honest evidence, hardened the serving layer, and
measured a real quality-vs-speed trade-off.* That's a senior engineer's story.
