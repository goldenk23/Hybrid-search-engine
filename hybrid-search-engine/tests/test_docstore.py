"""
Hermetic tests for SQLiteDocstore.

All tests use tmp_path — never data/docstore.sqlite.
"""
import pytest

from src.database.docstore import SQLiteDocstore

DOCS = [
    {"id": "1", "title": "Python Intro", "category": "tutorial",
     "body": "Python is a high-level programming language."},
    {"id": "2", "title": "FAISS Guide", "category": "search",
     "body": "FAISS enables fast approximate nearest-neighbour search."},
]


@pytest.fixture
def store(tmp_path):
    s = SQLiteDocstore(tmp_path / "test.sqlite")
    s.init()
    return s


# ------------------------------------------------------------------ round-trip

def test_upsert_and_retrieve(store):
    store.upsert_documents(DOCS)
    result = store.get_documents_by_ids(["1"])

    assert "1" in result
    assert result["1"]["title"] == "Python Intro"
    assert result["1"]["category"] == "tutorial"
    assert "high-level" in result["1"]["body"]


def test_all_fields_stored(store):
    store.upsert_documents(DOCS)
    result = store.get_documents_by_ids(["1", "2"])

    assert result["2"]["title"] == "FAISS Guide"
    assert result["2"]["category"] == "search"
    assert "nearest-neighbour" in result["2"]["body"]


def test_upsert_overwrites_existing(store):
    store.upsert_documents(DOCS)
    updated = [{"id": "1", "title": "Updated Title", "category": "updated",
                "body": "Updated body content for document one."}]
    store.upsert_documents(updated)

    result = store.get_documents_by_ids(["1"])
    assert result["1"]["title"] == "Updated Title"
    assert result["1"]["category"] == "updated"


def test_missing_id_not_in_result(store):
    store.upsert_documents(DOCS)
    result = store.get_documents_by_ids(["999"])
    assert result == {}


def test_empty_id_list_returns_empty(store):
    assert store.get_documents_by_ids([]) == {}


def test_upsert_empty_list_is_noop(store):
    store.upsert_documents([])
    assert store.count_documents() == 0


# ------------------------------------------------------------------ count

def test_count_documents(store):
    assert store.count_documents() == 0
    store.upsert_documents(DOCS)
    assert store.count_documents() == 2


def test_count_after_upsert_no_duplicates(store):
    store.upsert_documents(DOCS)
    store.upsert_documents(DOCS)   # same IDs — should overwrite, not grow
    assert store.count_documents() == 2


# ------------------------------------------------------------------ read-only mode

def test_read_only_can_read(tmp_path):
    # Write with a normal store first.
    rw = SQLiteDocstore(tmp_path / "ro.sqlite")
    rw.init()
    rw.upsert_documents(DOCS)

    # Open the same file read-only and verify data is accessible.
    ro = SQLiteDocstore(tmp_path / "ro.sqlite", read_only=True)
    result = ro.get_documents_by_ids(["1"])
    assert result["1"]["title"] == "Python Intro"


def test_read_only_rejects_writes(tmp_path):
    rw = SQLiteDocstore(tmp_path / "ro.sqlite")
    rw.init()

    ro = SQLiteDocstore(tmp_path / "ro.sqlite", read_only=True)
    # SQLite enforces mode=ro at the engine level — any write raises an exception.
    with pytest.raises((RuntimeError, Exception)):
        ro.upsert_documents(DOCS)


def test_read_only_init_is_noop(tmp_path):
    rw = SQLiteDocstore(tmp_path / "ro.sqlite")
    rw.init()
    rw.upsert_documents(DOCS)

    ro = SQLiteDocstore(tmp_path / "ro.sqlite", read_only=True)
    ro.init()   # should return immediately without error
    assert ro.count_documents() == 2


def test_read_only_does_not_create_dir(tmp_path):
    nonexistent = tmp_path / "newdir" / "store.sqlite"
    # In read-only mode, the parent dir must already exist (or SQLite will error).
    # The key assertion: no directory was created just by constructing the object.
    SQLiteDocstore(nonexistent, read_only=True)
    assert not nonexistent.parent.exists()


# ------------------------------------------------------------------ backward-compat alias

def test_get_document_by_id_alias(store):
    store.upsert_documents(DOCS)
    result = store.get_document_by_id(["1"])
    assert "1" in result
