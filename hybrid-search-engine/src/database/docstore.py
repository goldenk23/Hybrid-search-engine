import sqlite3
import zlib
from pathlib import Path
from typing import Any

from src.config import DOCSTORE_PATH

# Just making small comment

class SQLiteDocstore:
    """SQLite-backed document store with optional read-only mode."""

    def __init__(self, db_path: Path | None = None, *, read_only: bool = False):
        # Fall back to the configured production path when no path is given.
        self.db_path = db_path or DOCSTORE_PATH

        # Remember the flag so every method can check it.
        self.read_only = read_only

        # Only create the directory in write mode.
        # A read-only store should not silently create an empty folder —
        # if the DB file isn't there, that's a bug you want to see.
        if not read_only:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        if self.read_only:
            # SQLite URI syntax: "mode=ro" makes the engine itself reject writes.
            # resolve() → absolute path (no ".." segments).
            # as_posix() → forward slashes, which the SQLite URI parser requires
            # even on Windows.
            uri = f"file:{self.db_path.resolve().as_posix()}?mode=ro"
            # uri=True tells sqlite3 to treat the string as a URI, not a filename.
            return sqlite3.connect(uri, uri=True)

        return sqlite3.connect(self.db_path)

    def init(self) -> None:
        # Nothing to initialise in read-only mode — the table must already exist.
        if self.read_only:
            return
        with self.connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS documents (
                    id               TEXT PRIMARY KEY,
                    title            TEXT NOT NULL,
                    category         TEXT NOT NULL,
                    body_compressed  BLOB NOT NULL
                )
            """)

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _compress_text(text: str) -> bytes:
        return zlib.compress(text.encode("utf-8"), level=6)

    @staticmethod
    def _decompress_text(compressed: bytes) -> str:
        return zlib.decompress(compressed).decode("utf-8")

    # ------------------------------------------------------------------ writes

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        """Insert or update documents. Each dict must have 'id'; 'title',
        'category', and 'body' are optional and default to empty string."""
        if not documents:
            return

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO documents (id, title, category, body_compressed)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title           = excluded.title,
                    category        = excluded.category,
                    body_compressed = excluded.body_compressed
                """,
                [
                    (
                        str(doc["id"]),
                        doc.get("title", ""),
                        doc.get("category", ""),
                        self._compress_text(doc.get("body", "")),
                    )
                    for doc in documents
                ],
            )

    # ------------------------------------------------------------------ reads

    def get_documents_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}

        # One "?" per id — never interpolate user data directly into SQL.
        placeholders = ",".join("?" for _ in ids)

        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT id, title, category, body_compressed
                FROM documents
                WHERE id IN ({placeholders})
                """,
                ids,
            ).fetchall()

        documents: dict[str, dict[str, Any]] = {}
        for row in rows:
            documents[row["id"]] = {
                "id":       row["id"],
                "title":    row["title"],     # real stored value, not body[:100]
                "category": row["category"],  # real stored value, not hardcoded "msmarco"
                "body":     self._decompress_text(row["body_compressed"]),
            }

        return documents

    # ------------------------------------------------------------------ utils

    def count_documents(self) -> int:
        with self.connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0])

    def get_document_by_id(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Backward-compatible alias for older call sites."""
        return self.get_documents_by_ids(ids)
