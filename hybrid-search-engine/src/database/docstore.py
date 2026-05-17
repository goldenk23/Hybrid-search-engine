import sqlite3
from pathlib import Path
from typing import Any

from src.config import DOCSTORE_PATH


class SQLiteDocstore:
    """Simple document store using SQLite."""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DOCSTORE_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def init(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    body TEXT NOT NULL,
                    category TEXT
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category)")

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        """Insert or update documents in the local document store."""
        if not documents:
            return

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO documents (id, title, body, category)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    title = excluded.title,
                    body = excluded.body,
                    category = excluded.category
                """,
                [
                    (
                        str(doc["id"]),
                        doc.get("title", ""),
                        doc.get("body", ""),
                        doc.get("category", ""),
                    )
                    for doc in documents
                ],
            )

    def get_documents_by_ids(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        if not ids:
            return {}

        placeholders = ",".join("?" for _ in ids)
        with self.connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                f"""
                SELECT id, title, body, category
                FROM documents
                WHERE id IN ({placeholders})
                """,
                ids,
            ).fetchall()

        return {
            row["id"]: {
                "id": row["id"],
                "title": row["title"],
                "body": row["body"],
                "category": row["category"],
            }
            for row in rows
        }

    def get_document_by_id(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Backward-compatible alias for older call sites."""
        return self.get_documents_by_ids(ids)
