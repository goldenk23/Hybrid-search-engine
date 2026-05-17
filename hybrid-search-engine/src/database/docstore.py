import sqlite3
import zlib # For compressing document bodies before storage
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
                    body_compressed BLOB NOT NULL
                )
                """
            )
    @staticmethod
    def _compress_text(text: str) -> bytes:
        return zlib.compress(text.encode("utf-8"), level=6)# level=6 => Moderate compression level for a good balance of speed and size
    
    @staticmethod
    def _decompress_text(compressed: bytes) -> str:
        return zlib.decompress(compressed).decode("utf-8")

    def upsert_documents(self, documents: list[dict[str, Any]]) -> None:
        """Insert or update documents in the local document store."""
        if not documents:
            return

        with self.connect() as conn:
            conn.executemany(
                """
                INSERT INTO documents (id, body_compressed)
                VALUES (?, ?)
                ON CONFLICT(id) DO UPDATE SET
                body_compressed=excluded.body_compressed
                """,
                [
                    (
                        str(doc["id"]),
                        self._compress_text(doc.get("body", "")),
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
                SELECT id, body_compressed
                FROM documents
                WHERE id IN ({placeholders})
                """,
                ids,
            ).fetchall()
            
        documents = {}
        for row in rows:
            body = self._decompress_text(row["body_compressed"])
            documents[row["id"]] = {
                "id": row["id"],
                "title": body[:100],
                "body": body,
                "category": "msmarco",
            }

        return documents
    def get_document_by_id(self, ids: list[str]) -> dict[str, dict[str, Any]]:
        """Backward-compatible alias for older call sites."""
        return self.get_documents_by_ids(ids)
