"""
BM25 keyword search using Tantivy.
"""

import re
import time
import gc
from pathlib import Path
from typing import Any
from collections.abc import Iterable
from src.database.docstore import SQLiteDocstore
from src.config import BM25_INDEX_PATH, INDEX_DIR

import tantivy


WRITER_HEAP_SIZE_BYTES = 64_000_000
WRITER_NUM_THREADS = 1


class BM25Search:
    """BM25 search engine using Tantivy."""

    def __init__(self, index_path: Path | None = None, reset: bool = False, is_resuming: bool = False):
        self.index_path = index_path or BM25_INDEX_PATH
        self.docstore = SQLiteDocstore()
        self.docstore.init()
        self.schema = (
            tantivy.SchemaBuilder()
            .add_text_field("id", stored=True)
            .add_text_field("title", stored=False)
            .add_text_field("body", stored=False)
            .add_text_field("category", stored=False)
            .build()
        )

        # Delete existing index if reset is True (for reindexing)
        if reset and self.index_path.exists():
            import shutil
            shutil.rmtree(self.index_path)
            print(f"Reset: Deleted existing index at {self.index_path}")

        # Try to open existing index
        if self.index_path.exists():
            try:
                self.index = tantivy.Index.open(str(self.index_path))
            except (ValueError, Exception) as e:
                # Index directory exists but is corrupted/incomplete
                if is_resuming:
                    # If resuming from checkpoint, this is a critical error - don't auto-fix
                    print(f"ERROR opening existing index: {e}")
                    print(f"   Index path: {self.index_path}")
                    print(f"   To fix: use --reset flag to delete and recreate index")
                    raise RuntimeError(
                        f"Failed to open index at {self.index_path}. "
                        "Use --reset flag to delete and recreate the index."
                    ) from e
                else:
                    # Starting fresh with corrupted index dir - auto-recover by deleting and recreating
                    import shutil
                    print("WARNING: Index directory corrupted but not resuming - recreating fresh index...")
                    shutil.rmtree(self.index_path)
                    self.index_path.mkdir(parents=True, exist_ok=True)
                    self.index = tantivy.Index(self.schema, path=str(self.index_path))
        else:
            self.index_path.mkdir(parents=True, exist_ok=True)
            self.index = tantivy.Index(self.schema, path=str(self.index_path))

    def _create_writer(self):
        """Create a conservative writer for Windows-safe indexing."""
        return self.index.writer(
            heap_size=WRITER_HEAP_SIZE_BYTES,
            num_threads=WRITER_NUM_THREADS,
        )

    def _index_size_bytes(self) -> int:
        """Return the current on-disk size of the index directory."""
        if not self.index_path.exists():
            return 0

        return sum(
            path.stat().st_size
            for path in self.index_path.rglob("*")
            if path.is_file()
        )

    def committed_document_count(self) -> int:
        """Return the number of documents visible to a fresh Tantivy searcher."""
        return self.index.searcher().num_docs

    def _refresh_and_validate_commit(
        self,
        *,
        expected_min_docs: int,
        previous_committed_docs: int,
        label: str,
    ) -> int:
        """Reload searchers and verify committed document count never regresses."""
        self.index.reload()
        committed_docs = self.committed_document_count()
        index_size_mb = self._index_size_bytes() / 1024 / 1024

        print(
            f"     {label}: committed_docs={committed_docs:,}, "
            f"index_size={index_size_mb:.1f} MB"
        )

        if committed_docs < previous_committed_docs:
            raise RuntimeError(
                "Committed BM25 document count went backwards "
                f"({previous_committed_docs:,} -> {committed_docs:,}). "
                "Stop indexing and inspect the index directory before continuing."
            )

        if committed_docs < expected_min_docs:
            raise RuntimeError(
                "BM25 commit did not make all checkpoint documents durable "
                f"(expected at least {expected_min_docs:,}, got {committed_docs:,}). "
                "Checkpoint was not advanced."
            )

        return committed_docs

   # documents: list of dicts with keys: id, title, body, category, all documents are loaded into RAM before indexing.
    def add_documents(self, documents: list[dict[str, Any]], batch_size: int = 5000) -> None:
        """Add documents to the BM25 index.

        For large datasets, use add_documents_stream_with_checkpoint instead.
        Commits only at the end to avoid Windows file locking issues.
        """
        committed_docs = self.committed_document_count()
        writer = self._create_writer()
        
        try:
            for i, document in enumerate(documents):
                writer.add_document(
                    tantivy.Document(
                        id=str(document["id"]),
                        title=document.get("title", ""),
                        body=document.get("body", ""),
                        category=document.get("category", ""),
                    )
                )
                self.docstore.upsert_documents([document])

                # Print progress
                if (i + 1) % batch_size == 0:
                    print(f"  Indexed {i + 1:,} documents...")

            # Commit ALL documents at once at the end
            print(f"  Committing {len(documents):,} documents to index...")
            writer.commit()
            gc.collect()
            time.sleep(1.0)
            self._refresh_and_validate_commit(
                expected_min_docs=committed_docs + len(documents),
                previous_committed_docs=committed_docs,
                label="Commit verified",
            )
        finally:
            del writer
            gc.collect()

    def add_documents_stream_with_checkpoint(
        self,
        documents: Iterable[dict[str, Any]],
        checkpoint_manager,
        collection_path: Path | str | None = None,
        batch_size: int = 5000,
        checkpoint_interval: int = 25000,
        start_count: int = 0,
    ) -> int:
        """Add documents from an iterable with checkpoint and resume support.
        
        Checkpoint metadata and Tantivy commits are kept in lockstep:
        - Add documents with a writer until the checkpoint interval is reached
        - Commit that writer so the index is durable on disk
        - Save checkpoint metadata only after the commit succeeds
        - Open a new writer for the next interval
        - On resume: skip to the last committed document ID and continue adding
        
        Args:
            documents: Iterable of document dictionaries
            checkpoint_manager: IndexCheckpoint instance for saving progress
            collection_path: Path to collection file (for checkpoint metadata)
            batch_size: Progress logging interval
            checkpoint_interval: Save checkpoint every N documents
            start_count: Starting count for resume scenarios
        """
        count = start_count
        buffer_count = 0
        last_document_id = None
        committed_docs = self.committed_document_count()
        
        writer = self._create_writer()
        
        # Conservative sub-batch size for maximum stability
        sub_batch_size = 150
        docstore_batch = []
        docstore_batch_size = 1000
        
        try:
            for document in documents:
                # Retry logic for transient failures
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        writer.add_document(
                            tantivy.Document(
                                id=str(document["id"]),
                                title=document.get("title", ""),
                                body=document.get("body", ""),
                                category=document.get("category", ""),
                            )
                        )
                        last_document_id = str(document["id"])
                        break  # Success
                    except ValueError as e:
                        if attempt < max_retries - 1:
                            gc.collect()
                            time.sleep(0.5)
                        else:
                            raise

                count += 1
                buffer_count += 1
                docstore_batch.append(document)

                if len(docstore_batch) >= docstore_batch_size:
                    self.docstore.upsert_documents(docstore_batch)
                    docstore_batch.clear()
                
                # Give Tantivy's worker threads time to process
                if buffer_count % sub_batch_size == 0:
                    gc.collect()
                    time.sleep(0.05)
                
                # Print progress at batch interval
                if count % batch_size == 0:
                    print(f"  Indexed {count:,} documents...")
                
                # Commit before saving checkpoint metadata. After a crash, the
                # checkpoint will never point past durable index data.
                if count > start_count and (count - start_count) % checkpoint_interval == 0:
                    print(f"\n  Checkpoint at {count:,} documents...")
                    
                    if docstore_batch:
                        self.docstore.upsert_documents(docstore_batch)
                        docstore_batch.clear()

                    print("     Committing checkpoint batch to disk...")
                    writer.commit()
                    gc.collect()
                    time.sleep(1.0)
                    committed_docs = self._refresh_and_validate_commit(
                        expected_min_docs=count,
                        previous_committed_docs=committed_docs,
                        label="Checkpoint commit verified",
                    )

                    checkpoint_manager.save_checkpoint(
                        total_documents_indexed=count,
                        last_document_id=last_document_id,
                        collection_path=str(collection_path) if collection_path else "unknown",
                    )
                    print("     Checkpoint saved to metadata file")
                    print(f"     Index and checkpoint saved")
                    print()
                    del writer
                    gc.collect()
                    time.sleep(1.0)
                    writer = self._create_writer()

            # Final commit for the last partial checkpoint interval.
            print(f"\n  Final commit: Writing {count:,} documents to index...")
            if docstore_batch:
                self.docstore.upsert_documents(docstore_batch)
                docstore_batch.clear()

            writer.commit()
            gc.collect()
            time.sleep(1.0)
            print("     All documents committed successfully!")
            committed_docs = self._refresh_and_validate_commit(
                expected_min_docs=count,
                previous_committed_docs=committed_docs,
                label="Final commit verified",
            )
            if last_document_id:
                checkpoint_manager.save_checkpoint(
                    total_documents_indexed=count,
                    last_document_id=last_document_id,
                    collection_path=str(collection_path) if collection_path else "unknown",
                )
                print(f"     Final checkpoint saved at {count:,} documents")
            
        except (Exception, KeyboardInterrupt) as e:
            print(f"\nERROR during indexing: {e}")
            print(f"Saving checkpoint at {count:,} documents...")
            
            partial_commit_succeeded = False

            # Try to commit what we have so far (even though incomplete)
            try:
                print("   Attempting to commit partial batch...")
                if docstore_batch:
                    self.docstore.upsert_documents(docstore_batch)
                    docstore_batch.clear()

                writer.commit()
                committed_docs = self._refresh_and_validate_commit(
                    expected_min_docs=count,
                    previous_committed_docs=committed_docs,
                    label="Partial commit verified",
                )
                partial_commit_succeeded = True
                print(f"   Partial batch committed ({count:,} documents)")
                gc.collect()
                time.sleep(1.0)
            except Exception as commit_error:
                print(f"   Warning: Failed to commit: {commit_error}")
            
            # Save checkpoint only if the matching index commit succeeded.
            if last_document_id and partial_commit_succeeded:
                checkpoint_manager.save_checkpoint(
                    total_documents_indexed=count,
                    last_document_id=last_document_id,
                    collection_path=str(collection_path) if collection_path else "unknown",
                )
                print(f"   Checkpoint saved - can resume from document {last_document_id}")
            elif last_document_id:
                print("   Checkpoint not advanced because the partial commit failed")
            raise
        finally:
            # Ensure writer is properly cleaned up
            try:
                if 'writer' in locals():
                    del writer
            except:
                pass
            gc.collect()

        return count







    def search(self, query: str, top_k: int = 10) -> list[dict[str, Any]]:
        """Search the BM25 index and return ranked results."""
        searcher = self.index.searcher()

        # MS MARCO queries are natural-language questions, not Tantivy query syntax.
        # We remove punctuation so words like "don't" become "don t" instead of
        # causing parser syntax errors.
        cleaned_query = re.sub(r"[^\w\s]", " ", query)
        cleaned_query = re.sub(r"\s+", " ", cleaned_query).strip()

        if not cleaned_query:
            return []

        query_obj = self.index.parse_query(cleaned_query)
        hits = searcher.search(query_obj, top_k).hits

        doc_ids = []
        scores_by_id = {}

        for score, doc_address in hits:
            retrieved_doc = searcher.doc(doc_address)
            doc_id = str(retrieved_doc.get_first("id"))
            doc_ids.append(doc_id)
            scores_by_id[doc_id] = score

        docs_by_id = self.docstore.get_documents_by_ids(doc_ids)

        results = []
        for doc_id in doc_ids:
            document = docs_by_id.get(doc_id)
            if document is None:
                continue

            results.append(
                {
                    "id": doc_id,
                    "title": document.get("title", ""),
                    "body": document.get("body", ""),
                    "category": document.get("category", ""),
                    "score": scores_by_id[doc_id],
                }
            )

        return results
