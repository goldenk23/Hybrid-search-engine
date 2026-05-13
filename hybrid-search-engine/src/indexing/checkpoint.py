"""
Checkpoint management for resumable indexing.

Saves indexing progress to a JSON file so that if indexing crashes,
it can resume from the last checkpoint instead of starting over.
"""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional


class IndexCheckpoint:
    """Manages checkpoint files for resumable indexing."""

    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "indexing_checkpoint.json"

    def save_checkpoint(
        self,
        total_documents_indexed: int,
        last_document_id: str,
        collection_path: str,
        max_documents: Optional[int] = None,
        batch_size: int = 1000,
    ) -> None:
        """Save indexing progress to checkpoint file.
        
        Args:
            total_documents_indexed: Total number of documents indexed so far
            last_document_id: ID of the last indexed document
            collection_path: Path to the collection file being indexed
            max_documents: Maximum documents to index (None = all)
            batch_size: Documents per batch
        """
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "total_documents_indexed": total_documents_indexed,
            "last_document_id": last_document_id,
            "collection_path": str(collection_path),
            "max_documents": max_documents,
            "batch_size": batch_size,
            "checkpoint_version": 1,
        }

        # Write atomically to avoid corruption
        temp_file = self.checkpoint_file.with_suffix(".tmp")
        with open(temp_file, "w") as f:
            json.dump(checkpoint_data, f, indent=2)
        
        # Atomic rename
        temp_file.replace(self.checkpoint_file)
        print(
            f"  Checkpoint saved: {total_documents_indexed:,} documents "
            f"(at {datetime.now().strftime('%H:%M:%S')})"
        )

    def load_checkpoint(self) -> Optional[dict]:
        """Load the last checkpoint if it exists.
        
        Returns:
            Dictionary with checkpoint data, or None if no checkpoint exists
        """
        if not self.checkpoint_file.exists():
            return None

        try:
            with open(self.checkpoint_file, "r") as f:
                checkpoint = json.load(f)
            print(
                f"Loaded checkpoint: {checkpoint['total_documents_indexed']:,} "
                f"documents previously indexed"
            )
            print(f"  Last indexed at: {checkpoint['timestamp']}")
            return checkpoint
        except (json.JSONDecodeError, IOError, KeyError) as e:
            print(f"Warning: Could not load checkpoint: {e}")
            return None

    def clear_checkpoint(self) -> None:
        """Clear the checkpoint file after successful indexing."""
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()
            print("Checkpoint cleared (indexing completed successfully)")

    def get_checkpoint_status(self) -> Optional[dict]:
        """Get current checkpoint status without loading it."""
        if not self.checkpoint_file.exists():
            return None
        
        try:
            with open(self.checkpoint_file, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return None
