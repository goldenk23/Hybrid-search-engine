"""
Helpers for reading and writing artifact state files (manifests, checkpoints).

Four functions, one job each:

  sha256_path       — fingerprint a file or directory
  write_json_atomic — write JSON without risk of a half-written file
  load_json_required — read JSON and stop loudly if it is missing or corrupt
  reconcile_count   — decide the safe start_count after a crash
"""

import hashlib
import json
import os
from pathlib import Path


def sha256_path(path: Path) -> str:
    """Return a hex SHA-256 digest that fingerprints a file or a directory tree.

    For a file: hash its contents.
    For a directory: hash each file's relative path + contents in sorted order
    so the fingerprint is stable regardless of filesystem traversal order.

    Same bytes (or same tree) → same fingerprint every time.  This lets us
    prove an artifact hasn't changed or been swapped out for a same-size
    impostor from a different build.
    """
    digest = hashlib.sha256()

    if path.is_file():
        files = [path]
    else:
        # Sort so the hash is deterministic across OSes and Python versions.
        # Skip transient lock files (Tantivy writes *.lock files like
        # .tantivy-writer.lock / .tantivy-meta.lock). Those are runtime state,
        # not index content, and Tantivy creates/removes them when the index is
        # opened. Hashing them makes the fingerprint change between indexing and
        # serving even when the actual index bytes are identical — which would
        # make the API refuse to start on a correctly-copied index.
        files = sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and not p.name.endswith(".lock")
        )

    for file in files:
        # Include the relative path so renaming a file changes the fingerprint.
        # as_posix() keeps the separator stable: str() would emit "dir\file" on
        # Windows and "dir/file" on Linux, so a bundle prepared on Windows could
        # never pass verification on a Linux server.
        digest.update(file.relative_to(path.parent).as_posix().encode())
        # Stream in 1 MB chunks to avoid loading multi-GB indexes into RAM.
        with file.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)

    return digest.hexdigest()


def write_json_atomic(path: Path, value: dict) -> None:
    """Write *value* to *path* as JSON without risk of a partial write.

    Strategy: write to a sibling temp file first, then rename.
    A rename (os.replace) is atomic on every major OS — it either
    completes fully or does nothing.  If the process dies mid-write,
    the original file is untouched and the incomplete temp file is
    harmless (it will be overwritten on the next attempt).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(temporary, path)  # atomic swap


def load_json_required(path: Path) -> dict:
    """Read and return the JSON object at *path*, or raise RuntimeError.

    Two failure modes we refuse to silently paper over:
    - File missing or unreadable → OSError
    - File exists but contains invalid JSON → json.JSONDecodeError
    Both are wrapped in RuntimeError so callers get a clear message instead
    of a confusing low-level exception.

    We also reject non-object JSON (arrays, bare strings, etc.) because
    every state file in this project is expected to be a JSON object.
    """
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unreadable state file: {path}") from exc

    if not isinstance(value, dict):
        raise TypeError(
            f"Expected a JSON object in {path}, got {type(value).__name__}"
        )

    return value


def reconcile_count(checkpoint_count: int, durable_count: int) -> int:
    """Decide the safe start_count after a crash or interrupted run.

    Two scenarios after a crash:

    1. Checkpoint is AHEAD of the durable index
       (crash happened after saving the checkpoint but before the index write
       finished, or the index write was partial/corrupt).
       → We cannot trust either; stop loudly so the operator can investigate.

    2. Durable index is AHEAD of (or equal to) the checkpoint
       (crash happened after the index was written but before the checkpoint
       was updated — the common case with write-then-checkpoint ordering).
       → Trust the durable index count; the checkpoint is just behind.

    Returns the safe count to resume from (always durable_count in case 2).
    """
    if checkpoint_count > durable_count:
        raise RuntimeError(
            f"Checkpoint ({checkpoint_count:,}) is ahead of the durable index "
            f"({durable_count:,}). The index may be corrupt or incomplete. "
            "Inspect both files before resuming, or use --reset to start over."
        )
    return durable_count
