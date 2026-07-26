"""
Recompute artifact hashes from disk and update the manifest.

Run once after copying index files from another machine.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.artifact_state import sha256_path, load_json_required, write_json_atomic
from src.config import BM25_INDEX_PATH, VECTOR_INDEX_PATH, DOCSTORE_PATH, INDEX_DIR

manifest_path = INDEX_DIR / "artifact_manifest.json"
manifest = load_json_required(manifest_path)

print("Stored hashes (old):")
for k, v in manifest["artifact_sha256"].items():
    print(f"  {k}: {v}")

print("\nRecomputing from disk (this may take a minute for large indexes)...")

new_hashes = {}
for name, path in [("bm25", BM25_INDEX_PATH), ("vector", VECTOR_INDEX_PATH), ("docstore", DOCSTORE_PATH)]:
    if path.exists():
        h = sha256_path(path)
        new_hashes[name] = h
        print(f"  {name}: {h}")
    else:
        print(f"  {name}: MISSING — skipped")

manifest["artifact_sha256"] = new_hashes
write_json_atomic(manifest_path, manifest)
print("\nManifest updated. You can now start the API.")
