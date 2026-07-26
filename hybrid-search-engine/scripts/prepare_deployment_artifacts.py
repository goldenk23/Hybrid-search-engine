"""Stage a verified SQ8 artifact bundle without modifying source indexes."""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from src.indexing.artifact_state import load_json_required, sha256_path, write_json_atomic

REVISION_PATTERN = re.compile(r"^[0-9a-f]{40}$")
MIN_SAMPLE_COSINE = 0.95


def _base(index: faiss.Index) -> faiss.Index:
    return faiss.downcast_index(index.index) if hasattr(index, "index") else index


def _ids(index: faiss.Index) -> np.ndarray:
    if not hasattr(index, "id_map"):
        raise ValueError("Vector index must use IndexIDMap2 with durable document IDs")
    return faiss.vector_to_array(index.id_map)


def _validate_indexes(flat_path: Path, sq8_path: Path) -> tuple[faiss.Index, faiss.Index]:
    flat = faiss.read_index(str(flat_path))
    sq8 = faiss.read_index(str(sq8_path))
    sq8_base = _base(sq8)
    if type(sq8_base).__name__ != "IndexScalarQuantizer":
        raise ValueError(f"Expected an SQ8 index, got {type(sq8_base).__name__}")
    if sq8_base.sq.qtype != faiss.ScalarQuantizer.QT_8bit:
        raise ValueError("Quantized index is not FAISS QT_8bit")
    if (flat.ntotal, flat.d, flat.metric_type) != (sq8.ntotal, sq8.d, sq8.metric_type):
        raise ValueError("Flat and SQ8 index count, dimension, or metric differs")
    if not np.array_equal(_ids(flat), _ids(sq8)):
        raise ValueError("Flat and SQ8 document IDs differ")

    # ponytail: sample reconstruction catches a wrong/corrupt SQ8 cheaply; use a
    # full quantization-evaluation pipeline if deployment quality requirements grow.
    positions = np.linspace(0, flat.ntotal - 1, min(64, flat.ntotal), dtype=np.int64)
    flat_base, samples = _base(flat), []
    for position in positions:
        left = flat_base.reconstruct(int(position))
        right = sq8_base.reconstruct(int(position))
        denominator = np.linalg.norm(left) * np.linalg.norm(right)
        samples.append(float(np.dot(left, right) / denominator) if denominator else 1.0)
    if not np.isfinite(samples).all() or min(samples) < MIN_SAMPLE_COSINE:
        raise ValueError(f"SQ8 reconstruction check failed (minimum cosine={min(samples):.4f})")
    return flat, sq8


def prepare_deployment_artifacts(source_data: Path, output: Path, revision: str) -> Path:
    """Copy verified BM25/docstore/SQ8 artifacts into ``output/data`` atomically."""
    source_data = source_data.resolve()
    output = output.resolve()
    if not REVISION_PATTERN.fullmatch(revision):
        raise ValueError("embedding revision must be a lowercase 40-character Git commit")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    indexes = source_data / "indexes"
    manifest_path = indexes / "artifact_manifest.json"
    paths = {
        "bm25": indexes / "bm25_compact",
        "vector": indexes / "vector.faiss",
        "docstore": source_data / "docstore.sqlite",
    }
    sq8_path = indexes / "vector.sq8.faiss"
    manifest = load_json_required(manifest_path)
    if manifest.get("status") != "complete":
        raise ValueError("Source artifact manifest is not complete")
    counts = manifest.get("counts", {})
    if set(counts) < paths.keys() or len(set(counts.values())) != 1 or next(iter(counts.values()), 0) <= 0:
        raise ValueError(f"Source artifact counts are incomplete or inconsistent: {counts}")

    stored_hashes = manifest.get("artifact_sha256", {})
    source_hashes: dict[str, str] = {}
    for name, path in paths.items():
        if not path.exists() or name not in stored_hashes:
            raise FileNotFoundError(f"Required source artifact or hash is missing: {name}")
        source_hashes[name] = sha256_path(path)
        if source_hashes[name] != stored_hashes[name]:
            raise ValueError(f"Source {name} hash does not match its manifest")
    if not sq8_path.exists():
        raise FileNotFoundError(f"SQ8 index not found: {sq8_path}")

    flat, sq8 = _validate_indexes(paths["vector"], sq8_path)
    if flat.ntotal != counts["vector"] or sq8.ntotal != counts["vector"]:
        raise ValueError("Manifest and FAISS document counts differ")
    recorded_revision = manifest.get("embedding_revision", "")
    if recorded_revision and recorded_revision != revision:
        raise ValueError("Requested embedding revision differs from source manifest")

    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary}")
    staged_data = temporary / "data"
    staged_indexes = staged_data / "indexes"
    try:
        staged_indexes.mkdir(parents=True)
        shutil.copy2(paths["docstore"], staged_data / "docstore.sqlite")
        shutil.copytree(paths["bm25"], staged_indexes / "bm25_compact", ignore=shutil.ignore_patterns("*.lock"))
        shutil.copy2(sq8_path, staged_indexes / "vector.faiss")

        staged_paths = {
            "bm25": staged_indexes / "bm25_compact",
            "vector": staged_indexes / "vector.faiss",
            "docstore": staged_data / "docstore.sqlite",
        }
        manifest["embedding_revision"] = revision
        manifest["artifact_sha256"] = {
            name: sha256_path(path) for name, path in staged_paths.items()
        }
        manifest["deployment"] = {
            "index_type": "sq8",
            "prepared_at_utc": datetime.now(UTC).isoformat(),
            "source_generation_id": manifest.get("generation_id"),
            "source_flat_sha256": source_hashes["vector"],
            "source_sq8_sha256": sha256_path(sq8_path),
            "revision_source": "manifest" if recorded_revision else "operator_recovered",
        }
        write_json_atomic(staged_indexes / "artifact_manifest.json", manifest)

        for name, path in paths.items():
            if sha256_path(path) != source_hashes[name]:
                raise RuntimeError(f"Source {name} changed while staging")
        temporary.replace(output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output / "data"


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare an immutable SQ8 deployment bundle")
    parser.add_argument("--source-data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--embedding-revision", required=True)
    args = parser.parse_args()
    data_path = prepare_deployment_artifacts(
        args.source_data, args.output, args.embedding_revision
    )
    size = sum(path.stat().st_size for path in data_path.rglob("*") if path.is_file())
    print(f"Deployment artifacts ready: {data_path}")
    print(f"Size: {size / 1024 / 1024:.1f} MiB")
    print("Source artifacts were not modified.")


if __name__ == "__main__":
    main()
