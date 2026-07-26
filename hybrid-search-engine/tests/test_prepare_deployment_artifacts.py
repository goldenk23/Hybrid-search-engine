import json
from pathlib import Path

import faiss
import numpy as np
import pytest

from scripts.prepare_deployment_artifacts import prepare_deployment_artifacts
from src.indexing.artifact_state import sha256_path, write_json_atomic

REVISION = "a" * 40


def _source_data(root: Path) -> Path:
    data = root / "data"
    indexes = data / "indexes"
    bm25 = indexes / "bm25_compact"
    bm25.mkdir(parents=True)
    (bm25 / "segment.idx").write_bytes(b"bm25")
    (data / "docstore.sqlite").write_bytes(b"sqlite")

    vectors = np.asarray(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        dtype="float32",
    )
    ids = np.asarray([10, 20, 30, 40], dtype="int64")
    flat = faiss.IndexIDMap2(faiss.IndexFlatIP(4))
    flat.add_with_ids(vectors, ids)
    faiss.write_index(flat, str(indexes / "vector.faiss"))

    scalar = faiss.IndexScalarQuantizer(4, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT)
    scalar.train(vectors)
    sq8 = faiss.IndexIDMap2(scalar)
    sq8.add_with_ids(vectors, ids)
    faiss.write_index(sq8, str(indexes / "vector.sq8.faiss"))

    paths = {"bm25": bm25, "vector": indexes / "vector.faiss", "docstore": data / "docstore.sqlite"}
    write_json_atomic(indexes / "artifact_manifest.json", {
        "status": "complete", "generation_id": "test", "embedding_model": "all-MiniLM-L6-v2",
        "embedding_revision": "", "counts": {name: 4 for name in paths},
        "artifact_sha256": {name: sha256_path(path) for name, path in paths.items()},
    })
    return data


def test_prepares_sq8_bundle_without_mutating_source(tmp_path):
    source = _source_data(tmp_path)
    source_manifest = source / "indexes" / "artifact_manifest.json"
    before = source_manifest.read_bytes()
    output = tmp_path / "deployment"

    staged = prepare_deployment_artifacts(source, output, REVISION)

    assert source_manifest.read_bytes() == before
    manifest = json.loads((staged / "indexes" / "artifact_manifest.json").read_text())
    assert manifest["embedding_revision"] == REVISION
    assert manifest["deployment"]["index_type"] == "sq8"
    vector = staged / "indexes" / "vector.faiss"
    assert type(faiss.downcast_index(faiss.read_index(str(vector)).index)).__name__ == "IndexScalarQuantizer"
    assert manifest["artifact_sha256"]["vector"] == sha256_path(vector)
    with pytest.raises(FileExistsError):
        prepare_deployment_artifacts(source, output, REVISION)


def test_directory_fingerprint_uses_posix_separators(tmp_path):
    """Guards the cross-OS fingerprint: a bundle prepared on Windows has to verify
    on a Linux server, so the path bytes folded into the digest must not depend on
    os.sep. Using str(relative_path) here silently broke deployment."""
    import hashlib

    tree = tmp_path / "index"
    (tree / "nested").mkdir(parents=True)
    (tree / "nested" / "part.bin").write_bytes(b"payload")

    expected = hashlib.sha256()
    expected.update(b"index/nested/part.bin")
    expected.update(b"payload")

    assert sha256_path(tree) == expected.hexdigest()
