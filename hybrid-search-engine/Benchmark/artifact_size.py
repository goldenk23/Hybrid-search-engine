from pathlib import Path


def size_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def mib(num_bytes: int) -> float:
    return round(num_bytes / 1024 / 1024, 2)


artifact_sizes = {
    "bm25_compact_mib": mib(size_bytes(Path("data/indexes/bm25_compact"))),
    "vector_sq8_mib": mib(size_bytes(Path("data/indexes/vector.sq8.faiss"))),
    "docstore_mib": mib(size_bytes(Path("data/docstore.sqlite"))),
}
artifact_sizes["total_mib"] = round(sum(artifact_sizes.values()), 2)
print(artifact_sizes)