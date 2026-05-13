"""
Download MS MARCO dataset from HuggingFace (official source)
"""

from datasets import load_dataset
from pathlib import Path

# Create output directory
output_dir = Path("data/msmarco")
output_dir.mkdir(parents=True, exist_ok=True)

print("Downloading MS MARCO queries from HuggingFace...")
queries = load_dataset("microsoft/ms_marco", "v1.1", split="dev")

# Save queries.tsv (query_id \t query_text)
print("Saving queries...")
with open(output_dir / "queries.tsv", "w", encoding="utf-8") as f:
    for item in queries:
        query_id = item["query_id"]
        query_text = item["query"]
        f.write(f"{query_id}\t{query_text}\n")

print(f"✓ Saved {len(queries)} queries to queries.tsv")

# Download qrels if needed
print("\nDownloading MS MARCO qrels (relevance judgments)...")
qrels = load_dataset("microsoft/ms_marco", "v1.1", split="dev_qrels")

print(f"✓ Downloaded {len(qrels)} qrel entries")
print("\nMS MARCO data ready for evaluation!")
