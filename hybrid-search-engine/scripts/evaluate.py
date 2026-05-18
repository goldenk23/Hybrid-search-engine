# ============================================================
# LATENCY METRICS
# ============================================================
#
# Metric              | Min     | Max         | Meaning
# -------------------- | -------- | ----------- | -------
# p50 latency         | 0 ms    | No fixed max| Median query time
# p95 latency         | 0 ms    | No fixed max| Slow-query experience, 95% of queries are faster than this
#
# ============================================================
# 1. NDCG@10 (Normalized Discounted Cumulative Gain at 10)
# ============================================================
#
# What it measures:
#     Quality of the ordering of the top 10 search results.
#
# Intuition:
#     Rewards systems that place highly relevant documents
#     near the top of the ranking.
#
#     Penalizes relevant documents that appear lower
#     in the top 10 results.
#
# Range:
#     0 -> Worst ranking
#     1 -> Perfect ranking
#
# When to care:
#     Useful when user satisfaction depends heavily on
#     the quality of the first page of results.
#
# ============================================================
# 2. MRR (Mean Reciprocal Rank)
# ============================================================
#
# What it measures:
#     Position of the FIRST relevant result,
#     averaged across all queries.
#
# Intuition:
#     - If the first relevant result is usually at rank 1:
#           MRR ~= 1
#
#     - If the first relevant result is usually at rank 2:
#           MRR ~= 0.5
#
# Range:
#     0 -> No relevant results found
#     1 -> First result always relevant
#
# When to care:
#     Useful when users want to quickly find at least
#     one good/relevant result.
#
# ============================================================
# 3. Recall@100
# ============================================================
#
# What it measures:
#     Fraction of ALL relevant documents that appear
#     within the top 100 retrieved results.
#
# Intuition:
#     If a relevant document is missing from the top 100,
#     later stages (such as re-rankers) cannot recover it.
#
# Range:
#     0 -> No relevant documents retrieved
#     1 -> All relevant documents retrieved
#
# When to care:
#     Important for retrieval pipelines where the first
#     stage must preserve relevant documents for later
#     processing and re-ranking.
#
# ============================================================

    
"""
Usage:
    python scripts/evaluate.py `
  --queries data/msmarco/queries.dev.small.tsv `
  --qrels data/msmarco/qrels.dev.small.tsv `
  --max-queries 1000

"""

import argparse # handles command line arguments like --queries and  --qrels
import csv # reads the tsv files containing queries and relevance judgments
import sys# allows us to modify the Python path to include the project root for imports
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.config import DATA_DIR
from src.search.bm25 import BM25Search

# load queries and relavance judgments
def load_qrels(qrels_path: Path) -> dict[str, set[str]]:
    """
    Load relevance judgments.

    Expected qrels format:
        query_id<TAB>0<TAB>document_id<TAB>relevance
    """
    qrels=defaultdict(set)
    """
    defaultdict(set) creates a dictionary where each key
    automatically stores an empty set by default.

    Useful when mapping one key to multiple unique values
    without manually initializing the set first.
            Example:
        --------
        from collections import defaultdict

        qrels = defaultdict(set)

        qrels["query1"].add("doc1")
        qrels["query1"].add("doc2")

        Result:
        -------
        {
            "query1": {"doc1", "doc2"}
        }

    """
    
    with qrels_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        
        for row in reader:
            if len(row)<4:
                continue;
            query_id=row[0]
            document_id=row[2]
            relavance=int(row[3])
            
            if relavance>0:
                qrels[query_id].add(document_id)
    
    return qrels

def load_queries(queries_path: Path) -> dict[str, str]:
    """
    Load queries.

    Expected queries format:
        query_id<TAB>query_text
    """
    queries={}
    
    with queries_path.open("r", encoding="utf-8") as file:
        reader = csv.reader(file, delimiter="\t")
        
        for row in reader:
            if len(row)<2:
                continue;
            query_id=row[0]
            query_text=row[1]
            queries[query_id]=query_text
    
    return queries 


def compute_ndcg(ranked_doc_ids: list[str], relevant_id: set[str], k: int =10)-> float:
    """
    Compute NDCG@k for a single query.

    """
    dcg=0.0
    
    for index, doc_id in enumerate(ranked_doc_ids[:k]):
        if doc_id in relevant_id:
            # standard DCG formula: relevance / log2(position + 1)
            dcg+=1.0/np.log2(index+2)# index+2 because index starts at 0 but log2(1) should be the first position
     
     # compute best possbile dcg if all documents appeared in perfect order relative to relevant document in top k (for comparison)       
    ideal_dcg=0.0
    
    for index in range(min(len(relevant_id), k)):
        ideal_dcg+=1.0/np.log2(index+2)
        
    if ideal_dcg==0:
        return 0.0
    return dcg/ideal_dcg

def compute_mrr(ranked_doc_ids: list[str], relevant_id: set[str]) -> float:
    """
    Compute MRR for a single query.
    It is based on the position of first relevant document in the ranked list
    """
    for index, doc_id in enumerate(ranked_doc_ids):
        if doc_id in relevant_id:
            return 1.0/(index+1)
        
    return 0.0

def compute_recall(ranked_doc_ids: list[str], relevant_id: set[str], k: int =100) ->float:
    """
    Compute Recall@k for a single query.
    It is based on the fraction of relevant documents that appear in top k (retrieved documents)
    """
    if not relevant_id:
         return 0.0
    
    retrieved_documents=set(ranked_doc_ids[:k])
    found = retrieved_documents.intersection(relevant_id)
    return len(found)/len(relevant_id)


def evaluate_bm25(
    queries_path: Path,
    qrels_path: Path,
    max_queries: int | None=None,
) -> dict[str, float]:
    """
    Evaluate BM25 search engine on the given queries and relevance judgments.
    """
    queries = load_queries(queries_path)
    qrels = load_qrels(qrels_path)
    
    # create new dictionary(key value pair) containing only queries whose query_id exists in
    # qrels (relevance judgments) to ensure we only evaluate on queries with relevance judgments
     
    eval_queries = {
        query_id:query_test
        for query_id, query_test in queries.items()
            if query_id in qrels
    }
    # limiting the number of eval_queries to max_queries if it is provided 
    if max_queries is not None:
        eval_queries=dict(list(eval_queries.items())[:max_queries])
    
    bm25 = BM25Search()
    
    ndcg_scores = []
    mrr_scores = []
    recall_scores = []
    
    for query_id, query_text in tqdm(eval_queries.items(), desc="Evaluating BM25"):
        results = bm25.search(query_text, top_k=100)
        ranked_doc_ids = [result["id"] for result in results]
        relevant_id = qrels[query_id]
        
        ndcg_scores.append(compute_ndcg(ranked_doc_ids, relevant_id, k=10))  
        mrr_scores.append(compute_mrr(ranked_doc_ids, relevant_id)) 
        recall_scores.append(compute_recall(ranked_doc_ids, relevant_id, k=100)) 
        
    metrics = {
        "ndcg@10": float(np.mean(ndcg_scores)) if ndcg_scores else 0.0,
        "mrr": float(np.mean(mrr_scores)) if mrr_scores else 0.0,
        "recall@100": float(np.mean(recall_scores)) if recall_scores else 0.0,
        "queries": float(len(eval_queries)),
    }
    
    
    print("\nBM25 Baseline Results")
    print("=" * 60)
    print(f"NDCG@10:    {metrics['ndcg@10']:.4f}")
    print(f"MRR:        {metrics['mrr']:.4f}")
    print(f"Recall@100: {metrics['recall@100']:.4f}")
    print(f"Queries:    {int(metrics['queries'])}")
    print("=" * 60)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 search engine quality on MS MARCO queries.")
    parser.add_argument(
        "--queries",
        type = Path,
        default = DATA_DIR / "msmarco/queries.dev.small.tsv",
        help = "path to query TSV file"
    )
    
    parser.add_argument(
        "--qrels",
        type = Path,
        default = DATA_DIR / "msmarco/qrels.dev.small.tsv",
        help = "path to relevance judgments(qrels) TSV file"
    )
    
    parser.add_argument(
        "--max-queries",
        type = int,
        default = 100,
        help = "maximum number of queries to evaluate"
    )
    
    args = parser.parse_args()
    
    evaluate_bm25(
        queries_path=args.queries,
        qrels_path = args.qrels,
        max_queries = args.max_queries,
    )

# Run the main function when this script is executed not imported as a module
if __name__ == "__main__":
    main()
