# Hybrid Search Benchmark Results

Auto-generated. Re-run a benchmark to update.

| Corpus | System | Queries | NDCG@10 | MRR@10 | Recall@100 | p50 (ms) | p95 (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.5M | BM25 | 268 | 0.4167 | 0.3621 | 0.7774 | 38.64 | 62.36 |
| 1.5M | Vector | 268 | 0.6069 | 0.5522 | 0.9459 | 321.58 | 458.77 |
| 1.5M | Hybrid RRF (1.0 / 1.0) | 268 | 0.5646 | 0.5043 | 0.9496 | 359.74 | 535.20 |
| 1.5M | Weighted RRF (0.50 / 1.00) | 268 | 0.5948 | 0.5371 | 0.9571 | 520.12 | 712.38 |
| 1.5M | Weighted RRF (0.25 / 1.00) | 268 | 0.6016 | 0.5407 | 0.9459 | 360.18 | 557.29 |
| 1.5M | Hybrid + Cross-Encoder | 200 | 0.7289 | 0.6880 | 0.8625 | 1353.55 | 1676.01 |
