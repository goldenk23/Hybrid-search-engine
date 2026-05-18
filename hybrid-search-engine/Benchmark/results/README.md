# Hybrid Search Benchmark Results

This file is updated automatically after each benchmark run.

## Presentation Summary Table

| Corpus | System | Queries | NDCG@10 | MRR@10 | Recall@100 | p50 Latency (ms) | p95 Latency (ms) | Compact Size (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 750k | BM25 | 26 | 0.1454 | 0.1065 | 0.5769 | 10.64 | 17.45 | 652.37 |
| 750k | Vector | 26 | 0.3201 | 0.2416 | 0.9038 | 151.29 | 180.43 | 652.37 |
| 750k | Hybrid RRF | 26 | 0.1905 | 0.1213 | 0.8654 | 169.43 | 193.72 | 652.37 |
| 750k | Weighted RRF (BM25 0.50, Vector 1.00) | 26 | 0.2024 | 0.1352 | 0.9038 | 173.84 | 218.91 | 652.37 |
| 750k | Weighted RRF (BM25 0.25, Vector 1.00) | 26 | 0.2406 | 0.1494 | 0.9038 | 182.03 | 205.52 | 652.37 |
| 750k | Hybrid RRF + Cross-Encoder | 26 | 0.4470 | 0.3587 | 0.7308 | 1080.05 | 1421.28 | 652.37 |
| 1M | BM25 | 315 | 0.2052 | 0.1624 | 0.6444 | 17.91 | 33.83 | 762.61 |
| 1M | Vector | 315 | 0.3794 | 0.3105 | 0.9048 | 116.70 | 132.49 | 762.61 |
| 1M | Hybrid RRF | 315 | 0.3339 | 0.2710 | 0.8905 | 148.27 | 185.30 | 762.61 |
| 1M | Weighted RRF (BM25 0.50, Vector 1.00) | 315 | 0.3568 | 0.2874 | 0.9016 | 147.58 | 174.59 | 762.61 |
| 1M | Weighted RRF (BM25 0.25, Vector 1.00) | 315 | 0.3737 | 0.2992 | 0.9048 | 150.84 | 172.36 | 762.61 |
| 1M | Hybrid RRF + Cross-Encoder | 100 | 0.4068 | 0.3319 | 0.6400 | 844.87 | 991.17 | 762.61 |
| 2M | BM25 | 32 | 0.1311 | 0.1128 | 0.4375 | 58.27 | 101.31 | 1,342.24 |
| 2M | Vector | 32 | 0.1979 | 0.1716 | 0.8594 | 1185.48 | 1410.96 | 1,342.24 |
| 2M | Hybrid RRF | 32 | 0.2013 | 0.1574 | 0.8125 | 1203.61 | 1424.41 | 1,342.24 |
| 2M | Weighted RRF (BM25 0.50, Vector 1.00) | 32 | 0.2243 | 0.1787 | 0.8438 | 1134.21 | 1312.17 | 1,342.24 |
| 2M | Weighted RRF (BM25 0.25, Vector 1.00) | 32 | 0.1989 | 0.1549 | 0.8594 | 1092.96 | 1365.69 | 1,342.24 |
| 2M | Hybrid RRF + Cross-Encoder | 32 | 0.3174 | 0.2521 | 0.5312 | 790.57 | 1013.66 | 1,342.24 |

## Result Files

| Corpus | JSON | Report |
|---|---|---|
| 750k | `750k.json` | `750k.md` |
| 1M | `1M.json` | `1M.md` |
| 2M | `2M.json` | `2M.md` |

## Notes

- Re-running a benchmark with the same `--corpus-label` updates that corpus JSON and report.
- This README is rebuilt from all JSON files in this folder after every run.
- Higher retrieval metrics are better; lower latency metrics are better.
