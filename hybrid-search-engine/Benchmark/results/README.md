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
| 2M | BM25 | 85 | 0.1397 | 0.1169 | 0.5059 | 36.84 | 102.38 | 1,342.24 |
| 2M | Vector | 85 | 0.2491 | 0.1931 | 0.8059 | 209.59 | 235.50 | 1,342.24 |
| 2M | Hybrid RRF | 85 | 0.2302 | 0.1849 | 0.7882 | 259.17 | 298.27 | 1,342.24 |
| 2M | Weighted RRF (BM25 0.50, Vector 1.00) | 85 | 0.2564 | 0.2056 | 0.8118 | 242.07 | 274.51 | 1,342.24 |
| 2M | Weighted RRF (BM25 0.25, Vector 1.00) | 85 | 0.2664 | 0.2080 | 0.8059 | 245.06 | 282.67 | 1,342.24 |
| 2M | Hybrid RRF + Cross-Encoder | 85 | 0.3681 | 0.2962 | 0.6000 | 869.47 | 1086.97 | 1,342.24 |
| 2MA | BM25 | 153 | 0.1753 | 0.1423 | 0.5621 | 32.81 | 64.45 | 1,342.24 |
| 2MA | Vector | 153 | 0.3051 | 0.2423 | 0.8660 | 215.92 | 1218.71 | 1,342.24 |
| 2MA | Hybrid RRF | 153 | 0.2758 | 0.2225 | 0.8431 | 410.82 | 1355.02 | 1,342.24 |
| 2MA | Weighted RRF (BM25 0.50, Vector 1.00) | 153 | 0.3028 | 0.2382 | 0.8693 | 241.70 | 280.12 | 1,342.24 |
| 2MA | Weighted RRF (BM25 0.25, Vector 1.00) | 153 | 0.3139 | 0.2460 | 0.8660 | 1050.10 | 1207.97 | 1,342.24 |
| 2MA | Hybrid RRF + Cross-Encoder | 100 | 0.3607 | 0.2808 | 0.6200 | 1955.62 | 2196.24 | 1,342.24 |

## Result Files

| Corpus | JSON | Report |
|---|---|---|
| 750k | `750k.json` | `750k.md` |
| 1M | `1M.json` | `1M.md` |
| 2M | `2M.json` | `2M.md` |
| 2MA | `2MA.json` | `2MA.md` |

## Notes

- Re-running a benchmark with the same `--corpus-label` updates that corpus JSON and report.
- This README is rebuilt from all JSON files in this folder after every run.
- Higher retrieval metrics are better; lower latency metrics are better.
