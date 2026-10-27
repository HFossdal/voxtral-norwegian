# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.90 [94.34, 95.48] | 94.77 [94.20, 95.33] | 59.19 [58.40, 60.05] | 10.7% | 2.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.84 [5.61, 6.05] | 5.83 [5.61, 6.10] | 2.09 [2.00, 2.18] | 87.7% | 95.5% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- | --- |
| region = east | 2700 | 95.65 [94.75, 96.58] | 19.86 [19.03, 20.87] | 4.97 [4.70, 5.25] |
| region = mid | 521 | 96.72 [95.14, 98.62] | 27.98 [26.07, 29.92] | 6.67 [5.88, 7.50] |
| region = north | 412 | 90.64 [88.52, 92.67] | 21.48 [20.31, 22.69] | 4.52 [3.96, 5.05] |
| region = south | 480 | 93.23 [91.89, 94.76] | 18.92 [17.89, 20.05] | 4.95 [4.38, 5.54] |
| region = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 87.50 [75.00, 100.00] |
| region = west | 1294 | 95.06 [94.15, 95.93] | 24.48 [23.68, 25.32] | 7.82 [7.35, 8.29] |
| gender = female | 2352 | 94.37 [93.53, 95.23] | 20.50 [19.57, 21.68] | 4.83 [4.55, 5.12] |
| gender = male | 3055 | 95.30 [94.62, 96.02] | 22.81 [22.22, 23.37] | 6.60 [6.30, 6.90] |
| gender = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 87.50 [75.00, 100.00] |
| duration_bucket = 0-3s | 1179 | 104.08 [102.82, 105.55] | 35.62 [33.57, 37.94] | 9.03 [7.98, 10.09] |
| duration_bucket = 10s+ | 929 | 91.27 [90.09, 92.41] | 21.38 [20.21, 22.84] | 5.41 [5.00, 5.79] |
| duration_bucket = 3-6s | 1607 | 98.67 [97.75, 99.63] | 21.45 [20.57, 22.31] | 5.87 [5.40, 6.36] |
| duration_bucket = 6-10s | 1694 | 94.26 [93.38, 95.17] | 20.14 [19.56, 20.82] | 5.66 [5.30, 5.99] |
| seen_speaker = False | 3037 | 95.47 [94.77, 96.15] | 22.31 [21.73, 22.97] | 6.58 [6.27, 6.89] |
| seen_speaker = True | 2372 | 94.18 [93.32, 95.06] | 21.21 [20.35, 22.28] | 4.92 [4.67, 5.19] |
| has_number = False | 4775 | 94.77 [94.20, 95.33] | 20.44 [20.02, 20.89] | 5.83 [5.61, 6.10] |
| has_number = True | 634 | 95.63 [94.14, 97.03] | 29.99 [27.74, 33.04] | 5.86 [5.35, 6.50] |
| has_name = False | 3734 | 94.87 [94.19, 95.51] | 20.97 [20.29, 21.86] | 5.49 [5.24, 5.76] |
| has_name = True | 1675 | 94.94 [94.03, 95.99] | 23.40 [22.65, 24.12] | 6.49 [6.09, 6.90] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | -73.08 | [-73.77, -72.36] | 0.000 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -89.06 | [-89.66, -88.47] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 (480 ms): `/content/voxtral-norwegian/reports/runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-3B-2507 (language: nb): `/content/voxtral-norwegian/reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `/content/drive/MyDrive/voxtral-runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`
