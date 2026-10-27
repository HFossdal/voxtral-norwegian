# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.90 [94.34, 95.48] | 94.77 [94.20, 95.33] | 59.19 [58.40, 60.05] | 10.7% | 2.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.25 [8.98, 9.51] | 9.06 [8.77, 9.36] | 3.14 [3.03, 3.24] | 84.6% | 91.4% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) |
| --- | --- | --- | --- | --- |
| region = east | 2700 | 95.65 [94.75, 96.58] | 19.86 [19.03, 20.87] | 7.72 [7.38, 8.08] |
| region = mid | 521 | 96.72 [95.14, 98.62] | 27.98 [26.07, 29.92] | 9.35 [8.49, 10.27] |
| region = north | 412 | 90.64 [88.52, 92.67] | 21.48 [20.31, 22.69] | 7.94 [7.11, 8.73] |
| region = south | 480 | 93.23 [91.89, 94.76] | 18.92 [17.89, 20.05] | 8.03 [7.21, 8.85] |
| region = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 95.06 [94.15, 95.93] | 24.48 [23.68, 25.32] | 12.73 [12.11, 13.34] |
| gender = female | 2352 | 94.37 [93.53, 95.23] | 20.50 [19.57, 21.68] | 7.59 [7.23, 7.95] |
| gender = male | 3055 | 95.30 [94.62, 96.02] | 22.81 [22.22, 23.37] | 10.52 [10.13, 10.92] |
| gender = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 104.08 [102.82, 105.55] | 35.62 [33.57, 37.94] | 12.53 [11.33, 13.75] |
| duration_bucket = 10s+ | 929 | 91.27 [90.09, 92.41] | 21.38 [20.21, 22.84] | 8.79 [8.32, 9.27] |
| duration_bucket = 3-6s | 1607 | 98.67 [97.75, 99.63] | 21.45 [20.57, 22.31] | 9.35 [8.77, 9.98] |
| duration_bucket = 6-10s | 1694 | 94.26 [93.38, 95.17] | 20.14 [19.56, 20.82] | 9.06 [8.65, 9.51] |
| seen_speaker = False | 3037 | 95.47 [94.77, 96.15] | 22.31 [21.73, 22.97] | 10.37 [9.98, 10.78] |
| seen_speaker = True | 2372 | 94.18 [93.32, 95.06] | 21.21 [20.35, 22.28] | 7.89 [7.56, 8.26] |
| has_number = False | 4775 | 94.77 [94.20, 95.33] | 20.44 [20.02, 20.89] | 9.06 [8.77, 9.36] |
| has_number = True | 634 | 95.63 [94.14, 97.03] | 29.99 [27.74, 33.04] | 10.41 [9.73, 11.19] |
| has_name = False | 3734 | 94.87 [94.19, 95.51] | 20.97 [20.29, 21.86] | 8.88 [8.54, 9.21] |
| has_name = True | 1675 | 94.94 [94.03, 95.99] | 23.40 [22.65, 24.12] | 9.96 [9.46, 10.46] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | -73.08 | [-73.77, -72.36] | 0.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | -85.64 | [-86.25, -85.04] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 (480 ms): `/content/voxtral-norwegian/reports/runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-3B-2507 (language: nb): `/content/voxtral-norwegian/reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `/content/drive/MyDrive/voxtral-runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
