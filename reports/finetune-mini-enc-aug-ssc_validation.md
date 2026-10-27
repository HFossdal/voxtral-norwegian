# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.90 [94.34, 95.48] | 94.77 [94.20, 95.33] | 59.19 [58.40, 60.05] | 10.7% | 2.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | 6.17 [5.94, 6.39] | 6.15 [5.92, 6.40] | 2.22 [2.13, 2.32] | 87.4% | 94.6% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) |
| --- | --- | --- | --- | --- |
| region = east | 2700 | 95.65 [94.75, 96.58] | 19.86 [19.03, 20.87] | 5.32 [5.03, 5.62] |
| region = mid | 521 | 96.72 [95.14, 98.62] | 27.98 [26.07, 29.92] | 7.05 [6.23, 7.90] |
| region = north | 412 | 90.64 [88.52, 92.67] | 21.48 [20.31, 22.69] | 5.06 [4.43, 5.69] |
| region = south | 480 | 93.23 [91.89, 94.76] | 18.92 [17.89, 20.05] | 5.40 [4.80, 5.99] |
| region = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 112.50 [50.00, 175.00] |
| region = west | 1294 | 95.06 [94.15, 95.93] | 24.48 [23.68, 25.32] | 7.96 [7.49, 8.47] |
| gender = female | 2352 | 94.37 [93.53, 95.23] | 20.50 [19.57, 21.68] | 5.27 [4.97, 5.58] |
| gender = male | 3055 | 95.30 [94.62, 96.02] | 22.81 [22.22, 23.37] | 6.83 [6.54, 7.13] |
| gender = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 112.50 [50.00, 175.00] |
| duration_bucket = 0-3s | 1179 | 104.08 [102.82, 105.55] | 35.62 [33.57, 37.94] | 9.40 [8.38, 10.43] |
| duration_bucket = 10s+ | 929 | 91.27 [90.09, 92.41] | 21.38 [20.21, 22.84] | 5.82 [5.43, 6.18] |
| duration_bucket = 3-6s | 1607 | 98.67 [97.75, 99.63] | 21.45 [20.57, 22.31] | 6.28 [5.80, 6.81] |
| duration_bucket = 6-10s | 1694 | 94.26 [93.38, 95.17] | 20.14 [19.56, 20.82] | 5.86 [5.51, 6.19] |
| seen_speaker = False | 3037 | 95.47 [94.77, 96.15] | 22.31 [21.73, 22.97] | 6.75 [6.43, 7.08] |
| seen_speaker = True | 2372 | 94.18 [93.32, 95.06] | 21.21 [20.35, 22.28] | 5.44 [5.17, 5.74] |
| has_number = False | 4775 | 94.77 [94.20, 95.33] | 20.44 [20.02, 20.89] | 6.15 [5.92, 6.40] |
| has_number = True | 634 | 95.63 [94.14, 97.03] | 29.99 [27.74, 33.04] | 6.25 [5.72, 6.95] |
| has_name = False | 3734 | 94.87 [94.19, 95.51] | 20.97 [20.29, 21.86] | 5.82 [5.56, 6.09] |
| has_name = True | 1675 | 94.94 [94.03, 95.99] | 23.40 [22.65, 24.12] | 6.81 [6.41, 7.22] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | -73.08 | [-73.77, -72.36] | 0.000 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | -88.73 | [-89.31, -88.16] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 (480 ms): `/content/voxtral-norwegian/reports/runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-3B-2507 (language: nb): `/content/voxtral-norwegian/reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb): `/content/drive/MyDrive/voxtral-runs/20261025-042251_finetune-mini-enc-aug-ssc-step8000`
