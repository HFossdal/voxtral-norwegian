# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.90 [94.34, 95.48] | 94.77 [94.20, 95.33] | 59.19 [58.40, 60.05] | 10.7% | 2.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) | 11.71 [11.42, 12.01] | 11.68 [11.34, 12.01] | 3.92 [3.81, 4.04] | 82.7% | 93.7% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) |
| --- | --- | --- | --- | --- |
| region = east | 2700 | 95.65 [94.75, 96.58] | 19.86 [19.03, 20.87] | 9.83 [9.41, 10.21] |
| region = mid | 521 | 96.72 [95.14, 98.62] | 27.98 [26.07, 29.92] | 12.02 [11.02, 13.05] |
| region = north | 412 | 90.64 [88.52, 92.67] | 21.48 [20.31, 22.69] | 9.67 [8.69, 10.56] |
| region = south | 480 | 93.23 [91.89, 94.76] | 18.92 [17.89, 20.05] | 10.59 [9.67, 11.50] |
| region = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 95.06 [94.15, 95.93] | 24.48 [23.68, 25.32] | 15.94 [15.29, 16.60] |
| gender = female | 2352 | 94.37 [93.53, 95.23] | 20.50 [19.57, 21.68] | 9.67 [9.22, 10.09] |
| gender = male | 3055 | 95.30 [94.62, 96.02] | 22.81 [22.22, 23.37] | 13.26 [12.81, 13.69] |
| gender = unknown | 2 | 100.00 [100.00, 100.00] | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 104.08 [102.82, 105.55] | 35.62 [33.57, 37.94] | 13.29 [12.04, 14.56] |
| duration_bucket = 10s+ | 929 | 91.27 [90.09, 92.41] | 21.38 [20.21, 22.84] | 11.17 [10.65, 11.71] |
| duration_bucket = 3-6s | 1607 | 98.67 [97.75, 99.63] | 21.45 [20.57, 22.31] | 12.08 [11.41, 12.73] |
| duration_bucket = 6-10s | 1694 | 94.26 [93.38, 95.17] | 20.14 [19.56, 20.82] | 11.70 [11.22, 12.15] |
| seen_speaker = False | 3037 | 95.47 [94.77, 96.15] | 22.31 [21.73, 22.97] | 13.27 [12.81, 13.70] |
| seen_speaker = True | 2372 | 94.18 [93.32, 95.06] | 21.21 [20.35, 22.28] | 9.80 [9.40, 10.21] |
| has_number = False | 4775 | 94.77 [94.20, 95.33] | 20.44 [20.02, 20.89] | 11.68 [11.34, 12.01] |
| has_number = True | 634 | 95.63 [94.14, 97.03] | 29.99 [27.74, 33.04] | 11.91 [11.11, 12.84] |
| has_name = False | 3734 | 94.87 [94.19, 95.51] | 20.97 [20.29, 21.86] | 11.39 [11.01, 11.77] |
| has_name = True | 1675 | 94.94 [94.03, 95.99] | 23.40 [22.65, 24.12] | 12.33 [11.78, 12.85] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | -73.08 | [-73.77, -72.36] | 0.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) | -83.18 | [-83.78, -82.57] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 (480 ms): `/content/voxtral-norwegian/reports/runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-3B-2507 (language: nb): `/content/voxtral-norwegian/reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms): `/content/drive/MyDrive/voxtral-runs/20261023-045156_finetune-realtime-enc-8ep-aug-960ms-step8000`
