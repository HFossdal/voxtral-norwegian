# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms) | 24.73 [24.30, 25.15] | 24.38 [23.90, 24.82] | 8.69 [8.51, 8.86] | 65.1% | 75.9% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | 22.64 [22.21, 23.07] | 22.35 [21.89, 22.81] | 7.93 [7.75, 8.11] | 68.6% | 81.3% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | 15.23 [14.90, 15.57] | 15.07 [14.67, 15.43] | 5.07 [4.94, 5.20] | 75.8% | 90.4% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | 14.54 [14.21, 14.88] | 14.38 [14.00, 14.75] | 4.87 [4.73, 5.00] | 77.5% | 89.5% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | 12.77 [12.46, 13.07] | 12.68 [12.31, 13.03] | 4.23 [4.11, 4.36] | 80.2% | 91.8% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | 12.17 [11.87, 12.47] | 12.08 [11.75, 12.42] | 4.08 [3.96, 4.19] | 81.9% | 93.6% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| region = east | 2700 | 19.86 [19.03, 20.87] | 22.65 [22.04, 23.26] | 20.91 [20.31, 21.48] | 13.27 [12.80, 13.75] | 12.64 [12.20, 13.11] | 10.81 [10.40, 11.23] | 10.25 [9.84, 10.64] |
| region = mid | 521 | 27.98 [26.07, 29.92] | 24.80 [23.33, 26.32] | 22.25 [20.85, 23.73] | 15.13 [14.02, 16.25] | 14.50 [13.41, 15.59] | 12.53 [11.58, 13.51] | 11.98 [10.95, 12.99] |
| region = north | 412 | 21.48 [20.31, 22.69] | 22.79 [21.56, 24.17] | 20.06 [18.78, 21.33] | 12.96 [11.88, 14.03] | 12.43 [11.33, 13.47] | 10.62 [9.60, 11.64] | 10.14 [9.17, 11.10] |
| region = south | 480 | 18.92 [17.89, 20.05] | 22.69 [21.43, 23.88] | 20.26 [18.99, 21.44] | 13.86 [12.90, 14.79] | 13.11 [12.17, 14.03] | 11.64 [10.76, 12.52] | 11.30 [10.42, 12.14] |
| region = unknown | 2 | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] | 125.00 [75.00, 175.00] | 125.00 [75.00, 175.00] | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 24.48 [23.68, 25.32] | 29.67 [28.77, 30.55] | 27.43 [26.58, 28.30] | 19.86 [19.12, 20.59] | 19.03 [18.33, 19.77] | 17.32 [16.66, 17.98] | 16.53 [15.83, 17.21] |
| gender = female | 2352 | 20.50 [19.57, 21.68] | 22.50 [21.82, 23.14] | 20.41 [19.82, 21.02] | 12.95 [12.44, 13.44] | 12.38 [11.89, 12.84] | 10.58 [10.12, 10.99] | 10.00 [9.59, 10.41] |
| gender = male | 3055 | 22.81 [22.22, 23.37] | 26.43 [25.84, 27.02] | 24.33 [23.77, 24.93] | 16.96 [16.46, 17.46] | 16.19 [15.70, 16.64] | 14.44 [13.99, 14.89] | 13.83 [13.35, 14.28] |
| gender = unknown | 2 | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] | 125.00 [75.00, 175.00] | 125.00 [75.00, 175.00] | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 35.62 [33.57, 37.94] | 28.64 [26.91, 30.42] | 25.76 [24.05, 27.40] | 17.42 [16.04, 18.98] | 16.43 [15.03, 17.86] | 14.05 [12.83, 15.38] | 13.62 [12.42, 14.95] |
| duration_bucket = 10s+ | 929 | 21.38 [20.21, 22.84] | 23.56 [22.82, 24.30] | 21.31 [20.55, 22.01] | 14.62 [14.01, 15.21] | 14.11 [13.54, 14.69] | 12.21 [11.69, 12.81] | 11.70 [11.17, 12.24] |
| duration_bucket = 3-6s | 1607 | 21.45 [20.57, 22.31] | 25.76 [24.78, 26.67] | 24.12 [23.18, 25.05] | 15.60 [14.85, 16.33] | 14.76 [13.97, 15.47] | 13.08 [12.41, 13.76] | 12.40 [11.71, 13.11] |
| duration_bucket = 6-10s | 1694 | 20.14 [19.56, 20.82] | 24.48 [23.83, 25.14] | 22.39 [21.73, 23.02] | 15.17 [14.68, 15.68] | 14.47 [13.97, 14.99] | 12.85 [12.37, 13.39] | 12.20 [11.70, 12.69] |
| seen_speaker = False | 3037 | 22.31 [21.73, 22.97] | 26.33 [25.72, 26.93] | 24.16 [23.59, 24.73] | 16.95 [16.46, 17.43] | 16.21 [15.75, 16.68] | 14.37 [13.93, 14.84] | 13.86 [13.40, 14.31] |
| seen_speaker = True | 2372 | 21.21 [20.35, 22.28] | 22.76 [22.19, 23.32] | 20.76 [20.18, 21.33] | 13.11 [12.68, 13.56] | 12.49 [12.06, 12.96] | 10.80 [10.36, 11.23] | 10.10 [9.70, 10.51] |
| has_number = False | 4775 | 20.44 [20.02, 20.89] | 24.38 [23.90, 24.82] | 22.35 [21.89, 22.81] | 15.07 [14.67, 15.43] | 14.38 [14.00, 14.75] | 12.68 [12.31, 13.03] | 12.08 [11.75, 12.42] |
| has_number = True | 634 | 29.99 [27.74, 33.04] | 26.83 [25.70, 28.13] | 24.34 [23.24, 25.58] | 16.19 [15.23, 17.22] | 15.53 [14.58, 16.50] | 13.32 [12.43, 14.29] | 12.72 [11.88, 13.58] |
| has_name = False | 3734 | 20.97 [20.29, 21.86] | 24.37 [23.85, 24.92] | 22.27 [21.74, 22.79] | 14.80 [14.35, 15.22] | 14.13 [13.75, 14.54] | 12.33 [11.90, 12.72] | 11.82 [11.41, 12.21] |
| has_name = True | 1675 | 23.40 [22.65, 24.12] | 25.42 [24.69, 26.18] | 23.33 [22.63, 24.08] | 16.04 [15.42, 16.64] | 15.32 [14.76, 15.89] | 13.60 [13.08, 14.20] | 12.84 [12.32, 13.39] |

## Paired comparison against Voxtral-Mini-3B-2507 (language: nb)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms) | +2.91 | [+2.28, +3.49] | 1.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | +0.82 | [+0.17, +1.42] | 0.992 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | -6.59 | [-7.25, -6.06] | 0.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | -7.27 | [-7.93, -6.76] | 0.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | -9.05 | [-9.68, -8.51] | 0.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | -9.64 | [-10.28, -9.10] | 0.000 |

## Run directories

- Voxtral-Mini-3B-2507 (language: nb): `reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms): `reports/runs/20261020-123701_finetune-realtime-step2264`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms): `reports/runs/20261020-170946_finetune-realtime-3ep-step3396`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms): `reports/runs/20261020-232609_finetune-realtime-enc-step3396`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms): `reports/runs/20261021-102102_finetune-realtime-enc-5ep-step5660`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms): `reports/runs/20261021-205210_finetune-realtime-enc-5ep-aug-step5660`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms): `reports/runs/20261022-124210_finetune-realtime-enc-8ep-aug-step9056`
