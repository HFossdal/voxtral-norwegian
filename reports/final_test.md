# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | 21.28 [20.75, 21.82] | 20.21 [19.62, 20.91] | 10.11 [9.76, 10.47] | 59.9% | 24.0% | 5006 | 38 |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.97 [94.38, 95.59] | 95.03 [94.43, 95.66] | 60.18 [59.32, 61.05] | 9.4% | 3.7% | 5006 | 38 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | 24.07 [23.58, 24.56] | 23.86 [23.33, 24.35] | 8.63 [8.45, 8.84] | 62.7% | 84.6% | 5006 | 38 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |

## WER by slice

| slice | utts | Voxtral-Mini-3B-2507 (language: nb) | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- | --- | --- |
| region = east | 2541 | 20.56 [19.67, 21.58] | 94.65 [93.86, 95.54] | 22.01 [21.37, 22.68] | 5.25 [4.96, 5.57] |
| region = mid | 353 | 22.29 [20.89, 23.68] | 96.84 [94.39, 99.54] | 20.76 [19.39, 22.15] | 5.27 [4.62, 5.94] |
| region = north | 554 | 24.87 [23.47, 26.22] | 94.05 [92.43, 95.69] | 24.42 [23.19, 25.68] | 6.60 [5.99, 7.27] |
| region = south | 272 | 18.97 [17.31, 20.55] | 94.08 [92.04, 96.26] | 23.50 [21.94, 25.24] | 5.04 [4.28, 5.84] |
| region = west | 1286 | 21.12 [20.31, 21.92] | 95.55 [94.68, 96.54] | 28.31 [27.44, 29.29] | 7.05 [6.62, 7.49] |
| gender = female | 1941 | 20.38 [19.67, 21.09] | 94.46 [93.41, 95.47] | 22.92 [22.22, 23.69] | 5.38 [5.05, 5.71] |
| gender = male | 3065 | 21.80 [21.07, 22.66] | 95.27 [94.57, 95.94] | 24.74 [24.15, 25.28] | 6.23 [5.93, 6.51] |
| duration_bucket = 0-3s | 916 | 34.16 [31.82, 36.73] | 103.22 [101.65, 104.83] | 28.99 [26.98, 31.13] | 8.97 [7.89, 10.06] |
| duration_bucket = 10s+ | 909 | 20.69 [19.86, 21.64] | 91.95 [90.89, 93.15] | 22.44 [21.64, 23.21] | 5.57 [5.19, 5.94] |
| duration_bucket = 3-6s | 1549 | 21.20 [20.34, 22.07] | 98.24 [97.30, 99.36] | 26.30 [25.33, 27.33] | 6.22 [5.74, 6.71] |
| duration_bucket = 6-10s | 1632 | 20.16 [19.18, 21.38] | 94.56 [93.78, 95.50] | 23.53 [22.81, 24.27] | 5.63 [5.27, 5.98] |
| seen_speaker = False | 5006 | 21.28 [20.75, 21.82] | 94.97 [94.38, 95.59] | 24.07 [23.58, 24.56] | 5.91 [5.67, 6.14] |
| has_number = False | 4363 | 20.21 [19.62, 20.91] | 95.03 [94.43, 95.66] | 23.86 [23.33, 24.35] | 5.86 [5.63, 6.12] |
| has_number = True | 643 | 26.85 [25.61, 28.16] | 94.67 [93.18, 96.30] | 25.14 [23.93, 26.24] | 6.19 [5.65, 6.73] |
| has_name = False | 3467 | 19.83 [19.28, 20.37] | 95.34 [94.58, 96.07] | 23.53 [22.98, 24.10] | 5.37 [5.13, 5.60] |
| has_name = True | 1539 | 23.93 [22.80, 25.31] | 94.29 [93.40, 95.27] | 25.07 [24.27, 25.83] | 6.92 [6.51, 7.32] |

## Paired comparison against Voxtral-Mini-3B-2507 (language: nb)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | +73.70 | [+72.92, +74.46] | 1.000 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | +2.79 | [+2.08, +3.41] | 1.000 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -15.36 | [-15.98, -14.86] | 0.000 |

## Run directories

- Voxtral-Mini-3B-2507 (language: nb): `reports/runs/20261019-222846_baseline-mini-nb`
- Voxtral-Mini-4B-Realtime-2602 (480 ms): `reports/runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms): `reports/runs/20261020-170946_finetune-realtime-3ep-step3396`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`
