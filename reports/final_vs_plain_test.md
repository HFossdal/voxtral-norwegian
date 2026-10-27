# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | 24.07 [23.58, 24.56] | 23.86 [23.33, 24.35] | 8.63 [8.45, 8.84] | 62.7% | 84.6% | 5006 | 38 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2541 | 22.01 [21.37, 22.68] | 5.25 [4.96, 5.57] |
| region = mid | 353 | 20.76 [19.39, 22.15] | 5.27 [4.62, 5.94] |
| region = north | 554 | 24.42 [23.19, 25.68] | 6.60 [5.99, 7.27] |
| region = south | 272 | 23.50 [21.94, 25.24] | 5.04 [4.28, 5.84] |
| region = west | 1286 | 28.31 [27.44, 29.29] | 7.05 [6.62, 7.49] |
| gender = female | 1941 | 22.92 [22.22, 23.69] | 5.38 [5.05, 5.71] |
| gender = male | 3065 | 24.74 [24.15, 25.28] | 6.23 [5.93, 6.51] |
| duration_bucket = 0-3s | 916 | 28.99 [26.98, 31.13] | 8.97 [7.89, 10.06] |
| duration_bucket = 10s+ | 909 | 22.44 [21.64, 23.21] | 5.57 [5.19, 5.94] |
| duration_bucket = 3-6s | 1549 | 26.30 [25.33, 27.33] | 6.22 [5.74, 6.71] |
| duration_bucket = 6-10s | 1632 | 23.53 [22.81, 24.27] | 5.63 [5.27, 5.98] |
| seen_speaker = False | 5006 | 24.07 [23.58, 24.56] | 5.91 [5.67, 6.14] |
| has_number = False | 4363 | 23.86 [23.33, 24.35] | 5.86 [5.63, 6.12] |
| has_number = True | 643 | 25.14 [23.93, 26.24] | 6.19 [5.65, 6.73] |
| has_name = False | 3467 | 23.53 [22.98, 24.10] | 5.37 [5.13, 5.60] |
| has_name = True | 1539 | 25.07 [24.27, 25.83] | 6.92 [6.51, 7.32] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -18.15 | [-18.57, -17.73] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms): `reports/runs/20261020-170946_finetune-realtime-3ep-step3396`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`
