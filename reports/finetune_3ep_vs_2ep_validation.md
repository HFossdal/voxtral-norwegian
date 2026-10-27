# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms) | 24.73 [24.30, 25.15] | 24.38 [23.90, 24.82] | 8.69 [8.51, 8.86] | 65.1% | 75.9% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | 22.64 [22.21, 23.07] | 22.35 [21.89, 22.81] | 7.93 [7.75, 8.11] | 68.6% | 81.3% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 22.65 [22.04, 23.26] | 20.91 [20.31, 21.48] |
| region = mid | 521 | 24.80 [23.33, 26.32] | 22.25 [20.85, 23.73] |
| region = north | 412 | 22.79 [21.56, 24.17] | 20.06 [18.78, 21.33] |
| region = south | 480 | 22.69 [21.43, 23.88] | 20.26 [18.99, 21.44] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] |
| region = west | 1294 | 29.67 [28.77, 30.55] | 27.43 [26.58, 28.30] |
| gender = female | 2352 | 22.50 [21.82, 23.14] | 20.41 [19.82, 21.02] |
| gender = male | 3055 | 26.43 [25.84, 27.02] | 24.33 [23.77, 24.93] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] |
| duration_bucket = 0-3s | 1179 | 28.64 [26.91, 30.42] | 25.76 [24.05, 27.40] |
| duration_bucket = 10s+ | 929 | 23.56 [22.82, 24.30] | 21.31 [20.55, 22.01] |
| duration_bucket = 3-6s | 1607 | 25.76 [24.78, 26.67] | 24.12 [23.18, 25.05] |
| duration_bucket = 6-10s | 1694 | 24.48 [23.83, 25.14] | 22.39 [21.73, 23.02] |
| seen_speaker = False | 3037 | 26.33 [25.72, 26.93] | 24.16 [23.59, 24.73] |
| seen_speaker = True | 2372 | 22.76 [22.19, 23.32] | 20.76 [20.18, 21.33] |
| has_number = False | 4775 | 24.38 [23.90, 24.82] | 22.35 [21.89, 22.81] |
| has_number = True | 634 | 26.83 [25.70, 28.13] | 24.34 [23.24, 25.58] |
| has_name = False | 3734 | 24.37 [23.85, 24.92] | 22.27 [21.74, 22.79] |
| has_name = True | 1675 | 25.42 [24.69, 26.18] | 23.33 [22.63, 24.08] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | -2.09 | [-2.32, -1.84] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002264 (480 ms): `reports/runs/20261020-123701_finetune-realtime-step2264`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms): `reports/runs/20261020-170946_finetune-realtime-3ep-step3396`
