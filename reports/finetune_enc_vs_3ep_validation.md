# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | 22.64 [22.21, 23.07] | 22.35 [21.89, 22.81] | 7.93 [7.75, 8.11] | 68.6% | 81.3% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | 15.23 [14.90, 15.57] | 15.07 [14.67, 15.43] | 5.07 [4.94, 5.20] | 75.8% | 90.4% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 20.91 [20.31, 21.48] | 13.27 [12.80, 13.75] |
| region = mid | 521 | 22.25 [20.85, 23.73] | 15.13 [14.02, 16.25] |
| region = north | 412 | 20.06 [18.78, 21.33] | 12.96 [11.88, 14.03] |
| region = south | 480 | 20.26 [18.99, 21.44] | 13.86 [12.90, 14.79] |
| region = unknown | 2 | 112.50 [75.00, 150.00] | 125.00 [75.00, 175.00] |
| region = west | 1294 | 27.43 [26.58, 28.30] | 19.86 [19.12, 20.59] |
| gender = female | 2352 | 20.41 [19.82, 21.02] | 12.95 [12.44, 13.44] |
| gender = male | 3055 | 24.33 [23.77, 24.93] | 16.96 [16.46, 17.46] |
| gender = unknown | 2 | 112.50 [75.00, 150.00] | 125.00 [75.00, 175.00] |
| duration_bucket = 0-3s | 1179 | 25.76 [24.05, 27.40] | 17.42 [16.04, 18.98] |
| duration_bucket = 10s+ | 929 | 21.31 [20.55, 22.01] | 14.62 [14.01, 15.21] |
| duration_bucket = 3-6s | 1607 | 24.12 [23.18, 25.05] | 15.60 [14.85, 16.33] |
| duration_bucket = 6-10s | 1694 | 22.39 [21.73, 23.02] | 15.17 [14.68, 15.68] |
| seen_speaker = False | 3037 | 24.16 [23.59, 24.73] | 16.95 [16.46, 17.43] |
| seen_speaker = True | 2372 | 20.76 [20.18, 21.33] | 13.11 [12.68, 13.56] |
| has_number = False | 4775 | 22.35 [21.89, 22.81] | 15.07 [14.67, 15.43] |
| has_number = True | 634 | 24.34 [23.24, 25.58] | 16.19 [15.23, 17.22] |
| has_name = False | 3734 | 22.27 [21.74, 22.79] | 14.80 [14.35, 15.22] |
| has_name = True | 1675 | 23.33 [22.63, 24.08] | 16.04 [15.42, 16.64] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | -7.41 | [-7.76, -7.10] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-3ep step-003396 (480 ms): `reports/runs/20261020-170946_finetune-realtime-3ep-step3396`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms): `reports/runs/20261020-232609_finetune-realtime-enc-step3396`
