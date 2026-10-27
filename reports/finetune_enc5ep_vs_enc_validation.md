# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | 15.23 [14.90, 15.57] | 15.07 [14.67, 15.43] | 5.07 [4.94, 5.20] | 75.8% | 90.4% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | 14.54 [14.21, 14.88] | 14.38 [14.00, 14.75] | 4.87 [4.73, 5.00] | 77.5% | 89.5% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 13.27 [12.80, 13.75] | 12.64 [12.20, 13.11] |
| region = mid | 521 | 15.13 [14.02, 16.25] | 14.50 [13.41, 15.59] |
| region = north | 412 | 12.96 [11.88, 14.03] | 12.43 [11.33, 13.47] |
| region = south | 480 | 13.86 [12.90, 14.79] | 13.11 [12.17, 14.03] |
| region = unknown | 2 | 125.00 [75.00, 175.00] | 125.00 [75.00, 175.00] |
| region = west | 1294 | 19.86 [19.12, 20.59] | 19.03 [18.33, 19.77] |
| gender = female | 2352 | 12.95 [12.44, 13.44] | 12.38 [11.89, 12.84] |
| gender = male | 3055 | 16.96 [16.46, 17.46] | 16.19 [15.70, 16.64] |
| gender = unknown | 2 | 125.00 [75.00, 175.00] | 125.00 [75.00, 175.00] |
| duration_bucket = 0-3s | 1179 | 17.42 [16.04, 18.98] | 16.43 [15.03, 17.86] |
| duration_bucket = 10s+ | 929 | 14.62 [14.01, 15.21] | 14.11 [13.54, 14.69] |
| duration_bucket = 3-6s | 1607 | 15.60 [14.85, 16.33] | 14.76 [13.97, 15.47] |
| duration_bucket = 6-10s | 1694 | 15.17 [14.68, 15.68] | 14.47 [13.97, 14.99] |
| seen_speaker = False | 3037 | 16.95 [16.46, 17.43] | 16.21 [15.75, 16.68] |
| seen_speaker = True | 2372 | 13.11 [12.68, 13.56] | 12.49 [12.06, 12.96] |
| has_number = False | 4775 | 15.07 [14.67, 15.43] | 14.38 [14.00, 14.75] |
| has_number = True | 634 | 16.19 [15.23, 17.22] | 15.53 [14.58, 16.50] |
| has_name = False | 3734 | 14.80 [14.35, 15.22] | 14.13 [13.75, 14.54] |
| has_name = True | 1675 | 16.04 [15.42, 16.64] | 15.32 [14.76, 15.89] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | -0.69 | [-0.88, -0.49] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc step-003396 (480 ms): `reports/runs/20261020-232609_finetune-realtime-enc-step3396`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms): `reports/runs/20261021-102102_finetune-realtime-enc-5ep-step5660`
