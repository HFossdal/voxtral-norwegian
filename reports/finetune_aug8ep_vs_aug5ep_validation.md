# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | 12.77 [12.46, 13.07] | 12.68 [12.31, 13.03] | 4.23 [4.11, 4.36] | 80.2% | 91.8% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | 12.17 [11.87, 12.47] | 12.08 [11.75, 12.42] | 4.08 [3.96, 4.19] | 81.9% | 93.6% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 10.81 [10.40, 11.23] | 10.25 [9.84, 10.64] |
| region = mid | 521 | 12.53 [11.58, 13.51] | 11.98 [10.95, 12.99] |
| region = north | 412 | 10.62 [9.60, 11.64] | 10.14 [9.17, 11.10] |
| region = south | 480 | 11.64 [10.76, 12.52] | 11.30 [10.42, 12.14] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 17.32 [16.66, 17.98] | 16.53 [15.83, 17.21] |
| gender = female | 2352 | 10.58 [10.12, 10.99] | 10.00 [9.59, 10.41] |
| gender = male | 3055 | 14.44 [13.99, 14.89] | 13.83 [13.35, 14.28] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 14.05 [12.83, 15.38] | 13.62 [12.42, 14.95] |
| duration_bucket = 10s+ | 929 | 12.21 [11.69, 12.81] | 11.70 [11.17, 12.24] |
| duration_bucket = 3-6s | 1607 | 13.08 [12.41, 13.76] | 12.40 [11.71, 13.11] |
| duration_bucket = 6-10s | 1694 | 12.85 [12.37, 13.39] | 12.20 [11.70, 12.69] |
| seen_speaker = False | 3037 | 14.37 [13.93, 14.84] | 13.86 [13.40, 14.31] |
| seen_speaker = True | 2372 | 10.80 [10.36, 11.23] | 10.10 [9.70, 10.51] |
| has_number = False | 4775 | 12.68 [12.31, 13.03] | 12.08 [11.75, 12.42] |
| has_number = True | 634 | 13.32 [12.43, 14.29] | 12.72 [11.88, 13.58] |
| has_name = False | 3734 | 12.33 [11.90, 12.72] | 11.82 [11.41, 12.21] |
| has_name = True | 1675 | 13.60 [13.08, 14.20] | 12.84 [12.32, 13.39] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | -0.60 | [-0.79, -0.41] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms): `reports/runs/20261021-205210_finetune-realtime-enc-5ep-aug-step5660`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms): `reports/runs/20261022-124210_finetune-realtime-enc-8ep-aug-step9056`
