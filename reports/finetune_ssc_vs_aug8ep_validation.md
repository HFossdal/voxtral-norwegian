# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | 12.17 [11.87, 12.47] | 12.08 [11.75, 12.42] | 4.08 [3.96, 4.19] | 81.9% | 93.6% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.25 [8.98, 9.51] | 9.06 [8.77, 9.36] | 3.14 [3.03, 3.24] | 84.6% | 91.4% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 10.25 [9.84, 10.64] | 7.72 [7.38, 8.08] |
| region = mid | 521 | 11.98 [10.95, 12.99] | 9.35 [8.49, 10.27] |
| region = north | 412 | 10.14 [9.17, 11.10] | 7.94 [7.11, 8.73] |
| region = south | 480 | 11.30 [10.42, 12.14] | 8.03 [7.21, 8.85] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 16.53 [15.83, 17.21] | 12.73 [12.11, 13.34] |
| gender = female | 2352 | 10.00 [9.59, 10.41] | 7.59 [7.23, 7.95] |
| gender = male | 3055 | 13.83 [13.35, 14.28] | 10.52 [10.13, 10.92] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 13.62 [12.42, 14.95] | 12.53 [11.33, 13.75] |
| duration_bucket = 10s+ | 929 | 11.70 [11.17, 12.24] | 8.79 [8.32, 9.27] |
| duration_bucket = 3-6s | 1607 | 12.40 [11.71, 13.11] | 9.35 [8.77, 9.98] |
| duration_bucket = 6-10s | 1694 | 12.20 [11.70, 12.69] | 9.06 [8.65, 9.51] |
| seen_speaker = False | 3037 | 13.86 [13.40, 14.31] | 10.37 [9.98, 10.78] |
| seen_speaker = True | 2372 | 10.10 [9.70, 10.51] | 7.89 [7.56, 8.26] |
| has_number = False | 4775 | 12.08 [11.75, 12.42] | 9.06 [8.77, 9.36] |
| has_number = True | 634 | 12.72 [11.88, 13.58] | 10.41 [9.73, 11.19] |
| has_name = False | 3734 | 11.82 [11.41, 12.21] | 8.88 [8.54, 9.21] |
| has_name = True | 1675 | 12.84 [12.32, 13.39] | 9.96 [9.46, 10.46] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | -2.92 | [-3.14, -2.69] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms): `reports/runs/20261022-124210_finetune-realtime-enc-8ep-aug-step9056`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `reports/runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
