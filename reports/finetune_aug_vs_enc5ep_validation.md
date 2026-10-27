# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | 14.54 [14.21, 14.88] | 14.38 [14.00, 14.75] | 4.87 [4.73, 5.00] | 77.5% | 89.5% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | 12.77 [12.46, 13.07] | 12.68 [12.31, 13.03] | 4.23 [4.11, 4.36] | 80.2% | 91.8% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 12.64 [12.20, 13.11] | 10.81 [10.40, 11.23] |
| region = mid | 521 | 14.50 [13.41, 15.59] | 12.53 [11.58, 13.51] |
| region = north | 412 | 12.43 [11.33, 13.47] | 10.62 [9.60, 11.64] |
| region = south | 480 | 13.11 [12.17, 14.03] | 11.64 [10.76, 12.52] |
| region = unknown | 2 | 125.00 [75.00, 175.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 19.03 [18.33, 19.77] | 17.32 [16.66, 17.98] |
| gender = female | 2352 | 12.38 [11.89, 12.84] | 10.58 [10.12, 10.99] |
| gender = male | 3055 | 16.19 [15.70, 16.64] | 14.44 [13.99, 14.89] |
| gender = unknown | 2 | 125.00 [75.00, 175.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 16.43 [15.03, 17.86] | 14.05 [12.83, 15.38] |
| duration_bucket = 10s+ | 929 | 14.11 [13.54, 14.69] | 12.21 [11.69, 12.81] |
| duration_bucket = 3-6s | 1607 | 14.76 [13.97, 15.47] | 13.08 [12.41, 13.76] |
| duration_bucket = 6-10s | 1694 | 14.47 [13.97, 14.99] | 12.85 [12.37, 13.39] |
| seen_speaker = False | 3037 | 16.21 [15.75, 16.68] | 14.37 [13.93, 14.84] |
| seen_speaker = True | 2372 | 12.49 [12.06, 12.96] | 10.80 [10.36, 11.23] |
| has_number = False | 4775 | 14.38 [14.00, 14.75] | 12.68 [12.31, 13.03] |
| has_number = True | 634 | 15.53 [14.58, 16.50] | 13.32 [12.43, 14.29] |
| has_name = False | 3734 | 14.13 [13.75, 14.54] | 12.33 [11.90, 12.72] |
| has_name = True | 1675 | 15.32 [14.76, 15.89] | 13.60 [13.08, 14.20] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms) | -1.77 | [-2.01, -1.51] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep step-005660 (480 ms): `reports/runs/20261021-102102_finetune-realtime-enc-5ep-step5660`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-5ep-aug step-005660 (480 ms): `reports/runs/20261021-205210_finetune-realtime-enc-5ep-aug-step5660`
