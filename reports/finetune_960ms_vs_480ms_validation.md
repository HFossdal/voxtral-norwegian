# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | 12.17 [11.87, 12.47] | 12.08 [11.75, 12.42] | 4.08 [3.96, 4.19] | 81.9% | 93.6% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) | 11.71 [11.42, 12.01] | 11.68 [11.34, 12.01] | 3.92 [3.81, 4.04] | 82.7% | 93.7% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) |
| --- | --- | --- | --- |
| region = east | 2700 | 10.25 [9.84, 10.64] | 9.83 [9.41, 10.21] |
| region = mid | 521 | 11.98 [10.95, 12.99] | 12.02 [11.02, 13.05] |
| region = north | 412 | 10.14 [9.17, 11.10] | 9.67 [8.69, 10.56] |
| region = south | 480 | 11.30 [10.42, 12.14] | 10.59 [9.67, 11.50] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| region = west | 1294 | 16.53 [15.83, 17.21] | 15.94 [15.29, 16.60] |
| gender = female | 2352 | 10.00 [9.59, 10.41] | 9.67 [9.22, 10.09] |
| gender = male | 3055 | 13.83 [13.35, 14.28] | 13.26 [12.81, 13.69] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 100.00 [75.00, 125.00] |
| duration_bucket = 0-3s | 1179 | 13.62 [12.42, 14.95] | 13.29 [12.04, 14.56] |
| duration_bucket = 10s+ | 929 | 11.70 [11.17, 12.24] | 11.17 [10.65, 11.71] |
| duration_bucket = 3-6s | 1607 | 12.40 [11.71, 13.11] | 12.08 [11.41, 12.73] |
| duration_bucket = 6-10s | 1694 | 12.20 [11.70, 12.69] | 11.70 [11.22, 12.15] |
| seen_speaker = False | 3037 | 13.86 [13.40, 14.31] | 13.27 [12.81, 13.70] |
| seen_speaker = True | 2372 | 10.10 [9.70, 10.51] | 9.80 [9.40, 10.21] |
| has_number = False | 4775 | 12.08 [11.75, 12.42] | 11.68 [11.34, 12.01] |
| has_number = True | 634 | 12.72 [11.88, 13.58] | 11.91 [11.11, 12.84] |
| has_name = False | 3734 | 11.82 [11.41, 12.21] | 11.39 [11.01, 11.77] |
| has_name = True | 1675 | 12.84 [12.32, 13.39] | 12.33 [11.78, 12.85] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms) | -0.46 | [-0.68, -0.25] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms): `reports/runs/20261022-124210_finetune-realtime-enc-8ep-aug-step9056`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug-960ms step-008000 (960 ms): `reports/runs/20261023-045156_finetune-realtime-enc-8ep-aug-960ms-step8000`
