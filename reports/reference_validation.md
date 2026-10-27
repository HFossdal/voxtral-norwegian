# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | 12.17 [11.87, 12.47] | 12.08 [11.75, 12.42] | 4.08 [3.96, 4.19] | 81.9% | 93.6% | 5409 | 52 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.82 [21.27, 22.46] | 20.44 [20.02, 20.89] | 10.23 [9.84, 10.73] | 61.0% | 17.3% | 5409 | 52 |
| nb-whisper-large (language: no) | 14.16 [13.62, 14.66] | 12.69 [12.15, 13.20] | 9.96 [9.51, 10.37] | 88.4% | 17.9% | 5409 | 52 |
| nb-whisper-large-verbatim (language: no) | 5.16 [4.94, 5.36] | 4.02 [3.83, 4.22] | 2.82 [2.66, 3.00] | 93.0% | 34.2% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) | nb-whisper-large (language: no) | nb-whisper-large-verbatim (language: no) |
| --- | --- | --- | --- | --- | --- |
| region = east | 2700 | 10.25 [9.84, 10.64] | 19.86 [19.03, 20.87] | 12.56 [11.90, 13.27] | 4.69 [4.37, 5.03] |
| region = mid | 521 | 11.98 [10.95, 12.99] | 27.98 [26.07, 29.92] | 16.08 [14.30, 17.96] | 5.95 [5.23, 6.80] |
| region = north | 412 | 10.14 [9.17, 11.10] | 21.48 [20.31, 22.69] | 13.73 [12.30, 15.35] | 4.33 [3.69, 4.98] |
| region = south | 480 | 11.30 [10.42, 12.14] | 18.92 [17.89, 20.05] | 14.19 [12.55, 15.91] | 4.28 [3.73, 4.81] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] | 62.50 [50.00, 75.00] | 25.00 [0.00, 50.00] |
| region = west | 1294 | 16.53 [15.83, 17.21] | 24.48 [23.68, 25.32] | 16.45 [15.53, 17.42] | 6.30 [5.88, 6.78] |
| gender = female | 2352 | 10.00 [9.59, 10.41] | 20.50 [19.57, 21.68] | 12.58 [11.87, 13.22] | 4.39 [4.11, 4.70] |
| gender = male | 3055 | 13.83 [13.35, 14.28] | 22.81 [22.22, 23.37] | 15.36 [14.68, 16.00] | 5.74 [5.45, 6.07] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [75.00, 150.00] | 62.50 [50.00, 75.00] | 25.00 [0.00, 50.00] |
| duration_bucket = 0-3s | 1179 | 13.62 [12.42, 14.95] | 35.62 [33.57, 37.94] | 12.00 [10.73, 13.28] | 6.52 [5.64, 7.43] |
| duration_bucket = 10s+ | 929 | 11.70 [11.17, 12.24] | 21.38 [20.21, 22.84] | 13.77 [12.87, 14.71] | 5.00 [4.62, 5.36] |
| duration_bucket = 3-6s | 1607 | 12.40 [11.71, 13.11] | 21.45 [20.57, 22.31] | 11.52 [10.69, 12.28] | 5.28 [4.80, 5.79] |
| duration_bucket = 6-10s | 1694 | 12.20 [11.70, 12.69] | 20.14 [19.56, 20.82] | 16.31 [15.47, 17.20] | 4.99 [4.65, 5.37] |
| seen_speaker = False | 3037 | 13.86 [13.40, 14.31] | 22.31 [21.73, 22.97] | 14.69 [14.04, 15.33] | 5.52 [5.21, 5.82] |
| seen_speaker = True | 2372 | 10.10 [9.70, 10.51] | 21.21 [20.35, 22.28] | 13.50 [12.81, 14.26] | 4.71 [4.40, 5.02] |
| has_number = False | 4775 | 12.08 [11.75, 12.42] | 20.44 [20.02, 20.89] | 12.69 [12.15, 13.20] | 4.02 [3.83, 4.22] |
| has_number = True | 634 | 12.72 [11.88, 13.58] | 29.99 [27.74, 33.04] | 22.84 [21.45, 24.32] | 11.86 [10.98, 12.87] |
| has_name = False | 3734 | 11.82 [11.41, 12.21] | 20.97 [20.29, 21.86] | 14.22 [13.57, 14.85] | 4.96 [4.69, 5.22] |
| has_name = True | 1675 | 12.84 [12.32, 13.39] | 23.40 [22.65, 24.12] | 14.04 [13.26, 14.87] | 5.53 [5.17, 5.91] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | +9.64 | [+9.10, +10.28] | 1.000 |
| nb-whisper-large (language: no) | +1.98 | [+1.48, +2.48] | 1.000 |
| nb-whisper-large-verbatim (language: no) | -7.02 | [-7.33, -6.71] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-8ep-aug step-009056 (480 ms): `reports/runs/20261022-124210_finetune-realtime-enc-8ep-aug-step9056`
- Voxtral-Mini-3B-2507 (language: nb): `reports/runs/20261019-222846_baseline-mini-nb`
- nb-whisper-large (language: no): `reports/runs/20261022-152401_reference-nb-whisper-large`
- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
