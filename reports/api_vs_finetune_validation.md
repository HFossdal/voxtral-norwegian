# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| voxtral-mini-transcribe-3 (API, language: no) | 10.91 [10.59, 11.24] | 9.73 [9.43, 10.03] | 5.67 [5.45, 5.87] | 86.3% | 20.6% | 5409 | 52 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.25 [8.98, 9.51] | 9.06 [8.77, 9.36] | 3.14 [3.03, 3.24] | 84.6% | 91.4% | 5409 | 52 |
| nb-whisper-large-verbatim (language: no) | 5.16 [4.94, 5.36] | 4.02 [3.83, 4.22] | 2.82 [2.66, 3.00] | 93.0% | 34.2% | 5409 | 52 |

## WER by slice

| slice | utts | voxtral-mini-transcribe-3 (API, language: no) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | nb-whisper-large-verbatim (language: no) |
| --- | --- | --- | --- | --- |
| region = east | 2700 | 8.77 [8.33, 9.22] | 7.72 [7.38, 8.08] | 4.69 [4.37, 5.03] |
| region = mid | 521 | 15.18 [13.92, 16.48] | 9.35 [8.49, 10.27] | 5.95 [5.23, 6.80] |
| region = north | 412 | 12.54 [11.47, 13.51] | 7.94 [7.11, 8.73] | 4.33 [3.69, 4.98] |
| region = south | 480 | 8.80 [8.03, 9.56] | 8.03 [7.21, 8.85] | 4.28 [3.73, 4.81] |
| region = unknown | 2 | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] | 25.00 [0.00, 50.00] |
| region = west | 1294 | 13.50 [12.88, 14.12] | 12.73 [12.11, 13.34] | 6.30 [5.88, 6.78] |
| gender = female | 2352 | 9.64 [9.20, 10.10] | 7.59 [7.23, 7.95] | 4.39 [4.11, 4.70] |
| gender = male | 3055 | 11.86 [11.46, 12.29] | 10.52 [10.13, 10.92] | 5.74 [5.45, 6.07] |
| gender = unknown | 2 | 112.50 [75.00, 150.00] | 100.00 [75.00, 125.00] | 25.00 [0.00, 50.00] |
| duration_bucket = 0-3s | 1179 | 21.22 [19.42, 23.06] | 12.53 [11.33, 13.75] | 6.52 [5.64, 7.43] |
| duration_bucket = 10s+ | 929 | 10.24 [9.72, 10.83] | 8.79 [8.32, 9.27] | 5.00 [4.62, 5.36] |
| duration_bucket = 3-6s | 1607 | 11.00 [10.36, 11.71] | 9.35 [8.77, 9.98] | 5.28 [4.80, 5.79] |
| duration_bucket = 6-10s | 1694 | 9.72 [9.25, 10.21] | 9.06 [8.65, 9.51] | 4.99 [4.65, 5.37] |
| seen_speaker = False | 3037 | 11.47 [11.05, 11.90] | 10.37 [9.98, 10.78] | 5.52 [5.21, 5.82] |
| seen_speaker = True | 2372 | 10.21 [9.80, 10.69] | 7.89 [7.56, 8.26] | 4.71 [4.40, 5.02] |
| has_number = False | 4775 | 9.73 [9.43, 10.03] | 9.06 [8.77, 9.36] | 4.02 [3.83, 4.22] |
| has_number = True | 634 | 17.85 [16.82, 18.94] | 10.41 [9.73, 11.19] | 11.86 [10.98, 12.87] |
| has_name = False | 3734 | 10.62 [10.24, 11.00] | 8.88 [8.54, 9.21] | 4.96 [4.69, 5.22] |
| has_name = True | 1675 | 11.44 [10.94, 11.93] | 9.96 [9.46, 10.46] | 5.53 [5.17, 5.91] |

## Paired comparison against voxtral-mini-transcribe-3 (API, language: no)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | -1.65 | [-1.97, -1.34] | 0.000 |
| nb-whisper-large-verbatim (language: no) | -5.75 | [-6.01, -5.49] | 0.000 |

## Run directories

- voxtral-mini-transcribe-3 (API, language: no): `reports/runs/20261024-112447_reference-mistral-api-no`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `reports/runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
