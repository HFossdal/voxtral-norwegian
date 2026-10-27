# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.81 [9.51, 10.13] | 9.65 [9.31, 9.96] | 3.38 [3.26, 3.50] | 77.8% | 92.2% | 5006 | 38 |
| voxtral-mini-transcribe-3 (API, language: no) | 10.67 [10.36, 10.99] | 9.54 [9.23, 9.87] | 5.71 [5.50, 5.92] | 83.7% | 25.3% | 5006 | 38 |
| nb-whisper-large (language: no) | 14.40 [13.89, 14.89] | 13.05 [12.54, 13.61] | 10.49 [10.05, 10.93] | 85.7% | 22.3% | 5006 | 38 |
| nb-whisper-large-verbatim (language: no) | 5.02 [4.79, 5.23] | 3.83 [3.63, 4.02] | 2.81 [2.64, 2.97] | 91.9% | 34.2% | 5006 | 38 |

## WER by slice

| slice | utts | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | voxtral-mini-transcribe-3 (API, language: no) | nb-whisper-large (language: no) | nb-whisper-large-verbatim (language: no) |
| --- | --- | --- | --- | --- | --- | --- |
| region = east | 2541 | 5.25 [4.96, 5.57] | 8.42 [8.02, 8.82] | 9.79 [9.33, 10.28] | 12.85 [12.21, 13.55] | 4.53 [4.24, 4.82] |
| region = mid | 353 | 5.27 [4.62, 5.94] | 8.54 [7.63, 9.43] | 12.27 [11.26, 13.40] | 16.73 [14.75, 18.57] | 4.92 [4.27, 5.67] |
| region = north | 554 | 6.60 [5.99, 7.27] | 10.43 [9.57, 11.24] | 13.57 [12.66, 14.46] | 16.61 [15.11, 18.08] | 6.49 [5.79, 7.22] |
| region = south | 272 | 5.04 [4.28, 5.84] | 9.34 [8.13, 10.65] | 9.98 [8.71, 11.24] | 12.86 [10.84, 14.94] | 4.31 [3.56, 5.08] |
| region = west | 1286 | 7.05 [6.62, 7.49] | 12.27 [11.61, 12.94] | 10.62 [10.04, 11.21] | 15.72 [14.87, 16.69] | 5.37 [4.97, 5.78] |
| gender = female | 1941 | 5.38 [5.05, 5.71] | 9.28 [8.85, 9.77] | 10.27 [9.77, 10.77] | 13.70 [12.92, 14.43] | 4.71 [4.38, 5.08] |
| gender = male | 3065 | 6.23 [5.93, 6.51] | 10.12 [9.74, 10.51] | 10.90 [10.52, 11.30] | 14.81 [14.20, 15.44] | 5.20 [4.94, 5.46] |
| duration_bucket = 0-3s | 916 | 8.97 [7.89, 10.06] | 12.98 [11.62, 14.42] | 21.23 [18.98, 23.54] | 10.93 [9.59, 12.37] | 5.36 [4.44, 6.30] |
| duration_bucket = 10s+ | 909 | 5.57 [5.19, 5.94] | 9.13 [8.61, 9.65] | 10.34 [9.81, 10.92] | 13.78 [12.96, 14.72] | 5.14 [4.76, 5.57] |
| duration_bucket = 3-6s | 1549 | 6.22 [5.74, 6.71] | 10.56 [9.89, 11.21] | 10.31 [9.71, 10.96] | 10.91 [10.14, 11.69] | 5.08 [4.65, 5.52] |
| duration_bucket = 6-10s | 1632 | 5.63 [5.27, 5.98] | 9.54 [9.06, 10.01] | 9.80 [9.32, 10.27] | 17.31 [16.51, 18.22] | 4.84 [4.53, 5.16] |
| seen_speaker = False | 5006 | 5.91 [5.67, 6.14] | 9.81 [9.51, 10.13] | 10.67 [10.36, 10.99] | 14.40 [13.89, 14.89] | 5.02 [4.79, 5.23] |
| has_number = False | 4363 | 5.86 [5.63, 6.12] | 9.65 [9.31, 9.96] | 9.54 [9.23, 9.87] | 13.05 [12.54, 13.61] | 3.83 [3.63, 4.02] |
| has_number = True | 643 | 6.19 [5.65, 6.73] | 10.67 [9.92, 11.48] | 16.56 [15.68, 17.38] | 21.48 [20.17, 22.78] | 11.24 [10.53, 11.98] |
| has_name = False | 3467 | 5.37 [5.13, 5.60] | 9.09 [8.75, 9.46] | 10.18 [9.79, 10.57] | 13.88 [13.24, 14.48] | 4.66 [4.40, 4.92] |
| has_name = True | 1539 | 6.92 [6.51, 7.32] | 11.14 [10.57, 11.74] | 11.57 [11.03, 12.11] | 15.37 [14.54, 16.26] | 5.67 [5.29, 6.07] |

## Paired comparison against Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | +3.90 | [+3.67, +4.14] | 1.000 |
| voxtral-mini-transcribe-3 (API, language: no) | +4.76 | [+4.47, +5.06] | 1.000 |
| nb-whisper-large (language: no) | +8.49 | [+8.02, +8.95] | 1.000 |
| nb-whisper-large-verbatim (language: no) | -0.90 | [-1.13, -0.66] | 0.000 |

## Run directories

- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`
- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `reports/runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
- voxtral-mini-transcribe-3 (API, language: no): `reports/runs/20261024-112447_reference-mistral-api-no`
- nb-whisper-large (language: no): `reports/runs/20261022-152401_reference-nb-whisper-large`
- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
