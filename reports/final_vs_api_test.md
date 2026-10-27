# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| voxtral-mini-transcribe-3 (API, language: no) | 10.67 [10.36, 10.99] | 9.54 [9.23, 9.87] | 5.71 [5.50, 5.92] | 83.7% | 25.3% | 5006 | 38 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |

## WER by slice

| slice | utts | voxtral-mini-transcribe-3 (API, language: no) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2541 | 9.79 [9.33, 10.28] | 5.25 [4.96, 5.57] |
| region = mid | 353 | 12.27 [11.26, 13.40] | 5.27 [4.62, 5.94] |
| region = north | 554 | 13.57 [12.66, 14.46] | 6.60 [5.99, 7.27] |
| region = south | 272 | 9.98 [8.71, 11.24] | 5.04 [4.28, 5.84] |
| region = west | 1286 | 10.62 [10.04, 11.21] | 7.05 [6.62, 7.49] |
| gender = female | 1941 | 10.27 [9.77, 10.77] | 5.38 [5.05, 5.71] |
| gender = male | 3065 | 10.90 [10.52, 11.30] | 6.23 [5.93, 6.51] |
| duration_bucket = 0-3s | 916 | 21.23 [18.98, 23.54] | 8.97 [7.89, 10.06] |
| duration_bucket = 10s+ | 909 | 10.34 [9.81, 10.92] | 5.57 [5.19, 5.94] |
| duration_bucket = 3-6s | 1549 | 10.31 [9.71, 10.96] | 6.22 [5.74, 6.71] |
| duration_bucket = 6-10s | 1632 | 9.80 [9.32, 10.27] | 5.63 [5.27, 5.98] |
| seen_speaker = False | 5006 | 10.67 [10.36, 10.99] | 5.91 [5.67, 6.14] |
| has_number = False | 4363 | 9.54 [9.23, 9.87] | 5.86 [5.63, 6.12] |
| has_number = True | 643 | 16.56 [15.68, 17.38] | 6.19 [5.65, 6.73] |
| has_name = False | 3467 | 10.18 [9.79, 10.57] | 5.37 [5.13, 5.60] |
| has_name = True | 1539 | 11.57 [11.03, 12.11] | 6.92 [6.51, 7.32] |

## Paired comparison against voxtral-mini-transcribe-3 (API, language: no)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -4.76 | [-5.06, -4.47] | 0.000 |

## Run directories

- voxtral-mini-transcribe-3 (API, language: no): `reports/runs/20261024-112447_reference-mistral-api-no`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | Mistral API | Mini lr 2e-4 | Mini lr 2e-4 minus Mistral API | resamples where Mini lr 2e-4 is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5006 | 10.67 [10.36, 10.99] | 5.91 [5.67, 6.14] | -4.76 [-5.04, -4.47] | 0.000 |
| number-free utterances | 4363 | 9.54 [9.23, 9.87] | 5.86 [5.63, 6.12] | -3.68 [-3.97, -3.41] | 0.000 |
| number-free, spoken/standard forms folded | 4363 | 8.66 [8.35, 8.98] | 5.72 [5.49, 5.99] | -2.93 [-3.22, -2.66] | 0.000 |

Folding changes 8.1% of the reference words in the number-free utterances.

