# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nb-whisper-large-verbatim (language: no) | 5.02 [4.79, 5.23] | 3.83 [3.63, 4.02] | 2.81 [2.64, 2.97] | 91.9% | 34.2% | 5006 | 38 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |

## WER by slice

| slice | utts | nb-whisper-large-verbatim (language: no) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2541 | 4.53 [4.24, 4.82] | 5.25 [4.96, 5.57] |
| region = mid | 353 | 4.92 [4.27, 5.67] | 5.27 [4.62, 5.94] |
| region = north | 554 | 6.49 [5.79, 7.22] | 6.60 [5.99, 7.27] |
| region = south | 272 | 4.31 [3.56, 5.08] | 5.04 [4.28, 5.84] |
| region = west | 1286 | 5.37 [4.97, 5.78] | 7.05 [6.62, 7.49] |
| gender = female | 1941 | 4.71 [4.38, 5.08] | 5.38 [5.05, 5.71] |
| gender = male | 3065 | 5.20 [4.94, 5.46] | 6.23 [5.93, 6.51] |
| duration_bucket = 0-3s | 916 | 5.36 [4.44, 6.30] | 8.97 [7.89, 10.06] |
| duration_bucket = 10s+ | 909 | 5.14 [4.76, 5.57] | 5.57 [5.19, 5.94] |
| duration_bucket = 3-6s | 1549 | 5.08 [4.65, 5.52] | 6.22 [5.74, 6.71] |
| duration_bucket = 6-10s | 1632 | 4.84 [4.53, 5.16] | 5.63 [5.27, 5.98] |
| seen_speaker = False | 5006 | 5.02 [4.79, 5.23] | 5.91 [5.67, 6.14] |
| has_number = False | 4363 | 3.83 [3.63, 4.02] | 5.86 [5.63, 6.12] |
| has_number = True | 643 | 11.24 [10.53, 11.98] | 6.19 [5.65, 6.73] |
| has_name = False | 3467 | 4.66 [4.40, 4.92] | 5.37 [5.13, 5.60] |
| has_name = True | 1539 | 5.67 [5.29, 6.07] | 6.92 [6.51, 7.32] |

## Paired comparison against nb-whisper-large-verbatim (language: no)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | +0.90 | [+0.66, +1.13] | 1.000 |

## Run directories

- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | nb-whisper-large-verbatim | Mini lr 2e-4 | Mini lr 2e-4 minus nb-whisper-large-verbatim | resamples where Mini lr 2e-4 is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5006 | 5.02 [4.79, 5.23] | 5.91 [5.67, 6.14] | +0.90 [+0.68, +1.13] | 1.000 |
| number-free utterances | 4363 | 3.83 [3.63, 4.02] | 5.86 [5.63, 6.12] | +2.03 [+1.83, +2.25] | 1.000 |
| number-free, spoken/standard forms folded | 4363 | 3.72 [3.52, 3.91] | 5.72 [5.49, 5.99] | +2.00 [+1.79, +2.22] | 1.000 |

Folding changes 8.1% of the reference words in the number-free utterances.

