# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nb-whisper-large-verbatim (language: no) | 5.16 [4.94, 5.36] | 4.02 [3.83, 4.22] | 2.82 [2.66, 3.00] | 93.0% | 34.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.84 [5.61, 6.05] | 5.83 [5.61, 6.10] | 2.09 [2.00, 2.18] | 87.7% | 95.5% | 5409 | 52 |

## WER by slice

| slice | utts | nb-whisper-large-verbatim (language: no) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2700 | 4.69 [4.37, 5.03] | 4.97 [4.70, 5.25] |
| region = mid | 521 | 5.95 [5.23, 6.80] | 6.67 [5.88, 7.50] |
| region = north | 412 | 4.33 [3.69, 4.98] | 4.52 [3.96, 5.05] |
| region = south | 480 | 4.28 [3.73, 4.81] | 4.95 [4.38, 5.54] |
| region = unknown | 2 | 25.00 [0.00, 50.00] | 87.50 [75.00, 100.00] |
| region = west | 1294 | 6.30 [5.88, 6.78] | 7.82 [7.35, 8.29] |
| gender = female | 2352 | 4.39 [4.11, 4.70] | 4.83 [4.55, 5.12] |
| gender = male | 3055 | 5.74 [5.45, 6.07] | 6.60 [6.30, 6.90] |
| gender = unknown | 2 | 25.00 [0.00, 50.00] | 87.50 [75.00, 100.00] |
| duration_bucket = 0-3s | 1179 | 6.52 [5.64, 7.43] | 9.03 [7.98, 10.09] |
| duration_bucket = 10s+ | 929 | 5.00 [4.62, 5.36] | 5.41 [5.00, 5.79] |
| duration_bucket = 3-6s | 1607 | 5.28 [4.80, 5.79] | 5.87 [5.40, 6.36] |
| duration_bucket = 6-10s | 1694 | 4.99 [4.65, 5.37] | 5.66 [5.30, 5.99] |
| seen_speaker = False | 3037 | 5.52 [5.21, 5.82] | 6.58 [6.27, 6.89] |
| seen_speaker = True | 2372 | 4.71 [4.40, 5.02] | 4.92 [4.67, 5.19] |
| has_number = False | 4775 | 4.02 [3.83, 4.22] | 5.83 [5.61, 6.10] |
| has_number = True | 634 | 11.86 [10.98, 12.87] | 5.86 [5.35, 6.50] |
| has_name = False | 3734 | 4.96 [4.69, 5.22] | 5.49 [5.24, 5.76] |
| has_name = True | 1675 | 5.53 [5.17, 5.91] | 6.49 [6.09, 6.90] |

## Paired comparison against nb-whisper-large-verbatim (language: no)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | +0.68 | [+0.44, +0.92] | 1.000 |

## Run directories

- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | nb-whisper-large-verbatim | Mini lr 2e-4 | Mini lr 2e-4 minus nb-whisper-large-verbatim | resamples where Mini lr 2e-4 is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5409 | 5.16 [4.94, 5.36] | 5.84 [5.61, 6.05] | +0.68 [+0.46, +0.93] | 1.000 |
| number-free utterances | 4775 | 4.02 [3.83, 4.22] | 5.83 [5.61, 6.10] | +1.81 [+1.62, +2.03] | 1.000 |
| number-free, spoken/standard forms folded | 4775 | 3.88 [3.70, 4.08] | 5.69 [5.47, 5.95] | +1.81 [+1.61, +2.02] | 1.000 |

Folding changes 8.4% of the reference words in the number-free utterances.

