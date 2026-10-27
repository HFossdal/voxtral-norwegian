# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nb-whisper-large-verbatim (language: no) | 5.16 [4.94, 5.36] | 4.02 [3.83, 4.22] | 2.82 [2.66, 3.00] | 93.0% | 34.2% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | 6.17 [5.94, 6.39] | 6.15 [5.92, 6.40] | 2.22 [2.13, 2.32] | 87.4% | 94.6% | 5409 | 52 |

## WER by slice

| slice | utts | nb-whisper-large-verbatim (language: no) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2700 | 4.69 [4.37, 5.03] | 5.32 [5.03, 5.62] |
| region = mid | 521 | 5.95 [5.23, 6.80] | 7.05 [6.23, 7.90] |
| region = north | 412 | 4.33 [3.69, 4.98] | 5.06 [4.43, 5.69] |
| region = south | 480 | 4.28 [3.73, 4.81] | 5.40 [4.80, 5.99] |
| region = unknown | 2 | 25.00 [0.00, 50.00] | 112.50 [50.00, 175.00] |
| region = west | 1294 | 6.30 [5.88, 6.78] | 7.96 [7.49, 8.47] |
| gender = female | 2352 | 4.39 [4.11, 4.70] | 5.27 [4.97, 5.58] |
| gender = male | 3055 | 5.74 [5.45, 6.07] | 6.83 [6.54, 7.13] |
| gender = unknown | 2 | 25.00 [0.00, 50.00] | 112.50 [50.00, 175.00] |
| duration_bucket = 0-3s | 1179 | 6.52 [5.64, 7.43] | 9.40 [8.38, 10.43] |
| duration_bucket = 10s+ | 929 | 5.00 [4.62, 5.36] | 5.82 [5.43, 6.18] |
| duration_bucket = 3-6s | 1607 | 5.28 [4.80, 5.79] | 6.28 [5.80, 6.81] |
| duration_bucket = 6-10s | 1694 | 4.99 [4.65, 5.37] | 5.86 [5.51, 6.19] |
| seen_speaker = False | 3037 | 5.52 [5.21, 5.82] | 6.75 [6.43, 7.08] |
| seen_speaker = True | 2372 | 4.71 [4.40, 5.02] | 5.44 [5.17, 5.74] |
| has_number = False | 4775 | 4.02 [3.83, 4.22] | 6.15 [5.92, 6.40] |
| has_number = True | 634 | 11.86 [10.98, 12.87] | 6.25 [5.72, 6.95] |
| has_name = False | 3734 | 4.96 [4.69, 5.22] | 5.82 [5.56, 6.09] |
| has_name = True | 1675 | 5.53 [5.17, 5.91] | 6.81 [6.41, 7.22] |

## Paired comparison against nb-whisper-large-verbatim (language: no)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | +1.01 | [+0.76, +1.26] | 1.000 |

## Run directories

- nb-whisper-large-verbatim (language: no): `reports/runs/20261022-153138_reference-nb-whisper-large-verbatim`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb): `reports/runs/20261025-042251_finetune-mini-enc-aug-ssc-step8000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | nb-whisper-verbatim | Mini | Mini minus nb-whisper-verbatim | resamples where Mini is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5409 | 5.16 [4.94, 5.36] | 6.17 [5.94, 6.39] | +1.01 [+0.77, +1.27] | 1.000 |
| number-free utterances | 4775 | 4.02 [3.83, 4.22] | 6.15 [5.92, 6.40] | +2.13 [+1.92, +2.35] | 1.000 |
| number-free, spoken/standard forms folded | 4775 | 3.88 [3.70, 4.08] | 5.98 [5.75, 6.23] | +2.10 [+1.90, +2.31] | 1.000 |

Folding changes 8.4% of the reference words in the number-free utterances.

