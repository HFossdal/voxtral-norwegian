# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.25 [8.98, 9.51] | 9.06 [8.77, 9.36] | 3.14 [3.03, 3.24] | 84.6% | 91.4% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | 6.17 [5.94, 6.39] | 6.15 [5.92, 6.40] | 2.22 [2.13, 2.32] | 87.4% | 94.6% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2700 | 7.72 [7.38, 8.08] | 5.32 [5.03, 5.62] |
| region = mid | 521 | 9.35 [8.49, 10.27] | 7.05 [6.23, 7.90] |
| region = north | 412 | 7.94 [7.11, 8.73] | 5.06 [4.43, 5.69] |
| region = south | 480 | 8.03 [7.21, 8.85] | 5.40 [4.80, 5.99] |
| region = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [50.00, 175.00] |
| region = west | 1294 | 12.73 [12.11, 13.34] | 7.96 [7.49, 8.47] |
| gender = female | 2352 | 7.59 [7.23, 7.95] | 5.27 [4.97, 5.58] |
| gender = male | 3055 | 10.52 [10.13, 10.92] | 6.83 [6.54, 7.13] |
| gender = unknown | 2 | 100.00 [75.00, 125.00] | 112.50 [50.00, 175.00] |
| duration_bucket = 0-3s | 1179 | 12.53 [11.33, 13.75] | 9.40 [8.38, 10.43] |
| duration_bucket = 10s+ | 929 | 8.79 [8.32, 9.27] | 5.82 [5.43, 6.18] |
| duration_bucket = 3-6s | 1607 | 9.35 [8.77, 9.98] | 6.28 [5.80, 6.81] |
| duration_bucket = 6-10s | 1694 | 9.06 [8.65, 9.51] | 5.86 [5.51, 6.19] |
| seen_speaker = False | 3037 | 10.37 [9.98, 10.78] | 6.75 [6.43, 7.08] |
| seen_speaker = True | 2372 | 7.89 [7.56, 8.26] | 5.44 [5.17, 5.74] |
| has_number = False | 4775 | 9.06 [8.77, 9.36] | 6.15 [5.92, 6.40] |
| has_number = True | 634 | 10.41 [9.73, 11.19] | 6.25 [5.72, 6.95] |
| has_name = False | 3734 | 8.88 [8.54, 9.21] | 5.82 [5.56, 6.09] |
| has_name = True | 1675 | 9.96 [9.46, 10.46] | 6.81 [6.41, 7.22] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | -3.09 | [-3.31, -2.88] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `reports/runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb): `reports/runs/20261025-042251_finetune-mini-enc-aug-ssc-step8000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | Realtime | Mini | Mini minus Realtime | resamples where Mini is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5409 | 9.25 [8.98, 9.51] | 6.17 [5.94, 6.39] | -3.09 [-3.33, -2.87] | 0.000 |
| number-free utterances | 4775 | 9.06 [8.77, 9.36] | 6.15 [5.92, 6.40] | -2.91 [-3.15, -2.66] | 0.000 |
| number-free, spoken/standard forms folded | 4775 | 8.89 [8.61, 9.18] | 5.98 [5.75, 6.23] | -2.91 [-3.15, -2.65] | 0.000 |

Folding changes 8.4% of the reference words in the number-free utterances.

