# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | 6.17 [5.94, 6.39] | 6.15 [5.92, 6.40] | 2.22 [2.13, 2.32] | 87.4% | 94.6% | 5409 | 52 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.84 [5.61, 6.05] | 5.83 [5.61, 6.10] | 2.09 [2.00, 2.18] | 87.7% | 95.5% | 5409 | 52 |

## WER by slice

| slice | utts | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2700 | 5.32 [5.03, 5.62] | 4.97 [4.70, 5.25] |
| region = mid | 521 | 7.05 [6.23, 7.90] | 6.67 [5.88, 7.50] |
| region = north | 412 | 5.06 [4.43, 5.69] | 4.52 [3.96, 5.05] |
| region = south | 480 | 5.40 [4.80, 5.99] | 4.95 [4.38, 5.54] |
| region = unknown | 2 | 112.50 [50.00, 175.00] | 87.50 [75.00, 100.00] |
| region = west | 1294 | 7.96 [7.49, 8.47] | 7.82 [7.35, 8.29] |
| gender = female | 2352 | 5.27 [4.97, 5.58] | 4.83 [4.55, 5.12] |
| gender = male | 3055 | 6.83 [6.54, 7.13] | 6.60 [6.30, 6.90] |
| gender = unknown | 2 | 112.50 [50.00, 175.00] | 87.50 [75.00, 100.00] |
| duration_bucket = 0-3s | 1179 | 9.40 [8.38, 10.43] | 9.03 [7.98, 10.09] |
| duration_bucket = 10s+ | 929 | 5.82 [5.43, 6.18] | 5.41 [5.00, 5.79] |
| duration_bucket = 3-6s | 1607 | 6.28 [5.80, 6.81] | 5.87 [5.40, 6.36] |
| duration_bucket = 6-10s | 1694 | 5.86 [5.51, 6.19] | 5.66 [5.30, 5.99] |
| seen_speaker = False | 3037 | 6.75 [6.43, 7.08] | 6.58 [6.27, 6.89] |
| seen_speaker = True | 2372 | 5.44 [5.17, 5.74] | 4.92 [4.67, 5.19] |
| has_number = False | 4775 | 6.15 [5.92, 6.40] | 5.83 [5.61, 6.10] |
| has_number = True | 634 | 6.25 [5.72, 6.95] | 5.86 [5.35, 6.50] |
| has_name = False | 3734 | 5.82 [5.56, 6.09] | 5.49 [5.24, 5.76] |
| has_name = True | 1675 | 6.81 [6.41, 7.22] | 6.49 [6.09, 6.90] |

## Paired comparison against Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -0.33 | [-0.47, -0.17] | 0.000 |

## Run directories

- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc step-008000 (language: nb): `reports/runs/20261025-042251_finetune-mini-enc-aug-ssc-step8000`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | lr 1e-4 | lr 2e-4 | lr 2e-4 minus lr 1e-4 | resamples where lr 2e-4 is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5409 | 6.17 [5.94, 6.39] | 5.84 [5.61, 6.05] | -0.33 [-0.49, -0.18] | 0.000 |
| number-free utterances | 4775 | 6.15 [5.92, 6.40] | 5.83 [5.61, 6.10] | -0.31 [-0.49, -0.15] | 0.000 |
| number-free, spoken/standard forms folded | 4775 | 5.98 [5.75, 6.23] | 5.69 [5.47, 5.95] | -0.29 [-0.46, -0.14] | 0.000 |

Folding changes 8.4% of the reference words in the number-free utterances.

