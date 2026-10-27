# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | 9.81 [9.51, 10.13] | 9.65 [9.31, 9.96] | 3.38 [3.26, 3.50] | 77.8% | 92.2% | 5006 | 38 |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | 5.91 [5.67, 6.14] | 5.86 [5.63, 6.12] | 2.15 [2.05, 2.25] | 85.1% | 96.3% | 5006 | 38 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms) | Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2541 | 8.42 [8.02, 8.82] | 5.25 [4.96, 5.57] |
| region = mid | 353 | 8.54 [7.63, 9.43] | 5.27 [4.62, 5.94] |
| region = north | 554 | 10.43 [9.57, 11.24] | 6.60 [5.99, 7.27] |
| region = south | 272 | 9.34 [8.13, 10.65] | 5.04 [4.28, 5.84] |
| region = west | 1286 | 12.27 [11.61, 12.94] | 7.05 [6.62, 7.49] |
| gender = female | 1941 | 9.28 [8.85, 9.77] | 5.38 [5.05, 5.71] |
| gender = male | 3065 | 10.12 [9.74, 10.51] | 6.23 [5.93, 6.51] |
| duration_bucket = 0-3s | 916 | 12.98 [11.62, 14.42] | 8.97 [7.89, 10.06] |
| duration_bucket = 10s+ | 909 | 9.13 [8.61, 9.65] | 5.57 [5.19, 5.94] |
| duration_bucket = 3-6s | 1549 | 10.56 [9.89, 11.21] | 6.22 [5.74, 6.71] |
| duration_bucket = 6-10s | 1632 | 9.54 [9.06, 10.01] | 5.63 [5.27, 5.98] |
| seen_speaker = False | 5006 | 9.81 [9.51, 10.13] | 5.91 [5.67, 6.14] |
| has_number = False | 4363 | 9.65 [9.31, 9.96] | 5.86 [5.63, 6.12] |
| has_number = True | 643 | 10.67 [9.92, 11.48] | 6.19 [5.65, 6.73] |
| has_name = False | 3467 | 9.09 [8.75, 9.46] | 5.37 [5.13, 5.60] |
| has_name = True | 1539 | 11.14 [10.57, 11.74] | 6.92 [6.51, 7.32] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb) | -3.90 | [-4.14, -3.67] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime-enc-aug-ssc step-012000 (480 ms): `reports/runs/20261024-082207_finetune-realtime-enc-aug-ssc-step12000`
- Voxtral-Mini-3B-2507 + LoRA finetune-mini-enc-aug-ssc-lr2e-4 step-012000 (language: nb): `reports/runs/20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000`

## With the spelling convention removed

Numbers: only utterances without number words in the reference. Spoken forms: `voxnor.variants` folds spoken and standard Bokmål forms together on both sides (feminine and past-tense `-a`, `sjøl`/`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in the NPSC transcripts). Danish spellings are not folded.

| scoring | utts | Realtime | Mini lr 2e-4 | Mini lr 2e-4 minus Realtime | resamples where Mini lr 2e-4 is not better |
| --- | --- | --- | --- | --- | --- |
| all utterances, official scoring | 5006 | 9.81 [9.51, 10.13] | 5.91 [5.67, 6.14] | -3.90 [-4.14, -3.66] | 0.000 |
| number-free utterances | 4363 | 9.65 [9.31, 9.96] | 5.86 [5.63, 6.12] | -3.78 [-4.04, -3.53] | 0.000 |
| number-free, spoken/standard forms folded | 4363 | 9.49 [9.16, 9.80] | 5.72 [5.49, 5.99] | -3.76 [-4.02, -3.51] | 0.000 |

Folding changes 8.1% of the reference words in the number-free utterances.

