# Results on test

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Voxtral-Mini-4B-Realtime-2602 (480 ms) | 94.97 [94.38, 95.59] | 95.03 [94.43, 95.66] | 60.18 [59.32, 61.05] | 9.4% | 3.7% | 5006 | 38 |
| Voxtral-Mini-3B-2507 (language: nb) | 21.28 [20.75, 21.82] | 20.21 [19.62, 20.91] | 10.11 [9.76, 10.47] | 59.9% | 24.0% | 5006 | 38 |

## WER by slice

| slice | utts | Voxtral-Mini-4B-Realtime-2602 (480 ms) | Voxtral-Mini-3B-2507 (language: nb) |
| --- | --- | --- | --- |
| region = east | 2541 | 94.65 [93.86, 95.54] | 20.56 [19.67, 21.58] |
| region = mid | 353 | 96.84 [94.39, 99.54] | 22.29 [20.89, 23.68] |
| region = north | 554 | 94.05 [92.43, 95.69] | 24.87 [23.47, 26.22] |
| region = south | 272 | 94.08 [92.04, 96.26] | 18.97 [17.31, 20.55] |
| region = west | 1286 | 95.55 [94.68, 96.54] | 21.12 [20.31, 21.92] |
| gender = female | 1941 | 94.46 [93.41, 95.47] | 20.38 [19.67, 21.09] |
| gender = male | 3065 | 95.27 [94.57, 95.94] | 21.80 [21.07, 22.66] |
| duration_bucket = 0-3s | 916 | 103.22 [101.65, 104.83] | 34.16 [31.82, 36.73] |
| duration_bucket = 10s+ | 909 | 91.95 [90.89, 93.15] | 20.69 [19.86, 21.64] |
| duration_bucket = 3-6s | 1549 | 98.24 [97.30, 99.36] | 21.20 [20.34, 22.07] |
| duration_bucket = 6-10s | 1632 | 94.56 [93.78, 95.50] | 20.16 [19.18, 21.38] |
| seen_speaker = False | 5006 | 94.97 [94.38, 95.59] | 21.28 [20.75, 21.82] |
| has_number = False | 4363 | 95.03 [94.43, 95.66] | 20.21 [19.62, 20.91] |
| has_number = True | 643 | 94.67 [93.18, 96.30] | 26.85 [25.61, 28.16] |
| has_name = False | 3467 | 95.34 [94.58, 96.07] | 19.83 [19.28, 20.37] |
| has_name = True | 1539 | 94.29 [93.40, 95.27] | 23.93 [22.80, 25.31] |

## Paired comparison against Voxtral-Mini-4B-Realtime-2602 (480 ms)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| Voxtral-Mini-3B-2507 (language: nb) | -73.70 | [-74.46, -72.92] | 0.000 |

## Run directories

- Voxtral-Mini-4B-Realtime-2602 (480 ms): `/content/drive/MyDrive/voxtral-runs/20261019-203158_baseline-realtime`
- Voxtral-Mini-3B-2507 (language: nb): `/content/drive/MyDrive/voxtral-runs/20261019-222846_baseline-mini-nb`
