# Results on validation

WER and CER in %, with 95% bootstrap intervals.

| system | WER | WER, number-free utts | CER | name recall | number recall | scored | excluded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| voxtral-mini-transcribe-3 (API, language: auto) | 11.14 [10.83, 11.48] | 9.99 [9.68, 10.30] | 5.81 [5.59, 6.02] | 86.1% | 20.3% | 5409 | 52 |
| voxtral-mini-transcribe-3 (API, language: no) | 10.91 [10.59, 11.24] | 9.73 [9.43, 10.03] | 5.67 [5.45, 5.87] | 86.3% | 20.6% | 5409 | 52 |

## WER by slice

| slice | utts | voxtral-mini-transcribe-3 (API, language: auto) | voxtral-mini-transcribe-3 (API, language: no) |
| --- | --- | --- | --- |
| region = east | 2700 | 9.00 [8.57, 9.44] | 8.77 [8.33, 9.22] |
| region = mid | 521 | 15.43 [14.15, 16.79] | 15.18 [13.92, 16.48] |
| region = north | 412 | 12.79 [11.73, 13.80] | 12.54 [11.47, 13.51] |
| region = south | 480 | 9.08 [8.32, 9.86] | 8.80 [8.03, 9.56] |
| region = unknown | 2 | 112.50 [75.00, 150.00] | 112.50 [75.00, 150.00] |
| region = west | 1294 | 13.73 [13.10, 14.33] | 13.50 [12.88, 14.12] |
| gender = female | 2352 | 9.85 [9.40, 10.30] | 9.64 [9.20, 10.10] |
| gender = male | 3055 | 12.12 [11.71, 12.55] | 11.86 [11.46, 12.29] |
| gender = unknown | 2 | 112.50 [75.00, 150.00] | 112.50 [75.00, 150.00] |
| duration_bucket = 0-3s | 1179 | 23.52 [21.59, 25.44] | 21.22 [19.42, 23.06] |
| duration_bucket = 10s+ | 929 | 10.35 [9.81, 10.93] | 10.24 [9.72, 10.83] |
| duration_bucket = 3-6s | 1607 | 11.23 [10.58, 11.91] | 11.00 [10.36, 11.71] |
| duration_bucket = 6-10s | 1694 | 9.73 [9.28, 10.23] | 9.72 [9.25, 10.21] |
| seen_speaker = False | 3037 | 11.66 [11.23, 12.10] | 11.47 [11.05, 11.90] |
| seen_speaker = True | 2372 | 10.50 [10.05, 10.97] | 10.21 [9.80, 10.69] |
| has_number = False | 4775 | 9.99 [9.68, 10.30] | 9.73 [9.43, 10.03] |
| has_number = True | 634 | 17.95 [16.90, 19.04] | 17.85 [16.82, 18.94] |
| has_name = False | 3734 | 10.95 [10.56, 11.34] | 10.62 [10.24, 11.00] |
| has_name = True | 1675 | 11.51 [11.00, 11.99] | 11.44 [10.94, 11.93] |

## Paired comparison against voxtral-mini-transcribe-3 (API, language: auto)

| system | ΔWER (points) | 95% interval | share of resamples not better |
| --- | --- | --- | --- |
| voxtral-mini-transcribe-3 (API, language: no) | -0.24 | [-0.30, -0.16] | 0.000 |

## Run directories

- voxtral-mini-transcribe-3 (API, language: auto): `/content/drive/MyDrive/voxtral-runs/20261024-111452_reference-mistral-api-auto`
- voxtral-mini-transcribe-3 (API, language: no): `/content/drive/MyDrive/voxtral-runs/20261024-112447_reference-mistral-api-no`
