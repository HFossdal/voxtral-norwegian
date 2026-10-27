# Running on a cloud GPU

All real model runs happened on a single rented GPU (Colab, A100 40 GB).
This page covers the setup and the commands that produce every reported
result.

## What to rent

A single GPU with **bf16 support and at least 24 GB**: L4, A10G, A100 or
H100. Fine-tuning Voxtral Mini peaks at about 27 GB, so it needs an A100
or larger.
- Inference alone needs about 9.5 GB for Voxtral Mini 3B and more than
  16 GB for Voxtral Realtime 4B (model cards).
- Colab's free T4 has no bf16 support. Don't use it.
- Disk: about 20 GB for the two models (≈9 GB each), 10 GB for the NPSC
  Parquet files, and about 30 GB if the SSC clips are built.

## Setup

Every run records the git commit and whether the working tree had
uncommitted changes (`environment.json`), so run from a clean checkout.

**RunPod** (or any machine with a network volume at `/workspace`, so
models, data and runs survive restarts):

```sh
cd /workspace
git clone https://github.com/HFossdal/voxtral-norwegian.git && cd voxtral-norwegian
pip install -e .                      # installs the pinned torch/transformers
export HF_HOME=/workspace/hf          # model and dataset cache on the volume
R=/workspace/runs
```

**Colab** (L4 or A100 runtime):

```python
from google.colab import drive
drive.mount("/content/drive")
!git clone https://github.com/HFossdal/voxtral-norwegian.git /content/voxtral-norwegian
%cd /content/voxtral-norwegian
!pip install -e .
```

Then restart the runtime once, so the pinned versions are the ones that get
imported, and `%cd /content/voxtral-norwegian` again. Keep the HF cache on
local disk (re-downloading is faster than reading 9 GB from Drive) and put
runs on Drive, so they survive a disconnect. The notebooks in `notebooks/`
do all of this and then call one driver each.

In a Colab cell, a `!` command must be on one line. Put long values in
Python variables and pass them as `{name}`.

## The drivers

Each driver chains the steps, skips or resumes finished work when it is run
again, and zips the small outputs for download.

**Zero-shot baselines** (`notebooks/colab_baseline.ipynb`). Scores Realtime
and Mini with three language settings on validation, picks Mini's setting
on unseen-speaker validation, and scores the chosen systems on test:

```sh
python scripts/run_baseline.py --config configs/cloud.yaml --output-root $R
```

**Extra training data.** Cuts clips out of the Stortinget Speech Corpus;
the first 12 shards of the seeded order give the 69.4 h used here:

```sh
python scripts/build_ssc.py --out $R/ssc --shards 12
```

**Fine-tuning** (`notebooks/colab_finetune.ipynb`). Aligns the training
set, fine-tunes, picks the checkpoint on the selection set, scores it on
full validation and writes a report. The final model:

```sh
python scripts/run_finetune.py --config configs/cloud.yaml --output-root $R \
  --name finetune-mini-enc-aug-ssc-lr2e-4 \
  --set model.family=voxtral model.hf_id=mistralai/Voxtral-Mini-3B-2507 model.language=nb \
        train.lora_encoder=true train.augment=true train.extra_data=$R/ssc \
        train.epochs=3 train.learning_rate=2e-4 train.micro_batch_size=2 train.eval_every=2000
```

The streaming model is the same without the `model.*` overrides and the
learning rate, with `train.micro_batch_size=4`. Every other run's settings
are in its `config.yaml` under `reports/runs/`. Give each changed run its
own `--name`, so the skip-if-done logic does not mix runs up.

**Reference systems.** nb-whisper large and large-verbatim, and Mistral's
API model (needs `MISTRAL_API_KEY`), on validation:

```sh
python scripts/run_reference.py --config configs/cloud.yaml --output-root $R
python scripts/run_api_reference.py --config configs/cloud.yaml --output-root $R
```

**Test, once** (`notebooks/colab_test.ipynb`). Names the final model from
the full-validation results and writes that choice to `test_plan.json`
before scoring anything, then scores every system once on test. The API
part costs about $2 for the 9.1 h of test audio:

```sh
python scripts/run_test_eval.py --config configs/cloud.yaml --output-root $R
```

After a run, copy each run directory into `reports/runs/<run name>/`
(everything but the logs and adapters); the reports and the error analysis
work from those files.

## Throughput

Measured on an A100 40 GB:
- Fine-tuning Voxtral Mini (final model): 4.19 s per step of 16 clips,
  12,378 steps, about 14.4 h, peak 26.9 GB.
- Realtime evaluation, full validation (9.6 h of audio): 7.3-7.8x real
  time, about 1 h 20 min.
- Alignment of the training set: 11 min.
- The whole test evaluation: about 3 h 15 min.
