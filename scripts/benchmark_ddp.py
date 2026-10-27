"""Measure training speed on 1..N GPUs of one machine, for the scaling curve.

    python -m torch.distributed.run --standalone --nproc_per_node 4 scripts/benchmark_ddp.py \
        --config configs/cloud.yaml --set <the run's train.* settings> --out scaling.jsonl

Trains `--steps` optimizer steps of the real setup (same model, LoRA, batch
and micro-batch size) on a fixed random part of the training set, with no
selection evaluations. Appends one line to `--out`: seconds per step over
the steps after `--warmup`, and clips per second. Run it once per GPU
count; nothing from these runs is kept as a model.
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from finetune import prepare

from voxnor import distributed
from voxnor.config import add_override_argument, load_config
from voxnor.finetune import LOG_FILE, train
from voxnor.model import resolve_device
from voxnor.run_dir import environment_info

log = logging.getLogger("benchmark_ddp")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    add_override_argument(parser)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s: %(message)s")
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)  # one line per Hub request otherwise

    dist = distributed.init()
    cfg = load_config(args.config, args.set)
    torch.manual_seed(cfg.seed)
    device = distributed.process_device(resolve_device(cfg.runtime.device), dist)
    s = prepare(cfg, dist, device)

    # One epoch of exactly `steps` full batches.
    n = args.steps * cfg.train.batch_size
    rows = np.random.default_rng(cfg.seed).permutation(len(s.examples))[:n]
    examples = [s.examples[i] for i in sorted(rows)]
    train_cfg = replace(cfg.train, epochs=1, log_every=1, save_every=10**9)

    run_dir = Path(tempfile.mkdtemp(prefix="benchmark-")) if dist.is_main else None
    run_dir = distributed.broadcast_object(run_dir, dist)
    train(
        s.model,
        s.batches,
        examples,
        train_cfg,
        cfg.seed,
        run_dir,
        lambda step: None,
        dist,
    )
    if dist.is_main:
        entries = [json.loads(line) for line in (run_dir / LOG_FILE).open(encoding="utf-8")]
        seconds = {e["step"]: e["seconds"] for e in entries}
        last = max(seconds)
        per_step = (seconds[last] - seconds[args.warmup]) / (last - args.warmup)
        result = {
            "gpus": dist.world_size,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "steps": last,
            "warmup": args.warmup,
            "batch_size": cfg.train.batch_size,
            "micro_batch_size": cfg.train.micro_batch_size,
            "seconds_per_step": round(per_step, 3),
            "clips_per_second": round(cfg.train.batch_size / per_step, 2),
            "peak_gpu_gb": entries[-1]["peak_gpu_gb"],
            "environment": environment_info(),
        }
        with args.out.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        log.info("%s", result)
    distributed.barrier(dist)
    distributed.close(dist)


if __name__ == "__main__":
    main()
