"""Check that the configured model loads and runs one forward pass at the eval batch size.

Evaluation only runs the model forward, so that is what this checks, with the
peak GPU memory it needs. Backward passes are covered by the unit tests (tiny
models); training memory with LoRA is measured on the GPU. A full backward
pass through all parameters of a real model needs far more memory than
anything this project runs (it ran out of memory on a 40 GB A100 for Mini 3B).

    python scripts/smoke.py --config configs/local.yaml
    python scripts/smoke.py --config configs/cloud.yaml --set model.family=voxtral \
        model.hf_id=mistralai/Voxtral-Mini-3B-2507
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from voxnor.config import add_override_argument, load_config
from voxnor.model import count_parameters, dummy_batch, load_model, resolve_device
from voxnor.run_dir import create_run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    add_override_argument(parser)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    torch.manual_seed(cfg.seed)
    run_dir = create_run_dir(cfg)
    device = resolve_device(cfg.runtime.device)

    start = time.perf_counter()
    model = load_model(cfg.model, device)
    load_seconds = time.perf_counter() - start
    total, _ = count_parameters(model)

    batch = dummy_batch(model, batch_size=cfg.runtime.batch_size)
    start = time.perf_counter()
    with torch.inference_mode():
        out = model(**batch)
    step_seconds = time.perf_counter() - start

    result = {
        "family": cfg.model.family,
        "tiny": cfg.model.tiny,
        "device": str(device),
        "dtype": cfg.model.dtype,
        "parameters": total,
        "logits_shape": list(out.logits.shape),
        "loss": out.loss.item(),
        "load_seconds": round(load_seconds, 2),
        "forward_seconds": round(step_seconds, 2),
    }
    if device.type == "cuda":
        result["peak_gpu_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    if not torch.isfinite(out.loss):
        raise RuntimeError(f"Non-finite loss: {result}")

    (run_dir / "smoke.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
