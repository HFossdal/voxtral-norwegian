"""LoRA fine-tune Voxtral Realtime or Voxtral Mini on the aligned training set.

    python scripts/finetune.py --config configs/cloud.yaml \
        --set name=finetune-realtime train.alignments=<dir>/train.jsonl
    python scripts/finetune.py --resume <run dir>
    python -m torch.distributed.run --standalone --nproc_per_node 4 scripts/finetune.py ...

Trains a LoRA adapter for `train.epochs` epochs. `model.family=voxtral` trains
Voxtral Mini's offline transcription mode on the same examples, with the
`model.language` hint in its prompt. The base model and
every `train.eval_every` steps are scored on the selection set
(`train.selection_ids`, unseen-speaker validation). The training state is
saved every `train.save_every` steps, so `--resume` continues an interrupted
run. At the end, best.json names the adapter with the lowest selection WER.
Files in the run directory: see `voxnor.finetune`. Under torchrun, every GPU
trains on its part of each batch (`voxnor.distributed`).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

# Long and short batches alternate; without this, fragmentation alone can run
# the GPU out of memory.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import datasets
import torch
from peft import PeftModel
from transformers import ProcessorMixin

from voxnor import distributed
from voxnor.config import RunConfig, add_override_argument, load_config, save_config
from voxnor.data import AUDIO_COLUMN, load_split, read_id_list, select_rows, training_speakers
from voxnor.distributed import Dist
from voxnor.finetune import (
    OfflineBatches,
    RealtimeBatches,
    SelectionEval,
    TrainingBatches,
    choose_best,
    train,
)
from voxnor.model import count_parameters, load_model, resolve_device
from voxnor.realtime_stream import layout_from_processor
from voxnor.realtime_train import add_lora
from voxnor.run_dir import create_run_dir, setup_logging
from voxnor.train_data import Example, add_extra_data, read_alignments, select_examples
from voxnor.transcribe import build_transcriber, load_processor

log = logging.getLogger("finetune")


@dataclass
class Setup:
    """Everything training needs, loaded from the config."""

    processor: ProcessorMixin
    batches: TrainingBatches
    examples: list[Example]
    data_summary: dict
    selection_ds: datasets.Dataset
    model: PeftModel


def prepare(cfg: RunConfig, dist: Dist, device: torch.device) -> Setup:
    """Load the data and the model, and attach LoRA (also used by benchmark_ddp.py)."""
    # The main process downloads and builds the dataset caches; the others reuse them.
    with distributed.main_first(dist):
        processor = load_processor(cfg.model)
        realtime = cfg.model.family == "voxtral_realtime"
        layout = layout_from_processor(processor) if realtime else None
        train_ds = load_split(cfg.data, "train")
        alignments = read_alignments(Path(cfg.train.alignments))
        npsc_utterances = len(train_ds)
        if cfg.train.extra_data:
            extra = Path(cfg.train.extra_data)
            train_ds, alignments = add_extra_data(train_ds, alignments, extra)
        selection_ds = select_rows(
            load_split(cfg.data, "validation"), read_id_list(cfg.train.selection_ids)
        )
        model = load_model(cfg.model, device)
    train_meta = train_ds.remove_columns(AUDIO_COLUMN).to_pandas()
    examples, dropped = select_examples(train_meta, alignments, processor, layout)
    data_summary = {
        "utterances": len(train_meta),
        "npsc_utterances": npsc_utterances,
        "examples": len(examples),
        "hours": round(sum(e.seconds for e in examples) / 3600, 2),
        "extra_hours": round(
            sum(e.seconds for e in examples if e.row >= npsc_utterances) / 3600, 2
        ),
        "dropped": dict(dropped),
    }
    model = add_lora(
        model,
        cfg.train.lora_rank,
        cfg.train.lora_alpha,
        cfg.train.lora_dropout,
        encoder=cfg.train.lora_encoder,
    )
    total, trainable = count_parameters(model)
    log.info("Trainable parameters: %d of %d", trainable, total)
    if realtime:
        batches = RealtimeBatches(processor, layout, train_ds)
    else:
        batches = OfflineBatches(processor, cfg.model.hf_id, cfg.model.language, train_ds)
    return Setup(processor, batches, examples, data_summary, selection_ds, model)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume", help="Existing run directory; its saved config is used")
    add_override_argument(parser)
    args = parser.parse_args()
    dist = distributed.init()
    if args.resume:
        if args.set:
            parser.error("--set cannot be combined with --resume: the saved config is used as is")
        run_dir = Path(args.resume)
        cfg = load_config(run_dir / "config.yaml")
        # Settings added to the code since the run started take their
        # defaults; record them, so config.yaml says what the run used.
        if dist.is_main:
            save_config(cfg, run_dir / "config.yaml")
    else:
        cfg = load_config(args.config, args.set)
        if cfg.model.family not in ("voxtral_realtime", "voxtral") or cfg.model.adapter:
            parser.error("Fine-tuning starts from a pretrained Voxtral model")
        if not cfg.train.alignments:
            parser.error("Set train.alignments to the output of scripts/align_train.py")
        run_dir = create_run_dir(cfg) if dist.is_main else None
        run_dir = distributed.broadcast_object(run_dir, dist)
    if dist.is_main:
        setup_logging(run_dir / "finetune.log")
    log.info("Run directory %s", run_dir)
    torch.manual_seed(cfg.seed)
    device = distributed.process_device(resolve_device(cfg.runtime.device), dist)

    s = prepare(cfg, dist, device)
    if dist.is_main:
        (run_dir / "data_summary.json").write_text(
            json.dumps(s.data_summary, indent=2) + "\n", encoding="utf-8"
        )
    log.info("Training data: %s", s.data_summary)
    on_eval = SelectionEval(
        s.model,
        build_transcriber(cfg.model, s.model, s.processor, cfg.runtime.max_new_tokens),
        s.selection_ds,
        s.selection_ds.remove_columns(AUDIO_COLUMN).to_pandas(),
        training_speakers(cfg.data),
        run_dir,
        cfg.runtime.batch_size,
        dist,
    )
    train(
        s.model,
        s.batches,
        s.examples,
        cfg.train,
        cfg.seed,
        run_dir,
        on_eval,
        dist,
    )
    if dist.is_main:
        best = choose_best(run_dir)
        log.info("Best: %s", best)
    distributed.barrier(dist)
    distributed.close(dist)


if __name__ == "__main__":
    main()
