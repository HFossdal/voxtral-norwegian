"""Transcribe and score one or more splits with the configured model.

    python scripts/evaluate.py --config configs/local.yaml --splits validation
    python scripts/evaluate.py --config configs/cloud.yaml --set name=baseline-realtime \
        --splits validation
    python scripts/evaluate.py --resume runs/20261020-101500_baseline-realtime

Per split, the run directory gets predictions_<split>.jsonl (appended per
batch, so `--resume` continues an interrupted run), scored_<split>.csv and
metrics_<split>.json. The test split is only scored with --allow-test.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from voxnor.config import add_override_argument, load_config
from voxnor.data import AUDIO_COLUMN, load_split, training_speakers
from voxnor.evaluate import read_predictions, score_predictions, transcribe_to_file
from voxnor.mistral_api import MistralApiTranscriber
from voxnor.model import load_model, resolve_device
from voxnor.run_dir import create_run_dir, setup_logging
from voxnor.transcribe import build_transcriber, load_processor

log = logging.getLogger("evaluate")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--config")
    source.add_argument("--resume", help="Existing run directory; its saved config is used")
    add_override_argument(parser)
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument(
        "--allow-test", action="store_true", help="Required to score the test split"
    )
    args = parser.parse_args()

    if "test" in args.splits and not args.allow_test:
        parser.error("The test split is used once per final model; pass --allow-test")
    if args.resume:
        if args.set:
            parser.error("--set cannot be combined with --resume: the saved config is used as is")
        run_dir = Path(args.resume)
        cfg = load_config(run_dir / "config.yaml")
    else:
        cfg = load_config(args.config, args.set)
        run_dir = create_run_dir(cfg)
    setup_logging(run_dir / "evaluate.log")
    log.info("Run directory %s", run_dir)

    torch.manual_seed(cfg.seed)
    if cfg.model.family == "mistral_api":
        transcriber = MistralApiTranscriber(cfg.model.hf_id, cfg.model.language)
    else:
        model = load_model(cfg.model, resolve_device(cfg.runtime.device)).eval()
        processor = load_processor(cfg.model)
        transcriber = build_transcriber(cfg.model, model, processor, cfg.runtime.max_new_tokens)
    train_speakers = training_speakers(cfg.data)

    for split in args.splits:
        ds = load_split(cfg.data, split)
        predictions_path = run_dir / f"predictions_{split}.jsonl"
        transcribe_to_file(ds, transcriber, predictions_path, cfg.runtime.batch_size)

        metadata = ds.remove_columns(AUDIO_COLUMN).to_pandas()
        scored, summary = score_predictions(
            read_predictions(predictions_path), metadata, train_speakers
        )
        scored.to_csv(run_dir / f"scored_{split}.csv", index=False)
        (run_dir / f"metrics_{split}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        wer = summary["overall"]["wer"]
        unseen = summary["slices"]["seen_speaker"].get("False", {}).get("wer")
        log.info(
            "%s: WER %.2f%% [%.2f, %.2f] over %d utterances; unseen speakers: %s",
            split,
            100 * wer["value"],
            100 * wer["low"],
            100 * wer["high"],
            summary["utterances_scored"],
            f"{100 * unseen['value']:.2f}%" if unseen else "n/a",
        )


if __name__ == "__main__":
    main()
