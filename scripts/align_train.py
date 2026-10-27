"""Force-align the training transcripts to their audio, for the training targets.

    python scripts/align_train.py --config configs/cloud.yaml --out <dir>

Aligns `standardized_text` of every training utterance (test speakers left
out) with the Norwegian wav2vec2 CTC model and appends the word timings to
<dir>/train.jsonl. Re-run the same command after an interruption;
aligned utterances are skipped. Then writes <dir>/summary.json: how many
failed, the distribution of alignment scores, and the worst-scoring
utterances to look at by hand.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from voxnor.align import CtcAligner
from voxnor.config import add_override_argument, load_config
from voxnor.data import AUDIO_COLUMN, load_split
from voxnor.model import resolve_device
from voxnor.train_data import align_to_file, alignment_summary, read_alignments

ALIGNER = "NbAiLab/nb-wav2vec2-300m-bokmaal-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    add_override_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    ds = load_split(cfg.data, "train")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / "train.jsonl"
    aligner = CtcAligner.from_pretrained(ALIGNER, resolve_device(cfg.runtime.device))
    align_to_file(ds, aligner, path)

    metadata = ds.remove_columns(AUDIO_COLUMN).to_pandas()
    summary = {"aligner": ALIGNER, **alignment_summary(read_alignments(path), metadata)}
    (args.out / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "worst"}, indent=2))


if __name__ == "__main__":
    main()
