"""Re-transcribe some validation utterances one at a time and compare with a run's output.

    python scripts/check_batching.py --run <run dir> --n 24

Evaluation runs in batches. If batching (padding, masks) changed the output,
the run's numbers would describe the pipeline rather than the model. Half
of the sample is utterances whose batched output was empty, since those are
the most suspicious. Small differences can still come from bf16 arithmetic
on different shapes; empty outputs that become real text cannot.
Writes batching_check.json into the run directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from voxnor.config import load_config
from voxnor.data import decode_audio, load_split, select_rows
from voxnor.evaluate import read_predictions
from voxnor.model import load_model, resolve_device
from voxnor.transcribe import build_transcriber, load_processor

SPLIT = "validation"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--n", type=int, default=24)
    args = parser.parse_args()

    cfg = load_config(args.run / "config.yaml")
    batched = read_predictions(args.run / f"predictions_{SPLIT}.jsonl")
    sample = pick(batched, args.n, cfg.seed)

    torch.manual_seed(cfg.seed)
    model = load_model(cfg.model, resolve_device(cfg.runtime.device)).eval()
    processor = load_processor(cfg.model)
    transcriber = build_transcriber(cfg.model, model, processor, cfg.runtime.max_new_tokens)
    ds = select_rows(load_split(cfg.data, SPLIT), list(sample["utterance_id"]))

    rows = []
    for uid, audio in zip(ds["utterance_id"], ds["utterance_audio_file"], strict=True):
        single = transcriber.transcribe([decode_audio(audio)])[0]
        before = sample.set_index("utterance_id").at[uid, "hypothesis"]
        rows.append({"utterance_id": uid, "batched": before, "single": single})
    result = summarize(pd.DataFrame(rows))
    (args.run / "batching_check.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    for row in result["rows"]:
        if row["batched"] != row["single"]:
            print(f"\n{row['utterance_id']}")
            print(f"  batched: {row['batched']!r}\n  single:  {row['single']!r}")


def pick(predictions: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    empty = predictions["hypothesis"].str.strip() == ""
    half = n // 2
    return pd.concat(
        [
            predictions[empty].sample(min(half, int(empty.sum())), random_state=seed),
            predictions[~empty].sample(n - half, random_state=seed),
        ]
    )


def summarize(df: pd.DataFrame) -> dict:
    was_empty = df["batched"].str.strip() == ""
    now_empty = df["single"].str.strip() == ""
    return {
        "utterances": len(df),
        "identical": int((df["batched"] == df["single"]).sum()),
        "empty_batched": int(was_empty.sum()),
        "empty_batched_still_empty_single": int((was_empty & now_empty).sum()),
        "rows": df.to_dict("records"),
    }


if __name__ == "__main__":
    main()
