"""Create the fixed list of validation utterances that fine-tuning checkpoints are chosen on.

    python scripts/make_selection_ids.py --config configs/cloud.yaml --size 600

A seeded random sample of the validation utterances whose speakers are not in
the training set, so its WER estimates the WER on all of them.
Written to subsets/selection/validation.txt, which is committed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voxnor.config import load_config
from voxnor.data import load_metadata, training_speakers, write_id_list

OUT = Path("subsets/selection/validation.txt")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--size", type=int, default=600)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    validation = load_metadata(cfg.data, "validation")
    unseen = validation[~validation["speaker_id"].isin(training_speakers(cfg.data))]
    sample = unseen.sample(n=args.size, random_state=args.seed)
    header = (
        f"{args.size} of the {len(unseen)} validation utterances of {cfg.data.dataset_id} by\n"
        f"speakers not in training (exclusion {list(cfg.data.exclude_train_speakers_from)}).\n"
        f"Random sample, seed {args.seed}. {sample['duration'].sum() / 3600:.2f} h. "
        "Made by scripts/make_selection_ids.py."
    )
    write_id_list(OUT, sorted(sample["utterance_id"]), header)
    print(header)
    print(sample["region"].value_counts().to_string())


if __name__ == "__main__":
    main()
