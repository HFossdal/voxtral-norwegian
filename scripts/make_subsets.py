"""Create the fixed utterance-id lists used for small, reproducible runs.

The output is committed, so every run on every machine uses the same
utterances. Re-running with the same arguments reproduces the files exactly.

    python scripts/make_subsets.py --config configs/local.yaml --name local --size 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voxnor.config import load_config
from voxnor.data import SPLITS, exclude_speakers, load_metadata, stratified_sample, write_id_list


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--name", required=True, help="Subset directory name under subsets/")
    parser.add_argument("--size", type=int, required=True, help="Utterances per split")
    parser.add_argument(
        "--max-duration", type=float, default=8.0, help="Seconds; keeps CPU runs fast"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta = {split: load_metadata(cfg.data, split) for split in SPLITS}
    # Train ids come from speakers absent from validation and test, so the
    # subset stays valid whichever speaker exclusion a run uses.
    meta["train"] = exclude_speakers(meta["train"], meta["validation"], meta["test"])

    out_dir = Path("subsets") / args.name
    for split, df in meta.items():
        pool = df[df["duration"] <= args.max_duration]
        ids = stratified_sample(pool, args.size, by="region", seed=args.seed)
        header = (
            f"{args.size} {split} utterances of {cfg.data.dataset_id}, stratified by region.\n"
            f"duration <= {args.max_duration} s, seed {args.seed}. Made by scripts/make_subsets.py."
        )
        write_id_list(out_dir / f"{split}.txt", ids, header)
        print(f"{split}: {len(ids)} ids -> {out_dir / f'{split}.txt'}")


if __name__ == "__main__":
    main()
