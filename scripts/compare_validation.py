"""Compare two evaluated runs on validation (or test), also with the spelling convention removed.

    python scripts/compare_validation.py --config configs/local.yaml \
        --out reports/<name>_validation.md reports/runs/<run A> reports/runs/<run B>

Writes the standard table of scripts/report.py (paired against run A),
followed by the paired comparison on number-free utterances with spoken and
standard forms folded together. `--split test`
compares finished test runs; the spoken/standard pairs still come
from train and validation text only. Reads only metadata (no audio) and
the run directories.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from analyse_finetune import _included, convention_section

from voxnor.config import add_override_argument, load_config
from voxnor.data import load_metadata
from voxnor.report import load_run, render
from voxnor.variants import LEXICAL_PAIRS, corpus_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    add_override_argument(parser)
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--labels", nargs=2, help="Short names for the convention table")
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    val_meta = load_metadata(cfg.data, "validation")
    train_meta = load_metadata(cfg.data, "train")
    pairs = {**corpus_pairs([*train_meta.raw_text, *val_meta.raw_text]), **LEXICAL_PAIRS}

    runs = [load_run(directory, args.split) for directory in args.runs]
    a, b = (_included(run.scored) for run in runs)
    if list(a.utterance_id) != list(b.utterance_id):
        raise ValueError("The two runs must score the same utterances in the same order")
    labels = tuple(args.labels) if args.labels else (runs[0].label, runs[1].label)
    sections = [
        render(runs, args.split),
        *convention_section(a, b, pairs, labels, "## With the spelling convention removed"),
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
