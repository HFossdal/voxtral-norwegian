"""Run the fine-tuning with one command.

    python scripts/run_finetune.py --config configs/cloud.yaml --output-root <runs dir>

Steps, each skipped or resumed if an earlier invocation already did it:
  1. force-align the training set (scripts/align_train.py)
  2. fine-tune, choosing the checkpoint on the selection set (scripts/finetune.py)
  3. score the chosen adapter on the full validation split (scripts/evaluate.py)
  4. a validation report against the zero-shot baselines, and a zip of
     everything small, for download

The test split is not touched (`scripts/run_test_eval.py` scores it).
Re-run the same command after a disconnect; finished work is not redone. Extra `--set` overrides go
to the fine-tuning run (e.g. `--name finetune-lr2e-4 --set train.learning_rate=2e-4`);
the `model.*` ones (e.g. the delay) also go to the full validation.
`--gpus N` trains on N GPUs of this machine with torchrun; the full
validation still runs on one.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
BASELINES = ("20261019-203158_baseline-realtime", "20261019-222846_baseline-mini-nb")
# Small files worth keeping; logs, adapters and training state stay on Drive.
KEEP = (
    "config.yaml",
    "environment.json",
    "data_summary.json",
    "train_log.jsonl",
    "selection.jsonl",
    "best.json",
    "predictions_*.jsonl",
    "scored_*.csv",
    "metrics_*.json",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--name", default="finetune-realtime")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gpus", type=int, default=1)
    args = parser.parse_args()
    train = launcher(args.gpus)
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    step("1/4 align the training set")
    alignments = root / "alignments"
    if (alignments / "summary.json").exists():
        print(f"already done in {alignments}")
    else:
        run("align_train.py", "--config", args.config, "--out", str(alignments))

    step("2/4 fine-tune")
    train_dir = latest(root, args.name)
    if train_dir and (train_dir / "best.json").exists():
        print(f"already done in {train_dir}")
    elif train_dir:
        train("finetune.py", "--resume", str(train_dir))
    else:
        overrides = [
            f"output_root={root}",
            f"name={args.name}",
            f"train.alignments={alignments / 'train.jsonl'}",
            *args.set,
        ]
        train("finetune.py", "--config", args.config, "--set", *overrides)
        train_dir = latest(root, args.name)
    best = json.loads((train_dir / "best.json").read_text())
    print(json.dumps(best, indent=2))

    step("3/4 full validation with the chosen adapter")
    eval_name = f"{args.name}-step{best['step']}"
    eval_dir = latest(root, eval_name)
    if eval_dir and (eval_dir / "metrics_validation.json").exists():
        print(f"already done in {eval_dir}")
    elif eval_dir:
        run("evaluate.py", "--resume", str(eval_dir), "--splits", "validation")
    else:
        overrides = [
            f"output_root={root}",
            f"name={eval_name}",
            f"model.adapter={best['adapter']}",
            *(o for o in args.set if o.startswith("model.")),
        ]
        run("evaluate.py", "--config", args.config, "--set", *overrides, "--splits", "validation")
        eval_dir = latest(root, eval_name)

    step("4/4 report and results zip")
    report = root / "reports" / f"{args.name}_validation.md"
    baselines = [REPO / "reports" / "runs" / name for name in BASELINES]
    run(
        "report.py",
        "--split",
        "validation",
        "--out",
        str(report),
        *map(str, baselines),
        str(eval_dir),
    )
    archive = root / f"{args.name}_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report, report.relative_to(root))
        zf.write(alignments / "summary.json", "alignments/summary.json")
        for run_dir in (train_dir, eval_dir):
            for pattern in KEEP:
                for path in run_dir.glob(pattern):
                    zf.write(path, Path("runs") / run_dir.name / path.name)
            # Not the per-GPU parts of a multi-GPU run.
            for path in sorted((run_dir / "selection").glob("predictions_step-??????.jsonl")):
                zf.write(path, Path("runs") / run_dir.name / "selection" / path.name)
    print(f"\nDone. Download {archive}")


def latest(root: Path, name: str) -> Path | None:
    existing = sorted(root.glob(f"*_{name}"))
    return existing[-1] if existing else None


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], check=True, cwd=REPO)


def launcher(gpus: int):
    """`run`, or the same through torchrun with one process per GPU."""
    if gpus == 1:
        return run
    torchrun = [sys.executable, "-m", "torch.distributed.run", "--standalone"]

    def run_distributed(script: str, *args: str) -> None:
        command = [*torchrun, f"--nproc_per_node={gpus}", str(REPO / "scripts" / script), *args]
        subprocess.run(command, check=True, cwd=REPO)

    return run_distributed


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


if __name__ == "__main__":
    main()
