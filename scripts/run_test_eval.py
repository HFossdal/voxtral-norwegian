"""Run the test evaluation with one command.

    python scripts/run_test_eval.py --config configs/cloud.yaml --output-root <runs dir>

Steps, each skipped or resumed if an earlier invocation already did it:
  1. find the validation run of every system below; per model, the run with
     the lowest full-validation WER is that model's candidate, and the better
     candidate is the final model; written to test_plan.json before anything
     is scored on test
  2. score every system on test, once, in its validation run directory
  3. test reports and a zip for download

Every run and its full validation must be finished (scripts/run_finetune.py,
run_reference.py, run_api_reference.py). The Mistral API run needs the
MISTRAL_API_KEY environment variable. Nothing here is chosen on test.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASELINES = REPO / "reports" / "runs"
# Zero-shot baselines, already scored on test by scripts/run_baseline.py.
REALTIME = BASELINES / "20261019-203158_baseline-realtime"
MINI_NB = BASELINES / "20261019-222846_baseline-mini-nb"
# Fine-tuning runs, by name; the step scored is the one in best.json.
PLAIN = "finetune-realtime-3ep"  # the first recipe
# Candidate runs per model. A run that was never started is skipped;
# one that was started must be finished.
CANDIDATES = {
    "realtime": ("finetune-realtime-enc-aug-ssc",),
    "mini": (
        "finetune-mini-enc-aug-ssc",
        "finetune-mini-enc-aug-ssc-lr2e-4",
        "finetune-mini-enc-aug-ssc-lr5e-5",  # if run
    ),
}
# Context, not baselines.
REFERENCES = (
    "reference-mistral-api-no",
    "reference-nb-whisper-large",
    "reference-nb-whisper-large-verbatim",
)
KEEP = ("config.yaml", "environment.json", "predictions_*.jsonl", "scored_*.csv", "metrics_*.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root

    step("1/3 find the runs and name the final model (validation only)")
    plan_path = root / "test_plan.json"
    if plan_path.exists():
        # Fixed the first time; a later invocation only finishes the scoring.
        plan = json.loads(plan_path.read_text())
    else:
        plan = make_plan(root)
        plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan, indent=2))
    runs = {name: Path(directory) for name, directory in plan["runs"].items()}
    if not os.environ.get("MISTRAL_API_KEY") and any(
        is_api(d) and not (d / "metrics_test.json").exists() for d in runs.values()
    ):
        raise SystemExit("Set MISTRAL_API_KEY for the Mistral API run")

    step("2/3 test, once per system")
    for name, directory in runs.items():
        if (directory / "metrics_test.json").exists():
            print(f"{name}: already done in {directory}")
        else:
            print(f"{name}: scoring test in {directory}", flush=True)
            run("evaluate.py", "--resume", str(directory), "--splits", "test", "--allow-test")

    step("3/3 reports and results zip")
    final = runs[plan["final"]]
    other = runs[next(c for c in plan["candidates"].values() if c != plan["final"])]
    references = [runs[name] for name in REFERENCES]
    reports = {
        # The headline: zero-shot, the plain recipe and the final model.
        "final_test.md": [MINI_NB, REALTIME, runs[PLAIN], final],
        # What the later experiments added.
        "final_vs_plain_test.md": [runs[PLAIN], final],
        # The other candidate (streaming vs offline) and the references.
        "final_vs_others_test.md": [final, other, *references],
    }
    for filename, directories in reports.items():
        out = root / "reports" / filename
        run("report.py", "--split", "test", "--out", str(out), *map(str, directories))
    archive = root / "final_test_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename in reports:
            zf.write(root / "reports" / filename, Path("reports") / filename)
        zf.write(plan_path, plan_path.name)
        for directory in runs.values():
            for pattern in KEEP:
                for path in directory.glob(pattern):
                    zf.write(path, Path("runs") / directory.name / path.name)
    print(f"\nDone. Download {archive}")


def make_plan(root: Path) -> dict:
    """The validation run of every system, and the final model chosen on validation."""
    runs = {PLAIN: finetune_eval_dir(root, PLAIN)}
    wer, candidates = {}, {}
    for model, names in CANDIDATES.items():
        started = [name for name in names if latest(root, name) is not None]
        if not started:
            raise SystemExit(f"{model}: none of {names} was run in {root}")
        dirs = {name: finetune_eval_dir(root, name) for name in started}
        wer.update({name: validation_wer(d) for name, d in dirs.items()})
        candidates[model] = min(started, key=wer.get)
        runs[candidates[model]] = dirs[candidates[model]]
    for name in REFERENCES:
        directory = latest(root, name)
        if directory is None or not (directory / "metrics_validation.json").exists():
            raise SystemExit(f"{name}: no finished validation run in {root}")
        runs[name] = directory
    return {
        "rule": (
            "per model, the run with the lowest full-validation WER is its candidate; "
            "the final model is the better candidate"
        ),
        "validation_wer": wer,
        "candidates": candidates,
        "final": min(candidates.values(), key=wer.get),
        "runs": {name: str(directory) for name, directory in runs.items()},
    }


def finetune_eval_dir(root: Path, name: str) -> Path:
    """The full-validation run of the step a fine-tuning run chose."""
    train_dir = latest(root, name)
    if train_dir is None or not (train_dir / "best.json").exists():
        raise SystemExit(f"{name}: no finished fine-tuning run in {root}")
    best = json.loads((train_dir / "best.json").read_text())
    eval_dir = latest(root, f"{name}-step{best['step']}")
    if eval_dir is None or not (eval_dir / "metrics_validation.json").exists():
        raise SystemExit(f"{name}: full validation of step {best['step']} is not finished")
    return eval_dir


def validation_wer(directory: Path) -> float:
    metrics = json.loads((directory / "metrics_validation.json").read_text(encoding="utf-8"))
    return metrics["overall"]["wer"]["value"]


def is_api(directory: Path) -> bool:
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    return config["model"]["family"] == "mistral_api"


def latest(root: Path, name: str) -> Path | None:
    existing = sorted(root.glob(f"*_{name}"))
    return existing[-1] if existing else None


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], check=True, cwd=REPO)


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


if __name__ == "__main__":
    main()
