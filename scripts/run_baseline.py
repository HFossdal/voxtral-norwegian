"""Run the whole zero-shot baseline with one command.

    python scripts/run_baseline.py --config configs/cloud.yaml --output-root /workspace/runs

Steps, each skipped or resumed if an earlier invocation already did it:
  1. smoke-test both models (loads weights, reports GPU memory)
  2. validation: Realtime, and Mini with language auto / no / nb
  3. choose Mini's language: lowest WER on unseen-speaker validation
  4. test, once each, for Realtime and the chosen Mini setting
  5. reports and a zip of everything small, for download

Re-run the same command after a disconnect; finished work is not redone.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MINI = ["model.family=voxtral", "model.hf_id=mistralai/Voxtral-Mini-3B-2507"]
SYSTEMS = {
    "baseline-realtime": [],
    "baseline-mini-auto": [*MINI, "model.language=null"],
    "baseline-mini-no": [*MINI, "model.language=no"],
    "baseline-mini-nb": [*MINI, "model.language=nb"],
}
# Small files worth keeping from each run; logs stay behind.
KEEP = ("config.yaml", "environment.json", "predictions_*.jsonl", "scored_*.csv", "metrics_*.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    if not args.skip_smoke:
        step("1/5 smoke tests")
        for name in ("baseline-realtime", "baseline-mini-auto"):
            overrides = [f"output_root={root}", f"name=smoke-{name}", *SYSTEMS[name]]
            run("smoke.py", "--config", args.config, "--set", *overrides)

    step("2/5 validation")
    runs = {name: evaluate(args.config, root, name, "validation") for name in SYSTEMS}

    step("3/5 choose Mini's language setting")
    choice = choose_mini(runs)
    (root / "baseline_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(json.dumps(choice, indent=2))

    step("4/5 test (once per system)")
    finalists = ["baseline-realtime", choice["chosen"]]
    for name in finalists:
        evaluate(args.config, root, name, "test")

    step("5/5 reports and results zip")
    reports = root / "reports"
    run(
        "report.py",
        "--split",
        "validation",
        "--out",
        str(reports / "baseline_validation.md"),
        *(str(runs[name]) for name in SYSTEMS),
    )
    run(
        "report.py",
        "--split",
        "test",
        "--out",
        str(reports / "baseline_test.md"),
        *(str(runs[name]) for name in finalists),
    )
    archive = bundle(root, reports, [runs[name] for name in SYSTEMS])
    print(f"\nDone. Download {archive}")


def evaluate(
    config: str, root: Path, name: str, split: str, overrides: list[str] | None = None
) -> Path:
    """Run (or resume, or skip) one system on one split; return its run directory.

    `overrides` default to the system's entry in SYSTEMS.
    """
    existing = sorted(root.glob(f"*_{name}"))
    if existing:
        run_dir = existing[-1]
        if (run_dir / f"metrics_{split}.json").exists():
            print(f"{name} / {split}: already done in {run_dir}")
            return run_dir
        run("evaluate.py", "--resume", str(run_dir), "--splits", split, *_test_flag(split))
        return run_dir
    overrides = [f"output_root={root}", f"name={name}", *(overrides or SYSTEMS[name])]
    run(
        "evaluate.py",
        "--config",
        config,
        "--set",
        *overrides,
        "--splits",
        split,
        *_test_flag(split),
    )
    return sorted(root.glob(f"*_{name}"))[-1]


def choose_mini(runs: dict[str, Path]) -> dict:
    scores = {}
    for name in SYSTEMS:
        if not name.startswith("baseline-mini"):
            continue
        metrics = json.loads((runs[name] / "metrics_validation.json").read_text())
        scores[name] = metrics["slices"]["seen_speaker"]["False"]["wer"]["value"]
    chosen = min(scores, key=scores.get)
    return {
        "rule": "lowest WER on validation utterances by speakers not in training",
        "unseen_speaker_validation_wer": scores,
        "chosen": chosen,
    }


def bundle(root: Path, reports: Path, run_dirs: list[Path]) -> Path:
    archive = root / "baseline_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in [*reports.glob("*.md"), root / "baseline_choice.json"]:
            zf.write(path, path.relative_to(root))
        for run_dir in run_dirs:
            for pattern in KEEP:
                for path in run_dir.glob(pattern):
                    zf.write(path, Path("runs") / run_dir.name / path.name)
    return archive


def _test_flag(split: str) -> list[str]:
    return ["--allow-test"] if split == "test" else []


def run(script: str, *args: str) -> None:
    subprocess.run([sys.executable, str(REPO / "scripts" / script), *args], check=True, cwd=REPO)


def step(title: str) -> None:
    print(f"\n=== {title} ===", flush=True)


if __name__ == "__main__":
    main()
