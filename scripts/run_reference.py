"""Score the nb-whisper reference points on validation with one command.

    python scripts/run_reference.py --config configs/cloud.yaml --output-root <runs dir>

nb-whisper is context, not a baseline: it was trained on NPSC and other
parliament speech, so it has likely heard these speakers and meetings.
Greedy decoding and the standard normaliser, as for every system.
Validation only. Re-run the same command after a disconnect; finished work
is not redone.
"""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

from run_baseline import KEEP, run, step

WHISPER = ["model.family=whisper", "model.language=no", "runtime.batch_size=32"]
SYSTEMS = {
    "reference-nb-whisper-large": [*WHISPER, "model.hf_id=NbAiLab/nb-whisper-large"],
    "reference-nb-whisper-large-verbatim": [
        *WHISPER,
        "model.hf_id=NbAiLab/nb-whisper-large-verbatim",
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    step("1/2 validation")
    runs = [evaluate(args.config, root, name) for name in SYSTEMS]

    step("2/2 report and results zip")
    report = root / "reports" / "reference_validation.md"
    run("report.py", "--split", "validation", "--out", str(report), *map(str, runs))
    archive = root / "reference_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(report, report.relative_to(root))
        for run_dir in runs:
            for pattern in KEEP:
                for path in run_dir.glob(pattern):
                    zf.write(path, Path("runs") / run_dir.name / path.name)
    print(f"\nDone. Download {archive}")


def evaluate(config: str, root: Path, name: str) -> Path:
    """Run, resume or skip one system on validation; return its run directory."""
    existing = sorted(root.glob(f"*_{name}"))
    if existing:
        run_dir = existing[-1]
        if (run_dir / "metrics_validation.json").exists():
            print(f"{name}: already done in {run_dir}")
        else:
            run("evaluate.py", "--resume", str(run_dir), "--splits", "validation")
        return run_dir
    overrides = [f"output_root={root}", f"name={name}", *SYSTEMS[name]]
    run("evaluate.py", "--config", config, "--set", *overrides, "--splits", "validation")
    return sorted(root.glob(f"*_{name}"))[-1]


if __name__ == "__main__":
    main()
