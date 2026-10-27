"""Score Mistral's Voxtral Mini Transcribe 3 (API) on validation with one command.

    MISTRAL_API_KEY=... python scripts/run_api_reference.py --config configs/cloud.yaml \
        --output-root <runs dir>

A reference point, not a baseline: the model is API-only, and its training
data is unknown. Language auto and `no` are each scored on validation, and
the reported setting is the one with the lowest WER on unseen speakers, as
for Voxtral Mini. The API rejects `nb` (400 Bad Request, 2026-10-24).
Needs no GPU. Costs about $2 per validation pass ($0.0035 per audio minute,
checked 2026-10-23). Re-run the same command after a disconnect;
finished work is not redone.
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

from run_baseline import KEEP, evaluate, run, step

API = ["model.family=mistral_api", "model.hf_id=voxtral-mini-transcribe-3", "runtime.batch_size=32"]
SYSTEMS = {
    "reference-mistral-api-auto": [*API, "model.language=null"],
    "reference-mistral-api-no": [*API, "model.language=no"],
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)

    step("1/3 validation")
    runs = {
        name: evaluate(args.config, root, name, "validation", overrides)
        for name, overrides in SYSTEMS.items()
    }

    step("2/3 choose the language setting")
    scores = {}
    for name, run_dir in runs.items():
        metrics = json.loads((run_dir / "metrics_validation.json").read_text())
        scores[name] = metrics["slices"]["seen_speaker"]["False"]["wer"]["value"]
    choice = {
        "rule": "lowest WER on validation utterances by speakers not in training",
        "unseen_speaker_validation_wer": scores,
        "chosen": min(scores, key=scores.get),
    }
    (root / "api_reference_choice.json").write_text(json.dumps(choice, indent=2) + "\n")
    print(json.dumps(choice, indent=2))

    step("3/3 report and results zip")
    report = root / "reports" / "api_reference_validation.md"
    run("report.py", "--split", "validation", "--out", str(report), *map(str, runs.values()))
    archive = root / "api_reference_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in (report, root / "api_reference_choice.json"):
            zf.write(path, path.relative_to(root))
        for run_dir in runs.values():
            for pattern in KEEP:
                for path in run_dir.glob(pattern):
                    zf.write(path, Path("runs") / run_dir.name / path.name)
    print(f"\nDone. Download {archive}")


if __name__ == "__main__":
    main()
