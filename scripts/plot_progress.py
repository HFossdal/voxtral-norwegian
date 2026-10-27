"""Plot validation WER for every system and experiment, in the order they were run.

    python scripts/plot_progress.py

Reads metrics_validation.json of the run directories in reports/runs and
writes reports/figures/validation_progress.png. Needs the `analysis` extra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RUNS = Path("reports/runs")
# (label, run directory, kind); kinds set the colour.
SYSTEMS = [
    ("Mini, zero-shot (nb)", "20261019-222846_baseline-mini-nb", "mini"),
    ("Realtime, zero-shot", "20261019-203158_baseline-realtime", "realtime"),
    ("+ decoder LoRA, 2 epochs", "20261020-123701_finetune-realtime-step2264", "realtime"),
    ("3 epochs", "20261020-170946_finetune-realtime-3ep-step3396", "realtime"),
    ("+ encoder LoRA", "20261020-232609_finetune-realtime-enc-step3396", "realtime"),
    ("5 epochs", "20261021-102102_finetune-realtime-enc-5ep-step5660", "realtime"),
    ("+ augmentation", "20261021-205210_finetune-realtime-enc-5ep-aug-step5660", "realtime"),
    ("8 epochs", "20261022-124210_finetune-realtime-enc-8ep-aug-step9056", "realtime"),
    (
        "960 ms delay (side branch)",
        "20261023-045156_finetune-realtime-enc-8ep-aug-960ms-step8000",
        "side",
    ),
    (
        "+ 69 h SSC data",
        "20261024-082207_finetune-realtime-enc-aug-ssc-step12000",
        "realtime",
    ),
    ("Mini + LoRA, same setup", "20261025-042251_finetune-mini-enc-aug-ssc-step8000", "mini"),
    ("lr 2e-4", "20261025-234651_finetune-mini-enc-aug-ssc-lr2e-4-step12000", "mini"),
]
REFERENCES = [
    ("nb-whisper-large", "20261022-152401_reference-nb-whisper-large"),
    ("Mistral API (Transcribe 3)", "20261024-112447_reference-mistral-api-no"),
    ("nb-whisper-large-verbatim", "20261022-153138_reference-nb-whisper-large-verbatim"),
]
COLOURS = {"realtime": "#4C72B0", "mini": "#DD8452", "side": "#B0B0B0"}
X_MAX = 30


def wer(run: str) -> tuple[float, float, float]:
    metrics = json.loads((RUNS / run / "metrics_validation.json").read_text(encoding="utf-8"))
    w = metrics["overall"]["wer"]
    return w["value"] * 100, w["low"] * 100, w["high"] * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("reports/figures/validation_progress.png"))
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for y, (_, run, kind) in enumerate(SYSTEMS):
        value, low, high = wer(run)
        ax.barh(y, min(value, X_MAX), color=COLOURS[kind])
        if value > X_MAX:
            ax.text(X_MAX - 0.3, y, f"{value:.1f}% →", va="center", ha="right", color="white")
        else:
            ax.errorbar(value, y, xerr=[[value - low], [high - value]], color="black", capsize=2)
            box = {"facecolor": "white", "edgecolor": "none", "pad": 1}
            ax.text(high + 0.3, y, f"{value:.2f}", va="center", fontsize=8, bbox=box)
    for (label, run), style in zip(REFERENCES, (":", "-.", "--"), strict=True):
        value = wer(run)[0]
        ax.axvline(value, color="black", linestyle=style, lw=1, label=f"{label}: {value:.2f}")
    ax.set_yticks(range(len(SYSTEMS)), [label for label, _, _ in SYSTEMS])
    ax.invert_yaxis()
    ax.set_xlim(0, X_MAX)
    ax.set_xlabel("WER on validation (%), 95% bootstrap interval")
    ax.set_title("Voxtral on NPSC Bokmål, in the order the experiments ran")
    ax.legend(title="references (context, not baselines)", fontsize=8, loc="lower right")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
