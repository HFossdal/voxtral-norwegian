"""Plot selection-set WER over training for several fine-tuning runs.

    python scripts/plot_selection.py --out reports/figures/lr_sweep_selection.png \
        reports/runs/<run A> reports/runs/<run B> --labels "lr 1e-4" "lr 2e-4"

Reads selection.jsonl and best.json of each training run directory; the step
each run chose is marked. Needs the `analysis` extra.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def selection(run: Path) -> list[dict]:
    lines = (run / "selection.jsonl").read_text(encoding="utf-8").splitlines()
    return sorted((json.loads(line) for line in lines if line.strip()), key=lambda r: r["step"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--labels", nargs="+")
    parser.add_argument("--title", default="Selection-set WER during training")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    labels = args.labels or [run.name for run in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("Give one label per run")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for run, label in zip(args.runs, labels, strict=True):
        rows = selection(run)
        steps = [r["step"] for r in rows]
        line = ax.plot(steps, [100 * r["wer"] for r in rows], marker="o", ms=4, label=label)[0]
        low, high = ([100 * r[k] for r in rows] for k in ("wer_low", "wer_high"))
        ax.fill_between(steps, low, high, color=line.get_color(), alpha=0.15, lw=0)
        best_file = run / "best.json"
        if best_file.exists():
            best = json.loads(best_file.read_text(encoding="utf-8"))
            ax.plot(best["step"], 100 * best["selection_wer"], marker="*", ms=14,
                    color=line.get_color(), mec="black")  # fmt: skip
    ax.set_xlabel("training step")
    ax.set_ylabel("WER on the selection set (%), 95% interval")
    ax.set_ylim(bottom=0)
    ax.set_title(args.title)
    ax.grid(alpha=0.3)
    ax.legend(title="★ chosen step")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
