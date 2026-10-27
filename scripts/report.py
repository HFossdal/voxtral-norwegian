"""Compare evaluated runs on one split and write a Markdown table.

The first run is the baseline for paired comparisons.

    python scripts/report.py --split validation --out reports/baseline_validation.md \
        runs/<realtime-run> runs/<mini-auto-run> runs/<mini-no-run>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from voxnor.report import load_run, render


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_run(directory, args.split) for directory in args.runs]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(runs, args.split), encoding="utf-8")
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
