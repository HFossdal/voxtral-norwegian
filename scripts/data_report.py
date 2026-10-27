"""Write split statistics to reports/data.md and reports/data_stats.json.

Reads only the metadata columns, so it is quick on any machine.

    python scripts/data_report.py --config configs/local.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from voxnor.config import add_override_argument, load_config
from voxnor.data import SPLITS, load_metadata
from voxnor.data_report import build_report, render_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    add_override_argument(parser)
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    splits = {split: load_metadata(cfg.data, split) for split in SPLITS}
    report = build_report(splits)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data_stats.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "data.md").write_text(render_markdown(report, cfg.data.dataset_id), encoding="utf-8")
    print(f"Wrote {out_dir / 'data.md'} and {out_dir / 'data_stats.json'}")


if __name__ == "__main__":
    main()
