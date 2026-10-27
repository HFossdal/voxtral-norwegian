"""Markdown tables comparing evaluated runs on one split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from voxnor.analysis import word_counts
from voxnor.metrics import paired_bootstrap


@dataclass(frozen=True)
class EvaluatedRun:
    label: str
    directory: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    scored: pd.DataFrame


def load_run(directory: Path, split: str, label: str | None = None) -> EvaluatedRun:
    config = yaml.safe_load((directory / "config.yaml").read_text(encoding="utf-8"))
    metrics = json.loads((directory / f"metrics_{split}.json").read_text(encoding="utf-8"))
    scored = pd.read_csv(
        directory / f"scored_{split}.csv", keep_default_na=False, na_values={"excluded": [""]}
    )
    return EvaluatedRun(label or describe(config), directory, config, metrics, scored)


def describe(config: dict[str, Any]) -> str:
    model = config["model"]
    name = model["hf_id"].split("/")[-1]
    if model["tiny"]:
        name = f"TINY RANDOM {model['family']}"
    if model.get("adapter"):
        # Adapters live in <timestamp>_<run name>/adapters/step-N; the run
        # name tells the fine-tuning runs apart.
        adapter = Path(model["adapter"])
        run_name = adapter.parent.parent.name.split("_", 1)[-1]
        name += f" + LoRA {run_name} {adapter.name}"
    if model["family"] == "voxtral_realtime":
        return f"{name} ({model['transcription_delay_ms']} ms)"
    if model["family"] == "mistral_api":
        return f"{name} (API, language: {model.get('language') or 'auto'})"
    return f"{name} (language: {model.get('language') or 'auto'})"


def render(runs: list[EvaluatedRun], split: str) -> str:
    lines = [f"# Results on {split}", "", "WER and CER in %, with 95% bootstrap intervals.", ""]
    lines += _headline_table(runs)
    lines += ["", "## WER by slice", ""]
    lines += _slice_table(runs)
    if len(runs) > 1:
        lines += ["", f"## Paired comparison against {runs[0].label}", ""]
        lines += _paired_table(runs)
    lines += ["", "## Run directories", ""]
    lines += [f"- {run.label}: `{run.directory.as_posix()}`" for run in runs]
    return "\n".join(lines) + "\n"


def _headline_table(runs: list[EvaluatedRun]) -> list[str]:
    rows = [
        "| system | WER | WER, number-free utts | CER | name recall | number recall "
        "| scored | excluded |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for run in runs:
        m = run.metrics
        rows.append(
            f"| {run.label} | {_pct(m['overall']['wer'])} | {_pct(m['number_free']['wer'])} "
            f"| {_pct(m['overall']['cer'])} | {_share(m['names']['recall'])} "
            f"| {_share(m['numbers']['recall'])} | {m['utterances_scored']} "
            f"| {sum(m['excluded'].values())} |"
        )
    return rows


def _slice_table(runs: list[EvaluatedRun]) -> list[str]:
    keys = [
        (column, value) for column, values in runs[0].metrics["slices"].items() for value in values
    ]
    rows = [
        "| slice | utts | " + " | ".join(run.label for run in runs) + " |",
        "| --- | --- | " + " | ".join("---" for _ in runs) + " |",
    ]
    for column, value in keys:
        cells = [_pct(run.metrics["slices"][column].get(value, {}).get("wer")) for run in runs]
        n = runs[0].metrics["slices"][column][value]["utterances"]
        rows.append(f"| {column} = {value} | {n} | " + " | ".join(cells) + " |")
    return rows


def _paired_table(runs: list[EvaluatedRun]) -> list[str]:
    baseline = runs[0]
    rows = [
        "| system | ΔWER (points) | 95% interval | share of resamples not better |",
        "| --- | --- | --- | --- |",
    ]
    base = _included(baseline.scored)
    for run in runs[1:]:
        cand = _included(run.scored)
        if list(base["utterance_id"]) != list(cand["utterance_id"]):
            raise ValueError(
                f"{run.label} was scored on different utterances than {baseline.label}"
            )
        result = paired_bootstrap(word_counts(base), word_counts(cand))
        d = result.delta
        rows.append(
            f"| {run.label} | {100 * d.value:+.2f} | [{100 * d.low:+.2f}, {100 * d.high:+.2f}] "
            f"| {result.share_not_better:.3f} |"
        )
    return rows


def _included(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[scored["excluded"].isna()].sort_values("utterance_id").reset_index(drop=True)


def _pct(estimate: dict[str, float] | None) -> str:
    if not estimate:
        return "n/a"
    return (
        f"{100 * estimate['value']:.2f} [{100 * estimate['low']:.2f}, {100 * estimate['high']:.2f}]"
    )


def _share(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"
