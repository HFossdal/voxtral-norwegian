"""Error analysis of a fine-tuned model against Mini `nb` on validation.

    python scripts/analyse_finetune.py --config configs/local.yaml \
        --mini reports/runs/<mini-nb run> --tuned reports/runs/<tuned eval run> \
        --training reports/runs/<fine-tuning run>

Writes reports/finetune_analysis.md and figures in reports/figures/:
  1. the comparison with spoken/standard forms folded together, on
     number-free utterances (convention removed)
  2. error kinds on the selection set at every evaluated training step
  3. error kinds on full validation, with the most frequent examples
  4. WER per speaker for both models
Reads only metadata (no audio) and the run directories. Validation only.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from voxnor.analysis import name_tokens, word_counts
from voxnor.config import add_override_argument, load_config
from voxnor.data import load_metadata
from voxnor.error_analysis import KINDS, classify_utterance, errors_by_kind, frequent_words
from voxnor.evaluate import read_predictions
from voxnor.metrics import ErrorCounts, bootstrap_rate, paired_bootstrap, word_errors
from voxnor.normalize import normalize_hypothesis
from voxnor.report import load_run
from voxnor.variants import LEXICAL_PAIRS, corpus_pairs, fold_variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    add_override_argument(parser)
    parser.add_argument("--mini", type=Path, required=True)
    parser.add_argument("--tuned", type=Path, required=True)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/finetune_analysis.md"))
    parser.add_argument("--figures", type=Path, default=Path("reports/figures"))
    args = parser.parse_args()

    cfg = load_config(args.config, args.set)
    val_meta = load_metadata(cfg.data, "validation")
    train_meta = load_metadata(cfg.data, "train")
    pairs = {**corpus_pairs([*train_meta.raw_text, *val_meta.raw_text]), **LEXICAL_PAIRS}
    names = {
        u: name_tokens(t) for u, t in zip(val_meta.utterance_id, val_meta.raw_text, strict=True)
    }

    tuned_run = load_run(args.tuned, "validation")
    mini, tuned = (_included(r.scored) for r in (load_run(args.mini, "validation"), tuned_run))
    if list(mini.utterance_id) != list(tuned.utterance_id):
        raise ValueError("The two runs must score the same utterances in the same order")
    function_words = frequent_words(train_meta.standardized_text)
    vocabulary = {w for t in train_meta.standardized_text for w in t.split()}
    vocabulary |= {w for t in val_meta.standardized_text for w in t.split()}

    args.figures.mkdir(parents=True, exist_ok=True)
    sections = [
        "# Fine-tuned model vs Mini `nb`: error analysis (validation)",
        "",
        f"Fine-tuned model: {tuned_run.label}.",
        "",
        "Made by `scripts/analyse_finetune.py`; method and caveats in "
        "`reports/writeup.md`. Error rates are corpus-level, intervals "
        "are 95% bootstrap intervals over utterances.",
        "",
        *convention_section(mini, tuned, pairs),
        *trajectory_section(
            args.training, tuned, names, function_words, pairs, vocabulary, args.figures
        ),
        *kinds_section(mini, tuned, names, function_words, pairs),
        *speaker_section(mini, tuned, args.figures),
    ]
    args.out.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {args.out}")


def convention_section(
    mini: pd.DataFrame,
    tuned: pd.DataFrame,
    pairs: dict[str, str],
    labels: tuple[str, str] = ("Mini `nb`", "fine-tuned"),
    heading: str = "## 1. With the spelling convention removed",
):
    """WER of both runs with numbers, then spoken forms, taken out;
    also used by compare_validation.py."""
    a, b = labels
    number_free = ~_flag(mini.has_number) & ~_flag(tuned.has_number)
    rows = [
        ("all utterances, official scoring", slice(None), False),
        ("number-free utterances", number_free, False),
        ("number-free, spoken/standard forms folded", number_free, True),
    ]
    lines = [
        heading,
        "",
        "Numbers: only utterances without number words in the reference. "
        "Spoken forms: `voxnor.variants` folds spoken and standard Bokmål "
        "forms together on both sides (feminine and past-tense `-a`, `sjøl`/"
        "`selv`, `meiner`/`mener`, and the spoken|normalised pairs marked in "
        "the NPSC transcripts). Danish spellings are not folded.",
        "",
        f"| scoring | utts | {a} | {b} | {b} minus {a} | resamples where {b} is not better |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for label, rows_mask, fold in rows:
        m, t = mini[rows_mask], tuned[rows_mask]
        mc, tc = (_counts(d, pairs if fold else None) for d in (m, t))
        cmp = paired_bootstrap(mc, tc)
        lines.append(
            f"| {label} | {len(m)} | {_pct(bootstrap_rate(mc))} | {_pct(bootstrap_rate(tc))} | "
            f"{_pct(cmp.delta, sign='+')} | {cmp.share_not_better:.3f} |"
        )
    refs = [w for r in mini[number_free].reference for w in r.split()]
    changed = sum(fold_variants(w, pairs) != w for w in refs)
    lines += [
        "",
        f"Folding changes {changed / len(refs):.1%} of the reference words in the "
        "number-free utterances.",
        "",
    ]
    return lines


def trajectory_section(training, tuned, names, function_words, pairs, vocabulary, figures):
    refs = dict(zip(tuned.utterance_id, tuned.reference, strict=True))
    selection = sorted((training / "selection").glob("predictions_step-*.jsonl"))
    table = []
    for path in selection:
        step = int(path.stem.split("-")[-1])
        preds = read_predictions(path)
        preds = preds[preds.utterance_id.isin(refs)]
        hyps = [normalize_hypothesis(h) for h in preds.hypothesis]
        ref_list = [refs[u] for u in preds.utterance_id]
        n_ref = sum(len(r.split()) for r in ref_list)
        counts = errors_by_kind(
            region
            for u, r, h in zip(preds.utterance_id, ref_list, hyps, strict=True)
            for region in classify_utterance(r, h, names[u], function_words, pairs)
        )
        hyp_words = [w for h in hyps for w in h.split()]
        table.append(
            {
                "step": step,
                "utts": len(hyps),
                "wer": sum(counts.values()) / n_ref * 100,
                **{k: counts[k] / n_ref * 100 for k in KINDS},
                "empty": sum(not h for h in hyps) / len(hyps) * 100,
                "unknown words": sum(w not in vocabulary for w in hyp_words)
                / max(len(hyp_words), 1)
                * 100,
            }
        )
    df = pd.DataFrame(table)
    lines = [
        "## 2. What training fixes first (selection set)",
        "",
        f"The {df.utts.iloc[0]} scored selection utterances at every evaluated step. "
        "Error kinds in errors per 100 reference words; they add up to the WER. "
        "*Unknown words*: share of output words that never occur in a train or "
        "validation transcript (made-up words, other languages).",
        "",
        "| step | WER | " + " | ".join(KINDS) + " | empty outputs | unknown words |",
        "| --- " * (len(KINDS) + 4) + "|",
    ]
    for r in table:
        lines.append(
            f"| {r['step']} | {r['wer']:.1f} | "
            + " | ".join(f"{r[k]:.1f}" for k in KINDS)
            + f" | {r['empty']:.1f}% | {r['unknown words']:.1f}% |"
        )
    lines += ["", "![Error kinds by training step](figures/selection_error_kinds.png)", ""]
    _plot_trajectory(df[df.step > 0], figures / "selection_error_kinds.png")
    return lines


def kinds_section(mini, tuned, names, function_words, pairs):
    n_ref = tuned.ref_words.sum()
    per_model = {}
    for label, d in (("Mini `nb`", mini), ("fine-tuned", tuned)):
        per_model[label] = [
            region
            for u, r, h in zip(d.utterance_id, d.reference, d.hypothesis, strict=True)
            for region in classify_utterance(r, h, names[u], function_words, pairs)
        ]
    counts = {label: errors_by_kind(regions) for label, regions in per_model.items()}
    lines = [
        "## 3. Error kinds on full validation",
        "",
        "Errors per 100 reference words, by the first kind that applies (`voxnor.error_analysis`).",
        "",
        "| kind | Mini `nb` | fine-tuned | difference |",
        "| --- | --- | --- | --- |",
    ]
    for kind in KINDS:
        m, t = (counts[label][kind] / n_ref * 100 for label in counts)
        lines.append(f"| {kind} | {m:.2f} | {t:.2f} | {t - m:+.2f} |")
    m, t = (sum(c.values()) / n_ref * 100 for c in counts.values())
    lines += [f"| **all (WER)** | {m:.2f} | {t:.2f} | {t - m:+.2f} |", ""]
    for label, regions in per_model.items():
        lines += [f"Most frequent error regions, {label}:", ""]
        lines += ["| kind | reference → output (count) |", "| --- | --- |"]
        for kind in KINDS:
            top = Counter(
                (" ".join(r.reference) or "∅", " ".join(r.hypothesis) or "∅")
                for r in regions
                if r.kind == kind
            ).most_common(6)
            lines.append(
                f"| {kind} | " + ", ".join(f"*{a}* → *{b}* ({n})" for (a, b), n in top) + " |"
            )
        lines.append("")
    return lines


def speaker_section(mini, tuned, figures):
    both = tuned[["speaker_id", "region", "seen_speaker", "ref_words"]].copy()
    both["e_tuned"] = tuned.substitutions + tuned.deletions + tuned.insertions
    both["e_mini"] = (mini.substitutions + mini.deletions + mini.insertions).to_numpy()
    sp = (
        both.groupby(["region", "speaker_id", "seen_speaker"])
        .agg(words=("ref_words", "sum"), e_tuned=("e_tuned", "sum"), e_mini=("e_mini", "sum"))
        .reset_index()
    )
    sp["mini"] = sp.e_mini / sp.words * 100
    sp["tuned"] = sp.e_tuned / sp.words * 100
    sp = sp.sort_values(["region", "words"], ascending=[True, False])
    lines = [
        "## 4. WER per speaker",
        "",
        f"{len(sp)} speakers; the fine-tuned model is better for {(sp.tuned < sp.mini).sum()}. "
        "*Seen*: the speaker also occurs in the training data (validation "
        "speakers were not removed from training).",
        "",
        "![WER per speaker](figures/speaker_wer.png)",
        "",
        "| region | speaker | seen | reference words | Mini `nb` | fine-tuned |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in sp.itertuples(index=False):
        lines.append(
            f"| {r.region} | {r.speaker_id} | {'yes' if str(r.seen_speaker) == 'True' else 'no'} | "
            f"{r.words} | {r.mini:.1f} | {r.tuned:.1f} |"
        )
    _plot_speakers(sp, figures / "speaker_wer.png")
    return [*lines, ""]


def _plot_trajectory(df: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    bottom = pd.Series(0.0, index=df.index)
    for kind in KINDS:
        ax.bar(df.step.astype(str), df[kind], bottom=bottom, label=kind)
        bottom += df[kind]
    ax.set_xlabel("training step")
    ax.set_ylabel("errors per 100 reference words")
    ax.set_title("Selection set: error kinds after fine-tuning starts")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_speakers(sp: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # The `unknown` region is one speaker with 2 utterances; it would stretch the axes.
    sp = sp[sp.region != "unknown"]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for region, group in sp.groupby("region"):
        ax.scatter(group.mini, group.tuned, s=group.words / 40 + 5, label=region, alpha=0.7)
    limit = max(sp.mini.max(), sp.tuned.max()) + 2
    ax.plot([0, limit], [0, limit], color="grey", lw=1)
    ax.set_xlim(0, limit)
    ax.set_ylim(0, limit)
    ax.set_xlabel("Mini nb WER (%)")
    ax.set_ylabel("fine-tuned WER (%)")
    ax.set_title("WER per validation speaker (size: words)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _included(scored: pd.DataFrame) -> pd.DataFrame:
    return scored[scored.excluded.isna()].reset_index(drop=True)


def _flag(column: pd.Series) -> pd.Series:
    return column.astype(str) == "True"


def _counts(d: pd.DataFrame, pairs: dict[str, str] | None) -> list[ErrorCounts]:
    if pairs is None:
        return word_counts(d)
    return [
        word_errors(fold_variants(r, pairs), fold_variants(h, pairs))
        for r, h in zip(d.reference, d.hypothesis, strict=True)
    ]


def _pct(estimate, sign: str = "") -> str:
    v, lo, hi = (x * 100 for x in (estimate.value, estimate.low, estimate.high))
    return f"{v:{sign}.2f} [{lo:{sign}.2f}, {hi:{sign}.2f}]"


if __name__ == "__main__":
    main()
