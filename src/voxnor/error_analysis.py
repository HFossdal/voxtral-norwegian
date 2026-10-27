"""Where the word errors are: error regions sorted into kinds.

An error region is a run of consecutive non-matching words in the word
alignment of a reference and a hypothesis. It holds max(reference words,
hypothesis words) word errors, so the regions of an utterance add up to its
WER errors. Each region gets the first kind in `KINDS` that applies.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import jiwer

from voxnor.analysis import is_number_word
from voxnor.variants import fold_variants

KINDS = (
    "spacing",  # same letters, different word split: "helse regionene" / "helseregionene"
    "convention",  # spoken vs standard form (voxnor.variants)
    "number",  # a number word in the reference
    "name",  # a proper name in the reference
    "function word",  # only frequent short words on both sides: "å"/"og", "de"/"det"
    "near miss",  # similar spelling: at most a quarter of the characters differ
    "other",
)


@dataclass(frozen=True)
class ErrorRegion:
    reference: tuple[str, ...]
    hypothesis: tuple[str, ...]
    kind: str

    @property
    def errors(self) -> int:
        return max(len(self.reference), len(self.hypothesis))


def error_regions(reference: str, hypothesis: str) -> list[tuple[list[str], list[str]]]:
    """Runs of non-matching words, as (reference words, hypothesis words)."""
    if not reference.split():
        return [([], hypothesis.split())] if hypothesis.split() else []
    if not hypothesis.split():
        return [(reference.split(), [])]
    out = jiwer.process_words(reference, hypothesis)
    ref, hyp = out.references[0], out.hypotheses[0]
    regions: list[tuple[list[str], list[str]]] = []
    previous_equal = True
    for chunk in out.alignments[0]:
        if chunk.type == "equal":
            previous_equal = True
            continue
        if previous_equal:
            regions.append(([], []))
        regions[-1][0].extend(ref[chunk.ref_start_idx : chunk.ref_end_idx])
        regions[-1][1].extend(hyp[chunk.hyp_start_idx : chunk.hyp_end_idx])
        previous_equal = False
    return regions


def classify(
    ref_words: Sequence[str],
    hyp_words: Sequence[str],
    names: Iterable[str],
    function_words: Iterable[str],
    pairs: Mapping[str, str],
) -> str:
    ref_text, hyp_text = " ".join(ref_words), " ".join(hyp_words)
    names, function_words = set(names), set(function_words)
    if ref_words and hyp_words and ref_text.replace(" ", "") == hyp_text.replace(" ", ""):
        return "spacing"
    if fold_variants(ref_text, pairs) == fold_variants(hyp_text, pairs):
        return "convention"
    if any(is_number_word(w) for w in ref_words):
        return "number"
    if any(w in names for w in ref_words):
        return "name"
    if all(w in function_words for w in (*ref_words, *hyp_words)):
        return "function word"
    if ref_words and hyp_words and _char_distance(ref_text, hyp_text) <= len(ref_text) / 4:
        return "near miss"
    return "other"


def classify_utterance(
    reference: str,
    hypothesis: str,
    names: Iterable[str],
    function_words: Iterable[str],
    pairs: Mapping[str, str],
) -> list[ErrorRegion]:
    return [
        ErrorRegion(tuple(r), tuple(h), classify(r, h, names, function_words, pairs))
        for r, h in error_regions(reference, hypothesis)
    ]


def errors_by_kind(regions: Iterable[ErrorRegion]) -> Counter:
    counts = Counter({kind: 0 for kind in KINDS})
    for region in regions:
        counts[region.kind] += region.errors
    return counts


def frequent_words(references: Iterable[str], n: int = 50, max_length: int = 4) -> set[str]:
    """The `n` most frequent reference words of at most `max_length` letters."""
    counts = Counter(w for text in references for w in text.split() if len(w) <= max_length)
    return {w for w, _ in counts.most_common(n)}


def _char_distance(a: str, b: str) -> int:
    out = jiwer.process_characters(a, b)
    return out.substitutions + out.deletions + out.insertions
