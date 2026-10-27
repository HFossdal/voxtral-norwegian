"""Word and character error rates with bootstrap confidence intervals.

Error rates are corpus-level: total errors over total reference words, not a
mean of per-utterance rates. Confidence intervals resample utterances, which
is the unit the test set was sampled in. Before/after comparisons use a
paired bootstrap over the same resampled utterances.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import jiwer
import numpy as np


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int
    reference_length: int

    @property
    def errors(self) -> int:
        return self.substitutions + self.deletions + self.insertions


def word_errors(reference: str, hypothesis: str) -> ErrorCounts:
    out = jiwer.process_words(reference, hypothesis)
    return ErrorCounts(out.substitutions, out.deletions, out.insertions, len(reference.split()))


def char_errors(reference: str, hypothesis: str) -> ErrorCounts:
    """Character edits, counting spaces, on the normalised text."""
    out = jiwer.process_characters(reference, hypothesis)
    return ErrorCounts(out.substitutions, out.deletions, out.insertions, len(reference))


def error_rate(counts: Sequence[ErrorCounts]) -> float:
    total = sum(c.reference_length for c in counts)
    if total == 0:
        raise ValueError("Error rate is undefined for an empty reference set")
    return sum(c.errors for c in counts) / total


@dataclass(frozen=True)
class Estimate:
    value: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"value": self.value, "low": self.low, "high": self.high}


def bootstrap_rate(
    counts: Sequence[ErrorCounts], n_resamples: int = 1000, level: float = 0.95, seed: int = 0
) -> Estimate:
    """Corpus error rate with a percentile bootstrap interval over utterances."""
    errors, lengths = _arrays(counts)
    idx = np.random.default_rng(seed).integers(0, len(errors), size=(n_resamples, len(errors)))
    rates = errors[idx].sum(axis=1) / lengths[idx].sum(axis=1)
    low, high = np.quantile(rates, [(1 - level) / 2, (1 + level) / 2])
    return Estimate(error_rate(counts), float(low), float(high))


@dataclass(frozen=True)
class PairedComparison:
    baseline: float
    candidate: float
    # candidate - baseline: negative means the candidate makes fewer errors.
    delta: Estimate
    # Share of resamples in which the candidate is not better than the baseline.
    share_not_better: float

    def as_dict(self) -> dict:
        return {
            "baseline": self.baseline,
            "candidate": self.candidate,
            "delta": self.delta.as_dict(),
            "share_not_better": self.share_not_better,
        }


def paired_bootstrap(
    baseline: Sequence[ErrorCounts],
    candidate: Sequence[ErrorCounts],
    n_resamples: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Compare two systems scored on the same utterances, in the same order."""
    if len(baseline) != len(candidate):
        raise ValueError("Both systems must be scored on the same utterances")
    base_errors, lengths = _arrays(baseline)
    cand_errors, cand_lengths = _arrays(candidate)
    if not np.array_equal(lengths, cand_lengths):
        raise ValueError("Reference lengths differ: the utterances are not the same")

    idx = np.random.default_rng(seed).integers(0, len(lengths), size=(n_resamples, len(lengths)))
    total = lengths[idx].sum(axis=1)
    deltas = (cand_errors[idx].sum(axis=1) - base_errors[idx].sum(axis=1)) / total
    low, high = np.quantile(deltas, [(1 - level) / 2, (1 + level) / 2])
    base_rate, cand_rate = error_rate(baseline), error_rate(candidate)
    return PairedComparison(
        baseline=base_rate,
        candidate=cand_rate,
        delta=Estimate(cand_rate - base_rate, float(low), float(high)),
        share_not_better=float((deltas >= 0).mean()),
    )


def _arrays(counts: Sequence[ErrorCounts]) -> tuple[np.ndarray, np.ndarray]:
    if not counts:
        raise ValueError("No utterances to score")
    errors = np.array([c.errors for c in counts], dtype=np.int64)
    lengths = np.array([c.reference_length for c in counts], dtype=np.int64)
    if lengths.sum() == 0:
        raise ValueError("Error rate is undefined for an empty reference set")
    return errors, lengths
