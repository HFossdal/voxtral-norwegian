import pytest

from voxnor.metrics import (
    ErrorCounts,
    bootstrap_rate,
    char_errors,
    error_rate,
    paired_bootstrap,
    word_errors,
)


def test_word_errors_count_each_edit_type():
    counts = word_errors("a b c d", "a x c d e")
    assert (counts.substitutions, counts.deletions, counts.insertions) == (1, 0, 1)
    assert counts.reference_length == 4
    assert counts.errors == 2


def test_empty_hypothesis_is_all_deletions():
    assert word_errors("a b", "") == ErrorCounts(0, 2, 0, 2)


def test_char_errors_count_spaces():
    assert char_errors("ab c", "abc").errors == 1
    assert char_errors("ab c", "abc").reference_length == 4


def test_error_rate_is_corpus_level_not_mean_of_utterances():
    # Per-utterance rates 1.0 and 0.0 average to 0.5; corpus WER is 1 / 10.
    counts = [word_errors("a", "b"), word_errors("a b c d e f g h i", "a b c d e f g h i")]
    assert error_rate(counts) == pytest.approx(0.1)


def test_error_rate_can_exceed_one():
    assert error_rate([word_errors("a", "b c d")]) == 3.0


def test_error_rate_rejects_empty_references():
    with pytest.raises(ValueError):
        error_rate([ErrorCounts(0, 0, 1, 0)])


def make_counts(errors, length=10):
    return [ErrorCounts(e, 0, 0, length) for e in errors]


def test_bootstrap_interval_contains_estimate_and_is_reproducible():
    counts = make_counts([0, 1, 2, 3, 4, 5, 1, 2, 0, 3] * 20)
    est = bootstrap_rate(counts, seed=1)
    assert est.low <= est.value <= est.high
    assert est == bootstrap_rate(counts, seed=1)
    assert est.high - est.low < 0.1


def test_bootstrap_interval_is_degenerate_without_variance():
    est = bootstrap_rate(make_counts([2] * 50))
    assert est.value == est.low == est.high == pytest.approx(0.2)


def test_paired_bootstrap_detects_a_clear_improvement():
    baseline = make_counts([3, 4, 2, 5, 3] * 40)
    candidate = make_counts([1, 2, 1, 2, 1] * 40)
    result = paired_bootstrap(baseline, candidate)
    assert result.delta.value == pytest.approx(error_rate(candidate) - error_rate(baseline))
    assert result.delta.high < 0
    assert result.share_not_better == 0.0


def test_paired_bootstrap_sees_no_difference_between_identical_systems():
    counts = make_counts([0, 1, 2, 3] * 25)
    result = paired_bootstrap(counts, counts)
    assert result.delta.value == result.delta.low == result.delta.high == 0
    assert result.share_not_better == 1.0


def test_paired_bootstrap_requires_the_same_utterances():
    with pytest.raises(ValueError):
        paired_bootstrap(make_counts([1, 2]), make_counts([1]))
    with pytest.raises(ValueError):
        paired_bootstrap(make_counts([1, 2], length=10), make_counts([1, 2], length=5))
