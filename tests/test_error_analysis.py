from voxnor.error_analysis import (
    classify_utterance,
    error_regions,
    errors_by_kind,
    frequent_words,
)
from voxnor.metrics import word_errors
from voxnor.variants import LEXICAL_PAIRS

FUNCTION = {"å", "og", "de", "det", "en", "den", "er", "har"}


def kinds(ref, hyp, names=()):
    return [r.kind for r in classify_utterance(ref, hyp, names, FUNCTION, LEXICAL_PAIRS)]


def test_regions_add_up_to_word_errors():
    cases = [
        ("a b c d e", "a x c e f g"),
        ("a b c", ""),
        ("a b c", "a b c"),
        ("helse regionene med", "helseregionene med det"),
    ]
    for ref, hyp in cases:
        regions = classify_utterance(ref, hyp, (), FUNCTION, LEXICAL_PAIRS)
        assert sum(r.errors for r in regions) == word_errors(ref, hyp).errors
    assert sum(r.errors for r in classify_utterance("", "a b", (), FUNCTION, {})) == 2


def test_adjacent_edits_form_one_region():
    assert error_regions("a b c d", "a x y d") == [(["b", "c"], ["x", "y"])]
    assert error_regions("a b c d", "a x c y") == [(["b"], ["x"]), (["d"], ["y"])]


def test_kinds():
    assert kinds("helse regionene", "helseregionene") == ["spacing"]
    assert kinds("regjeringa har", "regjeringen har") == ["convention"]
    assert kinds("i totusenogsytten", "i totusenogatten") == ["number"]
    assert kinds("takk olaug bollestad", "takk ole baudestad", names={"olaug", "bollestad"}) == [
        "name"
    ]
    assert kinds("vi skal å se", "vi skal og se") == ["function word"]
    assert kinds("fast ansatte vikarer", "fast ansatte vikarar") == ["near miss"]
    assert kinds("dette er bra", "dette er mange ting") == ["other"]


def test_errors_by_kind_counts_every_kind():
    regions = classify_utterance("vi skal å se", "vi skal og se", (), FUNCTION, LEXICAL_PAIRS)
    counts = errors_by_kind(regions)
    assert counts["function word"] == 1 and counts["other"] == 0


def test_frequent_words_are_short_and_common():
    refs = ["og det er og", "det og en kommunestyre", "og"]
    assert frequent_words(refs, n=2) == {"og", "det"}
