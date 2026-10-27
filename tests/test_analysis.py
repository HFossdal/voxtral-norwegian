import pandas as pd
import pytest

from voxnor.analysis import (
    EXCLUDE_EMPTY,
    EXCLUDE_INAUDIBLE,
    is_number_word,
    name_tokens,
    number_tokens,
    score_utterances,
    summarize,
)


@pytest.mark.parametrize(
    "token",
    [
        "to",
        "tjuesytten",
        "totusenogsytten",
        "femogtjue",
        "hundre",
        "tredje",
        "tjueførste",
        "nittennittini",
        "millioner",
        "null",
    ],
)
def test_number_words(token):
    assert is_number_word(token)


@pytest.mark.parametrize(
    "token",
    [
        "en",
        "ett",
        "ei",
        "andre",
        "første",
        "og",
        "tusenvis",
        "toget",
        "prosent",
        "kroner",
        "tre1",
        "",
    ],
)
def test_not_number_words(token):
    assert not is_number_word(token)


def test_morpheme_order_is_not_checked():
    # Known limitation: any sequence of number parts is accepted. No such
    # non-number occurs in the corpus vocabulary (checked by listing all matches).
    assert is_number_word("tito")


def test_number_tokens_finds_compounds_in_running_text():
    assert number_tokens("i totusenogsytten fikk to av dem en pris") == ["totusenogsytten", "to"]


def test_name_tokens_uses_capitals_except_the_first_word():
    raw = "Stortingets møte er satt og Erna Solberg <ee> fra Vest-Agder og KrF"
    assert name_tokens(raw) == {"erna", "solberg", "vest", "agder", "krf"}


def prediction(uid, hypothesis, standardized, raw=None, **extra):
    row = {
        "utterance_id": uid,
        "hypothesis": hypothesis,
        "standardized_text": standardized,
        "raw_text": raw if raw is not None else standardized,
        "speaker_id": "s1",
        "region": "east",
        "gender": "female",
        "duration": 4.0,
        "seen_speaker": False,
    }
    row.update(extra)
    return row


def test_score_utterances_normalises_both_sides_and_flags_exclusions():
    scored = score_utterances(
        pd.DataFrame(
            [
                prediction("1", "Hei, Erna Solberg!", "hei erna solberg", raw="hei Erna Solberg"),
                prediction("2", "statsminister", "statsminister", raw="statsminister <INAUDIBLE>"),
                prediction("3", "", "mmm", raw="<mm>"),
                prediction("4", "I 2017 kom to.", "i totusenogsytten kom to"),
            ]
        )
    ).set_index("utterance_id")

    assert scored.loc["1", ["substitutions", "deletions", "insertions"]].tolist() == [0, 0, 0]
    assert scored.loc["1", "name_total"] == 2 and scored.loc["1", "name_hits"] == 2
    assert scored.loc["2", "excluded"] == EXCLUDE_INAUDIBLE
    assert scored.loc["3", "excluded"] == EXCLUDE_EMPTY
    # "2017" is one substitution for "totusenogsytten"; "to" is still recognised.
    assert scored.loc["4", "substitutions"] == 1
    assert scored.loc["4", "number_total"] == 2 and scored.loc["4", "number_hits"] == 1
    assert scored.loc["4", "hyp_has_digit"]


def test_summarize_excludes_flagged_utterances_and_separates_number_free():
    scored = score_utterances(
        pd.DataFrame(
            [
                prediction("1", "a b c d", "a b c d", region="east"),
                prediction("2", "a x c d", "a b c d", region="west"),
                prediction("3", "x x", "a b", raw="a b <INAUDIBLE>"),
                prediction("4", "i 2017", "i totusenogsytten"),
            ]
        )
    )
    summary = summarize(scored, n_resamples=50)
    assert summary["utterances_scored"] == 3
    assert summary["excluded"] == {EXCLUDE_INAUDIBLE: 1}
    assert summary["overall"]["wer"]["value"] == pytest.approx(2 / 10)
    assert summary["number_free"]["wer"]["value"] == pytest.approx(1 / 8)
    assert summary["slices"]["region"]["west"]["wer"]["value"] == pytest.approx(1 / 4)
    assert summary["numbers"]["utterances_with_digits_in_hypothesis"] == 1
