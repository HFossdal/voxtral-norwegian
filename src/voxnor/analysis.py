"""Per-utterance scoring and the slices used in the error analysis."""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from typing import Any

import jiwer
import pandas as pd

from voxnor.metrics import ErrorCounts, bootstrap_rate, char_errors, word_errors
from voxnor.normalize import (
    normalize_hypothesis,
    normalize_reference,
    score_form,
    to_reference_conventions,
)

# Norwegian number words are written as one compound ("totusenogsytten",
# "femogtjue"), so a token is a number if it splits entirely into these parts.
_CARDINALS = (
    "null en ett ei én to tre fire fem seks sju syv åtte ni ti elleve tolv tretten fjorten "
    "femten seksten sytten atten nitten tjue tretti tredve førti femti seksti sytti åtti "
    "nitti hundre tusen millioner million milliarder milliard"
).split()
_ORDINALS = (
    "første andre tredje fjerde femte sjette sjuende syvende åttende niende tiende ellevte "
    "tolvte trettende fjortende femtende sekstende syttende attende nittende tjuende "
    "trettiende tredevte førtiende femtiende sekstiende syttiende åttiende nittiende "
    "hundrede tusende"
).split()


def _alternation(words: list[str]) -> str:
    # Longest first, so "seksten" is tried before "seks".
    return "|".join(sorted(words, key=len, reverse=True))


_NUMBER_WORD = re.compile(
    rf"(?:(?:{_alternation(_CARDINALS)})(?:og)?)*(?:{_alternation(_CARDINALS + _ORDINALS)})"
)
# Number words that are far more often something else: articles ("en",
# "ett", "ei") and "andre"/"første" ("other"/"first").
_AMBIGUOUS = frozenset({"en", "ett", "ei", "én", "andre", "første"})

_CAPITALISED = re.compile(r"^[A-ZÆØÅ]")

EXCLUDE_INAUDIBLE = "reference has <INAUDIBLE>"
EXCLUDE_EMPTY = "empty reference after normalisation"
DURATION_BUCKETS = (0, 3, 6, 10, 15.5)


def is_number_word(token: str) -> bool:
    return token not in _AMBIGUOUS and _NUMBER_WORD.fullmatch(token) is not None


def number_tokens(text: str) -> list[str]:
    return [token for token in text.split() if is_number_word(token)]


def name_tokens(raw_text: str) -> set[str]:
    """Scored-form tokens of capitalised words in `raw_text`, except the first word.

    The transcripts capitalise proper names (people, places, parties,
    institutions). The first word is skipped because it is sometimes
    capitalised as a sentence start.
    """
    words = re.sub(r"<[^>]*>", " ", raw_text).split()[1:]
    names = [w for w in words if _CAPITALISED.match(w)]
    return set(score_form(to_reference_conventions(" ".join(names))).split())


def score_utterances(predictions: pd.DataFrame) -> pd.DataFrame:
    """Score each utterance and attach the fields the slices need.

    `predictions` needs: utterance_id, hypothesis, raw_text, standardized_text,
    speaker_id, region, gender, duration, seen_speaker.
    """
    rows = []
    for row in predictions.itertuples(index=False):
        ref = normalize_reference(row.standardized_text)
        hyp = normalize_hypothesis(row.hypothesis)
        excluded = None
        if "<INAUDIBLE>" in row.raw_text:
            excluded = EXCLUDE_INAUDIBLE
        elif not ref:
            excluded = EXCLUDE_EMPTY
        words = word_errors(ref, hyp)
        chars = char_errors(ref, hyp)
        names, numbers = name_tokens(row.raw_text), set(number_tokens(ref))
        name_hits, name_total = _token_recall(ref, hyp, names)
        number_hits, number_total = _token_recall(ref, hyp, numbers)
        rows.append(
            {
                "utterance_id": row.utterance_id,
                "reference": ref,
                "hypothesis": hyp,
                "excluded": excluded,
                "substitutions": words.substitutions,
                "deletions": words.deletions,
                "insertions": words.insertions,
                "ref_words": words.reference_length,
                "char_errors": chars.errors,
                "ref_chars": chars.reference_length,
                "has_number": number_total > 0,
                "hyp_has_digit": bool(re.search(r"\d", hyp)),
                "number_hits": number_hits,
                "number_total": number_total,
                "has_name": name_total > 0,
                "name_hits": name_hits,
                "name_total": name_total,
                "speaker_id": row.speaker_id,
                "region": row.region,
                "gender": row.gender,
                "duration": row.duration,
                "duration_bucket": _bucket(row.duration),
                "seen_speaker": bool(row.seen_speaker),
            }
        )
    return pd.DataFrame(rows)


def word_counts(scored: pd.DataFrame) -> list[ErrorCounts]:
    return [
        ErrorCounts(r.substitutions, r.deletions, r.insertions, r.ref_words)
        for r in scored.itertuples(index=False)
    ]


def char_counts(scored: pd.DataFrame) -> list[ErrorCounts]:
    # Character edit types are not needed separately; all go in substitutions.
    return [ErrorCounts(r.char_errors, 0, 0, r.ref_chars) for r in scored.itertuples(index=False)]


SLICES = ("region", "gender", "duration_bucket", "seen_speaker", "has_number", "has_name")


def summarize(scored: pd.DataFrame, n_resamples: int = 1000, seed: int = 0) -> dict[str, Any]:
    """Headline numbers, slices and exclusions for one system on one split."""
    included = scored[scored["excluded"].isna()]
    summary: dict[str, Any] = {
        "utterances_scored": len(included),
        "excluded": scored["excluded"].value_counts().to_dict(),
        "overall": _rates(included, n_resamples, seed),
        # Recognition without the number-formatting question.
        "number_free": _rates(included[~included["has_number"]], n_resamples, seed),
        "slices": {
            column: {
                str(value): _rates(group, n_resamples, seed)
                for value, group in included.groupby(column, sort=True)
            }
            for column in SLICES
        },
        "names": _recall(included, "name"),
        "numbers": {
            **_recall(included, "number"),
            "utterances_with_digits_in_hypothesis": int(included["hyp_has_digit"].sum()),
        },
    }
    return summary


def _rates(df: pd.DataFrame, n_resamples: int, seed: int) -> dict[str, Any]:
    if df.empty or df["ref_words"].sum() == 0:
        return {"utterances": len(df), "wer": None, "cer": None}
    words = word_counts(df)
    return {
        "utterances": len(df),
        "ref_words": int(df["ref_words"].sum()),
        "wer": bootstrap_rate(words, n_resamples=n_resamples, seed=seed).as_dict(),
        "cer": bootstrap_rate(char_counts(df), n_resamples=n_resamples, seed=seed).as_dict(),
        "substitutions": int(df["substitutions"].sum()),
        "deletions": int(df["deletions"].sum()),
        "insertions": int(df["insertions"].sum()),
    }


def _recall(df: pd.DataFrame, kind: str) -> dict[str, Any]:
    total = int(df[f"{kind}_total"].sum())
    hits = int(df[f"{kind}_hits"].sum())
    return {"tokens": total, "recognised": hits, "recall": hits / total if total else None}


def _token_recall(reference: str, hypothesis: str, targets: Iterable[str]) -> tuple[int, int]:
    """(hits, total) over reference tokens in `targets`, using the WER alignment."""
    targets = set(targets)
    ref_tokens = reference.split()
    positions = [i for i, token in enumerate(ref_tokens) if token in targets]
    if not positions:
        return 0, 0
    if not hypothesis:
        return 0, len(positions)
    equal: set[int] = set()
    for chunk in jiwer.process_words(reference, hypothesis).alignments[0]:
        if chunk.type == "equal":
            equal.update(range(chunk.ref_start_idx, chunk.ref_end_idx))
    return sum(i in equal for i in positions), len(positions)


def _bucket(duration: float) -> str:
    for low, high in itertools.pairwise(DURATION_BUCKETS):
        if low <= duration < high:
            return f"{low}-{high:g}s" if high != DURATION_BUCKETS[-1] else f"{low}s+"
    return f"{DURATION_BUCKETS[-2]}s+"
