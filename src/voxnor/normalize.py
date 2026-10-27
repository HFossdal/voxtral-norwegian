"""Text normalisation for scoring.

References are the dataset's own `standardized_text`. A model's output is
first brought into the same conventions (`to_reference_conventions`). Then
both sides pass through `score_form`, which folds differences that are not
recognition errors.

`to_reference_conventions(raw_text)` reproduces `standardized_text` for most
of the corpus; `scripts/check_normalizer.py` measures how closely.
"""

from __future__ import annotations

import re
import unicodedata

# Letter folding observed between `raw_text` and `standardized_text`
# in the training data. Norwegian æ, ø, å and é are kept.
LETTER_FOLDS = str.maketrans(
    {"ò": "o", "ô": "o", "ó": "o", "ü": "y", "ö": "ø", "ä": "a", "à": "a", "è": "e", "ç": "s"}
)
KEEP_LETTERS = set("æøåé")

# Hesitation tokens as the references write them (`<ee>` -> `eee`, ...), plus
# common ways a model might write a filled pause. Removed from both sides.
HESITATIONS = frozenset({"eee", "qqq", "mmm", "eh", "ehm", "øh", "øhm", "æh", "hm", "hmm", "mm"})

_HESITATION_TAG = re.compile(r"<(ee|qq|mm)>")
_TAG = re.compile(r"<[^>]*>")
# `spoken|normalised` pairs and `word|1` markers in the raw transcripts.
_ALTERNATIVE = re.compile(r"(\S+?)\|\S+")
# Keep letters, digits and hyphens joining two word characters.
_NON_WORD = re.compile(r"[^\w\s-]|_|(?<!\w)-|-(?!\w)")


def to_reference_conventions(text: str) -> str:
    """Bring free-form text (cased, punctuated) into `standardized_text` conventions.

    Lowercases, writes hesitation tags as the references do (`<ee>` -> `eee`),
    drops other `<...>` tags, keeps the first form of `a|b` pairs, folds
    accented letters as the references do, and removes punctuation except
    word-internal hyphens. Digits are kept as digits.
    """
    text = _HESITATION_TAG.sub(lambda m: f" {m.group(1)[0] * 3} ", text)
    text = _TAG.sub(" ", text)
    text = _ALTERNATIVE.sub(r"\1", text)
    text = unicodedata.normalize("NFC", text.lower()).translate(LETTER_FOLDS)
    text = "".join(_strip_accent(c) for c in text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def score_form(text: str) -> str:
    """The form both reference and hypothesis are scored in.

    `é` -> `e` (an emphasis mark, as in `én`), hyphens -> spaces (compound
    spelling, not recognition), hesitation tokens removed.
    """
    text = text.replace("é", "e").replace("-", " ")
    return " ".join(token for token in text.split() if token not in HESITATIONS)


def normalize_reference(standardized_text: str) -> str:
    return score_form(standardized_text)


def normalize_hypothesis(text: str) -> str:
    return score_form(to_reference_conventions(text))


def _strip_accent(char: str) -> str:
    if char in KEEP_LETTERS or char.isascii():
        return char
    decomposed = unicodedata.normalize("NFD", char)
    base = "".join(c for c in decomposed if not unicodedata.combining(c))
    return base or char
