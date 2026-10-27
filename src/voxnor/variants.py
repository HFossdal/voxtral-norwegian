"""Fold spoken and standard Bokmål word forms together, for analysis only.

NPSC references keep spoken forms (`regjeringa`, `sjøl`, `meiner`) where a
model may write the standard form (`regjeringen`, `selv`, `mener`). Both
are Norwegian, so this difference is a convention, not a recognition error.
`fold_variants` maps both forms to one, applied to reference and hypothesis
alike. The headline WER never uses it; it measures how much of a
comparison is convention.

Danish spellings (`præsident`, `netop`) are not folded: they are not
Norwegian.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from voxnor.analysis import is_number_word

# Spoken or optional Bokmål form -> the standard form: optional forms of
# official Bokmål, and spoken forms the NPSC transcripts keep (the most
# frequent such differences in the zero-shot outputs). Pairs that are
# also a real grammatical distinction (`dem`/`de`) are not folded.
LEXICAL_PAIRS: Mapping[str, str] = {
    "sjøl": "selv",
    "fram": "frem",
    "ei": "en",
    "heim": "hjem",
    "heime": "hjemme",
    "heilt": "helt",
    "anna": "annet",
    "veit": "vet",
    "syns": "synes",
    "blei": "ble",
    "meiner": "mener",
    "meine": "mene",
    "meint": "ment",
}

# Feminine definite nouns (`regjeringa`/`regjeringen`) and weak past tense
# (`løfta`/`løftet`): fold `-a`, `-en` and `-et` to one ending. A short stem
# is too likely to be a different word (`da`/`den`, `ha`/`het`).
_A_ENDING = re.compile(r"^(?P<stem>\w{4,}?)(?:a|en|et)$")

# Spoken|normalised pairs in raw transcripts: `kjem|kommer`. `word|1` markers
# are not pairs.
_RAW_PAIR = re.compile(r"(\S+)\|(\S+)")


def corpus_pairs(raw_texts: Iterable[str]) -> dict[str, str]:
    """Spoken -> normalised forms marked in NPSC `raw_text` (`kjem|kommer`).

    Number words are left out (number order, `femogtjue|tjuefem`); scoring keeps
    numbers as written. A spoken form with several normalisations keeps the
    most frequent one.
    """
    counts: dict[str, dict[str, int]] = {}
    for text in raw_texts:
        for spoken, normalised in _RAW_PAIR.findall(text.lower()):
            if normalised.isdigit() or is_number_word(spoken) or spoken == normalised:
                continue
            counts.setdefault(spoken, {}).setdefault(normalised, 0)
            counts[spoken][normalised] += 1
    return {spoken: max(forms, key=forms.get) for spoken, forms in counts.items()}


def fold_variants(text: str, pairs: Mapping[str, str] = LEXICAL_PAIRS) -> str:
    """Map each word to one form for all its spoken and standard variants."""
    return " ".join(_fold_word(word, pairs) for word in text.split())


def _fold_word(word: str, pairs: Mapping[str, str]) -> str:
    word = pairs.get(word, word)
    match = _A_ENDING.match(word)
    return f"{match['stem']}~" if match else word
