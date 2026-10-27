"""Extra training clips from the Stortinget Speech Corpus (SSC).

SSC segments (up to 30 s) come with two transcripts: the official
proceedings (edited, with digits and punctuation) and a wav2vec2 ASR
transcript (NPSC conventions, with recognition errors). Neither is verbatim.
Where both agree on a run of words, the words were very probably said as
written, so only those runs are kept:

1. normalise both transcripts and find runs of at least `min_run`
   agreeing words;
2. force-align the whole ASR transcript to the segment audio (`voxnor.align`);
3. cut each run into clips of at most `max_seconds`, at the widest pauses,
   with silence on both sides, so no word outside the run is in the clip.

The clip text is the ASR words of the run, hesitations included, because
those follow NPSC's `standardized_text` conventions.
"""

from __future__ import annotations

import difflib
import io
import re
import unicodedata
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import soundfile as sf

from voxnor.align import CtcAligner, WordTiming
from voxnor.data import SAMPLE_RATE
from voxnor.normalize import normalize_hypothesis

MIN_RUN = 8
MAX_SECONDS = 14.0
MIN_WORDS = 4
# A clip edge needs this much silence between the clip's word and the
# neighbouring word outside it; otherwise the edge word is dropped. 0.06 s is
# three aligner frames: set with scripts/tune_ssc_cuts.py.
MIN_EDGE_GAP = 0.06
PAD = 0.2

# Why a segment gives no clips. Counted per shard.
SKIP_SPLIT = "not in SSC's train split"
SKIP_SPEAKERS = "not exactly one speaker"
SKIP_LANGUAGE = "not Bokmål"
SKIP_HELD_OUT = "held-out speaker"
SKIP_DATE = "NPSC meeting date"
SKIP_NO_RUN = "no agreeing run"
SKIP_NOT_ALIGNED = "alignment failed"
SKIP_NO_CLIP = "no clip fits"

GENDERS = {"M": "male", "F": "female"}

# The columns of npsc_nb, so clips can join its training split.
CLIP_SCHEMA = pa.schema(
    [
        ("speaker_id", pa.string()),
        ("gender", pa.string()),
        ("utterance_id", pa.string()),
        ("language", pa.string()),
        ("raw_text", pa.string()),
        ("full_audio_file", pa.string()),
        ("original_data_split", pa.string()),
        ("region", pa.string()),
        ("duration", pa.float64()),
        ("start", pa.float64()),
        ("end", pa.float64()),
        ("utterance_audio_file", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("standardized_text", pa.string()),
    ]
)

# NPSC speakers whose surname changed before ParlaMint-NO was made; the
# ParlaMint IDs still carry the old initials.
RENAMED = {"Else-May Botten": "person.EBO", "Marianne Synnes": "person.MS"}


# --- speakers -------------------------------------------------------------


def _name_words(name: str) -> list[str]:
    name = unicodedata.normalize("NFC", name).lower()
    return re.sub(r"[^\w ]|_|\d", " ", name).split()


@dataclass(frozen=True)
class Person:
    id: str
    forenames: tuple[str, ...]
    surnames: tuple[str, ...]


def read_parlamint_people(path: str | Path) -> list[Person]:
    """Every name of every person in a ParlaMint `listPerson` TEI file."""
    ns = {"t": "http://www.tei-c.org/ns/1.0"}
    xml_id = "{http://www.w3.org/XML/1998/namespace}id"
    people = []
    for person in ET.parse(path).getroot().findall("t:person", ns):
        for name in person.findall("t:persName", ns):
            fore = " ".join(n.text or "" for n in name.findall("t:forename", ns))
            sur = " ".join(n.text or "" for n in name.findall("t:surname", ns))
            people.append(
                Person(person.get(xml_id), tuple(_name_words(fore)), tuple(_name_words(sur)))
            )
    return people


def match_speaker(name: str, people: Sequence[Person]) -> str | None:
    """ParlaMint ID of an NPSC speaker name, or None if not exactly one person fits.

    Exact full-name match first; otherwise the first forename and the last
    surname word (NPSC sometimes leaves out middle names or hyphens).
    """
    if name in RENAMED:
        return RENAMED[name]
    words = _name_words(name)
    if not words:
        return None
    exact = {p.id for p in people if list(p.forenames + p.surnames) == words}
    if len(exact) == 1:
        return exact.pop()
    if exact:
        return None
    loose = {
        p.id
        for p in people
        if p.forenames and p.surnames and p.forenames[0] == words[0] and p.surnames[-1] == words[-1]
    }
    return loose.pop() if len(loose) == 1 else None


# --- agreement and clips --------------------------------------------------


def agreeing_runs(
    proceedings: str, asr_words: Sequence[str], min_run: int
) -> list[tuple[int, int]]:
    """Inclusive spans of ASR word indices where both transcripts agree on >= `min_run` words.

    Both sides are compared in score form, so case, punctuation
    and hesitations do not matter; a hesitation inside a run stays in it.
    """
    tokens, word_of = [], []
    for i, word in enumerate(asr_words):
        for token in normalize_hypothesis(word).split():
            tokens.append(token)
            word_of.append(i)
    reference = normalize_hypothesis(proceedings).split()
    matcher = difflib.SequenceMatcher(None, reference, tokens, autojunk=False)
    runs = []
    for block in matcher.get_matching_blocks():
        if block.size < min_run:
            continue
        lo, hi = block.b, block.b + block.size - 1
        first, last = word_of[lo], word_of[hi]
        # A word split into several tokens (a hyphen) counts only if all agree.
        if lo > 0 and word_of[lo - 1] == first:
            first += 1
        if hi + 1 < len(word_of) and word_of[hi + 1] == last:
            last -= 1
        if last >= first:
            runs.append((first, last))
    return runs


def split_run(
    timings: Sequence[WordTiming], first: int, last: int, max_span: float
) -> list[tuple[int, int]]:
    """Cut words `first..last` into pieces spanning at most `max_span` seconds.

    Each cut goes at the widest pause in the second half of the piece.
    A single word longer than `max_span` is dropped.
    """
    pieces, start = [], first
    while start <= last:
        if timings[last].end - timings[start].start <= max_span:
            pieces.append((start, last))
            break
        begin = timings[start].start
        fits = [k for k in range(start, last + 1) if timings[k].end - begin <= max_span]
        if not fits:
            start += 1
            continue
        end = fits[-1]
        cut = max(
            range(start + (end - start) // 2, end + 1),
            key=lambda k: timings[k + 1].start - timings[k].end,
        )
        pieces.append((start, cut))
        start = cut + 1
    return pieces


@dataclass(frozen=True)
class ClipPlan:
    first: int  # ASR word indices, inclusive
    last: int
    start: float  # seconds in the segment
    end: float


def plan_clips(
    timings: Sequence[WordTiming],
    runs: Iterable[tuple[int, int]],
    seconds: float,
    max_seconds: float = MAX_SECONDS,
    min_words: int = MIN_WORDS,
    min_edge_gap: float = MIN_EDGE_GAP,
    pad: float = PAD,
) -> list[ClipPlan]:
    """Clip boundaries for agreeing runs, given word timings of all ASR words."""
    n, clips = len(timings), []
    for run_first, run_last in runs:
        for first, last in split_run(timings, run_first, run_last, max_seconds - 2 * pad):
            while first <= last and first > 0 and _gap(timings, first - 1) < min_edge_gap:
                first += 1
            while last >= first and last < n - 1 and _gap(timings, last) < min_edge_gap:
                last -= 1
            if last - first + 1 < min_words:
                continue
            start = max(0.0, timings[first].start - pad)
            if first > 0:
                start = max(start, (timings[first - 1].end + timings[first].start) / 2)
            end = min(seconds, timings[last].end + pad)
            if last < n - 1:
                end = min(end, (timings[last].end + timings[last + 1].start) / 2)
            clips.append(ClipPlan(first, last, round(start, 3), round(end, 3)))
    return clips


def _gap(timings: Sequence[WordTiming], k: int) -> float:
    """Silence between word k and word k + 1."""
    return timings[k + 1].start - timings[k].end


# --- segments -------------------------------------------------------------


def skip_reason(meta: dict, held_out: set[str], excluded_dates: set[str]) -> str | None:
    """Why an SSC segment (its JSON record) is not used, or None if it is a candidate."""
    if meta["split"] != "train":
        return SKIP_SPLIT
    speakers = meta.get("speakers") or []
    if meta.get("num_speakers") != 1 or len(speakers) != 1:
        return SKIP_SPEAKERS
    if speakers[0].get("language") != "nob":
        return SKIP_LANGUAGE
    if speakers[0].get("speaker_id") in held_out:
        return SKIP_HELD_OUT
    if meta["meeting_date"] in excluded_dates:
        return SKIP_DATE
    return None


def encode_wav(audio: np.ndarray) -> bytes:
    out = io.BytesIO()
    sf.write(out, audio, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return out.getvalue()


def _text(value) -> str | None:
    # Missing speaker fields come as None or as NaN (a float).
    return value if isinstance(value, str) and value else None


def segment_clips(
    meta: dict, audio: np.ndarray, aligner: CtcAligner, min_run: int = MIN_RUN
) -> tuple[list[dict], list[dict], str | None]:
    """Clip rows (NPSC columns) and their alignment records for one candidate segment.

    Returns the skip reason instead when no clip comes out of it.
    """
    asr_words = meta["transcription_text"].split()
    runs = agreeing_runs(meta["proceedings_text"], asr_words, min_run)
    if not runs:
        return [], [], SKIP_NO_RUN
    result = aligner.align_with_score(audio, asr_words)
    if result is None:
        return [], [], SKIP_NOT_ALIGNED
    timings, log_prob = result
    plans = plan_clips(timings, runs, len(audio) / SAMPLE_RATE)
    if not plans:
        return [], [], SKIP_NO_CLIP

    speaker = meta["speakers"][0]
    rows, records = [], []
    for k, plan in enumerate(plans):
        uid = f"ssc_{meta['segment_id']}_{k}"
        clip = audio[round(plan.start * SAMPLE_RATE) : round(plan.end * SAMPLE_RATE)]
        words = [
            [t.word, round(t.start - plan.start, 3), round(t.end - plan.start, 3)]
            for t in timings[plan.first : plan.last + 1]
        ]
        text = " ".join(w for w, _, _ in words)
        rows.append(
            {
                "speaker_id": speaker["speaker_id"],
                "gender": GENDERS.get(_text(speaker.get("gender")), "unknown"),
                "utterance_id": uid,
                "language": "nb-NO",
                "raw_text": text,
                "full_audio_file": meta["audio_path"],
                "original_data_split": "ssc_train",
                "region": _text(speaker.get("dialect")) or "unknown",
                "duration": round(len(clip) / SAMPLE_RATE, 3),
                "start": plan.start,
                "end": plan.end,
                "utterance_audio_file": {"bytes": encode_wav(clip), "path": f"{uid}.wav"},
                "standardized_text": text,
            }
        )
        records.append(
            {
                "utterance_id": uid,
                "n_samples": len(clip),
                "words": words,
                "log_prob": round(log_prob, 4),
            }
        )
    return rows, records, None
