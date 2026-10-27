"""The token stream of Voxtral Realtime: one token per 80 ms audio frame.

Layout of one sequence (offline mode, read from `mistral_common` and the
`transformers` processor, 2026-10-20):

    position 0                      BOS
    positions 1 .. L+D              [STREAMING_PAD]   (prompt)
    positions L+D+1 .. end          generated: [STREAMING_PAD] while nothing
                                    is due, otherwise [STREAMING_WORD] and the
                                    word's text tokens

L is the number of left-padding frames (32: silence prepended to the audio)
and D the delay in frames (480 ms = 6). Position p carries the audio
embedding of frame p, and the token at position p is predicted at p - 1.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

FRAME_SECONDS = 0.08
# Frames between our formula and where the model itself puts English words,
# with end times from a wav2vec2 CTC aligner: 1 (28 words), 2 (46), 3 (7).
# Absorbs the aligner's early word ends and the model's own convention.
EMISSION_OFFSET = 2


@dataclass(frozen=True)
class StreamLayout:
    n_left_pad: int
    delay: int
    bos_id: int
    pad_id: int
    word_id: int
    offset: int = 0

    @property
    def prompt_length(self) -> int:
        return 1 + self.n_left_pad + self.delay

    def emission_position(self, word_end: float) -> int:
        """Position of a word's first token, given when it ends in the clip."""
        frames = math.ceil(round(word_end / FRAME_SECONDS, 6))
        return self.n_left_pad + frames + self.delay + self.offset


IGNORE_INDEX = -100


def build_sequence(
    layout: StreamLayout,
    words: list[tuple[list[int], list[int], float]],
    n_positions: int,
) -> tuple[list[int], list[int]] | None:
    """Training input ids and labels for one clip, or None if the words don't fit.

    `words` holds, per word in spoken order: its tokens after a
    [STREAMING_WORD] marker, its tokens when it directly follows the previous
    word (with a leading space), and its end time in seconds. A word goes at
    its emission position; if the previous word's tokens still run there, it
    follows them without a marker. Prompt positions get no loss.
    """
    ids = [layout.bos_id] + [layout.pad_id] * (n_positions - 1)
    cursor = layout.prompt_length
    for marked_tokens, joined_tokens, end in words:
        slot = max(layout.emission_position(end), layout.prompt_length)
        if slot >= cursor:
            tokens = [layout.word_id, *marked_tokens]
            cursor = slot
        else:
            tokens = joined_tokens
        if cursor + len(tokens) > n_positions:
            return None
        ids[cursor : cursor + len(tokens)] = tokens
        cursor += len(tokens)
    labels = [IGNORE_INDEX] * layout.prompt_length + ids[layout.prompt_length :]
    return ids, labels


def tekken(processor):
    """The underlying `mistral_common` tokenizer of a Realtime processor."""
    return processor.tokenizer.tokenizer.instruct_tokenizer.tokenizer


def layout_from_processor(processor) -> StreamLayout:
    tokenizer = tekken(processor)
    return StreamLayout(
        n_left_pad=processor.mistral_common_audio_config.n_left_pad_tokens,
        delay=processor.num_delay_tokens,
        bos_id=tokenizer.bos_id,
        pad_id=tokenizer.get_special_token("[STREAMING_PAD]"),
        word_id=tokenizer.get_special_token("[STREAMING_WORD]"),
        offset=EMISSION_OFFSET,
    )


@dataclass(frozen=True)
class EmittedWord:
    text: str
    position: int  # of its first token ([STREAMING_WORD] when present)
    marked: bool  # preceded by [STREAMING_WORD]


def parse_stream(ids: list[int], pieces: list[str], layout: StreamLayout) -> list[EmittedWord]:
    """Split the generated part of a stream into words, with where each starts.

    A word starts at [STREAMING_WORD], or at a text token beginning with a
    space when words share an emission frame.
    """
    words: list[EmittedWord] = []
    text, start, marked = "", None, False

    def flush() -> None:
        if start is not None and text.strip():
            words.append(EmittedWord(text.strip(), start, marked))

    for pos in range(layout.prompt_length, len(ids)):
        token, piece = ids[pos], pieces[pos]
        if token == layout.pad_id:
            continue
        if token == layout.word_id:
            flush()
            text, start, marked = "", pos, True
        elif piece.startswith(" ") and text.strip():
            flush()
            text, start, marked = piece, pos, False
        else:
            if start is None:
                start, marked = pos, False
            text += piece
    flush()
    return words


def render_stream(ids: list[int], pieces: list[str], layout: StreamLayout) -> str:
    """Compact view of the generated part: '.' per pad, '|' per word marker."""
    out = []
    for pos in range(layout.prompt_length, len(ids)):
        if ids[pos] == layout.pad_id:
            out.append(".")
        elif ids[pos] == layout.word_id:
            out.append("|")
        else:
            out.append(f"[{pieces[pos]}]")
    return "".join(out)
