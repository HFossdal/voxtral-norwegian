"""Word timings from CTC forced alignment (for Voxtral Realtime training targets).

NPSC has no word timestamps, and a Realtime training target places every
word at a time derived from its end time. A wav2vec2 CTC model gives
per-frame character probabilities; Viterbi alignment of the known
transcript against them gives each character's frames, hence each word's.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

import numpy as np
import torch
from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor, Wav2Vec2ForCTC

from voxnor.data import SAMPLE_RATE


@dataclass(frozen=True)
class WordTiming:
    word: str
    start: float  # seconds from the start of the clip
    end: float


def ctc_viterbi(log_probs: np.ndarray, targets: list[int], blank: int) -> np.ndarray | None:
    """Most likely frame-level path through `targets`, or None if they cannot fit.

    `log_probs` is (frames, vocab). Returns, per frame, the index of the
    target being emitted, or -1 for a blank frame. Standard CTC topology:
    blanks between targets are optional, except between two equal targets.
    """
    frames = log_probs.shape[0]
    states = np.full(2 * len(targets) + 1, blank)
    states[1::2] = targets
    n = len(states)
    # A state may be entered from two states back only if it is a target that
    # differs from the previous target (skipping the blank between them).
    can_skip = np.zeros(n, dtype=bool)
    can_skip[3::2] = states[3::2] != states[1:-2:2]

    score = np.full(n, -np.inf)
    score[0] = log_probs[0, states[0]]
    if n > 1:
        score[1] = log_probs[0, states[1]]
    back = np.zeros((frames, n), dtype=np.int8)  # 0 stay, 1 from previous, 2 skip
    for t in range(1, frames):
        stay = score
        prev = np.concatenate(([-np.inf], score[:-1]))
        skip = np.where(can_skip, np.concatenate(([-np.inf, -np.inf], score[:-2])), -np.inf)
        candidates = np.stack([stay, prev, skip])
        back[t] = candidates.argmax(axis=0)
        score = candidates.max(axis=0) + log_probs[t, states]

    ends = [n - 1, n - 2] if n > 1 else [n - 1]
    state = max(ends, key=lambda s: score[s])
    if not np.isfinite(score[state]):
        return None
    path = np.empty(frames, dtype=np.int64)
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])
    return np.where(path % 2 == 1, path // 2, -1)


def path_log_prob(log_probs: np.ndarray, path: np.ndarray, targets: list[int], blank: int) -> float:
    """Mean per-frame log-probability of a `ctc_viterbi` path."""
    tokens = np.where(path >= 0, np.asarray(targets)[np.maximum(path, 0)], blank)
    return float(log_probs[np.arange(len(path)), tokens].mean())


def word_timings(
    path: np.ndarray, word_of_target: list[int], words: list[str], seconds_per_frame: float
) -> list[WordTiming]:
    """Turn a per-frame target path into one timing per word.

    A word with no alignable characters gets a zero-length span where the
    previous word ended.
    """
    spans: dict[int, list[int]] = {}
    for frame, target in enumerate(path):
        if target >= 0 and word_of_target[target] >= 0:
            spans.setdefault(word_of_target[target], []).append(frame)
    timings, last_end = [], 0.0
    for i, word in enumerate(words):
        if i in spans:
            start, end = spans[i][0] * seconds_per_frame, (spans[i][-1] + 1) * seconds_per_frame
        else:
            start = end = last_end
        timings.append(WordTiming(word, round(start, 3), round(end, 3)))
        last_end = end
    return timings


class CtcAligner:
    """Forced alignment of a known transcript with a wav2vec2 CTC model."""

    def __init__(
        self,
        model: Wav2Vec2ForCTC,
        features: Wav2Vec2FeatureExtractor,
        tokenizer: Wav2Vec2CTCTokenizer,
    ):
        self.model, self.features = model.eval(), features
        self.vocab = tokenizer.get_vocab()
        self.blank = tokenizer.pad_token_id
        self.separator = self.vocab[tokenizer.word_delimiter_token]
        # Some models are trained on upper-case text (English), others lower-case.
        self.upper = "a" not in self.vocab and "A" in self.vocab
        self.seconds_per_frame = float(np.prod(model.config.conv_stride)) / SAMPLE_RATE

    @classmethod
    def from_pretrained(cls, hf_id: str, device: str | torch.device = "cpu") -> CtcAligner:
        return cls(
            Wav2Vec2ForCTC.from_pretrained(hf_id).to(device),
            Wav2Vec2FeatureExtractor.from_pretrained(hf_id),
            Wav2Vec2CTCTokenizer.from_pretrained(hf_id),
        )

    def encode(self, words: list[str]) -> tuple[list[int], list[int]]:
        """Character ids with word separators, and the word index of each (-1: separator)."""
        targets, word_of_target = [], []
        for i, word in enumerate(words):
            chars = [self.vocab[c] for c in self._spell(word) if c in self.vocab]
            if not chars:
                continue
            if targets:
                targets.append(self.separator)
                word_of_target.append(-1)
            targets += chars
            word_of_target += [i] * len(chars)
        return targets, word_of_target

    @torch.inference_mode()
    def log_probs(self, audio: np.ndarray) -> np.ndarray:
        inputs = self.features(audio, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        logits = self.model(inputs.input_values.to(self.model.device)).logits[0]
        return torch.log_softmax(logits.float(), dim=-1).cpu().numpy()

    def align(self, audio: np.ndarray, words: list[str]) -> list[WordTiming] | None:
        """Timings for `words` (spoken in order in `audio`), or None if alignment fails."""
        result = self.align_with_score(audio, words)
        return result[0] if result else None

    def align_with_score(
        self, audio: np.ndarray, words: list[str]
    ) -> tuple[list[WordTiming], float] | None:
        """Timings and the path's mean log-probability per frame (low: poor fit)."""
        targets, word_of_target = self.encode(words)
        if not targets:
            return None
        log_probs = self.log_probs(audio)
        path = ctc_viterbi(log_probs, targets, self.blank)
        if path is None:
            return None
        timings = word_timings(path, word_of_target, words, self.seconds_per_frame)
        return timings, path_log_prob(log_probs, path, targets, self.blank)

    def _spell(self, word: str) -> str:
        # Fold accents the vocabulary lacks (é -> e); keep æ, ø, å when present.
        out = []
        for char in word.upper() if self.upper else word.lower():
            if char in self.vocab:
                out.append(char)
            else:
                base = unicodedata.normalize("NFD", char)[0]
                out.append(base)
        return "".join(out)
