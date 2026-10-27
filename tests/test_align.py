import itertools
import json

import numpy as np
import pytest
import torch
from transformers import (
    Wav2Vec2Config,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2ForCTC,
)

from voxnor.align import CtcAligner, WordTiming, ctc_viterbi, word_timings

BLANK, A, B = 0, 1, 2


def frames(*ids: int, vocab: int = 3) -> np.ndarray:
    """Log-probs where each frame strongly prefers the given id."""
    out = np.full((len(ids), vocab), np.log(0.01))
    for t, i in enumerate(ids):
        out[t, i] = np.log(0.98)
    return out


def test_viterbi_follows_the_evidence():
    path = ctc_viterbi(frames(A, A, BLANK, B, B), [A, B], BLANK)
    assert path.tolist() == [0, 0, -1, 1, 1]


def test_viterbi_needs_a_blank_between_equal_targets():
    assert ctc_viterbi(frames(A, BLANK, A), [A, A], BLANK).tolist() == [0, -1, 1]
    assert ctc_viterbi(frames(A, A), [A, A], BLANK) is None


def test_viterbi_rejects_more_targets_than_frames():
    assert ctc_viterbi(frames(A), [A, B], BLANK) is None


def test_viterbi_forces_the_transcript_against_the_evidence():
    # The audio "says" b everywhere, but the transcript is a then b.
    path = ctc_viterbi(frames(B, B, B, B), [A, B], BLANK)
    assert path[0] == 0 and path[-1] == 1


def test_word_timings_use_first_and_last_frame():
    path = np.array([-1, 0, 0, -1, 1, 2, -1])
    words = ["ab", "c"]
    timings = word_timings(path, [0, -1, 1], words, seconds_per_frame=0.02)
    # target 1 is the separator, so word "c" is target 2 at frame 5
    assert timings == [WordTiming("ab", 0.02, 0.06), WordTiming("c", 0.1, 0.12)]


def test_word_without_characters_gets_zero_length_after_previous():
    path = np.array([0, 0, -1])
    timings = word_timings(path, [0], ["ja", "§"], seconds_per_frame=0.02)
    assert timings[1] == WordTiming("§", 0.04, 0.04)


@pytest.fixture(scope="module")
def aligner(tmp_path_factory):
    chars = list("abcdefghijklmnopqrstuvwxyzæøå")
    vocab = {"[PAD]": 0, "|": 1, **{c: i + 2 for i, c in enumerate(chars)}}
    path = tmp_path_factory.mktemp("w2v") / "vocab.json"
    path.write_text(json.dumps(vocab), encoding="utf-8")
    tokenizer = Wav2Vec2CTCTokenizer(str(path), pad_token="[PAD]", word_delimiter_token="|")
    torch.manual_seed(0)
    config = Wav2Vec2Config(
        vocab_size=len(vocab),
        hidden_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=32,
        conv_dim=(8, 8),
        conv_kernel=(10, 3),
        conv_stride=(5, 2),
        num_conv_pos_embeddings=16,
        num_conv_pos_embedding_groups=2,
        pad_token_id=0,
    )
    return CtcAligner(Wav2Vec2ForCTC(config), Wav2Vec2FeatureExtractor(), tokenizer)


def test_encode_spells_words_with_separators(aligner):
    targets, word_of_target = aligner.encode(["blå", "én"])
    v = aligner.vocab
    assert targets == [v["b"], v["l"], v["å"], v["|"], v["e"], v["n"]]
    assert word_of_target == [0, 0, 0, -1, 1, 1]


def test_align_gives_ordered_timings_within_the_clip(aligner):
    audio = np.random.default_rng(0).standard_normal(16000).astype(np.float32)
    timings = aligner.align(audio, ["hei", "på", "deg"])
    assert [t.word for t in timings] == ["hei", "på", "deg"]
    assert all(a.end <= b.start for a, b in itertools.pairwise(timings))
    assert timings[0].start >= 0 and timings[-1].end <= 1.0
    assert aligner.seconds_per_frame == pytest.approx(10 / 16000)
