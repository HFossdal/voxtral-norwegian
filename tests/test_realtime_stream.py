import pytest

from voxnor.realtime_stream import StreamLayout, build_sequence, parse_stream, render_stream

BOS, PAD, WORD = 1, 32, 33
LAYOUT = StreamLayout(n_left_pad=2, delay=1, bos_id=BOS, pad_id=PAD, word_id=WORD)


def stream(*generated: int | str) -> tuple[list[int], list[str]]:
    """A stream with the prompt and the given generated tokens (str = text piece)."""
    ids, pieces = [BOS, PAD, PAD, PAD], ["<s>", "[P]", "[P]", "[P]"]
    for i, g in enumerate(generated):
        if isinstance(g, str):
            ids.append(1000 + i)
            pieces.append(g)
        else:
            ids.append(g)
            pieces.append({PAD: "[P]", WORD: "[W]"}[g])
    return ids, pieces


def test_prompt_length():
    assert LAYOUT.prompt_length == 4


@pytest.mark.parametrize(
    ("end", "position"),
    [(0.0, 3), (0.01, 4), (0.08, 4), (0.081, 5), (0.24, 6)],
)
def test_emission_position_counts_whole_frames(end, position):
    # left pad 2 + frames needed to contain the word end + delay 1
    assert LAYOUT.emission_position(end) == position


def test_parse_stream_splits_on_markers():
    ids, pieces = stream(PAD, WORD, "Hel", "lo", PAD, WORD, " world", PAD)
    words = parse_stream(ids, pieces, LAYOUT)
    assert [(w.text, w.position, w.marked) for w in words] == [
        ("Hello", 5, True),
        ("world", 9, True),
    ]


def test_parse_stream_splits_words_sharing_an_emission_frame():
    ids, pieces = stream(WORD, " the", " cat", PAD)
    words = parse_stream(ids, pieces, LAYOUT)
    assert [(w.text, w.position, w.marked) for w in words] == [("the", 4, True), ("cat", 6, False)]


def test_parse_stream_ignores_the_prompt_and_empty_markers():
    ids, pieces = stream(WORD, PAD, WORD, "ja")
    assert [w.text for w in parse_stream(ids, pieces, LAYOUT)] == ["ja"]


def test_render_stream():
    ids, pieces = stream(PAD, WORD, " hi", PAD)
    assert render_stream(ids, pieces, LAYOUT) == ".|[ hi]."


def test_build_sequence_places_marks_and_joins_words():
    words = [([100], [101], 0.08), ([200], [201], 0.16), ([300], [301], 0.40)]
    ids, labels = build_sequence(LAYOUT, words, n_positions=12)
    # "A" due at 4 -> marker+token at 4-5; "B" due at 5 but 5 is taken, so it
    # joins at 6 without a marker; "C" due at 8.
    assert ids == [BOS, PAD, PAD, PAD, WORD, 100, 201, PAD, WORD, 300, PAD, PAD]
    assert labels == [-100] * 4 + ids[4:]


def test_build_sequence_rejects_words_that_overflow_the_stream():
    assert build_sequence(LAYOUT, [([7, 8, 9], [7, 8, 9], 0.4)], n_positions=10) is None


def test_build_sequence_without_words_is_all_pads():
    ids, _ = build_sequence(LAYOUT, [], n_positions=6)
    assert ids == [BOS] + [PAD] * 5


def test_emission_offset_shifts_every_word():
    shifted = StreamLayout(n_left_pad=2, delay=1, bos_id=BOS, pad_id=PAD, word_id=WORD, offset=2)
    assert shifted.emission_position(0.08) == LAYOUT.emission_position(0.08) + 2
