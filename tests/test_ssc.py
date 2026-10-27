import json

import datasets
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tests.test_data import wav_bytes
from voxnor.align import WordTiming
from voxnor.data import SAMPLE_RATE, decode_audio
from voxnor.ssc import (
    CLIP_SCHEMA,
    SKIP_DATE,
    SKIP_HELD_OUT,
    SKIP_LANGUAGE,
    SKIP_NO_RUN,
    SKIP_SPEAKERS,
    SKIP_SPLIT,
    ClipPlan,
    agreeing_runs,
    match_speaker,
    plan_clips,
    read_parlamint_people,
    segment_clips,
    skip_reason,
    split_run,
)
from voxnor.train_data import add_extra_data, read_alignments


def test_agreeing_runs_ignore_case_punctuation_and_hesitations():
    proceedings = "Det er viktig. Vi må, som sagt, gjøre noe med dette i 2020!"
    asr = "at det er eee viktig vi må som sagt gjøre noe med dette i tjuetjue".split()
    # "at" is not in the proceedings; the run ends before the number.
    assert agreeing_runs(proceedings, asr, min_run=8) == [(1, 13)]
    assert agreeing_runs(proceedings, asr, min_run=14) == []


def test_agreeing_runs_drop_a_partly_agreeing_hyphenated_word():
    proceedings = "a b c d e f g h"
    asr = ["x-a", "b", "c", "d", "e", "f", "g", "h"]  # "x-a" -> tokens "x", "a"
    assert agreeing_runs(proceedings, asr, min_run=8) == [(1, 7)]


def timings(spans):
    return [WordTiming(f"w{i}", s, e) for i, (s, e) in enumerate(spans)]


def test_split_run_cuts_at_the_widest_pause_within_the_limit():
    # Ten 0.5 s words with 0.1 s pauses, except a 0.6 s pause after word 6.
    spans, t = [], 0.0
    for i in range(10):
        spans.append((t, t + 0.5))
        t += 0.5 + (0.6 if i == 6 else 0.1)
    words = timings(spans)
    pieces = split_run(words, 0, 9, max_span=5.0)
    assert pieces == [(0, 6), (7, 9)]
    for first, last in pieces:
        assert words[last].end - words[first].start <= 5.0
    assert split_run(words, 0, 9, max_span=100) == [(0, 9)]


def test_plan_clips_keep_silence_at_the_edges():
    # Words 0-9, 1 s apart; word 3 touches word 2 (no pause before it).
    spans = [(i, i + 0.6) for i in range(10)]
    spans[3] = (2.6, 3.6)
    words = timings(spans)
    clips = plan_clips(words, [(3, 8)], seconds=10.0, max_seconds=14, min_words=4)
    # Word 3 has no silence before it, so the clip starts at word 4.
    assert clips == [ClipPlan(4, 8, 3.8, 8.8)]
    assert plan_clips(words, [(3, 8)], seconds=10.0, min_words=6) == []
    # At the segment edges, the pad is limited by the audio only.
    assert plan_clips(words, [(0, 9)], seconds=9.7, min_words=4) == [ClipPlan(0, 9, 0.0, 9.7)]


def write_people(tmp_path):
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<listPerson xmlns="http://www.tei-c.org/ns/1.0">
  <person xml:id="person.ES"><persName><forename>Erna</forename>
    <surname>Solberg</surname></persName></person>
  <person xml:id="person.IME"><persName><forename>Ine Marie</forename>
    <surname>Eriksen Søreide</surname></persName></person>
  <person xml:id="person.KA1"><persName><forename>Kari</forename>
    <surname>Andersen</surname></persName></person>
  <person xml:id="person.KA2"><persName><forename>Kari Anne</forename>
    <surname>Andersen</surname></persName></person>
</listPerson>"""
    path = tmp_path / "people.xml"
    path.write_text(xml, encoding="utf-8")
    return read_parlamint_people(path)


def test_match_speaker(tmp_path):
    people = write_people(tmp_path)
    assert match_speaker("Erna Solberg", people) == "person.ES"
    # NPSC leaves out the middle name; first forename + last surname word.
    assert match_speaker("Ine Marie Søreide", people) == "person.IME"
    assert match_speaker("Kari Andersen", people) == "person.KA1"  # exact beats loose
    assert match_speaker("Kari Anne Andersen", people) == "person.KA2"
    assert match_speaker("Kari Marie Andersen", people) is None  # two loose matches
    assert match_speaker("Marianne Synnes", people) == "person.MS"  # renamed
    assert match_speaker("unknown", people) is None


def meta(**changes):
    record = {
        "segment_id": 7,
        "split": "train",
        "num_speakers": 1,
        "speakers": [
            {"speaker_id": "person.ES", "language": "nob", "dialect": "west", "gender": "F"}
        ],
        "meeting_date": "2014-06-11",
        "audio_path": "data/audio/2014/x.mp3",
        "duration": 6.0,
        "proceedings_text": "Vi må gjøre noe med dette nå, sier jeg.",
        "transcription_text": "eee vi må gjøre noe med dette nå sier jeg da",
    }
    return record | changes


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({}, None),
        ({"split": "eval"}, SKIP_SPLIT),
        ({"num_speakers": 2}, SKIP_SPEAKERS),
        ({"speakers": []}, SKIP_SPEAKERS),
        ({"speakers": [{"speaker_id": "person.X", "language": "nno"}]}, SKIP_LANGUAGE),
        ({"speakers": [{"speaker_id": "person.HO", "language": "nob"}]}, SKIP_HELD_OUT),
        ({"meeting_date": "2017-02-07"}, SKIP_DATE),
    ],
)
def test_skip_reason(changes, reason):
    assert skip_reason(meta(**changes), {"person.HO"}, {"2017-02-07"}) == reason


class EvenAligner:
    """Places word i at [0.5 i + 0.1, 0.5 i + 0.4] seconds."""

    def align_with_score(self, audio, words):
        return [WordTiming(w, 0.5 * i + 0.1, 0.5 * i + 0.4) for i, w in enumerate(words)], -0.2


def test_segment_clips_cut_the_agreeing_run_out_of_the_audio():
    audio = decode_audio({"bytes": wav_bytes(6.0)})
    rows, records, reason = segment_clips(meta(), audio, EvenAligner(), min_run=8)
    assert reason is None
    [row], [record] = rows, records
    # Words 1-9 agree ("vi" ... "jeg"); the hesitation before them and "da" do not.
    assert row["standardized_text"] == "vi må gjøre noe med dette nå sier jeg"
    # Cut halfway through the pauses next to the run.
    assert (row["start"], row["end"]) == (0.5, 5.0)
    assert row["speaker_id"] == "person.ES"
    assert (row["gender"], row["region"]) == ("female", "west")
    clip = decode_audio(row["utterance_audio_file"])
    assert len(clip) == record["n_samples"] == round(4.5 * SAMPLE_RATE)
    assert record["words"][0] == ["vi", pytest.approx(0.1), pytest.approx(0.4)]
    assert all(0 <= s <= e <= 4.5 for _, s, e in record["words"])
    assert pa.Table.from_pylist(rows, schema=CLIP_SCHEMA).num_rows == 1

    unknown = {
        "speaker_id": "person.ES",
        "language": "nob",
        "dialect": float("nan"),
        "gender": float("nan"),
    }
    [row], _, _ = segment_clips(meta(speakers=[unknown]), audio, EvenAligner(), min_run=8)
    assert (row["gender"], row["region"]) == ("unknown", "unknown")
    assert pa.Table.from_pylist([row], schema=CLIP_SCHEMA).num_rows == 1

    no_run = meta(proceedings_text="Helt andre ord her.")
    assert segment_clips(no_run, audio, EvenAligner(), min_run=8)[2] == SKIP_NO_RUN


def test_extra_clips_join_the_training_set(tmp_path):
    rows, records, _ = segment_clips(
        meta(), decode_audio({"bytes": wav_bytes(6.0)}), EvenAligner(), min_run=8
    )
    (tmp_path / "clips").mkdir()
    (tmp_path / "alignments").mkdir()
    pq.write_table(
        pa.Table.from_pylist(rows, schema=CLIP_SCHEMA), tmp_path / "clips" / "shard-00001.parquet"
    )
    (tmp_path / "alignments" / "shard-00001.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records), encoding="utf-8"
    )
    base_rows = [dict(r, utterance_id="npsc_1", speaker_id="npsc_3") for r in rows]
    base = datasets.Dataset(pa.Table.from_pylist(base_rows, schema=CLIP_SCHEMA))
    joined, alignments = add_extra_data(base, {"npsc_1": records[0]}, tmp_path)
    assert joined["utterance_id"] == ["npsc_1", rows[0]["utterance_id"]]
    assert set(alignments) == {"npsc_1", rows[0]["utterance_id"]}
    assert joined.features == base.features
    assert np.array_equal(
        decode_audio(joined[1]["utterance_audio_file"]),
        decode_audio(rows[0]["utterance_audio_file"]),
    )
    assert read_alignments(tmp_path / "alignments" / "shard-00001.jsonl").keys() == {
        rows[0]["utterance_id"]
    }
    with pytest.raises(ValueError):
        add_extra_data(base, {rows[0]["utterance_id"]: records[0]}, tmp_path)
