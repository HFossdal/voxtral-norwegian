import datasets
import numpy as np
import pandas as pd
import pytest

from tests.test_data import wav_bytes
from voxnor.evaluate import read_predictions, score_predictions, transcribe_to_file


class FakeTranscriber:
    """Returns a fixed transcript per audio length and records batch sizes."""

    def __init__(self, texts_by_length):
        self.texts_by_length = texts_by_length
        self.batches = []

    def transcribe(self, audios):
        self.batches.append([len(a) for a in audios])
        return [self.texts_by_length[len(a)] for a in audios]


def dataset(durations):
    return datasets.Dataset.from_list(
        [
            {
                "utterance_id": f"u{i}",
                "duration": d,
                "utterance_audio_file": {"bytes": wav_bytes(d), "path": f"{i}.wav"},
            }
            for i, d in enumerate(durations)
        ]
    )


DURATIONS = [0.1, 0.3, 0.2, 0.4, 0.5]
TEXTS = {int(d * 16000): f"tekst {i}" for i, d in enumerate(DURATIONS)}


def test_transcribes_longest_first_in_batches(tmp_path):
    fake = FakeTranscriber(TEXTS)
    path = tmp_path / "pred.jsonl"
    transcribe_to_file(dataset(DURATIONS), fake, path, batch_size=2)
    assert fake.batches == [[8000, 6400], [4800, 3200], [1600]]
    preds = read_predictions(path).set_index("utterance_id")
    assert preds.loc["u3", "hypothesis"] == "tekst 3"
    assert not preds["truncated"].any()


def test_resume_only_transcribes_missing_utterances(tmp_path):
    path = tmp_path / "pred.jsonl"
    transcribe_to_file(dataset(DURATIONS), FakeTranscriber(TEXTS), path, batch_size=2)
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:3]) + "\n", encoding="utf-8")  # simulate a crash

    fake = FakeTranscriber(TEXTS)
    transcribe_to_file(dataset(DURATIONS), fake, path, batch_size=2)
    assert sum(len(b) for b in fake.batches) == 2
    assert sorted(read_predictions(path)["utterance_id"]) == [f"u{i}" for i in range(5)]

    fake = FakeTranscriber(TEXTS)
    transcribe_to_file(dataset(DURATIONS), fake, path, batch_size=2)
    assert fake.batches == []


def test_per_utterance_truncation_flags_are_recorded(tmp_path):
    fake = FakeTranscriber(TEXTS)
    fake_transcribe = fake.transcribe

    def transcribe(audios):
        out = fake_transcribe(audios)
        fake.last_truncated = [len(a) == 8000 for a in audios]
        return out

    fake.transcribe = transcribe
    path = tmp_path / "pred.jsonl"
    transcribe_to_file(dataset(DURATIONS), fake, path, batch_size=2)
    truncated = read_predictions(path).set_index("utterance_id")["truncated"]
    assert truncated[truncated].index.tolist() == ["u4"]


def test_read_predictions_rejects_duplicates(tmp_path):
    path = tmp_path / "pred.jsonl"
    line = '{"utterance_id": "u0", "hypothesis": "a", "truncated": false}\n'
    path.write_text(line * 2, encoding="utf-8")
    with pytest.raises(ValueError):
        read_predictions(path)


def metadata(ids, speakers):
    return pd.DataFrame(
        {
            "utterance_id": ids,
            "speaker_id": speakers,
            "gender": "male",
            "region": "east",
            "duration": np.full(len(ids), 3.0),
            "raw_text": "hei på deg",
            "standardized_text": "hei på deg",
        }
    )


def test_score_predictions_marks_seen_speakers_and_requires_every_prediction():
    meta = metadata(["u0", "u1"], ["a", "b"])
    preds = pd.DataFrame(
        {"utterance_id": ["u0", "u1"], "hypothesis": ["Hei på deg.", "hei"], "truncated": False}
    )
    scored, summary = score_predictions(preds, meta, train_speakers={"a"})
    assert scored.set_index("utterance_id")["seen_speaker"].to_dict() == {"u0": True, "u1": False}
    assert summary["slices"]["seen_speaker"]["True"]["wer"]["value"] == 0
    assert summary["slices"]["seen_speaker"]["False"]["wer"]["value"] == pytest.approx(2 / 3)

    with pytest.raises(ValueError):
        score_predictions(preds.iloc[:1], meta, train_speakers=set())
