import io

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from voxnor.config import DataConfig
from voxnor.data import (
    METADATA_COLUMNS,
    SAMPLE_RATE,
    decode_audio,
    exclude_speakers,
    read_id_list,
    read_metadata,
    read_rows,
    stratified_sample,
    subset_cache_path,
    write_id_list,
)


def wav_bytes(seconds: float, sample_rate: int = SAMPLE_RATE, channels: int = 1) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    tone = 0.5 * np.sin(2 * np.pi * 440 * t)
    data = np.stack([tone] * channels, axis=1)
    buf = io.BytesIO()
    sf.write(buf, data, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def fake_rows(n: int) -> list[dict]:
    regions = ["east", "west", "north", "mid", "south"]
    return [
        {
            "utterance_id": f"npsc_{i}",
            "speaker_id": f"spk_{i % 7}",
            "gender": "female" if i % 2 else "male",
            "region": regions[i % 7 % len(regions)],
            "duration": 1.0,
            "full_audio_file": f"meeting_{i // 10}",
            "start": float(i),
            "end": float(i + 1),
            "raw_text": f"Tekst nummer {i}",
            "standardized_text": f"tekst nummer {i}",
            "utterance_audio_file": {"bytes": wav_bytes(0.1), "path": f"{i}.wav"},
        }
        for i in range(n)
    ]


@pytest.fixture
def parquet_path(tmp_path):
    path = tmp_path / "test-00000-of-00001.parquet"
    pq.write_table(pa.Table.from_pylist(fake_rows(25)), path, row_group_size=10)
    return path


def test_read_metadata_skips_audio(parquet_path):
    meta = read_metadata(parquet_path)
    assert list(meta.columns) == list(METADATA_COLUMNS)
    assert len(meta) == 25


def test_read_rows_returns_requested_ids_in_requested_order(parquet_path):
    ids = ["npsc_21", "npsc_3", "npsc_12"]
    table = read_rows(parquet_path, ids)
    assert table["utterance_id"].to_pylist() == ids
    assert table.num_columns == len(METADATA_COLUMNS) + 1


def test_read_rows_without_ids_returns_everything(parquet_path):
    assert read_rows(parquet_path, None).num_rows == 25


def test_read_rows_rejects_unknown_ids(parquet_path):
    with pytest.raises(KeyError):
        read_rows(parquet_path, ["npsc_3", "npsc_999"])


def test_read_rows_works_on_open_file_handles(parquet_path):
    # The laptop path reads from a remote file object rather than a local path.
    with open(parquet_path, "rb") as handle:
        assert read_rows(handle, ["npsc_5"])["utterance_id"].to_pylist() == ["npsc_5"]


def test_subset_cache_path_depends_on_ids_and_split(tmp_path):
    cfg = DataConfig(dataset_id="org/name", cache_dir=str(tmp_path))
    a = subset_cache_path(cfg, "test", ["1", "2"])
    assert a.parent == tmp_path / "org__name"
    assert a == subset_cache_path(cfg, "test", ["1", "2"])
    assert a != subset_cache_path(cfg, "test", ["1", "3"])
    assert a != subset_cache_path(cfg, "validation", ["1", "2"])


def test_decode_audio_returns_16k_mono_float32():
    audio = decode_audio({"bytes": wav_bytes(0.5)})
    assert audio.dtype == np.float32
    assert audio.shape == (SAMPLE_RATE // 2,)
    assert 0.4 < np.abs(audio).max() <= 0.5


def test_decode_audio_resamples_and_downmixes():
    audio = decode_audio({"bytes": wav_bytes(0.5, sample_rate=48_000, channels=2)})
    assert audio.ndim == 1
    assert abs(len(audio) - SAMPLE_RATE // 2) <= 1


def test_exclude_speakers_removes_every_held_out_speaker():
    train = pd.DataFrame({"speaker_id": ["a", "b", "c", "d"], "utterance_id": list("1234")})
    val = pd.DataFrame({"speaker_id": ["b"]})
    test = pd.DataFrame({"speaker_id": ["d", "e"]})
    assert exclude_speakers(train, val, test)["speaker_id"].tolist() == ["a", "c"]


def test_stratified_sample_is_balanced_deterministic_and_unique():
    meta = pd.DataFrame(fake_rows(200)).drop(columns="utterance_audio_file")
    a = stratified_sample(meta, 20, seed=1)
    assert a == stratified_sample(meta, 20, seed=1)
    assert a != stratified_sample(meta, 20, seed=2)
    assert len(set(a)) == 20
    counts = meta.set_index("utterance_id").loc[a, "region"].value_counts()
    assert counts.max() - counts.min() <= 1


def test_stratified_sample_gives_small_strata_everything():
    meta = pd.DataFrame(
        {"utterance_id": [str(i) for i in range(12)], "region": ["east"] * 10 + ["south"] * 2}
    )
    picked = stratified_sample(meta, 8)
    assert {"10", "11"} <= set(picked)
    assert len(picked) == 8


def test_stratified_sample_rejects_too_large_requests():
    meta = pd.DataFrame({"utterance_id": ["1", "2"], "region": ["east", "west"]})
    with pytest.raises(ValueError):
        stratified_sample(meta, 3)


def test_id_lists_round_trip_and_ignore_comments(tmp_path):
    path = tmp_path / "sub" / "test.txt"
    write_id_list(path, ["b", "a"], header="made by a test\nseed 0")
    assert read_id_list(path) == ["b", "a"]
    assert path.read_text(encoding="utf-8").startswith("# made by a test\n# seed 0\n")


def test_id_lists_reject_duplicates(tmp_path):
    path = tmp_path / "dup.txt"
    path.write_text("a\nb\na\n", encoding="utf-8")
    with pytest.raises(ValueError):
        read_id_list(path)
