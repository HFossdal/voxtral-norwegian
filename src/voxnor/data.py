"""Loading NPSC (Bokmål) utterances from the `scribe-project/npsc_nb` Parquet files.

Each split is a single Parquet file. Audio is stored inline as WAV bytes, so
metadata can be read without touching the audio, and audio is decoded here
with soundfile rather than through `datasets`' audio backend.
"""

from __future__ import annotations

import hashlib
import io
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

import datasets
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import soundfile as sf

from voxnor.config import DataConfig

Split = Literal["train", "validation", "test"]
SPLITS: tuple[Split, ...] = ("train", "validation", "test")
SAMPLE_RATE = 16_000
AUDIO_COLUMN = "utterance_audio_file"
METADATA_COLUMNS = (
    "utterance_id",
    "speaker_id",
    "gender",
    "region",
    "duration",
    "full_audio_file",
    "start",
    "end",
    "raw_text",
    "standardized_text",
)


def parquet_file(dataset_id: str, split: Split, download: bool):
    """Return a local path (downloaded and cached) or an open remote file handle."""
    from huggingface_hub import HfFileSystem, hf_hub_download

    fs = HfFileSystem()
    matches = fs.glob(f"datasets/{dataset_id}/data/{split}-*.parquet")
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one Parquet file for {dataset_id}/{split}, got {matches}"
        )
    repo_path = matches[0].removeprefix(f"datasets/{dataset_id}/")
    if download:
        return hf_hub_download(dataset_id, repo_path, repo_type="dataset")
    return fs.open(matches[0])


def read_metadata(source) -> pd.DataFrame:
    """All columns except the audio, for one split. Cheap even for remote files."""
    return pq.read_table(source, columns=list(METADATA_COLUMNS)).to_pandas()


def load_metadata(cfg: DataConfig, split: Split) -> pd.DataFrame:
    return read_metadata(parquet_file(cfg.dataset_id, split, download=False))


def load_split(cfg: DataConfig, split: Split) -> datasets.Dataset:
    """Utterances of one split, after subset selection and speaker exclusion.

    The audio column stays as undecoded bytes; use `decode_audio` per example.
    """
    ids = _selected_ids(cfg, split)
    if cfg.download:
        path = parquet_file(cfg.dataset_id, split, download=True)
        return select_rows(datasets.Dataset.from_parquet(path, keep_in_memory=False), ids)
    if ids is None:
        raise ValueError("download=False needs a subset_dir; a full split should be downloaded")

    # A subset spread over the split touches most row groups, and audio can
    # only be read a whole row group at a time, so the first fetch is slow.
    # Later runs read the cached rows.
    cached = subset_cache_path(cfg, split, ids)
    if not cached.exists():
        table = read_rows(parquet_file(cfg.dataset_id, split, download=False), ids)
        cached.parent.mkdir(parents=True, exist_ok=True)
        tmp = cached.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(cached)
    return datasets.Dataset(pq.read_table(cached))


def subset_cache_path(cfg: DataConfig, split: Split, ids: Sequence[str]) -> Path:
    digest = hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()[:16]
    return Path(cfg.cache_dir) / cfg.dataset_id.replace("/", "__") / f"{split}-{digest}.parquet"


def read_rows(source, ids: Sequence[str] | None) -> pa.Table:
    """Read full rows (with audio) from a Parquet file, row group by row group.

    Only row groups that contain a wanted id have their audio read, which keeps
    remote reads of small subsets cheap. Row order follows `ids` when given.
    """
    pf = pq.ParquetFile(source)
    if ids is None:
        return pf.read()
    wanted = pa.array(list(ids), type=pa.string())
    parts = []
    for i in range(pf.num_row_groups):
        group_ids = pf.read_row_group(i, columns=["utterance_id"])["utterance_id"]
        if pc.any(pc.is_in(group_ids, value_set=wanted)).as_py():
            group = pf.read_row_group(i)
            parts.append(group.filter(pc.is_in(group["utterance_id"], value_set=wanted)))
    table = pa.concat_tables(parts) if parts else pf.schema_arrow.empty_table()
    return _reorder(table, list(ids))


def select_rows(ds: datasets.Dataset, ids: Sequence[str] | None) -> datasets.Dataset:
    if ids is None:
        return ds
    return ds.select(_positions(ds["utterance_id"], ids))


def _reorder(table: pa.Table, ids: Sequence[str]) -> pa.Table:
    return table.take(_positions(table["utterance_id"].to_pylist(), ids))


def _positions(available: Iterable[str], ids: Sequence[str]) -> list[int]:
    position = {uid: i for i, uid in enumerate(available)}
    missing = [uid for uid in ids if uid not in position]
    if missing:
        raise KeyError(f"{len(missing)} subset ids not in the split, e.g. {missing[:3]}")
    return [position[uid] for uid in ids]


def _selected_ids(cfg: DataConfig, split: Split) -> list[str] | None:
    ids = read_id_list(Path(cfg.subset_dir) / f"{split}.txt") if cfg.subset_dir else None
    if split != "train" or not cfg.exclude_train_speakers_from:
        return ids

    held_out = [load_metadata(cfg, other) for other in cfg.exclude_train_speakers_from]
    keep = exclude_speakers(load_metadata(cfg, "train"), *held_out)["utterance_id"]
    if ids is None:
        return keep.tolist()
    keep_set = set(keep)
    return [uid for uid in ids if uid in keep_set]


def training_speakers(cfg: DataConfig) -> set[str]:
    """Speakers in the training set this config uses (all of it, not a subset)."""
    train = load_metadata(cfg, "train")
    held_out = [load_metadata(cfg, other) for other in cfg.exclude_train_speakers_from]
    return set(exclude_speakers(train, *held_out)["speaker_id"])


def exclude_speakers(train: pd.DataFrame, *held_out: pd.DataFrame) -> pd.DataFrame:
    """Training rows whose speaker does not occur in any of the held-out splits."""
    speakers = set().union(*(set(df["speaker_id"]) for df in held_out))
    return train[~train["speaker_id"].isin(speakers)]


def decode_audio(audio: dict) -> np.ndarray:
    """WAV bytes from the audio column -> float32 mono waveform at 16 kHz."""
    waveform, sample_rate = sf.read(io.BytesIO(audio["bytes"]), dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if sample_rate != SAMPLE_RATE:
        import librosa

        waveform = librosa.resample(waveform, orig_sr=sample_rate, target_sr=SAMPLE_RATE)
    return waveform.astype(np.float32, copy=False)


def read_id_list(path: str | Path) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    ids = [line.strip() for line in lines if line.strip() and not line.startswith("#")]
    duplicates = len(ids) - len(set(ids))
    if duplicates:
        raise ValueError(f"{path} contains {duplicates} duplicate ids")
    return ids


def write_id_list(path: str | Path, ids: Iterable[str], header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    comment = "".join(f"# {line}\n" for line in header.splitlines())
    path.write_text(comment + "".join(f"{uid}\n" for uid in ids), encoding="utf-8")


def stratified_sample(meta: pd.DataFrame, n: int, by: str = "region", seed: int = 0) -> list[str]:
    """`n` utterance ids, spread as evenly as possible across the values of `by`.

    Small strata contribute everything they have; the rest of the budget goes
    to larger strata. Deterministic for a given seed. Sorted by id.
    """
    if n > len(meta):
        raise ValueError(f"Asked for {n} utterances, only {len(meta)} available")
    groups = {key: g for key, g in meta.groupby(by, sort=True)}
    quota = dict.fromkeys(groups, 0)
    remaining = n
    while remaining:
        open_groups = [k for k in groups if quota[k] < len(groups[k])]
        share, extra = divmod(remaining, len(open_groups))
        for i, key in enumerate(open_groups):
            take = min(share + (i < extra), len(groups[key]) - quota[key])
            quota[key] += take
            remaining -= take
    picked = [
        groups[key].sample(n=k, random_state=seed)["utterance_id"] for key, k in quota.items() if k
    ]
    return sorted(pd.concat(picked).tolist())
