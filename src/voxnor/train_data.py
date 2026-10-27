"""Training data for both Voxtral models: word alignments, example selection, batches."""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from transformers import ProcessorMixin

from voxnor.align import CtcAligner, WordTiming
from voxnor.data import SAMPLE_RATE, decode_audio
from voxnor.realtime_stream import StreamLayout, build_sequence
from voxnor.realtime_train import n_positions, word_tokens

log = logging.getLogger(__name__)

# Why a training utterance is not used. Counted and reported.
DROP_INAUDIBLE = "transcript has <INAUDIBLE>"
DROP_NOT_ALIGNED = "alignment failed"
DROP_NO_FIT = "words don't fit the stream"
DROP_NOT_IN_FILE = "missing from the alignment file"


def align_to_file(ds: datasets.Dataset, aligner: CtcAligner, path: Path) -> None:
    """Append one JSON line of word timings per utterance, skipping ids already in `path`.

    An utterance without alignable characters (e.g. empty text) gets an empty
    word list; one whose transcript cannot be fitted to the audio gets
    `"words": null`. Resumes after an interruption like `transcribe_to_file`.
    """
    done = set()
    if path.exists():
        _drop_partial_line(path)
        done = set(read_alignments(path))
    todo = [i for i, uid in enumerate(ds["utterance_id"]) if uid not in done]
    log.info("Aligning %d utterances (%d already done)", len(todo), len(done))
    start = time.perf_counter()
    with path.open("a", encoding="utf-8") as out:
        for n, i in enumerate(todo, 1):
            row = ds[i]
            audio = decode_audio(row["utterance_audio_file"])
            words = row["standardized_text"].split()
            record = {"utterance_id": row["utterance_id"], "n_samples": len(audio)}
            result = aligner.align_with_score(audio, words) if words else ([], None)
            if result is None:
                record |= {"words": None, "log_prob": None}
            else:
                timings, score = result
                record |= {
                    "words": [[t.word, t.start, t.end] for t in timings],
                    "log_prob": None if score is None else round(score, 4),
                }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            if n % 500 == 0 or n == len(todo):
                out.flush()
                log.info("%d/%d aligned, %.1f s elapsed", n, len(todo), time.perf_counter() - start)


def _drop_partial_line(path: Path) -> None:
    """Cut an unfinished last line, left by a session killed mid-write."""
    data = path.read_bytes()
    if data and not data.endswith(b"\n"):
        path.write_bytes(data[: data.rfind(b"\n") + 1])


def read_alignments(path: Path) -> dict[str, dict]:
    with path.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    by_id = {r["utterance_id"]: r for r in records}
    if len(by_id) != len(records):
        raise ValueError(f"{path} has duplicate utterance ids")
    return by_id


def alignment_summary(alignments: dict[str, dict], metadata: pd.DataFrame, n_worst: int = 25):
    """Counts, score quantiles and the lowest-scoring utterances of an alignment file."""
    records = [alignments[uid] for uid in metadata["utterance_id"] if uid in alignments]
    scores = np.array([r["log_prob"] for r in records if r["log_prob"] is not None])
    text = dict(zip(metadata["utterance_id"], metadata["standardized_text"], strict=True))
    worst = sorted((r for r in records if r["log_prob"] is not None), key=lambda r: r["log_prob"])[
        :n_worst
    ]
    return {
        "utterances": len(metadata),
        "in_file": len(records),
        "failed": sum(r["words"] is None for r in records),
        "no_words": sum(r["words"] == [] for r in records),
        "log_prob_quantiles": {
            str(q): round(float(np.quantile(scores, q)), 4) for q in (0.01, 0.05, 0.25, 0.5)
        }
        if len(scores)
        else {},
        "worst": [
            {
                "utterance_id": r["utterance_id"],
                "log_prob": r["log_prob"],
                "text": text[r["utterance_id"]],
            }
            for r in worst
        ],
    }


def add_extra_data(
    ds: datasets.Dataset, alignments: dict[str, dict], extra_dir: Path
) -> tuple[datasets.Dataset, dict[str, dict]]:
    """Append the clips and word timings that scripts/build_ssc.py wrote to `extra_dir`.

    Rows keep the training set's columns, so the clips are used exactly like
    NPSC utterances.
    """
    parts = sorted((extra_dir / "clips").glob("*.parquet"))
    if not parts:
        raise FileNotFoundError(f"No clips in {extra_dir / 'clips'}")
    extra = datasets.Dataset.from_parquet([str(p) for p in parts], keep_in_memory=False)
    extra_alignments = {}
    for path in sorted((extra_dir / "alignments").glob("*.jsonl")):
        extra_alignments |= read_alignments(path)
    if set(alignments) & set(extra_alignments):
        raise ValueError("Extra clips share utterance ids with the training set")
    extra = extra.select_columns(ds.column_names).cast(ds.features)
    return datasets.concatenate_datasets([ds, extra]), alignments | extra_alignments


def timings_of(record: dict) -> list[WordTiming]:
    return [WordTiming(w, s, e) for w, s, e in record["words"]]


@dataclass(frozen=True)
class Example:
    row: int  # position in the training split's Dataset
    utterance_id: str
    seconds: float
    timings: list[WordTiming]


def select_examples(
    metadata: pd.DataFrame,
    alignments: dict[str, dict],
    processor: ProcessorMixin,
    layout: StreamLayout | None,
) -> tuple[list[Example], Counter]:
    """The training utterances usable as targets, and why the others are not.

    `metadata` rows must be in the order of the Dataset the examples index.
    With a `layout` (Realtime), clips whose words don't fit the stream are
    dropped; without one (Voxtral Mini), that check is skipped.
    """
    examples, dropped = [], Counter()
    for row, utt in enumerate(metadata.itertuples(index=False)):
        record = alignments.get(utt.utterance_id)
        if "<INAUDIBLE>" in utt.raw_text:
            # Words were spoken that the transcript lacks, so the alignment
            # and the target would both be wrong there; scoring drops them too.
            reason = DROP_INAUDIBLE
        elif record is None:
            reason = DROP_NOT_IN_FILE
        elif record["words"] is None:
            reason = DROP_NOT_ALIGNED
        else:
            timings = timings_of(record)
            if layout is not None and not _fits(processor, layout, timings, record["n_samples"]):
                reason = DROP_NO_FIT
            else:
                seconds = record["n_samples"] / SAMPLE_RATE
                examples.append(Example(row, utt.utterance_id, seconds, timings))
                continue
        dropped[reason] += 1
    return examples, dropped


def _fits(processor, layout: StreamLayout, timings: list[WordTiming], n_samples: int) -> bool:
    words = word_tokens(processor, timings, space_after_marker=True)
    return build_sequence(layout, words, n_positions(processor, n_samples)) is not None


def length_grouped_batches(
    durations: Sequence[float], batch_size: int, seed: int, pool_batches: int = 50
) -> list[list[int]]:
    """Shuffled batches of indices, each holding clips of similar length.

    The indices are shuffled, cut into pools of `pool_batches` batches, sorted
    by duration within a pool and split into batches; then the batches are
    shuffled. Similar lengths mean less padding, the pools keep it random.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(durations)).tolist()
    pool = batch_size * pool_batches
    batches = []
    for start in range(0, len(order), pool):
        part = sorted(order[start : start + pool], key=lambda i: durations[i])
        batches += [part[j : j + batch_size] for j in range(0, len(part), batch_size)]
    return [batches[i] for i in rng.permutation(len(batches))]
