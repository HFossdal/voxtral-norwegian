"""Transcribe a split to a resumable predictions file, then score it."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import datasets
import pandas as pd

from voxnor.analysis import score_utterances, summarize
from voxnor.data import decode_audio
from voxnor.transcribe import Transcriber

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = ("utterance_id", "hypothesis", "truncated")


def transcribe_to_file(
    ds: datasets.Dataset, transcriber: Transcriber, path: Path, batch_size: int
) -> None:
    """Append one JSON line per utterance to `path`, skipping ids already in it.

    Lines are flushed after every batch, so an interrupted run (e.g. a Colab
    disconnect) resumes where it stopped. Longest clips go first: batches then
    hold similar lengths (less padding), and running out of GPU memory shows
    up in the first batch rather than at the end.
    """
    done = set(read_predictions(path)["utterance_id"]) if path.exists() else set()
    durations = ds["duration"]
    todo = [i for i, uid in enumerate(ds["utterance_id"]) if uid not in done]
    todo.sort(key=lambda i: -durations[i])
    if not todo:
        log.info("All %d utterances already transcribed in %s", len(done), path)
        return
    log.info("Transcribing %d utterances (%d already done)", len(todo), len(done))

    start, audio_seconds = time.perf_counter(), 0.0
    with path.open("a", encoding="utf-8") as out:
        for n, offset in enumerate(range(0, len(todo), batch_size)):
            batch = ds.select(todo[offset : offset + batch_size])
            audios = [decode_audio(a) for a in batch["utterance_audio_file"]]
            hypotheses = transcriber.transcribe(audios)
            # Only the offline model can be cut off; Realtime output length is fixed.
            truncated = getattr(transcriber, "last_truncated", None) or [False] * len(hypotheses)
            for uid, hyp, cut in zip(batch["utterance_id"], hypotheses, truncated, strict=True):
                line = {"utterance_id": uid, "hypothesis": hyp, "truncated": cut}
                out.write(json.dumps(line, ensure_ascii=False) + "\n")
            out.flush()
            audio_seconds += sum(batch["duration"])
            if n % 20 == 0 or offset + batch_size >= len(todo):
                elapsed = time.perf_counter() - start
                log.info(
                    "%d/%d utterances, %.1fx real time",
                    min(offset + batch_size, len(todo)),
                    len(todo),
                    audio_seconds / elapsed,
                )


def read_predictions(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows, columns=list(PREDICTION_COLUMNS))
    if df["utterance_id"].duplicated().any():
        raise ValueError(f"{path} has duplicate utterance ids")
    return df


def score_predictions(
    predictions: pd.DataFrame, metadata: pd.DataFrame, train_speakers: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Join predictions with references and metadata; return per-utterance scores and summary."""
    missing = set(metadata["utterance_id"]) - set(predictions["utterance_id"])
    if missing:
        raise ValueError(
            f"{len(missing)} utterances have no prediction, e.g. {sorted(missing)[:3]}"
        )
    merged = metadata.merge(predictions, on="utterance_id", how="left", validate="one_to_one")
    merged["seen_speaker"] = merged["speaker_id"].isin(train_speakers)
    scored = score_utterances(merged)
    summary = summarize(scored)
    summary["truncated_utterances"] = int(merged["truncated"].sum())
    return scored, summary
