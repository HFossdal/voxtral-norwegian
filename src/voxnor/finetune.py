"""LoRA fine-tuning with resumable state: Voxtral Realtime or
Voxtral Mini's offline transcription mode.

A run directory holds:
    train_state.pt          latest LoRA weights, optimizer, scheduler, step, RNG
    train_log.jsonl         loss etc. every `log_every` steps (after a resume,
                            steps between the last save and the interruption
                            appear twice; the later entry is the one that counts)
    adapters/step-NNNNNN/   the adapter at every evaluation step (PEFT format)
    selection/              predictions on the selection set, per evaluated step
    selection.jsonl         WER on the selection set per evaluated step
    best.json               the chosen step (lowest selection WER)

On several GPUs (`voxnor.distributed`), only the main process writes
these files; all processes read the training state.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import datasets
import numpy as np
import pandas as pd
import torch
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import ProcessorMixin, get_linear_schedule_with_warmup

from voxnor.augment import augment
from voxnor.config import TrainConfig
from voxnor.data import decode_audio
from voxnor.distributed import (
    SINGLE,
    Dist,
    barrier,
    broadcast_parameters,
    shard,
    step_seed,
    sum_gradients,
    sum_scalar,
)
from voxnor.evaluate import read_predictions, score_predictions, transcribe_to_file
from voxnor.offline_train import build_offline_batch, text_loss
from voxnor.realtime_stream import IGNORE_INDEX, StreamLayout
from voxnor.realtime_train import build_batch, fits_stream, loss_breakdown
from voxnor.train_data import Example, length_grouped_batches
from voxnor.transcribe import Transcriber

log = logging.getLogger(__name__)

STATE_FILE = "train_state.pt"
LOG_FILE = "train_log.jsonl"
SELECTION_FILE = "selection.jsonl"
BEST_FILE = "best.json"


class TrainingBatches(Protocol):
    """Turns examples into model inputs with labels, for one model family."""

    def make(self, examples: Sequence[Example], rng: np.random.Generator | None) -> dict: ...

    def breakdown(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]: ...


@dataclass
class RealtimeBatches:
    """Time-aligned token streams, augmented if an `rng` is given."""

    processor: ProcessorMixin
    layout: StreamLayout
    ds: datasets.Dataset

    def make(self, examples: Sequence[Example], rng: np.random.Generator | None) -> dict:
        return make_batch(self.processor, self.layout, self.ds, examples, rng)

    def breakdown(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return loss_breakdown(logits, labels, self.layout.pad_id)


@dataclass
class OfflineBatches:
    """Voxtral Mini's transcription prompt + the transcript."""

    processor: ProcessorMixin
    model_id: str
    language: str | None
    ds: datasets.Dataset

    def make(self, examples: Sequence[Example], rng: np.random.Generator | None) -> dict:
        clips = load_clips(self.ds, examples)
        if rng is not None:
            clips = [augment(audio, timings, rng) for audio, timings in clips]
        texts = [" ".join(w.word for w in e.timings) for e in examples]
        audios = [audio for audio, _ in clips]
        return build_offline_batch(self.processor, self.model_id, self.language, audios, texts)

    def breakdown(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        return text_loss(logits, labels)


def epoch_batches(examples: Sequence[Example], batch_size: int, seed: int, epoch: int):
    """The batches of one epoch; fixed by seed and epoch, so a resumed run sees the same."""
    return length_grouped_batches(
        [e.seconds for e in examples], batch_size, seed=1000 * seed + epoch
    )


def adapter_dir(run_dir: Path, step: int) -> Path:
    return run_dir / "adapters" / f"step-{step:06d}"


def is_eval_step(step: int, cfg: TrainConfig, total_steps: int) -> bool:
    return step % cfg.eval_every == 0 or step == total_steps


def train(
    model: PeftModel,
    batches: TrainingBatches,
    examples: Sequence[Example],
    cfg: TrainConfig,
    seed: int,
    run_dir: Path,
    on_eval: Callable[[int], None],
    dist: Dist = SINGLE,
) -> None:
    """Train, or continue from `run_dir`'s saved state, to the end of the last epoch.

    `on_eval(step)` runs at step 0 and every `eval_every` steps, in every
    process; it must be idempotent, since a resumed run calls it again for
    its starting step.

    With several processes, each takes a contiguous part of every batch. When
    that part is a whole number of micro-batches, the micro-batches, their
    augmentation and so the gradient are those of a single-GPU run; only the
    dropout masks differ.
    """
    batches_per_epoch = len(epoch_batches(examples, cfg.batch_size, seed, 0))
    total_steps = batches_per_epoch * cfg.epochs
    params = [p for p in model.parameters() if p.requires_grad]
    broadcast_parameters(params, dist)
    optimizer = torch.optim.AdamW(params, lr=cfg.learning_rate, weight_decay=0.0)
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)
    state_path = run_dir / STATE_FILE
    step = load_state(state_path, model, optimizer, scheduler) if state_path.exists() else 0
    log.info(
        "Step %d of %d (%d per epoch, %d processes)",
        step,
        total_steps,
        batches_per_epoch,
        dist.world_size,
    )
    if is_eval_step(step, cfg, total_steps):
        on_eval(step)

    start = time.perf_counter()
    order, order_epoch = [], -1
    log_file = (run_dir / LOG_FILE).open("a", encoding="utf-8") if dist.is_main else None
    try:
        while step < total_steps:
            epoch, k = divmod(step, batches_per_epoch)
            if epoch != order_epoch:
                order, order_epoch = epoch_batches(examples, cfg.batch_size, seed, epoch), epoch
            chosen = [examples[i] for i in order[k]]
            first, stop = shard(len(chosen), dist)
            micro_batches = [
                batches.make(
                    chosen[j : min(j + cfg.micro_batch_size, stop)],
                    # Seeded by step, so a resumed run augments the same way.
                    np.random.default_rng([seed, step, j]) if cfg.augment else None,
                )
                for j in range(first, stop, cfg.micro_batch_size)
            ]

            if dist.enabled:
                torch.manual_seed(step_seed(seed, step, dist))
            model.train()
            loss, out, labels = accumulate_gradients(model, micro_batches, dist)
            sum_gradients(params, dist)
            grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if log_file and out is not None and (step == 1 or step % cfg.log_every == 0):
                with torch.no_grad():  # on the last micro-batch only
                    parts = batches.breakdown(out.logits, labels)
                entry = {
                    "step": step,
                    "epoch": epoch,
                    "loss": round(loss, 4),
                    **{key: round(v, 4) for key, v in parts.items()},
                    "grad_norm": round(grad_norm.item(), 4),
                    "lr": scheduler.get_last_lr()[0],
                    "seconds": round(time.perf_counter() - start, 1),
                    "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)
                    if torch.cuda.is_available()
                    else None,
                }
                log_file.write(json.dumps(entry) + "\n")
                log_file.flush()
                log.info("%s", entry)
            del out, labels, micro_batches
            saving = step % cfg.save_every == 0 or is_eval_step(step, cfg, total_steps)
            if saving and dist.is_main:
                save_state(state_path, model, optimizer, scheduler, step)
            if is_eval_step(step, cfg, total_steps):
                on_eval(step)
    finally:
        if log_file:
            log_file.close()


def accumulate_gradients(model: PeftModel, micro_batches: Sequence[dict], dist: Dist = SINGLE):
    """Backward over micro-batches, with the gradient of one batch made of all of them.

    The model's loss is a mean over a micro-batch's target tokens, so each is
    weighted by its share of the batch's target tokens, counted over all
    processes; summing the processes' gradients then gives the batch's.
    Returns the batch loss and the last micro-batch's output and labels (on
    the device; None for a process without micro-batches).
    """
    counts = [int((b["labels"][:, 1:] != IGNORE_INDEX).sum()) for b in micro_batches]
    total = sum_scalar(sum(counts), dist, model.device)
    loss, out, batch = 0.0, None, {"labels": None}
    for batch, count in zip(micro_batches, counts, strict=True):
        batch = {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}
        batch["input_features"] = batch["input_features"].to(model.dtype)
        out = model(**batch)
        (out.loss * (count / total)).backward()
        loss += out.loss.item() * count / total
    return sum_scalar(loss, dist, model.device), out, batch["labels"]


def make_batch(
    processor: ProcessorMixin,
    layout: StreamLayout,
    ds: datasets.Dataset,
    examples: Sequence[Example],
    rng: np.random.Generator | None = None,
) -> dict[str, torch.Tensor]:
    """Realtime model inputs for `examples`, augmented if an `rng` is given."""
    clips = load_clips(ds, examples)
    if rng is not None:
        augmented = [augment(audio, timings, rng) for audio, timings in clips]
        # A faster clip can have too little time for its words: keep that one as recorded.
        clips = [
            new
            if fits_stream(processor, layout, len(new[0]), new[1], space_after_marker=True)
            else old
            for new, old in zip(augmented, clips, strict=True)
        ]
    batch, kept = build_batch(
        processor, layout, [a for a, _ in clips], [t for _, t in clips], space_after_marker=True
    )
    if len(kept) != len(examples):
        raise ValueError("A selected example does not fit its stream")  # checked in selection
    return batch


def load_clips(ds: datasets.Dataset, examples: Sequence[Example]) -> list[tuple[np.ndarray, list]]:
    """Each example's audio and word timings."""
    rows = ds.select([e.row for e in examples])
    if list(rows["utterance_id"]) != [e.utterance_id for e in examples]:
        raise ValueError("Examples do not match the dataset rows they index")
    return [
        (decode_audio(a), e.timings)
        for a, e in zip(rows["utterance_audio_file"], examples, strict=True)
    ]


def save_state(path: Path, model: PeftModel, optimizer, scheduler, step: int) -> None:
    state = {
        "step": step,
        "lora": get_peft_model_state_dict(model),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # never leaves a half-written state behind


def load_state(path: Path, model: PeftModel, optimizer, scheduler) -> int:
    state = torch.load(path, map_location="cpu", weights_only=True)
    set_peft_model_state_dict(model, state["lora"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["rng"])
    if state["cuda_rng"]:
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state["step"]


class SelectionEval:
    """Saves the adapter and scores the selection set at an evaluation step."""

    def __init__(
        self,
        model: PeftModel,
        transcriber: Transcriber,
        ds: datasets.Dataset,
        metadata: pd.DataFrame,
        train_speakers: set[str],
        run_dir: Path,
        batch_size: int,
        dist: Dist = SINGLE,
    ):
        self.model, self.transcriber, self.ds = model, transcriber, ds
        self.metadata, self.train_speakers = metadata, train_speakers
        self.run_dir, self.batch_size, self.dist = run_dir, batch_size, dist

    def __call__(self, step: int) -> None:
        if step in read_selection(self.run_dir):
            return
        if step > 0 and self.dist.is_main:  # step 0 is the base model: no adapter
            self.model.save_pretrained(adapter_dir(self.run_dir, step))
        predictions = self.run_dir / "selection" / f"predictions_step-{step:06d}.jsonl"
        predictions.parent.mkdir(exist_ok=True)
        self.model.eval()
        self.transcribe(predictions)
        if self.dist.is_main:
            self.score(step, predictions)
        barrier(self.dist)  # every process sees the result before training on

    def transcribe(self, predictions: Path) -> None:
        if not self.dist.enabled:
            transcribe_to_file(self.ds, self.transcriber, predictions, self.batch_size)
            return
        # Each process transcribes its part; the main process joins the parts.
        part = predictions.with_suffix(f".part{self.dist.rank}.jsonl")
        first, stop = shard(len(self.ds), self.dist)
        rows = self.ds.select(range(first, stop))
        transcribe_to_file(rows, self.transcriber, part, self.batch_size)
        barrier(self.dist)
        if self.dist.is_main:
            parts = [
                predictions.with_suffix(f".part{r}.jsonl") for r in range(self.dist.world_size)
            ]
            joined = "".join(p.read_text(encoding="utf-8") for p in parts)
            predictions.write_text(joined, encoding="utf-8")

    def score(self, step: int, predictions: Path) -> None:
        _, summary = score_predictions(
            read_predictions(predictions), self.metadata, self.train_speakers
        )
        wer, cer = summary["overall"]["wer"], summary["overall"]["cer"]
        result = {
            "step": step,
            "wer": wer["value"],
            "wer_low": wer["low"],
            "wer_high": wer["high"],
            "cer": cer["value"],
            "utterances_scored": summary["utterances_scored"],
        }
        with (self.run_dir / SELECTION_FILE).open("a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")
        log.info("Selection set at step %d: WER %.2f%%", step, 100 * wer["value"])


def read_selection(run_dir: Path) -> dict[int, dict]:
    path = run_dir / SELECTION_FILE
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return {r["step"]: r for r in map(json.loads, filter(str.strip, f))}


def choose_best(run_dir: Path) -> dict:
    """The evaluated step with the lowest selection WER (step 0, the base model, excluded)."""
    results = read_selection(run_dir)
    trained = {s: r for s, r in results.items() if s > 0}
    if not trained:
        raise ValueError(f"No evaluated training step in {run_dir}")
    step = min(trained, key=lambda s: (trained[s]["wer"], s))
    best = {
        "rule": "lowest WER on the selection set (unseen-speaker validation)",
        "step": step,
        "adapter": str(adapter_dir(run_dir, step)),
        "selection_wer": trained[step]["wer"],
        "base_model_selection_wer": results.get(0, {}).get("wer"),
        "evaluated_steps": sorted(results),
    }
    (run_dir / BEST_FILE).write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    return best
