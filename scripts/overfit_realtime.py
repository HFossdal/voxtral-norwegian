"""Feasibility check: can LoRA teach Voxtral Realtime a few Norwegian clips?

    python scripts/overfit_realtime.py --config configs/cloud.yaml --out <dir>

Takes the first --n utterances of the committed train subset, aligns their
words, transcribes them with the base model, trains a decoder LoRA
adapter on just these clips for --steps steps, and transcribes them again.
Go criterion: the loss falls towards zero and the clips come out close to
their transcripts. This shows the training sequences are in a form the model
can learn and decode; it says nothing about generalisation.

Writes overfit.json into --out and prints the losses and transcripts.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from voxnor.align import CtcAligner
from voxnor.config import add_override_argument, load_config
from voxnor.data import decode_audio, parquet_file, read_id_list, read_rows
from voxnor.metrics import error_rate, word_errors
from voxnor.model import load_model, resolve_device
from voxnor.normalize import normalize_hypothesis, normalize_reference
from voxnor.realtime_stream import layout_from_processor, render_stream, tekken
from voxnor.realtime_train import add_lora, build_batch, loss_breakdown
from voxnor.transcribe import RealtimeTranscriber, load_processor

ALIGNER = "NbAiLab/nb-wav2vec2-300m-bokmaal-v2"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument(
        "--no-marker-space",
        action="store_true",
        help="Write a word's first token without a leading space after [STREAMING_WORD]",
    )
    add_override_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg.runtime.device)
    torch.manual_seed(cfg.seed)

    ids = read_id_list("subsets/local/train.txt")[: args.n]
    rows = read_rows(parquet_file(cfg.data.dataset_id, "train", download=False), ids)
    audios = [decode_audio(a) for a in rows.column("utterance_audio_file").to_pylist()]
    texts = rows.column("standardized_text").to_pylist()

    aligner = CtcAligner.from_pretrained(ALIGNER, device)
    timings = [
        aligner.align(audio, text.split()) or [] for audio, text in zip(audios, texts, strict=True)
    ]
    del aligner

    model = load_model(cfg.model, device)
    processor = load_processor(cfg.model)
    transcriber = RealtimeTranscriber(model, processor, cfg.model.transcription_delay_ms)
    layout = layout_from_processor(processor)
    batch, kept = build_batch(
        processor, layout, audios, timings, space_after_marker=not args.no_marker_space
    )
    batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    batch["input_features"] = batch["input_features"].to(model.dtype)
    audios, texts = [audios[i] for i in kept], [texts[i] for i in kept]

    model.eval()
    before = transcriber.transcribe(audios)

    model = add_lora(model, rank=args.rank, alpha=2 * args.rank, dropout=0.0)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    model.train()
    log, start = [], time.perf_counter()
    for step in range(1, args.steps + 1):
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step == 1 or step % 10 == 0:
            entry = {"step": step, "loss": round(out.loss.item(), 4)}
            with torch.no_grad():
                parts = loss_breakdown(out.logits, batch["labels"], layout.pad_id)
            entry |= {k: round(v, 4) for k, v in parts.items()}
            log.append(entry)
            print(entry, flush=True)
    train_seconds = time.perf_counter() - start

    model.eval()
    after = transcriber.transcribe(audios)
    tokenizer = tekken(processor)
    stream_ids = transcriber.generate_ids(audios[:1])[0].tolist()
    target_ids = batch["input_ids"][0, : len(stream_ids)].tolist()

    result = {
        "utterances": [
            {"id": ids[i], "reference": t, "before": b, "after": a}
            for i, t, b, a in zip(kept, texts, before, after, strict=True)
        ],
        "wer_before": _wer(texts, before),
        "wer_after": _wer(texts, after),
        "log": log,
        "train_seconds": round(train_seconds, 1),
        "trainable_parameters": sum(p.numel() for p in params),
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)
        if torch.cuda.is_available()
        else None,
        "target_stream_first_clip": _render(target_ids, tokenizer, layout),
        "output_stream_first_clip": _render(stream_ids, tokenizer, layout),
        "args": vars(args) | {"out": str(args.out)},
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overfit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for u in result["utterances"]:
        print(f"\nref:    {u['reference']}\nbefore: {u['before']}\nafter:  {u['after']}")
    print(f"\ntarget: {result['target_stream_first_clip']}")
    print(f"output: {result['output_stream_first_clip']}")
    print(f"\nWER before {result['wer_before']:.1%}, after {result['wer_after']:.1%}")


def _wer(references: list[str], hypotheses: list[str]) -> float:
    return error_rate(
        [
            word_errors(normalize_reference(r), normalize_hypothesis(h))
            for r, h in zip(references, hypotheses, strict=True)
        ]
    )


def _render(ids: list[int], tokenizer, layout) -> str:
    return render_stream(ids, [tokenizer.id_to_piece(i) for i in ids], layout)


if __name__ == "__main__":
    main()
