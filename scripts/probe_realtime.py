"""Look at Voxtral Realtime's raw token stream and calibrate word placement.

    python scripts/probe_realtime.py --config configs/cloud.yaml --out <dir>

1. English LibriSpeech clips (a language the model supports): its own
   streams, and for each emitted word the offset between where the model put
   it and where our formula (`StreamLayout.emission_position`) puts it,
   using word end times from an English CTC aligner. Offsets near a
   constant mean the formula reproduces the model's own convention. The
   first run (2026-10-20) found a median of 2 frames, now built into the
   formula as `EMISSION_OFFSET`, so a re-run should centre on 0.
2. Norwegian validation clips: the streams, and word timings from the
   Norwegian aligner, to check the aligner by eye.

Writes probe.json into --out and prints a summary.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download

from voxnor.align import CtcAligner
from voxnor.config import add_override_argument, load_config
from voxnor.data import decode_audio, load_split, read_id_list, select_rows
from voxnor.model import load_model, resolve_device
from voxnor.realtime_stream import layout_from_processor, parse_stream, render_stream, tekken
from voxnor.transcribe import RealtimeTranscriber, load_processor

EN_ALIGNER = "facebook/wav2vec2-base-960h"
NB_ALIGNER = "NbAiLab/nb-wav2vec2-300m-bokmaal-v2"
N_CLIPS = 4


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True, type=Path)
    add_override_argument(parser)
    args = parser.parse_args()
    cfg = load_config(args.config, args.set)
    device = resolve_device(cfg.runtime.device)

    model = load_model(cfg.model, device).eval()
    processor = load_processor(cfg.model)
    transcriber = RealtimeTranscriber(model, processor, cfg.model.transcription_delay_ms)
    layout = layout_from_processor(processor)
    tokenizer = tekken(processor)

    clips = []
    for lang, audios, texts, aligner_id in (
        ("en", *english_clips(), EN_ALIGNER),
        ("nb", *norwegian_clips(cfg), NB_ALIGNER),
    ):
        aligner = CtcAligner.from_pretrained(aligner_id, device)
        for audio, text in zip(audios, texts, strict=True):
            ids = transcriber.generate_ids([audio])[0].tolist()
            pieces = [tokenizer.id_to_piece(i) for i in ids]
            emitted = parse_stream(ids, pieces, layout)
            timings = aligner.align(audio, text.split()) or []
            clips.append(
                {
                    "lang": lang,
                    "reference": text,
                    "seconds": round(len(audio) / 16000, 2),
                    "stream": render_stream(ids, pieces, layout),
                    "eos_in_stream": tokenizer.eos_id in ids,
                    "emitted": [vars(w) for w in emitted],
                    "timings": [vars(t) for t in timings],
                    "offsets": offsets(emitted, timings, layout),
                }
            )
        del aligner

    summary = summarize(clips, layout)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "probe.json").write_text(
        json.dumps({"summary": summary, "clips": clips}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for clip in clips:
        print(f"\n[{clip['lang']}] {clip['reference']}\n{clip['stream']}")
    print("\n" + json.dumps(summary, indent=2))


def english_clips() -> tuple[list[np.ndarray], list[str]]:
    path = hf_hub_download(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean/validation/0000.parquet",
        repo_type="dataset",
        revision="refs/convert/parquet",
    )
    rows = pd.read_parquet(path).head(N_CLIPS)
    return [decode_audio(a) for a in rows["audio"]], list(rows["text"])


def norwegian_clips(cfg) -> tuple[list[np.ndarray], list[str]]:
    ids = read_id_list("subsets/local/validation.txt")[:N_CLIPS]
    ds = select_rows(load_split(cfg.data, "validation"), ids)
    return [decode_audio(a) for a in ds["utterance_audio_file"]], list(ds["standardized_text"])


def offsets(emitted, timings, layout) -> list[int]:
    """Model position minus formula position, for emitted words matching the reference."""
    hyp = [_plain(w.text) for w in emitted]
    ref = [_plain(t.word) for t in timings]
    matcher = difflib.SequenceMatcher(None, ref, hyp, autojunk=False)
    out = []
    for block in matcher.get_matching_blocks():
        for k in range(block.size):
            word, timing = emitted[block.b + k], timings[block.a + k]
            out.append(word.position - layout.emission_position(timing.end))
    return out


def summarize(clips: list[dict], layout) -> dict:
    english = [c for c in clips if c["lang"] == "en"]
    all_offsets = [o for c in english for o in c["offsets"]]
    emitted = [w for c in clips for w in c["emitted"]]
    return {
        "layout": vars(layout),
        "english_matched_words": len(all_offsets),
        "english_offset_counts": dict(sorted(Counter(all_offsets).items())),
        "english_offset_median": float(np.median(all_offsets)) if all_offsets else None,
        "share_of_words_with_marker": round(sum(w["marked"] for w in emitted) / len(emitted), 3)
        if emitted
        else None,
        "streams_with_eos": sum(c["eos_in_stream"] for c in clips),
        "cuda": torch.cuda.is_available(),
    }


def _plain(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


if __name__ == "__main__":
    main()
