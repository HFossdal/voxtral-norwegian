"""Training batches for Voxtral Realtime, and the LoRA setup both models use."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import PreTrainedModel, ProcessorMixin

from voxnor.align import WordTiming
from voxnor.data import SAMPLE_RATE
from voxnor.realtime_stream import IGNORE_INDEX, StreamLayout, build_sequence, tekken

# Decoder attention and MLP projections; the audio encoder has layers with the
# same names, so the pattern is anchored to the language model.
DECODER_LORA_TARGETS = (
    r"model\.language_model\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
)
# The same projections in the audio encoder, plus the projector that maps its
# output into the decoder's embedding space. Voxtral Mini's encoder is
# Whisper's, with `out_proj`, `fc1` and `fc2`; Realtime has none of these.
ENCODER_LORA_TARGETS = (
    r"model\.audio_tower\.layers\.\d+\."
    r"(self_attn\.([qkvo]|out)_proj|mlp\.(gate|up|down)_proj|fc[12])"
    r"|model\.multi_modal_projector\.linear_[12]"
)


def n_positions(processor: ProcessorMixin, n_samples: int) -> int:
    """Stream length the processor gives a clip: left pad + audio frames + right pad."""
    audio_config = processor.mistral_common_audio_config
    frames = math.ceil(n_samples / audio_config.raw_audio_length_per_tok)
    return audio_config.n_left_pad_tokens + frames + processor.num_right_pad_tokens


def word_tokens(
    processor: ProcessorMixin, timings: Sequence[WordTiming], space_after_marker: bool
) -> list[tuple[list[int], list[int], float]]:
    """Per word: tokens after a marker, tokens when joined to the previous word, end time."""
    tokenizer = tekken(processor)
    marker_prefix = " " if space_after_marker else ""

    def encode(text: str) -> list[int]:
        return tokenizer.encode(text, bos=False, eos=False)

    return [(encode(marker_prefix + t.word), encode(" " + t.word), t.end) for t in timings]


def fits_stream(
    processor: ProcessorMixin,
    layout: StreamLayout,
    n_samples: int,
    timings: Sequence[WordTiming],
    space_after_marker: bool,
) -> bool:
    """Whether a clip of `n_samples` has room for all its words in the stream."""
    words = word_tokens(processor, timings, space_after_marker)
    return build_sequence(layout, words, n_positions(processor, n_samples)) is not None


def build_batch(
    processor: ProcessorMixin,
    layout: StreamLayout,
    audios: Sequence[np.ndarray],
    timings: Sequence[Sequence[WordTiming]],
    space_after_marker: bool,
) -> tuple[dict[str, torch.Tensor], list[int]]:
    """Model inputs with labels for the clips whose words fit, and the indices kept.

    Clips are padded with silence to the longest one; positions past a
    clip's own end get no loss.
    """
    sequences, kept = [], []
    for i, (audio, clip_timings) in enumerate(zip(audios, timings, strict=True)):
        words = word_tokens(processor, clip_timings, space_after_marker)
        seq = build_sequence(layout, words, n_positions(processor, len(audio)))
        if seq is not None:
            sequences.append(seq)
            kept.append(i)
    if not sequences:
        raise ValueError("No clip in the batch fits its words into its stream")

    inputs = processor([audios[i] for i in kept], sampling_rate=SAMPLE_RATE, return_tensors="pt")
    total = math.ceil(inputs["input_features"].shape[-1] / processor.audio_length_per_tok)
    input_ids = torch.full((len(kept), total), layout.pad_id)
    labels = torch.full((len(kept), total), IGNORE_INDEX)
    for row, (ids, labs) in enumerate(sequences):
        input_ids[row, : len(ids)] = torch.tensor(ids)
        labels[row, : len(labs)] = torch.tensor(labs)
    batch = {
        "input_ids": input_ids,
        "labels": labels,
        "input_features": inputs["input_features"],
        "num_delay_tokens": inputs["num_delay_tokens"],
    }
    return batch, kept


def add_lora(
    model: PreTrainedModel, rank: int, alpha: int, dropout: float, encoder: bool = False
) -> PeftModel:
    """LoRA on the language decoder, and on the audio encoder if `encoder`; the rest is frozen."""
    targets = f"{DECODER_LORA_TARGETS}|{ENCODER_LORA_TARGETS}" if encoder else DECODER_LORA_TARGETS
    config = LoraConfig(r=rank, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets)
    return get_peft_model(model, config)


def loss_breakdown(logits: torch.Tensor, labels: torch.Tensor, pad_id: int) -> dict[str, float]:
    """Mean cross-entropy on pad targets and on the rest (markers and text), separately."""
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    per_token = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).view_as(labels)
    pad = labels == pad_id
    text = (labels != IGNORE_INDEX) & ~pad
    return {
        "pad_loss": per_token[pad].mean().item() if pad.any() else float("nan"),
        "text_loss": per_token[text].mean().item() if text.any() else float("nan"),
        "text_accuracy": (logits.argmax(-1)[text] == labels[text]).float().mean().item()
        if text.any()
        else float("nan"),
    }
