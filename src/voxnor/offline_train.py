"""Training batches for Voxtral Mini's offline transcription mode.

The input is the model's own transcription prompt, as in zero-shot
evaluation (`OfflineTranscriber`): the audio (padded to 30 s), the language
hint, `[TRANSCRIBE]`. The target is the transcript and the end token, with
the loss on those tokens only. The transcript is the same word sequence the
Realtime model is trained on (`standardized_text`, via the alignment file),
so both models learn the same text.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from transformers import ProcessorMixin

from voxnor.data import SAMPLE_RATE
from voxnor.realtime_stream import IGNORE_INDEX, tekken


def target_tokens(processor: ProcessorMixin, text: str) -> list[int]:
    """The transcript's tokens and the end token (no leading space: the model
    starts a transcript with a word token)."""
    tokenizer = tekken(processor)
    return [*tokenizer.encode(text, bos=False, eos=False), tokenizer.eos_id]


def build_offline_batch(
    processor: ProcessorMixin,
    model_id: str,
    language: str | None,
    audios: Sequence[np.ndarray],
    texts: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Prompt + transcript per clip, right-padded; labels only on the transcript."""
    inputs = processor.apply_transcription_request(
        audio=list(audios),
        model_id=model_id,
        language=language,
        sampling_rate=SAMPLE_RATE,
        format=["wav"] * len(audios),
        return_tensors="pt",
    )
    # The processor pads prompts on the left for generation; take each one unpadded.
    prompts = [
        ids[mask.bool()].tolist()
        for ids, mask in zip(inputs["input_ids"], inputs["attention_mask"], strict=True)
    ]
    targets = [target_tokens(processor, text) for text in texts]
    length = max(len(p) + len(t) for p, t in zip(prompts, targets, strict=True))
    pad_id = processor.tokenizer.pad_token_id
    input_ids = torch.full((len(prompts), length), pad_id)
    labels = torch.full((len(prompts), length), IGNORE_INDEX)
    attention_mask = torch.zeros((len(prompts), length), dtype=torch.long)
    for row, (prompt, target) in enumerate(zip(prompts, targets, strict=True)):
        end = len(prompt) + len(target)
        input_ids[row, :end] = torch.tensor(prompt + target)
        labels[row, len(prompt) : end] = torch.tensor(target)
        attention_mask[row, :end] = 1
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "input_features": inputs["input_features"],
    }


def text_loss(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    """Mean cross-entropy and next-token accuracy on the transcript tokens."""
    logits, labels = logits[:, :-1].float(), labels[:, 1:]
    target = labels != IGNORE_INDEX
    loss = torch.nn.functional.cross_entropy(logits[target], labels[target])
    accuracy = (logits.argmax(-1)[target] == labels[target]).float().mean()
    return {"text_loss": loss.item(), "text_accuracy": accuracy.item()}
