"""Batched greedy transcription for both Voxtral families behind one interface."""

from __future__ import annotations

import warnings
from typing import Protocol

import numpy as np
import torch
from transformers import AutoProcessor, PreTrainedModel, ProcessorMixin

from voxnor.config import ModelConfig
from voxnor.data import SAMPLE_RATE

MS_PER_REALTIME_TOKEN = 80


class Transcriber(Protocol):
    def transcribe(self, audios: list[np.ndarray]) -> list[str]: ...


def load_processor(cfg: ModelConfig) -> ProcessorMixin:
    # Tiny models use the real processor too (same vocabulary and audio format).
    processor = AutoProcessor.from_pretrained(cfg.hf_id)
    if cfg.family == "voxtral_realtime":
        # Set before anything reads the stream layout from the processor, so
        # training targets and decoding use the same delay.
        processor.mistral_common_audio_config.transcription_delay_ms = cfg.transcription_delay_ms
    return processor


def build_transcriber(
    cfg: ModelConfig, model: PreTrainedModel, processor: ProcessorMixin, max_new_tokens: int
) -> Transcriber:
    if cfg.family == "voxtral_realtime":
        return RealtimeTranscriber(model, processor, cfg.transcription_delay_ms)
    if cfg.family == "whisper":
        return WhisperTranscriber(model, processor, cfg.language, max_new_tokens)
    return OfflineTranscriber(model, processor, cfg.hf_id, cfg.language, max_new_tokens)


class RealtimeTranscriber:
    """Voxtral Realtime in offline mode: the whole clip is given at once.

    The model emits one token per 80 ms of audio, so output length is fixed
    by the audio length and no `max_new_tokens` is needed.
    """

    def __init__(self, model: PreTrainedModel, processor: ProcessorMixin, delay_ms: int):
        audio_config = processor.mistral_common_audio_config
        if audio_config.transcription_delay_ms != delay_ms:
            audio_config.transcription_delay_ms = delay_ms
        if processor.num_delay_tokens * MS_PER_REALTIME_TOKEN != delay_ms:
            raise ValueError(
                f"Processor uses {processor.num_delay_tokens} delay tokens, "
                f"which is not {delay_ms} ms"
            )
        self.model, self.processor = model, processor

    @torch.inference_mode()
    def generate_ids(self, audios: list[np.ndarray]) -> torch.Tensor:
        """The full token streams (prompt included, special tokens kept)."""
        inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with warnings.catch_warnings():
            # The model sets max_length to its number of audio tokens, but
            # transformers still warns about a "model-agnostic default". Checked:
            # output length equals the audio-token count for 6.7, 10 and 15 s clips.
            warnings.filterwarnings(
                "ignore", message="Using the model-agnostic default `max_length`"
            )
            return self.model.generate(**inputs, do_sample=False)

    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        output = self.generate_ids(audios)
        return [t.strip() for t in self.processor.batch_decode(output, skip_special_tokens=True)]


class OfflineTranscriber:
    """Voxtral Mini's dedicated transcription mode, optionally with a language hint."""

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        hf_id: str,
        language: str | None,
        max_new_tokens: int,
    ):
        self.model, self.processor = model, processor
        self.hf_id, self.language, self.max_new_tokens = hf_id, language, max_new_tokens
        # Set after each batch: which outputs hit max_new_tokens (likely cut off).
        self.last_truncated: list[bool] = []

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        inputs = self.processor.apply_transcription_request(
            audio=audios,
            model_id=self.hf_id,
            language=self.language,
            sampling_rate=SAMPLE_RATE,
            format=["wav"] * len(audios),
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        eos = self.processor.tokenizer.eos_token_id
        self.last_truncated = (~(new_tokens == eos).any(dim=1)).tolist()
        texts = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return [t.strip() for t in texts]


class WhisperTranscriber:
    """Whisper (here nb-whisper, a reference point): greedy, no timestamps.

    Clips are under 15 s, so each fits in Whisper's single 30 s window.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        processor: ProcessorMixin,
        language: str | None,
        max_new_tokens: int,
    ):
        self.model, self.processor = model, processor
        self.language, self.max_new_tokens = language, max_new_tokens
        self.last_truncated: list[bool] = []

    @torch.inference_mode()
    def transcribe(self, audios: list[np.ndarray]) -> list[str]:
        inputs = self.processor(audios, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        features = inputs.input_features.to(self.model.device, dtype=self.model.dtype)
        output = self.model.generate(
            features,
            language=self.language,
            task="transcribe",
            return_timestamps=False,
            do_sample=False,
            num_beams=1,
            max_new_tokens=self.max_new_tokens,
        )
        # Whisper's output may lack the final end token, and pads with it, so a
        # cut-off output is one with max_new_tokens real tokens.
        eos = self.model.generation_config.eos_token_id
        self.last_truncated = ((output != eos).sum(dim=1) >= self.max_new_tokens).tolist()
        texts = self.processor.batch_decode(output, skip_special_tokens=True)
        return [t.strip() for t in texts]
