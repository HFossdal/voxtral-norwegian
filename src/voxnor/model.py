"""Model construction: real pretrained checkpoints or tiny random ones for tests."""

from __future__ import annotations

import torch
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    VoxtralConfig,
    VoxtralForConditionalGeneration,
    VoxtralRealtimeConfig,
    VoxtralRealtimeForConditionalGeneration,
    WhisperConfig,
    WhisperForConditionalGeneration,
)

from voxnor.config import ModelConfig, ModelFamily

_MODEL_CLASSES: dict[ModelFamily, type[PreTrainedModel]] = {
    "voxtral_realtime": VoxtralRealtimeForConditionalGeneration,
    "voxtral": VoxtralForConditionalGeneration,
    "whisper": WhisperForConditionalGeneration,
}

# Small enough to run a forward and backward pass on a laptop CPU in about a
# second, while keeping everything the real processors depend on: layer types,
# 128 mel bins, projector shapes, the real vocabulary size and audio token id,
# and Voxtral Mini's 30 s (1500-position) audio chunks. So a tiny model can be
# driven by the real tokenizer and feature extractor end to end.
TINY_VOCAB_SIZE = 131_072
TINY_AUDIO_TOKEN_ID = 24  # Voxtral-Mini-3B's real audio token id


def load_model(cfg: ModelConfig, device: str | torch.device = "cpu") -> PreTrainedModel:
    dtype = getattr(torch, cfg.dtype)
    cls = _MODEL_CLASSES[cfg.family]
    if cfg.tiny:
        torch.manual_seed(0)
        model = cls(tiny_config(cfg.family)).to(dtype)
        if cfg.family == "whisper":
            # Language and task token ids for generate().
            model.generation_config = GenerationConfig.from_pretrained(cfg.hf_id)
    else:
        model = cls.from_pretrained(cfg.hf_id, dtype=dtype)
    if cfg.adapter:
        from peft import PeftModel

        # Merged, so inference runs at the base model's speed and the result is
        # an ordinary model for the transcribers.
        model = PeftModel.from_pretrained(model, cfg.adapter).merge_and_unload()
    return model.to(device)


def tiny_config(family: ModelFamily) -> VoxtralRealtimeConfig | VoxtralConfig | WhisperConfig:
    text = {
        "vocab_size": TINY_VOCAB_SIZE,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }
    if family == "voxtral_realtime":
        audio = {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "head_dim": 16,
            "vocab_size": TINY_VOCAB_SIZE,
        }
        return VoxtralRealtimeConfig(text_config=text, audio_config=audio)
    if family == "voxtral":
        audio = {
            "hidden_size": 32,
            # The projector folds 4 encoder frames into one audio token, so
            # intermediate_size must be 4 * hidden_size, as in the real model.
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "max_source_positions": 1500,
        }
        return VoxtralConfig(
            text_config={"model_type": "llama", **text},
            audio_config=audio,
            audio_token_id=TINY_AUDIO_TOKEN_ID,
        )
    if family == "whisper":
        # Whisper large-v3's vocabulary, 128 mel bins and 30 s input, as nb-whisper.
        return WhisperConfig(
            vocab_size=51_866,
            num_mel_bins=128,
            d_model=32,
            encoder_layers=2,
            decoder_layers=2,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=64,
            decoder_ffn_dim=64,
            max_source_positions=1500,
            max_target_positions=448,
            decoder_start_token_id=50258,
            pad_token_id=50257,
            bos_token_id=50257,
            eos_token_id=50257,
        )
    raise ValueError(f"Unknown model family {family!r}")


def dummy_batch(
    model: PreTrainedModel, batch_size: int = 2, num_audio_tokens: int = 8
) -> dict[str, torch.Tensor | int]:
    """Random inputs with the shapes `model` expects, plus labels.

    Used to check that a model (tiny or real) runs forward and backward. It is
    not a training example: the labels are random token ids.
    """
    cfg = model.config
    vocab = cfg.text_config.vocab_size
    mel_bins = cfg.audio_config.num_mel_bins
    gen = torch.Generator().manual_seed(0)
    dtype = model.dtype

    if isinstance(cfg, VoxtralRealtimeConfig):
        # One text position per `audio_length_per_tok` mel frames; audio and
        # text embeddings are summed position by position.
        num_frames = num_audio_tokens * cfg.audio_length_per_tok
        input_ids = torch.randint(0, vocab, (batch_size, num_audio_tokens), generator=gen)
        batch = {
            "input_ids": input_ids,
            "input_features": torch.randn(batch_size, mel_bins, num_frames, generator=gen),
            "num_delay_tokens": cfg.default_num_delay_tokens,
        }
    elif isinstance(cfg, VoxtralConfig):
        # Whisper-style encoder: fixed-length mel chunks; each chunk becomes
        # max_source_positions / 4 audio placeholder tokens in input_ids.
        num_frames = cfg.audio_config.max_source_positions * 2
        tokens_per_chunk = cfg.audio_config.max_source_positions // 4
        num_text = 4
        audio_ids = torch.full((batch_size, tokens_per_chunk), cfg.audio_token_id)
        text_ids = torch.randint(0, vocab, (batch_size, num_text), generator=gen)
        text_ids[text_ids == cfg.audio_token_id] = 0
        input_ids = torch.cat([audio_ids, text_ids], dim=1)
        batch = {
            "input_ids": input_ids,
            "input_features": torch.randn(batch_size, mel_bins, num_frames, generator=gen),
        }
    else:
        raise TypeError(f"Unsupported config type {type(cfg).__name__}")

    batch["input_features"] = batch["input_features"].to(dtype)
    batch["attention_mask"] = torch.ones_like(input_ids)
    batch["labels"] = input_ids.clone()
    return {k: v.to(model.device) if torch.is_tensor(v) else v for k, v in batch.items()}


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Return (total, trainable) parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Config asks for device 'cuda', but CUDA is not available")
    return torch.device(device)
