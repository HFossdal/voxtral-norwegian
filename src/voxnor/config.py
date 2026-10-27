"""Typed run configuration loaded from YAML, with dotted command-line overrides.

A run is fully described by its resolved config, the git commit and the seed.
Unknown keys are errors, so a typo in a YAML file can't silently fall back to
a default.
"""

from __future__ import annotations

import argparse
import dataclasses
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

# "whisper" and "mistral_api" are for reference points only (nb-whisper and
# Mistral's API models), never fine-tuned here.
ModelFamily = Literal["voxtral_realtime", "voxtral", "whisper", "mistral_api"]


@dataclass(frozen=True)
class ModelConfig:
    family: ModelFamily = "voxtral_realtime"
    # For "mistral_api", the API model name (e.g. voxtral-mini-transcribe-3).
    hf_id: str = "mistralai/Voxtral-Mini-4B-Realtime-2602"
    # Randomly initialised miniature of `family`: no download, CPU-friendly.
    # Only for testing the code paths; its outputs are meaningless.
    tiny: bool = False
    dtype: Literal["float32", "bfloat16", "float16"] = "bfloat16"
    # Realtime models only. Must be a multiple of 80 ms; 480 ms is the
    # model card's recommended setting.
    transcription_delay_ms: int = 480
    # Voxtral Mini, Whisper and the API: ISO 639-1 language hint, or None to auto-detect.
    # The Realtime processor always auto-detects.
    language: str | None = None
    # Directory of a saved LoRA adapter (PEFT format), merged into the weights
    # after loading. None: the pretrained model as is.
    adapter: str | None = None


@dataclass(frozen=True)
class DataConfig:
    dataset_id: str = "scribe-project/npsc_nb"
    # Directory with one `<split>.txt` file of utterance ids per split.
    # None means the full split.
    subset_dir: str | None = None
    # True: download each split's Parquet once to the HF cache and memory-map it.
    # False: fetch only the row groups that hold the needed rows over HTTP
    # (laptop subsets), and cache those rows in `cache_dir`.
    download: bool = True
    cache_dir: str = "data/cache"
    # Drop training utterances by speakers who also occur in these splits,
    # so that evaluation on them measures unseen speakers.
    exclude_train_speakers_from: tuple[Literal["validation", "test"], ...] = ()


@dataclass(frozen=True)
class RuntimeConfig:
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size: int = 8
    # Voxtral Mini generation cap. Utterances are < 15 s (~50 words), so a
    # transcript that hits this limit is almost surely a decoding loop.
    max_new_tokens: int = 256


@dataclass(frozen=True)
class TrainConfig:
    """LoRA fine-tuning of Voxtral Realtime or Voxtral Mini."""

    # Word timings from scripts/align_train.py (one JSON line per utterance).
    alignments: str | None = None
    # Extra clips with their word timings, added to the training set: the
    # output directory of scripts/build_ssc.py. None: NPSC only.
    extra_data: str | None = None
    # Checkpoints are chosen on these validation utterances.
    selection_ids: str = "subsets/selection/validation.txt"
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    # Also LoRA on the audio encoder and the projector; decoder only if False.
    lora_encoder: bool = False
    # Speed perturbation and noise on training clips (voxnor.augment).
    augment: bool = False
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    epochs: int = 2
    batch_size: int = 16  # clips per optimizer step
    # Clips per forward pass; gradients are accumulated over the batch. 16 at
    # once runs out of memory on a 40 GB A100: the loss over the 131k-token
    # vocabulary takes several 2 GB float copies of the logits.
    micro_batch_size: int = 8
    max_grad_norm: float = 1.0
    log_every: int = 20
    # Saving the training state lets an interrupted run resume;
    # evaluation on the selection set also saves the adapter.
    save_every: int = 250
    eval_every: int = 500


@dataclass(frozen=True)
class RunConfig:
    name: str = "unnamed"
    seed: int = 0
    output_root: str = "runs"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        if self.model.transcription_delay_ms % 80 != 0:
            raise ValueError(
                f"transcription_delay_ms must be a multiple of 80, "
                f"got {self.model.transcription_delay_ms}"
            )
        if self.model.family == "voxtral_realtime" and self.model.language is not None:
            raise ValueError(
                "The Realtime processor always auto-detects the language; "
                "model.language must be null"
            )
        for key in (
            "runtime.batch_size",
            "train.batch_size",
            "train.micro_batch_size",
            "train.save_every",
            "train.eval_every",
        ):
            section, name = key.split(".")
            if getattr(getattr(self, section), name) < 1:
                raise ValueError(f"{key} must be positive")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class _Yaml12Loader(yaml.SafeLoader):
    """SafeLoader with YAML 1.2 booleans: only true/false.

    YAML 1.1 also reads yes/no/on/off as booleans, which turns the language
    code for Norwegian, `no`, into False.
    """


_Yaml12Loader.yaml_implicit_resolvers = {
    first: [(tag, regexp) for tag, regexp in resolvers if tag != "tag:yaml.org,2002:bool"]
    for first, resolvers in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
_Yaml12Loader.add_implicit_resolver(
    "tag:yaml.org,2002:bool",
    re.compile(r"^(?:true|True|TRUE|false|False|FALSE)$"),
    list("tTfF"),
)


def _load_yaml(text: str) -> Any:
    return yaml.load(text, Loader=_Yaml12Loader)


def load_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    """Load a YAML config and apply `section.key=value` overrides on top."""
    raw = _load_yaml(Path(path).read_text(encoding="utf-8")) or {}
    for override in overrides or []:
        _apply_override(raw, override)
    return _build(RunConfig, raw, prefix="")


def add_override_argument(parser: argparse.ArgumentParser) -> None:
    """`--set key=value ...`; repeated flags accumulate instead of replacing each other."""
    parser.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE")


def save_config(config: RunConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(_plain(config.to_dict()), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def _plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, tuple | list):
        return [_plain(v) for v in value]
    return value


def _apply_override(raw: dict[str, Any], override: str) -> None:
    key, sep, value = override.partition("=")
    if not sep or not key:
        raise ValueError(f"Override must look like 'section.key=value', got {override!r}")
    *parents, leaf = key.split(".")
    node = raw
    for part in parents:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ValueError(f"Cannot override {key!r}: {part!r} is not a section")
    # YAML parsing gives `8` -> int, `true` -> bool, `null` -> None, like in the files.
    node[leaf] = _load_yaml(value)


def _build(cls: type, raw: dict[str, Any], prefix: str) -> Any:
    if not isinstance(raw, dict):
        raise TypeError(f"Expected a mapping for {prefix or 'the config root'}, got {raw!r}")
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(raw) - set(fields)
    if unknown:
        raise KeyError(f"Unknown config key(s) {sorted(prefix + k for k in unknown)}")

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for name, value in raw.items():
        hint = hints[name]
        if dataclasses.is_dataclass(hint):
            kwargs[name] = _build(hint, value, prefix=f"{prefix}{name}.")
        elif typing.get_origin(hint) is tuple:
            # YAML has no tuples; sequences are stored as tuples so configs stay hashable.
            if not isinstance(value, list | tuple):
                raise TypeError(f"{prefix}{name} must be a list, got {value!r}")
            (item_hint, _ellipsis) = typing.get_args(hint)
            for i, item in enumerate(value):
                _check_type(item, item_hint, f"{prefix}{name}[{i}]")
            kwargs[name] = tuple(value)
        else:
            _check_type(value, hint, f"{prefix}{name}")
            kwargs[name] = value
    return cls(**kwargs)


def _check_type(value: Any, hint: Any, key: str) -> None:
    origin = typing.get_origin(hint)
    if origin is Literal:
        if value not in typing.get_args(hint):
            raise ValueError(f"{key} must be one of {typing.get_args(hint)}, got {value!r}")
        return
    if origin in (typing.Union, types.UnionType):
        allowed = tuple(type(None) if a is type(None) else a for a in typing.get_args(hint))
    else:
        allowed = (hint,)
    # bool is a subclass of int; `batch_size: true` should still be rejected.
    if isinstance(value, bool) and bool not in allowed:
        raise TypeError(f"{key} must be {hint}, got {value!r}")
    if not isinstance(value, allowed):
        raise TypeError(f"{key} must be {hint}, got {value!r}")
