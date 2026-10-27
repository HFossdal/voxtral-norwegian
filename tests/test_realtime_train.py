import numpy as np
import pytest
import torch

from voxnor.align import WordTiming
from voxnor.config import ModelConfig
from voxnor.model import load_model
from voxnor.realtime_stream import IGNORE_INDEX, layout_from_processor
from voxnor.realtime_train import add_lora, build_batch, loss_breakdown, n_positions

REALTIME_ID = "mistralai/Voxtral-Mini-4B-Realtime-2602"


@pytest.fixture(scope="module")
def processor():
    # The real processor (tokenizer + feature extractor), from the local
    # Hugging Face cache only: tests never use the network.
    from transformers import AutoProcessor

    try:
        return AutoProcessor.from_pretrained(REALTIME_ID, local_files_only=True)
    except OSError:
        pytest.skip("Realtime processor not in the local Hugging Face cache")


def clips():
    rng = np.random.default_rng(0)
    audios = [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16000, 24000)]
    timings = [
        [WordTiming("hei", 0.1, 0.4), WordTiming("på", 0.45, 0.6)],
        [WordTiming("takk", 0.2, 0.5), WordTiming("president", 0.6, 1.2)],
    ]
    return audios, timings


def test_n_positions_matches_the_processor(processor):
    audio = np.zeros(32000, dtype=np.float32)
    frames = processor([audio], return_tensors="pt")["input_features"].shape[-1]
    assert n_positions(processor, len(audio)) == frames // processor.audio_length_per_tok == 74


def test_build_batch_shapes_and_loss_mask(processor):
    layout = layout_from_processor(processor)
    audios, timings = clips()
    batch, kept = build_batch(processor, layout, audios, timings, space_after_marker=True)
    assert kept == [0, 1]
    ids, labels = batch["input_ids"], batch["labels"]
    assert ids.shape == labels.shape
    assert ids.shape[1] * 8 == batch["input_features"].shape[-1]
    assert (labels[:, : layout.prompt_length] == IGNORE_INDEX).all()
    # The shorter clip gets no loss past its own end.
    assert (labels[0, n_positions(processor, 16000) :] == IGNORE_INDEX).all()
    # Words in both clips are far enough apart to each get a marker.
    assert (ids == layout.word_id).sum(dim=1).tolist() == [2, 2]


def test_lora_overfits_a_tiny_realtime_model(processor):
    layout = layout_from_processor(processor)
    audios, timings = clips()
    batch, _ = build_batch(processor, layout, audios, timings, space_after_marker=True)
    cfg = ModelConfig(family="voxtral_realtime", hf_id=REALTIME_ID, tiny=True, dtype="float32")
    model = add_lora(load_model(cfg), rank=4, alpha=8, dropout=0.0)
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    assert trainable and all("language_model" in n and "lora_" in n for n in trainable)

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=3e-2)
    losses = []
    for _ in range(25):
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    # A tiny random model can't get far: its frozen output layer has small
    # random weights after a fixed norm, which caps how confident it can be.
    # The loss still has to fall clearly.
    assert losses[-1] < losses[0] - 0.3
    parts = loss_breakdown(model(**batch).logits, batch["labels"], layout.pad_id)
    assert set(parts) == {"pad_loss", "text_loss", "text_accuracy"}


def test_encoder_lora_trains_the_audio_side(processor):
    layout = layout_from_processor(processor)
    audios, timings = clips()
    batch, _ = build_batch(processor, layout, audios, timings, space_after_marker=True)
    cfg = ModelConfig(family="voxtral_realtime", hf_id=REALTIME_ID, tiny=True, dtype="float32")
    model = add_lora(load_model(cfg), rank=4, alpha=8, dropout=0.0, encoder=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert all("lora_" in n for n in trainable)
    for part in ("language_model", "audio_tower", "multi_modal_projector"):
        assert any(part in n for n in trainable), part

    model(**batch).loss.backward()
    # lora_B starts at zero, so it is the factor that gets a gradient first.
    encoder_grads = [
        p.grad for n, p in model.named_parameters() if "audio_tower" in n and "lora_B" in n
    ]
    assert encoder_grads and all(g is not None for g in encoder_grads)
    assert any(g.abs().sum() > 0 for g in encoder_grads)
