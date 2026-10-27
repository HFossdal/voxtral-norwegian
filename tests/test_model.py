import pytest
import torch

from voxnor.config import ModelConfig
from voxnor.model import count_parameters, dummy_batch, load_model

FAMILIES = ["voxtral_realtime", "voxtral"]


@pytest.fixture(params=FAMILIES)
def tiny_model(request):
    return load_model(ModelConfig(family=request.param, tiny=True, dtype="float32"))


def test_tiny_model_is_small(tiny_model):
    # Mostly the (real-size) embedding table and LM head.
    total, _ = count_parameters(tiny_model)
    assert total < 20_000_000


def test_tiny_model_is_deterministic():
    cfg = ModelConfig(tiny=True, dtype="float32")
    a, b = load_model(cfg), load_model(cfg)
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters(), strict=True):
        assert torch.equal(pa, pb), name


def test_forward_gives_logits_per_position_and_finite_loss(tiny_model):
    batch = dummy_batch(tiny_model)
    out = tiny_model(**batch)
    batch_size, seq_len = batch["input_ids"].shape
    assert out.logits.shape == (batch_size, seq_len, tiny_model.config.text_config.vocab_size)
    assert torch.isfinite(out.loss)


def test_backward_reaches_audio_encoder_and_decoder(tiny_model):
    tiny_model(**dummy_batch(tiny_model)).loss.backward()
    grads = {n: p.grad for n, p in tiny_model.named_parameters()}
    for part in ("audio_tower", "language_model", "multi_modal_projector"):
        assert any(g is not None and g.abs().sum() > 0 for n, g in grads.items() if part in n), part


def test_audio_changes_the_output(tiny_model):
    # Guards against a batch layout where the audio is silently ignored.
    batch = dummy_batch(tiny_model)
    with torch.no_grad():
        a = tiny_model(**batch).logits
        batch["input_features"] = torch.zeros_like(batch["input_features"])
        b = tiny_model(**batch).logits
    assert not torch.allclose(a, b)
