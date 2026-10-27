import re
from dataclasses import replace

import numpy as np
import pytest
import torch

from tests.test_finetune import CFG, dataset, examples
from voxnor.config import ModelConfig
from voxnor.finetune import OfflineBatches, accumulate_gradients, train
from voxnor.model import load_model
from voxnor.offline_train import build_offline_batch, target_tokens, text_loss
from voxnor.realtime_stream import IGNORE_INDEX
from voxnor.realtime_train import ENCODER_LORA_TARGETS, add_lora
from voxnor.transcribe import build_transcriber, load_processor

MINI = ModelConfig(
    family="voxtral", hf_id="mistralai/Voxtral-Mini-3B-2507", tiny=True, dtype="float32"
)


@pytest.fixture(scope="module")
def mini_processor():
    return load_processor(MINI)


def clips():
    rng = np.random.default_rng(0)
    return [rng.standard_normal(n).astype(np.float32) * 0.1 for n in (16_000, 40_000)]


def test_offline_batch_puts_the_loss_on_the_transcript_only(mini_processor):
    texts = ["hei på deg", "takk president eee"]
    batch = build_offline_batch(mini_processor, MINI.hf_id, "nb", clips(), texts)
    assert batch["input_ids"].shape == batch["labels"].shape == batch["attention_mask"].shape
    eos = target_tokens(mini_processor, "x")[-1]
    for row, text in enumerate(texts):
        target = target_tokens(mini_processor, text)
        labels = batch["labels"][row]
        labelled = (labels != IGNORE_INDEX).nonzero().flatten()
        # One contiguous span: the transcript and the end token, right after the prompt.
        assert labels[labelled].tolist() == target and target[-1] == eos
        assert labelled.tolist() == list(range(labelled[0], labelled[0] + len(target)))
        assert batch["input_ids"][row, labelled].tolist() == target
        # The prompt ends with the language hint and [TRANSCRIBE], as in evaluation.
        end = int(labelled[-1]) + 1
        assert batch["attention_mask"][row].tolist() == [1] * end + [0] * (len(labels) - end)
    # Every clip is padded to 30 s, so both prompts have the same length.
    starts = [(batch["labels"][r] != IGNORE_INDEX).nonzero()[0].item() for r in range(2)]
    assert starts[0] == starts[1]


def test_encoder_lora_covers_minis_whisper_encoder():
    model = add_lora(load_model(MINI), rank=4, alpha=8, dropout=0.0, encoder=True)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    audio_side = [n for n in trainable if "audio_tower" in n or "projector" in n]
    for part in ("self_attn.out_proj", "fc1", "fc2", "multi_modal_projector.linear_1"):
        assert any(part in n for n in audio_side), part


def test_the_wider_encoder_pattern_leaves_realtime_unchanged():
    # The pattern of the Realtime runs so far.
    before = (
        r"model\.audio_tower\.layers\.\d+\.(self_attn\.[qkvo]_proj|mlp\.(gate|up|down)_proj)"
        r"|model\.multi_modal_projector\.linear_[12]"
    )
    names = [n for n, _ in load_model(ModelConfig(tiny=True, dtype="float32")).named_modules()]
    assert [n for n in names if re.fullmatch(ENCODER_LORA_TARGETS, n)] == [
        n for n in names if re.fullmatch(before, n)
    ]


def test_offline_lora_learns_a_transcript(mini_processor):
    model = add_lora(load_model(MINI), rank=4, alpha=8, dropout=0.0, encoder=True)
    batch = build_offline_batch(mini_processor, MINI.hf_id, "nb", clips()[:1], ["hei på deg"])
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-2)
    losses = []
    for _ in range(15):
        loss, out, labels = accumulate_gradients(model, [batch])
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss)
    # As for Realtime: the tiny model's frozen output layer caps its confidence,
    # so the loss only has to fall clearly.
    assert losses[-1] < losses[0] - 0.3
    assert set(text_loss(out.logits, labels)) == {"text_loss", "text_accuracy"}


def test_offline_training_runs_and_its_model_transcribes(mini_processor, tmp_path):
    model = add_lora(load_model(MINI), rank=4, alpha=8, dropout=0.0, encoder=True)
    batches = OfflineBatches(mini_processor, MINI.hf_id, "nb", dataset())
    evaluated = []
    cfg = replace(CFG, augment=True, lora_dropout=0.0)
    train(model, batches, examples(), cfg, 0, tmp_path, evaluated.append)
    assert evaluated == [0, 2, 4]
    model.eval()
    transcriber = build_transcriber(replace(MINI, language="nb"), model, mini_processor, 4)
    texts = transcriber.transcribe(clips())
    assert len(texts) == 2 and all(isinstance(t, str) for t in texts)
