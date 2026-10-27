import json
from dataclasses import replace

import datasets
import numpy as np
import pandas as pd
import pytest
import torch
from peft import get_peft_model_state_dict

from tests.test_data import wav_bytes
from tests.test_realtime_train import REALTIME_ID, processor  # noqa: F401 (fixture)
from voxnor.align import WordTiming
from voxnor.config import ModelConfig, TrainConfig
from voxnor.finetune import (
    BEST_FILE,
    SELECTION_FILE,
    RealtimeBatches,
    SelectionEval,
    accumulate_gradients,
    adapter_dir,
    choose_best,
    make_batch,
    read_selection,
    train,
)
from voxnor.model import load_model
from voxnor.realtime_stream import layout_from_processor
from voxnor.realtime_train import add_lora
from voxnor.train_data import Example

TINY = ModelConfig(family="voxtral_realtime", hf_id=REALTIME_ID, tiny=True, dtype="float32")
CFG = TrainConfig(
    batch_size=2,
    micro_batch_size=1,
    epochs=2,
    learning_rate=1e-2,
    warmup_steps=0,
    log_every=1,
    save_every=1,
    eval_every=2,
    lora_rank=4,
    lora_alpha=8,
    lora_dropout=0.1,
)
SECONDS = [1.0, 1.5, 1.0, 1.25]


def dataset():
    return datasets.Dataset.from_list(
        [
            {
                "utterance_id": f"u{i}",
                "utterance_audio_file": {"bytes": wav_bytes(s), "path": f"{i}.wav"},
            }
            for i, s in enumerate(SECONDS)
        ]
    )


def examples():
    words = [WordTiming("hei", 0.1, 0.4), WordTiming("på", 0.5, 0.7)]
    return [Example(i, f"u{i}", s, words) for i, s in enumerate(SECONDS)]


def tiny_lora(encoder=False):
    return add_lora(
        load_model(TINY), CFG.lora_rank, CFG.lora_alpha, CFG.lora_dropout, encoder=encoder
    )


class Interrupt(Exception):
    pass


def run(processor, run_dir, on_eval):  # noqa: F811
    run_dir.mkdir(exist_ok=True)
    model = tiny_lora()
    layout = layout_from_processor(processor)
    # With augmentation, so resuming must also reproduce the augmented audio.
    cfg = replace(CFG, augment=True)
    train(
        model, RealtimeBatches(processor, layout, dataset()), examples(), cfg, 0, run_dir, on_eval
    )
    return model


def lora_weights(model):
    return {k: v.clone() for k, v in get_peft_model_state_dict(model).items()}


def test_resumed_training_ends_where_an_uninterrupted_run_does(processor, tmp_path):  # noqa: F811
    evaluated = []
    full = lora_weights(run(processor, tmp_path / "a", evaluated.append))
    assert evaluated == [0, 2, 4]  # 2 batches per epoch, 2 epochs

    def interrupt_at_2(step):
        if step == 2:
            raise Interrupt

    with pytest.raises(Interrupt):
        run(processor, tmp_path / "b", interrupt_at_2)
    resumed_evals = []
    resumed = lora_weights(run(processor, tmp_path / "b", resumed_evals.append))
    assert resumed_evals == [2, 4]
    assert full.keys() == resumed.keys()
    assert all(torch.equal(full[k], resumed[k]) for k in full)

    log = [json.loads(line) for line in (tmp_path / "a" / "train_log.jsonl").open()]
    assert [e["step"] for e in log] == [1, 2, 3, 4]
    assert {"loss", "pad_loss", "text_loss", "lr", "grad_norm"} <= set(log[0])


class FixedTranscriber:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audios):
        return [self.text] * len(audios)


def selection_metadata():
    return pd.DataFrame(
        {
            "utterance_id": ["u0", "u1"],
            "speaker_id": ["s1", "s2"],
            "gender": ["female", "male"],
            "region": ["east", "west"],
            "duration": [1.0, 1.5],
            "raw_text": ["hei på deg", "hei på deg"],
            "standardized_text": ["hei på deg", "hei på deg"],
        }
    )


@pytest.mark.parametrize("encoder", [False, True])
def test_selection_eval_saves_the_adapter_and_scores_once(tmp_path, encoder):
    model = tiny_lora(encoder)
    with torch.no_grad():  # LoRA B starts at zero; make the adapter change something
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()
    ds = dataset().select([0, 1]).add_column("duration", [1.0, 1.5])
    evaluate = SelectionEval(
        model, FixedTranscriber("hei på"), ds, selection_metadata(), set(), tmp_path, 2
    )
    evaluate(0)
    evaluate(2)
    evaluate(2)  # already done: skipped
    results = read_selection(tmp_path)
    assert list(results) == [0, 2]
    assert results[2]["wer"] == pytest.approx(1 / 3)
    assert not adapter_dir(tmp_path, 0).exists()
    assert (adapter_dir(tmp_path, 2) / "adapter_config.json").exists()

    # A saved adapter loads back merged, with the same outputs.
    batch = {
        "input_ids": torch.tensor([[1, 32, 32, 33]]),
        "input_features": torch.randn(1, 128, 32),
        "num_delay_tokens": 6,
    }
    model.eval()
    merged = load_model(replace(TINY, adapter=str(adapter_dir(tmp_path, 2)))).eval()
    base = load_model(TINY).eval()
    with torch.no_grad():
        adapted = model(**batch).logits
        assert torch.allclose(adapted, merged(**batch).logits, atol=1e-4)
        assert not torch.allclose(adapted, base(**batch).logits, atol=1e-2)


def test_choose_best_ignores_the_base_model(tmp_path):
    lines = [{"step": 0, "wer": 0.1}, {"step": 500, "wer": 0.3}, {"step": 1000, "wer": 0.2}]
    (tmp_path / SELECTION_FILE).write_text("".join(json.dumps(r) + "\n" for r in lines))
    best = choose_best(tmp_path)
    assert (best["step"], best["selection_wer"], best["base_model_selection_wer"]) == (
        1000,
        0.2,
        0.1,
    )
    assert json.loads((tmp_path / BEST_FILE).read_text())["step"] == 1000


def test_micro_batches_give_the_gradient_of_the_whole_batch(processor):  # noqa: F811
    layout = layout_from_processor(processor)
    ds, chosen = dataset(), examples()[:2]
    model = tiny_lora().eval()  # no dropout, so both runs see the same network
    with torch.no_grad():  # nonzero B, so A gets a gradient too
        for name, param in model.named_parameters():
            if "lora_B" in name:
                param.normal_()

    def gradients(micro_batches):
        model.zero_grad()
        loss, _, _ = accumulate_gradients(model, micro_batches)
        grads = {n: p.grad.clone() for n, p in model.named_parameters() if p.requires_grad}
        return loss, grads

    whole_loss, whole = gradients([make_batch(processor, layout, ds, chosen)])
    parts = [make_batch(processor, layout, ds, [e]) for e in chosen]
    split_loss, split = gradients(parts)
    assert split_loss == pytest.approx(whole_loss, rel=1e-4)
    assert all(torch.allclose(whole[n], split[n], rtol=1e-3, atol=1e-6) for n in whole)


def test_make_batch_augments_reproducibly(processor):  # noqa: F811
    layout = layout_from_processor(processor)
    ds, chosen = dataset(), examples()[:2]
    plain = make_batch(processor, layout, ds, chosen)
    seeded = [make_batch(processor, layout, ds, chosen, np.random.default_rng(s)) for s in (3, 3)]
    assert torch.equal(seeded[0]["input_features"], seeded[1]["input_features"])
    assert torch.equal(seeded[0]["labels"], seeded[1]["labels"])
    others = [make_batch(processor, layout, ds, chosen, np.random.default_rng(s)) for s in range(6)]
    assert any(
        b["input_features"].shape != plain["input_features"].shape
        or not torch.equal(b["input_features"], plain["input_features"])
        for b in others
    )
