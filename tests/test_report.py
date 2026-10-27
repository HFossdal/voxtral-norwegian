import json

import pandas as pd
import pytest

from voxnor.analysis import score_utterances, summarize
from voxnor.config import ModelConfig, RunConfig, save_config
from voxnor.report import describe, load_run, render


def write_run(directory, model, hypotheses):
    directory.mkdir()
    save_config(RunConfig(model=model), directory / "config.yaml")
    refs = ["a b c d", "e f g h", "i j k l"]
    predictions = pd.DataFrame(
        {
            "utterance_id": ["u0", "u1", "u2"],
            "hypothesis": hypotheses,
            "standardized_text": refs,
            "raw_text": refs,
            "speaker_id": "s",
            "region": "east",
            "gender": "male",
            "duration": 2.0,
            "seen_speaker": False,
        }
    )
    scored = score_utterances(predictions)
    scored.to_csv(directory / "scored_validation.csv", index=False)
    summary = summarize(scored, n_resamples=50)
    (directory / "metrics_validation.json").write_text(json.dumps(summary), encoding="utf-8")


def test_describe_marks_tiny_models_and_settings():
    real = RunConfig(model=ModelConfig()).to_dict()
    assert describe(real) == "Voxtral-Mini-4B-Realtime-2602 (480 ms)"
    tiny = RunConfig(model=ModelConfig(tiny=True)).to_dict()
    assert describe(tiny).startswith("TINY RANDOM")
    mini = RunConfig(model=ModelConfig(family="voxtral", language="no")).to_dict()
    assert describe(mini).endswith("(language: no)")
    adapter = "/runs/20261020-095301_finetune-realtime/adapters/step-002000"
    tuned = RunConfig(model=ModelConfig(adapter=adapter)).to_dict()
    assert describe(tuned) == (
        "Voxtral-Mini-4B-Realtime-2602 + LoRA finetune-realtime step-002000 (480 ms)"
    )


def test_render_includes_headline_slices_and_paired_comparison(tmp_path):
    write_run(tmp_path / "a", ModelConfig(), ["a x c d", "e f x h", "i j k x"])
    write_run(tmp_path / "b", ModelConfig(family="voxtral"), ["a b c d", "e f g h", "i j k x"])
    runs = [load_run(tmp_path / name, "validation") for name in ("a", "b")]
    text = render(runs, "validation")
    assert "25.00 [" in text  # baseline WER 3/12
    assert "| region = east | 3 |" in text
    assert "-16.67" in text  # 1/12 - 3/12


def test_paired_comparison_refuses_different_utterances(tmp_path):
    write_run(tmp_path / "a", ModelConfig(), ["a", "e", "i"])
    write_run(tmp_path / "b", ModelConfig(), ["a", "e", "i"])
    run_b = load_run(tmp_path / "b", "validation")
    run_b.scored.loc[0, "utterance_id"] = "other"
    with pytest.raises(ValueError):
        render([load_run(tmp_path / "a", "validation"), run_b], "validation")
