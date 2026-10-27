from pathlib import Path

import pytest

from voxnor.config import RunConfig, load_config, save_config

CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


@pytest.mark.parametrize("name", ["local", "cloud"])
def test_shipped_profiles_load(name):
    cfg = load_config(CONFIG_DIR / f"{name}.yaml")
    assert cfg.name == name


def test_profiles_differ_only_where_intended():
    local = load_config(CONFIG_DIR / "local.yaml").to_dict()
    cloud = load_config(CONFIG_DIR / "cloud.yaml").to_dict()
    assert local["model"]["family"] == cloud["model"]["family"]
    assert local["model"]["hf_id"] == cloud["model"]["hf_id"]
    assert local["model"]["tiny"] and not cloud["model"]["tiny"]
    assert local["data"]["subset_dir"] is not None and not local["data"]["download"]
    assert cloud["data"]["subset_dir"] is None and cloud["data"]["download"]
    # Which speakers are excluded from training changes what the eval measures.
    exclusion = "exclude_train_speakers_from"
    assert local["data"][exclusion] == cloud["data"][exclusion]


def write(tmp_path, text):
    path = tmp_path / "cfg.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_missing_keys_take_defaults(tmp_path):
    cfg = load_config(write(tmp_path, "name: x\n"))
    assert cfg.model == RunConfig().model


def test_empty_file_gives_defaults(tmp_path):
    assert load_config(write(tmp_path, "")) == RunConfig()


def test_overrides_are_parsed_as_yaml(tmp_path):
    cfg = load_config(
        write(tmp_path, "name: x\n"),
        [
            "runtime.batch_size=16",
            "model.tiny=true",
            "data.subset_dir=null",
            "data.exclude_train_speakers_from=[test]",
            "model.family=voxtral",
        ],
    )
    assert cfg.runtime.batch_size == 16
    assert cfg.model.tiny is True
    assert cfg.data.subset_dir is None
    assert cfg.data.exclude_train_speakers_from == ("test",)
    assert cfg.model.family == "voxtral"


@pytest.mark.parametrize(
    ("text", "overrides", "error"),
    [
        ("modle:\n  tiny: true\n", [], KeyError),
        ("model:\n  tinny: true\n", [], KeyError),
        ("", ["runtime.batchsize=4"], KeyError),
        ("model:\n  family: wav2vec2\n", [], ValueError),
        ("runtime:\n  batch_size: '8'\n", [], TypeError),
        ("runtime:\n  batch_size: true\n", [], TypeError),
        ("runtime:\n  batch_size: 0\n", [], ValueError),
        ("model:\n  transcription_delay_ms: 500\n", [], ValueError),
        ("model: true\n", [], TypeError),
        ("", ["no_equals_sign"], ValueError),
        ("data:\n  exclude_train_speakers_from: test\n", [], TypeError),
        ("data:\n  exclude_train_speakers_from: [train]\n", [], ValueError),
        ("seed: 0\n", ["seed.inner=1"], ValueError),
    ],
)
def test_invalid_configs_are_rejected(tmp_path, text, overrides, error):
    with pytest.raises(error):
        load_config(write(tmp_path, text), overrides)


@pytest.mark.parametrize("code", ["no", "nb", "on", "yes"])
def test_language_codes_stay_strings(tmp_path, code):
    # YAML 1.1 would read `no` (Norwegian) as False.
    base = "model:\n  family: voxtral\n"
    assert load_config(write(tmp_path, base), [f"model.language={code}"]).model.language == code
    from_file = load_config(write(tmp_path, base + f"  language: {code}\n"))
    assert from_file.model.language == code


def test_realtime_model_rejects_a_language_hint(tmp_path):
    with pytest.raises(ValueError):
        load_config(write(tmp_path, "model:\n  language: nb\n"))


def test_save_and_reload_round_trips(tmp_path):
    cfg = load_config(CONFIG_DIR / "local.yaml", ["runtime.batch_size=3"])
    save_config(cfg, tmp_path / "resolved.yaml")
    assert load_config(tmp_path / "resolved.yaml") == cfg


def test_repeated_set_flags_accumulate():
    import argparse

    from voxnor.config import add_override_argument

    parser = argparse.ArgumentParser()
    add_override_argument(parser)
    args = parser.parse_args(["--set", "output_root=/x", "--set", "name=a", "seed=1"])
    assert args.set == ["output_root=/x", "name=a", "seed=1"]
    assert parser.parse_args([]).set == []
