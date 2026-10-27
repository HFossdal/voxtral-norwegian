import json
from datetime import datetime

import pytest

from voxnor.config import RunConfig, load_config


def test_run_dir_records_config_and_environment(tmp_path):
    from voxnor.run_dir import create_run_dir

    cfg = RunConfig(name="t", output_root=str(tmp_path))
    path = create_run_dir(cfg, now=datetime(2026, 1, 2, 3, 4, 5))

    assert path.name == "20260102-030405_t"
    assert load_config(path / "config.yaml") == cfg
    env = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    assert env["packages"]["transformers"] is not None
    assert "git_commit" in env


def test_run_dir_never_overwrites(tmp_path):
    from voxnor.run_dir import create_run_dir

    cfg = RunConfig(name="t", output_root=str(tmp_path))
    now = datetime(2026, 1, 2, 3, 4, 5)
    create_run_dir(cfg, now=now)
    with pytest.raises(FileExistsError):
        create_run_dir(cfg, now=now)
