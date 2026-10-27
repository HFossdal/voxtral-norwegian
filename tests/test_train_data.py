import json

import datasets
import numpy as np
import pandas as pd
import pytest

from tests.test_align import aligner  # noqa: F401 (fixture)
from tests.test_data import wav_bytes
from tests.test_realtime_train import processor  # noqa: F401 (fixture)
from voxnor.realtime_stream import layout_from_processor
from voxnor.train_data import (
    DROP_INAUDIBLE,
    DROP_NO_FIT,
    DROP_NOT_ALIGNED,
    DROP_NOT_IN_FILE,
    align_to_file,
    alignment_summary,
    length_grouped_batches,
    read_alignments,
    select_examples,
)


def test_batches_cover_every_index_once():
    durations = np.random.default_rng(0).uniform(1, 15, size=103)
    batches = length_grouped_batches(durations, batch_size=8, seed=0)
    assert sorted(i for b in batches for i in b) == list(range(103))
    assert all(len(b) <= 8 for b in batches)


def test_batches_hold_neighbours_in_length_within_a_pool():
    durations = np.random.default_rng(1).uniform(1, 15, size=64)
    batches = length_grouped_batches(durations, batch_size=8, seed=0, pool_batches=8)
    ranked = np.argsort(durations).tolist()
    chunks = [sorted(ranked[i : i + 8]) for i in range(0, 64, 8)]
    assert sorted(sorted(b) for b in batches) == sorted(chunks)


def test_batches_depend_on_the_seed_only():
    durations = list(range(50))
    first = length_grouped_batches(durations, 4, seed=3, pool_batches=2)
    assert first == length_grouped_batches(durations, 4, seed=3, pool_batches=2)
    assert first != length_grouped_batches(durations, 4, seed=4, pool_batches=2)


def clip_dataset(texts):
    return datasets.Dataset.from_list(
        [
            {
                "utterance_id": f"u{i}",
                "standardized_text": text,
                "utterance_audio_file": {"bytes": wav_bytes(0.5), "path": f"{i}.wav"},
            }
            for i, text in enumerate(texts)
        ]
    )


def test_align_to_file_resumes_and_repairs_a_cut_off_line(aligner, tmp_path):  # noqa: F811
    path = tmp_path / "train.jsonl"
    align_to_file(clip_dataset(["hei på deg", ""]), aligner, path)
    with path.open("a", encoding="utf-8") as f:
        f.write('{"utterance_id": "u2", "n_sa')  # killed mid-write
    align_to_file(clip_dataset(["hei på deg", "", "takk"]), aligner, path)

    records = read_alignments(path)
    assert list(records) == ["u0", "u1", "u2"]
    assert [w[0] for w in records["u0"]["words"]] == ["hei", "på", "deg"]
    assert records["u0"]["n_samples"] == 8000
    assert records["u1"] == {"utterance_id": "u1", "n_samples": 8000, "words": [], "log_prob": None}
    assert records["u2"]["log_prob"] < 0


def test_alignment_summary_counts_failures_and_lists_the_worst():
    alignments = {
        "a": {"utterance_id": "a", "words": [["x", 0, 1]], "log_prob": -0.5},
        "b": {"utterance_id": "b", "words": None, "log_prob": None},
        "c": {"utterance_id": "c", "words": [["y", 0, 1]], "log_prob": -2.0},
    }
    meta = pd.DataFrame({"utterance_id": ["a", "b", "c", "d"], "standardized_text": list("xzyw")})
    summary = alignment_summary(alignments, meta, n_worst=1)
    assert (summary["utterances"], summary["in_file"], summary["failed"]) == (4, 3, 1)
    assert summary["worst"] == [{"utterance_id": "c", "log_prob": -2.0, "text": "y"}]
    json.dumps(summary)


def test_select_examples_keeps_fitting_clips_and_counts_the_rest(processor):  # noqa: F811
    layout = layout_from_processor(processor)
    one_second = 16000
    meta = pd.DataFrame(
        {
            "utterance_id": ["ok", "inaudible", "missing", "failed", "crowded"],
            "raw_text": ["hei", "hei <INAUDIBLE>", "hei", "hei", "x"],
        }
    )
    many = [[f"ord{i}", 0.9, 0.95] for i in range(40)]  # 40 words at the very end
    alignments = {
        "ok": {"words": [["hei", 0.1, 0.4]], "n_samples": one_second},
        "inaudible": {"words": [["hei", 0.1, 0.4]], "n_samples": one_second},
        "failed": {"words": None, "n_samples": one_second},
        "crowded": {"words": many, "n_samples": one_second},
    }
    examples, dropped = select_examples(meta, alignments, processor, layout)
    assert [(e.row, e.utterance_id, e.seconds) for e in examples] == [(0, "ok", 1.0)]
    assert examples[0].timings[0].word == "hei"
    assert dropped == {
        DROP_INAUDIBLE: 1,
        DROP_NOT_IN_FILE: 1,
        DROP_NOT_ALIGNED: 1,
        DROP_NO_FIT: 1,
    }


@pytest.mark.parametrize("size", [0, 1])
def test_batches_of_tiny_inputs(size):
    assert [i for b in length_grouped_batches([1.0] * size, 4, seed=0) for i in b] == list(
        range(size)
    )
