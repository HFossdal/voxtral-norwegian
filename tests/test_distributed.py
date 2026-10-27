import json
import os
import socket
from dataclasses import replace

import pytest
import torch
import torch.multiprocessing as mp

from tests.test_finetune import (
    CFG,
    TINY,
    FixedTranscriber,
    dataset,
    examples,
    lora_weights,
    selection_metadata,
)
from tests.test_realtime_train import REALTIME_ID, processor  # noqa: F401 (fixture)
from voxnor import distributed
from voxnor.distributed import Dist, shard
from voxnor.finetune import RealtimeBatches, SelectionEval, read_selection, train
from voxnor.model import load_model
from voxnor.realtime_stream import layout_from_processor
from voxnor.realtime_train import add_lora

# No dropout, so one process and two processes train the same network.
DIST_CFG = replace(CFG, augment=True, lora_dropout=0.0)


@pytest.mark.parametrize("world_size", [1, 2, 3, 4, 8])
@pytest.mark.parametrize("n", [0, 1, 2, 5, 16])
def test_shards_cover_every_item_once_and_the_main_process_gets_one(world_size, n):
    parts = [shard(n, Dist(rank, world_size)) for rank in range(world_size)]
    assert [i for first, stop in parts for i in range(first, stop)] == list(range(n))
    if n:
        assert parts[0][1] > parts[0][0]


def test_single_process_helpers_do_nothing():
    single = distributed.SINGLE
    assert not single.enabled and single.is_main
    assert distributed.broadcast_object("x", single) == "x"
    assert distributed.sum_scalar(2.5, single, torch.device("cpu")) == 2.5
    device = torch.device("cuda")
    assert distributed.process_device(device, single) == device
    assert distributed.process_device(device, Dist(1, 2, 1)) == torch.device("cuda", 1)


def tiny_run(rank, run_dir, proc, dist):
    model = load_model(TINY)
    if rank:
        torch.manual_seed(rank)  # a different LoRA start: training must broadcast rank 0's
    model = add_lora(model, DIST_CFG.lora_rank, DIST_CFG.lora_alpha, 0.0)
    ds = dataset().select([0, 1]).add_column("duration", [1.0, 1.5])
    on_eval = SelectionEval(
        model, FixedTranscriber("hei på"), ds, selection_metadata(), set(), run_dir, 1, dist
    )
    layout = layout_from_processor(proc)
    batches = RealtimeBatches(proc, layout, dataset())
    train(model, batches, examples(), DIST_CFG, 0, run_dir, on_eval, dist)
    return model


def worker(rank, world_size, port, run_dir, out):
    from transformers import AutoProcessor

    os.environ |= {
        "RANK": str(rank),
        "LOCAL_RANK": str(rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": str(port),
    }
    dist = distributed.init()
    proc = AutoProcessor.from_pretrained(REALTIME_ID, local_files_only=True)
    model = tiny_run(rank, run_dir, proc, dist)
    if dist.is_main:
        torch.save(lora_weights(model), out)
    distributed.close(dist)


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_two_processes_train_like_one(processor, tmp_path):  # noqa: F811
    single_dir, multi_dir = tmp_path / "single", tmp_path / "multi"
    single_dir.mkdir()
    multi_dir.mkdir()
    single = lora_weights(tiny_run(0, single_dir, processor, distributed.SINGLE))

    out = tmp_path / "multi.pt"
    mp.spawn(worker, args=(2, free_port(), multi_dir, out), nprocs=2, join=True)
    multi = torch.load(out, weights_only=True)
    assert multi.keys() == single.keys()
    assert all(torch.allclose(single[k], multi[k], rtol=1e-4, atol=1e-6) for k in single)

    # The main process writes the files, once.
    log = [json.loads(line) for line in (multi_dir / "train_log.jsonl").open()]
    assert [e["step"] for e in log] == [1, 2, 3, 4]
    assert list(read_selection(multi_dir)) == [0, 2, 4]
    assert read_selection(multi_dir)[4]["wer"] == read_selection(single_dir)[4]["wer"]
    # Selection predictions: one part per process, joined in order.
    joined = (multi_dir / "selection" / "predictions_step-000004.jsonl").read_text()
    assert [json.loads(line)["utterance_id"] for line in joined.splitlines()] == ["u0", "u1"]
