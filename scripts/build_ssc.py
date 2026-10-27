"""Cut extra training clips out of the Stortinget Speech Corpus.

    python scripts/build_ssc.py --out <dir> --hours 60
    python -m torch.distributed.run --standalone --nproc_per_node 4 scripts/build_ssc.py \
        --out <dir> --shards 43

Goes through the shards of the SSC WebDataset mirror in a fixed random order
(seeded), keeps segments that pass `voxnor.ssc.skip_reason`, and cuts the
runs where proceedings and ASR agree into clips (`voxnor.ssc`). Stops after
the shard that brings the clips to `--hours`. Per shard it writes:

    <dir>/clips/shard-NNNNN.parquet      clips, same columns as npsc_nb
    <dir>/alignments/shard-NNNNN.jsonl   their word timings (align_train format)
    <dir>/stats/shard-NNNNN.json         segments used and skipped, with reasons

Then <dir>/summary.json for all shards. `train.extra_data=<dir>` adds the
clips to fine-tuning. Re-run the same command after an interruption;
finished shards are not redone.

`--shards N` takes the first N shards of the same order instead, so a larger
build contains every clip of a smaller one. Under torchrun the N shards are
split between the GPUs; `--hours` needs one process, since where it
stops depends on the shards before.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tarfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from align_train import ALIGNER
from huggingface_hub import hf_hub_download

from voxnor import distributed
from voxnor.align import CtcAligner
from voxnor.data import decode_audio, read_id_list
from voxnor.model import resolve_device
from voxnor.ssc import CLIP_SCHEMA, MIN_RUN, segment_clips, skip_reason

SSC = "Aalto-Speech-Synthesis/stortinget_speech_corpus_v1.0"
SSC_REVISION = "1518ba50076a3b37031b0bb14920d7304f205dd9"
N_SHARDS = 145
LISTS = Path(__file__).resolve().parents[1] / "subsets" / "ssc"

log = logging.getLogger("build_ssc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, type=Path)
    size = parser.add_mutually_exclusive_group(required=True)
    size.add_argument("--hours", type=float, help="Stop after the shard that reaches this")
    size.add_argument("--shards", type=int, help="Exactly the first N shards of the order")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-run", type=int, default=MIN_RUN)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    dist = distributed.init()
    if dist.enabled and args.shards is None:
        parser.error("Several processes need --shards")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout, force=True
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    for sub in ("clips", "alignments", "stats"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    held_out = set(read_id_list(LISTS / "held_out_speakers.txt"))
    dates = set(read_id_list(LISTS / "npsc_meeting_dates.txt"))
    order = np.random.default_rng(args.seed).permutation(N_SHARDS).tolist()
    device = distributed.process_device(resolve_device(args.device), dist)
    aligner = None

    def stats_of(shard: int) -> dict:
        nonlocal aligner
        stats_path = args.out / "stats" / f"shard-{shard:05d}.json"
        if not stats_path.exists():
            if aligner is None:
                aligner = CtcAligner.from_pretrained(ALIGNER, device)
            build_shard(shard, args.out, aligner, held_out, dates, args.min_run)
        return json.loads(stats_path.read_text(encoding="utf-8"))

    hours, done = 0.0, []
    if args.shards is not None:
        chosen = order[: args.shards]
        mine = chosen[dist.rank :: dist.world_size]
        for shard in mine:
            stats = stats_of(shard)
            log.info("shard %d: %.1f h of clips", shard, stats["clip_hours"])
        distributed.barrier(dist)  # every process has finished its shards
        if not dist.is_main:
            distributed.close(dist)
            return
        done = [stats_of(shard) for shard in chosen]  # all built now: only read
        hours = sum(s["clip_hours"] for s in done)
    else:
        for shard in order:
            if hours >= args.hours:
                break
            stats = stats_of(shard)
            hours += stats["clip_hours"]
            done.append(stats)
            log.info(
                "shard %d: %.1f h of clips, %.1f h in total", shard, stats["clip_hours"], hours
            )

    skipped = sum((Counter(s["skipped"]) for s in done), Counter())
    summary = {
        "source": f"{SSC}@{SSC_REVISION}",
        "aligner": ALIGNER,
        "seed": args.seed,
        "min_run": args.min_run,
        "shards": [s["shard"] for s in done],
        "segments": sum(s["segments"] for s in done),
        "segment_hours": round(sum(s["segment_hours"] for s in done), 2),
        "used_segments": sum(s["used_segments"] for s in done),
        "skipped": dict(skipped.most_common()),
        "clips": sum(s["clips"] for s in done),
        "clip_hours": round(hours, 2),
        "speakers": len(set().union(*(s["speakers"] for s in done))),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    distributed.close(dist)


def build_shard(
    shard: int,
    out: Path,
    aligner: CtcAligner,
    held_out: set[str],
    dates: set[str],
    min_run: int,
) -> None:
    """Clips, alignments and stats of one shard; the stats file is written last."""
    name = f"shard-{shard:05d}"
    tar_path = hf_hub_download(SSC, f"{name}.tar", repo_type="dataset", revision=SSC_REVISION)
    clips_path = out / "clips" / f"{name}.parquet"
    tmp = clips_path.with_name(clips_path.name + ".tmp")
    rows, records, skipped = [], [], Counter()
    n_clips, clip_seconds = 0, 0.0
    segments, segment_seconds, speakers = 0, 0.0, set()
    start = time.perf_counter()
    # Written in parts: a shard's clips are ~15 h of WAV, too much to hold.
    with pq.ParquetWriter(tmp, CLIP_SCHEMA) as writer:
        for meta, mp3 in segments_of(tar_path):
            segments += 1
            segment_seconds += meta["duration"]
            reason = skip_reason(meta, held_out, dates)
            if reason is None:
                clip_rows, clip_records, reason = segment_clips(
                    meta, decode_audio({"bytes": mp3}), aligner, min_run
                )
                rows += clip_rows
                records += clip_records
            if reason is None:
                speakers.add(meta["speakers"][0]["speaker_id"])
            else:
                skipped[reason] += 1
            if len(rows) >= 500:
                n_clips, clip_seconds = flush(writer, rows, n_clips, clip_seconds)
            if segments % 500 == 0:
                log.info("%s: %d segments, %.0f s", name, segments, time.perf_counter() - start)
        n_clips, clip_seconds = flush(writer, rows, n_clips, clip_seconds)
    tmp.replace(clips_path)

    write_atomic(
        out / "alignments" / f"{name}.jsonl",
        lambda p: p.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
        ),
    )
    stats = {
        "shard": shard,
        "segments": segments,
        "segment_hours": round(segment_seconds / 3600, 3),
        "used_segments": segments - sum(skipped.values()),
        "skipped": dict(skipped),
        "clips": n_clips,
        "clip_hours": round(clip_seconds / 3600, 3),
        "speakers": sorted(speakers),
        "seconds": round(time.perf_counter() - start, 1),
    }
    write_atomic(
        out / "stats" / f"{name}.json",
        lambda p: p.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8"),
    )


def segments_of(tar_path: str):
    """(JSON record, MP3 bytes) per segment of a WebDataset shard."""
    pending: dict[str, dict] = {}
    with tarfile.open(tar_path) as tf:
        for member in tf:
            if not member.isfile():
                continue
            key, _, ext = member.name.rpartition(".")
            pending.setdefault(key, {})[ext] = tf.extractfile(member).read()
            if {"json", "mp3"} <= pending[key].keys():
                parts = pending.pop(key)
                yield json.loads(parts["json"]), parts["mp3"]


def flush(writer: pq.ParquetWriter, rows: list[dict], n: int, seconds: float):
    """Write `rows` as one row group and empty the list; return the running totals."""
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=CLIP_SCHEMA))
        n, seconds = n + len(rows), seconds + sum(r["duration"] for r in rows)
        rows.clear()
    return n, seconds


def write_atomic(path: Path, write) -> None:
    tmp = path.with_name(path.name + ".tmp")
    write(tmp)
    tmp.replace(path)


if __name__ == "__main__":
    main()
