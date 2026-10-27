"""Measure how much SSC audio each clip-cutting setting keeps, before a build.

    python scripts/tune_ssc_cuts.py --segments 50

Aligns the first `--segments` candidate segments of one SSC shard once, then
re-cuts them with every combination of `min_edge_gap` and `min_words`
(`voxnor.ssc.plan_clips`) and prints, per setting, the share of the
segment audio kept as clips, the clip count and mean length, and the share
of agreeing words kept. Also prints the pauses between agreeing words.
Needs a GPU to be quick (~1 s per segment); on a laptop CPU it is ~40 s.
"""

from __future__ import annotations

import argparse

import numpy as np
from align_train import ALIGNER
from build_ssc import LISTS, SSC, SSC_REVISION, segments_of
from huggingface_hub import hf_hub_download

from voxnor.align import CtcAligner
from voxnor.data import SAMPLE_RATE, decode_audio, read_id_list
from voxnor.model import resolve_device
from voxnor.ssc import MIN_RUN, agreeing_runs, plan_clips, skip_reason

EDGE_GAPS = (0.0, 0.02, 0.04, 0.06, 0.1)
MIN_WORDS = (4, 6)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", type=int, default=50)
    parser.add_argument("--segments", type=int, default=50)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    held_out = set(read_id_list(LISTS / "held_out_speakers.txt"))
    dates = set(read_id_list(LISTS / "npsc_meeting_dates.txt"))
    aligner = CtcAligner.from_pretrained(ALIGNER, resolve_device(args.device))
    path = hf_hub_download(
        SSC, f"shard-{args.shard:05d}.tar", repo_type="dataset", revision=SSC_REVISION
    )

    aligned = []  # (seconds, timings, runs)
    for meta, mp3 in segments_of(path):
        if len(aligned) == args.segments:
            break
        if skip_reason(meta, held_out, dates):
            continue
        words = meta["transcription_text"].split()
        runs = agreeing_runs(meta["proceedings_text"], words, MIN_RUN)
        audio = decode_audio({"bytes": mp3})
        result = aligner.align_with_score(audio, words)
        if result is not None:
            aligned.append((len(audio) / SAMPLE_RATE, result[0], runs))

    total = sum(seconds for seconds, _, _ in aligned)
    run_words = sum(last - first + 1 for _, _, runs in aligned for first, last in runs)
    gaps = np.array(
        [t[k + 1].start - t[k].end for _, t, runs in aligned for a, b in runs for k in range(a, b)]
    )
    print(f"{len(aligned)} candidate segments, {total / 3600:.2f} h, {run_words} agreeing words")
    quantiles = {q: round(float(np.quantile(gaps, q)), 3) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    print(f"pauses between agreeing words (s): {quantiles}")
    print("min_edge_gap  min_words  audio kept  clips  mean s  words kept")
    for gap in EDGE_GAPS:
        for min_words in MIN_WORDS:
            clips = [
                c
                for seconds, timings, runs in aligned
                for c in plan_clips(timings, runs, seconds, min_words=min_words, min_edge_gap=gap)
            ]
            kept = sum(c.end - c.start for c in clips)
            words = sum(c.last - c.first + 1 for c in clips)
            print(
                f"{gap:12.2f}  {min_words:9d}  {kept / total:10.1%}  {len(clips):5d}"
                f"  {kept / max(len(clips), 1):6.1f}  {words / max(run_words, 1):10.1%}"
            )


if __name__ == "__main__":
    main()
