"""Write the speakers and meeting dates that SSC clips must not come from.

    python scripts/make_ssc_lists.py --config configs/cloud.yaml

- subsets/ssc/held_out_speakers.txt: ParlaMint IDs of every speaker in
  NPSC validation and test, so that adding SSC to training keeps test
  speakers out and the unseen-speaker validation slice unseen.
- subsets/ssc/npsc_meeting_dates.txt: the dates of all NPSC meetings, so
  that no NPSC audio enters training a second time (validation and test
  meetings above all).

`npsc_nb` has only speaker numbers. Names come from the upstream NPSC
metadata (`npsc_<sentence_id>` = NPSC's `sentence_id`, checked for every
utterance), and names are matched to ParlaMint-NO's person list. Fails if a
held-out speaker cannot be matched. Both files are committed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

from voxnor.config import load_config
from voxnor.data import SPLITS, load_metadata, write_id_list
from voxnor.ssc import match_speaker, read_parlamint_people

NPSC = "NbAiLab/NPSC"
NPSC_REVISION = "e906435cc445e38961d86f3ebf53a7b38603b903"
PARLAMINT_COMMIT = "230144d62da0011007732d85447e8fb402e4a81b"
PARLAMINT_URL = (
    f"https://raw.githubusercontent.com/clarin-eric/ParlaMint/{PARLAMINT_COMMIT}"
    "/Samples/ParlaMint-NO/ParlaMint-NO-listPerson.xml"
)
OUT = Path("subsets/ssc")


def npsc_sentences() -> pd.DataFrame:
    files = [
        f
        for f in HfApi().list_repo_files(NPSC, repo_type="dataset", revision=NPSC_REVISION)
        if f.startswith("data/") and f.endswith(".json")
    ]
    rows = []
    for name in files:
        path = hf_hub_download(NPSC, name, repo_type="dataset", revision=NPSC_REVISION)
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    df = pd.DataFrame(rows)
    df["utterance_id"] = "npsc_" + df["sentence_id"].astype(str)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    nb = pd.concat(
        [load_metadata(cfg.data, split).assign(split=split) for split in SPLITS],
        ignore_index=True,
    )
    npsc = npsc_sentences()
    joined = nb.merge(npsc, on="utterance_id", how="left", suffixes=("", "_npsc"))
    same = joined["speaker_id"] == "npsc_" + joined["speaker_id_npsc"].astype("Int64").astype(str)
    if not same.all():
        raise ValueError(f"{(~same).sum()} npsc_nb utterances do not match NPSC's metadata")

    held = joined[joined["split"] != "train"]
    names = held.groupby("speaker_id")["speaker_name"].agg(set)
    if (names.map(len) != 1).any():
        raise ValueError("A held-out speaker has more than one name in NPSC")
    names = names.map(lambda s: next(iter(s)))
    unnamed = names[names == "unknown"]

    people = read_parlamint_people(cached_download(PARLAMINT_URL))
    ids = {sid: match_speaker(name, people) for sid, name in names.items() if name != "unknown"}
    missing = [names[sid] for sid, pid in ids.items() if pid is None]
    if missing:
        raise ValueError(f"No unique ParlaMint person for {missing}")

    unknown_utts = held["speaker_id"].isin(unnamed.index).sum()
    header = (
        f"ParlaMint-NO IDs of the {len(ids)} named speakers in NPSC validation and test\n"
        f"({cfg.data.dataset_id}); {len(unnamed)} unnamed speaker ID(s) ('unknown',\n"
        f"{unknown_utts} utterances) cannot be matched. Names from {NPSC}@{NPSC_REVISION[:7]},\n"
        f"people from ParlaMint@{PARLAMINT_COMMIT[:7]}. Made by scripts/make_ssc_lists.py."
    )
    write_id_list(OUT / "held_out_speakers.txt", sorted(set(ids.values())), header)
    print(header)

    dates = sorted({f"{d[:4]}-{d[4:6]}-{d[6:]}" for d in npsc["meeting_date"]})
    header = (
        f"The {len(dates)} NPSC meeting dates ({NPSC}@{NPSC_REVISION[:7]}), all splits.\n"
        "SSC segments from these dates are not used. Made by scripts/make_ssc_lists.py."
    )
    write_id_list(OUT / "npsc_meeting_dates.txt", dates, header)
    print(header)


def cached_download(url: str) -> Path:
    """Download a small file once into the HF cache folder."""
    import urllib.request

    from huggingface_hub.constants import HF_HUB_CACHE

    path = Path(HF_HUB_CACHE) / "voxnor" / Path(url).name
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


if __name__ == "__main__":
    main()
