import pandas as pd

from voxnor.data_report import build_report, render_markdown, speaker_overlap, split_summary


def meta(rows):
    df = pd.DataFrame(
        rows, columns=["speaker_id", "region", "duration", "full_audio_file", "raw_text"]
    )
    df["gender"] = "female"
    return df


TRAIN = meta(
    [
        ("a", "east", 3600.0, "m1", "hei <ee> vi|1 sees"),
        ("b", "west", 1800.0, "m1", "god dag"),
        ("c", "north", 1800.0, "m2", "<INAUDIBLE>"),
    ]
)
TEST = meta(
    [
        ("a", "east", 1800.0, "m3", "hei"),
        ("d", "west", 1800.0, "m3", "<mm>"),
    ]
)


def test_split_summary_counts_hours_and_markers():
    summary = split_summary(TRAIN)
    assert summary["hours"] == 2.0
    assert summary["speakers"] == 3
    assert summary["hours_by_region"] == {"east": 1.0, "north": 0.5, "west": 0.5}
    assert summary["utterances_with"]["hesitation <ee>"] == 1
    assert summary["utterances_with"]["numbered word marker"] == 1
    assert summary["utterances_with"]["<INAUDIBLE>"] == 1


def test_speaker_overlap_separates_seen_and_unseen_speakers():
    overlap = speaker_overlap(TRAIN, TEST)
    assert overlap["speakers_seen_in_train"] == 1
    assert overlap["utterance_share_seen"] == 0.5
    assert overlap["unseen_hours_by_region"] == {"west": 0.5}
    assert overlap["shared_meetings"] == 0


def test_report_renders_valid_markdown_tables():
    report = build_report({"train": TRAIN, "validation": TEST, "test": TEST})
    markdown = render_markdown(report, "x/y")
    tables = [block for block in markdown.split("\n\n") if block.startswith("|")]
    assert tables
    for table in tables:
        # Every row has as many cells as the header: no stray "|" inside a cell.
        widths = {row.count("|") for row in table.splitlines()}
        assert len(widths) == 1, table
