import pytest

from voxnor.normalize import (
    normalize_hypothesis,
    normalize_reference,
    score_form,
    to_reference_conventions,
)

# (raw_text, standardized_text) pairs taken from the corpus.
CORPUS_PAIRS = [
    ("Stortingets møte er lovlig satt", "stortingets møte er lovlig satt"),
    (
        "det er for Vest-Agder fylke Ånen Werdal",
        "det er for vest-agder fylke ånen werdal",
    ),
    (
        "heldigvis har vi|1 stort sett vært heldig med oppfølginga",
        "heldigvis har vi stort sett vært heldig med oppfølginga",
    ),
    (
        "fra toogførti|førtito tusen til femogtredve|trettifem tusen",
        "fra toogførti tusen til femogtredve tusen",
    ),
    (
        "flertallet i Stortinget <ee> stiller seg bak",
        "flertallet i stortinget eee stiller seg bak",
    ),
    ("statsminister Erna <INAUDIBLE>", "statsminister erna"),
    ("og det er jo et paradoks òg opp mot", "og det er jo et paradoks og opp mot"),
    ("vårt|1 engasjement for gründerkapital", "vårt engasjement for grynderkapital"),
    ("regjeringen har stilt krav om barne- og voksenhabilitering", None),
]


@pytest.mark.parametrize(("raw", "standardized"), [p for p in CORPUS_PAIRS if p[1]])
def test_reproduces_standardized_text(raw, standardized):
    assert to_reference_conventions(raw) == standardized


def test_trailing_hyphen_of_elided_compound_is_dropped():
    assert to_reference_conventions("barne- og voksenhabilitering") == "barne og voksenhabilitering"


@pytest.mark.parametrize(
    ("hypothesis", "expected"),
    [
        ("Hei, Stortinget!", "hei stortinget"),
        ("«Sitat»: det – er (slik).", "sitat det er slik"),  # noqa: RUF001
        ("Det er 25 prosent.", "det er 25 prosent"),
        ("CO2-utslipp", "co2 utslipp"),
        ("Vest-Agder", "vest agder"),
        ("Øh, jeg mener ... eh, ja", "jeg mener ja"),
        ("én gang", "en gang"),
        ("Zürich og Malmö", "zyrich og malmø"),
        ("naïve café", "naive cafe"),
        ("statsrådens", "statsrådens"),
        ("Stortinget's", "stortinget s"),
        ("", ""),
        ("  \n\t ", ""),
    ],
)
def test_normalize_hypothesis(hypothesis, expected):
    assert normalize_hypothesis(hypothesis) == expected


def test_reference_and_hypothesis_agree_on_the_same_text():
    standardized = "det er en av grunnene til at vi valgte å øke antallet"
    assert normalize_reference(standardized) == normalize_hypothesis(
        standardized.capitalize() + "."
    )


def test_score_form_removes_hesitations_only_as_whole_tokens():
    assert score_form("eee hmm mmm eeee hmmm") == "eeee hmmm"


def test_hesitation_only_reference_becomes_empty():
    assert normalize_reference("mmm") == ""


def test_normalisation_is_idempotent():
    for raw, _ in CORPUS_PAIRS:
        once = normalize_hypothesis(raw)
        assert normalize_hypothesis(once) == once
