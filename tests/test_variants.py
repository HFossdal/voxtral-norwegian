from voxnor.variants import corpus_pairs, fold_variants


def test_spoken_and_standard_forms_fold_together():
    assert fold_variants("regjeringa") == fold_variants("regjeringen")
    assert fold_variants("vi har løfta opp") == fold_variants("vi har løftet opp")
    assert fold_variants("jeg meiner sjøl at ei sak") == fold_variants("jeg mener selv at en sak")


def test_different_words_stay_different():
    for a, b in [("da", "den"), ("det", "de"), ("dem", "de"), ("regjering", "regjeringen")]:
        assert fold_variants(a) != fold_variants(b)


def test_danish_spelling_is_not_folded():
    assert fold_variants("præsident") != fold_variants("president")
    assert fold_variants("netop") != fold_variants("nettopp")


def test_corpus_pairs_skip_markers_and_numbers():
    texts = ["Det kjem|kommer an", "vi|1 kjem|kommer", "femogtjue|tjuefem kjem|kjem"]
    assert corpus_pairs(texts) == {"kjem": "kommer"}
    assert fold_variants("kjem", corpus_pairs(texts)) == fold_variants("kommer")
