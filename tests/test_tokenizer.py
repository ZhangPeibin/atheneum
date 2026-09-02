from __future__ import annotations

import atheneum
from atheneum.text.tokenizer import ENGLISH_STOPWORDS, token_frequencies, tokenize


def test_lowercase_and_split():
    assert tokenize("Reciprocal Rank Fusion") == ["reciprocal", "rank", "fusion"]


def test_stopwords_removed_by_default():
    tokens = tokenize("the rank of a document")
    assert "the" not in tokens
    assert "of" not in tokens
    assert "rank" in tokens


def test_stopwords_kept_when_requested():
    tokens = tokenize("the rank", remove_stopwords=False)
    assert "the" in tokens


def test_single_character_words_dropped_but_digits_kept():
    assert tokenize("a x 42") == ["42"]


def test_accents_and_case_folding_match():
    assert tokenize("Café") == tokenize("café")
    assert tokenize("CAFÉ") == tokenize("cafe")


def test_fullwidth_latin_folds_to_ascii():
    # NFKC folds full-width forms, so a query typed in ASCII finds text stored
    # with full-width characters.
    assert tokenize("ＦＴＳ５") == tokenize("fts5")


def test_cjk_yields_unigrams_and_bigrams():
    tokens = tokenize("混合检索")
    assert "混" in tokens
    assert "混合" in tokens
    assert "合检" in tokens
    assert "检索" in tokens


def test_cjk_mixed_with_latin_keeps_both():
    tokens = tokenize("使用 BM25 做词法匹配")
    assert "bm25" in tokens
    assert "词法" in tokens


def test_empty_and_none_like_input():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("...") == []


def test_apostrophes_and_underscores_stay_inside_words():
    tokens = tokenize("node.value don't snake_case")
    assert "don't" in tokens
    assert "snake_case" in tokens
    assert "node" in tokens
    assert "value" in tokens


def test_token_frequencies_counts():
    assert token_frequencies(["a", "b", "a", "a"]) == {"a": 3, "b": 1}


def test_tokenize_is_idempotent_on_repeated_calls():
    text = "Okapi BM25 saturates term frequency 混合检索"
    assert tokenize(text) == tokenize(text)


def test_public_reexport_matches_internal():
    assert atheneum.tokenize is tokenize


def test_english_stopwords_are_lowercase():
    assert all(word == word.lower() for word in ENGLISH_STOPWORDS)
