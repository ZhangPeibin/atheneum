from __future__ import annotations

import math

import numpy as np
import pytest

from atheneum.index.bm25 import BM25Index, BM25Params


@pytest.fixture
def index() -> BM25Index:
    built = BM25Index()
    built.add_batch(
        [
            "the quick brown fox jumps over the lazy dog",
            "a lazy dog sleeps all day long",
            "the quick brown fox is clever and fast",
            "stock market prices rose sharply today",
        ]
    )
    return built


def test_relevant_document_ranks_first(index: BM25Index):
    hits = index.search("quick brown fox", top_k=3)
    assert hits
    assert hits[0][0] in (0, 2)


def test_uncommon_term_beats_common_term(index: BM25Index):
    # "fox" appears in two docs, "dog" in two, but "sleeps" only in one, so a
    # query for the rare term must rank its document first.
    hits = index.search("sleeps", top_k=4)
    assert hits[0][0] == 1


def test_no_matching_terms_returns_nothing(index: BM25Index):
    assert index.search("zzzqqq", top_k=5) == []


def test_scores_are_positive_and_descending(index: BM25Index):
    hits = index.search("lazy dog", top_k=4)
    assert all(score > 0 for _, score in hits)
    assert [s for _, s in hits] == sorted((s for _, s in hits), reverse=True)


def test_top_k_limit_honoured(index: BM25Index):
    assert len(index.search("the", top_k=2)) <= 2


def test_empty_index_returns_nothing():
    assert BM25Index().search("anything", top_k=5) == []


def test_zero_top_k_returns_nothing(index: BM25Index):
    assert index.search("fox", top_k=0) == []


def test_repeated_query_term_is_not_double_counted(index: BM25Index):
    single = index.search("fox")
    doubled = index.search("fox fox")
    assert single == doubled


def test_idf_matches_the_okapi_definition(index: BM25Index):
    # "sleeps" occurs in exactly 1 of 4 documents.
    expected = math.log((4 - 1 + 0.5) / (1 + 0.5))
    assert index.idf("sleeps") == pytest.approx(expected)


def test_negative_idf_is_floored_not_inverted():
    # A term in more than half the corpus has a negative raw idf. It must come
    # back small and positive, never negative and never above a rarer term.
    built = BM25Index()
    built.add("ubiquitous uniquealpha")
    built.add("ubiquitous uniquebeta")
    built.add("ubiquitous uniquesomethingelse")
    built.finalize()
    raw_negative = math.log((3 - 3 + 0.5) / (3 + 0.5))
    assert raw_negative < 0
    assert built.idf("ubiquitous") > 0
    assert built.idf("ubiquitous") < built.idf("uniquealpha")


def test_length_normalization_prefers_shorter_document_at_equal_frequency():
    # Both documents mention the query term once. Only length differs, so the
    # normalization term alone decides the winner.
    built = BM25Index(BM25Params(k1=1.5, b=0.75))
    built.add("zebras graze")
    built.add("zebras graze " + "padding words " * 30)
    hits = dict(built.search("zebras", top_k=2))
    assert hits[0] > hits[1]


def test_high_term_frequency_beats_short_length():
    # BM25 saturates but does not stop rewarding repetition; asserting otherwise
    # would misunderstand the algorithm.
    built = BM25Index(BM25Params(k1=1.5, b=0.75))
    built.add("zebra")
    built.add("zebra " * 40 + "and a great deal of unrelated filler text entirely")
    hits = dict(built.search("zebra", top_k=2))
    assert hits[1] > hits[0]


def test_b_zero_disables_length_normalization():
    text_a = "alpha beta gamma"
    text_b = "alpha beta gamma " + "padding words " * 60
    for b in (0.0, 0.75):
        built = BM25Index(BM25Params(k1=1.5, b=b))
        built.add(text_a)
        built.add(text_b)
        hits = dict(built.search("alpha", top_k=2))
        if b == 0.0:
            # With no normalization the long doc wins on raw term count, because
            # "padding" never appears in the query.
            assert hits[1] >= hits[0]
        else:
            assert hits[0] > hits[1]


def test_export_and_load_round_trip_is_score_identical(index: BM25Index):
    restored = BM25Index()
    restored.load(index.export_term_frequencies())
    assert restored.search("lazy dog", top_k=4) == index.search("lazy dog", top_k=4)


def test_add_frequencies_avoids_retokensizing():
    built = BM25Index()
    built.add_frequencies({"fox": 2, "quick": 1})
    built.add_frequencies({"dog": 3})
    built.finalize()
    hits = built.search("fox", top_k=2)
    assert hits[0][0] == 0
    assert len(built) == 2


def test_score_all_is_dense_and_matches_sparse(index: BM25Index):
    dense = index.score_all("quick fox")
    assert dense.shape == (4,)
    for position, score in index.search("quick fox", top_k=4):
        assert dense[position] == pytest.approx(score)


def test_average_document_length_and_vocabulary(index: BM25Index):
    from atheneum.text.tokenizer import tokenize

    lengths = [len(tokenize(text)) for text in (
        "the quick brown fox jumps over the lazy dog",
        "a lazy dog sleeps all day long",
        "the quick brown fox is clever and fast",
        "stock market prices rose sharply today",
    )]
    assert index.average_document_length == pytest.approx(sum(lengths) / len(lengths))
    assert index.vocabulary_size > 10


def test_empty_chunk_does_not_divide_by_zero():
    built = BM25Index()
    built.add("")
    built.add("real content here")
    built.finalize()
    scores = built.score_all("real content")
    assert np.isfinite(scores).all()
    assert scores[1] > scores[0]


@pytest.mark.parametrize(
    "kwargs",
    [{"k1": 0.0}, {"k1": -1.0}, {"b": -0.1}, {"b": 1.5}, {"epsilon": 0.0}],
)
def test_invalid_params_rejected(kwargs: dict):
    with pytest.raises(ValueError):
        BM25Params(**kwargs)


def test_results_are_reproducible():
    built = BM25Index()
    for i in range(50):
        built.add(f"document {i} shares the common term and has unique token zzz{i}")
    first = built.search("common term unique", top_k=10)
    second = built.search("common term unique", top_k=10)
    assert first == second


def test_ties_broken_by_index_order():
    built = BM25Index()
    built.add("identical text")
    built.add("identical text")
    hits = built.search("identical", top_k=2)
    assert [i for i, _ in hits] == [0, 1]


def test_cjk_query_matches_cjk_document():
    built = BM25Index()
    built.add("混合检索使用倒数排名融合来合并多个排序列表")
    built.add("The agent loop executes tool calls")
    hits = built.search("如何合并排序列表", top_k=2)
    assert hits[0][0] == 0
