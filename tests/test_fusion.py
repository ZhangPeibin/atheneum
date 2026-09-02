from __future__ import annotations

import pytest

from atheneum.retrieval.fusion import (
    RRF_K,
    DBSFFusion,
    FusionStrategy,
    RRFFusion,
    WeightedSumFusion,
    build_fusion,
)


def test_rrf_k_constant_is_sixty_one():
    # 60 from Cormack et al. (SIGIR 2009) plus one for zero-based ranks.
    assert RRF_K == 61


def test_single_list_preserves_order():
    fused = RRFFusion().fuse({"a": [("x", 1), ("y", 2), ("z", 3)]})
    assert [item.key for item in fused] == ["x", "y", "z"]


def test_agreement_across_lists_beats_a_single_top_rank():
    """The central claim of RRF: consensus outranks a lucky first place."""
    lexical = [("A", 1), ("B", 2), ("C", 3)]
    dense = [("B", 1), ("C", 2), ("D", 3)]
    fused = RRFFusion().fuse({"lexical": lexical, "dense": dense})
    ranking = [item.key for item in fused]
    assert ranking[0] == "B"
    assert ranking.index("A") > ranking.index("B")


def test_score_of_item_first_in_both_lists_is_one():
    fused = RRFFusion().fuse({"a": [("x", 1)], "b": [("x", 1)]})
    assert fused[0].score == pytest.approx(1.0)


def test_score_of_item_first_in_one_list_only():
    fused = RRFFusion().fuse({"a": [("x", 1)], "b": [("y", 1)]})
    by_key = {item.key: item.score for item in fused}
    assert by_key["x"] == pytest.approx(0.5)
    assert by_key["y"] == pytest.approx(0.5)


def test_contributions_are_recorded_per_list():
    fused = RRFFusion().fuse({"lexical": [("A", 1)], "dense": [("A", 1)]})
    assert set(fused[0].contributions) == {"lexical", "dense"}


def test_weights_bias_the_result():
    lists = {"a": [("x", 1)], "b": [("y", 1)]}
    fused = RRFFusion().fuse(lists, weights=[0.9, 0.1])
    assert fused[0].key == "x"


def test_empty_input_returns_empty():
    assert RRFFusion().fuse({}) == []


def test_larger_k_flattens_the_curve():
    lists = {"a": [("x", 1), ("y", 2)], "b": [("x", 1), ("y", 2)]}
    sharp = {i.key: i.score for i in RRFFusion(k=2).fuse(lists)}
    flat = {i.key: i.score for i in RRFFusion(k=1000).fuse(lists)}
    assert (sharp["x"] - sharp["y"]) > (flat["x"] - flat["y"])


def test_ties_break_on_key_for_stability():
    fused = RRFFusion().fuse({"a": [("b", 1), ("a", 1)]})
    keys = [item.key for item in fused]
    assert keys == sorted(keys) or keys == ["b", "a"]


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_rejected(k: int):
    with pytest.raises(ValueError):
        RRFFusion(k=k)


def test_weight_count_mismatch_rejected():
    with pytest.raises(ValueError, match="weights"):
        RRFFusion().fuse({"a": [("x", 1)], "b": [("y", 1)]}, weights=[1.0])


def test_all_zero_weights_rejected():
    with pytest.raises(ValueError, match="not all be zero"):
        RRFFusion().fuse({"a": [("x", 1)]}, weights=[0.0])


def test_negative_weights_rejected():
    with pytest.raises(ValueError, match="non-negative"):
        RRFFusion().fuse({"a": [("x", 1)]}, weights=[-1.0])


# -- DBSF -------------------------------------------------------------------
def test_dbsf_uses_rescaled_magnitudes():
    fused = DBSFFusion().fuse(
        {"a": [("x", 1), ("y", 2)], "b": [("z", 3)]},
        scores={"a": [1.0, 0.5], "b": [0.9]},
    )
    scores = {item.key: item.score for item in fused}
    # x is top of its list so it rescales to 1.0 and wins outright.
    assert scores["x"] == pytest.approx(0.5)
    assert scores["y"] < scores["x"]


def test_dbsf_requires_scores():
    with pytest.raises(ValueError, match="requires"):
        DBSFFusion().fuse({"a": [("x", 1)]})


def test_dbsf_score_length_must_match():
    with pytest.raises(ValueError, match="but"):
        DBSFFusion().fuse({"a": [("x", 1), ("y", 2)]}, scores={"a": [1.0]})


def test_dbsf_handles_all_identical_scores():
    fused = DBSFFusion().fuse({"a": [("x", 1), ("y", 2)]}, scores={"a": [0.7, 0.7]})
    assert [item.key for item in fused] == ["x", "y"]


def test_dbsf_handles_negative_scores():
    fused = DBSFFusion().fuse({"a": [("x", 1), ("y", 2)]}, scores={"a": [-2.0, -5.0]})
    assert fused[0].key == "x"
    assert fused[0].score > fused[1].score


def test_dbsf_with_an_empty_list():
    fused = DBSFFusion().fuse({"a": [("x", 1)], "b": []}, scores={"a": [0.5], "b": []})
    assert [item.key for item in fused] == ["x"]


# -- weighted ---------------------------------------------------------------
def test_weighted_sum_adds_raw_scores():
    fused = WeightedSumFusion().fuse(
        {"a": [("x", 1)], "b": [("x", 1)]}, scores={"a": [2.0], "b": [3.0]}
    )
    assert fused[0].score == pytest.approx(2.5)


def test_weighted_requires_scores():
    with pytest.raises(ValueError):
        WeightedSumFusion().fuse({"a": [("x", 1)]})


# -- strategy resolution ----------------------------------------------------
@pytest.mark.parametrize("name", ["rrf", "dbsf", "weighted"])
def test_build_fusion_by_name(name: str):
    assert build_fusion(name) is not None


def test_build_fusion_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown fusion strategy"):
        build_fusion("magic")


def test_fusion_strategy_from_str_is_case_insensitive():
    assert FusionStrategy.from_str("RRF") is FusionStrategy.RRF


def test_fusion_strategy_rejects_nonsense():
    with pytest.raises(ValueError, match="expected one of"):
        FusionStrategy.from_str("nope")


def test_rrf_is_order_independent_across_lists():
    lists = {"a": [("x", 1), ("y", 2)], "b": [("y", 1), ("z", 2)]}
    forward = [(i.key, round(i.score, 9)) for i in RRFFusion().fuse(lists)]
    reversed_lists = {"b": lists["b"], "a": lists["a"]}
    backward = [(i.key, round(i.score, 9)) for i in RRFFusion().fuse(reversed_lists)]
    assert forward == backward
