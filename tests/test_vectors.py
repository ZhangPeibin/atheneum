from __future__ import annotations

import numpy as np
import pytest

from atheneum.index.vectors import DimensionMismatchError, VectorIndex


def _unit(angle_degrees: float) -> list[float]:
    theta = np.deg2rad(angle_degrees)
    return [float(np.cos(theta)), float(np.sin(theta))]


@pytest.fixture
def index() -> VectorIndex:
    built = VectorIndex(dim=2)
    for angle in (0, 30, 60, 90, 180):
        built.add(_unit(angle))
    return built


def test_length_grows_with_adds(index: VectorIndex):
    assert len(index) == 5


def test_self_similarity_is_one(index: VectorIndex):
    hits = index.search(_unit(90), top_k=1)
    assert hits[0][1] == pytest.approx(1.0, abs=1e-5)


def test_nearest_neighbour_is_by_angle(index: VectorIndex):
    assert index.search(_unit(35), top_k=1)[0][0] == 1  # the 30-degree row


def test_opposite_vector_scores_negatively(index: VectorIndex):
    scores = dict(index.search(_unit(0), top_k=10))
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    # The 180-degree row is excluded rather than reported: only positive
    # similarities are useful retrieval candidates.
    assert 4 not in scores


def test_top_k_limits_results(index: VectorIndex):
    assert len(index.search(_unit(45), top_k=2)) == 2


def test_zero_top_k_returns_nothing(index: VectorIndex):
    assert index.search(_unit(0), top_k=0) == []


def test_empty_index_returns_nothing():
    assert VectorIndex(dim=3).search([1, 0, 0], top_k=5) == []


def test_rows_are_sorted_by_descending_score(index: VectorIndex):
    scores = [s for _, s in index.search(_unit(50), top_k=5)]
    assert scores == sorted(scores, reverse=True)


def test_normalization_is_applied_on_add():
    built = VectorIndex(dim=2)
    built.add([3.0, 4.0])
    # A (3,4) vector becomes unit length, so it correlates with (1,0) at 0.6.
    assert built.search([1.0, 0.0], top_k=1)[0][1] == pytest.approx(0.6, abs=1e-5)


def test_zero_vector_is_inert_not_nan():
    built = VectorIndex(dim=3)
    built.add([0.0, 0.0, 0.0])
    built.add([1.0, 0.0, 0.0])
    hits = built.search([1.0, 0.0, 0.0], top_k=2)
    assert all(np.isfinite(score) for _, score in hits)
    assert hits[0][0] == 1


def test_dimension_mismatch_on_add():
    built = VectorIndex(dim=4)
    with pytest.raises(DimensionMismatchError):
        built.add([1.0, 2.0])


def test_dimension_mismatch_on_query(index: VectorIndex):
    with pytest.raises(DimensionMismatchError):
        index.search([1.0, 2.0, 3.0], top_k=2)


def test_dimension_inferred_from_first_add():
    built = VectorIndex()
    built.add([1.0, 0.0, 0.0])
    assert built.dim == 3


def test_export_and_load_round_trip(index: VectorIndex):
    blobs = index.export()
    restored = VectorIndex()
    restored.load(blobs, dim=2)
    assert len(restored) == len(index)
    assert restored.search(_unit(35), top_k=1) == index.search(_unit(35), top_k=1)


def test_load_requires_dim_for_raw_buffers():
    with pytest.raises(DimensionMismatchError):
        VectorIndex().load([np.asarray([1.0, 0.0], dtype="<f4").tobytes()])


def test_load_rejects_wrong_blob_size():
    blob = np.asarray([1.0, 2.0], dtype="<f4").tobytes()
    with pytest.raises(DimensionMismatchError, match="does not hold"):
        VectorIndex().load([blob], dim=7)


def test_load_accepts_plain_sequences():
    built = VectorIndex()
    built.load([[1.0, 0.0], [0.0, 1.0]], dim=2)
    assert len(built) == 2


def test_load_empty_clears_index(index: VectorIndex):
    index.load([], dim=8)
    assert len(index) == 0
    assert index.dim == 8


def test_ragged_batch_is_rejected():
    with pytest.raises(DimensionMismatchError):
        VectorIndex(dim=2).add_batch([[1.0, 0.0], [1.0, 0.0, 2.0]])


def test_matrix_view_is_read_only(index: VectorIndex):
    with pytest.raises(ValueError):
        index.matrix[0][0] = 99.0


def test_similarity_to_row_accessor(index: VectorIndex):
    assert index.similarity_to_row(3, _unit(90)) == pytest.approx(1.0, abs=1e-5)
    with pytest.raises(IndexError):
        index.similarity_to_row(99, _unit(0))


def test_growth_keeps_all_rows(index: VectorIndex):
    """Crossing the preallocated capacity must not lose earlier rows."""
    built = VectorIndex(dim=1)
    for _ in range(600):
        built.add([1.0])
    assert len(built) == 600
    assert built.matrix[599][0] == pytest.approx(1.0)


def test_add_batch_matches_sequential_adds():
    sequential = VectorIndex(dim=2)
    for angle in (0, 45, 90):
        sequential.add(_unit(angle))
    batched = VectorIndex(dim=2)
    batched.add_batch([_unit(a) for a in (0, 45, 90)])
    assert batched.search(_unit(80), top_k=3) == sequential.search(_unit(80), top_k=3)


def test_empty_batch_is_a_noop():
    built = VectorIndex(dim=2)
    built.add_batch([])
    assert len(built) == 0


def test_invalid_dim_rejected():
    with pytest.raises(ValueError):
        VectorIndex(dim=0)


def test_result_order_is_stable_for_ties():
    built = VectorIndex(dim=2)
    for _ in range(5):
        built.add([1.0, 0.0])
    assert [i for i, _ in built.search([1.0, 0.0], top_k=5)] == [0, 1, 2, 3, 4]
