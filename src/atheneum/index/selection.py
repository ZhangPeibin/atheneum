"""Deterministic top-k selection over a dense score vector.

Both retrievers need this and both need it to behave identically, so it lives in
one place. The subtlety is worth stating: ``np.argpartition`` selects the right
*number* of top elements in O(n) but chooses arbitrarily among ties, and sorting
afterwards fixes the order of the chosen few without fixing *which* few were
chosen. Ten identical rows asked for the top four returned rows 4, 6, 7 and 8.

Since reproducible ranking is a stated guarantee of this package, ties are
resolved to the lowest row index. Selection stays O(n) by partitioning only to
find the k-th score, then taking every row that beats it and filling the
remainder from the tied rows in ascending index order.
"""

from __future__ import annotations

import numpy as np

__all__ = ["top_k_indices"]


def top_k_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    """Row indices of the ``top_k`` highest positive scores.

    Ordered by descending score, ties broken by ascending index. Rows scoring
    zero or below are excluded, because a zero is "no match" rather than a weak
    match, and returning it would fill result slots with noise.
    """
    if top_k <= 0 or scores.size == 0:
        return np.empty(0, dtype=np.int64)

    positive = np.flatnonzero(scores > 0)
    if positive.size == 0:
        return np.empty(0, dtype=np.int64)
    if positive.size <= top_k:
        selected = positive
    else:
        negated = -scores[positive]
        # O(n): the k-th smallest negated score is the k-th largest real score.
        threshold = np.partition(negated, top_k - 1)[top_k - 1]
        better = positive[negated < threshold]
        if better.size >= top_k:
            selected = better
        else:
            tied = positive[negated == threshold]
            need = top_k - better.size
            # `positive` is ascending, so `tied` is too: taking the first `need`
            # is exactly "lowest index wins".
            selected = np.concatenate([better, tied[:need]])

    # `selected` is ascending, so a stable sort on descending score leaves tied
    # rows in ascending index order.
    return selected[np.argsort(-scores[selected], kind="stable")]
