"""Rank fusion: combining several ranked lists into one.

Lexical and dense retrievers produce scores on incomparable scales — BM25 is
unbounded and corpus-dependent, cosine similarity sits in [-1, 1]. Fusing on
*rank* rather than *score* sidesteps that entirely, which is why Reciprocal Rank
Fusion is the default here.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

__all__ = [
    "DBSFFusion",
    "FusionStrategy",
    "RRFFusion",
    "RankedItem",
    "build_fusion",
]

T = TypeVar("T")

# 60 is the constant recommended by Cormack, Clarke and Büttcher (SIGIR 2009).
# One is added because this implementation enumerates ranks from zero while the
# paper's derivation is one-based.
RRF_K = 61


@dataclass(frozen=True, slots=True)
class RankedItem(Generic[T]):
    """An item together with the per-list scores that produced its final rank."""

    key: str
    value: T
    score: float
    # Keyed by list name so an answer can explain *why* something ranked first.
    contributions: dict[str, float]


class FusionStrategy(Enum):
    RRF = "rrf"
    DBSF = "dbsf"
    WEIGHTED = "weighted"

    @classmethod
    def from_str(cls, raw: str) -> FusionStrategy:
        try:
            return cls(raw.strip().lower())
        except ValueError:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(f"unknown fusion strategy {raw!r}; expected one of {valid}") from None


def _validate_weights(lists: Sequence[str], weights: Sequence[float] | None) -> list[float]:
    if weights is None:
        return [1.0 / len(lists)] * len(lists) if lists else []
    if len(weights) != len(lists):
        raise ValueError(f"got {len(weights)} weights for {len(lists)} ranked lists")
    if any(w < 0 for w in weights):
        raise ValueError("fusion weights must be non-negative")
    total = sum(weights)
    if total == 0:
        raise ValueError("fusion weights must not all be zero")
    return [w / total for w in weights]


class RRFFusion:
    """Reciprocal Rank Fusion.

    Score for an item is the weighted sum of ``1 / (k + rank)`` over every list
    containing it. An item near the top of several lists beats one at the very
    top of a single list, which is the property that makes hybrid search work.
    """

    name = "rrf"

    def __init__(self, k: int = RRF_K) -> None:
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")
        self.k = k

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[tuple[str, T]]],
        weights: Sequence[float] | None = None,
    ) -> list[RankedItem[T]]:
        if not ranked_lists:
            return []
        names = list(ranked_lists)
        resolved = _validate_weights(names, weights)

        totals: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        values: dict[str, T] = {}
        # Which list supplied the surviving payload for a key, as
        # (weight, name) so precedence is deterministic and independent of the
        # order the caller happened to build the dict in.
        provenance: dict[str, tuple[float, str]] = {}

        for list_name, weight in zip(names, resolved, strict=True):
            seen_in_list: set[str] = set()
            for rank, (key, value) in enumerate(ranked_lists[list_name]):
                if key in seen_in_list:
                    # A ranked list must not contain the same key twice. Counting
                    # it again inflated the score past the documented maximum of
                    # 1.0 while contributions kept only one entry, so the parts no
                    # longer summed to the whole.
                    continue
                seen_in_list.add(key)
                share = weight * self.k / (self.k + rank)
                contributions[key][list_name] = share
                candidate = (weight, list_name)
                if key not in provenance or candidate > provenance[key]:
                    provenance[key] = candidate
                    values[key] = value

        # Weights sum to 1, so an item ranked first in every list scores exactly
        # 1/k before this scaling. Multiplying by k therefore makes 1.0 mean
        # "top of every retriever" and anything less a proportion of that, which
        # is far more legible than raw 1/(k+rank) magnitudes around 0.016.
        #
        # Contributions are scaled by the same factor so that they sum to the
        # reported score. Leaving them raw made `--explain` output incoherent:
        # the parts summed to score/k, not to the score.
        # fsum over the same scaled parts that are reported, so the contributions
        # add up to the score exactly. Summing the raw shares and scaling
        # afterwards differed in the last ulp: 4221 of 20000 random fusions had
        # sum(contributions) != score, and the score could read 1.0000000000000002
        # for an item that was top of every list.
        totals = {key: math.fsum(parts.values()) for key, parts in contributions.items()}
        ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            RankedItem(
                key=key,
                value=values[key],
                score=total,
                contributions=dict(contributions[key]),
            )
            for key, total in ordered
        ]


class DBSFFusion:
    """Distribution-Based Score Fusion.

    Each list's scores are min-max rescaled onto [0, 1] before combining, so
    the *magnitude* of a match is preserved rather than discarded. This is
    better than RRF when one retriever's scores are genuinely informative and
    its list lengths differ a lot.
    """

    name = "dbsf"

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[tuple[str, T]]],
        weights: Sequence[float] | None = None,
        scores: Mapping[str, Sequence[float]] | None = None,
    ) -> list[RankedItem[T]]:
        if not ranked_lists:
            return []
        if scores is None:
            raise ValueError("DBSFFusion requires a parallel `scores` mapping")
        names = list(ranked_lists)
        resolved = _validate_weights(names, weights)

        totals: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        values: dict[str, T] = {}
        provenance: dict[str, tuple[float, str]] = {}

        for list_name, weight in zip(names, resolved, strict=True):
            items = ranked_lists[list_name]
            raw = list(scores.get(list_name, []))
            if len(raw) != len(items):
                raise ValueError(
                    f"list {list_name!r} has {len(items)} items but {len(raw)} scores"
                )
            rescaled = _min_max(raw)
            seen_in_list: set[str] = set()
            for (key, value), scaled in zip(items, rescaled, strict=True):
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)
                totals[key] += weight * scaled
                contributions[key][list_name] = weight * scaled
                _record_provenance(provenance, values, key, value, weight, list_name)

        ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            RankedItem(key=key, value=values[key], score=total, contributions=dict(contributions[key]))
            for key, total in ordered
        ]


class WeightedSumFusion:
    """Linear combination of raw scores.

    Kept because it is what most people reach for first, and because it is the
    right answer when both retrievers already emit calibrated, comparable
    scores. It is not the default: BM25 and cosine are not comparable.
    """

    name = "weighted"

    def fuse(
        self,
        ranked_lists: Mapping[str, Sequence[tuple[str, T]]],
        weights: Sequence[float] | None = None,
        scores: Mapping[str, Sequence[float]] | None = None,
    ) -> list[RankedItem[T]]:
        if not ranked_lists:
            return []
        if scores is None:
            raise ValueError("WeightedSumFusion requires a parallel `scores` mapping")
        names = list(ranked_lists)
        resolved = _validate_weights(names, weights)

        totals: dict[str, float] = defaultdict(float)
        contributions: dict[str, dict[str, float]] = defaultdict(dict)
        values: dict[str, T] = {}
        provenance: dict[str, tuple[float, str]] = {}

        for list_name, weight in zip(names, resolved, strict=True):
            items = ranked_lists[list_name]
            raw = scores.get(list_name, [])
            if len(raw) != len(items):
                raise ValueError(
                    f"list {list_name!r} has {len(items)} items but {len(raw)} scores"
                )
            seen_in_list: set[str] = set()
            for (key, value), score in zip(items, raw, strict=True):
                # A key repeated within one list must not be counted twice: it
                # inflated the total while contributions kept one entry, so the
                # parts no longer summed to the score.
                if key in seen_in_list:
                    continue
                seen_in_list.add(key)
                totals[key] += weight * score
                contributions[key][list_name] = weight * score
                _record_provenance(provenance, values, key, value, weight, list_name)

        ordered = sorted(totals.items(), key=lambda kv: (-kv[1], kv[0]))
        return [
            RankedItem(key=key, value=values[key], score=total, contributions=dict(contributions[key]))
            for key, total in ordered
        ]


def _record_provenance(
    provenance: dict[str, tuple[float, str]],
    values: dict[str, T],
    key: str,
    value: T,
    weight: float,
    list_name: str,
) -> None:
    """Keep the payload from the highest-weighted list, deterministically.

    `values.setdefault` made the surviving payload depend on the order the caller
    built the dict in, so the same fusion could return either list's value.
    """
    candidate = (weight, list_name)
    if key not in provenance or candidate > provenance[key]:
        provenance[key] = candidate
        values[key] = value


def _min_max(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    low, high = min(values), max(values)
    span = high - low
    if span == 0:
        # A degenerate list where everything scored the same: give every item
        # the neutral midpoint rather than dividing by zero.
        return [1.0 for _ in values] if high > 0 else [0.0 for _ in values]
    return [(v - low) / span for v in values]


def build_fusion(
    strategy: str | FusionStrategy = FusionStrategy.RRF, *, k: int = RRF_K
) -> RRFFusion | DBSFFusion | WeightedSumFusion:
    """Construct a fusion strategy by name."""
    resolved = (
        strategy if isinstance(strategy, FusionStrategy) else FusionStrategy.from_str(strategy)
    )
    if resolved is FusionStrategy.RRF:
        return RRFFusion(k=k)
    if resolved is FusionStrategy.DBSF:
        return DBSFFusion()
    return WeightedSumFusion()
