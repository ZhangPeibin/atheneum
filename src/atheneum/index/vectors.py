"""Dense vector search without a vector database.

Vectors live in one contiguous float32 matrix, so a query is a single
matrix-vector product. Rows are L2-normalized on insertion, which turns cosine
similarity into a dot product and removes a per-query normalization pass.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from atheneum.index.selection import top_k_indices

__all__ = ["VectorIndex"]

_INITIAL_CAPACITY = 256


class DimensionMismatchError(ValueError):
    """Raised when a vector does not match the index dimensionality."""


class VectorIndex:
    """Append-only dense index with brute-force cosine similarity."""

    def __init__(self, dim: int | None = None) -> None:
        if dim is not None and dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        self._dim = dim
        self._matrix: np.ndarray | None = None
        self._count = 0
        self._capacity = 0

    def __len__(self) -> int:
        return self._count

    @property
    def dim(self) -> int | None:
        return self._dim

    @property
    def matrix(self) -> np.ndarray:
        """The normalized vectors as a (count, dim) array.

        A copy, not a view. ``setflags(write=False)`` on the slice only made the
        slice itself immutable: ``matrix.base`` was still the writable internal
        capacity buffer, so a caller could mutate stored vectors and read rows
        beyond ``count`` (shape (256, dim) for a 3-vector index). This accessor
        is for inspection and tests, never on the query path, so the copy is free
        where it matters.
        """
        if self._matrix is None:
            return np.empty((0, 0), dtype=np.float32)
        out = self._matrix[: self._count].copy()
        out.setflags(write=False)
        return out

    @staticmethod
    def normalize(vector: Sequence[float] | np.ndarray) -> np.ndarray:
        array = np.asarray(vector, dtype=np.float32).ravel()
        norm = float(np.linalg.norm(array))
        if norm == 0.0:
            # A zero vector has no direction; returning it unchanged keeps the
            # all-zeros row inert instead of producing NaN similarities.
            return array
        return array / np.float32(norm)

    def add(self, vector: Sequence[float] | np.ndarray) -> int:
        row = self.normalize(vector)
        if not np.isfinite(row).all():
            # A NaN row normalizes to NaN and then silently never matches, so the
            # chunk quietly disappears from dense retrieval.
            raise ValueError("vectors must be finite; NaN and inf cannot be indexed")
        self._dim = self._ensure_dim(row.size)
        self._ensure_capacity(self._count + 1)
        assert self._matrix is not None  # for type checkers; _ensure_capacity set it
        self._matrix[self._count] = row
        self._count += 1
        return self._count - 1

    def add_batch(self, vectors: Iterable[Sequence[float] | np.ndarray]) -> None:
        rows = [self.normalize(v) for v in vectors]
        if not rows:
            return
        width = rows[0].size
        for row in rows:
            if row.size != width:
                raise DimensionMismatchError(
                    f"all vectors in a batch must share a dimension; got {width} and {row.size}"
                )
        self._dim = self._ensure_dim(width)
        stacked = np.vstack(rows).astype(np.float32, copy=False)
        self._ensure_capacity(self._count + len(stacked))
        assert self._matrix is not None
        self._matrix[self._count : self._count + len(stacked)] = stacked
        self._count += len(stacked)

    def load(self, vectors: Sequence[bytes] | Sequence[Sequence[float]], dim: int | None = None) -> None:
        """Rebuild from persisted rows. ``bytes`` rows are raw float32 buffers."""
        if len(vectors) == 0:
            self._dim = dim
            self._matrix = None
            self._count = 0
            self._capacity = 0
            return

        rows: list[np.ndarray] = []
        for entry in vectors:
            if isinstance(entry, bytes | bytearray | memoryview):
                buffer = bytes(entry)
                if len(buffer) % 4:
                    raise DimensionMismatchError(
                        f"buffer of {len(buffer)} bytes is not a whole number of float32 values"
                    )
                # Infer the width from the buffer when the caller did not state it,
                # so `load(export())` round-trips. Requiring dim made the natural
                # pairing of the two methods fail outright.
                width = dim if dim is not None else self._dim
                if width is None:
                    width = len(buffer) // 4
                if len(buffer) != width * 4:
                    raise DimensionMismatchError(
                        f"buffer of {len(buffer)} bytes does not hold {width} float32 values"
                    )
                rows.append(np.frombuffer(buffer, dtype=np.float32).copy())
            else:
                rows.append(np.asarray(entry, dtype=np.float32).ravel())

        width = dim or rows[0].size
        for row in rows:
            if row.size != width:
                raise DimensionMismatchError(
                    f"expected dimension {width}, got a vector of size {row.size}"
                )
        if self._dim is not None and self._dim != width:
            # Honour a dim declared at construction rather than silently adopting
            # whatever the persisted rows happen to have.
            raise DimensionMismatchError(
                f"index was built with dimension {self._dim}; cannot load {width}-d vectors"
            )
        self._dim = width
        # Normalize on load as well as on add: load() used to store rows verbatim,
        # so a non-unit blob made the dot product stop being cosine similarity.
        # Already-unit rows are left alone, because normalizing a second time in
        # float32 perturbs the last bits and near-tied scores could flip order
        # across an export/load round-trip.
        normalized = [
            row if _is_unit(row) else self.normalize(row) for row in rows
        ]
        for position, row in enumerate(normalized):
            if not np.isfinite(row).all():
                raise DimensionMismatchError(
                    f"row {position} is not finite (NaN or inf) and cannot be indexed"
                )
        self._count = len(rows)
        self._capacity = len(rows)
        self._matrix = np.vstack(normalized).astype(np.float32, copy=False)

    def export(self) -> list[bytes]:
        """Serialize rows as raw little-endian float32 buffers."""
        if self._matrix is None or self._count == 0:
            return []
        return [self._matrix[i].tobytes() for i in range(self._count)]

    def search(self, query: Sequence[float] | np.ndarray, top_k: int = 10) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(row_index, cosine_similarity)`` pairs."""
        if self._count == 0 or top_k <= 0 or self._matrix is None:
            return []

        vector = self.normalize(query)
        if vector.size != self._dim:
            raise DimensionMismatchError(
                f"query dimension {vector.size} does not match index dimension {self._dim}"
            )

        similarities = self._matrix[: self._count] @ vector
        return _top_k(similarities, top_k)

    def similarity_to_row(self, row_index: int, query: Sequence[float] | np.ndarray) -> float:
        if self._matrix is None or not 0 <= row_index < self._count:
            raise IndexError(f"row {row_index} is out of range for {self._count} vectors")
        return float(self._matrix[row_index] @ self.normalize(query))

    def _ensure_dim(self, width: int) -> int:
        if width <= 0:
            raise DimensionMismatchError("vectors must have a positive dimension")
        if self._dim is None:
            self._dim = width
        elif self._dim != width:
            raise DimensionMismatchError(
                f"index was built with dimension {self._dim}; cannot add a {width}-dimensional vector"
            )
        return self._dim

    def _ensure_capacity(self, needed: int) -> None:
        assert self._dim is not None
        if self._matrix is None:
            self._capacity = max(_INITIAL_CAPACITY, needed)
            self._matrix = np.zeros((self._capacity, self._dim), dtype=np.float32)
            return
        if needed <= self._capacity:
            return
        # Amortize reallocation: doubling keeps append O(1) on average without
        # holding unbounded slack for small corpora.
        self._capacity = max(needed, self._capacity * 2)
        grown = np.zeros((self._capacity, self._dim), dtype=np.float32)
        grown[: self._count] = self._matrix[: self._count]
        self._matrix = grown


def _is_unit(row: np.ndarray, tolerance: float = 1e-6) -> bool:
    """True when a row is already L2-normalized, so it need not be touched."""
    if row.size == 0:
        return False
    norm = float(np.linalg.norm(row))
    return abs(norm - 1.0) <= tolerance


def _top_k(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    return [(int(i), float(scores[i])) for i in top_k_indices(scores, top_k)]
