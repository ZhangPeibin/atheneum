"""Okapi BM25 over an inverted index.

Scores are computed by gathering postings for the query terms only, so cost
scales with the number of matching chunks rather than the size of the corpus.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from atheneum.index.selection import top_k_indices
from atheneum.text.tokenizer import token_frequencies, tokenize

__all__ = ["BM25Index", "BM25Params"]


@dataclass(frozen=True, slots=True)
class BM25Params:
    """Saturation and length-normalization constants.

    These are the conventional Okapi defaults; they are exposed because the
    right value of ``b`` genuinely varies with how uniform chunk lengths are.
    """

    k1: float = 1.5
    b: float = 0.75
    # Terms appearing in more than half the corpus produce a negative raw IDF,
    # which would let a ubiquitous word penalise the documents containing it.
    # The floor rescales those terms to a small positive value instead.
    epsilon: float = 0.25

    def __post_init__(self) -> None:
        if self.k1 <= 0:
            raise ValueError(f"k1 must be positive, got {self.k1}")
        if not 0.0 <= self.b <= 1.0:
            raise ValueError(f"b must be within [0, 1], got {self.b}")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")


@dataclass(slots=True)
class _Posting:
    chunk_indices: np.ndarray
    term_frequencies: np.ndarray


class BM25Index:
    """In-memory inverted index with Okapi BM25 scoring."""

    def __init__(self, params: BM25Params | None = None) -> None:
        self.params = params or BM25Params()
        self._term_frequencies: list[dict[str, int]] = []
        self._doc_lengths: list[int] = []
        self._postings: dict[str, _Posting] = {}
        self._idf: dict[str, float] = {}
        self._avgdl: float = 0.0
        self._dirty = False

    def __len__(self) -> int:
        return len(self._term_frequencies)

    @property
    def average_document_length(self) -> float:
        return self._avgdl

    @property
    def vocabulary_size(self) -> int:
        return len(self._idf)

    def add(self, text: str, tokens: Sequence[str] | None = None) -> int:
        """Append a chunk and return its internal index."""
        freqs = token_frequencies(tokens if tokens is not None else tokenize(text))
        self._term_frequencies.append(freqs)
        self._doc_lengths.append(sum(freqs.values()))
        self._dirty = True
        return len(self._term_frequencies) - 1

    def add_frequencies(self, freqs: dict[str, int]) -> int:
        """Append a chunk from an already-computed term-frequency map.

        Reloading a persisted index through this path avoids re-tokenizing every
        chunk, which is the difference between a millisecond and a minute on a
        large corpus.
        """
        cleaned = {term: int(count) for term, count in freqs.items() if count > 0}
        self._term_frequencies.append(cleaned)
        self._doc_lengths.append(sum(cleaned.values()))
        self._dirty = True
        return len(self._term_frequencies) - 1

    def add_batch(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.add(text)
        self.finalize()

    def load(self, term_frequencies: Sequence[dict[str, int]]) -> None:
        """Rebuild the index from persisted term frequencies."""
        self._term_frequencies = [dict(tf) for tf in term_frequencies]
        self._doc_lengths = [sum(tf.values()) for tf in self._term_frequencies]
        self._dirty = True
        self.finalize()

    def export_term_frequencies(self) -> list[dict[str, int]]:
        return [dict(tf) for tf in self._term_frequencies]

    def finalize(self) -> None:
        """Build the inverted index and IDF table. Safe to call repeatedly."""
        if not self._dirty:
            return
        corpus_size = len(self._term_frequencies)
        if corpus_size == 0:
            self._postings = {}
            self._idf = {}
            self._avgdl = 0.0
            self._dirty = False
            return

        occurrences: dict[str, list[int]] = defaultdict(list)
        frequencies: dict[str, list[int]] = defaultdict(list)
        document_frequency: dict[str, int] = defaultdict(int)

        for position, freqs in enumerate(self._term_frequencies):
            for term, count in freqs.items():
                occurrences[term].append(position)
                frequencies[term].append(count)
                document_frequency[term] += 1

        self._postings = {
            term: _Posting(
                chunk_indices=np.asarray(occurrences[term], dtype=np.int64),
                term_frequencies=np.asarray(frequencies[term], dtype=np.float64),
            )
            for term in occurrences
        }
        self._idf = self._compute_idf(document_frequency, corpus_size)
        self._avgdl = float(np.mean(self._doc_lengths)) if self._doc_lengths else 0.0
        self._dirty = False

    def _compute_idf(self, document_frequency: dict[str, int], corpus_size: int) -> dict[str, float]:
        raw = {
            term: math.log((corpus_size - df + 0.5) / (df + 0.5))
            for term, df in document_frequency.items()
        }
        if not raw:
            return {}
        average_idf = sum(raw.values()) / len(raw)
        floor = self.params.epsilon * average_idf
        # The floor is only meaningful when the average is positive; otherwise
        # clamping would push common terms below the rare ones.
        if floor <= 0:
            floor = self.params.epsilon
        return {term: idf if idf > 0 else floor for term, idf in raw.items()}

    def idf(self, term: str) -> float:
        return self._idf.get(term, 0.0)

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """Return up to ``top_k`` ``(chunk_index, score)`` pairs, best first."""
        self.finalize()
        if not self._term_frequencies or top_k <= 0:
            return []

        scores = np.zeros(len(self._term_frequencies), dtype=np.float64)
        scored_any = False
        k1, b = self.params.k1, self.params.b
        doc_lengths = np.asarray(self._doc_lengths, dtype=np.float64)
        avgdl = self._avgdl or 1.0

        seen: set[str] = set()
        for term in tokenize(query):
            if term in seen:
                # A repeated query term must not double-count; BM25 treats the
                # query as a set of terms weighted by their IDF.
                continue
            seen.add(term)
            posting = self._postings.get(term)
            if posting is None:
                continue
            idf = self._idf[term]
            tf = posting.term_frequencies
            dl = doc_lengths[posting.chunk_indices]
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            # denominator can only be zero if tf and k1 are both zero, which the
            # parameter validation rules out; guard anyway against empty chunks.
            contribution = np.divide(
                idf * (tf * (k1 + 1.0)),
                denominator,
                out=np.zeros_like(tf),
                where=denominator != 0,
            )
            scores[posting.chunk_indices] += contribution
            scored_any = True

        if not scored_any:
            return []
        return _top_k(scores, top_k)

    def score_all(self, query: str) -> np.ndarray:
        """Dense score vector over every chunk. Used by fusion and debugging."""
        self.finalize()
        scores = np.zeros(len(self._term_frequencies), dtype=np.float64)
        doc_lengths = np.asarray(self._doc_lengths, dtype=np.float64)
        avgdl = self._avgdl or 1.0
        k1, b = self.params.k1, self.params.b
        # dict.fromkeys, not set(): set iteration order depends on
        # PYTHONHASHSEED, and because floating-point addition is not associative
        # the accumulated score changed in its last bits between processes. That
        # broke the reproducibility guarantee and let score_all() disagree with
        # search() by an ulp on the same query.
        for term in dict.fromkeys(tokenize(query)):
            posting = self._postings.get(term)
            if posting is None:
                continue
            tf = posting.term_frequencies
            dl = doc_lengths[posting.chunk_indices]
            denominator = tf + k1 * (1.0 - b + b * (dl / avgdl))
            contribution = np.divide(
                self._idf[term] * (tf * (k1 + 1.0)),
                denominator,
                out=np.zeros_like(tf),
                where=denominator != 0,
            )
            scores[posting.chunk_indices] += contribution
        return scores


def _top_k(scores: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    """Highest-scoring chunks, ties broken by lowest index. See index/selection."""
    return [(int(i), float(scores[i])) for i in top_k_indices(scores, top_k)]
