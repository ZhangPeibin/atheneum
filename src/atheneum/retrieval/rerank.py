"""Re-ranking: a second, more expensive pass over a small candidate set.

Fusion already produces a good ordering. Re-ranking is opt-in because it costs
either an extra model call or a cross-encoder pass, and on a small local corpus
the gain is often not worth the latency.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from atheneum.text.tokenizer import tokenize

__all__ = ["CrossEncoderReranker", "LexicalOverlapReranker", "Reranker", "build_reranker"]


class Reranker(Protocol):
    name: str

    def rerank(self, query: str, documents: Sequence[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        """Return ``(key, score)`` pairs, best first, at most ``top_k`` long."""
        ...


@dataclass(slots=True)
class LexicalOverlapReranker:
    """Cheap deterministic re-ranker based on query-term coverage.

    Fusion ranks by how well a chunk *matches* the query; this ranks by how much
    of the query a chunk *covers*. Coverage is the property that matters when a
    multi-part question is answered across several chunks, and it is free to
    compute.
    """

    name: str = "overlap"
    coverage_weight: float = 2.0
    density_weight: float = 1.0

    def rerank(self, query: str, documents: Sequence[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        query_terms = set(tokenize(query))
        if not query_terms:
            return [(key, 0.0) for key, _ in documents[:top_k]]

        scored: list[tuple[str, float]] = []
        for key, text in documents:
            terms = set(tokenize(text))
            if not terms:
                scored.append((key, 0.0))
                continue
            overlap = query_terms & terms
            coverage = len(overlap) / len(query_terms)
            # Density penalises a long chunk that happens to contain the words
            # but buries them in unrelated text.
            density = len(overlap) / len(terms)
            scored.append(
                (key, self.coverage_weight * coverage + self.density_weight * density)
            )

        scored.sort(key=lambda kv: (-kv[1], kv[0]))
        return scored[:top_k]


_HEADING_RE = re.compile(r"^\s*(#{1,6}\s|[-=]{3,}\s*$)")
_DIGIT_RE = re.compile(r"\d")


@dataclass
class CrossEncoderReranker:
    """Neural cross-encoder reranker, backed by sentence-transformers.

    Lazy-loaded: importing atheneum must never pull in torch, so the model is
    only constructed on first use and the failure to install the optional
    dependency surfaces where it is actually relevant.
    """

    name: str = "cross-encoder"
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _model: Any = None

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder  # type: ignore[import-not-found]
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise RuntimeError(
                    "CrossEncoderReranker needs `pip install sentence-transformers`"
                ) from exc

            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: Sequence[tuple[str, str]], top_k: int) -> list[tuple[str, float]]:
        if not documents:
            return []
        model = self._ensure_model()
        pairs = [(query, text) for _, text in documents]
        scores: Any = model.predict(pairs)
        ranked = sorted(
            ((key, float(score)) for (key, _), score in zip(documents, scores, strict=True)),
            key=lambda kv: (-kv[1], kv[0]),
        )
        return ranked[:top_k]


def build_reranker(spec: str | dict[str, Any] | None) -> Reranker | None:
    """Resolve a reranker, or ``None`` to skip re-ranking entirely."""
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = {"kind": spec}
    kind = str(spec.get("kind", "")).lower()
    if kind in {"", "none", "off", "false"}:
        return None
    if kind in {"overlap", "lexical"}:
        return LexicalOverlapReranker(
            coverage_weight=float(spec.get("coverage_weight", 2.0)),
            density_weight=float(spec.get("density_weight", 1.0)),
        )
    if kind in {"cross-encoder", "crossencoder", "neural"}:
        return CrossEncoderReranker(model_name=str(spec.get("model", CrossEncoderReranker.model_name)))
    raise ValueError(f"unknown reranker kind {kind!r}; expected none, overlap, or cross-encoder")
