"""Embedding backends.

``HashingEmbedder`` is the default and ships with the package: it is
deterministic, dependency-free, and runs offline, which is what lets the whole
product be installed, indexed, queried and tested with no network access and no
API key.

It is a hashed bag-of-terms projection, not a neural embedder. It captures term
and phrase overlap well and generalises to unseen vocabulary, but it does not
understand paraphrase the way a trained model does. Point ``--embedder`` at a
networked backend when you want true semantic matching; the retrieval pipeline
is identical either way.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, cast, runtime_checkable

import numpy as np

from atheneum.text.tokenizer import tokenize

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "OllamaEmbedder",
    "OpenAIEmbedder",
    "SentenceTransformerEmbedder",
    "build_embedder",
]


@runtime_checkable
class Embedder(Protocol):
    """A backend that turns text into a fixed-width vector.

    ``name`` is deliberately not a protocol member: implementations expose it as
    a class attribute, and requiring an instance attribute made every concrete
    embedder fail the protocol check. Callers read it with getattr, which is what
    ``describe()`` already does.
    """

    dim: int

    def embed(self, text: str) -> np.ndarray: ...

    def embed_many(self, texts: Sequence[str]) -> np.ndarray: ...


class HashingEmbedder:
    """Deterministic hashed term/phrase projection.

    Each token is hashed into one of ``dim`` buckets and each adjacent token
    pair into a second set of buckets, so word order contributes signal that a
    pure bag-of-words embedding loses. Weights are sublinear in term frequency
    and the result is L2-normalized.
    """

    name = "hashing"

    def __init__(self, dim: int = 512, *, bigram_weight: float = 0.5) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        if bigram_weight < 0:
            raise ValueError(f"bigram_weight must be non-negative, got {bigram_weight}")
        self.dim = dim
        self.bigram_weight = bigram_weight

    def embed(self, text: str) -> np.ndarray:
        return cast("np.ndarray[Any, Any]", self.embed_many([text])[0])

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            if not tokens:
                continue
            vector = out[row]
            counts: dict[int, float] = {}
            for token in tokens:
                bucket = _bucket(token, self.dim)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
            if self.bigram_weight and len(tokens) > 1:
                for first, second in itertools.pairwise(tokens):
                    # Salt the bigram so it lands in a different bucket space
                    # than its constituent unigrams.
                    bucket = _bucket(f"{first}\x1f{second}", self.dim, salt=b"\x01")
                    counts[bucket] = counts.get(bucket, 0.0) + self.bigram_weight
            for bucket, count in counts.items():
                # Sublinear scaling: a term repeated ten times is not ten times
                # as indicative of topic.
                vector[bucket] += np.float32(1.0 + np.log(np.float32(count)))
            norm = float(np.linalg.norm(vector))
            if norm > 0:
                vector /= np.float32(norm)
        return out

    def to_dict(self) -> dict[str, object]:
        return {"kind": self.name, "dim": self.dim, "bigram_weight": self.bigram_weight}


def _bucket(feature: str, dim: int, salt: bytes = b"") -> int:
    encoded = feature.encode("utf-8")
    if salt:
        digest = hashlib.blake2b(encoded, digest_size=8, salt=salt).digest()
    else:
        digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


@dataclass(slots=True)
class _HttpEmbedder:
    """Shared plumbing for embedders that call an HTTP endpoint.

    ``name`` is a ClassVar, not a field. As a field it defaulted to "http" and
    every subclass instance reported that instead of its own name, so
    ``describe()`` could not tell OpenAI from Ollama -- which silently defeated
    the index/embedder mismatch guard that depends on it.
    """

    dim: int = 0
    # repr=False: the generated dataclass repr printed the API key verbatim.
    api_key: str | None = field(default=None, repr=False)
    timeout: float = 60.0
    max_retries: int = 3

    name: ClassVar[str] = "http"

    def _client(self) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise RuntimeError(
                f"{type(self).name} needs the network extra; install with "
                "`pip install atheneum[net]`"
            ) from exc
        return httpx

    def _post_with_retry(self, url: str, headers: dict[str, str], payload: dict[str, Any]) -> dict[str, Any]:
        httpx = self._client()
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = httpx.post(url, headers=headers, json=payload, timeout=self.timeout)
            except httpx.HTTPError as exc:
                # Transport-level only. raise_for_status() used to be inside this
                # try, and HTTPStatusError subclasses HTTPError, so a permanent
                # 401 was retried three times with backoff and the real cause was
                # buried under "failed: status 401".
                last_error = exc
                if attempt < self.max_retries - 1:
                    _sleep_backoff(attempt)
                continue
            if response.status_code == 429 or response.status_code >= 500:
                last_error = RuntimeError(f"{url} returned {response.status_code}")
                if attempt < self.max_retries - 1:
                    _sleep_backoff(attempt, response.headers.get("retry-after"))
                continue
            if response.status_code >= 400:
                # Permanent: fail immediately with the status and body visible.
                raise RuntimeError(
                    f"embedding request to {url} failed with status "
                    f"{response.status_code}: {response.text[:200]}"
                )
            return cast(dict[str, Any], response.json())
        raise RuntimeError(f"embedding request to {url} failed after {self.max_retries} attempts: {last_error}") from last_error


def _ordered_embeddings(items: Any, batch: Sequence[str]) -> list[list[float]]:
    """Put embeddings back into input order using the API's index field.

    Three separate ways this used to misalign vectors against texts:
    sorting on the raw value ordered string indices lexicographically
    ("0","10","1","2"), a missing index defaulted to 0 and shifted every
    following row, and duplicates resolved by arrival order.
    """
    if not isinstance(items, list):
        raise RuntimeError(f"embedding API returned {type(items).__name__} instead of a list")

    indexed: dict[int, list[float]] = {}
    for position, item in enumerate(items):
        if not isinstance(item, dict) or "embedding" not in item:
            raise RuntimeError(f"embedding API returned a malformed item at position {position}")
        raw_index = item.get("index", position)
        try:
            index = int(raw_index)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"embedding API returned a non-integer index {raw_index!r}") from exc
        if not 0 <= index < len(batch):
            raise RuntimeError(f"embedding API returned out-of-range index {index} for a batch of {len(batch)}")
        if index in indexed:
            raise RuntimeError(f"embedding API returned index {index} twice")
        indexed[index] = item["embedding"]

    missing = [i for i in range(len(batch)) if i not in indexed]
    if missing:
        raise RuntimeError(f"embedding API omitted indices {missing[:5]} of {len(batch)}")
    return [indexed[i] for i in range(len(batch))]


def _sleep_backoff(attempt: int, retry_after: str | None = None) -> None:
    import time

    delay = min(2**attempt * 0.25, 4.0)
    if retry_after:
        with contextlib.suppress(ValueError):
            delay = max(delay, min(float(retry_after), 30.0))
    time.sleep(delay)


class OpenAIEmbedder(_HttpEmbedder):
    """OpenAI-compatible ``/v1/embeddings`` endpoint (OpenAI, Azure, vLLM, LM Studio)."""

    name = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dim: int = 1536,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openai.com/v1",
        batch_size: int = 64,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(dim=dim, api_key=api_key, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch_size = batch_size

    def embed(self, text: str) -> np.ndarray:
        return cast("np.ndarray[Any, Any]", self.embed_many([text])[0])

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            data = self._post_with_retry(
                f"{self.base_url}/embeddings",
                self._headers(),
                {"model": self.model, "input": batch},
            )
            rows.extend(_ordered_embeddings(data.get("data"), batch))
        matrix = np.asarray(rows, dtype=np.float32)
        if matrix.shape[0] != len(texts):
            # Without this the caller silently receives the wrong number of
            # vectors and every later chunk carries its neighbour's embedding --
            # the same class of silent misattribution as a desynced index, and
            # far harder to notice because nothing raises.
            raise RuntimeError(
                f"embedding API returned {matrix.shape[0]} vectors for {len(texts)} inputs"
            )
        return cast("np.ndarray[Any, Any]", matrix)

    def _headers(self) -> dict[str, str]:
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError(
                "no API key for the OpenAI embedder; set OPENAI_API_KEY or pass api_key="
            )
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}


class OllamaEmbedder(_HttpEmbedder):
    """Local Ollama ``/api/embed`` endpoint."""

    name = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dim: int = 768,
        *,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> None:
        super().__init__(dim=dim, api_key=None, timeout=timeout, max_retries=max_retries)
        self.model = model
        self.base_url = (base_url or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")

    def embed(self, text: str) -> np.ndarray:
        return cast("np.ndarray[Any, Any]", self.embed_many([text])[0])

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        data = self._post_with_retry(
            f"{self.base_url}/api/embed",
            {"Content-Type": "application/json"},
            {"model": self.model, "input": list(texts)},
        )
        return cast("np.ndarray[Any, Any]", np.asarray(data["embeddings"], dtype=np.float32))


class SentenceTransformerEmbedder:
    """Local neural embeddings via sentence-transformers, if the user installs it."""

    name = "sentence-transformers"

    def __init__(self, model: str = "all-MiniLM-L6-v2", *, device: str | None = None) -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "SentenceTransformerEmbedder needs `pip install sentence-transformers`"
            ) from exc
        self._model = SentenceTransformer(model, device=device)
        self.dim = int(self._model.get_sentence_embedding_dimension())

    def embed(self, text: str) -> np.ndarray:
        return np.asarray(self._model.encode(text, normalize_embeddings=True), dtype=np.float32)

    def embed_many(self, texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            self._model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )


def build_embedder(spec: str | dict[str, Any] | None, *, default_dim: int = 512) -> Embedder:
    """Resolve an embedder from a name, a config mapping, or None.

    Accepting a plain string keeps the CLI simple (``--embedder hashing``),
    while the mapping form lets a config file carry model names and dimensions.
    """
    if spec is None:
        return HashingEmbedder(dim=default_dim)
    if isinstance(spec, Embedder):
        return spec
    if isinstance(spec, str):
        spec = {"kind": spec}

    kind = str(spec.get("kind", "hashing")).lower()
    if kind in {"hashing", "hash", "default", "offline"}:
        return HashingEmbedder(
            dim=int(spec.get("dim", default_dim)),
            bigram_weight=float(spec.get("bigram_weight", 0.5)),
        )
    if kind in {"openai", "openai-compatible"}:
        return OpenAIEmbedder(
            model=str(spec.get("model", "text-embedding-3-small")),
            dim=int(spec.get("dim", 1536)),
            api_key=spec.get("api_key"),
            base_url=str(spec.get("base_url", "https://api.openai.com/v1")),
            batch_size=int(spec.get("batch_size", 64)),
        )
    if kind == "ollama":
        return OllamaEmbedder(
            model=str(spec.get("model", "nomic-embed-text")),
            dim=int(spec.get("dim", 768)),
            base_url=spec.get("base_url"),
        )
    if kind in {"sentence-transformers", "st", "local-neural"}:
        return SentenceTransformerEmbedder(
            model=str(spec.get("model", "all-MiniLM-L6-v2")), device=spec.get("device")
        )
    raise ValueError(
        f"unknown embedder kind {kind!r}; expected one of "
        "hashing, openai, ollama, sentence-transformers"
    )


def iter_batches(items: Iterable[str], size: int) -> Iterable[list[str]]:
    batch: list[str] = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def describe(embedder: Embedder) -> str:
    """A stable string identity, used to detect index/embedder mismatches."""
    return json.dumps(
        {"name": getattr(embedder, "name", type(embedder).__name__), "dim": embedder.dim},
        sort_keys=True,
    )
