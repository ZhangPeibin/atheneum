"""The retrieval pipeline: ingest, index, and search one local corpus.

``Corpus`` is the single entry point most users need. It owns the SQLite store,
keeps the in-memory BM25 and vector indexes aligned with it, and exposes one
``search`` whose mode selects lexical, dense, or fused retrieval.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from atheneum.core.types import Chunk, Document
from atheneum.index.bm25 import BM25Index, BM25Params
from atheneum.index.store import Store
from atheneum.index.vectors import VectorIndex
from atheneum.retrieval.embedders import Embedder, HashingEmbedder, describe
from atheneum.retrieval.fusion import RankedItem, RRFFusion, build_fusion
from atheneum.retrieval.rerank import Reranker, build_reranker
from atheneum.text.splitter import SplitterConfig, split_document
from atheneum.text.tokenizer import token_frequencies, tokenize

logger = logging.getLogger("atheneum.corpus")

__all__ = ["Corpus", "CorpusConfig", "EmbedderMismatchError", "SearchResult"]

SearchMode = Literal["hybrid", "lexical", "vector"]
VALID_MODES = ("hybrid", "lexical", "vector")

DEFAULT_DB = "atheneum.db"
_EMBEDDER_KEY = "embedder"


@dataclass(slots=True)
class CorpusConfig:
    splitter: SplitterConfig = field(default_factory=SplitterConfig)
    bm25: BM25Params = field(default_factory=BM25Params)
    fusion: str = "rrf"
    fusion_k: int = 61
    # Measured on the bundled evaluation set (`python -m atheneum.evaluate`):
    # equal weights score MRR 0.802 against 0.833 for lexical alone, because the
    # default hashed embedder is a weak second opinion and RRF gives it equal say.
    # Weighting it down to 0.3 recovers MRR 0.837 at identical recall. Raise the
    # vector weight when using a neural embedder, whose signal is far stronger.
    fusion_weights: dict[str, float] = field(
        default_factory=lambda: {"bm25": 0.7, "vector": 0.3}
    )
    reranker: str | dict[str, Any] | None = None
    # Candidates drawn from each retriever before fusion. Oversampling then
    # re-ordering is what makes fusion worth doing at all.
    candidate_multiplier: int = 4
    min_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_size": self.splitter.chunk_size,
            "chunk_overlap": self.splitter.chunk_overlap,
            "k1": self.bm25.k1,
            "b": self.bm25.b,
            "fusion": self.fusion,
            "fusion_k": self.fusion_k,
            "fusion_weights": dict(self.fusion_weights),
            "reranker": self.reranker,
            "candidate_multiplier": self.candidate_multiplier,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


@dataclass(frozen=True, slots=True)
class SearchResult:
    """One fused hit plus everything needed to cite and explain it."""

    chunk: Chunk
    score: float
    contributions: dict[str, float] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def text(self) -> str:
        return self.chunk.text

    def citation(self, index: int) -> str:
        return f"[{index}] {self.chunk.source}#chunk{self.chunk.ordinal}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk.id,
            "doc_id": self.chunk.doc_id,
            "source": self.chunk.source,
            "ordinal": self.chunk.ordinal,
            "score": round(self.score, 8),
            "contributions": {k: round(v, 8) for k, v in self.contributions.items()},
            "text": self.chunk.text,
        }


class EmbedderMismatchError(RuntimeError):
    """The corpus was indexed by a different embedding model than the active one."""


class Corpus:
    """A locally indexed body of documents."""

    def __init__(
        self,
        store: Store,
        embedder: Embedder | None = None,
        config: CorpusConfig | None = None,
    ) -> None:
        self.store = store
        self.embedder: Embedder = embedder or HashingEmbedder()
        self.config = config or CorpusConfig()
        self._bm25 = BM25Index(self.config.bm25)
        self._vectors = VectorIndex()
        # Row order of both in-memory indexes. Search results are produced by
        # retrievers as row numbers, so this list is the join key back to chunks
        # and must stay append-only and in positional order.
        self._rows: list[Chunk] = []
        self._loaded = False
        # Revision numbers as of the last load. Two counters, so an external
        # writer that only appended can be folded in cheaply while one that
        # deleted rows forces a full reload.
        self._loaded_append = -1
        self._loaded_structure = -1
        self._guard_embedder_mismatch()

    # -- construction -------------------------------------------------------
    @classmethod
    def open(
        cls,
        path: str | os.PathLike[str] = DEFAULT_DB,
        *,
        embedder: Embedder | None = None,
        config: CorpusConfig | None = None,
    ) -> Corpus:
        return cls(Store(Path(path)), embedder=embedder, config=config)

    @classmethod
    def in_memory(
        cls, *, embedder: Embedder | None = None, config: CorpusConfig | None = None
    ) -> Corpus:
        return cls(Store(":memory:"), embedder=embedder, config=config)

    def _guard_embedder_mismatch(self) -> None:
        """Refuse to mix vectors from two different embedding models.

        Mismatched dimensions fail loudly on their own. Two models that happen
        to share a dimension silently produce nonsense ranking, which is far
        worse because nothing reports it — hence recording the embedder that
        built the index and comparing on open.
        """
        recorded = self.store.get_meta(_EMBEDDER_KEY)
        current = describe(self.embedder)
        if recorded is None:
            self.store.set_meta(_EMBEDDER_KEY, current)
        elif recorded != current:
            raise EmbedderMismatchError(
                f"this corpus was indexed with {recorded} but the active embedder is {current}. "
                "Re-index with `ath index --rebuild`, or configure the original embedder."
            )

    # -- ingestion ----------------------------------------------------------
    def add_document(self, document: Document) -> int:
        return self.add_documents([document])

    def add_text(self, source: str, text: str, *, title: str | None = None, **metadata: Any) -> int:
        return self.add_document(
            Document(source=source, content=text, title=title or source, metadata=metadata)
        )

    def add_documents(self, documents: Iterable[Document], *, batch_size: int = 64) -> int:
        """Chunk, tokenize, embed and store documents. Returns chunks added."""
        total = 0
        batch: list[Document] = []
        for document in documents:
            batch.append(document)
            if len(batch) >= batch_size:
                total += self._index_batch(batch)
                batch = []
        if batch:
            total += self._index_batch(batch)
        return total

    def _index_batch(self, documents: Sequence[Document]) -> int:
        added = 0
        for document in documents:
            chunks = split_document(document, self.config.splitter)
            # Embedding happens outside the transaction: it can be slow and, for a
            # networked embedder, can fail, and neither should hold a write lock.
            embeddings = self._embed([c.text for c in chunks]) if chunks else []
            frequencies = [token_frequencies(tokenize(c.text)) for c in chunks]

            # Document row and chunk rows commit together. Splitting them left an
            # orphan document with zero chunks whenever embedding failed, and
            # because put_document then reported "already present" on every retry,
            # that document could never be indexed at all.
            with self.store.transaction():
                if not self.store._put_document_row(document):
                    logger.debug("document %s already indexed", document.source)
                    continue
                if not chunks:
                    logger.warning("document %s produced no chunks", document.source)
                    continue
                self.store._put_chunk_rows(chunks, frequencies, embeddings)
            added += len(chunks)
        return added

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        matrix = self.embedder.embed_many(texts)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError(
                f"embedder {getattr(self.embedder, 'name', '?')} returned shape "
                f"{matrix.shape} for {len(texts)} inputs"
            )
        if self._vectors.dim is not None and matrix.shape[1] != self._vectors.dim:
            raise ValueError(
                f"embedder produced {matrix.shape[1]}-d vectors but the index holds "
                f"{self._vectors.dim}-d vectors"
            )
        return [row.astype(float).tolist() for row in matrix]

    def add_path(self, path: str | os.PathLike[str], *, title: str | None = None) -> int:
        from atheneum.ingest import read_file

        return self.add_document(read_file(Path(path), title=title))

    def add_paths(
        self,
        paths: Iterable[str | os.PathLike[str]],
        *,
        patterns: Sequence[str] = ("*",),
        exclude: Sequence[str] = (),
        limit: int | None = None,
    ) -> int:
        """Index every readable file under ``paths``.

        A file that cannot be read is logged and skipped rather than aborting the
        walk, because one binary blob in a tree should not fail the ingest.
        """
        from atheneum.ingest import discover_files, read_file

        documents: list[Document] = []
        for root in paths:
            for found in discover_files(Path(root), patterns=patterns, exclude=exclude):
                # Checked before reading, not after: `limit=0` used to index one
                # whole file because the test happened once the append was done.
                if limit is not None and len(documents) >= limit:
                    break
                try:
                    documents.append(read_file(found))
                except Exception as exc:
                    logger.warning("skipping %s: %s", found, exc)
            if limit is not None and len(documents) >= limit:
                break
        return self.add_documents(documents) if documents else 0

    def delete_document(self, doc_id: str) -> int:
        # No local bookkeeping needed: the delete bumped structure_revision, and
        # _ensure_ready reloads when it sees that change.
        return self.store.delete_document(doc_id)

    # -- index maintenance --------------------------------------------------
    def _ensure_ready(self) -> None:
        """Bring the in-memory indexes up to date with SQLite.

        Freshness is decided by the store's revision counters rather than by a
        flag this instance set itself, so a second Corpus writing to the same
        file is detected. Counting rows was not enough: a concurrent delete could
        leave the count unchanged while the contents differed, and the positional
        alignment between the BM25 index, the vector matrix and `_rows` would
        silently attribute results to the wrong chunks.
        """
        structure = self.store.structure_revision
        append = self.store.append_revision
        if not self._loaded or structure != self._loaded_structure:
            self._rebuild_rows()
        elif append != self._loaded_append:
            self._extend_rows()
        self._loaded_append = append
        self._loaded_structure = structure

    def _rebuild_rows(self) -> None:
        """Load the whole corpus from SQLite into both in-memory indexes."""
        self._bm25 = BM25Index(self.config.bm25)
        self._vectors = VectorIndex()
        self._rows = []

        frequencies: list[dict[str, int]] = []
        blobs: list[bytes] = []
        dim = int(getattr(self.embedder, "dim", 0)) or 0
        missing = 0
        for row in self.store.iter_rows():
            frequencies.append(row.term_frequencies)
            blobs.append(row.embedding or b"")
            self._rows.append(row.chunk)
            if not row.embedding:
                missing += 1

        self._bm25.load(frequencies)
        if blobs and dim:
            zero = [0.0] * dim
            vectors = [_blob_to_list(blob, dim) if blob else zero for blob in blobs]
            self._vectors.load(vectors, dim=dim)
        if missing:
            logger.warning(
                "%d of %d chunks have no stored vector and were given a zero "
                "vector; they are invisible to dense retrieval",
                missing,
                len(blobs),
            )
        self._loaded = True

    def _extend_rows(self) -> None:
        """Fold in chunks stored since the last load.

        Appending is correct because chunk positions are monotonic, so rows
        already in memory keep their index in both in-memory structures.
        """
        already_loaded = len(self._rows)
        dim = int(self._vectors.dim or getattr(self.embedder, "dim", 0) or 0)
        for index, row in enumerate(self.store.iter_rows()):
            if index < already_loaded:
                continue
            self._bm25.add_frequencies(row.term_frequencies)
            if dim:
                if row.embedding:
                    self._vectors.add(_blob_to_list(row.embedding, dim))
                else:
                    # A zero vector, not a skip. Skipping desynchronized the
                    # matrix from _rows by one slot, so every later vector row
                    # was attributed to the wrong chunk: an incremental search
                    # returned v4.md where a fresh rebuild returned v5.md at the
                    # identical score. The three structures must advance together.
                    self._vectors.add([0.0] * dim)
                    logger.warning(
                        "chunk %s has no stored vector; it is invisible to dense retrieval",
                        row.chunk.id,
                    )
            self._rows.append(row.chunk)
        self._bm25.finalize()

    def invalidate(self) -> None:
        """Force indexes to reload from SQLite on the next search."""
        self._loaded = False
        self._loaded_append = -1
        self._loaded_structure = -1

    def configure(self, **overrides: Any) -> CorpusConfig:
        """Update retrieval parameters on a live corpus.

        Chunking settings are not included: they were applied at ingest time and
        changing them requires `rebuild()`, which re-splits from stored documents.
        """
        cfg = self.config
        for key, value in overrides.items():
            if not hasattr(cfg, key):
                raise AttributeError(f"CorpusConfig has no field {key!r}")
            if key == "splitter":
                raise AttributeError(
                    "splitter changes require rebuild(); stored chunks were already cut "
                    "under the previous configuration"
                )
            setattr(cfg, key, value)
        # BM25 parameters are baked into the built index, so force a rebuild of it.
        self._bm25 = BM25Index(cfg.bm25)
        self._loaded = False
        self._pending = False
        return cfg

    def rebuild(self) -> dict[str, Any]:
        """Re-chunk and re-embed every stored document."""
        documents = [
            doc
            for doc_id in self.store.get_document_ids()
            if (doc := self.store.get_document(doc_id)) is not None
        ]
        # keep_documents=True: the documents are the raw material for the new
        # chunks. Clearing them first meant that any failure part-way through
        # re-embedding destroyed data permanently -- a two-document corpus came
        # back with one document and zero chunks after a failed rebuild.
        # Re-chunk the documents that are still stored. Going through
        # add_documents would skip every one of them as "already indexed" and
        # leave the corpus with documents but no chunks.
        prepared: list[tuple[list[Chunk], list[dict[str, int]], list[list[float]]]] = []
        for document in documents:
            chunks = split_document(document, self.config.splitter)
            if not chunks:
                continue
            prepared.append(
                (
                    chunks,
                    [token_frequencies(tokenize(c.text)) for c in chunks],
                    self._embed([c.text for c in chunks]),
                )
            )

        # All embedding work is finished before anything is deleted, and the wipe
        # and the re-insert then commit as one transaction. An embedder failure
        # part-way through used to leave the corpus cleared with a handful of
        # orphan documents at zero chunks and no way to tell it had happened.
        # Holding every embedding in memory is the price; rebuild is an explicit
        # maintenance command over a corpus that already fits in `_rows`.
        with self.store.transaction():
            self.store._clear_index_rows(keep_documents=True)
            for chunks, frequencies, embeddings in prepared:
                self.store._put_chunk_rows(chunks, frequencies, embeddings)

        self.invalidate()
        self._rows = []
        self._ensure_ready()
        return self.stats()

    # -- search -------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 10,
        *,
        mode: SearchMode = "hybrid",
        source_filter: str | None = None,
        candidate_multiplier: int | None = None,
    ) -> list[SearchResult]:
        """Retrieve the chunks most relevant to ``query``.

        ``hybrid`` runs both retrievers and fuses the two ranked lists. That is
        the default because the two signals fail in opposite ways: BM25 misses
        paraphrase, dense search misses rare exact identifiers, and fusion keeps
        whichever one fired.
        """
        if top_k <= 0 or not query.strip():
            return []
        if mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}, got {mode!r}")

        self._ensure_ready()
        if not self._rows:
            return []

        pool = max(top_k, top_k * (candidate_multiplier or self.config.candidate_multiplier))

        ranked_lists: dict[str, list[tuple[str, Chunk]]] = {}
        raw_scores: dict[str, list[float]] = {}

        if mode in ("hybrid", "lexical"):
            hits = self._lexical(query, pool)
            if hits:
                ranked_lists["bm25"] = [(c.id, c) for c, _ in hits]
                raw_scores["bm25"] = [s for _, s in hits]
        if mode in ("hybrid", "vector"):
            hits = self._vector(query, pool)
            if hits:
                ranked_lists["vector"] = [(c.id, c) for c, _ in hits]
                raw_scores["vector"] = [s for _, s in hits]

        if not ranked_lists:
            return []

        weights = [self.config.fusion_weights.get(name, 1.0) for name in ranked_lists]
        fused = self._fuse(ranked_lists, raw_scores, weights)

        reranker = build_reranker(self.config.reranker) if self.config.reranker else None
        if reranker is not None:
            fused = self._rerank(query, fused, reranker, top_k)

        results = [
            SearchResult(chunk=item.value, score=item.score, contributions=item.contributions)
            for item in fused
        ]
        if source_filter:
            results = [r for r in results if source_filter in r.chunk.source]
        if self.config.min_score > 0:
            results = [r for r in results if r.score >= self.config.min_score]
        return results[:top_k]

    def _fuse(
        self,
        ranked_lists: dict[str, list[tuple[str, Chunk]]],
        raw_scores: dict[str, list[float]],
        weights: Sequence[float],
    ) -> list[RankedItem[Chunk]]:
        if self.config.fusion == "rrf":
            return RRFFusion(k=self.config.fusion_k).fuse(ranked_lists, weights)
        strategy = build_fusion(self.config.fusion, k=self.config.fusion_k)
        fused = strategy.fuse(ranked_lists, weights, scores=raw_scores)  # type: ignore[call-arg]
        return list(fused)

    def _rerank(
        self, query: str, fused: list[RankedItem[Chunk]], reranker: Reranker, top_k: int
    ) -> list[RankedItem[Chunk]]:
        # The expensive stage sees only the fused head; that is the whole point
        # of a two-stage retrieve-then-rerank design.
        window = fused[: max(top_k * self.config.candidate_multiplier, top_k)]
        if not window:
            return fused
        pairs = [(item.key, item.value.text) for item in window]
        rescored = reranker.rerank(query, pairs, len(window))
        by_key = {item.key: item for item in window}
        rebuilt: list[RankedItem[Chunk]] = []
        for key, score in rescored:
            original = by_key.get(key)
            if original is None:
                continue
            rebuilt.append(
                RankedItem(
                    key=key,
                    value=original.value,
                    score=score,
                    contributions={**original.contributions, f"rerank:{reranker.name}": score},
                )
            )
        # An empty rerank result means the reranker saw nothing, not that the
        # corpus is empty; keep the fusion order rather than returning nothing.
        return rebuilt or fused

    def _lexical(self, query: str, pool: int) -> list[tuple[Chunk, float]]:
        return [(self._rows[i], s) for i, s in self._bm25.search(query, top_k=pool) if i < len(self._rows)]

    def _vector(self, query: str, pool: int) -> list[tuple[Chunk, float]]:
        if self._vectors.dim is None or len(self._vectors) == 0:
            return []
        try:
            vector = self.embedder.embed(query)
        except Exception as exc:
            # Dense search is an enhancement over lexical; an embedder outage
            # should lower answer quality, not fail the request outright.
            logger.warning("embedder failed (%s); continuing with lexical-only results", exc)
            return []
        return [
            (self._rows[row], s) for row, s in self._vectors.search(vector, top_k=pool) if row < len(self._rows)
        ]

    def lexical_scores(self, query: str) -> dict[str, float]:
        """Dense BM25 score per chunk id. Used by the evaluation harness."""
        self._ensure_ready()
        dense = self._bm25.score_all(query)
        return {
            self._rows[i].id: float(score)
            for i, score in enumerate(dense)
            if score > 0 and i < len(self._rows)
        }

    # -- accessors ----------------------------------------------------------
    def get_chunk(self, chunk_id: str) -> Chunk | None:
        return self.store.get_chunk_by_id(chunk_id)

    def get_chunks(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        return self.store.get_chunks_by_id(chunk_ids)

    def chunks(self) -> list[Chunk]:
        self._ensure_ready()
        return list(self._rows)

    def stats(self) -> dict[str, Any]:
        info = self.store.stats().as_dict()
        info["embedder"] = json.loads(describe(self.embedder))
        info["config"] = self.config.to_dict()
        return info

    def sources(self, limit: int = 200) -> list[dict[str, Any]]:
        return self.store.list_documents(limit=limit)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> Corpus:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _blob_to_list(blob: bytes, dim: int) -> list[float]:
    import numpy as np

    array = np.frombuffer(blob, dtype="<f4")
    if dim and array.size != dim:
        # A row written by a different-dimension embedder cannot be used; pad or
        # truncate so one bad row does not fail the entire load.
        fixed = np.zeros(dim, dtype=np.float32)
        n = min(dim, array.size)
        fixed[:n] = array[:n]
        array = fixed
    return [float(value) for value in array.tolist()]
