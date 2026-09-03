"""SQLite persistence for documents, chunks, term statistics and vectors.

SQLite is the only store. There is no server to run, no schema to provision,
and the whole index is a single file that can be copied or deleted.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import zlib
from collections.abc import Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from atheneum.core.types import Chunk, Document

__all__ = ["CorpusStats", "Store", "StoredRow"]

SCHEMA_VERSION = 1

# Well under SQLite's default SQLITE_MAX_VARIABLE_NUMBER.
_SQL_VARIABLE_LIMIT = 900

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id        TEXT PRIMARY KEY,
    source    TEXT NOT NULL,
    title     TEXT,
    mime_type TEXT NOT NULL DEFAULT 'text/plain',
    content   TEXT NOT NULL,
    metadata  TEXT NOT NULL DEFAULT '{}',
    added_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
    pos      INTEGER PRIMARY KEY AUTOINCREMENT,
    id       TEXT NOT NULL UNIQUE,
    doc_id   TEXT NOT NULL,
    source   TEXT NOT NULL,
    ordinal  INTEGER NOT NULL,
    text     TEXT NOT NULL,
    metadata TEXT NOT NULL DEFAULT '{}',
    tf       BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_doc ON chunks(doc_id);

CREATE TABLE IF NOT EXISTS vectors (
    pos       INTEGER PRIMARY KEY,
    dim       INTEGER NOT NULL,
    embedding BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id         TEXT PRIMARY KEY,
    title      TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    seq        INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    content    TEXT NOT NULL DEFAULT '',
    payload    TEXT NOT NULL DEFAULT '{}',
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS messages_session ON messages(session_id, seq);
"""


@dataclass(frozen=True, slots=True)
class StoredRow:
    """A chunk row plus its positional index and serialized term frequencies."""

    pos: int
    chunk: Chunk
    term_frequencies: dict[str, int]
    embedding: bytes | None


@dataclass(frozen=True, slots=True)
class CorpusStats:
    documents: int
    chunks: int
    vectors: int
    dimension: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "chunks": self.chunks,
            "vectors": self.vectors,
            "dimension": self.dimension,
        }


def encode_term_frequencies(freqs: dict[str, int]) -> bytes:
    """Compress a term-frequency map for storage.

    zlib-wrapped JSON keeps the format inspectable with the stdlib while
    cutting the size of the repetitive term keys substantially.
    """
    payload = json.dumps(freqs, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return zlib.compress(payload.encode("utf-8"), level=6)


def decode_term_frequencies(blob: bytes) -> dict[str, int]:
    decoded = json.loads(zlib.decompress(blob).decode("utf-8"))
    return {str(term): int(count) for term, count in decoded.items()}


class Store:
    """SQLite store, safe to share across threads but not concurrent writers.

    ``check_same_thread`` is disabled because WSGI/ASGI servers dispatch
    synchronous handlers onto a thread pool, and a connection created at startup
    would otherwise be rejected on first use. Correctness is preserved by a
    reentrant lock around every connection use: SQLite cannot multiplex writes on
    one connection, so access is serialized rather than shared.

    WAL mode still allows a separate reader connection to proceed during a write,
    which is what makes the HTTP API responsive while the CLI is ingesting.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.path),
            detect_types=0,
            isolation_level=None,  # explicit transactions via begin()/commit()
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._configure()
        self._migrate()

    def _configure(self) -> None:
        conn = self._conn
        with self._lock:
            conn.execute("PRAGMA foreign_keys = ON")
            # WAL lets a reader proceed while the indexer writes, which is what
            # makes `ath serve` usable during a long `ath ingest`.
            if str(self.path) != ":memory:":
                conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")

    def _migrate(self) -> None:
        # executescript commits implicitly before it runs, so it must not be
        # wrapped in an explicit transaction.
        with self._lock:
            self._conn.executescript(_SCHEMA)
        existing = self._get_meta("schema_version")
        if existing is None:
            with self.transaction():
                self._set_meta("schema_version", str(SCHEMA_VERSION))
        elif int(existing) > SCHEMA_VERSION:
            raise RuntimeError(
                f"index at {self.path} was written by a newer atheneum "
                f"(schema {existing}); this build supports up to {SCHEMA_VERSION}"
            )

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._conn
        # The lock is reentrant, so nested helpers such as put_document calling
        # set_meta still work without deadlocking.
        with self._lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except Exception:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    @contextmanager
    def cursor(self) -> Iterator[sqlite3.Cursor]:
        with self._lock, closing(self._conn.cursor()) as cur:
            yield cur

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # -- meta ---------------------------------------------------------------
    def _get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else None

    def _set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def get_meta(self, key: str) -> str | None:
        return self._get_meta(key)

    # -- revision counters --------------------------------------------------
    # Two counters rather than one, so an in-memory index can tell "rows were
    # appended" (cheap incremental extend) from "rows were deleted" (the
    # positional alignment is gone and a full reload is required). Collapsing
    # them into one counter would force a full reload after every write, or --
    # worse -- let a delete be mistaken for an append and silently misattribute
    # search results to the wrong chunks.
    def _bump(self, key: str) -> None:
        self._set_meta(key, str(int(self._get_meta(key) or 0) + 1))

    @property
    def append_revision(self) -> int:
        return int(self._get_meta("append_revision") or 0)

    @property
    def structure_revision(self) -> int:
        return int(self._get_meta("structure_revision") or 0)

    def revisions(self) -> tuple[int, int]:
        """Both counters read as one snapshot: (structure, append).

        Reading them separately is not safe: a delete plus an append committing
        between the two queries looks like "appended only", which sends an
        incremental loader down the wrong path.
        """
        with self._lock:
            rows = self._conn.execute(
                "SELECT key, value FROM meta WHERE key IN ('structure_revision', 'append_revision')"
            ).fetchall()
        values = {row["key"]: int(row["value"]) for row in rows}
        return values.get("structure_revision", 0), values.get("append_revision", 0)

    def set_meta(self, key: str, value: str) -> None:
        with self.transaction():
            self._set_meta(key, value)

    # -- documents ----------------------------------------------------------
    def put_document(self, document: Document) -> bool:
        """Insert a document. Returns False if an identical one already exists.

        The existence check happens inside the transaction: doing it outside let
        two threads both see "absent" and both insert.
        """
        with self.transaction():
            return self._put_document_row(document)

    def _put_document_row(self, document: Document) -> bool:
        """Insert a document row. Assumes a transaction is already open."""
        doc_id = document.id
        if self._conn.execute("SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone():
            return False
        self._conn.execute(
            "INSERT INTO documents(id, source, title, mime_type, content, metadata, added_at) "
            "VALUES(?, ?, ?, ?, ?, ?, ?)",
            (
                doc_id,
                document.source,
                document.title,
                document.mime_type,
                document.content,
                json.dumps(document.metadata, ensure_ascii=False, sort_keys=True),
                time.time(),
            ),
        )
        return True

    def get_document(self, doc_id: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT source, title, mime_type, content, metadata FROM documents WHERE id = ?",
                (doc_id,),
            ).fetchone()
        if row is None:
            return None
        return Document(
            source=row["source"],
            content=row["content"],
            title=row["title"],
            mime_type=row["mime_type"],
            metadata=json.loads(row["metadata"]),
        )

    def find_document_by_source(self, source: str) -> Document | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM documents WHERE source = ? ORDER BY added_at DESC LIMIT 1",
                (source,),
            ).fetchone()
        return self.get_document(row["id"]) if row else None

    def list_documents(self, limit: int = 100) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
            "SELECT d.id, d.source, d.title, d.added_at, "
            "       (SELECT COUNT(*) FROM chunks c WHERE c.doc_id = d.id) AS chunk_count "
                "FROM documents d ORDER BY d.added_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_document(self, doc_id: str) -> int:
        """Delete a document and its chunks. Returns the number of chunks removed."""
        with self.transaction():
            count = self._conn.execute(
                "SELECT COUNT(*) AS n FROM chunks WHERE doc_id = ?", (doc_id,)
            ).fetchone()["n"]
            self._conn.execute(
                "DELETE FROM vectors WHERE pos IN (SELECT pos FROM chunks WHERE doc_id = ?)",
                (doc_id,),
            )
            self._conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            self._conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            if count:
                self._bump("structure_revision")
        return int(count)

    # -- chunks + vectors ---------------------------------------------------
    def put_chunks(
        self,
        chunks: Sequence[Chunk],
        term_frequencies: Sequence[dict[str, int]],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[int]:
        """Persist chunks with their term frequencies and optional vectors.

        Returns the positional indices assigned, in input order. Positions are
        the join key between the BM25 index and the vector matrix, so both must
        be built in this same order.
        """
        if len(chunks) != len(term_frequencies):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(term_frequencies)} term-frequency maps"
            )
        if embeddings is not None and len(embeddings) != len(chunks):
            raise ValueError(
                f"got {len(chunks)} chunks but {len(embeddings)} embeddings"
            )

        with self.transaction():
            return self._put_chunk_rows(chunks, term_frequencies, embeddings)

    def _put_chunk_rows(
        self,
        chunks: Sequence[Chunk],
        term_frequencies: Sequence[dict[str, int]],
        embeddings: Sequence[Sequence[float]] | None = None,
    ) -> list[int]:
        """Insert chunk rows. Assumes a transaction is already open."""
        positions: list[int] = []
        inserted = False
        for index, (chunk, freqs) in enumerate(zip(chunks, term_frequencies, strict=True)):
            duplicate = self._conn.execute(
                "SELECT pos FROM chunks WHERE id = ?", (chunk.id,)
            ).fetchone()
            if duplicate is not None:
                positions.append(int(duplicate["pos"]))
                continue
            cur = self._conn.execute(
                "INSERT INTO chunks(id, doc_id, source, ordinal, text, metadata, tf) "
                "VALUES(?, ?, ?, ?, ?, ?, ?)",
                (
                    chunk.id,
                    chunk.doc_id,
                    chunk.source,
                    chunk.ordinal,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                    encode_term_frequencies(freqs),
                ),
            )
            pos = int(cur.lastrowid or 0)
            positions.append(pos)
            inserted = True
            if embeddings is not None:
                vector = embeddings[index]
                array = _as_float32_bytes(vector)
                self._conn.execute(
                    "INSERT INTO vectors(pos, dim, embedding) VALUES(?, ?, ?) "
                    "ON CONFLICT(pos) DO UPDATE SET dim = excluded.dim, "
                    "embedding = excluded.embedding",
                    (pos, len(vector), array),
                )
        if inserted:
            self._bump("append_revision")
        return positions

    def chunk_count(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])

    def iter_rows(self, batch_size: int = 500) -> list[StoredRow]:
        """Every chunk in positional order.

        Rows are materialised under the lock and yielded outside it. Holding the
        lock across a yield let a slow consumer block every writer for as long as
        it liked, which is the opposite of what WAL mode is for.
        """
        with self.cursor() as cur:
            cur.execute(
                "SELECT c.pos, c.id, c.doc_id, c.source, c.ordinal, c.text, c.metadata, c.tf, "
                "       v.dim AS vdim, v.embedding AS vemb "
                "FROM chunks c LEFT JOIN vectors v ON v.pos = c.pos "
                "ORDER BY c.pos"
            )
            fetched = cur.fetchall()
        return [
            StoredRow(
                pos=int(row["pos"]),
                chunk=Chunk(
                    id=row["id"],
                    doc_id=row["doc_id"],
                    source=row["source"],
                    ordinal=int(row["ordinal"]),
                    text=row["text"],
                    metadata=json.loads(row["metadata"]),
                ),
                term_frequencies=decode_term_frequencies(row["tf"]),
                embedding=row["vemb"],
            )
            for row in fetched
        ]

    def get_chunk_by_id(self, chunk_id: str) -> Chunk | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, doc_id, source, ordinal, text, metadata FROM chunks WHERE id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            return None
        return Chunk(
            id=row["id"],
            doc_id=row["doc_id"],
            source=row["source"],
            ordinal=int(row["ordinal"]),
            text=row["text"],
            metadata=json.loads(row["metadata"]),
        )

    def get_chunks_by_id(self, chunk_ids: Sequence[str]) -> list[Chunk]:
        if not chunk_ids:
            return []
        # Batched: a single IN list above SQLite's bound-variable limit raises
        # "too many SQL variables" rather than degrading gracefully.
        rows: list[sqlite3.Row] = []
        with self._lock:
            for start in range(0, len(chunk_ids), _SQL_VARIABLE_LIMIT):
                batch = list(chunk_ids[start : start + _SQL_VARIABLE_LIMIT])
                placeholders = ",".join("?" for _ in batch)
                rows.extend(
                    self._conn.execute(
                        "SELECT id, doc_id, source, ordinal, text, metadata FROM chunks "
                        f"WHERE id IN ({placeholders})",
                        tuple(batch),
                    ).fetchall()
                )
        by_id = {
            row["id"]: Chunk(
                id=row["id"],
                doc_id=row["doc_id"],
                source=row["source"],
                ordinal=int(row["ordinal"]),
                text=row["text"],
                metadata=json.loads(row["metadata"]),
            )
            for row in rows
        }
        # Preserve the caller's ordering; missing ids are skipped rather than
        # raising, because a stale id should degrade gracefully.
        return [by_id[cid] for cid in chunk_ids if cid in by_id]

    def _clear_index_rows(self, *, keep_documents: bool = False) -> None:
        """Delete vectors and chunks. Assumes a transaction is already open."""
        self._conn.execute("DELETE FROM vectors")
        self._conn.execute("DELETE FROM chunks")
        if not keep_documents:
            self._conn.execute("DELETE FROM documents")
        self._bump("structure_revision")

    def clear_index(self, *, keep_documents: bool = False) -> None:
        """Delete all vectors and chunks, optionally the documents too.

        Re-chunking needs chunks and vectors dropped while documents are kept,
        because the documents are the raw material for the new chunks.
        """
        with self.transaction():
            self._clear_index_rows(keep_documents=keep_documents)

    def get_document_ids(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute("SELECT id FROM documents ORDER BY added_at").fetchall()
        return [str(r["id"]) for r in rows]

    # -- sessions -----------------------------------------------------------
    def create_session(self, session_id: str, title: str | None = None) -> None:
        now = time.time()
        with self.transaction():
            self._conn.execute(
                "INSERT INTO sessions(id, title, created_at, updated_at) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at",
                (session_id, title, now, now),
            )

    def append_message(self, session_id: str, role: str, content: str, payload: dict[str, Any] | None = None) -> None:
        now = time.time()
        with self.transaction():
            self._conn.execute(
                "INSERT INTO messages(session_id, role, content, payload, created_at) "
                "VALUES(?, ?, ?, ?, ?)",
                (
                    session_id,
                    role,
                    content,
                    json.dumps(payload or {}, ensure_ascii=False, sort_keys=True),
                    now,
                ),
            )
            self._conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?", (now, session_id)
            )

    def load_messages(self, session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
        query = "SELECT role, content, payload, created_at FROM messages WHERE session_id = ? ORDER BY seq"
        with self._lock:
            rows = self._conn.execute(query, (session_id,)).fetchall()
        if limit is not None and limit >= 0:
            rows = rows[-limit:] if limit else []
        return [
            {"role": r["role"], "content": r["content"], "payload": json.loads(r["payload"]),
             "created_at": r["created_at"]}
            for r in rows
        ]

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
            "SELECT s.id, s.title, s.created_at, s.updated_at, "
            "       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.id) AS message_count "
                "FROM sessions s ORDER BY s.updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    # -- stats --------------------------------------------------------------
    def stats(self) -> CorpusStats:
        with self._lock:
            documents = int(self._conn.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"])
            chunks = int(self._conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"])
            row = self._conn.execute("SELECT COUNT(*) AS n, MAX(dim) AS d FROM vectors").fetchone()
        return CorpusStats(
            documents=documents,
            chunks=chunks,
            vectors=int(row["n"]),
            dimension=int(row["d"]) if row["d"] is not None else None,
        )


def _as_float32_bytes(vector: Sequence[float]) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype="<f4").tobytes()
