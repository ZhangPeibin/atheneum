from __future__ import annotations

import pytest

from atheneum.core.types import Chunk, Document
from atheneum.index.store import Store, decode_term_frequencies, encode_term_frequencies
from atheneum.text.tokenizer import token_frequencies, tokenize


def _doc(source: str = "a.md", content: str = "Hello world.") -> Document:
    return Document(source=source, content=content, title=source)


def _chunks(document: Document) -> list[Chunk]:
    from atheneum.text.splitter import split_document

    return split_document(document)


@pytest.fixture
def store(tmp_path) -> Store:
    built = Store(tmp_path / "s.db")
    yield built
    built.close()


# -- schema -----------------------------------------------------------------
def test_schema_is_recorded(store: Store):
    assert store.get_meta("schema_version") == "1"


def test_reopening_an_existing_database_is_safe(tmp_path):
    path = tmp_path / "s.db"
    first = Store(path)
    first.set_meta("marker", "value")
    first.close()
    second = Store(path)
    assert second.get_meta("documents") is None
    assert second.get_meta("marker") == "value"
    second.close()


def test_newer_schema_version_is_rejected(tmp_path):
    path = tmp_path / "s.db"
    store = Store(path)
    store.set_meta("schema_version", "999")
    store.close()
    with pytest.raises(RuntimeError, match="newer atheneum"):
        Store(path)


# -- documents --------------------------------------------------------------
def test_put_document_reports_duplicates(store: Store):
    assert store.put_document(_doc()) is True
    assert store.put_document(_doc()) is False


def test_document_round_trips_with_metadata(store: Store):
    document = Document(source="x.md", content="body", title="T", metadata={"tag": "v"})
    store.put_document(document)
    loaded = store.get_document(document.id)
    assert loaded is not None
    assert loaded.source == "x.md"
    assert loaded.title == "T"
    assert loaded.metadata == {"tag": "v"}


def test_find_document_by_source(store: Store):
    store.put_document(_doc("found.md", "content"))
    assert store.find_document_by_source("found.md") is not None
    assert store.find_document_by_source("missing.md") is None


def test_get_missing_document_returns_none(store: Store):
    assert store.get_document("nope") is None


def test_list_documents_reports_chunk_counts(store: Store):
    document = _doc("multi.md", "One. Two. Three. " * 20)
    store.put_document(document)
    chunks = _chunks(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    rows = store.list_documents()
    assert rows[0]["chunk_count"] == len(chunks)


def test_delete_document_removes_its_chunks(store: Store):
    document = _doc("doomed.md", "Some content here.")
    store.put_document(document)
    chunks = _chunks(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    assert store.delete_document(document.id) == len(chunks)
    assert store.chunk_count() == 0
    assert store.get_document(document.id) is None


# -- chunks and vectors -----------------------------------------------------
def test_chunk_positions_are_monotonic(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    frequencies = [token_frequencies(tokenize(c.text)) for c in chunks]
    positions = store.put_chunks(chunks, frequencies)
    assert positions == sorted(positions)
    assert len(set(positions)) == len(positions)


def test_duplicate_chunk_ids_are_not_reinserted(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    frequencies = [token_frequencies(tokenize(c.text)) for c in chunks]
    first = store.put_chunks(chunks, frequencies)
    second = store.put_chunks(chunks, frequencies)
    assert first == second
    assert store.chunk_count() == len(chunks)


def test_embeddings_are_stored_and_streamed_back(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    frequencies = [token_frequencies(tokenize(c.text)) for c in chunks]
    vectors = [[1.0, 2.0, 3.0] for _ in chunks]
    store.put_chunks(chunks, frequencies, vectors)
    rows = list(store.iter_rows())
    assert len(rows) == len(chunks)
    assert rows[0].embedding is not None


def test_mismatched_lengths_are_rejected(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    frequencies = [token_frequencies(tokenize(c.text)) for c in chunks]
    with pytest.raises(ValueError, match="term-frequency"):
        store.put_chunks(chunks, [])
    with pytest.raises(ValueError, match="embeddings"):
        store.put_chunks(chunks, frequencies, [[1.0], [2.0]])


def test_iter_rows_returns_positions_in_order(store: Store):
    for name in ("a.md", "b.md", "c.md"):
        document = _doc(name, f"Content for {name}. Sentence two.")
        store.put_document(document)
        chunks = _chunks(document)
        store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    rows = list(store.iter_rows())
    assert [r.pos for r in rows] == sorted(r.pos for r in rows)
    assert [r.chunk.source for r in rows] == ["a.md", "b.md", "c.md"]


def test_get_chunks_by_id_preserves_order_and_skips_missing(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    fetched = store.get_chunks_by_id([chunks[0].id, "does-not-exist"])
    assert [c.id for c in fetched] == [chunks[0].id]
    assert store.get_chunks_by_id([]) == []


def test_clear_index_can_keep_documents(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    store.clear_index(keep_documents=True)
    assert store.chunk_count() == 0
    assert store.get_document(document.id) is not None
    store.clear_index(keep_documents=False)
    assert store.get_document(document.id) is None


# -- sessions ---------------------------------------------------------------
def test_session_messages_round_trip(store: Store):
    store.create_session("s1", title="Test")
    store.append_message("s1", "user", "hello")
    store.append_message("s1", "assistant", "hi", {"turn": 1})
    rows = store.load_messages("s1")
    assert [r["role"] for r in rows] == ["user", "assistant"]
    assert rows[1]["payload"] == {"turn": 1}


def test_load_messages_limit_keeps_the_tail(store: Store):
    store.create_session("s2")
    for i in range(10):
        store.append_message("s2", "user", f"m{i}")
    assert [r["content"] for r in store.load_messages("s2", limit=3)] == ["m7", "m8", "m9"]


def test_sessions_are_listed_by_recent_activity(store: Store):
    store.create_session("old")
    store.create_session("new")
    store.append_message("new", "user", "x")
    ids = [row["id"] for row in store.list_sessions()]
    assert ids[0] == "new"


# -- stats and helpers ------------------------------------------------------
def test_stats_counts_everything(store: Store):
    document = _doc()
    store.put_document(document)
    chunks = _chunks(document)
    store.put_chunks(
        chunks,
        [token_frequencies(tokenize(c.text)) for c in chunks],
        [[1.0, 0.0] for _ in chunks],
    )
    stats = store.stats()
    assert stats.documents == 1
    assert stats.chunks == len(chunks)
    assert stats.vectors == len(chunks)
    assert stats.dimension == 2
    assert stats.as_dict()["chunks"] == len(chunks)


def test_stats_on_an_empty_store(store: Store):
    stats = store.stats()
    assert stats.chunks == 0
    assert stats.dimension is None


def test_term_frequency_codec_round_trip():
    payload = {"hello": 3, "world": 1, "caféd": 9}
    assert decode_term_frequencies(encode_term_frequencies(payload)) == payload


def test_term_frequency_codec_is_compact():
    payload = {f"term{i}": i for i in range(500)}
    assert len(encode_term_frequencies(payload)) < len(str(payload))


def test_store_context_manager_closes(tmp_path):
    with Store(tmp_path / "cm.db") as store:
        assert store.stats().documents == 0


def test_wal_mode_enabled_for_file_databases(store: Store):
    mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
    assert str(mode).lower() == "wal"
