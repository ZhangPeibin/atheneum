"""Regressions from adversarial code review.

Every test here pins a defect that a reviewer reproduced and that the rest of the
suite did not catch. They are grouped by the review round that found them so the
provenance stays readable.
"""

from __future__ import annotations

import asyncio
import collections
import itertools
import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pytest

from atheneum.agent.loop import Agent, AgentConfig
from atheneum.agent.tools import Tool, ToolRegistry, tool
from atheneum.core.types import Message, ToolCall, ToolResult
from atheneum.index.bm25 import BM25Index
from atheneum.index.selection import top_k_indices
from atheneum.index.vectors import VectorIndex
from atheneum.providers.base import Generation, GenerationRequest, Provider, ProviderError, Usage
from atheneum.providers.offline import OfflineProvider, parse_evidence
from atheneum.retrieval.fusion import RRFFusion
from atheneum.retrieval.pipeline import Corpus
from atheneum.text.splitter import SplitterConfig, split_text
from atheneum.text.tokenizer import tokenize

# ---------------------------------------------------------------------------
# Round 1 — text layer and fusion
# ---------------------------------------------------------------------------


def _missing_chars(original: str, chunks: list[str]) -> int:
    """Count non-whitespace characters present in the input but absent from the output."""
    before = collections.Counter(c for c in original if not c.isspace())
    after = collections.Counter(c for c in "".join(chunks) if not c.isspace())
    return sum(count - after.get(char, 0) for char, count in before.items() if count > after.get(char, 0))


@pytest.mark.parametrize(
    ("name", "text", "config"),
    [
        ("heading-only", "# Title\n## Section\n### Sub", SplitterConfig()),
        ("long-body-short-tail", ("This is a sentence about retrieval. " * 8) + "\n# End", SplitterConfig()),
        ("overlap-near-chunk-size", "word " * 48, SplitterConfig(chunk_size=20, chunk_overlap=19)),
        (
            "text-around-atomic-fence",
            "text before\n```python\nx = 1\ny = 2\n```\ntext after",
            SplitterConfig(),
        ),
    ],
)
def test_splitter_never_loses_content(name: str, text: str, config: SplitterConfig):
    """The old min_chunk_chars filter discarded text silently.

    With chunk_size=20 and overlap=19 a 240-character input produced 240 short
    pieces, all were filtered out, and the `or pieces[:1]` fallback kept a single
    20-character chunk — 73% of the document vanished from the index.
    """
    chunks = split_text(text, config)
    assert chunks, name
    assert _missing_chars(text, chunks) == 0, f"{name} lost content"


def test_cjk_commas_survive_splitting():
    text = "混合检索使用倒数排名融合，向量检索负责语义匹配，融合之后再做重排序，可以提高召回质量，" * 3
    chunks = split_text(text, SplitterConfig(chunk_size=60, chunk_overlap=10))
    assert _missing_chars(text, chunks) == 0


def test_short_pieces_merge_rather_than_multiply():
    """min_chunk_chars still does useful work: it merges, it just never drops."""
    text = "# A\n\ntiny\n\n# B\n\nAnother very small section here."
    chunks = split_text(text, SplitterConfig(chunk_size=200, chunk_overlap=20))
    assert len(chunks) < 5
    assert _missing_chars(text, chunks) == 0


def test_rrf_ignores_a_duplicate_key_within_one_list():
    """A repeated key used to be counted twice, pushing the score past 1.0."""
    fused = RRFFusion().fuse({"a": [("x", "va"), ("x", "va2")], "b": [("x", "vb")]})
    top = fused[0]
    assert top.score <= 1.0 + 1e-12
    assert top.score == pytest.approx(sum(top.contributions.values()))


def test_rrf_contributions_sum_to_the_reported_score():
    fused = RRFFusion().fuse({"a": [("x", 1)], "b": [("x", 1)]})[0]
    assert fused.score == pytest.approx(1.0)
    assert sum(fused.contributions.values()) == pytest.approx(fused.score)


def test_rrf_payload_does_not_depend_on_dict_order():
    forward = RRFFusion().fuse({"a": [("x", "va")], "b": [("x", "vb")]})[0]
    backward = RRFFusion().fuse({"b": [("x", "vb")], "a": [("x", "va")]})[0]
    assert forward.value == backward.value
    assert forward.score == backward.score


def test_rrf_prefers_the_higher_weighted_lists_payload():
    fused = RRFFusion().fuse({"weak": [("x", "from-weak")], "strong": [("x", "from-strong")]}, weights=[0.2, 0.8])[0]
    assert fused.value == "from-strong"


def test_hangul_compatibility_jamo_are_retained():
    """U+3130-318F was missing from the CJK ranges, so lone jamo were dropped."""
    assert "ᄂ" in tokenize("ㅎㅏㄴ") or "ㄴ" in tokenize("ㅎㅏㄴ")
    assert tokenize("한국어") == ["한", "국", "어", "한국", "국어"]


def test_zero_width_joiner_does_not_break_a_cjk_bigram():
    assert tokenize("中\u200d文") == tokenize("中文")
    assert "中文" in tokenize("中\u200d文")


# ---------------------------------------------------------------------------
# Round 2 — index layer determinism
# ---------------------------------------------------------------------------

_SCORE_ALL_SCRIPT = (
    "from atheneum.index.bm25 import BM25Index;"
    "b = BM25Index();"
    "b.add('alpha beta beta gamma gamma gamma gamma gamma');"
    "[b.add(t) for t in ('alpha', 'beta', 'gamma')];"
    "b.finalize();"
    "print(b.score_all('alpha beta gamma').tobytes().hex())"
)


def test_score_all_is_bit_reproducible_across_processes():
    """Iterating a set made float accumulation order depend on PYTHONHASHSEED."""
    outputs = set()
    for seed in ("0", "1", "7", "42", "999"):
        result = subprocess.run(
            [sys.executable, "-c", _SCORE_ALL_SCRIPT],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        assert result.returncode == 0, result.stderr
        outputs.add(result.stdout.strip())
    assert len(outputs) == 1, f"score_all varied with PYTHONHASHSEED: {outputs}"


def test_score_all_agrees_with_search():
    index = BM25Index()
    index.add("alpha beta beta gamma gamma gamma gamma gamma")
    for term in ("alpha", "beta", "gamma"):
        index.add(term)
    index.finalize()
    dense = index.score_all("alpha beta gamma")
    for position, score in index.search("alpha beta gamma", top_k=4):
        assert dense[position] == score


@pytest.mark.parametrize("top_k", [1, 4, 9, 10, 11])
def test_ties_resolve_to_the_lowest_index_in_both_retrievers(top_k: int):
    """argpartition picks arbitrarily among ties; sorting afterwards did not help."""
    vectors = VectorIndex(dim=2)
    for _ in range(10):
        vectors.add([1.0, 0.0])
    expected = list(range(min(top_k, 10)))
    assert [i for i, _ in vectors.search([1.0, 0.0], top_k=top_k)] == expected

    lexical = BM25Index()
    for _ in range(10):
        lexical.add("identical text here")
    lexical.finalize()
    assert [i for i, _ in lexical.search("identical", top_k=top_k)] == expected


def test_top_k_indices_selection_is_directly_testable():
    scores = np.array([0.0, 5.0, 5.0, 1.0, 5.0])
    assert list(top_k_indices(scores, 2)) == [1, 2]
    assert list(top_k_indices(scores, 3)) == [1, 2, 4]
    assert list(top_k_indices(scores, 0)) == []
    assert list(top_k_indices(np.zeros(5), 3)) == []


def test_matrix_view_does_not_expose_a_writable_base():
    """matrix.base was the internal capacity buffer, so callers could mutate rows."""
    index = VectorIndex(dim=4)
    for _ in range(3):
        index.add([1.0, 0.0, 0.0, 0.0])
    view = index.matrix
    assert view.shape == (3, 4)
    with pytest.raises(ValueError):
        view[0, 0] = 9.0
    base = getattr(view, "base", None)
    if base is not None:
        with pytest.raises((ValueError, AttributeError)):
            base[0, 0] = 9.0
    assert index.matrix[0, 0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Round 3 — re-review of the round-1 fixes
# ---------------------------------------------------------------------------


class _Scripted(Provider):
    name = "scripted"

    def __init__(self, generations: list[Generation]) -> None:
        self.generations = generations
        self.calls = 0

    def complete(self, request: GenerationRequest) -> Generation:
        generation = self.generations[min(self.calls, len(self.generations) - 1)]
        self.calls += 1
        return generation


class _FailingMidStream(Provider):
    """Emits text and usage, then fails — the case the error path must preserve."""

    name = "failing"

    def __init__(self, text: str, usage: Usage) -> None:
        self._text = text
        self._usage = usage

    def complete(self, request: GenerationRequest) -> Generation:
        raise ProviderError("boom")

    def stream(self, request: GenerationRequest):
        from atheneum.providers.base import TextDelta, UsageEvent

        yield TextDelta(text=self._text)
        yield UsageEvent(usage=self._usage)
        raise ProviderError("boom")


@pytest.fixture
def noop_registry() -> ToolRegistry:
    def noop() -> str:
        """Does nothing."""
        return "ok"

    return ToolRegistry([tool(noop)])


def test_stream_error_path_keeps_usage_reported_before_the_failure():
    """Usage the provider emitted before raising was silently dropped."""
    agent = Agent(_FailingMidStream("partial", Usage(7, 3)), ToolRegistry())
    run = list(agent.stream("q"))[-1].run
    assert run.stopped_reason == "provider_error"
    assert run.usage.total_tokens == 10
    assert run.answer == "partial"


def test_stream_error_path_falls_back_to_earlier_turns(noop_registry: ToolRegistry):
    """run() fell back to earlier text on failure; stream() returned only the last turn's."""
    generations = [
        Generation(
            text="earlier finding",
            tool_calls=[ToolCall(id="c0", name="noop", arguments={})],
            finish_reason="tool_calls",
            usage=Usage(10, 5),
        )
    ]

    class ThenFail(_Scripted):
        def complete(self, request: GenerationRequest) -> Generation:
            if self.calls == 0:
                return super().complete(request)
            raise ProviderError("boom")

    direct = Agent(ThenFail(generations), noop_registry, config=AgentConfig(max_turns=4)).run("q")
    streamed = list(Agent(ThenFail(generations), noop_registry, config=AgentConfig(max_turns=4)).stream("q"))[-1].run
    assert direct.answer == streamed.answer == "earlier finding"
    assert direct.usage == streamed.usage


@pytest.mark.parametrize("exc", [asyncio.CancelledError, GeneratorExit])
def test_cancellation_signals_are_not_converted_into_tool_data(exc):
    """Catching BaseException must not defeat task cancellation."""

    def cancel() -> str:
        """Raises a cancellation signal."""
        raise exc

    with pytest.raises(BaseException) as info:
        ToolRegistry([tool(cancel)]).execute(ToolCall(id="c", name="cancel", arguments={}))
    assert isinstance(info.value, exc)


def test_a_raising_repr_does_not_abort_the_run():
    """Serialisation happened outside the guard, so a bad __repr__ escaped."""

    class Unprintable:
        def __repr__(self) -> str:
            raise RuntimeError("cannot represent")

    def returns_unprintable() -> object:
        """Returns an object that cannot be serialised."""
        return Unprintable()

    result = ToolRegistry([tool(returns_unprintable)]).execute(
        ToolCall(id="c", name="returns_unprintable", arguments={})
    )
    # Serialisation degrades to a typed marker instead of raising, so the run
    # continues and nothing from the hostile __repr__ reaches the model.
    assert not result.is_error
    assert "unserialisable" in result.content
    assert "cannot represent" not in result.content


@pytest.mark.parametrize("raw", ["1e999", "nan", "Infinity", "-Infinity", "NaN"])
def test_non_finite_evidence_scores_are_rejected(raw: str):
    """float() accepts these, and inf/nan sort ahead of every real score."""
    message = Message.tool(
        ToolResult(
            call_id="c",
            name="search",
            content=json.dumps({"results": [{"source": "a.md", "text": "A real passage.", "score": raw}]}),
        )
    )
    assert parse_evidence([message]) == []


def test_finite_and_absent_scores_are_still_accepted():
    def evidence_for(score: object) -> list:
        message = Message.tool(
            ToolResult(
                call_id="c",
                name="search",
                content=json.dumps({"results": [{"source": "a.md", "text": "A real passage.", "score": score}]}),
            )
        )
        return parse_evidence([message])

    assert evidence_for(0.5)[0].score == 0.5
    assert evidence_for(None)[0].score == 0.0
    assert evidence_for(-1.0)[0].score == -1.0


def test_a_bad_entry_does_not_displace_good_evidence():
    payload = {
        "results": [
            {"source": "bad.md", "text": "Poisoned passage with a huge score.", "score": "1e999"},
            {"source": "good.md", "text": "Genuine passage about retrieval.", "score": 0.9},
        ]
    }
    message = Message.tool(ToolResult(call_id="c", name="search", content=json.dumps(payload)))
    evidence = parse_evidence([message])
    assert [item.source for item in evidence] == ["good.md"]


def test_multiple_usage_events_in_one_turn_do_not_double_count(noop_registry: ToolRegistry):
    """openai_compat emits one cumulative usage per turn, so the loop must replace."""

    class MultiUsage(Provider):
        name = "multi"

        def complete(self, request: GenerationRequest) -> Generation:
            return Generation(text="done", usage=Usage(4, 2))

        def stream(self, request: GenerationRequest):
            from atheneum.providers.base import TextDelta, UsageEvent

            yield TextDelta(text="done")
            yield UsageEvent(usage=Usage(4, 2))
            yield UsageEvent(usage=Usage(4, 2))

    streamed = list(Agent(MultiUsage(), noop_registry).stream("q"))[-1].run
    direct = Agent(MultiUsage(), noop_registry).run("q")
    assert streamed.usage == direct.usage == Usage(4, 2)


def test_offline_provider_still_answers_after_the_guards():
    provider = OfflineProvider()
    payload = {"results": [{"source": "a.md", "ordinal": 0, "text": "Reciprocal rank fusion merges lists.", "score": 0.9}]}
    request = GenerationRequest(
        messages=[
            Message.user("what is fusion"),
            Message.tool(ToolResult(call_id="c", name="search", content=json.dumps(payload))),
        ]
    )
    generation = provider.complete(request)
    assert "Reciprocal rank fusion" in generation.text
    assert "a.md" in generation.text


# ---------------------------------------------------------------------------
# Round 4 — index alignment (2 blockers) and filesystem security (2 highs)
# ---------------------------------------------------------------------------

import sqlite3  # noqa: E402

from atheneum.core.types import Document  # noqa: E402
from atheneum.ingest import discover_files, read_file  # noqa: E402


def test_a_vectorless_chunk_does_not_desync_the_incremental_path(tmp_path):
    """BLOCKER: the matrix fell one row behind _rows and misattributed results.

    An incremental search returned v4.md where a fresh rebuild of the same
    database returned v5.md, at an identical score of 1.0 — a silently wrong
    citation, which is the worst failure mode a retrieval system has.
    """
    db = str(tmp_path / "align.db")
    corpus = Corpus.open(db)
    corpus.add_text("v1.md", "alpha alpha unique one")
    corpus.add_text("v2.md", "beta beta unique two")
    corpus.search("alpha", top_k=2)  # force a full load
    corpus.add_text("v4.md", "gamma gamma unique four")  # incremental path

    with sqlite3.connect(db) as conn:
        pos = conn.execute("SELECT pos FROM chunks WHERE source LIKE '%v4%'").fetchone()[0]
        conn.execute("DELETE FROM vectors WHERE pos = ?", (pos,))

    corpus.add_text("v5.md", "delta delta unique five")
    incremental = [h.chunk.source for h in corpus.search("delta unique five", top_k=3, mode="vector")]
    corpus.close()

    fresh = Corpus.open(db)
    rebuilt = [h.chunk.source for h in fresh.search("delta unique five", top_k=3, mode="vector")]
    fresh.close()

    assert incremental == rebuilt
    assert incremental[0] == "v5.md"


def test_a_failed_rebuild_does_not_destroy_documents(tmp_path):
    """BLOCKER: rebuild() cleared the documents table before re-embedding."""
    db = str(tmp_path / "rebuild.db")
    corpus = Corpus.open(db)
    corpus.add_text("d1.md", "first document content here")
    corpus.add_text("d2.md", "second document content here")
    assert corpus.stats()["documents"] == 2

    class Failing:
        name = "failing"
        dim = 512

        def embed(self, text: str) -> np.ndarray:
            raise RuntimeError("embedder offline")

        def embed_many(self, texts) -> np.ndarray:
            raise RuntimeError("embedder offline")

    corpus.embedder = Failing()
    with pytest.raises(RuntimeError):
        corpus.rebuild()

    stats = corpus.stats()
    assert stats["documents"] == 2, "a failed rebuild must not lose source documents"
    corpus.close()

    reopened = Corpus.open(db)
    assert reopened.stats()["documents"] == 2
    reopened.close()


def test_an_orphaned_document_can_be_reindexed(tmp_path):
    """MAJOR: a failure between put_document and put_chunks wedged it forever."""
    db = str(tmp_path / "orphan.db")
    corpus = Corpus.open(db)

    class FailsOnSecondBatch:
        name = "flaky"
        dim = 8
        calls = 0

        def embed(self, text: str) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

        def embed_many(self, texts) -> np.ndarray:
            type(self).calls += 1
            if type(self).calls >= 2:
                raise RuntimeError("transient failure")
            return np.zeros((len(texts), 8), dtype=np.float32)

    corpus.embedder = FailsOnSecondBatch()
    with pytest.raises(RuntimeError):
        corpus.add_documents(
            [Document(source="h1.md", content="one"), Document(source="h2.md", content="two")]
        )

    class Working:
        name = "working"
        dim = 8

        def embed(self, text: str) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

        def embed_many(self, texts) -> np.ndarray:
            return np.zeros((len(texts), 8), dtype=np.float32)

    corpus.embedder = Working()
    # h2 must not be permanently skipped as "already indexed".
    assert corpus.add_text("h2.md", "two") >= 1
    counts = {row["source"]: row["chunk_count"] for row in corpus.sources()}
    assert counts["h2.md"] >= 1
    corpus.close()


def test_concurrent_writer_is_detected_by_revision_counters(tmp_path):
    """MAJOR: a count-based skip could not see another instance's delete."""
    db = str(tmp_path / "shared.db")
    a = Corpus.open(db)
    a.add_text("a1.md", "alpha one")
    a.add_text("a2.md", "beta two")
    a.add_text("a3.md", "gamma three")

    b = Corpus.open(db)
    b.search("alpha", top_k=3)  # b loads three rows

    doc_id = next(row["id"] for row in a.sources() if row["source"] == "a2.md")
    a.delete_document(doc_id)
    a.add_text("a4.md", "delta four")

    hits = [h.chunk.source for h in b.search("beta two", top_k=3)]
    assert "a2.md" not in hits, "b served a chunk that another writer deleted"
    assert any(h.chunk.source == "a4.md" for h in b.search("delta four", top_k=3))
    a.close()
    b.close()


def test_limit_zero_indexes_nothing(tmp_path):
    """MINOR: the limit was checked after the append, so 0 indexed one file."""
    source = tmp_path / "tree"
    source.mkdir()
    (source / "x.md").write_text("some content here", encoding="utf-8")
    corpus = Corpus.in_memory()
    assert corpus.add_paths([source], limit=0) == 0
    assert corpus.stats()["documents"] == 0
    corpus.close()


def test_symlinked_file_does_not_escape_the_indexed_tree(tmp_path):
    """HIGH: `is_file()` follows symlinks, so follow_symlinks=False was a lie."""
    outside = tmp_path / "secret.md"
    outside.write_text("TOP-SECRET-OUTSIDE-CONTENT", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.md").write_text("legitimate content", encoding="utf-8")
    (root / "link.md").symlink_to(outside)

    found = {p.name for p in discover_files(root)}
    assert found == {"real.md"}
    assert not any("secret" in str(p) for p in discover_files(root))


def test_fifo_and_device_nodes_are_refused_not_hung(tmp_path):
    """MEDIUM: opening a FIFO for reading blocks forever."""
    fifo = tmp_path / "pipe.md"
    os.mkfifo(fifo)
    assert list(discover_files(tmp_path)) == []
    with pytest.raises(ValueError, match="not a regular file"):
        read_file(fifo)


def test_dot_env_is_never_indexed(tmp_path):
    """MEDIUM: `.env` holds API keys and was in TEXT_SUFFIXES."""
    (tmp_path / ".env").write_text("API_KEY=sk-indexed-secret", encoding="utf-8")
    (tmp_path / ".env.local").write_text("API_KEY=sk-other", encoding="utf-8")
    (tmp_path / "notes.md").write_text("real prose content", encoding="utf-8")
    found = {p.name for p in discover_files(tmp_path)}
    assert found == {"notes.md"}


def test_a_deeply_nested_argument_does_not_raise_recursion_error():
    """HIGH: the coercion error message used repr(), which blew the stack.

    The RecursionError happened while *building* the TypeError, inside the except
    handler, so it escaped execute() and aborted the agent run.
    """
    from atheneum.agent.tools import ToolRegistry, tool

    def want_str(items: str) -> int:
        """Wants a string."""
        return len(items)

    deep: object = [0]
    for _ in range(6000):
        deep = [deep]

    result = ToolRegistry([tool(want_str)]).execute(
        ToolCall(id="c", name="want_str", arguments={"items": deep})
    )
    assert result.is_error
    assert "must be a string" in result.content


def test_get_chunks_by_id_survives_a_huge_id_list(tmp_path):
    """LOW: one IN list above SQLite's variable limit raised OperationalError."""
    from atheneum.index.store import Store

    store = Store(tmp_path / "many.db")
    document = Document(source="x.md", content="hello world content")
    store.put_document(document)
    from atheneum.text.splitter import split_document

    chunks = split_document(document)
    store.put_chunks(chunks, [{} for _ in chunks])
    assert len(store.get_chunks_by_id([chunks[0].id] * 40_000)) == 40_000
    store.close()


# ---------------------------------------------------------------------------
# Round 5 — store locking, rebuild atomicity, digest budget, config validation
# ---------------------------------------------------------------------------

import threading  # noqa: E402
import time  # noqa: E402

from atheneum.agent.builtin_tools import build_corpus_tools  # noqa: E402
from atheneum.agent.memory import compact, estimate_tokens, summarize_messages  # noqa: E402
from atheneum.config import Config  # noqa: E402
from atheneum.index.store import Store  # noqa: E402


def test_an_uncommitted_row_is_not_visible_to_another_thread(tmp_path):
    """MAJOR: reads ran with no lock, inside another thread's open transaction."""
    store = Store(tmp_path / "lock.db")
    secret = Document(source="secret.md", content="UNCOMMITTED-SECRET")
    observed: list[bool] = []

    def writer() -> None:
        try:
            with store.transaction():
                store._put_document_row(secret)
                time.sleep(0.4)
                raise RuntimeError("rollback")
        except RuntimeError:
            pass

    thread = threading.Thread(target=writer)
    thread.start()
    time.sleep(0.15)
    observed.append(store.get_document(secret.id) is not None)
    thread.join()

    assert observed == [False], "a reader saw a row that was later rolled back"
    assert store.get_document(secret.id) is None
    store.close()


def test_a_failed_rebuild_leaves_the_index_untouched(tmp_path):
    """MAJOR: the wipe committed before re-embedding, so a failure left orphans."""
    db = str(tmp_path / "atomic.db")
    corpus = Corpus.open(db)
    for i in (1, 2, 3):
        corpus.add_text(f"doc{i}.txt", f"content for document {i} here")
    before = sorted((r["source"], r["chunk_count"]) for r in corpus.sources())

    class Flaky:
        name = "flaky"
        dim = 8
        calls = 0

        def embed(self, text: str) -> np.ndarray:
            return np.zeros(8, dtype=np.float32)

        def embed_many(self, texts) -> np.ndarray:
            type(self).calls += 1
            if type(self).calls >= 2:
                raise RuntimeError("BOOM")
            return np.zeros((len(texts), 8), dtype=np.float32)

    corpus.embedder = Flaky()
    with pytest.raises(RuntimeError):
        corpus.rebuild()

    after = sorted((r["source"], r["chunk_count"]) for r in corpus.sources())
    assert after == before
    assert corpus.stats()["chunks"] == 3
    assert not [s for s, n in after if n == 0], "no document may be left with zero chunks"
    corpus.close()


def test_iter_rows_does_not_hold_the_lock_while_iterating(tmp_path):
    """MINOR: the generator held an RLock across yields, blocking every writer."""
    from atheneum.text.splitter import split_document
    from atheneum.text.tokenizer import token_frequencies, tokenize

    store = Store(tmp_path / "iter.db")
    document = Document(source="a.md", content="hello world content that is long enough to chunk")
    store.put_document(document)
    chunks = split_document(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    rows = store.iter_rows()
    assert isinstance(rows, list)

    outcome: list[str] = []

    def writer() -> None:
        try:
            store.set_meta("key", "value")
            outcome.append("ok")
        except Exception as exc:
            outcome.append(type(exc).__name__)

    thread = threading.Thread(target=writer)
    thread.start()
    thread.join(timeout=5)
    assert outcome == ["ok"], f"a writer was blocked or failed: {outcome}"
    assert rows
    store.close()


@pytest.mark.parametrize("budget", [1, 4, 8])
def test_digest_is_omitted_when_the_header_alone_cannot_fit(budget: int):
    """MINOR: the digest used to exceed the budget it was handed."""
    messages = [Message.user("fact " + "x" * 40)]
    assert summarize_messages(messages, budget) is None


def test_digest_stays_within_a_budget_it_can_meet():
    messages = [Message.user("fact " + "x" * 40)]
    digest = summarize_messages(messages, 40)
    assert digest is not None
    assert estimate_tokens(digest.content) <= 40


def test_compact_does_not_add_a_digest_that_overflows_the_remainder():
    messages = [Message.user("y" * 4000), Message.user("final question")]
    result = compact(messages, 60)
    assert result[-1].content == "final question"
    oversized = [
        m for m in result if m.role.value == "system" and estimate_tokens(m.content) > 60
    ]
    assert not oversized


def test_a_retrieved_finding_survives_a_tight_budget():
    finding = Message.tool(
        ToolResult(call_id="c", name="search", content="CRITICAL-FINDING " + "z" * 400)
    )
    prose = [Message.user(f"chatter {i} " + "q" * 30) for i in range(3)]
    digest = summarize_messages([*prose, finding], 30)
    assert digest is not None
    assert "CRITICAL-FINDING" in digest.content
    assert estimate_tokens(digest.content) <= 30


def test_config_rejects_settings_that_cannot_work():
    for kwargs in (
        {"top_k": 0},
        {"fusion_k": 0},
        {"chunk_overlap": 5000, "chunk_size": 500},
        {"max_turns": -5},
        {"token_budget": -1},
        {"embedder_dim": 0},
    ):
        with pytest.raises(ValueError, match="invalid configuration"):
            Config(**kwargs)


def test_store_reads_are_serialized_with_writes(tmp_path):
    """Every read path must take the lock; this exercises the main ones at once."""
    from atheneum.text.splitter import split_document
    from atheneum.text.tokenizer import token_frequencies, tokenize

    store = Store(tmp_path / "reads.db")
    document = Document(source="a.md", content="hello world content that is long enough to chunk")
    store.put_document(document)
    chunks = split_document(document)
    store.put_chunks(chunks, [token_frequencies(tokenize(c.text)) for c in chunks])
    errors: list[str] = []

    def reader() -> None:
        try:
            for _ in range(30):
                store.get_document(document.id)
                store.list_documents()
                store.stats()
                store.chunk_count()
                store.find_document_by_source("a.md")
                store.get_document_ids()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    def writer() -> None:
        try:
            for i in range(30):
                store.set_meta(f"k{i}", str(i))
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=reader) for _ in range(3)] + [threading.Thread(target=writer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
    assert not errors, errors
    store.close()


# ---------------------------------------------------------------------------
# Round 6 — resource bounds on ingest and tool results
# ---------------------------------------------------------------------------


def test_add_paths_streams_instead_of_materialising_every_document(tmp_path):
    """LOW: the full Document list was built up front, so a big tree was held in RAM."""
    import inspect

    from atheneum.retrieval.pipeline import Corpus

    source = inspect.getsource(Corpus.add_paths)
    assert "def stream()" in source, "add_paths should build a lazy generator"

    tree = tmp_path / "tree"
    tree.mkdir()
    for i in range(12):
        (tree / f"n{i}.md").write_text(f"document number {i} has content", encoding="utf-8")
    corpus = Corpus.in_memory()
    assert corpus.add_paths([tree]) == 12
    assert corpus.stats()["documents"] == 12
    assert corpus.add_paths([tree], limit=3) == 0  # already indexed
    corpus.close()


def test_add_paths_respects_limit_without_reading_extra_files(tmp_path):
    tree = tmp_path / "tree"
    tree.mkdir()
    for i in range(10):
        (tree / f"n{i}.md").write_text(f"document number {i} has content", encoding="utf-8")
    corpus = Corpus.in_memory()
    assert corpus.stats()["documents"] == 0
    corpus.add_paths([tree], limit=3)
    assert corpus.stats()["documents"] == 3
    corpus.close()


def test_a_huge_tool_result_is_bounded_before_serialisation():
    """LOW: json.dumps ran to completion and only then got truncated."""
    from atheneum.agent.tools import ToolRegistry, tool

    def big(n: int) -> list:
        """Returns many items."""
        return list(range(n))

    result = ToolRegistry([tool(big)]).execute(ToolCall(id="c", name="big", arguments={"n": 500_000}))
    assert not result.is_error
    assert len(result.content) < 20_000
    assert "showing the first" in result.content


def test_a_huge_dict_result_is_bounded_too():
    from atheneum.agent.tools import ToolRegistry, tool

    def wide(n: int) -> dict:
        """Returns many keys."""
        return {f"key{i}": i for i in range(n)}

    result = ToolRegistry([tool(wide)]).execute(ToolCall(id="c", name="wide", arguments={"n": 100_000}))
    assert not result.is_error
    # The element-bound note is prefixed, so character truncation cannot cut it.
    assert result.content.startswith("[showing the first")
    assert "keys]" in result.content[:80]
    assert len(result.content) < 200_000


def test_an_unserialisable_result_never_calls_repr():
    """NIT: json.dumps(default=str) fell back to repr, leaking internals."""
    from atheneum.agent.tools import ToolRegistry, tool

    class Hostile:
        def __repr__(self) -> str:
            raise RuntimeError("no repr for you")

    def returns_hostile() -> object:
        """Returns something json cannot handle."""
        return {"k": Hostile()}

    result = ToolRegistry([tool(returns_hostile)]).execute(
        ToolCall(id="c", name="returns_hostile", arguments={})
    )
    # The serializer's own failure text must not be forwarded: str() falls back to
    # __repr__, so a hostile object could otherwise write into model-visible output.
    assert "no repr for you" not in result.content
    assert "unserialisable" in result.content


# ---------------------------------------------------------------------------
# Round 7 — Anthropic wire format (1 blocker) and HTTP API hardening
# ---------------------------------------------------------------------------

from atheneum.providers.anthropic import _build_messages, _coerce_input, _decode  # noqa: E402


def test_parallel_tool_results_share_one_user_message():
    """BLOCKER: two results became two adjacent user messages -> HTTP 400.

    Anthropic requires alternating roles, so any turn with parallel tool calls
    was rejected outright.
    """
    messages = [
        Message.assistant("", [ToolCall(id="t1", name="a", arguments={}), ToolCall(id="t2", name="b", arguments={})]),
        Message.tool(ToolResult(call_id="t1", name="a", content="r1")),
        Message.tool(ToolResult(call_id="t2", name="b", content="r2", is_error=True)),
    ]
    encoded = _build_messages(messages)
    assert [m["role"] for m in encoded] == ["assistant", "user"]
    assert len(encoded[-1]["content"]) == 2
    assert [b["is_error"] for b in encoded[-1]["content"]] == [False, True]


def test_no_two_adjacent_messages_share_a_role():
    cases = [
        [Message.assistant("a"), Message.user(""), Message.assistant("b")],
        [Message.assistant("x"), Message.assistant("y")],
        [Message.user("p"), Message.user("q")],
        [Message.assistant("", [ToolCall(id="t", name="a", arguments={})]),
         Message.tool(ToolResult(call_id="t", name="a", content="r")),
         Message.tool(ToolResult(call_id="t2", name="b", content="r2")),
         Message.assistant("done")],
    ]
    for messages in cases:
        roles = [m["role"] for m in _build_messages(messages)]
        assert all(a != b for a, b in itertools.pairwise(roles)), roles


def test_is_error_comes_from_the_flag_not_the_text():
    """Sniffing for a leading '{"error"' misfired in both directions."""
    call = Message.assistant("", [ToolCall(id="t", name="a", arguments={})])

    false_positive = _build_messages(
        [call, Message.tool(ToolResult(call_id="t", name="a", content='{"error handling": "documented"}'))]
    )[-1]["content"][0]
    assert false_positive["is_error"] is False

    false_negative = _build_messages(
        [call, Message.tool(ToolResult(call_id="t", name="a", content="Error: file not found", is_error=True))]
    )[-1]["content"][0]
    assert false_negative["is_error"] is True


def test_tool_input_arriving_as_a_json_string_is_parsed():
    """MAJOR: dict() on a string raised ValueError and crashed complete()."""
    assert _coerce_input('{"a": 1}') == {"a": 1}
    assert _coerce_input({"a": 1}) == {"a": 1}
    assert _coerce_input(None) == {}
    bad = _coerce_input("not json")
    assert bad["_raw"] == "not json" and "_parse_error" in bad

    generation = _decode(
        "m",
        {"content": [{"type": "tool_use", "id": "x", "name": "n", "input": '{"a": 1}'}],
         "stop_reason": "tool_use", "usage": {}},
    )
    assert generation.tool_calls[0].arguments == {"a": 1}
    assert generation.finish_reason == "tool_calls"


@pytest.mark.parametrize(
    ("stop_reason", "expected"),
    [("end_turn", "stop"), ("max_tokens", "length"), ("pause_turn", "error"), ("refusal", "error")],
)
def test_stop_reason_is_not_flattened_to_stop(stop_reason: str, expected: str):
    """pause_turn means "continue" and refusal means "declined"; neither is an answer."""
    generation = _decode(
        "m", {"content": [{"type": "text", "text": "hi"}], "stop_reason": stop_reason, "usage": {}}
    )
    assert generation.finish_reason == expected


def test_thinking_and_unknown_blocks_are_ignored_without_raising():
    generation = _decode(
        "m",
        {
            "content": [
                {"type": "thinking", "thinking": "internal"},
                {"type": "something-new", "data": 1},
                {"type": "text", "text": "visible"},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 4},
        },
    )
    assert generation.text == "visible"
    assert generation.usage.total_tokens == 7


def test_message_carries_is_error_from_its_tool_result():
    message = Message.tool(ToolResult(call_id="c", name="search", content="boom", is_error=True))
    assert message.is_error is True
    assert message.to_dict()["is_error"] is True
    assert Message.user("x").to_dict().get("is_error") is None


# --- HTTP API hardening ----------------------------------------------------

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from atheneum.api.http import create_app  # noqa: E402
from atheneum.config import Config as _Config  # noqa: E402


def _client(tmp_path, name: str) -> TestClient:
    return TestClient(create_app(_Config(db=os.path.join(str(tmp_path), name))))


@pytest.mark.parametrize("source", ["trail\n", " pad ", "lead\nline", "tab\tsep"])
def test_source_with_whitespace_is_rejected(tmp_path, source: str):
    """MEDIUM: `$` matched before a trailing newline, so "trail\\n" was stored."""
    client = _client(tmp_path, "ws.db")
    with client:
        assert client.post("/documents", json={"source": source, "content": "body"}).status_code == 422


def test_streaming_endpoint_parameters_are_bounded(tmp_path):
    """MEDIUM: /ask/stream accepted max_turns=999999 while POST /ask did not."""
    client = _client(tmp_path, "stream.db")
    with client:
        assert client.get("/ask/stream", params={"query": "hi", "max_turns": 999999}).status_code == 422
        assert client.get("/ask/stream", params={"query": "hi", "top_k": 99999}).status_code == 422
        assert client.get("/ask/stream", params={"query": ""}).status_code == 422
        assert client.get("/ask/stream", params={"query": "hi", "max_turns": 4}).status_code == 200


def test_a_rejected_body_is_not_echoed_back(tmp_path):
    """LOW: the default handler reflected 5 MB of input in a 422 response."""
    client = _client(tmp_path, "echo.db")
    with client:
        response = client.post("/documents", json={"source": "big.md", "content": "x" * (5 * 1024 * 1024)})
        assert response.status_code == 422
        assert len(response.content) < 2000


def test_the_content_limit_is_bytes_not_characters(tmp_path):
    """LOW: 4.19M CJK characters is a 12.6 MB body and used to be accepted."""
    client = _client(tmp_path, "bytes.db")
    with client:
        response = client.post("/documents", json={"source": "cjk.md", "content": "中" * (2 * 1024 * 1024)})
        assert response.status_code == 422
        assert "bytes" in response.text


def test_interactive_docs_are_disabled_when_auth_is_on(tmp_path, monkeypatch):
    """LOW: /docs and /redoc were reachable unauthenticated with a token set."""
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "s3cret")
    client = _client(tmp_path, "docs.db")
    with client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/health").status_code == 200


def test_a_whitespace_only_token_does_not_fake_auth(tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_API_TOKEN", "   ")
    client = _client(tmp_path, "blank.db")
    with client:
        # /health no longer advertises the auth mode at all, so an anonymous
        # caller cannot learn whether a token is configured.
        assert "auth" not in client.get("/health").json()
        assert client.get("/stats").status_code == 200


# ---------------------------------------------------------------------------
# Round 8 — final-review findings: embedders, tools, providers, fusion, text
# ---------------------------------------------------------------------------


from atheneum.retrieval.embedders import (  # noqa: E402
    OllamaEmbedder,
    OpenAIEmbedder,
    _ordered_embeddings,
)
from atheneum.retrieval.rerank import LexicalOverlapReranker  # noqa: E402


def test_embeddings_are_reinserted_in_input_order_not_lexicographic():
    """BLOCKER: sorting on the raw index field ordered "10" before "2"."""
    items = [
        {"index": "3", "embedding": [3.0]},
        {"index": "0", "embedding": [0.0]},
        {"index": "2", "embedding": [2.0]},
        {"index": "1", "embedding": [1.0]},
    ]
    rows = _ordered_embeddings(items, ["a", "b", "c", "d"])
    assert [r[0] for r in rows] == [0.0, 1.0, 2.0, 3.0]


def test_a_missing_index_falls_back_to_position_not_zero():
    """Defaulting to 0 shifted every later row into the wrong slot."""
    items = [{"embedding": [99.0]}, {"embedding": [1.0]}]
    rows = _ordered_embeddings(items, ["a", "b"])
    assert [r[0] for r in rows] == [99.0, 1.0]


def test_an_ambiguous_index_raises_instead_of_picking_an_order():
    """A defaulted index that collides with an explicit one cannot be guessed."""
    items = [{"index": 1, "embedding": [1.0]}, {"embedding": [99.0]}]
    with pytest.raises(RuntimeError, match="index 1 twice"):
        _ordered_embeddings(items, ["a", "b"])


@pytest.mark.parametrize(
    "items",
    [
        [{"index": 0, "embedding": [1.0]}, {"index": 0, "embedding": [2.0]}],  # duplicate
        [{"index": 5, "embedding": [1.0]}],  # out of range
        [{"index": 0}],  # missing embedding key
        "notalist",  # wrong container
    ],
)
def test_malformed_embedding_payloads_raise_instead_of_misaligning(items: object):
    with pytest.raises(RuntimeError):
        _ordered_embeddings(items, ["a", "b"])


def test_network_embedders_keep_their_own_names():
    """MAJOR: a dataclass field named `name` clobbered the subclass attribute.

    Both reported name="http", so describe() could not distinguish backends and
    the index/embedder mismatch guard was defeated.
    """
    assert OpenAIEmbedder().name == "openai"
    assert OllamaEmbedder().name == "ollama"


def test_an_api_key_never_appears_in_a_repr():
    embedder = OpenAIEmbedder(api_key="sk-SECRET-VALUE-123")
    assert "sk-SECRET-VALUE-123" not in repr(embedder)


def test_a_timed_out_tool_releases_the_agent():
    """MAJOR: with-block shutdown(wait=True) made the timeout report but not help."""

    def sleeper() -> str:
        """Blocks for two seconds."""
        time.sleep(2.0)
        return "never"

    started = time.perf_counter()
    result = ToolRegistry([tool(sleeper)]).execute(
        ToolCall(id="c", name="sleeper", arguments={}), timeout=0.3
    )
    elapsed = time.perf_counter() - started
    assert result.is_error
    assert "timeout" in result.content
    assert elapsed < 1.0, f"a 0.3s timeout waited {elapsed:.2f}s"


def test_a_hanging_tool_stalls_the_whole_agent_run_without_a_timeout():
    from atheneum.agent.loop import Agent, AgentConfig
    from atheneum.providers.base import Generation, Provider

    class Once(Provider):
        name = "once"

        def __init__(self, gens: list[Generation]) -> None:
            self.gens = gens
            self.i = 0

        def complete(self, request: GenerationRequest) -> Generation:
            value = self.gens[min(self.i, len(self.gens) - 1)]
            self.i += 1
            return value

    def sleeper() -> str:
        """Blocks briefly."""
        time.sleep(0.4)
        return "x"

    gens = [
        Generation(text="", tool_calls=[ToolCall(id="c", name="sleeper", arguments={})], finish_reason="tool_calls"),
        Generation(text="done", finish_reason="stop"),
    ]
    started = time.perf_counter()
    run = Agent(Once(gens), ToolRegistry([tool(sleeper)]), config=AgentConfig(max_turns=3, tool_timeout=0.1)).run(
        "q"
    )
    assert time.perf_counter() - started < 0.35
    assert run.steps[0].results[0].is_error
    assert run.answer == "done"


def test_duplicate_tool_call_ids_are_executed_once():
    """MINOR: a repeated id ran twice and emitted duplicate tool_use_ids."""
    calls = [ToolCall(id="dup", name="noop", arguments={}), ToolCall(id="dup", name="noop", arguments={})]

    def noop() -> str:
        """Does nothing."""
        return "ok"

    from atheneum.agent.loop import Agent, AgentConfig
    from atheneum.providers.base import Generation, Provider

    class Once(Provider):
        name = "once"

        def __init__(self, gens: list[Generation]) -> None:
            self.gens = gens
            self.i = 0

        def complete(self, request: GenerationRequest) -> Generation:
            value = self.gens[min(self.i, len(self.gens) - 1)]
            self.i += 1
            return value

    gens = [
        Generation(text="", tool_calls=calls, finish_reason="tool_calls"),
        Generation(text="done", finish_reason="stop"),
    ]
    run = Agent(Once(gens), ToolRegistry([tool(noop)]), config=AgentConfig(max_turns=3)).run("q")
    assert len(run.steps[0].results) == 1


def test_read_source_requires_an_exact_or_unambiguous_filename():
    """MAJOR: a substring match served src/ab.py for read_source("b.py")."""
    corpus = Corpus.in_memory()
    corpus.add_text("src/ab.py", "ALPHA-DOC-CONTENT")
    corpus.add_text("src/b.py", "BRAVO-DOC-CONTENT")
    registry = build_corpus_tools(corpus)

    def call(source: str) -> dict:
        import json as _json

        return _json.loads(registry.execute(ToolCall(id="c", name="read_source", arguments={"source": source})).content)

    assert call("b.py")["source"] == "src/b.py"
    assert call("")["error"] == "invalid_arguments"
    assert call("nope")["error"] == "not_found"
    assert call("ab")["error"] == "not_found"
    corpus.add_text("other/b.py", "OTHER-B-DOC")
    ambiguous = call("b.py")
    assert ambiguous["error"] == "ambiguous_source"
    assert len(ambiguous["candidates"]) == 2
    corpus.close()


def test_search_reports_an_invalid_mode_as_a_tool_result():
    """MINOR: it raised ValueError out of the tool instead of describing the options."""
    import json as _json

    corpus = Corpus.in_memory()
    corpus.add_text("a.md", "token bucket rate limiting")
    registry = build_corpus_tools(corpus)
    payload = _json.loads(
        registry.execute(ToolCall(id="c", name="search", arguments={"query": "bucket", "mode": "bogus"})).content
    )
    assert payload["error"] == "invalid_arguments"
    assert "hybrid" in str(payload["message"])
    corpus.close()


def test_weighted_and_distribution_fusions_reject_duplicate_keys_too():
    """MINOR: only RRF had the fix; these still double-counted a repeated key."""
    from atheneum.retrieval.fusion import DBSFFusion, WeightedSumFusion

    weighted = WeightedSumFusion().fuse({"a": [("k", "A"), ("k", "A2")]}, scores={"a": [1.0, 3.0]})[0]
    assert weighted.score == pytest.approx(sum(weighted.contributions.values()))

    distributed = DBSFFusion().fuse(
        {"a": [("k", "A"), ("k", "A2")], "b": [("k", "B")]}, scores={"a": [1.0, 0.5], "b": [0.9]}
    )[0]
    assert distributed.score == pytest.approx(sum(distributed.contributions.values()))


def test_heavier_list_wins_the_payload_regardless_of_dict_order():
    from atheneum.retrieval.fusion import WeightedSumFusion

    first = WeightedSumFusion().fuse(
        {"a": [("k", "A")], "b": [("k", "B")]}, scores={"a": [1.0], "b": [3.0]}, weights=[0.9, 0.1]
    )[0]
    # Weights are positional by design, so the same logical weighting of "a"
    # means a reordered weight list once the declared order flips.
    second = WeightedSumFusion().fuse(
        {"b": [("k", "B")], "a": [("k", "A")]}, scores={"a": [1.0], "b": [3.0]}, weights=[0.1, 0.9]
    )[0]
    assert first.value == second.value == "A"


def test_rerankers_treat_a_non_positive_top_k_as_no_results():
    """MINOR: a negative top_k sliced from the end and returned results anyway."""
    reranker = LexicalOverlapReranker()
    docs = [("a", "token bucket rate limiting"), ("b", "unrelated prose")]
    assert reranker.rerank("token bucket", docs, -1) == []
    assert reranker.rerank("token bucket", docs, 0) == []


def test_paragraph_breaks_are_a_real_splitting_level():
    """The documented cascade claimed paragraphs; PARAGRAPH_SEPARATORS was dead."""
    from atheneum.text.splitter import _split_paragraphs

    parts = _split_paragraphs("One two three.\n\nFour five six.\n\n\nSeven eight.")
    assert parts == ["One two three.", "Four five six.", "Seven eight."]


def test_chunking_still_loses_nothing_with_paragraph_packing():
    import collections

    text = ("Para one has content.\n\n" * 30) + ("x" * 3000) + "\n\nTail para."
    chunks = split_text(text, SplitterConfig(chunk_size=120, chunk_overlap=20))
    before = collections.Counter(c for c in text if not c.isspace())
    after = collections.Counter(c for c in "".join(chunks) if not c.isspace())
    assert all(count <= after.get(char, 0) for char, count in before.items())


def test_cjk_stopwords_are_applied_to_unigrams_only():
    from atheneum.text.tokenizer import tokenize

    tokens = tokenize("我的队列")
    assert "队列" in tokens
    assert "的" not in tokens
    assert "我的" in tokens  # bigrams survive; meaning lives in the pair


def test_giant_title_and_metadata_are_rejected():
    """LOW: only `content` was size-capped; one request grew the db to 34 MB."""
    client = _client(tempfile.mkdtemp(), "fields.db")
    with client:
        big = "T" * 8_000_000
        assert client.post("/documents", json={"source": "u.md", "content": "x", "title": big}).status_code == 422
        assert (
            client.post(
                "/documents", json={"source": "u.md", "content": "x", "metadata": {"k": big}}
            ).status_code
            == 422
        )
        assert client.get("/health").json().get("auth") is None


# ---------------------------------------------------------------------------
# Round 9 — concurrency, embedding validation, provider pairing
# ---------------------------------------------------------------------------

from atheneum.index.vectors import DimensionMismatchError  # noqa: E402
from atheneum.providers.anthropic import _validate_pairing  # noqa: E402
from atheneum.retrieval.embedders import (  # noqa: E402
    _validate_matrix,
    describe,
)


def test_a_published_index_is_never_mutated_under_a_reader(tmp_path):
    """BLOCKER: search + reconfigure raced and misattributed results."""
    corpus = Corpus.open(str(tmp_path / "race.db"))
    for i in range(60):
        corpus.add_text(f"w{i}.md", f"alpha{i} beta{i} gamma document {i} with body text")

    errors: list[str] = []
    misattributed: list[str] = []

    def reader() -> None:
        try:
            for _ in range(120):
                for hit in corpus.search("alpha7 beta7", top_k=3):
                    stored = corpus.get_chunk(hit.chunk.id)
                    if stored is None or stored.text != hit.text:
                        misattributed.append(hit.chunk.source)
        except BaseException as exc:
            errors.append(f"{type(exc).__name__}: {exc}")

    def writer() -> None:
        try:
            for i in range(40):
                corpus.add_text(f"x{i}.md", f"alpha{i} inserted material {i} during searching")
                corpus.configure(fusion_k=61 + i % 5)
        except BaseException as exc:
            errors.append(f"writer {type(exc).__name__}: {exc}")

    threads = [threading.Thread(target=reader) for _ in range(4)]
    threads += [threading.Thread(target=writer) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=45)

    assert not [t for t in threads if t.is_alive()], "a thread hung"
    assert not errors, errors[:2]
    assert not misattributed, f"hits attributed to the wrong chunk: {misattributed[:3]}"
    rows, bm25, vectors = corpus._pub
    assert len(rows) == len(bm25) == len(vectors) == corpus.store.chunk_count()
    corpus.close()


def test_a_store_transaction_plus_invalidate_does_not_deadlock_a_searcher(tmp_path):
    """BLOCKER: the refresh lock and the Store lock were taken in opposite orders."""
    corpus = Corpus.open(str(tmp_path / "dl.db"))
    corpus.add_text("a.md", "alpha beta content for deadlock probing")
    corpus.search("alpha", top_k=2)
    outcome: dict[str, str] = {}

    def writer() -> None:
        try:
            for _ in range(60):
                with corpus.store.transaction():
                    corpus.store.set_meta("k", "v")
                    corpus.invalidate()
            outcome["A"] = "done"
        except BaseException as exc:
            outcome["A"] = f"{type(exc).__name__}: {exc}"

    def reader() -> None:
        try:
            for _ in range(600):
                corpus.search("alpha beta", top_k=2)
            outcome["B"] = "done"
        except BaseException as exc:
            outcome["B"] = f"{type(exc).__name__}: {exc}"

    a = threading.Thread(target=writer)
    b = threading.Thread(target=reader)
    a.start()
    b.start()
    a.join(timeout=25)
    b.join(timeout=25)
    assert not a.is_alive() and not b.is_alive(), f"DEADLOCK: {outcome}"
    assert outcome.get("A") == "done", outcome.get("A")
    assert outcome.get("B") == "done", outcome.get("B")
    corpus.close()


def test_reconfigure_is_not_lost_by_an_in_flight_reload(tmp_path):
    """BLOCKER: an unlocked configure() left bm25 empty while rows were stored."""
    corpus = Corpus.open(str(tmp_path / "cfg.db"))
    for i in range(30):
        corpus.add_text(f"d{i}.md", f"unique{i} filler words for the index")
    corpus.search("unique1", top_k=2)
    for i in range(25):
        corpus.configure(fusion_k=61 + i)
        corpus.add_text(f"l{i}.md", f"later{i} content added after reconfiguring")
    corpus.search("later5", top_k=2)
    rows, bm25, _vectors = corpus._pub
    assert len(bm25) == len(rows) == corpus.store.chunk_count()
    assert bm25.search("later5", top_k=3), "BM25 index lost the newest chunks"
    corpus.close()


@pytest.mark.parametrize(
    "rows,texts,dim,fragment",
    [
        ([[1.0, 2.0]], ["a", "b"], 2, "1 vectors for 2 inputs"),
        ([[1.0, 2.0], [1.0]], ["a", "b"], 2, "width 1"),
        ([[float("nan"), 2.0], [1.0, 2.0]], ["a", "b"], 2, "non-finite"),
        ([[1.0, 2.0, 3.0], [1.0, 2.0]], ["a", "b"], 2, "width 3"),
        ("notalist", ["a"], 2, "instead of a list"),
    ],
)
def test_malformed_embedding_matrices_are_rejected(rows, texts, dim, fragment):
    """BLOCKER (Ollama path): no count or shape check attached wrong vectors."""
    with pytest.raises(RuntimeError, match=fragment):
        _validate_matrix(rows, texts, dim)


def test_a_float_index_is_not_truncated_onto_the_wrong_row():
    with pytest.raises(RuntimeError, match="non-integer index"):
        _ordered_embeddings(
            [{"index": 1.9, "embedding": [1.0]}, {"index": 0, "embedding": [0.0]}], ["a", "b"]
        )


def test_describe_distinguishes_models_at_the_same_dimension():
    """MAJOR: a model swap used to look identical, defeating the mismatch guard."""
    a = OpenAIEmbedder(model="text-embedding-3-small", dim=768)
    b = OpenAIEmbedder(model="something-else", dim=768, base_url="http://vllm:8000/v1")
    assert describe(a) != describe(b)


def test_load_normalizes_and_refuses_a_foreign_dimension():
    """MAJOR: load() stored raw blobs, so dot product stopped being cosine."""
    index = VectorIndex(dim=2)
    index.load([[3.0, 4.0], [0.0, 2.0]], dim=2)
    # A (3,4) row loaded raw would score 3.0 against (1,0); normalized it is 0.6.
    assert index.similarity_to_row(0, [1.0, 0.0]) == pytest.approx(0.6, abs=1e-5)
    assert max(s for _, s in index.search([1.0, 0.0], top_k=2)) <= 1.0 + 1e-6
    with pytest.raises(DimensionMismatchError):
        VectorIndex(dim=4).load([[1.0, 2.0, 3.0]], dim=3)


def test_non_finite_vectors_cannot_be_indexed():
    index = VectorIndex(dim=2)
    with pytest.raises(ValueError, match="finite"):
        index.add([float("nan"), 1.0])
    with pytest.raises(ValueError, match="finite"):
        index.add([float("inf"), 1.0])
    assert len(index) == 0


@pytest.mark.parametrize(
    "messages,fragment",
    [
        (
            [
                Message.assistant(
                    "", [ToolCall(id="t1", name="a", arguments={}), ToolCall(id="t2", name="b", arguments={})]
                ),
                Message.tool(ToolResult(call_id="t1", name="a", content="r")),
            ],
            "never answered",
        ),
        ([Message.tool(ToolResult(call_id="nope", name="a", content="r"))], "no matching tool call"),
        (
            [
                Message.assistant("", [ToolCall(id="", name="a", arguments={})]),
                Message.tool(ToolResult(call_id="", name="a", content="r")),
            ],
            "empty id",
        ),
    ],
)
def test_inconsistent_tool_pairing_is_rejected_locally(messages, fragment):
    """MAJOR: these serialized fine and failed only as a remote HTTP 400."""
    with pytest.raises(ProviderError, match=fragment):
        _validate_pairing(messages)


def test_a_well_formed_pairing_is_accepted():
    _validate_pairing(
        [
            Message.assistant("", [ToolCall(id="t1", name="a", arguments={})]),
            Message.tool(ToolResult(call_id="t1", name="a", content="r")),
        ]
    )


def test_a_non_positive_result_limit_cannot_disable_truncation():
    """LOW: `result_limit=0` meant "no limit" and returned the whole payload."""
    def big() -> str:
        """Returns a very large string."""
        return "y" * 50_000

    registry = ToolRegistry([tool(big)])
    for bad in (0, -1):
        content = registry.execute(
            ToolCall(id="c", name="big", arguments={}), result_limit=bad
        ).content
        assert len(content) < 30_000, f"result_limit={bad} disabled the cap"
    with pytest.raises(ValueError, match="result_limit must be positive"):
        AgentConfig(result_limit=0)


def test_approval_is_enforced_by_the_registry_not_only_the_loop():
    """MAJOR: requires_approval was a tool property only one caller honoured."""
    ran: list[str] = []

    def wipe() -> str:
        """Destructive."""
        ran.append("ran")
        return "wiped"

    registry = ToolRegistry([tool(wipe, name="wipe", requires_approval=True)])
    direct = registry.execute(ToolCall(id="c", name="wipe", arguments={}))
    assert direct.is_error and not ran, "execute() ran an approval-required tool"
    assert json.loads(direct.content)["error"] == "not_approved"

    def broken(_call: ToolCall, _tool: Tool) -> bool:
        raise RuntimeError("callback exploded")

    assert registry.execute(
        ToolCall(id="c", name="wipe", arguments={}), approver=broken
    ).is_error
    assert not ran, "a raising approver must decline, not approve"
    assert registry.execute(
        ToolCall(id="c", name="wipe", arguments={}), approver=lambda _c, _t: True
    ).content == "wiped"


# ---------------------------------------------------------------------------
# Round 10 — embedding payload validation and approval ordering
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rows,texts,dim",
    [
        ([[1.0, 2.0], [3.0, 4.0]], ["a", "b"], 0),      # zero width
        ([[1.0, 2.0, 3.0]], ["a"], 2),                  # too wide
        ([[1.0, 2.0]], ["a", "b", "c"], 2),             # too few
    ],
)
def test_validate_matrix_rejects_unusable_shapes(rows, texts, dim):
    """A zero-width matrix passes every other check and then matches nothing."""
    with pytest.raises(RuntimeError):
        _validate_matrix(rows, texts, dim)


@pytest.mark.parametrize("row", [[1.0, 2.0], (1.0, 2.0), np.array([1.0, 2.0], dtype=np.float32)])
def test_validate_matrix_accepts_any_sequence_row(row):
    """Accepting a tuple but rejecting a numpy row made the guard deserialization-dependent."""
    matrix = _validate_matrix([row, [3.0, 4.0]], ["a", "b"], 2)
    assert matrix.shape == (2, 2)
    assert np.isfinite(matrix).all()


@pytest.mark.parametrize("item", [{"index": 0, "embedding": None}, {"index": 0}, {"index": 0, "embedding": "x"}])
def test_a_missing_or_null_embedding_is_a_clean_error(item):
    """`embedding: None` used to raise a bare TypeError from a subscript."""
    with pytest.raises(RuntimeError, match="malformed item"):
        _ordered_embeddings([item], ["a"])


def test_approval_is_decided_before_arguments_are_processed():
    """A refused call must not report invalid_arguments.

    Coercing first told the model to fix arguments for an operation it was never
    allowed to perform.
    """
    def destruct(unused: int = 0) -> str:
        """Destructive."""
        return "ran"

    registry = ToolRegistry([tool(destruct, name="destruct", requires_approval=True)])
    bad = ToolCall(id="c", name="destruct", arguments={"bogus": 1})

    assert json.loads(registry.execute(bad).content)["error"] == "not_approved"
    assert json.loads(registry.execute(bad, approver=lambda _c, _d: False).content)["error"] == "not_approved"
    # Once approved, normal validation applies.
    assert json.loads(registry.execute(bad, approver=lambda _c, _d: True).content)["error"] == "invalid_arguments"
    good = ToolCall(id="c", name="destruct", arguments={})
    assert registry.execute(good, approver=lambda _c, _d: True).content == "ran"


# ---------------------------------------------------------------------------
# Round 11 — credential exclusion and serialisation cost
# ---------------------------------------------------------------------------

from atheneum.ingest import is_sensitive_name  # noqa: E402


@pytest.mark.parametrize(
    "name",
    [
        ".env", ".ENV", "Env", "env", ".env2", ".env.local", ".env.production",
        "config.env", "app.env.sample", ".env.production.local", "secrets.yaml", "secret.json",
        "id_ed25519", "id_rsa", "id_ecdsa", "credentials", "credentials.json",
        ".npmrc", ".netrc", ".pgpass", "server.pem", "tls.key", "bundle.p12",
        "signing.asc", "vault.kdbx", "private_key.txt", "keystore.jks", "token",
    ],
)
def test_credential_shaped_filenames_are_not_indexed(tmp_path, name: str):
    """Exact-name matching let .env2, config.env, id_ed25519 and secrets.yaml through.

    Indexed credentials land in SQLite and become retrievable, so a model can
    quote a private key back in an answer.
    """
    assert is_sensitive_name(name), name
    (tmp_path / name).write_text("SECRET=1", encoding="utf-8")
    (tmp_path / "keep.md").write_text("legitimate prose", encoding="utf-8")
    assert {p.name for p in discover_files(tmp_path)} == {"keep.md"}


@pytest.mark.parametrize(
    "name", ["notes.md", "README.md", "deploy.yml", "environment.md", "environ.py", "envoy.yaml", "main.go"]
)
def test_ordinary_filenames_are_not_over_blocked(tmp_path, name: str):
    """The other failure mode: excluding everything that merely contains 'env'."""
    assert not is_sensitive_name(name), name
    (tmp_path / name).write_text("legitimate content here", encoding="utf-8")
    assert {p.name for p in discover_files(tmp_path)} == {name}


def test_symlinked_files_do_not_escape_even_when_directories_are_pruned(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.md").write_text("TOP-SECRET", encoding="utf-8")
    root = tmp_path / "root"
    root.mkdir()
    (root / "real.md").write_text("legit", encoding="utf-8")
    (root / "link.md").symlink_to(outside / "secret.md")
    (root / "dirlink").symlink_to(outside)
    assert {p.name for p in discover_files(root)} == {"real.md"}
    # Opting in is explicit and does follow, which is the documented behaviour.
    followed = {p.name for p in discover_files(root, follow_symlinks=True)}
    assert "link.md" in followed


def test_bounding_a_collection_caps_work_not_just_output():
    """2000 elements of 200 KB each serialized in full before truncation."""
    def nested() -> object:
        """Returns nested collections."""
        return [{"k": "v" * 200_000} for _ in range(2000)]

    started = time.perf_counter()
    result = ToolRegistry([tool(nested)]).execute(ToolCall(id="c", name="nested", arguments={}))
    elapsed = time.perf_counter() - started
    assert not result.is_error
    assert len(result.content) < 30_000
    assert "long values truncated" in result.content
    assert elapsed < 0.5, f"serialisation still costs {elapsed:.2f}s"


def test_the_truncation_note_is_not_repeated_per_element():
    def many() -> list:
        """Returns many long strings."""
        return ["x" * 50_000 for _ in range(50)]

    content = ToolRegistry([tool(many)]).execute(ToolCall(id="c", name="many", arguments={})).content
    assert content.count("long values truncated") == 1
