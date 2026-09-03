"""Regressions from adversarial code review.

Every test here pins a defect that a reviewer reproduced and that the rest of the
suite did not catch. They are grouped by the review round that found them so the
provenance stays readable.
"""

from __future__ import annotations

import asyncio
import collections
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from atheneum.agent.loop import Agent, AgentConfig
from atheneum.agent.tools import ToolRegistry, tool
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
    assert result.is_error
    assert "could not be serialised" in result.content


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
