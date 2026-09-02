from __future__ import annotations

import json

import pytest

import atheneum
from atheneum.agent.builtin_tools import build_corpus_tools
from atheneum.agent.loop import Agent, AgentConfig, AgentRun
from atheneum.core.types import Message, Role, ToolCall, ToolResult
from atheneum.providers.base import Generation, GenerationRequest, Provider, Usage
from atheneum.providers.offline import Evidence, OfflineProvider, parse_evidence


def _evidence_message(source: str, ordinal: int, text: str, score: float = 0.5) -> Message:
    payload = {"results": [{"source": source, "ordinal": ordinal, "text": text, "score": score}]}
    return Message.tool(ToolResult(call_id="c1", name="search", content=json.dumps(payload)))


# -- turn one: it should ask for retrieval ----------------------------------
def test_first_turn_requests_a_search(provider: OfflineProvider):
    request = GenerationRequest(
        messages=[Message.system("s"), Message.user("what is fusion")],
        tools={"search": {"description": "", "parameters": {"type": "object", "properties": {}}}},
    )
    generation = provider.complete(request)
    assert generation.finish_reason == "tool_calls"
    assert len(generation.tool_calls) == 1
    assert generation.tool_calls[0].name == "search"
    assert generation.tool_calls[0].arguments == {"query": "what is fusion"}


def test_tool_call_ids_are_stable_across_runs(provider: OfflineProvider):
    def ids() -> list[str]:
        request = GenerationRequest(messages=[Message.user("same question")], tools={"search": {}})
        return [c.id for c in provider.complete(request).tool_calls]

    assert ids() == ids()


def test_no_search_tool_means_no_tool_call(provider: OfflineProvider):
    request = GenerationRequest(messages=[Message.user("hello")], tools={})
    generation = provider.complete(request)
    assert generation.tool_calls == []
    assert generation.finish_reason == "stop"


# -- evidence parsing -------------------------------------------------------
def test_parse_evidence_reads_search_results():
    evidence = parse_evidence([_evidence_message("a.md", 0, "First passage.", 0.9)])
    assert len(evidence) == 1
    assert evidence[0].source == "a.md"
    assert evidence[0].score == pytest.approx(0.9)


def test_parse_evidence_sorts_by_score():
    evidence = parse_evidence(
        [
            _evidence_message("low.md", 0, "Low scoring passage.", 0.1),
            _evidence_message("high.md", 0, "High scoring passage.", 0.9),
        ]
    )
    assert [e.source for e in evidence] == ["high.md", "low.md"]
    assert [e.rank for e in evidence] == [0, 1]


def test_parse_evidence_ignores_non_json():
    message = Message.tool(ToolResult(call_id="c", name="search", content="not json at all"))
    assert parse_evidence([message]) == []


def test_parse_evidence_ignores_other_tools():
    message = Message(
        role=Role.TOOL,
        name="weather",
        content=json.dumps({"results": [{"source": "x", "text": "sunny"}]}),
    )
    assert parse_evidence([message]) == []


def test_parse_evidence_skips_blank_passages():
    payload = {"results": [{"source": "a.md", "text": "   "}, {"source": "b.md", "text": "real"}]}
    evidence = parse_evidence([Message.tool(ToolResult(call_id="c", name="search", content=json.dumps(payload)))])
    assert [e.source for e in evidence] == ["b.md"]


def test_parse_evidence_accepts_content_alias():
    payload = {"results": [{"source": "a.md", "content": "Alternative key."}]}
    evidence = parse_evidence([Message.tool(ToolResult(call_id="c", name="search", content=json.dumps(payload)))])
    assert evidence[0].text == "Alternative key."


# -- answer composition -----------------------------------------------------
def test_answer_includes_sentences_from_evidence(provider: OfflineProvider):
    evidence = [
        Evidence(source="a.md", ordinal=0, text="Reciprocal rank fusion merges ranked lists.", rank=0, score=0.9),
    ]
    answer = provider.compose("what is fusion", evidence)
    assert "Reciprocal rank fusion merges ranked lists." in answer


def test_answer_lists_sources_with_coordinates(provider: OfflineProvider):
    evidence = [
        Evidence(source="notes.md", ordinal=3, text="The k1 parameter controls saturation of term frequency.", rank=0, score=0.8),
    ]
    answer = provider.compose("what controls saturation", evidence)
    assert "Sources:" in answer
    assert "notes.md" in answer
    assert "chunk 3" in answer
    assert "[1]" in answer


def test_citations_are_numbered_from_one(provider: OfflineProvider):
    evidence = [
        Evidence(source="a.md", ordinal=0, text="First relevant sentence about fusion.", rank=0, score=0.9),
        Evidence(source="b.md", ordinal=1, text="Second relevant sentence about fusion.", rank=1, score=0.7),
    ]
    answer = provider.compose("fusion", evidence)
    assert "[1]" in answer and "[2]" in answer
    assert "[0]" not in answer


def test_answer_is_deterministic(provider: OfflineProvider):
    evidence = [
        Evidence(source="a.md", ordinal=0, text="Alpha bravo charlie about ranking.", rank=0, score=0.9),
        Evidence(source="b.md", ordinal=2, text="Delta echo foxtrot about ranking.", rank=1, score=0.5),
    ]
    assert provider.compose("ranking", evidence) == provider.compose("ranking", evidence)


def test_duplicate_sentences_are_not_repeated(provider: OfflineProvider):
    text = "This exact sentence appears in both passages."
    evidence = [
        Evidence(source="a.md", ordinal=0, text=text, rank=0, score=0.9),
        Evidence(source="b.md", ordinal=0, text=text, rank=1, score=0.8),
    ]
    answer = provider.compose("sentence", evidence)
    assert answer.count(text) == 1


def test_sentences_are_ordered_by_passage_not_by_score():
    provider = OfflineProvider(max_sentences=3)
    evidence = [
        Evidence(source="a.md", ordinal=0, text="Alpha one is relevant here about fusion.", rank=0, score=0.9),
        Evidence(source="a.md", ordinal=0, text="Alpha two is relevant here about fusion.", rank=0, score=0.9),
        Evidence(source="a.md", ordinal=0, text="Alpha three is relevant here about fusion.", rank=0, score=0.9),
    ]
    answer = provider.compose("fusion", evidence)
    positions = [answer.index(s) for s in ("Alpha one", "Alpha two", "Alpha three")]
    assert positions == sorted(positions)


def test_no_evidence_gives_the_fallback_note():
    provider = OfflineProvider(fallback_note="NOTHING FOUND")
    assert provider.compose("anything", []) == "NOTHING FOUND"


def test_unmatched_sentences_fall_back_to_leading_text():
    provider = OfflineProvider()
    evidence = [
        Evidence(source="a.md", ordinal=0, text="A long passage with plenty of content but no query terms.", rank=0, score=0.3),
    ]
    answer = provider.compose("zzzzzqqqq", evidence)
    assert "A long passage" in answer


def test_max_sentences_is_respected():
    provider = OfflineProvider(max_sentences=2)
    evidence = [
        Evidence(source=f"{c}.md", ordinal=0, text=f"Sentence number {i} explains fusion clearly.", rank=i, score=1.0 - i / 10)
        for i, c in enumerate("abcdefgh")
    ]
    answer = provider.compose("fusion", evidence)
    body = answer.split("Sources:")[0]
    assert body.count("[") == 2


def test_usage_is_reported(provider: OfflineProvider):
    request = GenerationRequest(messages=[_evidence_message("a.md", 0, "Fusion merges lists."), Message.user("fusion")])
    generation = provider.complete(request)
    assert generation.usage.total_tokens > 0
    assert generation.model == "offline"


# -- the offline provider as a full agent -----------------------------------
@pytest.fixture
def provider() -> OfflineProvider:
    return OfflineProvider()


@pytest.fixture
def agent_corpus() -> atheneum.Corpus:
    corpus = atheneum.Corpus.in_memory()
    corpus.add_text(
        "docs/fusion.md",
        "# Fusion\n\nReciprocal rank fusion merges several ranked lists by awarding one over a constant "
        "plus the rank. The constant sixty was recommended by Cormack and colleagues at SIGIR 2009.",
    )
    corpus.add_text(
        "docs/bm25.md",
        "# BM25\n\nOkapi BM25 controls term frequency saturation with the k1 parameter. The b parameter "
        "controls length normalization between zero and one.",
    )
    return corpus


def test_agent_with_offline_provider_produces_a_cited_answer(agent_corpus: atheneum.Corpus):
    agent = Agent(OfflineProvider(), build_corpus_tools(agent_corpus), config=AgentConfig(max_turns=4))
    run = agent.run("what does the k1 parameter control")
    assert run.stopped_reason == "final_answer"
    assert run.turns == 2
    assert run.tool_call_count == 1
    assert "k1 parameter" in run.answer or "k1" in run.answer
    assert "docs/bm25.md" in run.answer
    assert "[1]" in run.answer


def test_agent_run_exposes_evidence(agent_corpus: atheneum.Corpus):
    agent = Agent(OfflineProvider(), build_corpus_tools(agent_corpus))
    run = agent.run("what is rank fusion")
    assert run.evidence
    payload = json.loads(run.evidence[0].content)
    assert payload["results"]


def test_agent_reports_no_results_for_an_empty_corpus():
    corpus = atheneum.Corpus.in_memory()
    agent = Agent(OfflineProvider(), build_corpus_tools(corpus))
    run = agent.run("anything at all")
    assert run.stopped_reason == "final_answer"
    assert "No indexed material matched" in run.answer


def test_two_identical_runs_are_byte_identical(agent_corpus: atheneum.Corpus):
    def once() -> str:
        agent = Agent(OfflineProvider(), build_corpus_tools(agent_corpus))
        return agent.run("what is length normalization").answer

    assert once() == once()


def test_agent_stream_emits_text_and_done(agent_corpus: atheneum.Corpus):
    agent = Agent(OfflineProvider(), build_corpus_tools(agent_corpus))
    events = list(agent.stream("what is rank fusion"))
    kinds = {event.kind for event in events}
    assert "text" in kinds
    assert events[-1].kind == "done"
    assert isinstance(events[-1].run, AgentRun)
    assert events[-1].run.answer


def test_stream_and_run_agree_on_the_answer(agent_corpus: atheneum.Corpus):
    tools = build_corpus_tools(agent_corpus)
    streamed = "".join(
        event.text for event in Agent(OfflineProvider(), tools).stream("what is fusion") if event.kind == "text"
    )
    direct = Agent(OfflineProvider(), build_corpus_tools(agent_corpus)).run("what is fusion").answer
    assert streamed.replace(" ", "").startswith(direct.replace(" ", "")[:40])


# -- a scripted provider, to exercise loop mechanics -------------------------
class ScriptedProvider(Provider):
    """Stands in for a hosted model so loop behaviour can be tested offline."""

    name = "scripted"

    def __init__(self, generations: list[Generation]) -> None:
        self.generations = generations
        self.calls = 0

    def complete(self, request: GenerationRequest) -> Generation:
        if self.calls >= len(self.generations):
            return Generation(text="default tail")
        generation = self.generations[self.calls]
        self.calls += 1
        return generation


def test_loop_terminates_at_max_turns():
    loop_forever = Generation(text="", tool_calls=[ToolCall(id="c", name="noop", arguments={})], finish_reason="tool_calls")
    provider = ScriptedProvider([loop_forever] * 20)
    registry = build_registry_with_noop()
    agent = Agent(provider, registry, config=AgentConfig(max_turns=3))
    run = agent.run("go")
    assert run.stopped_reason == "max_turns"
    assert run.turns == 3
    assert provider.calls == 3


def build_registry_with_noop():
    from atheneum.agent.tools import ToolRegistry, tool

    return ToolRegistry([tool(lambda: "ok", name="noop")])


def test_direct_answer_takes_one_turn():
    provider = ScriptedProvider([Generation(text="done", finish_reason="stop")])
    run = Agent(provider, build_registry_with_noop()).run("hi")
    assert run.answer == "done"
    assert run.turns == 1
    assert run.stopped_reason == "final_answer"


def test_tool_error_is_fed_back_not_raised():
    calls = [
        Generation(text="", tool_calls=[ToolCall(id="c1", name="missing", arguments={})], finish_reason="tool_calls"),
        Generation(text="recovered after the error", finish_reason="stop"),
    ]
    provider = ScriptedProvider(calls)
    run = Agent(provider, build_registry_with_noop()).run("go")
    assert run.answer == "recovered after the error"
    assert run.steps[0].results[0].is_error is True
    assert "unknown_tool" in run.steps[0].results[0].content


def test_usage_accumulates_across_turns():
    calls = [
        Generation(text="", tool_calls=[ToolCall(id="c", name="noop", arguments={})], finish_reason="tool_calls", usage=Usage(10, 5)),
        Generation(text="final", finish_reason="stop", usage=Usage(20, 7)),
    ]
    run = Agent(ScriptedProvider(calls), build_registry_with_noop()).run("go")
    assert run.usage == Usage(30, 12)


def test_history_is_passed_into_the_next_request():
    seen: list[int] = []

    class Recording(ScriptedProvider):
        def complete(self, request: GenerationRequest) -> Generation:
            seen.append(len(request.messages))
            return super().complete(request)

    provider = Recording([
        Generation(text="", tool_calls=[ToolCall(id="c", name="noop", arguments={})], finish_reason="tool_calls"),
        Generation(text="done"),
    ])
    Agent(provider, build_registry_with_noop()).run("go")
    assert seen[1] > seen[0]


def test_previous_history_is_included():
    provider = ScriptedProvider([Generation(text="answered")])
    prior = [Message.user("earlier question"), Message.assistant("earlier answer")]
    run = Agent(provider, build_registry_with_noop()).run("follow up", history=prior)
    assert any("earlier question" in m.content for m in run.messages)


def test_run_as_dict_is_serializable():
    provider = ScriptedProvider([Generation(text="answered")])
    run = Agent(provider, build_registry_with_noop()).run("q")
    payload = json.loads(json.dumps(run.as_dict()))
    assert payload["answer"] == "answered"
    assert payload["stopped_reason"] == "final_answer"


def test_provider_failure_is_reported_not_raised():
    from atheneum.providers.base import ProviderError

    class Failing(Provider):
        name = "failing"

        def complete(self, request: GenerationRequest) -> Generation:
            raise ProviderError("upstream exploded", status=500)

    run = Agent(Failing(), build_registry_with_noop()).run("q")
    assert run.stopped_reason == "provider_error"
    assert "upstream exploded" in run.error
    assert run.ok is False


def test_tool_request_without_any_tools_is_reported():
    from atheneum.agent.tools import ToolRegistry

    provider = ScriptedProvider([
        Generation(text="", tool_calls=[ToolCall(id="c", name="search", arguments={})], finish_reason="tool_calls")
    ])
    run = Agent(provider, ToolRegistry()).run("q")
    assert run.stopped_reason == "no_tools"
    assert "none are registered" in run.error


def test_approval_is_sought_for_destructive_tools():
    from atheneum.agent.tools import ToolRegistry, tool

    registry = ToolRegistry([tool(lambda: "ran", name="danger", requires_approval=True)])
    provider = ScriptedProvider([
        Generation(text="", tool_calls=[ToolCall(id="c", name="danger", arguments={})], finish_reason="tool_calls"),
        Generation(text="done"),
    ])
    # No confirm callback configured: the tool must not run.
    run = Agent(provider, registry).run("q")
    assert run.steps[0].results[0].is_error
    assert "not_approved" in run.steps[0].results[0].content


def test_approval_callback_can_allow_a_tool():
    from atheneum.agent.tools import ToolRegistry, tool

    registry = ToolRegistry([tool(lambda: "ran", name="danger", requires_approval=True)])
    provider = ScriptedProvider([
        Generation(text="", tool_calls=[ToolCall(id="c", name="danger", arguments={})], finish_reason="tool_calls"),
        Generation(text="done"),
    ])
    run = Agent(provider, registry, confirm=lambda call, definition: True).run("q")
    assert run.steps[0].results[0].is_error is False
    assert run.steps[0].results[0].content == "ran"


def test_a_raising_approval_callback_fails_closed():
    from atheneum.agent.tools import ToolRegistry, tool

    def explode(call, definition):
        raise RuntimeError("callback is broken")

    registry = ToolRegistry([tool(lambda: "ran", name="danger", requires_approval=True)])
    provider = ScriptedProvider([
        Generation(text="", tool_calls=[ToolCall(id="c", name="danger", arguments={})], finish_reason="tool_calls"),
        Generation(text="done"),
    ])
    run = Agent(provider, registry, confirm=explode).run("q")
    assert "not_approved" in run.steps[0].results[0].content


@pytest.mark.parametrize("turns", [0, -1])
def test_invalid_max_turns_rejected(turns: int):
    with pytest.raises(ValueError):
        AgentConfig(max_turns=turns)


def test_run_tool_call_count_counts_every_turn():
    calls = [
        Generation(text="", tool_calls=[ToolCall(id="a", name="noop", arguments={})], finish_reason="tool_calls"),
        Generation(text="", tool_calls=[ToolCall(id="b", name="noop", arguments={})], finish_reason="tool_calls"),
        Generation(text="done"),
    ]
    run = Agent(ScriptedProvider(calls), build_registry_with_noop()).run("q")
    assert run.tool_call_count == 2
    assert run.turns == 3
