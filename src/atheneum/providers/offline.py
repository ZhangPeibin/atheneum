"""The built-in offline provider.

This is not a language model and does not pretend to be one. It is a
deterministic extractive engine that speaks the same protocol as a real model:
it asks for tool calls, reads their results, and writes an answer with inline
citations.

Shipping it as a first-class provider rather than a test fixture is deliberate.
It means ``pip install atheneum && ath ask "..."`` produces a real, cited
answer with no API key, no download and no network — so the product is
demonstrable on sight, and the entire agent loop is testable in CI on a machine
with nothing configured.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from atheneum.core.types import Message, ToolCall
from atheneum.providers.base import Generation, GenerationRequest, Provider, StreamEvent, Usage
from atheneum.text.tokenizer import tokenize

__all__ = ["Evidence", "OfflineProvider", "parse_evidence"]

_SENTENCE_RE = re.compile(r"(?<=[.!?。！？;；\n])\s+")

# Heading markers and fence lines are document structure. Quoting "## Decision" as
# if it were an answer sentence is noise, so they are filtered out before scoring.
_STRUCTURAL_RE = re.compile(r"^\s*(?:#{1,6}\s|```|[-=]{3,}$|>\s)")

# The tool names whose results carry retrievable evidence. Kept as a tuple so an
# integrator adding a new search tool can extend it explicitly.
EVIDENCE_TOOLS = ("search", "atheneum_search", "lookup", "read")


@dataclass(frozen=True, slots=True)
class Evidence:
    """One retrieved passage, as the provider sees it."""

    source: str
    ordinal: int
    text: str
    rank: int
    score: float
    chunk_id: str = ""


class OfflineProvider(Provider):
    """Deterministic extractive answering over retrieved evidence.

    Behaviour is a function of the input alone: no randomness, no clock, no
    network. Two runs on the same corpus and question return byte-identical
    answers, which is what makes it usable as a regression baseline.
    """

    name = "offline"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        *,
        search_tool: str = "search",
        max_sentences: int = 6,
        max_evidence: int = 8,
        min_sentence_chars: int = 24,
        fallback_note: str = (
            "No indexed material matched the question. Ingest documents with "
            "`ath index PATH...` and ask again, or answer with a generative model "
            "via `ath ask -m openai \"...\"`."
        ),
    ) -> None:
        self.search_tool = search_tool
        self.max_sentences = max_sentences
        self.max_evidence = max_evidence
        self.min_sentence_chars = min_sentence_chars
        self.fallback_note = fallback_note

    # -- provider interface -------------------------------------------------
    def complete(self, request: GenerationRequest) -> Generation:
        query = request.last_user_message()
        evidence = parse_evidence(request.tool_results())

        if not evidence:
            if self.search_tool in request.tools and not _already_searched(request):
                # Ask for retrieval first. This makes the offline path a genuine
                # two-turn agent run rather than a single-shot lookup.
                return Generation(
                    text="",
                    tool_calls=[
                        ToolCall(
                            id=_call_id(query),
                            name=self.search_tool,
                            arguments={"query": query},
                        )
                    ],
                    finish_reason="tool_calls",
                    usage=Usage(prompt_tokens=_estimate_tokens(query), completion_tokens=0),
                    model=self.name,
                )
            if not request.tool_results():
                # Nothing retrieved and nothing to ask for: answer from whatever
                # the caller put in the prompt.
                text = _echo_answer(query, request)
                return Generation(text=text, finish_reason="stop", model=self.name,
                                  usage=Usage(prompt_tokens=_estimate_tokens(query)))
            return Generation(
                text=self.fallback_note,
                finish_reason="stop",
                model=self.name,
                usage=Usage(prompt_tokens=_estimate_tokens(query)),
            )

        answer = self.compose(query, evidence)
        return Generation(
            text=answer,
            finish_reason="stop",
            model=self.name,
            usage=Usage(
                prompt_tokens=sum(_estimate_tokens(e.text) for e in evidence),
                completion_tokens=_estimate_tokens(answer),
            ),
        )

    # -- composition --------------------------------------------------------
    def compose(self, query: str, evidence: Sequence[Evidence]) -> str:
        """Build a cited answer from retrieved passages."""
        if not evidence:
            return self.fallback_note

        query_terms = set(tokenize(query))
        sentences = _extract_sentences(evidence, query_terms)
        if not sentences:
            # No sentence cleared the overlap bar; fall back to the leading lines
            # of the best passages so the answer still carries real content.
            sentences = _leading_sentences(evidence, self.max_sentences, self.min_sentence_chars)
        if not sentences:
            return self.fallback_note

        chosen = _dedupe(sentences)[: self.max_sentences]
        # Citation numbers are assigned by source order, not by which sentence
        # was picked, so [1] always means the same passage within one answer.
        used = {item.evidence_rank for item in chosen}
        numbering = {rank: index + 1 for index, rank in enumerate(sorted(used))}

        body_lines = [f"{item.text} [{numbering[item.evidence_rank]}]" for item in chosen]
        lines = ["\n".join(body_lines)]

        sources = ["Sources:"]
        for rank in sorted(used):
            item = evidence[rank]
            sources.append(f"  [{numbering[rank]}] {item.source} (chunk {item.ordinal}, score {item.score:.4f})")
        lines.append("\n".join(sources))
        return "\n\n".join(lines)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        from atheneum.providers.base import TextDelta, ToolCallDelta, UsageEvent

        generation = self.complete(request)
        for call in generation.tool_calls:
            yield ToolCallDelta(call=call)
        if generation.text:
            # Stream a sentence at a time so the CLI can render progressively.
            for piece in re.split(r"(?<=[.!?。！？])\s+", generation.text):
                if piece:
                    yield TextDelta(text=piece + " " if piece else "")
        yield UsageEvent(usage=generation.usage)


@dataclass(frozen=True, slots=True)
class _Sentence:
    text: str
    evidence_rank: int
    score: float
    position: int


def parse_evidence(messages: Sequence[Message]) -> list[Evidence]:
    """Read retrieved passages out of tool result messages.

    Accepts the JSON emitted by the built-in search tool and ignores anything
    else, so an unrelated tool result never masquerades as evidence.
    """
    collected: list[Evidence] = []
    for message in messages:
        if message.name and message.name not in EVIDENCE_TOOLS:
            continue
        payload = _load_json(message.content)
        if payload is None:
            continue
        results = payload.get("results")
        if not isinstance(results, list):
            continue
        for entry in results:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text") or entry.get("content")
            if not isinstance(text, str) or not text.strip():
                continue
            collected.append(
                Evidence(
                    source=str(entry.get("source", "unknown")),
                    ordinal=int(entry.get("ordinal", entry.get("chunk_ordinal", 0)) or 0),
                    text=text.strip(),
                    rank=len(collected),
                    score=float(entry.get("score", 0.0) or 0.0),
                    chunk_id=str(entry.get("chunk_id", "") or ""),
                )
            )
    # Best-scoring passages first, with source and ordinal breaking ties so the
    # ordering never depends on dict iteration or insertion luck.
    collected.sort(key=lambda e: (-e.score, e.source, e.ordinal))
    for position, item in enumerate(collected):
        object.__setattr__(item, "rank", position)
    return collected


def _extract_sentences(evidence: Sequence[Evidence], query_terms: set[str]) -> list[_Sentence]:
    scored: list[_Sentence] = []
    for item in evidence[: _limit_for(evidence)]:
        terms = Counter(tokenize(item.text))
        for position, sentence in enumerate(_split_sentences(item.text)):
            stripped = sentence.strip()
            if len(stripped) < 20 or _STRUCTURAL_RE.match(stripped):
                continue
            if query_terms:
                hits = sum(1 for term in query_terms if term in terms)
                if hits == 0:
                    continue
            else:
                hits = 0
            # Passage rank dominates because retrieval already decided relevance;
            # term coverage then picks which sentence inside it to show.
            coverage = hits / len(query_terms) if query_terms else 0.0
            score = (1.0 / (1.0 + item.rank)) + coverage - (position * 0.02)
            scored.append(_Sentence(stripped, item.rank, score, position))
    scored.sort(key=lambda s: (-s.score, s.evidence_rank, s.position))
    return scored


def _limit_for(evidence: Sequence[Evidence]) -> int:
    return max(1, min(8, len(evidence)))


def _leading_sentences(
    evidence: Sequence[Evidence], count: int, min_chars: int
) -> list[_Sentence]:
    out: list[_Sentence] = []
    for item in evidence[:count]:
        for position, sentence in enumerate(_split_sentences(item.text)):
            stripped = sentence.strip()
            if len(stripped) >= min_chars and not _STRUCTURAL_RE.match(stripped):
                out.append(_Sentence(sentence.strip(), item.rank, 1.0 / (1 + item.rank), position))
                break
    return out


def _split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENTENCE_RE.split(text) if p.strip()]
    return parts or [text.strip()]


def _dedupe(sentences: Sequence[_Sentence]) -> list[_Sentence]:
    seen: set[str] = set()
    kept: list[_Sentence] = []
    for sentence in sentences:
        key = re.sub(r"\W+", "", sentence.text.lower())[:120]
        if not key or key in seen:
            continue
        seen.add(key)
        kept.append(sentence)
    # Present the answer in evidence order rather than score order, so the
    # sentences read as a passage rather than a scatter of unrelated facts.
    kept.sort(key=lambda s: (s.evidence_rank, s.position))
    return kept


def _already_searched(request: GenerationRequest) -> bool:
    """Stop the loop when retrieval has already run and returned nothing.

    Without this the provider would keep emitting the same search call and the
    agent would burn every turn on it.
    """
    for message in request.messages:
        for call in message.tool_calls:
            if call.name == request_search_tool(request) or call.name in EVIDENCE_TOOLS:
                return True
    return any(message.name in EVIDENCE_TOOLS for message in request.messages)


def request_search_tool(request: GenerationRequest) -> str:
    return "search"


def _echo_answer(query: str, request: GenerationRequest) -> str:
    context = [m.content for m in request.messages if m.role.value == "user" and m.content]
    if len(context) > 1:
        return f"Answer (offline extractive mode, no corpus configured):\n{context[-1]}"
    return (
        "Answer (offline extractive mode): no documents are indexed yet. "
        "Run `ath index PATH...` to add some, then ask again."
    )


def _load_json(raw: str) -> dict[str, Any] | None:
    text = raw.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def _call_id(query: str) -> str:
    # Must not use the builtin hash(): it is salted per process for strings, and
    # this provider's reproducibility guarantee depends on stable identifiers.
    digest = hashlib.blake2b(query.encode("utf-8"), digest_size=8).hexdigest()
    return f"call_{digest}"


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
