"""The agent loop.

One synchronous core function drives everything; ``stream`` is the same loop
emitting events rather than a second implementation. That is a deliberate
choice: in the largest agent runtime I studied, the non-streaming, streaming and
sync variants were three near-identical copies of a very long function, and
keeping them in step is pure maintenance cost.

Termination is bounded by ``max_turns``. A provider that keeps asking for tools
forever is a real failure mode, and the loop must not be able to hang.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from atheneum.agent.memory import ConversationMemory
from atheneum.agent.tools import Tool, ToolRegistry
from atheneum.core.types import Message, ToolCall, ToolResult
from atheneum.providers.base import (
    Generation,
    GenerationRequest,
    Provider,
    ProviderError,
    TextDelta,
    ToolCallDelta,
    Usage,
)

logger = logging.getLogger("atheneum.agent")

__all__ = ["Agent", "AgentConfig", "AgentEvent", "AgentRun", "Step", "ToolApproved"]

StoppedReason = Literal["final_answer", "max_turns", "no_tools", "provider_error"]

ToolApproved = Callable[[ToolCall, Tool], bool]

DEFAULT_SYSTEM_PROMPT = """You are Atheneum, a research assistant answering questions from a local indexed corpus.

Work rules:
- Answer from retrieved evidence, never from recall. If the corpus has nothing \
relevant, say so plainly and name what you searched for.
- Call the `search` tool before answering a factual question. Refine the query \
rather than repeating it if the first search is unhelpful.
- Cite the passages you used with their bracketed numbers.
- Prefer several narrow searches over one broad one for multi-part questions.
- Say what is uncertain. Do not paper over a gap in the evidence.
"""


@dataclass(slots=True)
class AgentConfig:
    max_turns: int = 8
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    token_budget: int = 8000
    keep_ratio: float = 0.7
    # Tool output is the usual culprit when a context window blows up, so it is
    # capped at the registry boundary rather than trusting tools to be brief.
    result_limit: int = 20_000
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if self.max_turns <= 0:
            raise ValueError(f"max_turns must be positive, got {self.max_turns}")


@dataclass(frozen=True, slots=True)
class Step:
    """One provider turn and the tool results that followed it."""

    turn: int
    generation: Generation
    results: tuple[ToolResult, ...] = ()

    @property
    def requested_tools(self) -> list[str]:
        return [call.name for call in self.generation.tool_calls]


@dataclass(frozen=True, slots=True)
class AgentRun:
    """The complete outcome of one agent invocation."""

    answer: str
    messages: tuple[Message, ...] = ()
    steps: tuple[Step, ...] = ()
    usage: Usage = field(default_factory=Usage)
    stopped_reason: StoppedReason = "final_answer"
    error: str | None = None

    @property
    def turns(self) -> int:
        return len(self.steps)

    @property
    def tool_call_count(self) -> int:
        return sum(len(s.generation.tool_calls) for s in self.steps)

    @property
    def ok(self) -> bool:
        return self.error is None and self.stopped_reason in ("final_answer", "max_turns")

    @property
    def evidence(self) -> list[ToolResult]:
        return [result for step in self.steps for result in step.results]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "turns": self.turns,
            "tool_calls": self.tool_call_count,
            "stopped_reason": self.stopped_reason,
            "usage": self.usage.as_dict(),
            "steps": [
                {
                    "turn": step.turn,
                    "requested_tools": step.requested_tools,
                    "text": step.generation.text,
                    "results": [
                        {"name": r.name, "is_error": r.is_error, "content": r.content}
                        for r in step.results
                    ],
                }
                for step in self.steps
            ],
        }


@dataclass(frozen=True, slots=True)
class AgentEvent:
    kind: Literal["text", "tool_call", "tool_result", "step", "done"]
    turn: int = 0
    text: str = ""
    tool_call: ToolCall | None = None
    tool_result: ToolResult | None = None
    run: AgentRun | None = None


class Agent:
    """A provider plus tools, run until it answers or runs out of turns."""

    def __init__(
        self,
        provider: Provider,
        tools: ToolRegistry | Iterable[Tool] | None = None,
        *,
        config: AgentConfig | None = None,
        confirm: ToolApproved | None = None,
    ) -> None:
        self.provider = provider
        self.config = config or AgentConfig()
        self.tools = _as_registry(tools)
        # Approval is opt-in per tool via `requires_approval`; a caller with no
        # callback registered cannot be shown a prompt, so such tools are refused
        # rather than run unattended.
        self.confirm = confirm

    # -- public API ---------------------------------------------------------
    def run(self, query: str, *, history: Sequence[Message] | None = None) -> AgentRun:
        """Answer ``query``, executing tools as the provider requests them."""
        memory = self._new_memory(history)
        memory.add_user(query)
        steps: list[Step] = []
        usage = Usage()

        for turn in range(1, self.config.max_turns + 1):
            request = GenerationRequest(
                messages=memory.window(),
                tools=self.tools.schemas(),
                temperature=self.config.temperature,
            )
            try:
                generation = self.provider.complete(request)
            except ProviderError as exc:
                logger.error("provider failed on turn %d: %s", turn, exc)
                return AgentRun(
                    answer=usage_and_partial(steps),
                    messages=tuple(memory.messages),
                    steps=tuple(steps),
                    usage=usage,
                    stopped_reason="provider_error",
                    error=str(exc),
                )

            usage = usage + generation.usage
            assistant = Message.assistant(generation.text, generation.tool_calls)
            memory.add(assistant)

            if not generation.tool_calls:
                steps.append(Step(turn=turn, generation=generation))
                return AgentRun(
                    answer=generation.text or usage_and_partial(steps),
                    messages=tuple(memory.messages),
                    steps=tuple(steps),
                    usage=usage,
                    stopped_reason="final_answer",
                )

            if not self.tools:
                # The provider asked for a tool that does not exist. Tell it so
                # explicitly instead of looping on a request that can never work.
                steps.append(Step(turn=turn, generation=generation))
                return AgentRun(
                    answer=generation.text,
                    messages=tuple(memory.messages),
                    steps=tuple(steps),
                    usage=usage,
                    stopped_reason="no_tools",
                    error="provider requested tools but none are registered",
                )

            results = self._execute_calls(generation.tool_calls, memory, turn)
            steps.append(Step(turn=turn, generation=generation, results=tuple(results)))

        # Turns exhausted. The answer is whatever the last turn produced, which
        # for a well-behaved provider is empty or partial — reported as such.
        last_text = steps[-1].generation.text if steps else ""
        return AgentRun(
            answer=last_text,
            messages=tuple(memory.messages),
            steps=tuple(steps),
            usage=usage,
            stopped_reason="max_turns",
            error=None,
        )

    def stream(self, query: str, *, history: Sequence[Message] | None = None) -> Iterator[AgentEvent]:
        """Same loop, emitting events. Text is streamed when the provider can."""
        memory = self._new_memory(history)
        memory.add_user(query)
        steps: list[Step] = []
        usage = Usage()

        for turn in range(1, self.config.max_turns + 1):
            request = GenerationRequest(
                messages=memory.window(), tools=self.tools.schemas(), temperature=self.config.temperature
            )
            text_parts: list[str] = []
            calls: list[ToolCall] = []
            try:
                for event in self.provider.stream(request):
                    if isinstance(event, TextDelta):
                        text_parts.append(event.text)
                        yield AgentEvent(kind="text", turn=turn, text=event.text)
                    elif isinstance(event, ToolCallDelta):
                        calls.append(event.call)
                        yield AgentEvent(kind="tool_call", turn=turn, tool_call=event.call)
                    else:
                        usage = usage + event.usage
            except ProviderError as exc:
                yield AgentEvent(
                    kind="done",
                    turn=turn,
                    run=AgentRun(
                        answer="".join(text_parts),
                        messages=tuple(memory.messages),
                        steps=tuple(steps),
                        usage=usage,
                        stopped_reason="provider_error",
                        error=str(exc),
                    ),
                )
                return

            generation = Generation(
                text="".join(text_parts),
                tool_calls=calls,
                finish_reason="tool_calls" if calls else "stop",
                usage=usage,
                model=self.provider.name,
            )
            memory.add(Message.assistant(generation.text, generation.tool_calls))

            if not calls:
                steps.append(Step(turn=turn, generation=generation))
                yield AgentEvent(
                    kind="done",
                    turn=turn,
                    run=AgentRun(
                        answer=generation.text,
                        messages=tuple(memory.messages),
                        steps=tuple(steps),
                        usage=usage,
                        stopped_reason="final_answer",
                    ),
                )
                return

            results = self._execute_calls(calls, memory, turn)
            for result in results:
                yield AgentEvent(kind="tool_result", turn=turn, tool_result=result)
            steps.append(Step(turn=turn, generation=generation, results=tuple(results)))

        yield AgentEvent(
            kind="done",
            turn=self.config.max_turns,
            run=AgentRun(
                answer=steps[-1].generation.text if steps else "",
                messages=tuple(memory.messages),
                steps=tuple(steps),
                usage=usage,
                stopped_reason="max_turns",
            ),
        )

    # -- internals ----------------------------------------------------------
    def _new_memory(self, history: Sequence[Message] | None) -> ConversationMemory:
        memory = ConversationMemory(
            token_budget=self.config.token_budget,
            keep_ratio=self.config.keep_ratio,
            system_prompt=self.config.system_prompt,
        )
        for message in history or ():
            memory.add(message)
        return memory

    def _execute_calls(self, calls: Sequence[ToolCall], memory: ConversationMemory, turn: int) -> list[ToolResult]:
        results: list[ToolResult] = []
        for call in calls:
            definition = self.tools.get(call.name)
            if definition is not None and definition.requires_approval and (
                self.confirm is None or not self._approved(call, definition)
            ):
                    declined = ToolResult(
                        call_id=call.id,
                        name=call.name,
                        content=(
                            '{"error":"not_approved","message":"The user declined this '
                            'operation or no approval handler is configured. Answer '
                            'without it or ask a different question."}'
                        ),
                        is_error=True,
                    )
                    memory.add(Message.tool(declined))
                    results.append(declined)
                    continue
            result = self.tools.execute(call, result_limit=self.config.result_limit)
            memory.add(Message.tool(result))
            logger.debug("turn %d: %s -> error=%s", turn, call.name, result.is_error)
            results.append(result)
        return results

    def _approved(self, call: ToolCall, definition: Tool) -> bool:
        assert self.confirm is not None
        try:
            return bool(self.confirm(call, definition))
        except Exception as exc:
            logger.warning("approval callback raised %s; treating %s as declined", type(exc).__name__, call.name)
            return False


def usage_and_partial(steps: Sequence[Step]) -> str:
    """Best available text when a run ended without a clean final answer."""
    for step in reversed(list(steps)):
        if step.generation.text.strip():
            return step.generation.text
    return ""


def _as_registry(tools: ToolRegistry | Iterable[Tool] | None) -> ToolRegistry:
    if isinstance(tools, ToolRegistry):
        return tools
    return ToolRegistry(tools or ())


def render_history(messages: Iterable[Message]) -> str:  # pragma: no cover - debug helper
    return "\n".join(f"{m.role.value}: {m.content[:120]}" for m in messages)
