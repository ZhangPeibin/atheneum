"""The provider interface.

A provider turns a list of messages plus tool schemas into either text or tool
calls. Everything above this layer — the agent loop, the CLI, the HTTP API — is
provider-agnostic, so swapping the model never touches retrieval or memory.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from atheneum.core.types import Message, ToolCall

__all__ = [
    "FinishReason",
    "Generation",
    "GenerationRequest",
    "Provider",
    "ProviderError",
    "StreamEvent",
    "TextDelta",
    "ToolCallDelta",
    "Usage",
]

FinishReason = Literal["stop", "tool_calls", "length", "error"]


class ProviderError(RuntimeError):
    """A provider could not produce a response."""

    def __init__(self, message: str, *, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status = status
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    """Everything a provider needs to produce one turn."""

    messages: Sequence[Message]
    tools: dict[str, dict[str, Any]] = field(default_factory=dict)
    temperature: float = 0.0
    max_tokens: int | None = None
    stop: Sequence[str] = ()
    # Providers that cannot honour a request shape (no tool support, say) need
    # this to be explicit rather than discovering it by failing.
    require_tools: bool = False

    def last_user_message(self) -> str:
        for message in reversed(list(self.messages)):
            if message.role.value == "user":
                return message.content
        return ""

    def tool_results(self) -> list[Message]:
        """Tool results that arrived after the most recent user turn.

        Scoping to the current turn matters: a provider reasoning about "what
        did the tools say" should see this turn's evidence, not the whole
        session's history of them.
        """
        messages = list(self.messages)
        start = 0
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].role.value == "user":
                start = index
                break
        return [m for m in messages[start:] if m.role.value == "tool"]


@dataclass(frozen=True, slots=True)
class Generation:
    """One completed provider turn."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: FinishReason = "stop"
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls) and self.finish_reason == "tool_calls"


@dataclass(frozen=True, slots=True)
class TextDelta:
    text: str


@dataclass(frozen=True, slots=True)
class ToolCallDelta:
    call: ToolCall


@dataclass(frozen=True, slots=True)
class UsageEvent:
    usage: Usage


StreamEvent = TextDelta | ToolCallDelta | UsageEvent


class Provider(ABC):
    """Base class for all model providers."""

    name: str = "base"
    supports_tools: bool = True
    supports_streaming: bool = True

    @abstractmethod
    def complete(self, request: GenerationRequest) -> Generation:
        """Produce one turn."""

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        """Yield events for one turn.

        The default implementation completes and emits the text whole. Providers
        that can truly stream should override this so callers get incremental
        output.
        """
        generation = self.complete(request)
        if generation.tool_calls:
            for call in generation.tool_calls:
                yield ToolCallDelta(call=call)
        if generation.text:
            yield TextDelta(text=generation.text)
        yield UsageEvent(usage=generation.usage)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "supports_tools": self.supports_tools,
            "supports_streaming": self.supports_streaming,
        }


def render_transcript(messages: Sequence[Message]) -> str:
    """Flatten a conversation to plain text for providers without message APIs."""
    lines: list[str] = []
    for message in messages:
        prefix = message.role.value
        if message.name:
            prefix = f"{prefix}:{message.name}"
        body = message.content
        if message.tool_calls:
            body = (body + "\n" if body else "") + json.dumps(
                {
                    "tool_calls": [
                        {"id": c.id, "name": c.name, "arguments": c.arguments}
                        for c in message.tool_calls
                    ]
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        lines.append(f"{prefix}\n{body}")
    return "\n\n".join(lines)
