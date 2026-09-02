"""Core domain types shared across the whole package.

These are plain dataclasses with no I/O and no third-party imports so that every
other module can depend on them without creating cycles.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

__all__ = [
    "Chunk",
    "Document",
    "Message",
    "Role",
    "ScoredChunk",
    "ToolCall",
    "ToolResult",
    "chunk_id",
    "document_id",
]


def document_id(source: str, content: str) -> str:
    """Stable content-addressed id for a document."""
    digest = hashlib.sha256(f"{source}\x00{content}".encode()).hexdigest()
    return digest[:32]


def chunk_id(doc_id: str, ordinal: int, text: str) -> str:
    """Stable id for a chunk within a document.

    The ordinal is included so that identical paragraphs appearing twice in one
    document still get distinct ids and remain independently retrievable.
    """
    digest = hashlib.sha256(f"{doc_id}\x00{ordinal}\x00{text}".encode()).hexdigest()
    return digest[:32]


@dataclass(frozen=True, slots=True)
class Document:
    """A unit of ingestion: one file, one URL, or one pasted string."""

    source: str
    content: str
    title: str | None = None
    mime_type: str = "text/plain"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return document_id(self.source, self.content)

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("Document.source must be a non-empty string")


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable slice of a document."""

    id: str
    doc_id: str
    source: str
    ordinal: int
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_estimate(self) -> int:
        return max(1, len(self.text) // 4)


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    """A chunk plus the score that ranked it, and why."""

    chunk: Chunk
    score: float
    # Per-retriever scores, keyed by retriever name. Kept for explainability so a
    # caller can see whether a hit came from lexical or semantic search.
    score_breakdown: dict[str, float] = field(default_factory=dict)

    @property
    def source(self) -> str:
        return self.chunk.source

    @property
    def text(self) -> str:
        return self.chunk.text

    def citation(self, index: int) -> str:
        return f"[{index}] {self.chunk.source}#{self.chunk.ordinal}"


class Role(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"id": self.id, "name": self.name, "arguments": self.arguments},
            sort_keys=True,
            ensure_ascii=False,
        )


@dataclass(frozen=True, slots=True)
class ToolResult:
    """The outcome of executing a ToolCall.

    Errors are represented as a normal result with ``is_error=True`` rather than
    as exceptions, so the agent loop can feed them back to the model and let it
    recover.
    """

    call_id: str
    name: str
    content: str
    is_error: bool = False


@dataclass(slots=True)
class Message:
    """One turn of a conversation."""

    role: Role
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None
    name: str | None = None
    created_at: float = field(default_factory=time.time)

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str = "", tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=list(tool_calls or []))

    @classmethod
    def tool(cls, result: ToolResult) -> Message:
        return cls(
            role=Role.TOOL,
            content=result.content,
            tool_call_id=result.call_id,
            name=result.name,
        )

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            data["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments} for c in self.tool_calls
            ]
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.name:
            data["name"] = self.name
        return data
