"""Conversation memory with bounded, predictable context compaction.

Compaction is where agent runtimes most often go wrong: too aggressive and the
agent forgets what it was doing, too lax and the request exceeds the model's
window. The semantics here are stated in fractions of a token budget and pinned
by tests, because a "sliding window" that silently drops the entire history is a
real bug that has shipped in this space.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from atheneum.core.types import Message, Role, ToolCall, ToolResult

__all__ = ["ConversationMemory", "compact", "estimate_tokens"]

# Digest line priorities. When the token budget cannot hold everything, a
# retrieved finding outvalues a restatement of the agent's own question.
FINDINGS, INTENT, PROSE = 0, 1, 2


def estimate_tokens(text: str) -> int:
    """Cheap token estimate.

    Four characters per token is the usual English average. CJK needs a stricter
    count because a single Han character is often a whole token on its own, so
    those are weighted at one character per token.
    """
    if not text:
        return 0
    cjk = sum(1 for char in text if _is_cjk_char(char))
    other = len(text) - cjk
    # Ceiling division, so a CJK-only string contributes exactly its length and
    # is not inflated by a phantom token for its empty Latin remainder.
    return cjk + (other + 3) // 4


def _is_cjk_char(char: str) -> bool:
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF
        or 0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xAC00 <= code <= 0xD7AF
    )


def message_tokens(message: Message) -> int:
    total = estimate_tokens(message.content) + 4  # role and framing overhead
    for call in message.tool_calls:
        total += estimate_tokens(call.name) + estimate_tokens(str(call.arguments)) + 6
    return total


@dataclass(slots=True)
class ConversationMemory:
    """Session history plus the budget it must fit inside.

    keep_ratio is the fraction of ``token_budget`` reserved for the most recent
    turns. The remainder is what the summary of older turns may occupy. It is
    deliberately a ratio rather than a message count, because one tool result can
    be larger than twenty short exchanges.
    """

    token_budget: int = 8000
    keep_ratio: float = 0.7
    system_prompt: str | None = None
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.token_budget <= 0:
            raise ValueError(f"token_budget must be positive, got {self.token_budget}")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError(f"keep_ratio must be in (0, 1], got {self.keep_ratio}")

    def add(self, message: Message) -> None:
        self.messages.append(message)

    def add_user(self, content: str) -> Message:
        message = Message.user(content)
        self.add(message)
        return message

    def add_assistant(
        self, content: str = "", tool_calls: Sequence[ToolCall] | None = None
    ) -> Message:
        message = Message.assistant(content, list(tool_calls or []))
        self.add(message)
        return message

    def add_tool_result(self, result: ToolResult) -> Message:
        message = Message.tool(result)
        self.add(message)
        return message

    def clear(self) -> None:
        self.messages.clear()

    def __len__(self) -> int:
        return len(self.messages)

    @property
    def total_tokens(self) -> int:
        return sum(message_tokens(m) for m in self.messages)

    def window(self) -> list[Message]:
        """Messages to send now: the system prompt plus a compacted history."""
        history = compact(self.messages, self.token_budget, keep_ratio=self.keep_ratio)
        if self.system_prompt:
            return [Message.system(self.system_prompt), *history]
        return history


def compact(
    messages: Sequence[Message],
    token_budget: int,
    *,
    keep_ratio: float = 0.7,
    summarize: bool = True,
) -> list[Message]:
    """Fit ``messages`` inside ``token_budget``.

    The tail is preserved verbatim up to ``keep_ratio`` of the budget. Older
    turns, if any are dropped, collapse into a single synthetic summary message
    rather than vanishing — an agent that loses its own earlier findings will
    redo work or contradict itself.

    At least the final message always survives, so compaction can never return
    an empty window no matter how small the budget is.
    """
    if token_budget <= 0:
        raise ValueError(f"token_budget must be positive, got {token_budget}")
    if not messages:
        return []

    costs = [message_tokens(m) for m in messages]
    if sum(costs) <= token_budget:
        return list(messages)

    recent_budget = max(1, int(token_budget * keep_ratio))

    tail: list[Message] = []
    used = 0
    for index in range(len(messages) - 1, -1, -1):
        cost = costs[index]
        # The first iteration is unconditional: the newest message must survive
        # even when it alone exceeds the budget.
        if tail and used + cost > recent_budget:
            break
        tail.insert(0, messages[index])
        used += cost

    dropped = list(messages[: len(messages) - len(tail)])
    if not dropped:
        return tail

    if not summarize:
        # Returning the tail alone is right when the caller wants raw recency and
        # is accepting that older findings disappear.
        return tail

    summary = summarize_messages(dropped, max(1, token_budget - used))
    return [summary, *tail] if summary is not None else tail


def summarize_messages(dropped: Sequence[Message], token_budget: int) -> Message | None:
    """Collapse older turns into one deterministic digest.

    Extractive rather than model-generated on purpose: it must work offline, and
    it must be reproducible so a test can assert on its content.

    When the budget only affords a few lines they are chosen by information
    value. A retrieved figure in a tool result is what lets the agent keep
    working; a restatement of its own question is not.
    """
    if not dropped or token_budget <= 0:
        return None

    candidates: list[tuple[int, int, str]] = []

    for position, message in enumerate(dropped):
        if message.role is Role.SYSTEM:
            continue
        if message.tool_calls:
            names = ", ".join(
                f"{call.name}({json_preview(call.arguments)})" for call in message.tool_calls
            )
            candidates.append((INTENT, position, f"assistant requested tools: {names}"))
        if message.role is Role.TOOL:
            preview = _first_clause(message.content, 240)
            if preview:
                candidates.append((FINDINGS, position, f"tool {message.name or '?'} returned: {preview}"))
            continue
        preview = _first_clause(message.content, 120)
        if preview:
            candidates.append((PROSE, position, f"{message.role.value}: {preview}"))

    if not candidates:
        return None

    head = "Summary of earlier conversation:\n"
    remaining = max(1, token_budget - estimate_tokens(head))

    # Spend the budget highest-value-first, preferring the most recent line
    # within a class, but emit chronologically so the digest reads as history.
    chosen: list[tuple[int, str]] = []
    for _rank, position, line in sorted(candidates, key=lambda entry: (entry[0], -entry[1])):
        cost = estimate_tokens(f"- {line}")
        if cost <= remaining:
            chosen.append((position, line))
            remaining -= cost
            continue
        if not chosen:
            # Nothing fits at all: truncate the single most valuable line rather
            # than returning no digest, which would discard the whole history.
            room = max(1, remaining * 4)
            chosen.append((position, line[:room].rsplit(" ", 1)[0] + "…"))
        # Otherwise skip it. A lower-value line that fits is worth more here than
        # a higher-value line reduced to fragments.

    if not chosen:
        return None

    chosen.sort()
    return Message.system(head + "\n".join(f"- {line}" for _, line in chosen))


def json_preview(value: object, limit: int = 80) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _first_clause(text: str, limit: int) -> str:
    stripped = " ".join((text or "").split())
    if not stripped:
        return ""
    if len(stripped) <= limit:
        return stripped
    return stripped[:limit].rsplit(" ", 1)[0] + "…"
