"""Anthropic Messages API provider.

Implemented separately from the OpenAI-compatible provider rather than as
another profile, because the wire format genuinely differs: the system prompt is
a top-level parameter instead of a message, and tool results travel back as
content blocks inside a user message instead of their own message role.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from collections.abc import Iterator, Sequence
from typing import Any

from atheneum.core.types import Message, Role, ToolCall
from atheneum.providers.base import (
    Generation,
    GenerationRequest,
    Provider,
    ProviderError,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    Usage,
    UsageEvent,
)

__all__ = ["AnthropicProvider"]

_RETRYABLE = {408, 409, 425, 429, 500, 502, 503, 504}
_ANTHROPIC_VERSION = "2023-06-01"


class AnthropicProvider(Provider):
    name = "anthropic"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-5",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = 120.0,
        max_retries: int = 4,
        max_tokens: int = 4096,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.base_url = (
            base_url or os.environ.get("ANTHROPIC_BASE_URL") or "https://api.anthropic.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.default_max_tokens = max_tokens

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                "no API key for the Anthropic provider; set ANTHROPIC_API_KEY or run "
                "`ath keys set anthropic <key>`"
            )
        return {
            "x-api-key": self.api_key,
            "anthropic-version": _ANTHROPIC_VERSION,
            "content-type": "application/json",
        }

    def _request(self, payload: dict[str, Any], *, stream: bool) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError(
                "network providers need the `net` extra: pip install atheneum[net]"
            ) from exc

        url = f"{self.base_url}/messages"
        last: Exception | None = None
        for attempt in range(self.max_retries):
            client = httpx.Client(timeout=self.timeout)
            # See OpenAICompatibleProvider._request for why this flag exists:
            # only a successfully built stream context transfers ownership.
            handed_over = False
            try:
                if stream:
                    context = client.stream("POST", url, headers=self._headers(), json=payload)
                    handed_over = True
                    return client, context
                response = client.post(url, headers=self._headers(), json=payload)
                if response.status_code in _RETRYABLE:
                    last = ProviderError(
                        f"anthropic returned {response.status_code}",
                        status=response.status_code,
                        retryable=True,
                    )
                    retry_after = response.headers.get("retry-after")
                    _backoff(attempt, retry_after)
                    continue
                if response.status_code >= 400:
                    raise ProviderError(
                        f"anthropic returned {response.status_code}: {response.text[:400]}",
                        status=response.status_code,
                    )
                try:
                    return response.json()
                except ValueError as exc:
                    # A 200 with a non-JSON body means a proxy answered instead of
                    # the API; the raw ValueError missed the loop's
                    # `except ProviderError` and surfaced as a traceback.
                    raise ProviderError(
                        f"{url} returned status {response.status_code} with a non-JSON "
                        f"body: {response.text[:200]!r}"
                    ) from exc
            except ProviderError:
                raise
            except httpx.HTTPError as exc:
                last = exc
                _backoff(attempt, None)
            finally:
                if not handed_over:
                    client.close()
        raise ProviderError(f"anthropic request failed after {self.max_retries} attempts: {last}") from last

    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        system_parts = [m.content for m in request.messages if m.role is Role.SYSTEM and m.content]
        converted = _build_messages(request.messages)
        if not converted:
            raise ProviderError("no non-system messages to send")
        _validate_pairing(request.messages)
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": request.max_tokens or self.default_max_tokens,
            "messages": converted,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.tools:
            payload["tools"] = [
                {
                    "name": name,
                    "description": spec.get("description", ""),
                    "input_schema": spec.get("parameters", {"type": "object", "properties": {}}),
                }
                for name, spec in request.tools.items()
            ]
        if request.temperature:
            payload["temperature"] = request.temperature
        if stream:
            payload["stream"] = True
        return payload

    def complete(self, request: GenerationRequest) -> Generation:
        data = self._request(self._payload(request, stream=False), stream=False)
        return _decode(self.model, data)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        client, context = self._request(self._payload(request, stream=True), stream=True)
        blocks: dict[int, dict[str, Any]] = {}
        usage = Usage()
        try:
            with context as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")[:400]
                    raise ProviderError(f"anthropic stream failed: {body}", status=response.status_code)
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:].strip())
                    except json.JSONDecodeError:
                        continue
                    kind = event.get("type")
                    if kind == "message_start":
                        # Prompt tokens are only reported here; ignoring it left
                        # streaming cost tracking permanently at zero.
                        start_usage = (event.get("message") or {}).get("usage") or {}
                        usage = Usage(
                            int(start_usage.get("input_tokens", 0) or 0),
                            int(start_usage.get("output_tokens", 0) or 0),
                        )
                    elif kind == "content_block_start":
                        block = event.get("content_block") or {}
                        if block.get("type") == "tool_use":
                            index = int(event.get("index", 0))
                            blocks[index] = {"id": block.get("id", ""), "name": block.get("name", ""), "input": ""}
                    elif kind == "content_block_delta":
                        delta = event.get("delta") or {}
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield TextDelta(text=delta["text"])
                        elif delta.get("type") == "input_json_delta":
                            index = int(event.get("index", 0))
                            # A delta for an index that never got a start event
                            # used to be dropped, losing the tool call entirely.
                            slot = blocks.setdefault(index, {"id": "", "name": "", "input": ""})
                            slot["input"] += delta.get("partial_json", "")
                    elif kind == "message_delta":
                        raw = event.get("usage") or {}
                        usage = Usage(usage.prompt_tokens, int(raw.get("output_tokens", 0) or 0))
                    elif kind == "message_stop":
                        pass
        finally:
            client.close()

        for index in sorted(blocks):
            entry = blocks[index]
            try:
                arguments = json.loads(entry["input"]) if entry["input"].strip() else {}
            except json.JSONDecodeError:
                arguments = {"_raw": entry["input"], "_parse_error": "invalid JSON in tool input"}
            yield ToolCallDelta(
                call=ToolCall(id=entry["id"] or f"toolu_{index}", name=entry["name"], arguments=arguments)
            )
        yield UsageEvent(usage=usage)


def _blocks_for(message: Message) -> list[dict[str, Any]]:
    """Blocks contributed by a single non-tool message."""
    blocks: list[dict[str, Any]] = []
    if message.content:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        blocks.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
    return blocks


def _tool_result_block(message: Message) -> dict[str, Any]:
    # message.is_error is authoritative. Sniffing the text for a leading
    # '{"error"' misfired in both directions: a passage documenting an error
    # schema was reported as a failed call, while a real failure rendered as
    # plain text was reported as a success.
    return {
        "type": "tool_result",
        "tool_use_id": message.tool_call_id or "",
        "content": message.content,
        "is_error": message.is_error,
    }


def _build_messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """Map a conversation onto Anthropic messages with alternating roles.

    Two rules the previous implementation broke, each of which is an HTTP 400:
    consecutive tool results became two adjacent ``user`` messages instead of one
    user message with several tool_result blocks (so any parallel tool call
    failed), and an empty message was dropped outright, which could leave two
    adjacent ``assistant`` messages. Merging same-role neighbours fixes both.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role is Role.SYSTEM:
            continue
        if message.role is Role.TOOL:
            block = _tool_result_block(message)
            last = out[-1] if out else None
            # Any preceding user message, not only one made entirely of
            # tool_results: Anthropic permits text and tool_result blocks to share
            # a user message. Restricting it left [user("hello"), tool, tool]
            # encoding as two adjacent user messages, which is an HTTP 400 -- and
            # that sequence is reachable from caller-supplied history or from
            # memory compaction dropping the assistant tool-call turn.
            if last is not None and last["role"] == "user":
                last["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
            continue

        blocks = _blocks_for(message)
        if not blocks:
            # Dropping it could put two same-role messages next to each other;
            # skipping here is fine because the merge keeps roles alternating.
            continue
        role = "assistant" if message.role is Role.ASSISTANT else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"].extend(blocks)
        else:
            out.append({"role": role, "content": blocks})
    return out


def _validate_pairing(messages: Sequence[Message]) -> None:
    """Reject a conversation the API would bounce, with a message that says why.

    Every case below serializes cleanly and fails only as a remote HTTP 400,
    which is useless for debugging: an unanswered tool_use, a tool_result whose
    id matches no tool_use, and a blank tool_use_id all mean the history this
    object was handed is inconsistent -- reachable from caller-supplied history
    and from context compaction dropping an assistant tool-call turn.
    """
    issued: set[str] = set()
    answered: set[str] = set()
    for message in messages:
        if message.role is Role.TOOL:
            if not message.tool_call_id:
                raise ProviderError(
                    f"tool result for {message.name or 'unknown'} carries no tool_call_id"
                )
            answered.add(message.tool_call_id)
            continue
        for call in message.tool_calls:
            if not call.id:
                raise ProviderError(f"tool call {call.name!r} has an empty id")
            issued.add(call.id)

    unanswered = sorted(issued - answered)
    if unanswered:
        raise ProviderError(f"tool calls were never answered: {unanswered}")
    orphaned = sorted(answered - issued)
    if orphaned:
        raise ProviderError(f"tool results have no matching tool call: {orphaned}")


def _coerce_input(raw: Any) -> dict[str, Any]:
    """Tool input as an object, or as a JSON string that some gateways emit."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw, "_parse_error": "tool input was not valid JSON"}
        return parsed if isinstance(parsed, dict) else {"_raw": raw, "_parse_error": "tool input was not an object"}
    return {}


def _decode(model: str, data: dict[str, Any]) -> Generation:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id") or f"toolu_{len(calls)}",
                    name=block.get("name", ""),
                    arguments=_coerce_input(block.get("input")),
                )
            )
    raw_usage = data.get("usage") or {}
    usage = Usage(
        prompt_tokens=int(raw_usage.get("input_tokens", 0) or 0),
        completion_tokens=int(raw_usage.get("output_tokens", 0) or 0),
    )
    stop_reason = data.get("stop_reason")
    if calls:
        reason = "tool_calls"
    elif stop_reason == "max_tokens":
        reason = "length"
    elif stop_reason in {"pause_turn", "refusal"}:
        # Neither is a completed answer: pause_turn means "send me again to
        # continue" and refusal means the model declined. Reporting either as
        # "stop" let the loop present a non-answer as a final one.
        reason = "error"
    else:
        reason = "stop"
    return Generation(
        text="".join(texts),
        tool_calls=calls,
        finish_reason=reason,  # type: ignore[arg-type]
        usage=usage,
        model=data.get("model", model),
        raw=data,
    )


def _backoff(attempt: int, retry_after: str | None) -> None:
    delay = 0.5 * (2**attempt)
    if retry_after:
        with contextlib.suppress(ValueError):
            delay = max(delay, min(float(retry_after), 30.0))
    time.sleep(min(delay, 30.0))
