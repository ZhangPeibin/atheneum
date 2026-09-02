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
from collections.abc import Iterator
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
                return response.json()
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
        converted = [
            block
            for message in request.messages
            if message.role is not Role.SYSTEM
            for block in [_encode(message)]
            if block is not None
        ]
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
                    if kind == "content_block_start":
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
                            if index in blocks:
                                blocks[index]["input"] += delta.get("partial_json", "")
                    elif kind == "message_delta":
                        raw = event.get("usage") or {}
                        usage = Usage(usage.prompt_tokens, int(raw.get("output_tokens", 0) or 0))
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


def _encode(message: Message) -> dict[str, Any] | None:
    """Map one conversation message onto an Anthropic message object."""
    if message.role is Role.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content,
                    "is_error": message.content.lstrip().startswith('{"error"'),
                }
            ],
        }
    role = "assistant" if message.role is Role.ASSISTANT else "user"
    content: list[dict[str, Any]] = []
    if message.content:
        content.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        content.append(
            {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        )
    if not content:
        return None
    return {"role": role, "content": content}


def _decode(model: str, data: dict[str, Any]) -> Generation:
    texts: list[str] = []
    calls: list[ToolCall] = []
    for block in data.get("content") or []:
        if block.get("type") == "text":
            texts.append(block.get("text", ""))
        elif block.get("type") == "tool_use":
            calls.append(
                ToolCall(
                    id=block.get("id", f"toolu_{len(calls)}"),
                    name=block.get("name", ""),
                    arguments=dict(block.get("input") or {}),
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
