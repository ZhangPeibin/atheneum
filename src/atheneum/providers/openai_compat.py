"""OpenAI-compatible chat completions provider.

One implementation covers OpenAI, Azure OpenAI, DeepSeek, Moonshot, vLLM, LM
Studio and Ollama's ``/v1`` surface, because they all speak this wire format.
That interoperability is the reason to target it first rather than writing a
provider per vendor.
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

__all__ = ["OpenAICompatibleProvider"]

_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Parameters that several OpenAI-compatible servers reject outright. Sending
# them only when the caller asks keeps the provider working against the widest
# range of servers, which is the entire point of "compatible".
_OMIT_ZERO_TEMPERATURE = ("presence_penalty", "frequency_penalty")


class OpenAICompatibleProvider(Provider):
    name = "openai"
    supports_tools = True
    supports_streaming = True

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        env_prefix: str = "OPENAI",
        timeout: float = 120.0,
        max_retries: int = 4,
        extra_headers: dict[str, str] | None = None,
        supports_tools: bool = True,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get(f"{env_prefix}_API_KEY") or os.environ.get("OPENAI_API_KEY")
        self.base_url = (
            base_url or os.environ.get(f"{env_prefix}_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.extra_headers = dict(extra_headers or {})
        self.supports_tools = supports_tools

    # -- transport ----------------------------------------------------------
    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise ProviderError(
                "no API key configured for the OpenAI-compatible provider. Set "
                "OPENAI_API_KEY, run `ath keys set openai <key>`, or use the "
                "built-in `offline` provider."
            )
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        headers.update(self.extra_headers)
        return headers

    def _request(self, path: str, payload: dict[str, Any], *, stream: bool = False) -> Any:
        try:
            import httpx
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ProviderError(
                "network providers need the `net` extra: pip install atheneum[net]"
            ) from exc

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            client = httpx.Client(timeout=self.timeout)
            # Set only once the streaming context is successfully built, because
            # from that point the caller owns the client and stream()'s finally
            # closes it. Every other exit path -- retry, 4xx, missing API key,
            # transport error -- must close here or the socket leaks.
            handed_over = False
            try:
                if stream:
                    context = client.stream(
                        "POST", self._url(path), headers=self._headers(), json=payload
                    )
                    handed_over = True
                    return client, context
                response = client.post(self._url(path), headers=self._headers(), json=payload)
                status = response.status_code
                if status in _RETRYABLE_STATUS:
                    last_error = ProviderError(
                        f"{self._url(path)} returned {status}", status=status, retryable=True
                    )
                    retry_after = response.headers.get("retry-after")
                    _backoff(attempt, retry_after)
                    continue
                if status >= 400:
                    raise ProviderError(
                        f"{self._url(path)} returned {status}: {response.text[:400]}",
                        status=status,
                        retryable=False,
                    )
                return response.json()
            except ProviderError:
                raise
            except httpx.HTTPError as exc:
                last_error = exc
                _backoff(attempt, None)
            finally:
                if not handed_over:
                    client.close()
        raise ProviderError(
            f"request to {self._url(path)} failed after {self.max_retries} attempts: {last_error}",
            retryable=True,
        ) from last_error

    # -- wire format --------------------------------------------------------
    def _payload(self, request: GenerationRequest, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [_encode_message(m) for m in request.messages],
        }
        if stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}
        if request.tools and self.supports_tools:
            payload["tools"] = [
                {"type": "function", "function": {"name": n, "description": s.get("description", ""),
                 "parameters": s.get("parameters", {"type": "object", "properties": {}})}}
                for n, s in request.tools.items()
            ]
        if request.temperature:
            payload["temperature"] = request.temperature
        if request.max_tokens:
            payload["max_tokens"] = request.max_tokens
        if request.stop:
            payload["stop"] = list(request.stop)
        return payload

    def complete(self, request: GenerationRequest) -> Generation:
        data = self._request("/chat/completions", self._payload(request, stream=False))
        return _decode(self.model, data)

    def stream(self, request: GenerationRequest) -> Iterator[StreamEvent]:
        client, context = self._request("/chat/completions", self._payload(request, stream=True), stream=True)
        pending: dict[int, dict[str, Any]] = {}
        usage = Usage()
        try:
            with context as response:
                if response.status_code >= 400:
                    body = response.read().decode("utf-8", "replace")[:400]
                    raise ProviderError(f"stream from {self.model} failed: {body}", status=response.status_code)
                for line in response.iter_lines():
                    if not line or not line.startswith("data:"):
                        continue
                    chunk = line[5:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except json.JSONDecodeError:
                        continue
                    if data.get("usage"):
                        usage = _decode_usage(data["usage"])
                    for choice in data.get("choices", []):
                        delta = choice.get("delta") or {}
                        text = delta.get("content")
                        if text:
                            yield TextDelta(text=text)
                        for index, call in enumerate(delta.get("tool_calls") or []):
                            slot = pending.setdefault(index, {"id": "", "name": "", "arguments": ""})
                            slot["id"] = call.get("id") or slot["id"]
                            function = call.get("function") or {}
                            slot["name"] = function.get("name") or slot["name"]
                            slot["arguments"] += function.get("arguments") or ""
        finally:
            client.close()

        for index in sorted(pending):
            call = pending[index]
            if not call["name"]:
                continue
            try:
                arguments = json.loads(call["arguments"]) if call["arguments"] else {}
            except json.JSONDecodeError:
                # Malformed arguments must not silently become an empty call;
                # surface it so the loop can report the error back to the model.
                arguments = {"_raw": call["arguments"], "_parse_error": "invalid JSON in tool arguments"}
            yield ToolCallDelta(call=ToolCall(id=call["id"] or f"call_{index}", name=call["name"], arguments=arguments))
        yield UsageEvent(usage=usage)


def _decode(model: str, data: dict[str, Any]) -> Generation:
    choices = data.get("choices") or []
    if not choices:
        raise ProviderError(f"provider returned no choices for {model}")
    message = choices[0].get("message") or {}
    finish = choices[0].get("finish_reason") or "stop"
    calls: list[ToolCall] = []
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        raw = function.get("arguments") or "{}"
        try:
            arguments = json.loads(raw) if isinstance(raw, str) else dict(raw)
        except json.JSONDecodeError:
            arguments = {"_raw": raw, "_parse_error": "invalid JSON in tool arguments"}
        calls.append(ToolCall(id=call.get("id") or f"call_{len(calls)}", name=function.get("name", ""), arguments=arguments))
    reason = "tool_calls" if calls else ("length" if finish == "length" else "stop")
    return Generation(
        text=message.get("content") or "",
        tool_calls=calls,
        finish_reason=reason,  # type: ignore[arg-type]
        usage=_decode_usage(data.get("usage") or {}),
        model=data.get("model", model),
        raw=data,
    )


def _decode_usage(raw: dict[str, Any]) -> Usage:
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens", 0) or 0),
        completion_tokens=int(raw.get("completion_tokens", 0) or 0),
    )


def _encode_message(message: Message) -> dict[str, Any]:
    data: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.role is Role.ASSISTANT and message.tool_calls:
        data["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments, ensure_ascii=False)},
            }
            for call in message.tool_calls
        ]
    if message.role is Role.TOOL:
        data["tool_call_id"] = message.tool_call_id or ""
        if message.name:
            data["name"] = message.name
    return data


def _backoff(attempt: int, retry_after: str | None) -> None:
    delay = 0.5 * (2**attempt)
    if retry_after:
        with contextlib.suppress(ValueError):
            delay = max(delay, min(float(retry_after), 30.0))
    time.sleep(min(delay, 30.0))
