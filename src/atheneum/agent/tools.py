"""Tool declaration and execution.

A tool is an ordinary Python function. Its JSON schema is derived from the
signature, so the model sees the same contract the code enforces — and there is
no second place to keep in sync.

Errors are data, not exceptions: a failing tool returns an error ToolResult that
the agent loop hands back to the model so it can correct itself. Aborting the
whole run because one call had a bad argument is how agent loops become useless.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import types
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, TypeVar, Union, get_args, get_origin, get_type_hints, overload

from atheneum.core.types import ToolCall, ToolResult

logger = logging.getLogger("atheneum.tools")

__all__ = ["Tool", "ToolError", "ToolRegistry", "tool"]


class ToolError(Exception):
    """Raised only for tool *registration* problems, never during execution."""


JSONType = Literal["string", "number", "integer", "boolean", "object", "array"]

_PRIMITIVES: dict[type, JSONType] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


@dataclass(frozen=True, slots=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any] = field(repr=False, default=lambda **_: None)
    # A tool marked destructive is the hook for a confirmation prompt; the loop
    # consults it, the registry only records it.
    requires_approval: bool = False

    def schema(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "name": self.name,
            "parameters": self.parameters,
            "requires_approval": self.requires_approval,
        }


_F = TypeVar("_F", bound=Callable[..., Any])


@overload
def tool(function: _F) -> Tool: ...


@overload
def tool(
    function: None = None,
    *,
    name: str | None = ...,
    description: str | None = ...,
    requires_approval: bool = ...,
) -> Callable[[_F], Tool]: ...


def tool(
    function: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
) -> Tool | Callable[[Callable[..., Any]], Tool]:
    """Register a function as a model-callable tool.

    Usable bare (``@tool``) or with arguments (``@tool(requires_approval=True)``).
    """

    def wrap(func: Callable[..., Any]) -> Tool:
        definition = Tool(
            name=name or func.__name__,
            description=description or _summary_from_docstring(func.__doc__ or ""),
            parameters=schema_for_function(func),
            function=func,
            requires_approval=requires_approval,
        )
        return definition

    if function is not None:
        return wrap(function)
    return wrap


def _summary_from_docstring(doc: str) -> str:
    if not doc:
        return ""
    lines = [line.strip() for line in doc.strip().splitlines()]
    # Take up to the first blank line: that is the summary, the rest is detail
    # the model does not need on every request.
    summary: list[str] = []
    for line in lines:
        if not line:
            break
        summary.append(line)
    return " ".join(summary)[:600]


def schema_for_function(function: Callable[..., Any]) -> dict[str, Any]:
    """Build a JSON-Schema `parameters` object from a signature."""
    try:
        hints = get_type_hints(function)
    except Exception as exc:
        raw = getattr(function, "__annotations__", {})
        if any(isinstance(value, str) for value in raw.values()):
            # A string annotation that could not be resolved would silently turn
            # into "type": "string", telling the model the wrong thing about
            # every argument. Fail loudly instead.
            raise ToolError(
                f"cannot resolve type annotations for tool {function.__name__}: {exc}. "
                "Import the annotated types at module level so they can be resolved."
            ) from exc
        hints = raw

    properties: dict[str, Any] = {}
    required: list[str] = []
    signature = inspect.signature(function)

    for parameter_name, parameter in signature.parameters.items():
        if parameter.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            # **kwargs cannot be described to a model, so it cannot be honoured.
            raise ToolError(
                f"tool {function.__name__} declares {parameter.kind.name.lower()}; "
                "every parameter must be named so it can appear in the schema"
            )
        annotation = hints.get(parameter_name)
        if annotation is None:
            # Defaulting to "string" would silently rewrite 4 into "4" and let a
            # numeric tool receive corrupted arguments with nothing reporting it.
            raise ToolError(
                f"tool {function.__name__} has unannotated parameter {parameter_name!r}; "
                "annotate it so the model is given the right schema"
            )
        properties[parameter_name] = schema_for_type(annotation, parameter_name)

        has_default = parameter.default is not inspect.Parameter.empty
        if (not has_default and parameter.kind is not inspect.Parameter.KEYWORD_ONLY) or not has_default:
            required.append(parameter_name)

    schema: dict[str, Any] = {"type": "object", "properties": properties}
    if required:
        schema["required"] = required
    schema["additionalProperties"] = False
    return schema


def schema_for_type(annotation: Any, label: str = "") -> dict[str, Any]:
    origin = get_origin(annotation)

    if origin is Literal:
        choices = list(get_args(annotation))
        kinds = {_json_type(type(choice).__name__, type(choice)) for choice in choices}
        schema: dict[str, Any] = {"enum": choices}
        if len(kinds) == 1:
            schema["type"] = next(iter(kinds))
        return schema

    if origin in (Union, types.UnionType):
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            # ``Optional[X]`` is documented as X plus a note; a nullable union
            # with several members is beyond what these models handle reliably.
            schema = schema_for_type(args[0], label)
            schema["nullable"] = True
            return schema
        return {"anyOf": [schema_for_type(a, label) for a in args]}

    if origin in (list, set, frozenset, Sequence, Iterable):
        item_args = get_args(annotation)
        item = schema_for_type(item_args[0], label) if item_args else {"type": "string"}
        return {"type": "array", "items": item}

    if origin is dict or annotation is dict:
        return {"type": "object", "additionalProperties": True}

    if annotation is Any or annotation is None:
        return {"type": "string"}

    if inspect.isclass(annotation):
        return {"type": _PRIMITIVES.get(annotation, "string")}

    return {"type": "string"}


def _json_type(_name: str, cls: type) -> JSONType:
    return _PRIMITIVES.get(cls, "string")


class ToolRegistry:
    """Named collection of tools, with validated dispatch."""

    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for item in tools:
            self.register(item)

    def register(self, definition: Tool) -> Tool:
        if not definition.name:
            raise ToolError("a tool must have a name")
        if definition.name in self._tools:
            logger.info("overwriting previously registered tool %s", definition.name)
        self._tools[definition.name] = definition
        return definition

    def add(self, function: Callable[..., Any], **kwargs: Any) -> Tool:
        definition = tool(function, **kwargs)
        return self.register(definition)

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    def names(self) -> list[str]:
        return sorted(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self, only: Sequence[str] | None = None) -> dict[str, dict[str, Any]]:
        if only is None:
            return {name: t.schema() for name, t in sorted(self._tools.items())}
        allowed = set(only)
        return {name: t.schema() for name, t in sorted(self._tools.items()) if name in allowed}

    def select(self, allowed: Iterable[str]) -> ToolRegistry:
        """Return a registry limited to ``allowed`` names.

        Tool allow-listing is the cheapest meaningful guardrail: it lets a
        read-only agent be *proven* read-only rather than merely prompted to be.
        """
        wanted = set(allowed)
        return ToolRegistry(t for name, t in self._tools.items() if name in wanted)

    def execute(self, call: ToolCall, *, result_limit: int = 20_000) -> ToolResult:
        """Run one tool call, converting every failure into an error result."""
        definition = self._tools.get(call.name)
        if definition is None:
            available = ", ".join(self.names()) or "none"
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=json.dumps(
                    {
                        "error": "unknown_tool",
                        "message": f"no tool named {call.name!r} is registered",
                        "available": available,
                    }
                ),
                is_error=True,
            )

        try:
            arguments = _coerce(call.arguments, definition.parameters)
        except (ValueError, TypeError) as exc:
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=json.dumps(
                    {
                        "error": "invalid_arguments",
                        "message": str(exc),
                        "expected": definition.parameters,
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
            )

        try:
            produced = definition.function(**arguments)
        except (KeyboardInterrupt, SystemExit, asyncio.CancelledError, GeneratorExit):
            # Process- and task-level signals, not tool failures. Swallowing a
            # Ctrl-C so the model could "recover" from it would be hostile, and
            # turning CancelledError into data defeats cancellation entirely.
            raise
        except BaseException as exc:
            # BaseException rather than Exception: a library raising a custom
            # BaseException subclass used to abort the whole agent run, which
            # contradicts this module's contract that tool errors are data.
            logger.warning("tool %s raised %s: %s", call.name, type(exc).__name__, exc)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=json.dumps(
                    {
                        "error": type(exc).__name__,
                        "message": str(exc),
                        "hint": "correct the arguments and call this tool again, or use another tool",
                    },
                    ensure_ascii=False,
                ),
                is_error=True,
            )

        try:
            content = _stringify(produced, limit=result_limit)
        except Exception as exc:
            # Serialisation happens outside the call above, so a tool returning an
            # object whose __repr__ raises would abort the run despite the
            # contract that tool failures are data.
            logger.warning("tool %s result could not be serialised: %s", call.name, exc)
            return ToolResult(
                call_id=call.id,
                name=call.name,
                content=json.dumps(
                    {
                        "error": type(exc).__name__,
                        "message": f"tool succeeded but its result could not be serialised: {exc}",
                    }
                ),
                is_error=True,
            )
        return ToolResult(call_id=call.id, name=call.name, content=content, is_error=False)


def _coerce(raw: dict[str, Any], schema: dict[str, Any]) -> dict[str, Any]:
    """Validate and lightly coerce arguments against the declared schema."""
    if not isinstance(raw, dict):
        raise TypeError(f"tool arguments must be an object, got {type(raw).__name__}")

    properties: dict[str, Any] = schema.get("properties", {})
    required: list[str] = schema.get("required", [])
    for missing in required:
        if missing not in raw or raw[missing] is None:
            raise ValueError(f"missing required argument {missing!r}")

    unknown = set(raw) - set(properties)
    if unknown and schema.get("additionalProperties") is False:
        # Reject rather than drop: silently ignoring a parameter the model
        # believed it set produces confidently wrong answers.
        raise ValueError(
            f"unexpected argument(s) {sorted(unknown)}; allowed are {sorted(properties)}"
        )

    coerced: dict[str, Any] = {}
    for key, value in raw.items():
        spec = properties.get(key)
        if spec is None:
            coerced[key] = value
            continue
        coerced[key] = _coerce_value(key, value, spec)
    return coerced


def _coerce_value(key: str, value: Any, spec: dict[str, Any]) -> Any:
    expected = spec.get("type")
    if value is None and spec.get("nullable"):
        return None

    if expected == "integer":
        if isinstance(value, bool):
            raise TypeError(f"argument {key!r} must be an integer, got a boolean")
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().lstrip("-").isdigit():
            return int(value)
        if isinstance(value, float) and value.is_integer():
            return int(value)
        raise TypeError(f"argument {key!r} must be an integer, got {_describe(value)}")

    if expected == "number":
        if isinstance(value, bool) or not isinstance(value, int | float | str):
            raise TypeError(f"argument {key!r} must be a number, got {_describe(value)}")
        try:
            return float(value)
        except ValueError as exc:
            raise TypeError(f"argument {key!r} must be a number, got {_describe(value)}") from exc

    if expected == "boolean":
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in {"true", "false"}:
            return value.lower() == "true"
        raise TypeError(f"argument {key!r} must be a boolean, got {_describe(value)}")

    if expected == "string":
        if isinstance(value, str):
            return value
        # Models sometimes emit a number where a string was requested; the
        # intent is unambiguous, so render it instead of failing the call.
        if isinstance(value, int | float) and not isinstance(value, bool):
            return str(value)
        raise TypeError(f"argument {key!r} must be a string, got {_describe(value)}")

    if expected == "array":
        if isinstance(value, str):
            raise TypeError(f"argument {key!r} must be an array, got a string")
        if not isinstance(value, list):
            raise TypeError(f"argument {key!r} must be an array, got {type(value).__name__}")
        item_spec = spec.get("items")
        return [_coerce_value(f"{key}[{i}]", item, item_spec) for i, item in enumerate(value)] if item_spec else list(value)

    if expected == "object":
        if not isinstance(value, dict):
            raise TypeError(f"argument {key!r} must be an object, got {type(value).__name__}")
        return dict(value)

    if "enum" in spec and value not in spec["enum"]:
        raise ValueError(f"argument {key!r} must be one of {spec['enum']}, got {_describe(value)}")
    return value


def _describe(value: Any, limit: int = 60) -> str:
    """A short, safe description of a value for an error message.

    Never ``repr()``: formatting a deeply nested list raises RecursionError while
    building the message, which escaped the except handler and aborted the agent
    run instead of becoming an error result. It also leaked object internals into
    model-visible text.
    """
    name = type(value).__name__
    try:
        text = str(value)
    except Exception:
        return f"<{name}>"
    if len(text) > limit:
        text = text[:limit] + "…"
    return f"{name}({text})"


def _stringify(value: Any, *, limit: int) -> str:
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
        except (TypeError, ValueError):
            text = repr(value)
    if limit > 0 and len(text) > limit:
        # Truncation is reported inside the payload so the model knows the
        # result is partial and can narrow its query.
        omitted = len(text) - limit
        text = text[:limit] + f"\n... [truncated {omitted} characters; narrow the query]"
    return text
