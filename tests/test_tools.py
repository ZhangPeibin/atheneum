from __future__ import annotations

import json
from typing import Literal

import pytest

from atheneum.agent.tools import Tool, ToolError, ToolRegistry, schema_for_function, tool
from atheneum.core.types import ToolCall


@tool(name="adder", description="Add two integers.")
def adder(a: int, b: int) -> int:
    return a + b


@tool
def greeter(name: str, excited: bool = False) -> str:
    """Greet someone by name."""
    return f"hi {name}" + ("!" if excited else "")


def divide(numerator: float, denominator: float) -> dict:
    """Divide two numbers."""
    if denominator == 0:
        raise ZeroDivisionError("denominator must not be zero")
    return {"quotient": numerator / denominator}


@tool
def tagged(items: list[str], mode: str = "full") -> list[str]:
    """Tag each item."""
    return [f"{mode}:{i}" for i in items]


def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


registry = ToolRegistry([adder, greeter, tool(divide), tagged, tool(boom)])


# -- schema derivation ------------------------------------------------------
def test_required_parameters_are_detected():
    schema = adder.parameters
    assert schema["required"] == ["a", "b"]
    assert schema["properties"]["a"]["type"] == "integer"


def test_optional_parameters_are_not_required():
    schema = greeter.parameters
    assert schema["required"] == ["name"]
    assert schema["properties"]["excited"]["type"] == "boolean"


def test_docstring_first_line_becomes_description():
    assert greeter.description == "Greet someone by name."
    assert adder.description == "Add two integers."


def test_explicit_description_wins():
    assert tagged.description == "Tag each item."


def test_list_parameter_becomes_array_of_strings():
    assert tagged.parameters["properties"]["items"] == {"type": "array", "items": {"type": "string"}}


def test_additional_properties_are_closed():
    assert adder.parameters["additionalProperties"] is False


def test_literal_becomes_an_enum():
    def pick(choice: Literal["a", "b"]) -> str:
        return choice

    schema = schema_for_function(pick)
    assert schema["properties"]["choice"]["enum"] == ["a", "b"]
    assert schema["properties"]["choice"]["type"] == "string"


def test_optional_annotation_is_nullable():
    def maybe(value: int | None = None) -> int:
        return value or 0

    schema = schema_for_function(maybe)
    assert schema["properties"]["value"]["type"] == "integer"
    assert schema["properties"]["value"]["nullable"] is True
    assert schema.get("required", []) == []


def test_var_kwargs_are_rejected():
    def loose(**kwargs: int) -> int:
        return 0

    with pytest.raises(ToolError, match="named so it can appear"):
        schema_for_function(loose)


def test_multiline_docstring_keeps_only_the_summary():
    def documented(value: str) -> str:
        """One line summary.

        Elaborate guidance that should not reach the schema.
        """
        return value

    assert schema_for_function(documented)
    assert tool(documented).description == "One line summary."


# -- registry ---------------------------------------------------------------
def test_registry_lists_names():
    assert registry.names() == ["adder", "boom", "divide", "greeter", "tagged"]


def test_registry_schemas_are_keyed_by_name():
    assert set(registry.schemas()) == set(registry.names())


def test_select_limits_available_tools():
    limited = registry.select(["adder", "greeter"])
    assert limited.names() == ["adder", "greeter"]
    assert len(limited) == 2


def test_select_ignores_unknown_names():
    assert registry.select(["adder", "ghost"]).names() == ["adder"]


def test_contains_and_get():
    assert "adder" in registry
    assert "ghost" not in registry
    assert registry.get("adder") is adder
    assert registry.get("ghost") is None


def test_duplicate_registration_overwrites():
    def replacement(a: int) -> int:
        return a

    built = ToolRegistry([adder])
    definition = tool(replacement, name="adder")
    built.register(definition)
    assert len(built) == 1
    assert built.get("adder") is definition


def test_nameless_tool_is_rejected():
    with pytest.raises(ToolError, match="must have a name"):
        ToolRegistry([Tool(name="", description="", parameters={}, function=lambda: None)])


# -- execution --------------------------------------------------------------
def test_successful_call():
    result = registry.execute(ToolCall(id="c1", name="adder", arguments={"a": 2, "b": 3}))
    assert result.is_error is False
    assert result.content == "5"


def test_call_with_defaults():
    result = registry.execute(ToolCall(id="c2", name="greeter", arguments={"name": "Ada"}))
    assert result.content == "hi Ada"


def test_call_returning_a_mapping_is_json():
    result = registry.execute(ToolCall(id="c3", name="divide", arguments={"numerator": 9, "denominator": 3}))
    assert json.loads(result.content) == {"quotient": 3.0}


def test_unknown_tool_names_the_available_options():
    result = registry.execute(ToolCall(id="c4", name="ghost", arguments={}))
    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"] == "unknown_tool"
    assert "adder" in payload["available"]


def test_missing_required_argument_becomes_an_error_result():
    result = registry.execute(ToolCall(id="c5", name="adder", arguments={"a": 1}))
    assert result.is_error
    assert "missing required argument" in json.loads(result.content)["message"]


def test_unexpected_argument_is_rejected_not_dropped():
    result = registry.execute(ToolCall(id="c6", name="adder", arguments={"a": 1, "b": 2, "c": 3}))
    assert result.is_error
    assert "unexpected argument" in json.loads(result.content)["message"]


def test_tool_exception_is_captured_as_data():
    result = registry.execute(ToolCall(id="c7", name="boom", arguments={}))
    assert result.is_error
    payload = json.loads(result.content)
    assert payload["error"] == "RuntimeError"
    assert "correct the arguments" in payload["hint"]


def test_division_by_zero_surfaces_the_tools_message():
    result = registry.execute(ToolCall(id="c8", name="divide", arguments={"numerator": 1, "denominator": 0}))
    assert result.is_error
    assert "must not be zero" in json.loads(result.content)["message"]


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": "2", "b": "3"}, 5),          # numeric strings
        ({"a": 2.0, "b": 3}, 5),            # integral float
    ],
)
def test_light_coercion_of_integers(raw, expected):
    result = registry.execute(ToolCall(id="c9", name="adder", arguments=raw))
    assert result.content == str(expected)


def test_boolean_argument_must_not_be_a_number():
    result = registry.execute(ToolCall(id="c10", name="greeter", arguments={"name": "x", "excited": 1}))
    assert result.is_error


def test_string_is_accepted_where_a_string_was_declared_from_a_number():
    result = registry.execute(ToolCall(id="c11", name="greeter", arguments={"name": 42}))
    assert result.content == "hi 42"


def test_list_argument_must_not_be_a_bare_string():
    result = registry.execute(ToolCall(id="c12", name="tagged", arguments={"items": "not-a-list"}))
    assert result.is_error


def test_list_items_are_coerced_individually():
    result = registry.execute(ToolCall(id="c13", name="tagged", arguments={"items": ["a", "b"]}))
    assert json.loads(result.content) == ["full:a", "full:b"]


def test_arguments_must_be_an_object():
    result = registry.execute(ToolCall(id="c14", name="adder", arguments=["not", "a", "dict"]))  # type: ignore[arg-type]
    assert result.is_error


def test_result_is_truncated_with_a_marker():
    long_tool = tool(lambda: "x" * 5000, name="longy")
    built = ToolRegistry([long_tool])
    result = built.execute(ToolCall(id="c15", name="longy", arguments={}), result_limit=100)
    assert len(result.content) < 5000
    assert "truncated" in result.content


def test_truncation_reports_the_omitted_amount():
    long_tool = tool(lambda: "y" * 1000, name="longy2")
    built = ToolRegistry([long_tool])
    result = built.execute(ToolCall(id="c16", name="longy2", arguments={}), result_limit=10)
    assert "990 characters" in result.content


def test_result_id_and_name_are_echoed():
    result = registry.execute(ToolCall(id="unique", name="adder", arguments={"a": 1, "b": 1}))
    assert result.call_id == "unique"
    assert result.name == "adder"


def test_non_json_return_value_is_stringified():
    built = ToolRegistry([tool(lambda: (1, 2), name="tup")])
    result = built.execute(ToolCall(id="c17", name="tup", arguments={}))
    assert "1" in result.content and "2" in result.content


def test_registry_add_accepts_a_plain_function():
    def double(x: int) -> int:
        return x * 2

    built = ToolRegistry()
    built.add(double, name="double")
    result = built.execute(ToolCall(id="c18", name="double", arguments={"x": 4}))
    assert result.content == "8"


def test_unannotated_parameter_is_rejected():
    from atheneum.agent.tools import ToolError

    def sloppy(x):
        return x

    with pytest.raises(ToolError, match="unannotated parameter 'x'"):
        tool(sloppy)
