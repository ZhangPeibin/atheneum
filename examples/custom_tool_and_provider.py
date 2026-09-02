#!/usr/bin/env python3
"""Adding a custom tool and a custom provider.

Two extension points matter most, and both are exercised here:

1. A tool is a plain annotated function. Its JSON schema is derived from the
   signature, so there is nothing else to keep in sync.
2. A provider is any object implementing ``complete``. Registering one by name
   makes it available everywhere a provider name is accepted.

Runs offline.

    python examples/custom_tool_and_provider.py
"""

from __future__ import annotations

import datetime as dt
import math

from atheneum.agent.builtin_tools import build_corpus_tools
from atheneum.agent.loop import Agent, AgentConfig
from atheneum.agent.tools import ToolRegistry, tool
from atheneum.core.types import Message, ToolCall
from atheneum.providers.base import Generation, GenerationRequest, Provider, Usage
from atheneum.providers.registry import registry
from atheneum.retrieval.pipeline import Corpus


# ---------------------------------------------------------------------------
# 1. A custom tool. Annotate every parameter: the annotation *is* the schema.
# ---------------------------------------------------------------------------
@tool(name="circle_area", description="Compute the area of a circle from its radius.")
def circle_area(radius: float) -> dict[str, float]:
    if radius < 0:
        # Raising is fine here. The registry converts any exception into an error
        # result, which the model sees and can correct.
        raise ValueError("radius must not be negative")
    return {"radius": radius, "area": round(math.pi * radius**2, 4)}


@tool(name="today", description="Report the current date in ISO-8601 form.")
def today() -> str:
    return dt.date.today().isoformat()


def show_schema() -> None:
    built = ToolRegistry([circle_area, today])
    print("── derived JSON schema ─────────────────────────────────")
    for name, schema in built.schemas().items():
        print(f"  {name}")
        print(f"    description: {schema['description']}")
        print(f"    parameters:  {schema['parameters']}")
    print()

    print("── executing, including a failing call ─────────────────")
    for call in (
        ToolCall(id="1", name="circle_area", arguments={"radius": 3}),
        ToolCall(id="2", name="circle_area", arguments={"radius": -1}),
        ToolCall(id="3", name="circle_area", arguments={"diameter": 6}),
        ToolCall(id="4", name="no_such_tool", arguments={}),
    ):
        result = built.execute(call)
        print(f"  {call.name}{call.arguments} -> error={result.is_error}")
        print(f"    {result.content[:120]}")
    print()


# ---------------------------------------------------------------------------
# 2. A custom provider. This one is deliberately naive so the contract is clear.
# ---------------------------------------------------------------------------
class ShoutingProvider(Provider):
    """Returns the retrieved evidence verbatim, uppercased.

    Real providers call a model. The only obligation is to return a Generation,
    and to ask for tools by returning tool_calls with finish_reason="tool_calls".
    """

    name = "shouting"

    def __init__(self, prefix: str = "EVIDENCE") -> None:
        self.prefix = prefix
        self.turns = 0

    def complete(self, request: GenerationRequest) -> Generation:
        self.turns += 1
        if self.turns == 1 and "search" in request.tools:
            return Generation(
                tool_calls=[
                    ToolCall(id="c1", name="search", arguments={"query": request.last_user_message()})
                ],
                finish_reason="tool_calls",
                usage=Usage(prompt_tokens=len(request.last_user_message()) // 4),
            )
        evidence = [m.content for m in request.tool_results()]
        body = (evidence[0][:400] if evidence else "NOTHING FOUND").upper()
        return Generation(text=f"{self.prefix}: {body}", finish_reason="stop")


def show_provider() -> None:
    registry.register("shouting", "custom", ShoutingProvider)
    print("── registered providers (excerpt) ─────────────────────")
    print(f"  {'shouting' in registry.names()=}")
    print()

    corpus = Corpus.in_memory()
    corpus.add_text(
        "notes.md",
        "Okapi BM25 controls term frequency saturation with k1 and length "
        "normalization with b. Reciprocal rank fusion merges ranked lists.",
    )

    tools = build_corpus_tools(corpus)
    agent = Agent(registry.create("shouting"), tools, config=AgentConfig(max_turns=3))
    run = agent.run("what does k1 control")
    print("── agent run with the custom provider ─────────────────")
    print(f"  {run.answer[:220]}")
    print(f"  turns={run.turns} tool_calls={run.tool_call_count} stopped={run.stopped_reason}")
    print()

    print("── the same corpus, default offline provider ──────────")
    default = Agent(registry.create("offline"), build_corpus_tools(corpus))
    print(default.run("what does k1 control").answer)
    corpus.close()


def main() -> int:
    show_schema()
    show_provider()
    print("\nHistory is just a list of messages:")
    for message in (Message.user("earlier"), Message.assistant("noted")):
        print(f"  {message.role.value}: {message.content}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
