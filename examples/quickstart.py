#!/usr/bin/env python3
"""End-to-end example: index, retrieve, and answer with citations.

Runs offline with no API key and no network. Prints every stage so the pipeline
is visible rather than magic.

    python examples/quickstart.py
"""

from __future__ import annotations

import atheneum
from atheneum.agent.builtin_tools import build_corpus_tools
from atheneum.agent.loop import Agent, AgentConfig
from atheneum.retrieval.pipeline import Corpus
from atheneum.text.splitter import SplitterConfig

DOCS = {
    "adr/001-rate-limiting.md": """# ADR 001 — Rate limiting

## Decision
We will rate limit at the edge using a token bucket per API key, permitting bursts
up to 200 requests while enforcing a long-run average of 20 requests per second.

## Consequences
Early rejection keeps one abusive tenant from degrading every other tenant. The
cost is that legitimate bursts above 200 requests are refused rather than queued.
""",
    "adr/002-queue-depth.md": """# ADR 002 — Queue depth

## Decision
The ingest queue is capped at 10,000 messages. When the cap is reached producers
receive backpressure instead of the queue growing without bound.

## Consequences
Memory usage becomes predictable under load. Producers must handle a rejected
enqueue, which they previously did not.
""",
    "runbook/incident-2026-04.md": """# Incident 2026-04-11

Ingest lag grew for 40 minutes because the consumer restarted in a loop. The queue
depth hit its cap and producers began receiving backpressure.

## Resolution
The restart loop was caused by a missing environment variable. Rate limiting was
not the cause, but the token bucket did prevent the retries from reaching the API.
""",
}


def main() -> int:
    print(f"atheneum {atheneum.__version__}\n")

    corpus = Corpus.in_memory()
    corpus.config.splitter = SplitterConfig(chunk_size=400, chunk_overlap=40)
    for source, text in DOCS.items():
        corpus.add_text(source, text, title=source.split("/")[-1])

    stats = corpus.stats()
    print(f"indexed {stats['documents']} documents into {stats['chunks']} chunks")
    print(f"embedder: {stats['embedder']['name']} (dim {stats['embedder']['dim']})")
    print(f"fusion: {stats['config']['fusion']} k={stats['config']['fusion_k']}\n")

    print("── retrieval ────────────────────────────────────────────")
    query = "what did we decide about queue depth"
    for hit in corpus.search(query, top_k=3):
        breakdown = "  ".join(f"{name}={value:.5f}" for name, value in sorted(hit.contributions.items()))
        print(f"  {hit.score:.5f}  {hit.chunk.source}#chunk{hit.chunk.ordinal}")
        print(f"           {breakdown}")
    print()

    print("── lexical vs vector vs hybrid ─────────────────────────")
    for mode in ("lexical", "vector", "hybrid"):
        hits = corpus.search(query, top_k=1, mode=mode)
        source = hits[0].chunk.source if hits else "(none)"
        print(f"  {mode:<8} -> {source}")
    print()

    print("── agent, offline provider ─────────────────────────────")
    agent = Agent(
        atheneum.get_provider("offline"),
        build_corpus_tools(corpus),
        config=AgentConfig(max_turns=4),
    )
    run = agent.run(query)
    print(run.answer)
    print()
    print(f"turns={run.turns} tool_calls={run.tool_call_count} stopped={run.stopped_reason}")
    print(f"tokens(prompt/completion/total)={run.usage.prompt_tokens}/"
          f"{run.usage.completion_tokens}/{run.usage.total_tokens}")

    corpus.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
