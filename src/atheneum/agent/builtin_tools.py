"""Tools that operate on a :class:`~atheneum.retrieval.pipeline.Corpus`.

These are the tools the agent is actually pointed at. Each returns a plain dict
so the JSON schema generator can describe its inputs and the model sees
structured output rather than a printed blob.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from atheneum.agent.tools import ToolRegistry, tool
from atheneum.retrieval.pipeline import Corpus

__all__ = ["build_corpus_tools"]


def build_corpus_tools(
    corpus: Corpus,
    *,
    default_top_k: int = 5,
    max_top_k: int = 25,
    modes: Sequence[str] = ("hybrid", "lexical", "vector"),
) -> ToolRegistry:
    """Create a tool registry whose tools read from ``corpus``.

    Every tool here is read-only, so the resulting agent cannot mutate the
    corpus no matter what the provider asks for.
    """

    @tool(
        name="search",
        description=(
            "Search the indexed corpus. Returns the most relevant passages with "
            "their source, chunk number and score. Use mode 'lexical' for exact "
            "identifiers or error strings, 'vector' for concepts, and the default "
            "'hybrid' otherwise."
        ),
    )
    def search(query: str, top_k: int = default_top_k, mode: str = "hybrid") -> dict[str, Any]:
        if not query.strip():
            return {"query": query, "results": [], "note": "empty query"}
        if mode not in modes:
            raise ValueError(f"mode must be one of {list(modes)}, got {mode!r}")
        bounded = max(1, min(int(top_k), max_top_k))
        results = corpus.search(query, top_k=bounded, mode=mode)  # type: ignore[arg-type]
        return {
            "query": query,
            "mode": mode,
            "count": len(results),
            "results": [entry.as_dict() for entry in results],
        }

    @tool(
        name="read_chunk",
        description="Fetch the full text of one chunk by its chunk_id, for when a search snippet is truncated.",
    )
    def read_chunk(chunk_id: str) -> dict[str, Any]:
        chunk = corpus.get_chunk(chunk_id)
        if chunk is None:
            # Returning a not-found result rather than raising lets the model see
            # the miss and try a different id instead of retrying blindly.
            return {"error": "not_found", "chunk_id": chunk_id}
        return {
            "chunk_id": chunk.id,
            "source": chunk.source,
            "ordinal": chunk.ordinal,
            "text": chunk.text,
            "metadata": chunk.metadata,
        }

    @tool(name="read_source", description="Read a whole document by source path, capped at a character limit.")
    def read_source(source: str, max_chars: int = 8000) -> dict[str, Any]:
        document = corpus.store.find_document_by_source(source)
        if document is None:
            matches = [row for row in corpus.sources(limit=1000) if source in row["source"]]
            if not matches:
                return {"error": "not_found", "source": source}
            document = corpus.store.get_document(matches[0]["id"])
        if document is None:
            return {"error": "not_found", "source": source}
        bounded = max(200, min(int(max_chars), 40_000))
        text = document.content
        return {
            "source": document.source,
            "title": document.title,
            "total_chars": len(text),
            "text": text[:bounded] + ("\n…[truncated]" if len(text) > bounded else ""),
        }

    @tool(name="list_sources", description="List the documents currently indexed, with chunk counts.")
    def list_sources(limit: int = 50) -> dict[str, Any]:
        rows = corpus.sources(limit=max(1, min(int(limit), 500)))
        return {
            "count": len(rows),
            "sources": [
                {"source": r["source"], "title": r.get("title"), "chunks": r.get("chunk_count", 0)}
                for r in rows
            ],
        }

    @tool(name="corpus_info", description="Report index size, embedding backend and active retrieval settings.")
    def corpus_info() -> dict[str, Any]:
        return corpus.stats()

    return ToolRegistry([search, read_chunk, read_source, list_sources, corpus_info])


def tool_descriptions(registry: ToolRegistry) -> list[str]:
    lines: list[str] = []
    for name in registry.names():
        definition = registry.get(name)
        if definition is not None:
            lines.append(f"{name}: {definition.description}")
    return lines
