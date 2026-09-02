"""Atheneum: a local-first AI research engine.

Index documents into SQLite, retrieve them with hybrid BM25 + vector search
fused by Reciprocal Rank Fusion, and answer questions through an agent loop that
cites what it found.

Everything works with no API key and no server. The built-in ``offline``
provider is a deterministic extractive engine that speaks the same protocol as a
hosted model, so ``pip install atheneum && ath ask "..."`` gives a real cited
answer on a machine with no network access.

    >>> import atheneum
    >>> corpus = atheneum.Corpus.in_memory()
    >>> _ = corpus.add_text("notes.md", "Reciprocal rank fusion combines ranked lists.")
    >>> [hit.chunk.source for hit in corpus.search("how do we combine ranked lists", top_k=1)]
    ['notes.md']
"""

from __future__ import annotations

import logging as _logging
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version
from typing import Any

from atheneum.config import Config, config_path, default_db_path, load_config, save_config
from atheneum.core.types import Chunk, Document, Message, Role, ToolCall, ToolResult
from atheneum.index.bm25 import BM25Index, BM25Params
from atheneum.index.store import Store
from atheneum.index.vectors import VectorIndex
from atheneum.providers.base import (
    Generation,
    GenerationRequest,
    Provider,
    ProviderError,
    Usage,
)
from atheneum.providers.offline import OfflineProvider
from atheneum.providers.registry import (
    PROVIDER_PROFILES,
    get_provider,
    resolve_provider,
)
from atheneum.retrieval.embedders import HashingEmbedder
from atheneum.retrieval.fusion import RRF_K, RRFFusion, build_fusion
from atheneum.retrieval.pipeline import Corpus, CorpusConfig, EmbedderMismatchError, SearchResult
from atheneum.retrieval.rerank import build_reranker
from atheneum.text.splitter import SplitterConfig
from atheneum.text.tokenizer import tokenize

# Libraries must not configure logging for their callers; a null handler lets
# messages surface only if the application sets up a handler.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

__all__ = [
    "PROVIDER_PROFILES",
    "RRF_K",
    "BM25Index",
    "BM25Params",
    "Chunk",
    "Config",
    "Corpus",
    "CorpusConfig",
    "Document",
    "EmbedderMismatchError",
    "Generation",
    "GenerationRequest",
    "HashingEmbedder",
    "Message",
    "OfflineProvider",
    "Provider",
    "ProviderError",
    "RRFFusion",
    "Role",
    "SearchResult",
    "SplitterConfig",
    "Store",
    "ToolCall",
    "ToolRegistry",
    "ToolResult",
    "Usage",
    "VectorIndex",
    "__version__",
    "ask",
    "build_fusion",
    "build_reranker",
    "config_path",
    "default_db_path",
    "get_provider",
    "load_config",
    "resolve_provider",
    "save_config",
    "search",
    "tokenize",
    "tool",
]


def _resolve_version() -> str:
    try:
        return _version("atheneum")
    except PackageNotFoundError:  # running from a source checkout that is not installed
        return "0.1.0"


__version__ = _resolve_version()


def search(
    query: str,
    *,
    db: str | None = None,
    top_k: int = 5,
    mode: str = "hybrid",
) -> list[SearchResult]:
    """One-shot retrieval against a corpus on disk."""
    corpus = Corpus.open(db) if db else Corpus.in_memory()
    try:
        return corpus.search(query, top_k=top_k, mode=mode)  # type: ignore[arg-type]
    finally:
        corpus.close()


def ask(
    query: str,
    *,
    db: str | None = None,
    provider: Any = None,
    top_k: int = 5,
    max_turns: int = 8,
) -> Any:
    """One-shot grounded answer. Returns an :class:`~atheneum.agent.loop.AgentRun`."""
    from atheneum.agent.builtin_tools import build_corpus_tools
    from atheneum.agent.loop import Agent, AgentConfig

    corpus = Corpus.open(db) if db else Corpus.in_memory()
    try:
        tools = build_corpus_tools(corpus, default_top_k=top_k)
        runner = Agent(
            resolve_provider(provider),
            tools,
            config=AgentConfig(max_turns=max_turns),
        )
        return runner.run(query)
    finally:
        corpus.close()


def __getattr__(name: str) -> Any:
    # The agent subpackage imports lazily so that `import atheneum` never pays
    # for the loop machinery when only retrieval is wanted.
    if name in {"Agent", "AgentConfig", "AgentRun", "Step"}:
        from atheneum.agent import loop

        return getattr(loop, name)
    if name == "ToolRegistry":
        from atheneum.agent.tools import ToolRegistry

        return ToolRegistry
    if name == "tool":
        from atheneum.agent.tools import tool

        return tool
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
