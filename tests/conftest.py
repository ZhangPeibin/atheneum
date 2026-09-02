"""Shared fixtures. Everything here runs offline with no credentials."""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator

import pytest

os.environ.setdefault("ATHENEUM_NO_PLUGINS", "1")

import atheneum
from atheneum.core.types import Document
from atheneum.retrieval.pipeline import Corpus, CorpusConfig
from atheneum.text.splitter import SplitterConfig

BM25_DOC = """# Okapi BM25

Okapi BM25 ranks documents by term frequency with saturation. The k1 parameter controls how quickly
additional occurrences stop mattering, while the b parameter controls length normalization. A b value
of zero disables length normalization entirely.

Terms appearing in more than half the collection receive a negative raw inverse document frequency,
which implementations floor at a fraction of the average idf.
"""

FUSION_DOC = """# Rank fusion

Reciprocal rank fusion merges several ranked lists by awarding each document a score of one over a
constant plus its rank, then summing across lists. Cormack, Clarke and Buttcher proposed it at
SIGIR 2009 with a constant of sixty.

Fusing on rank rather than score avoids calibrating an unbounded BM25 score against bounded cosine
similarity.
"""

AGENT_DOC = """# Agent loops

An agent loop samples a model, executes the tool calls it requested, appends the results and samples
again. Every implementation needs a hard bound on iterations, because a model that keeps requesting
the same failing tool never terminates. When a tool raises, the conventional choice is to return the
error to the model as the tool result instead of aborting the run.
"""

CHINESE_DOC = """# 混合检索

混合检索使用倒数排名融合来合并多个排序列表。BM25 负责词法匹配，向量检索负责语义匹配。
融合之后再做重排序，可以显著提高召回质量。
"""

CODE_DOC = '''# Rendering

The renderer walks the tree depth first.

```python
def render(node):
    for child in node.children:
        yield render(child)
    return node.value
```

Inline code like `node.value` should survive chunking.
'''


@pytest.fixture
def documents() -> list[Document]:
    return [
        Document(source="docs/bm25.md", content=BM25_DOC, title="BM25"),
        Document(source="docs/fusion.md", content=FUSION_DOC, title="Fusion"),
        Document(source="docs/agents.md", content=AGENT_DOC, title="Agents"),
        Document(source="docs/multilingual.md", content=CHINESE_DOC, title="混合检索"),
    ]


@pytest.fixture
def corpus(documents: list[Document]) -> Iterator[Corpus]:
    built = Corpus.in_memory(
        config=CorpusConfig(splitter=SplitterConfig(chunk_size=500, chunk_overlap=80))
    )
    built.add_documents(documents)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture
def empty_corpus() -> Iterator[Corpus]:
    built = Corpus.in_memory()
    try:
        yield built
    finally:
        built.close()


@pytest.fixture
def db_path(tmp_path) -> Iterator[str]:
    path = tmp_path / "corpus.db"
    yield str(path)
    for extra in ("", "-wal", "-shm"):
        with contextlib.suppress(FileNotFoundError):
            os.unlink(str(path) + extra)


@pytest.fixture
def offline_provider() -> atheneum.OfflineProvider:
    return atheneum.OfflineProvider()


@pytest.fixture
def code_document() -> Document:
    return Document(source="src/render.md", content=CODE_DOC, title="Rendering")
