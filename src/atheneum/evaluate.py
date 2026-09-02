"""Retrieval evaluation and benchmarking.

Two things live here because both need the same metrics code:

``run_evaluation`` scores a small labelled dataset that ships with the package.
It is the answer to "does hybrid retrieval actually beat each retriever on
its own, in this implementation" — a claim the README cannot make on taste.

``benchmark_paths`` times ingestion and query latency on real files, which is how
the brute-force vector ceiling gets measured rather than guessed at.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from atheneum.core.types import Document
from atheneum.retrieval.pipeline import Corpus, CorpusConfig, SearchMode, SearchResult
from atheneum.text.splitter import SplitterConfig

__all__ = [
    "QueryCase",
    "RetrievalMetrics",
    "eval_case",
    "measure",
    "recall_at_k",
    "reciprocal_rank",
    "run_evaluation",
]


@dataclass(frozen=True, slots=True)
class QueryCase:
    """One query with the set of passages that would answer it."""

    query: str
    relevant: frozenset[str]
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"query": self.query, "relevant": sorted(self.relevant), "note": self.note}


# A deliberately small, fully synthetic corpus so the numbers below are
# reproducible on any machine with no downloads.
_SAMPLES: tuple[tuple[str, str], ...] = (
    (
        "docs/bm25.md",
        "# Okapi BM25\n\n"
        "Okapi BM25 ranks documents by term frequency with saturation. The parameter k1 controls how quickly "
        "additional occurrences stop mattering, while b controls length normalization. A b of 0 disables length "
        "normalization entirely and a b of 1 applies it fully.\n\n"
        "Terms appearing in more than half the collection receive a negative raw inverse document frequency. "
        "Implementations floor those values at a fraction of the average idf so that common words cannot "
        "penalize the documents containing them.\n",
    ),
    (
        "docs/fusion.md",
        "# Rank fusion\n\n"
        "Reciprocal rank fusion merges several ranked lists by awarding each document a score of one over a "
        "constant plus its rank, then summing across lists. Cormack, Clarke and Buttcher introduced the technique "
        "at SIGIR 2009 and recommended a constant of sixty.\n\n"
        "Fusing on rank rather than score avoids the problem that BM25 scores are unbounded while cosine "
        "similarity is not, so no calibration step is needed between retrievers.\n",
    ),
    (
        "docs/chunking.md",
        "# Chunking\n\n"
        "Chunk size trades recall against precision. Small chunks retrieve precisely but lose surrounding "
        "context, while large chunks provide context yet dilute the embedding and waste the model's context "
        "window on irrelevant sentences.\n\n"
        "A practical compromise is hierarchical splitting: break on paragraph boundaries first, then sentence "
        "boundaries, then characters, and carry a small overlap between adjacent chunks so a fact spanning a "
        "boundary is not lost.\n",
    ),
    (
        "docs/sqlite.md",
        "# Storing the index\n\n"
        "SQLite full text search uses a contentless FTS5 table and returns results through the built in "
        "bm25() ranking function. Its default tokenizer splits on whitespace and punctuation, which means Han "
        "characters are not segmented into words at all.\n\n"
        "Storing embeddings as packed float32 blobs in a ordinary column lets a whole vector table be loaded "
        "into one numpy matrix, turning similarity search into a single matrix vector product.\n",
    ),
    (
        "docs/agents.md",
        "# Agent loops\n\n"
        "An agent loop repeatedly samples a model, executes any tool calls it requested, appends the results, "
        "and samples again until the model stops asking for tools. Every implementation needs a hard bound on "
        "iterations because a model that keeps requesting the same failing tool will otherwise never terminate.\n\n"
        "When a tool raises an exception the conventional choice is to serialize the error back to the model as "
        "the tool result so it can correct its arguments, rather than aborting the run.\n",
    ),
    (
        "docs/embeddings.md",
        "# Embeddings\n\n"
        "A hashing embedding projects tokens and token pairs into a fixed width vector using a signed hash, "
        "so the representation needs no trained model, no vocabulary and no download. It captures term and "
        "phrase overlap and generalizes to unseen text, but it does not model paraphrase the way a neural "
        "encoder trained on contrastive objectives can.\n\n"
        "Sublinear frequency scaling and L2 normalization keep a long passage from dominating purely because it "
        "repeats a word.\n",
    ),
    (
        "docs/deploy.md",
        "# Deployment\n\n"
        "Self hosted retrieval stacks commonly require Docker Compose, a search server such as Elasticsearch or "
        "Vespa, and a Postgres database. The dominant support complaint against such products is that a fresh "
        "clone does not come up: containers fail health checks and the web UI reports a server connection error.\n\n"
        "An embedded database removes that entire failure class because there is no service left to start.\n",
    ),
    (
        "docs/citations.md",
        "# Citations\n\n"
        "A retrieval answer should name the passages it used. Returning source and chunk ordinal alongside the "
        "text lets a renderer emit numbered references the reader can verify, which is the main defence against "
        "a fluent but unsupported summary.\n\n"
        "Recording the per retriever contribution for each surviving passage also lets a debug view explain why "
        "something ranked first.\n",
    ),
    # --- distractors -----------------------------------------------------
    # Adjacent-topic passages that share vocabulary with the queries above but
    # do not answer them. Without these the dataset is trivially separable and
    # every retriever scores 1.0, which proves nothing.
    (
        "docs/glossary.md",
        "# Glossary\n\n"
        "Corpus: the body of indexed text. Passage: one chunk of a document. Rank: the position of a "
        "passage in an ordered result list. Score: a numeric measure of relevance whose scale depends "
        "entirely on the algorithm that produced it, so scores from different algorithms are not "
        "comparable without normalization.\n",
    ),
    (
        "docs/tokenizers.md",
        "# Tokenizers\n\n"
        "A tokenizer turns text into terms. Whitespace tokenizers are fast but wrong for languages "
        "written without spaces. Subword tokenizers such as byte pair encoding are trained on a corpus "
        "and produce a fixed vocabulary. Character n-gram tokenizers need no training data and handle "
        "unseen words by construction.\n",
    ),
    (
        "docs/caching.md",
        "# Caching\n\n"
        "A query cache stores answers keyed by the normalized query string. Hit rate depends on query "
        "distribution; a long tail of unique questions keeps it low. Invalidation is the hard part, "
        "because a cache entry becomes wrong the moment the underlying index changes.\n",
    ),
    (
        "docs/evaluation-metrics.md",
        "# Evaluation metrics\n\n"
        "Recall at k measures what fraction of relevant passages appear in the top k results. Precision "
        "at k measures how many of those k are relevant. Mean reciprocal rank rewards putting the first "
        "relevant passage as high as possible. Normalized discounted cumulative gain additionally "
        "accounts for graded relevance and position.\n",
    ),
    (
        "docs/stopwords.md",
        "# Stopwords\n\n"
        "Removing common function words shrinks the index and removes terms whose inverse document "
        "frequency is near zero anyway. The risk is losing phrases whose meaning depends on them, such "
        "as 'to be or not to be'. Most modern engines keep stopwords and let the weighting handle them.\n",
    ),
    (
        "docs/stemming.md",
        "# Stemming and lemmatization\n\n"
        "A stemmer reduces words to a crude root, so running and runs collapse to run. A lemmatizer uses "
        "a dictionary and returns a real word. Stemming is faster and occasionally wrong in a way that "
        "merges unrelated words, which quietly degrades precision.\n",
    ),
    (
        "docs/monitoring.md",
        "# Monitoring\n\n"
        "Emit query latency percentiles rather than averages, because retrieval latency is skewed by "
        "rare large queries. Track the fraction of queries that return no results at all; a sudden rise "
        "usually means an ingestion failure rather than a change in user behaviour.\n",
    ),
    (
        "docs/multitenancy.md",
        "# Multi tenancy\n\n"
        "Isolating tenants can be done with a separate index per tenant or with a filter column on a "
        "shared index. Separate indexes give strong isolation and poor resource efficiency at small "
        "scale. A shared index with filters is the opposite trade.\n",
    ),
    (
        "docs/incremental-indexing.md",
        "# Incremental indexing\n\n"
        "Re-indexing an entire corpus on every change is simple and becomes unbearable as the corpus "
        "grows. Incremental updates require stable identifiers for passages so that a changed document "
        "replaces its own old passages rather than duplicating them.\n",
    ),
    (
        "docs/reranking-cost.md",
        "# Reranking cost\n\n"
        "A cross encoder reads the query and the passage together, which is far more accurate than "
        "comparing independent embeddings and far more expensive. It is therefore applied only to a "
        "small candidate set produced by a cheap first stage.\n",
    ),
)

_CASES: tuple[QueryCase, ...] = (
    QueryCase("what does the b parameter control", frozenset({"docs/bm25.md"}), "length normalization"),
    QueryCase("negative inverse document frequency floor", frozenset({"docs/bm25.md"}), "exact terminology"),
    QueryCase("Cormack Clarke Buttcher SIGIR 2009", frozenset({"docs/fusion.md"}), "named entities"),
    QueryCase("how to merge several ranked lists", frozenset({"docs/fusion.md"}), "paraphrase of heading"),
    QueryCase("small chunks lose surrounding context", frozenset({"docs/chunking.md"}), ""),
    QueryCase("why does chunk overlap matter", frozenset({"docs/chunking.md"}), "reasoning over prose"),
    QueryCase("FTS5 tokenizer does not segment Han characters", frozenset({"docs/sqlite.md"}), "specific claim"),
    QueryCase("packed float32 blobs into one matrix", frozenset({"docs/sqlite.md"}), "implementation detail"),
    QueryCase("a model keeps calling the same failing tool", frozenset({"docs/agents.md"}), "termination"),
    QueryCase("serialize the exception back to the model", frozenset({"docs/agents.md"}), "error handling"),
    QueryCase("no trained model and no vocabulary needed", frozenset({"docs/embeddings.md"}), ""),
    QueryCase("fresh clone does not come up containers fail health checks", frozenset({"docs/deploy.md"}), "pain point"),
    QueryCase("numbered references the reader can verify", frozenset({"docs/citations.md"}), ""),
    QueryCase("explain why a passage ranked first", frozenset({"docs/citations.md", "docs/fusion.md"}), "two plausible"),
    # --- paraphrase queries ---------------------------------------------
    # Deliberately low lexical overlap with the target passage. These are the
    # cases that separate a dense retriever from a keyword one; if every query
    # shared vocabulary with its answer, the comparison would be meaningless.
    QueryCase(
        "how do you combine results from two different search systems",
        frozenset({"docs/fusion.md"}),
        "paraphrase: no shared content words with 'rank fusion'",
    ),
    QueryCase(
        "why do scores from different algorithms need normalizing",
        frozenset({"docs/fusion.md", "docs/glossary.md"}),
        "paraphrase with a distractor also plausible",
    ),
    QueryCase(
        "make long documents count for less than short ones",
        frozenset({"docs/bm25.md"}),
        "paraphrase of length normalization",
    ),
    QueryCase(
        "how often should a word appear before extra mentions stop helping",
        frozenset({"docs/bm25.md"}),
        "paraphrase of term frequency saturation",
    ),
    QueryCase(
        "what happens when you cut text into pieces that are too small",
        frozenset({"docs/chunking.md"}),
        "paraphrase of the precision/context trade-off",
    ),
    QueryCase(
        "a way to store vectors without running a separate database service",
        frozenset({"docs/sqlite.md"}),
        "paraphrase of packed blobs in a column",
    ),
    QueryCase(
        "software that needs no server and starts immediately",
        frozenset({"docs/deploy.md"}),
        "paraphrase of the embedded database claim",
    ),
    QueryCase(
        "an embedding method that works without downloading a trained model",
        frozenset({"docs/embeddings.md"}),
        "paraphrase of hashing embeddings",
    ),
    # --- distractor bait ------------------------------------------------
    # Queries whose vocabulary overlaps a distractor more than the true answer.
    QueryCase(
        "what makes the first relevant result appear as early as possible",
        frozenset({"docs/evaluation-metrics.md"}),
        "MRR is defined in the metrics distractor, not in fusion",
    ),
    QueryCase(
        "how to reduce a word to its root form",
        frozenset({"docs/stemming.md"}),
        "stemming distractor is the true answer here",
    ),
    QueryCase(
        "which latency statistic hides rare slow queries",
        frozenset({"docs/monitoring.md"}),
        "averages versus percentiles",
    ),
    QueryCase(
        "keeping passage identifiers stable when a document changes",
        frozenset({"docs/incremental-indexing.md"}),
        "incremental indexing",
    ),
)


@dataclass(slots=True)
class RetrievalMetrics:
    """Averages over the evaluation queries."""

    name: str
    recall_at_k: float = 0.0
    precision_at_k: float = 0.0
    mrr: float = 0.0
    hit_rate: float = 0.0
    mean_latency_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "recall_at_k": round(self.recall_at_k, 4),
            "precision_at_k": round(self.precision_at_k, 4),
            "mrr": round(self.mrr, 4),
            "hit_rate": round(self.hit_rate, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 4),
        }


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    hits = len(set(retrieved[:k]) & set(relevant))
    total = len(set(relevant))
    return hits / total if total else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    wanted = set(relevant)
    if k <= 0:
        return 0.0
    return len(set(retrieved[:k]) & wanted) / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    wanted = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in wanted:
            return 1.0 / position
    return 0.0


def build_eval_corpus(config: CorpusConfig | None = None) -> Corpus:
    """Index the bundled sample documents in memory."""
    corpus = Corpus.in_memory(
        config=config
        or CorpusConfig(splitter=SplitterConfig(chunk_size=900, chunk_overlap=150))
    )
    corpus.add_documents([Document(source=source, content=text, title=source) for source, text in _SAMPLES])
    return corpus


def run_evaluation(*, top_k: int = 5, modes: Sequence[SearchMode] = ("hybrid", "lexical", "vector")) -> dict[str, Any]:
    """Score each retrieval mode over the labelled queries."""
    corpus = build_eval_corpus()
    per_mode: dict[str, RetrievalMetrics] = {mode: RetrievalMetrics(name=mode) for mode in modes}
    details: list[dict[str, Any]] = []

    try:
        for case in _CASES:
            row: dict[str, Any] = {"query": case.query, "relevant": sorted(case.relevant), "note": case.note}
            for mode in modes:
                start = time.perf_counter()
                results = corpus.search(case.query, top_k=top_k * 2, mode=mode)
                elapsed = (time.perf_counter() - start) * 1000.0
                sources = [r.chunk.source for r in results]
                metrics = per_mode[mode]
                metrics.recall_at_k += recall_at_k(sources, case.relevant, top_k) / len(_CASES)
                metrics.precision_at_k += precision_at_k(sources, case.relevant, top_k) / len(_CASES)
                metrics.mrr += reciprocal_rank(sources, case.relevant) / len(_CASES)
                metrics.hit_rate += (1.0 if set(sources[:top_k]) & set(case.relevant) else 0.0) / len(_CASES)
                metrics.mean_latency_ms += elapsed / len(_CASES)
                row[mode] = {
                    "recall_at_k": round(recall_at_k(sources, case.relevant, top_k), 4),
                    "mrr": round(reciprocal_rank(sources, case.relevant), 4),
                    "top_sources": list(dict.fromkeys(sources))[:3],
                }
            # Headline figure for the primary mode, so callers can filter weak
            # queries without knowing which retriever names were evaluated.
            primary = modes[0]
            row["recall_at_k"] = row[primary]["recall_at_k"]
            row["mrr"] = row[primary]["mrr"]
            row["k"] = top_k
            details.append(row)
    finally:
        corpus.close()

    payload = {
        "corpus": {
            "documents": len(_SAMPLES),
            "chunks": corpus_store_size(),
        },
        "k": top_k,
        "query_count": len(_CASES),
        "results": [metrics.as_dict() for metrics in per_mode.values()],
        "hybrid_beats_both_retrievers": _hybrid_wins(per_mode, modes),
        "queries": details,
    }
    return payload


def corpus_store_size() -> int:
    corpus = build_eval_corpus()
    try:
        return int(corpus.stats()["chunks"])
    finally:
        corpus.close()


def _hybrid_wins(per_mode: dict[str, RetrievalMetrics], modes: Sequence[str]) -> bool | None:
    """True only if fusion beats *every* single retriever on both recall and MRR.

    Deliberately strict. A looser rule — beating either metric against either
    retriever — reported a win here while equal-weight RRF was actually scoring
    below lexical-only on MRR, which is the kind of flattering number that gets
    quoted in a README and then fails in production.
    """
    if "hybrid" not in per_mode:
        return None
    hybrid = per_mode["hybrid"]
    others = [m for name, m in per_mode.items() if name != "hybrid" and name in modes]
    if not others:
        return None
    return all(
        hybrid.recall_at_k >= m.recall_at_k and hybrid.mrr >= m.mrr for m in others
    )


def measure(fn: Callable[[], Any], *, repeats: int = 5) -> dict[str, float]:
    """Time a callable, returning mean and fastest wall time in milliseconds."""
    timings: list[float] = []
    result: Any = None
    for _ in range(max(1, repeats)):
        start = time.perf_counter()
        result = fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    return {
        "mean_ms": statistics.fmean(timings),
        "min_ms": min(timings),
        "max_ms": max(timings),
        "stdev_ms": statistics.pstdev(timings) if len(timings) > 1 else 0.0,
        "result_count": len(result) if isinstance(result, list) else 1,
    }


def benchmark_paths(
    paths: Sequence[str | Path],
    *,
    queries: Sequence[str] = (),
    top_k: int = 5,
    mode: SearchMode | None = None,
    db: str | None = None,
    repeats: int = 20,
) -> dict[str, Any]:
    """Ingest real files and time retrieval against them.

    ``mode`` defaults to hybrid; ``None`` is accepted because CLI flags pass
    through unset options as ``None`` and the caller should not have to know.
    """
    resolved_mode: SearchMode = mode or "hybrid"
    target = Path(db) if db else Path(".bench_corpus.db")
    if target.exists():
        target.unlink()
    corpus = Corpus.open(target, config=CorpusConfig(splitter=SplitterConfig(chunk_size=1000, chunk_overlap=200)))
    try:
        start = time.perf_counter()
        added = corpus.add_paths(paths)
        ingest_ms = (time.perf_counter() - start) * 1000.0

        stats = corpus.stats()
        query_names = list(queries) or _sample_queries(corpus)
        timings: list[dict[str, Any]] = []
        for query in query_names:
            timing = measure(_timed_search(corpus, query, top_k, resolved_mode), repeats=repeats)
            hits = corpus.search(query, top_k=top_k, mode=resolved_mode)
            timings.append({"query": query, **timing, "hits": len(hits)})
    finally:
        corpus.close()
        target.unlink(missing_ok=True)
        Path(str(target) + "-wal").unlink(missing_ok=True)
        Path(str(target) + "-shm").unlink(missing_ok=True)

    latencies = [row["mean_ms"] for row in timings]
    return {
        "mode": resolved_mode,
        "corpus": stats,
        "ingest": {"chunks": added, "wall_ms": round(ingest_ms, 2), "chunks_per_second": round(added / (ingest_ms / 1000.0), 1) if ingest_ms > 0 else None},
        "queries": timings,
        "summary": {
            "query_count": len(timings),
            "mean_query_ms": round(statistics.fmean(latencies), 3) if latencies else None,
            "p95_query_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1], 3) if latencies else None,
        },
    }


def _timed_search(corpus: Corpus, query: str, top_k: int, mode: SearchMode) -> Callable[[], list[SearchResult]]:
    """Return a zero-argument thunk so `measure` has a concrete type to infer."""

    def run() -> list[SearchResult]:
        return list(corpus.search(query, top_k=top_k, mode=mode))

    return run


def _sample_queries(corpus: Corpus, count: int = 6) -> list[str]:
    """Take distinctive terms from the corpus so timings reflect real work."""
    chunks = corpus.chunks()
    queries: list[str] = []
    for chunk in chunks[:: max(1, len(chunks) // max(1, count))][:count]:
        words = [w for w in chunk.text.split() if len(w) > 6]
        if words:
            queries.append(" ".join(words[:4]))
    return queries or ["retrieval ranking fusion"]


def as_json_report(report: dict[str, Any]) -> str:
    return json.dumps(report, indent=2, sort_keys=True, default=str)


def eval_case(query: str, relevant: Sequence[str], note: str = "") -> QueryCase:
    return QueryCase(query=query, relevant=frozenset(relevant), note=note)


def metrics_to_rows(metrics: Iterable[RetrievalMetrics]) -> list[dict[str, Any]]:
    return [asdict(m) for m in metrics]
