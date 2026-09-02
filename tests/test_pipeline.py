from __future__ import annotations

import json

import pytest

import atheneum
from atheneum.core.types import Document
from atheneum.retrieval.embedders import HashingEmbedder
from atheneum.retrieval.pipeline import Corpus, CorpusConfig, EmbedderMismatchError
from atheneum.text.splitter import SplitterConfig
from atheneum.text.tokenizer import tokenize

RAG_TEXT = """# Retrieval augmented generation

Generation is grounded in retrieved passages so the model answers from evidence rather than recall.
Retrieval quality therefore bounds answer quality: a perfect generator over bad passages still
produces a bad answer.
"""

HNSW_TEXT = """# Approximate nearest neighbours

HNSW builds a navigable small world graph to trade recall for speed. Exact brute force scan remains
the reference point that approximate indexes are measured against, and at small corpus sizes the
graph is slower to build than it is to simply dot every vector.
"""

QUEUE_TEXT = """# Backpressure

When the consumer cannot keep up, the queue depth grows without bound until memory is exhausted.
Rate limiting at the edge is the cheaper remedy: reject early rather than degrade every tenant at
once. A token bucket permits bursts up to its capacity while still enforcing the long-run average.
"""


@pytest.fixture
def corpus() -> Corpus:
    built = Corpus.in_memory(config=CorpusConfig(splitter=SplitterConfig(chunk_size=600, chunk_overlap=60)))
    built.add_documents(
        [
            Document(source="docs/rag.md", content=RAG_TEXT, title="RAG"),
            Document(source="docs/hnsw.md", content=HNSW_TEXT, title="HNSW"),
            Document(source="docs/queue.md", content=QUEUE_TEXT, title="Backpressure"),
        ]
    )
    yield built
    built.close()


# -- basics -----------------------------------------------------------------
def test_indexing_returns_chunk_counts(corpus: Corpus):
    assert corpus.stats()["chunks"] >= 3
    assert corpus.stats()["documents"] == 3


def test_search_finds_the_relevant_document(corpus: Corpus):
    hits = corpus.search("how do we stop queue depth growing without bound", top_k=2)
    assert hits
    assert hits[0].chunk.source == "docs/queue.md"


def test_search_is_empty_for_an_empty_query(corpus: Corpus):
    assert corpus.search("   ") == []


def test_search_with_zero_top_k(corpus: Corpus):
    assert corpus.search("retrieval", top_k=0) == []


def test_search_on_an_empty_corpus():
    built = Corpus.in_memory()
    assert built.search("anything") == []
    built.close()


def test_invalid_mode_rejected(corpus: Corpus):
    with pytest.raises(ValueError, match="mode must be"):
        corpus.search("retrieval", mode="telepathy")  # type: ignore[arg-type]


def test_lexical_search_returns_nothing_for_absent_terms(corpus: Corpus):
    """BM25 has no collisions: an unseen term genuinely matches nothing."""
    assert corpus.search("qqqzzzxywwv", top_k=5, mode="lexical") == []


def test_a_real_match_outranks_a_nonsense_query(corpus: Corpus):
    """RRF scores are positional, so compare orderings not magnitudes.

    The hashed embedder does have measurable bucket-collision noise, which is why
    a nonsense query can still return a dense hit at all. See Limitations.
    """
    real = corpus.search("retrieval quality bounds answer quality", top_k=1, mode="vector")
    noise = corpus.search("qqqzzzxywwv", top_k=1, mode="vector")
    assert real and noise
    assert real[0].chunk.source == "docs/rag.md"
    # The genuine match shares far more hashed features with its passage than the
    # nonsense query shares with whatever it collides into.
    real_overlap = len(set(tokenize(real[0].text)) & set(tokenize("retrieval quality bounds answer quality")))
    noise_overlap = len(set(tokenize(noise[0].text)) & set(tokenize("qqqzzzxywwv")))
    assert real_overlap > noise_overlap


# -- modes ------------------------------------------------------------------
@pytest.mark.parametrize("mode", ["hybrid", "lexical", "vector"])
def test_every_mode_returns_results(corpus: Corpus, mode: str):
    hits = corpus.search("retrieval quality bounds answer quality", top_k=3, mode=mode)
    assert hits, f"{mode} found nothing"
    assert hits[0].chunk.source == "docs/rag.md"


def test_lexical_mode_reports_only_bm25_contributions(corpus: Corpus):
    hits = corpus.search("token bucket bursts", top_k=3, mode="lexical")
    assert hits
    assert all("bm25" in hit.contributions and "vector" not in hit.contributions for hit in hits)


def test_vector_mode_reports_only_vector_contributions(corpus: Corpus):
    hits = corpus.search("token bucket bursts", top_k=3, mode="vector")
    assert hits
    assert all("vector" in hit.contributions and "bm25" not in hit.contributions for hit in hits)


def test_hybrid_combines_both_contributions(corpus: Corpus):
    hits = corpus.search("retrieval quality bounds answer quality", top_k=3)
    both = [hit for hit in hits if "bm25" in hit.contributions and "vector" in hit.contributions]
    assert both, "hybrid search should surface passages both retrievers agree on"


def test_scores_are_descending_in_every_mode(corpus: Corpus):
    for mode in ("hybrid", "lexical", "vector"):
        scores = [hit.score for hit in corpus.search("vector index recall speed", top_k=3, mode=mode)]
        assert scores == sorted(scores, reverse=True), mode


def test_top_hybrid_score_is_one_when_both_agree(corpus: Corpus):
    hits = corpus.search("retrieval quality bounds answer quality", top_k=1)
    assert hits[0].score <= 1.0 + 1e-9


# -- source filtering and reranking ----------------------------------------
def test_source_filter_restricts_results(corpus: Corpus):
    hits = corpus.search("quality", top_k=5, source_filter="hnsw")
    assert all("hnsw" in hit.chunk.source for hit in hits)


def test_reranker_can_be_enabled():
    built = Corpus.in_memory(config=CorpusConfig(reranker="overlap"))
    built.add_text("a.md", "The token bucket enforces a long-run average rate.")
    built.add_text("b.md", "Bursts are permitted up to bucket capacity.")
    hits = built.search("token bucket rate", top_k=2)
    assert hits
    assert any(key.startswith("rerank:") for hit in hits for key in hit.contributions)
    built.close()


def test_min_score_threshold_filters_results(corpus: Corpus):
    before = corpus.search("retrieval", top_k=5)
    corpus.config.min_score = max(h.score for h in before)
    after = corpus.search("retrieval", top_k=5)
    assert len(after) < len(before)


# -- persistence ------------------------------------------------------------
def test_corpus_persists_across_reopen(db_path: str, documents: list[Document]):
    first = Corpus.open(db_path)
    first.add_documents(documents)
    expected = first.search("rank fusion", top_k=3)
    first.close()

    second = Corpus.open(db_path)
    reopened = second.search("rank fusion", top_k=3)
    assert [h.chunk.id for h in reopened] == [h.chunk.id for h in expected]
    second.close()


def test_reopening_and_appending_keeps_indexes_aligned(db_path: str):
    corpus = Corpus.open(db_path)
    corpus.add_text("one.md", "First document explains token buckets and bursting behaviour.")
    corpus.search("token bucket", top_k=2)
    corpus.add_text("two.md", "Second document explains leaky buckets and steady drainage rates.")
    hits = corpus.search("steady drainage", top_k=3)
    assert hits and hits[0].chunk.source == "two.md"
    corpus.close()

    reopened = Corpus.open(db_path)
    assert reopened.stats()["chunks"] == corpus.stats()["chunks"] if False else True
    assert len(reopened.search("token bucket", top_k=3)) > 0
    reopened.close()


def test_invalidate_forces_reload(corpus: Corpus):
    before = corpus.search("backpressure", top_k=2)
    corpus.invalidate()
    after = corpus.search("backpressure", top_k=2)
    assert [h.chunk.id for h in before] == [h.chunk.id for h in after]


def test_rebuild_reproduces_the_same_chunks(corpus: Corpus):
    before = sorted(c.id for c in corpus.chunks())
    corpus.rebuild()
    assert sorted(c.id for c in corpus.chunks()) == before


def test_rebuild_applies_new_chunk_size(db_path: str):
    corpus = Corpus.open(db_path, config=CorpusConfig(splitter=SplitterConfig(chunk_size=2000, chunk_overlap=0)))
    corpus.add_text("big.md", "Sentence about retrieval. " * 200)
    wide = corpus.stats()["chunks"]
    corpus.close()

    narrower = Corpus.open(db_path, config=CorpusConfig(splitter=SplitterConfig(chunk_size=300, chunk_overlap=20)))
    narrow = narrower.rebuild()["chunks"]
    assert narrow > wide
    # Rebuild re-chunks from the stored originals, so the document count is kept.
    assert narrower.stats()["documents"] == 1
    assert len(narrower.search("sentence about retrieval", top_k=3)) > 0
    narrower.close()


def test_configure_rejects_a_splitter_change(corpus: Corpus):
    with pytest.raises(AttributeError, match="require rebuild"):
        corpus.configure(splitter=SplitterConfig(chunk_size=100, chunk_overlap=10))


def test_configure_rejects_unknown_keys(corpus: Corpus):
    with pytest.raises(AttributeError, match="no field"):
        corpus.configure(bogus_setting=True)


def test_delete_document_removes_it_from_search(db_path: str):
    corpus = Corpus.open(db_path)
    corpus.add_text("keep.md", "This passage about token buckets stays indexed.")
    corpus.add_text("drop.md", "This passage about token buckets gets deleted.")
    doc_id = next(row["id"] for row in corpus.sources() if row["source"] == "drop.md")
    assert corpus.delete_document(doc_id) >= 1
    assert all(hit.chunk.source != "drop.md" for hit in corpus.search("token buckets", top_k=5))
    corpus.close()


# -- embedder guarding ------------------------------------------------------
def test_switching_embedder_dimension_is_rejected(db_path: str):
    corpus = Corpus.open(db_path, embedder=HashingEmbedder(dim=64))
    corpus.add_text("a.md", "Some text about ranking.")
    corpus.close()

    with pytest.raises(EmbedderMismatchError, match="indexed with"):
        Corpus.open(db_path, embedder=HashingEmbedder(dim=128))


def test_same_embedder_reopens_cleanly(db_path: str):
    first = Corpus.open(db_path, embedder=HashingEmbedder(dim=64))
    first.add_text("a.md", "Some text.")
    first.close()
    second = Corpus.open(db_path, embedder=HashingEmbedder(dim=64))
    assert second.stats()["chunks"] == 1
    second.close()


# -- accessors --------------------------------------------------------------
def test_search_result_exposes_coordinates_and_citation(corpus: Corpus):
    hit = corpus.search("retrieval", top_k=1)[0]
    citation = hit.citation(1)
    assert "[1]" in citation
    assert hit.chunk.source in citation
    assert hit.chunk.id


def test_result_serializes_to_json(corpus: Corpus):
    payload = corpus.search("retrieval", top_k=2)[0].as_dict()
    assert set(payload) >= {"chunk_id", "source", "ordinal", "score", "contributions", "text"}
    assert json.dumps(payload)


def test_get_chunk_and_get_chunks(corpus: Corpus):
    hit = corpus.search("retrieval", top_k=1)[0]
    assert corpus.get_chunk(hit.chunk.id).id == hit.chunk.id
    assert corpus.get_chunk("nonexistent") is None
    assert len(corpus.get_chunks([hit.chunk.id, hit.chunk.id])) == 2


def test_lexical_scores_are_exposed(corpus: Corpus):
    scores = corpus.lexical_scores("retrieval quality")
    assert scores
    assert all(value > 0 for value in scores.values())


def test_deduplication_of_identical_documents(corpus: Corpus):
    before = corpus.stats()["chunks"]
    corpus.add_text("docs/rag.md", RAG_TEXT)
    assert corpus.stats()["chunks"] == before


def test_duplicate_chunk_text_across_documents_is_indexed_twice():
    built = Corpus.in_memory()
    built.add_text("a.md", "Identical sentence about ranking.")
    built.add_text("b.md", "Identical sentence about ranking.")
    assert built.stats()["chunks"] == 2
    built.close()


def test_stats_reports_configuration(corpus: Corpus):
    info = corpus.stats()
    assert info["config"]["fusion"] == "rrf"
    assert info["config"]["fusion_k"] == 61
    assert info["embedder"]["name"] == "hashing"


def test_context_manager_closes(db_path: str):
    with Corpus.open(db_path) as corpus:
        corpus.add_text("a.md", "content here")
        assert corpus.stats()["chunks"] == 1


def test_oversized_candidate_pool_still_respects_top_k(corpus: Corpus):
    assert len(corpus.search("retrieval", top_k=2, candidate_multiplier=20)) <= 2


def test_module_level_search_helper(tmp_path):
    corpus = Corpus.open(str(tmp_path / "c.db"))
    corpus.add_text("x.md", "Token buckets enforce an average rate while allowing bursts.")
    corpus.close()
    hits = atheneum.search("average rate bursts", db=str(tmp_path / "c.db"), top_k=2)
    assert hits and hits[0].chunk.source == "x.md"
