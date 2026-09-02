from __future__ import annotations

import numpy as np
import pytest

from atheneum.retrieval.embedders import (
    HashingEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    build_embedder,
    describe,
    iter_batches,
)


def test_output_shape_matches_input_count():
    embedder = HashingEmbedder(dim=64)
    matrix = embedder.embed_many(["one", "two", "three"])
    assert matrix.shape == (3, 64)


def test_vectors_are_l2_normalized():
    embedder = HashingEmbedder(dim=64)
    for row in embedder.embed_many(["a phrase", "another phrase here"]):
        assert float(np.linalg.norm(row)) == pytest.approx(1.0, abs=1e-5)


def test_deterministic_across_instances():
    a = HashingEmbedder(dim=128).embed("reciprocal rank fusion")
    b = HashingEmbedder(dim=128).embed("reciprocal rank fusion")
    assert np.allclose(a, b)


def test_deterministic_across_processes_for_stable_features():
    # blake2b rather than the salted builtin hash() is what guarantees this.
    embedder = HashingEmbedder(dim=256)
    assert np.array_equal(embedder.embed("hello world"), embedder.embed("hello world"))


def test_similar_text_is_closer_than_dissimilar():
    embedder = HashingEmbedder(dim=512)
    anchor = embedder.embed("reciprocal rank fusion merges ranked lists")
    near = embedder.embed("rank fusion combines several ranked lists")
    far = embedder.embed("the bakery sold sourdough bread at dawn")
    assert float(anchor @ near) > float(anchor @ far)


def test_bigram_weight_adds_order_signal():
    embedder = HashingEmbedder(dim=512, bigram_weight=1.0)
    forward = embedder.embed("alpha beta gamma delta epsilon zeta")
    backward = embedder.embed("zeta epsilon delta gamma beta alpha")
    unigram_only = HashingEmbedder(dim=512, bigram_weight=0.0)
    same_a = unigram_only.embed("alpha beta gamma delta epsilon zeta")
    same_b = unigram_only.embed("zeta epsilon delta gamma beta alpha")
    # An anagram must look identical without bigrams and different with them.
    assert float(np.linalg.norm(same_a - same_b)) == pytest.approx(0.0, abs=1e-6)
    assert float(np.linalg.norm(forward - backward)) > 0.05


def test_empty_text_embeds_to_zero_vector():
    embedder = HashingEmbedder(dim=32)
    assert float(np.linalg.norm(embedder.embed(""))) == 0.0
    assert float(np.linalg.norm(embedder.embed("the a of"))) == 0.0  # all stopwords


def test_punctuation_only_is_inert():
    embedder = HashingEmbedder(dim=32)
    assert float(np.linalg.norm(embedder.embed("!!! ??? ..."))) == 0.0


def test_cjk_text_produces_a_vector():
    embedder = HashingEmbedder(dim=128)
    vector = embedder.embed("混合检索使用倒数排名融合")
    assert float(np.linalg.norm(vector)) == pytest.approx(1.0, abs=1e-5)


def test_cjk_query_matches_cjk_document_embedding():
    embedder = HashingEmbedder(dim=512)
    doc = embedder.embed("混合检索使用倒数排名融合来合并多个排序列表")
    query = embedder.embed("如何合并排序列表")
    other = embedder.embed("今天的天气非常好适合出门散步")
    assert float(doc @ query) > float(doc @ other)


def test_case_and_accent_insensitive():
    embedder = HashingEmbedder(dim=256)
    assert np.allclose(embedder.embed("Café Resume"), embedder.embed("cafe resume"))


def test_batch_and_incremental_agree():
    embedder = HashingEmbedder(dim=128)
    texts = ["alpha beta", "gamma delta", "epsilon"]
    batched = embedder.embed_many(texts)
    individual = np.vstack([embedder.embed(t) for t in texts])
    assert np.allclose(batched, individual)


def test_repetition_cannot_inflate_magnitude():
    # The guarantee normalization provides: a long passage cannot outrank a short
    # one merely by being long, because every vector has unit length.
    embedder = HashingEmbedder(dim=256)
    once = embedder.embed("zebra")
    many = embedder.embed("zebra " * 50)
    assert float(np.linalg.norm(once)) == pytest.approx(1.0, abs=1e-5)
    assert float(np.linalg.norm(many)) == pytest.approx(1.0, abs=1e-5)


def test_sublinear_scaling_without_bigrams():
    # With bigrams off, the two texts share exactly one feature, and sublinear
    # frequency scaling means the vector is still unit-normalized onto it.
    embedder = HashingEmbedder(dim=256, bigram_weight=0.0)
    once = embedder.embed("zebra")
    many = embedder.embed("zebra " * 50)
    assert float(once @ many) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("dim", [0, -8])
def test_invalid_dim_rejected(dim: int):
    with pytest.raises(ValueError):
        HashingEmbedder(dim=dim)


def test_negative_bigram_weight_rejected():
    with pytest.raises(ValueError):
        HashingEmbedder(dim=32, bigram_weight=-0.1)


def test_zero_bigram_weight_is_allowed():
    assert HashingEmbedder(dim=32, bigram_weight=0.0).embed("a b").size == 32


# -- build_embedder ---------------------------------------------------------
def test_build_defaults_to_hashing():
    assert isinstance(build_embedder(None), HashingEmbedder)


@pytest.mark.parametrize("spec", ["hashing", "hash", "default", "offline"])
def test_hashing_aliases(spec: str):
    assert isinstance(build_embedder(spec), HashingEmbedder)


def test_build_from_mapping_with_options():
    embedder = build_embedder({"kind": "hashing", "dim": 77})
    assert embedder.dim == 77


def test_build_openai_carries_settings():
    embedder = build_embedder({"kind": "openai", "model": "text-embedding-3-large", "dim": 3072})
    assert isinstance(embedder, OpenAIEmbedder)
    assert embedder.model == "text-embedding-3-large"
    assert embedder.dim == 3072


def test_build_ollama():
    embedder = build_embedder({"kind": "ollama", "model": "mxbai-embed-large"})
    assert isinstance(embedder, OllamaEmbedder)
    assert embedder.model == "mxbai-embed-large"


def test_build_unknown_kind_rejected():
    with pytest.raises(ValueError, match="unknown embedder kind"):
        build_embedder("quantum")


def test_build_passes_through_an_embedder_instance():
    instance = HashingEmbedder(dim=16)
    assert build_embedder(instance) is instance


def test_build_sentence_transformers_reports_missing_dependency(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "sentence_transformers", None)
    with pytest.raises((RuntimeError, ImportError)):
        build_embedder({"kind": "sentence-transformers"})


# -- describe / batches -----------------------------------------------------
def test_describe_is_stable_json():
    text = describe(HashingEmbedder(dim=64))
    assert '"dim": 64' in text
    assert text == describe(HashingEmbedder(dim=64))


def test_describe_distinguishes_dimensions():
    assert describe(HashingEmbedder(dim=64)) != describe(HashingEmbedder(dim=128))


@pytest.mark.parametrize(
    "items,size,expected",
    [
        ([], 3, []),
        ([1, 2, 3], 3, [[1, 2, 3]]),
        ([1, 2, 3, 4], 3, [[1, 2, 3], [4]]),
        ([1, 2], 5, [[1, 2]]),
    ],
)
def test_iter_batches(items, size, expected):
    assert list(iter_batches(items, size)) == expected


def test_openai_embedder_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="no API key"):
        OpenAIEmbedder(api_key=None).embed("hello")


def test_ollama_base_url_defaults_to_loopback(monkeypatch):
    monkeypatch.delenv("OLLAMA_HOST", raising=False)
    assert OllamaEmbedder().base_url.startswith("http://127.0.0.1")


def test_ollama_embed_many_empty_returns_empty_matrix():
    matrix = OllamaEmbedder().embed_many([])
    assert matrix.shape == (0, 768)
