from __future__ import annotations

import pytest

from atheneum.core.types import Document
from atheneum.text.splitter import SplitterConfig, split_document, split_text


def test_short_text_is_one_chunk():
    assert split_text("A single short sentence.") == ["A single short sentence."]


def test_empty_text_yields_nothing():
    assert split_text("") == []
    assert split_text("   \n  ") == []


def test_chunks_respect_size_limit():
    text = " ".join(f"Sentence number {i} about retrieval ranking." for i in range(200))
    chunks = split_text(text, SplitterConfig(chunk_size=300, chunk_overlap=50))
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_carries_text_forward():
    text = "Alpha bravo charlie. " * 60
    without = split_text(text, SplitterConfig(chunk_size=200, chunk_overlap=0))
    with_overlap = split_text(text, SplitterConfig(chunk_size=200, chunk_overlap=60))
    # Overlap must duplicate some content; disjoint windows would produce more
    # chunks for the same budget.
    assert len(with_overlap) >= len(without)


def test_overlap_never_makes_progress_impossible():
    """A tail carry equal to the whole chunk would loop forever."""
    text = "One. Two. Three. Four. Five. Six. Seven. Eight. " * 10
    chunks = split_text(text, SplitterConfig(chunk_size=40, chunk_overlap=39))
    assert chunks
    joined = " ".join(chunks)
    assert "One." in joined


def test_overlap_larger_than_chunk_is_rejected():
    with pytest.raises(ValueError, match="smaller than"):
        SplitterConfig(chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError, match="smaller than"):
        SplitterConfig(chunk_size=100, chunk_overlap=150)


@pytest.mark.parametrize(
    "kwargs",
    [{"chunk_size": 0}, {"chunk_size": -5}, {"chunk_overlap": -1}],
)
def test_invalid_config_rejected(kwargs: dict):
    with pytest.raises(ValueError):
        SplitterConfig(**kwargs)


def test_sentence_boundaries_preferred_mid_chunk():
    text = "First sentence here. Second sentence here. Third sentence is here. Fourth one too."
    chunks = split_text(text, SplitterConfig(chunk_size=60, chunk_overlap=0))
    for chunk in chunks:
        # No chunk should end in the middle of a period-terminated sentence.
        assert not chunk.rstrip().endswith(("sentence her", "sentenc"))


def test_cjk_text_without_latin_punctuation_still_splits():
    text = "混合检索使用倒数排名融合。向量检索负责语义匹配。词法匹配由BM25完成。重排序提高质量。" * 6
    chunks = split_text(text, SplitterConfig(chunk_size=80, chunk_overlap=10))
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def test_code_fence_is_not_split_through():
    text = (
        "Introduction paragraph with enough words to matter for the size budget.\n\n"
        "```python\n"
        + "\n".join(f"def function_{i}(value):\n    return value * {i}" for i in range(30))
        + "\n```\n\nClosing paragraph.\n"
    )
    chunks = split_text(text, SplitterConfig(chunk_size=200, chunk_overlap=0, respect_code_fences=True))
    fenced = [c for c in chunks if "```" in c]
    assert fenced, "expected a chunk containing the code fence"
    for chunk in fenced:
        assert chunk.count("```") % 2 == 0, "fence markers must be balanced inside a chunk"


def test_respecting_fences_off_allows_splitting_them():
    text = "```python\n" + ("x = 1\n" * 200) + "```\n"
    chunks = split_text(text, SplitterConfig(chunk_size=100, chunk_overlap=0, respect_code_fences=False))
    assert any(chunk.count("```") == 1 for chunk in chunks)


def test_heading_starts_a_new_section():
    text = ("Body of the first section with prose. " * 4) + "\n\n## Second\n" + ("More text here. " * 40)
    chunks = split_text(text, SplitterConfig(chunk_size=200, chunk_overlap=0))
    assert any(chunk.startswith("## Second") for chunk in chunks)


def test_oversized_sentence_is_hard_cut():
    text = "x" * 1000
    chunks = split_text(text, SplitterConfig(chunk_size=100, chunk_overlap=0))
    assert all(len(c) <= 100 for c in chunks)
    assert "".join(chunks).startswith("xxxx")


def test_split_document_assigns_stable_ids_and_ordinals():
    document = Document(source="a.md", content="One. Two. Three. " * 40)
    chunks = split_document(document, SplitterConfig(chunk_size=40, chunk_overlap=5))
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert {c.doc_id for c in chunks} == {document.id}
    assert all(c.source == "a.md" for c in chunks)
    # Chunk ids are content-addressed, so re-splitting the same document must
    # reproduce the same identifiers.
    again = split_document(document, SplitterConfig(chunk_size=40, chunk_overlap=5))
    assert [c.id for c in again] == [c.id for c in chunks]


def test_identical_paragraphs_get_distinct_ids():
    document = Document(source="dup.md", content="Same text here.\n\n" * 3)
    chunks = split_document(document)
    assert len(chunks) == len({c.id for c in chunks})


def test_title_defaults_into_chunk_metadata():
    document = Document(source="t.md", content="Hello world.")
    chunk = split_document(document)[0]
    assert chunk.metadata["title"] == "t.md"

    titled = Document(source="t.md", content="Hello world.", title="Explicit")
    assert split_document(titled)[0].metadata["title"] == "Explicit"


def test_min_chunk_chars_drops_fragile_remainders():
    text = "A long enough opening sentence for retrieval. " * 5 + "ab"
    chunks = split_text(text, SplitterConfig(chunk_size=60, chunk_overlap=0, min_chunk_chars=20))
    assert all(len(c.strip()) >= 20 for c in chunks)
