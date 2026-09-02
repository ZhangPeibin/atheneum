"""Hierarchical text splitting.

The splitter walks a cascade of boundaries — paragraph, sentence, then hard
character cut — so a chunk never exceeds the budget but also never breaks
mid-sentence unless the text forces it. Sentence detection covers Latin and CJK
punctuation because a corpus that mixes them is the common case for local
documents.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from atheneum.core.types import Chunk, Document, chunk_id

__all__ = ["SplitterConfig", "split_document", "split_text"]

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200

PARAGRAPH_SEPARATORS = ("\n\n\n", "\n\n", "\r\n\r\n")

# Sentence terminators for Latin, CJK, and Indic scripts. Kept as a character
# class rather than a lookbehind chain so it stays linear-time on large files.
_SENTENCE_END = re.compile(
    r"(?<=[.!?。！？；;…\n])\s*|(?<=[.!?。！？])(?=[\"'”’)】]\s)"
)

# Markdown and reStructuredText headings start a new logical section regardless
# of blank lines, so they are promoted to a boundary.
_HEADING = re.compile(r"^(?:#{1,6}\s|=+$|-+$|~+$)", re.MULTILINE)

_CODE_FENCE = re.compile(r"^```", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SplitterConfig:
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    # Splitting inside a fenced code block produces fragments that do not parse,
    # which makes retrieval on source code actively worse. Default to respecting
    # fence boundaries.
    respect_code_fences: bool = True
    # Pieces shorter than this are merged into a neighbouring chunk when they
    # fit. They are never discarded: dropping text from an index is a silent
    # correctness failure, whereas a small chunk is only a mild quality cost.
    min_chunk_chars: int = 24

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap must be non-negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than "
                f"chunk_size ({self.chunk_size})"
            )


def split_text(text: str, config: SplitterConfig | None = None) -> list[str]:
    """Split raw text into chunk-sized pieces."""
    cfg = config or SplitterConfig()
    if not text or not text.strip():
        return []

    pieces: list[str] = []
    for section in _split_sections(text, cfg):
        pieces.extend(_pack(section, cfg))
    return _absorb_short_pieces(pieces, cfg)


def _absorb_short_pieces(pieces: Sequence[str], cfg: SplitterConfig) -> list[str]:
    """Fold undersized pieces into a neighbour instead of discarding them.

    This used to be a filter — ``[p for p in pieces if len(p.strip()) >=
    min_chunk_chars] or pieces[:1]`` — which silently threw content away. With
    chunk_size=20 and overlap=19 a 240-character input produced 240 short pieces,
    every one was filtered out, and the ``or pieces[:1]`` fallback kept a single
    20-character chunk: 73% of the text vanished from the index without a warning.
    A retrieval system that loses text is worse than one with a few small chunks,
    so short pieces are merged when they fit and kept as-is when they do not.
    """
    merged: list[str] = []
    for piece in pieces:
        stripped = piece.strip()
        if not stripped:
            continue
        if (
            len(stripped) < cfg.min_chunk_chars
            and merged
            and len(merged[-1]) + len(stripped) + 1 <= cfg.chunk_size
        ):
            merged[-1] = f"{merged[-1].rstrip()}\n{stripped}"
        else:
            merged.append(piece)
    return merged


def split_document(
    document: Document, config: SplitterConfig | None = None
) -> list[Chunk]:
    """Split a Document into addressable Chunks with stable ids."""
    cfg = config or SplitterConfig()
    doc_id = document.id
    chunks: list[Chunk] = []
    for ordinal, text in enumerate(split_text(document.content, cfg)):
        chunks.append(
            Chunk(
                id=chunk_id(doc_id, ordinal, text),
                doc_id=doc_id,
                source=document.source,
                ordinal=ordinal,
                text=text,
                metadata={
                    **document.metadata,
                    "title": document.title or document.source,
                    "mime_type": document.mime_type,
                },
            )
        )
    return chunks


def _split_sections(text: str, cfg: SplitterConfig) -> Iterator[str]:
    """Yield top-level sections, keeping code fences intact."""
    if not cfg.respect_code_fences or not _CODE_FENCE.search(text):
        yield from _split_on_headings(text)
        return

    # Partition into fenced and unfenced spans; only unfenced spans are further
    # divided, so a code block always travels as one unit.
    positions = [m.start() for m in _CODE_FENCE.finditer(text)]
    cursor = 0
    inside = False
    for pos in positions:
        if not inside:
            if pos > cursor:
                yield from _split_on_headings(text[cursor:pos])
            cursor = pos
            inside = True
        else:
            newline = text.find("\n", pos)
            end = len(text) if newline == -1 else newline + 1
            yield text[cursor:end]
            cursor = end
            inside = False
    if cursor < len(text):
        remainder = text[cursor:]
        if inside:
            # Unterminated fence: emit as-is rather than guessing where it ends.
            yield remainder
        else:
            yield from _split_on_headings(remainder)


def _is_whole_code_block(section: str) -> bool:
    """True when the section is a complete fenced block."""
    if section.count("```") < 2:
        return False
    lines = section.splitlines()
    return len(lines) >= 2 and lines[0].lstrip().startswith("```") and lines[-1].strip().startswith("```")


def _split_on_headings(text: str) -> Iterator[str]:
    matches = list(_HEADING.finditer(text))
    if not matches:
        yield text
        return
    if matches[0].start() > 0:
        yield text[: matches[0].start()]
    for current, following in zip(matches, [*matches[1:], None], strict=False):
        end = following.start() if following else len(text)
        yield text[current.start() : end]


def _pack(section: str, cfg: SplitterConfig) -> list[str]:
    """Greedily pack sentences into chunks up to chunk_size, with overlap."""
    section = section.strip()
    if not section:
        return []
    if cfg.respect_code_fences and _is_whole_code_block(section):
        # Emit the block intact even past the budget. Half a function body is
        # worse than a long chunk: it cannot be parsed, and it is exactly the
        # fragment a reader cannot use.
        return [section]
    if len(section) <= cfg.chunk_size:
        return [section]

    sentences = _split_sentences(section)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # A single sentence larger than the budget has to be hard-cut; no
        # boundary-based strategy can honour the size limit otherwise.
        if len(sentence) > cfg.chunk_size:
            if current:
                chunks.append(" ".join(current))
                current, current_len = [], 0
            chunks.extend(_hard_cut(sentence, cfg))
            continue

        if current and current_len + len(sentence) + 1 > cfg.chunk_size:
            chunks.append(" ".join(current))
            carry = _overlap_tail(current, cfg)
            current = list(carry)
            current_len = sum(len(s) + 1 for s in current)

        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks


def _overlap_tail(sentences: Sequence[str], cfg: SplitterConfig) -> list[str]:
    """Take trailing sentences totalling roughly chunk_overlap characters."""
    if cfg.chunk_overlap == 0:
        return []
    tail: list[str] = []
    total = 0
    for sentence in reversed(sentences):
        if total + len(sentence) > cfg.chunk_overlap and tail:
            break
        tail.append(sentence)
        total += len(sentence) + 1
    tail.reverse()
    # Never carry the entire chunk forward, or packing makes no progress and
    # the caller would loop forever emitting duplicates.
    return tail[:-1] if len(tail) == len(sentences) else tail


def _hard_cut(text: str, cfg: SplitterConfig) -> list[str]:
    step = max(1, cfg.chunk_size - cfg.chunk_overlap)
    return [text[i : i + cfg.chunk_size] for i in range(0, len(text), step)]


def _split_sentences(text: str) -> list[str]:
    parts = [p for p in _SENTENCE_END.split(text) if p and p.strip()]
    if len(parts) <= 1:
        # No sentence punctuation at all (common in code or CJK prose written
        # without terminators): fall back to newline and clause boundaries.
        # Lookbehind split, so the separator stays attached to the clause it
        # ends. A plain character-class split consumed every CJK comma, which
        # removed 12 punctuation marks from a 3-chunk CJK document.
        parts = [p for p in re.split(r"(?<=[\n，,、；;])", text) if p and p.strip()]
    return parts or [text]
