"""Unicode-aware tokenization for lexical retrieval.

A single tokenizer is used for both indexing and querying so that term
statistics stay consistent. It handles Latin script with word tokens and CJK
script with overlapping bigrams, which is the standard lightweight approach
when no dictionary-based segmenter is available.
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Iterable, Sequence

__all__ = ["CJK_STOPWORDS", "ENGLISH_STOPWORDS", "token_frequencies", "tokenize"]

# Han, Hiragana, Katakana, Hangul. These scripts are written without spaces
# between words, so character-level n-grams are used instead of word splitting.
_CJK_RANGES = (
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0x20000, 0x2A6DF),  # CJK Extension B
)

_WORD_RE = re.compile(r"[a-z0-9]+(?:['\-_][a-z0-9]+)*")

ENGLISH_STOPWORDS = frozenset(
    ["a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these", "they", "this", "to", "was", "will", "with"]
)

# Function words that carry no retrieval signal in CJK queries.
CJK_STOPWORDS = frozenset(["的", "了", "和", "是", "在", "我", "有", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好"])


def _is_cjk(char: str) -> bool:
    code = ord(char)
    return any(start <= code <= end for start, end in _CJK_RANGES)


def _normalize(text: str) -> str:
    """Fold case, full-width forms and accents so queries match indexed text.

    NFKC first, because it folds full-width Latin and half-width Katakana into
    their canonical forms — that matters for CJK documents mixing in ASCII
    identifiers. Then accents are stripped via NFD decomposition, so a query
    typed as "cafe" still finds "café". Hangul recomposes afterwards; Han
    ideographs have no decomposition and pass through untouched.
    """
    folded = unicodedata.normalize("NFKC", text).casefold()
    decomposed = unicodedata.normalize("NFD", folded)
    stripped = "".join(char for char in decomposed if unicodedata.category(char) != "Mn")
    return unicodedata.normalize("NFC", stripped)


def tokenize(text: str, *, remove_stopwords: bool = True) -> list[str]:
    """Split text into retrieval terms.

    Latin runs yield word tokens; CJK runs yield single characters plus
    overlapping bigrams. Bigrams are what give CJK queries precision, since a
    lone Han character is usually far too ambiguous to rank on.
    """
    if not text:
        return []

    normalized = _normalize(text)
    tokens: list[str] = []
    cjk_run: list[str] = []

    def flush_cjk() -> None:
        if not cjk_run:
            return
        chars = list(cjk_run)
        cjk_run.clear()
        tokens.extend(chars)
        for first, second in itertools.pairwise(chars):
            tokens.append(first + second)

    # Walk the normalized string once, alternating between CJK accumulation and
    # Latin word extraction so that mixed-language text keeps positional order.
    latin_buffer: list[str] = []
    for char in normalized:
        if _is_cjk(char):
            if latin_buffer:
                tokens.extend(_words_from("".join(latin_buffer), remove_stopwords))
                latin_buffer.clear()
            cjk_run.append(char)
        else:
            flush_cjk()
            latin_buffer.append(char)
    if latin_buffer:
        tokens.extend(_words_from("".join(latin_buffer), remove_stopwords))
    flush_cjk()

    return tokens


def _words_from(segment: str, remove_stopwords: bool) -> list[str]:
    words = _WORD_RE.findall(segment)
    if not remove_stopwords:
        return words
    kept: list[str] = []
    for word in words:
        if word in ENGLISH_STOPWORDS:
            continue
        # A bare punctuation-joined fragment like "-" adds noise, not signal.
        if len(word) < 2 and not word.isdigit():
            continue
        kept.append(word)
    return kept


def token_frequencies(tokens: Iterable[str]) -> dict[str, int]:
    """Count term occurrences, preserving first-seen order for stable output."""
    freqs: dict[str, int] = {}
    for token in tokens:
        freqs[token] = freqs.get(token, 0) + 1
    return freqs


def distinct_terms(tokens: Sequence[str]) -> set[str]:
    return set(tokens)
