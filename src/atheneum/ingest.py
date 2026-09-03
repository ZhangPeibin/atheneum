"""Turning files on disk into Documents.

Everything here is best-effort by design: one unreadable file in a tree should
cost you that file, not the ingest run.
"""

from __future__ import annotations

import fnmatch
import logging
import os
import stat
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

from atheneum.core.types import Document

logger = logging.getLogger("atheneum.ingest")

__all__ = [
    "MAX_FILE_BYTES",
    "SENSITIVE_NAMES",
    "discover_files",
    "is_sensitive_name",
    "looks_binary",
    "read_file",
    "read_text",
]

# A single chunking pass over a multi-hundred-megabyte file would dominate the
# ingest and produce a chunk set nobody can read through.
MAX_FILE_BYTES = 8 * 1024 * 1024

TEXT_SUFFIXES = frozenset(
    {
        ".txt", ".md", ".markdown", ".rst", ".adoc", ".org", ".tex",
        ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".json",
        ".jsonl", ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
        ".go", ".rs", ".java", ".kt", ".scala", ".c", ".h", ".cc", ".cpp",
        ".hpp", ".cs", ".rb", ".php", ".sh", ".bash", ".zsh", ".fish", ".sql",
        ".html", ".htm", ".xml", ".css", ".scss", ".less", ".svg", ".vue",
        ".svelte", ".csv", ".tsv", ".log", ".diff", ".patch",
    }
)

# Filenames that are configuration or credentials rather than prose. `.env` in
# particular holds API keys, and indexing it writes those keys into SQLite where
# any later retrieval can surface them to a model.
SENSITIVE_NAMES = frozenset(
    {
        ".netrc", ".pgpass", ".npmrc", ".pypirc", ".git-credentials",
        "credentials", "token", "tokens",
        # Exact match, not a prefix: "env*" would also swallow legitimate files
        # such as environment.md.
        "env",
    }
)

# Exact names are not enough: `.env2`, `config.env`, `Env`, `id_ed25519` and
# `secrets.yaml` all sailed through an exact-match set and were indexed, keys and
# all. Match the shapes credential files actually take instead.
_SENSITIVE_PREFIXES = (".env", "id_", "secret", "credential", "private_key", "keystore")
_SENSITIVE_SUFFIXES = (
    ".pem", ".key", ".p12", ".pfx", ".jks", ".keystore", ".kdbx", ".gpg", ".asc", ".env",
)


def is_sensitive_name(name: str) -> bool:
    """True for filenames that plausibly hold credentials.

    Case-insensitive and pattern-based. The cost of a false positive is one file
    the caller can index explicitly by path; the cost of a false negative is a
    private key sitting in a searchable database that a model can quote back.
    """
    lowered = name.lower()
    if lowered in SENSITIVE_NAMES:
        return True
    # ".env" is matched as a substring, not a prefix or suffix: real names include
    # `.env.local`, `config.env`, `app.env.sample` and `.env.production.local`.
    # `environment.md` and `envoy.yaml` contain "env" but not ".env", so the
    # stricter pattern does not over-block them.
    if ".env" in lowered:
        return True
    if any(lowered.startswith(prefix) for prefix in _SENSITIVE_PREFIXES):
        return True
    return any(lowered.endswith(suffix) for suffix in _SENSITIVE_SUFFIXES)

DEFAULT_EXCLUDES = (
    ".git", ".hg", ".svn", "node_modules", "__pycache__", ".venv", "venv", "dist",
    "build", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".next", ".tox",
)


def looks_binary(path: Path, sample_bytes: int = 8192) -> bool:
    """Sniff a file for binary content.

    A NUL byte in the opening sample is the signal used by git and most search
    engines; it does not occur in valid UTF-8 or ASCII text.
    """
    try:
        with path.open("rb") as handle:
            sample = handle.read(sample_bytes)
    except OSError:
        return True
    if not sample:
        return False
    if b"\x00" in sample:
        return True
    # If the decodable fraction is very low, treat it as binary rather than
    # indexing mojibake.
    try:
        sample.decode("utf-8")
        return False
    except UnicodeDecodeError:
        pass
    decoded = 0
    for encoding in ("latin-1", "utf-16", "cp1252"):
        try:
            sample.decode(encoding)
            decoded += 1
        except (UnicodeDecodeError, LookupError):
            continue
    return decoded >= 2


def read_text(path: Path, *, max_bytes: int = MAX_FILE_BYTES) -> str:
    """Read a file as text, trying common encodings before giving up."""
    size = path.stat().st_size
    with path.open("rb") as handle:
        raw = handle.read(max_bytes + 1)
    if size > max_bytes or len(raw) > max_bytes:
        logger.warning("truncating %s from %d to %d bytes", path, size, max_bytes)
        raw = raw[:max_bytes]

    # utf-8-sig first: it also decodes BOM-less UTF-8, but strips the marker when
    # present so it cannot pollute the first chunk's text and term statistics.
    for encoding in ("utf-8-sig", "gb18030", "big5", "shift_jis", "cp1252", "latin-1"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    # latin-1 cannot fail, so this is only reached if a future edit adds a
    # stricter encoding to the front of the list.
    return raw.decode("utf-8", errors="replace")


def _is_regular_file(path: Path) -> bool:
    """True only for a regular file.

    `Path.is_file()` follows symlinks and returns True for a FIFO target too, so
    it is not enough on its own: opening a FIFO for reading blocks forever.
    """
    try:
        return stat.S_ISREG(os.stat(path).st_mode)
    except OSError:
        return False


def read_file(path: str | os.PathLike[str], *, title: str | None = None) -> Document:
    """Read one file into a Document addressed by its resolved path."""
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"no such file: {resolved}")
    if resolved.is_dir():
        raise IsADirectoryError(f"{resolved} is a directory; use discover_files() to walk it")
    if not _is_regular_file(resolved):
        # Without this a FIFO or a device node blocks the read indefinitely.
        raise ValueError(f"{resolved} is not a regular file")
    if looks_binary(resolved):
        raise ValueError(f"{resolved} appears to be binary; text formats only")

    content = read_text(resolved)
    suffix = resolved.suffix.lower().lstrip(".")
    return Document(
        source=str(resolved),
        content=content,
        title=title or resolved.name,
        mime_type=_mime_for(suffix),
        metadata={"path": str(resolved), "name": resolved.name, "suffix": suffix, "size": len(content)},
    )


def _mime_for(suffix: str) -> str:
    return {
        "md": "text/markdown",
        "markdown": "text/markdown",
        "py": "text/x-python",
        "js": "text/javascript",
        "ts": "text/typescript",
        "json": "application/json",
        "yaml": "application/x-yaml",
        "yml": "application/x-yaml",
        "html": "text/html",
        "csv": "text/csv",
        "sql": "text/x-sql",
    }.get(suffix, "text/plain")


def discover_files(
    root: str | os.PathLike[str],
    *,
    patterns: Sequence[str] = ("*",),
    exclude: Sequence[str] = DEFAULT_EXCLUDES,
    max_files: int = 20_000,
    follow_symlinks: bool = False,
) -> Iterator[Path]:
    """Yield readable text files under ``root`` matching any of ``patterns``.

    Symlinks are not followed by default: a symlinked directory outside the tree
    silently indexed would leak files the user never pointed at.
    """
    base = Path(root).expanduser()
    if base.is_file():
        yield base
        return
    if not base.is_dir():
        raise FileNotFoundError(f"no such directory: {base}")

    excludes = tuple(exclude) if exclude else DEFAULT_EXCLUDES
    count = 0
    visited: set[tuple[int, int]] = set()

    for dirpath, dirnames, filenames in os.walk(base, followlinks=follow_symlinks):
        # Prune in place so os.walk does not descend into ignored trees.
        dirnames[:] = sorted(
            d
            for d in dirnames
            if d not in excludes and not any(fnmatch.fnmatch(d, pat) for pat in excludes)
        )
        if not follow_symlinks:
            here = Path(dirpath)
            try:
                key = (here.stat()[stat.ST_DEV], here.stat()[stat.ST_INO])
            except OSError:
                continue
            if key in visited:
                dirnames[:] = []
                continue
            visited.add(key)

        for name in sorted(filenames):
            if name in excludes or is_sensitive_name(name):
                logger.debug("skipping sensitive or excluded file %s", name)
                continue
            candidate = Path(dirpath) / name
            if not _matches_any(name, patterns):
                continue
            if candidate.suffix.lower() not in TEXT_SUFFIXES and candidate.suffix:
                continue
            if not follow_symlinks and candidate.is_symlink():
                # `is_file()` follows symlinks, so a link inside the tree pointing
                # at /etc/passwd used to be read and indexed even with
                # follow_symlinks=False -- the docstring promised otherwise.
                logger.debug("skipping symlink %s", candidate)
                continue
            if not _is_regular_file(candidate):
                continue
            yield candidate
            count += 1
            if count >= max_files:
                logger.warning("stopped after %d files under %s", max_files, base)
                return


def _matches_any(name: str, patterns: Iterable[str]) -> bool:
    for pattern in patterns:
        if pattern in {"*", ""}:
            return True
        if fnmatch.fnmatch(name, pattern) or fnmatch.fnmatch(name.lower(), pattern.lower()):
            return True
    return False


def documents_from_paths(
    paths: Iterable[str | os.PathLike[str]], *, patterns: Sequence[str] = ("*",)
) -> Iterator[Document]:
    for root in paths:
        for found in discover_files(root, patterns=patterns):
            try:
                yield read_file(found)
            except Exception as exc:
                logger.warning("skipping %s: %s", found, exc)
