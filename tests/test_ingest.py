from __future__ import annotations

import pytest

from atheneum.ingest import (
    MAX_FILE_BYTES,
    discover_files,
    documents_from_paths,
    looks_binary,
    read_file,
    read_text,
)


def write(tmp_path, name: str, body: str | bytes):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body) if isinstance(body, bytes) else path.write_text(body, encoding="utf-8")
    return path


# -- read_file --------------------------------------------------------------
def test_read_a_markdown_file(tmp_path):
    path = write(tmp_path, "note.md", "# Title\n\nBody text.")
    document = read_file(path)
    assert document.source == str(path.resolve())
    assert document.title == "note.md"
    assert document.mime_type == "text/markdown"
    assert "Body text." in document.content


def test_metadata_records_size_and_suffix(tmp_path):
    path = write(tmp_path, "a.py", "print(1)")
    document = read_file(path)
    assert document.metadata["suffix"] == "py"
    assert document.metadata["size"] == len("print(1)")
    assert document.mime_type == "text/x-python"


def test_explicit_title_wins(tmp_path):
    document = read_file(write(tmp_path, "a.md", "x"), title="Chosen")
    assert document.title == "Chosen"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_file(tmp_path / "ghost.md")


def test_directory_raises(tmp_path):
    (tmp_path / "dir").mkdir()
    with pytest.raises(IsADirectoryError):
        read_file(tmp_path / "dir")


def test_binary_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="binary"):
        read_file(write(tmp_path, "blob.txt", b"\x00\x01\x02\x03binary"))


def test_png_signature_is_detected_as_binary(tmp_path):
    with pytest.raises(ValueError, match="binary"):
        read_file(write(tmp_path, "x.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 40))


# -- encoding ---------------------------------------------------------------
def test_utf8_with_bom_is_stripped(tmp_path):
    path = write(tmp_path, "bom.md", "﻿header text")
    assert read_text(path).startswith("header")


def test_gbk_encoded_file_is_decoded(tmp_path):
    body = "混合检索使用倒数排名融合"
    path = write(tmp_path, "gbk.txt", body.encode("gb18030"))
    assert read_text(path) == body


def test_latin1_fallback_keeps_content_readable(tmp_path):
    path = write(tmp_path, "l1.txt", "café résumé".encode("cp1252"))
    assert "caf" in read_text(path)


def test_utf16_is_not_mistaken_for_text(tmp_path):
    # UTF-16 ASCII text contains NUL bytes, so it is correctly rejected as binary
    # rather than indexed as mojibake.
    path = write(tmp_path, "u16.txt", "hello world".encode("utf-16"))
    assert looks_binary(path) is True


def test_oversized_file_is_truncated_not_rejected(tmp_path):
    body = "x" * (MAX_FILE_BYTES + 5000)
    path = write(tmp_path, "big.txt", body)
    assert len(read_text(path)) == MAX_FILE_BYTES


def test_empty_file_reads_as_empty_string(tmp_path):
    assert read_text(write(tmp_path, "empty.md", "")) == ""


def test_looks_binary_on_an_empty_file(tmp_path):
    assert looks_binary(write(tmp_path, "e.txt", "")) is False


# -- discovery --------------------------------------------------------------
def test_discovery_finds_text_files(tmp_path):
    write(tmp_path, "a.md", "one")
    write(tmp_path, "b.py", "two")
    write(tmp_path, "c.bin", b"\x00\x00binary")
    found = {p.name for p in discover_files(tmp_path)}
    assert found == {"a.md", "b.py"}


def test_default_excludes_prune_hidden_directories(tmp_path):
    write(tmp_path, "keep.md", "keep")
    write(tmp_path, "node_modules/pkg/index.js", "noise")
    write(tmp_path, ".git/config", "noise")
    found = {p.name for p in discover_files(tmp_path)}
    assert found == {"keep.md", "index.js"} - {"index.js"}


def test_explicit_exclude_overrides_the_defaults(tmp_path):
    write(tmp_path, "keep.md", "a")
    write(tmp_path, "node_modules/x.md", "b")
    found = {p.name for p in discover_files(tmp_path, exclude=("nodemodules",))}
    assert "x.md" in found


def test_patterns_filter_by_glob(tmp_path):
    write(tmp_path, "a.md", "x")
    write(tmp_path, "b.py", "y")
    assert {p.name for p in discover_files(tmp_path, patterns=("*.md",))} == {"a.md"}


def test_multiple_patterns(tmp_path):
    write(tmp_path, "a.md", "x")
    write(tmp_path, "b.py", "y")
    write(tmp_path, "c.txt", "z")
    found = {p.name for p in discover_files(tmp_path, patterns=("*.md", "*.txt"))}
    assert found == {"a.md", "c.txt"}


def test_patterns_are_case_insensitive(tmp_path):
    write(tmp_path, "README.MD", "x")
    assert {p.name for p in discover_files(tmp_path, patterns=("*.md",))} == {"README.MD"}


def test_a_single_file_path_yields_itself(tmp_path):
    path = write(tmp_path, "solo.md", "x")
    assert list(discover_files(path)) == [path]


def test_missing_directory_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(discover_files(tmp_path / "ghost"))


def test_max_files_caps_the_walk(tmp_path):
    for i in range(20):
        write(tmp_path, f"n{i}.md", "x")
    assert len(list(discover_files(tmp_path, max_files=5))) == 5


def test_nested_directories_are_walked(tmp_path):
    write(tmp_path, "a/b/c/deep.md", "x")
    assert {p.name for p in discover_files(tmp_path)} == {"deep.md"}


def test_unknown_suffixes_without_extension_are_skipped(tmp_path):
    write(tmp_path, "Makefile", "all:\n\techo hi")
    write(tmp_path, "data.parquet", b"\x00parquet")
    found = {p.name for p in discover_files(tmp_path)}
    assert "data.parquet" not in found


def test_documents_from_paths_skips_unreadable_files(tmp_path):
    write(tmp_path, "good.md", "# Good\n\nContent here.")
    write(tmp_path, "bad.md", b"\x00\x00binary")
    documents = list(documents_from_paths([tmp_path]))
    assert [d.title for d in documents] == ["good.md"]


def test_documents_have_resolved_paths(tmp_path):
    path = write(tmp_path, "rel.md", "x")
    document = next(documents_from_paths([str(path)]))
    assert document.source == str(path.resolve())
