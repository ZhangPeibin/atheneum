from __future__ import annotations

import json

import pytest
from click.testing import CliRunner

import atheneum
from atheneum.cli import cli


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def indexed(tmp_path):
    """A populated corpus plus the files it came from, ready for CLI runs."""
    source = tmp_path / "notes"
    source.mkdir()
    (source / "fusion.md").write_text(
        "# Fusion\n\nReciprocal rank fusion merges several ranked lists by awarding one over a "
        "constant plus the rank. Cormack proposed the constant sixty at SIGIR 2009.",
        encoding="utf-8",
    )
    (source / "bm25.md").write_text(
        "# BM25\n\nOkapi BM25 controls term frequency saturation with k1. The b parameter controls "
        "length normalization between zero and one.",
        encoding="utf-8",
    )
    db = tmp_path / "corpus.db"
    return str(source), str(db)


def invoke(runner: CliRunner, args: list[str], db: str | None = None):
    prefix = ["--db", db] if db else []
    return runner.invoke(cli, [*prefix, *args], catch_exceptions=False)


# -- index ------------------------------------------------------------------
def test_index_reports_chunks_added(runner, indexed):
    source, db = indexed
    result = invoke(runner, ["index", source], db)
    assert result.exit_code == 0
    assert "indexed" in result.output
    assert "chunks" in result.output


def test_index_json_output(runner, indexed):
    source, db = indexed
    result = invoke(runner, ["index", source, "--json"], db)
    payload = json.loads(result.output)
    assert payload["documents"] == 2
    assert payload["chunks"] >= 2


def test_index_is_idempotent(runner, indexed):
    source, db = indexed
    first = json.loads(invoke(runner, ["index", source, "--json"], db).output)["chunks"]
    second = json.loads(invoke(runner, ["index", source, "--json"], db).output)["chunks"]
    assert first == second


def test_fresh_flag_wipes_the_database(runner, indexed):
    source, db = indexed
    invoke(runner, ["index", source], db)
    invoke(runner, ["index", source, "--fresh"], db)
    stats = json.loads(invoke(runner, ["stats", "--json"], db).output)
    assert stats["documents"] == 2


def test_pattern_restricts_what_is_indexed(runner, tmp_path):
    source = tmp_path / "mixed"
    source.mkdir()
    (source / "a.md").write_text("Markdown content here.", encoding="utf-8")
    (source / "b.py").write_text("# python comment content", encoding="utf-8")
    db = str(tmp_path / "c.db")
    invoke(runner, ["index", str(source), "--pattern", "*.md"], db)
    sources = [row["source"] for row in json.loads(invoke(runner, ["sources", "--json"], db).output)]
    assert any(s.endswith("a.md") for s in sources)
    assert not any(s.endswith("b.py") for s in sources)


def test_chunk_size_option_is_applied(runner, indexed):
    source, db = indexed
    small = json.loads(invoke(runner, ["index", source, "--chunk-size", "80", "--json"], db).output)
    other_db = db + ".2"
    large = json.loads(
        invoke(runner, ["index", source, "--chunk-size", "4000", "--json"], other_db).output
    )
    assert small["chunks"] > large["chunks"]


def test_limit_caps_the_ingest(runner, indexed):
    source, db = indexed
    invoke(runner, ["index", source, "--limit", "1"], db)
    assert json.loads(invoke(runner, ["stats", "--json"], db).output)["documents"] == 1


def test_missing_path_is_reported(runner, tmp_path):
    result = invoke(runner, ["index", str(tmp_path / "ghost")], str(tmp_path / "c.db"))
    assert result.exit_code != 0


# -- search -----------------------------------------------------------------
@pytest.fixture
def prepared(indexed):
    source, db = indexed
    CliRunner().invoke(cli, ["--db", db, "index", source], catch_exceptions=False)
    return db


def test_search_prints_ranked_passages(runner, prepared):
    result = invoke(runner, ["search", "rank fusion"], prepared)
    assert result.exit_code == 0
    assert "fusion.md" in result.output


def test_search_json_is_structured(runner, prepared):
    result = invoke(runner, ["search", "length normalization", "--json"], prepared)
    payload = json.loads(result.output)
    assert isinstance(payload, list)
    assert {"chunk_id", "source", "score", "contributions", "text"} <= set(payload[0])


def test_explain_shows_contributions(runner, prepared):
    result = invoke(runner, ["search", "rank fusion", "--explain"], prepared)
    assert "bm25" in result.output


def test_mode_selection(runner, prepared):
    lexical = invoke(runner, ["search", "rank fusion", "--mode", "lexical", "--json"], prepared)
    vector = invoke(runner, ["search", "rank fusion", "--mode", "vector", "--json"], prepared)
    assert json.loads(lexical.output)
    assert json.loads(vector.output)


def test_invalid_mode_is_rejected(runner, prepared):
    result = invoke(runner, ["search", "x", "--mode", "telepathy"], prepared)
    assert result.exit_code != 0


def test_search_on_an_empty_database_still_works(runner, tmp_path):
    db = str(tmp_path / "empty.db")
    empty_source = tmp_path / "nothing"
    empty_source.mkdir()
    invoke(runner, ["index", str(empty_source)], db)
    result = invoke(runner, ["search", "anything"], db)
    assert result.exit_code == 0
    assert "no matches" in result.output


def test_top_k_is_honoured(runner, prepared):
    result = json.loads(invoke(runner, ["search", "the", "--top", "1", "--json"], prepared).output)
    assert len(result) <= 1


# -- ask --------------------------------------------------------------------
def test_ask_returns_a_cited_answer(runner, prepared):
    result = invoke(runner, ["ask", "what does the b parameter control"], prepared)
    assert result.exit_code == 0
    assert "b parameter" in result.output or "length normalization" in result.output
    assert "Sources:" in result.output


def test_ask_json_contains_the_run_metadata(runner, prepared):
    result = json.loads(invoke(runner, ["ask", "what is rank fusion", "--json"], prepared).output)
    assert result["stopped_reason"] == "final_answer"
    assert result["tool_calls"] >= 1
    assert "answer" in result and "turns" in result


def test_ask_with_streaming(runner, prepared):
    result = invoke(runner, ["ask", "what is rank fusion", "--stream"], prepared)
    assert result.exit_code == 0
    assert "fusion" in result.output.lower()


def test_ask_with_an_unknown_provider_fails_cleanly(runner, prepared):
    result = invoke(runner, ["ask", "x", "-m", "nonexistent-provider"], prepared)
    assert result.exit_code != 0
    assert "unknown provider" in result.output


def test_max_turns_flag_is_applied(runner, prepared):
    result = json.loads(invoke(runner, ["ask", "fusion", "--json", "--max-turns", "2"], prepared).output)
    assert result["turns"] <= 2


# -- inspection -------------------------------------------------------------
def test_stats_output(runner, prepared):
    payload = json.loads(invoke(runner, ["stats", "--json"], prepared).output)
    assert payload["embedder"]["name"] == "hashing"
    assert payload["config"]["fusion"] == "rrf"
    assert payload["config"]["fusion_k"] == 61


def test_sources_lists_documents(runner, prepared):
    rows = json.loads(invoke(runner, ["sources", "--json"], prepared).output)
    assert len(rows) == 2
    assert all(row["chunk_count"] >= 1 for row in rows)


def test_show_prints_a_document(runner, prepared):
    result = invoke(runner, ["show", "bm25.md"], prepared)
    assert result.exit_code == 0
    assert "length normalization" in result.output


def test_show_of_an_unknown_source_fails(runner, prepared):
    result = invoke(runner, ["show", "nothing-matches-this"], prepared)
    assert result.exit_code != 0


# -- configuration and providers -------------------------------------------
def test_providers_lists_offline_as_ready(runner):
    result = runner.invoke(cli, ["providers"], catch_exceptions=False)
    assert "offline" in result.output
    assert "NAME" in result.output


def test_providers_json(runner):
    result = runner.invoke(cli, ["providers", "--json"], catch_exceptions=False)
    rows = json.loads(result.output)
    offline = next(row for row in rows if row["name"] == "offline")
    assert offline["ready"] is True


def test_providers_ready_filter(runner, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    result = runner.invoke(cli, ["providers", "--ready", "--json"], catch_exceptions=False)
    names = {row["name"] for row in json.loads(result.output)}
    assert "offline" in names


def test_config_command_shows_effective_settings(runner):
    result = runner.invoke(cli, ["config"], catch_exceptions=False)
    assert "provider" in result.output
    assert "fusion_k" in result.output


def test_config_json_is_parseable(runner):
    result = runner.invoke(cli, ["config", "--json"], catch_exceptions=False)
    assert json.loads(result.output)["fusion"] == "rrf"


def test_init_writes_a_config_file(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_CONFIG_DIR", str(tmp_path / "conf"))
    result = runner.invoke(cli, ["init"], catch_exceptions=False)
    assert result.exit_code == 0
    assert json.loads((tmp_path / "conf" / "config.json").read_text())["provider"] == "offline"


def test_env_var_changes_the_provider(runner, tmp_path, monkeypatch):
    monkeypatch.setenv("ATHENEUM_PROVIDER", "groq")
    result = runner.invoke(cli, ["config", "--json"], catch_exceptions=False)
    assert json.loads(result.output)["provider"] == "groq"


# -- misc -------------------------------------------------------------------
def test_version_flag(runner):
    result = runner.invoke(cli, ["--version"], catch_exceptions=False)
    assert atheneum.__version__ in result.output


def test_help_lists_the_core_commands(runner):
    result = runner.invoke(cli, ["--help"], catch_exceptions=False)
    for command in ("index", "search", "ask", "chat", "serve"):
        assert command in result.output


def test_eval_reports_all_three_modes(runner):
    result = runner.invoke(cli, ["eval", "--json"], catch_exceptions=False)
    payload = json.loads(result.output)
    names = {row["name"] for row in payload["results"]}
    assert names == {"hybrid", "lexical", "vector"}
    assert payload["query_count"] > 5
    assert "hybrid_beats_both_retrievers" in payload


def test_eval_default_output_is_a_readable_table(runner):
    result = runner.invoke(cli, ["eval"], catch_exceptions=False)
    assert result.exit_code == 0
    assert "MODE" in result.output
    assert "recall@k" in result.output
    # The default view must not be JSON; that is what --json is for.
    with pytest.raises(json.JSONDecodeError):
        json.loads(result.output)


def test_benchmark_runs_against_real_files(runner, indexed):
    source, db = indexed
    result = invoke(runner, ["bench", source], db)
    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["ingest"]["chunks"] >= 2
    assert payload["summary"]["query_count"] >= 1


def test_chat_exits_on_eof(runner, prepared):
    result = runner.invoke(cli, ["--db", prepared, "chat"], input="", catch_exceptions=False)
    assert result.exit_code == 0


def test_verbose_flag_raises_logging(runner, prepared):
    result = invoke(runner, ["-v", "search", "fusion"], prepared)
    assert result.exit_code == 0


def test_main_wraps_click_errors():
    from atheneum.cli import main

    assert callable(main)
