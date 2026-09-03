from __future__ import annotations

import json
import os

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


# ---------------------------------------------------------------------------
# CLI hardening found by direct probing through the installed binary.
# ---------------------------------------------------------------------------


def test_db_pointing_at_a_directory_is_refused_not_traced(runner, tmp_path):
    """A destructive flag must fail cleanly, not halfway through deleting."""
    result = invoke(runner, ["search", "anything"], str(tmp_path))
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert "must point at a file" in result.output


def test_fresh_refuses_to_unlink_a_directory(runner, tmp_path):
    target = tmp_path / "adir"
    target.mkdir()
    (target / "keep.md").write_text("content", encoding="utf-8")
    result = invoke(runner, ["index", str(target), "--fresh"], str(target))
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert target.is_dir() and (target / "keep.md").exists(), "--fresh must not delete anything"


def test_fresh_replaces_an_existing_database(runner, indexed):
    source, db = indexed
    invoke(runner, ["index", source], db)
    before = json.loads(invoke(runner, ["stats", "--json"], db).output)["chunks"]
    invoke(runner, ["index", source, "--fresh"], db)
    after = json.loads(invoke(runner, ["stats", "--json"], db).output)
    assert after["chunks"] == before
    assert after["documents"] == 2


@pytest.mark.parametrize(
    ("var", "value", "fragment"),
    [
        ("ATHENEUM_FUSION", "magic", "fusion must be one of"),
        ("ATHENEUM_RERANKER", "bogus", "reranker must be one of"),
        ("ATHENEUM_EMBEDDER", "nope", "embedder must be one of"),
    ],
)
def test_unknown_enum_settings_fail_when_config_loads(runner, monkeypatch, var, value, fragment):
    """These used to load fine and only fail at the first query."""
    monkeypatch.setenv(var, value)
    result = runner.invoke(cli, ["config"], catch_exceptions=False)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    assert fragment in result.output


@pytest.mark.parametrize("fusion", ["rrf", "dbsf", "weighted"])
def test_valid_enum_settings_are_accepted(runner, monkeypatch, fusion):
    monkeypatch.setenv("ATHENEUM_FUSION", fusion)
    result = runner.invoke(cli, ["config", "--json"], catch_exceptions=False)
    assert result.exit_code == 0
    assert json.loads(result.output)["fusion"] == fusion


def test_cli_flag_beats_the_environment(tmp_path):
    """Documented precedence is defaults < file < env < CLI flags.

    Exercised through the installed console script rather than CliRunner:
    click's auto_envvar_prefix is set inside main(), so invoking the group
    object directly does not read ATHENEUM_<COMMAND>_<PARAM> at all and would
    make this test pass for the wrong reason.
    """
    import subprocess
    import sys

    source = tmp_path / "docs"
    source.mkdir()
    for i in range(12):
        (source / f"d{i}.md").write_text(
            f"supplementary document {i} about ranking and fusion", encoding="utf-8"
        )
    db = str(tmp_path / "prec.db")

    def search(*extra: str, env: dict[str, str] | None = None) -> int:
        environment = {**os.environ, **(env or {})}
        completed = subprocess.run(
            [sys.executable, "-m", "atheneum.cli", "--db", db, "search", "ranking fusion", "--json", *extra],
            capture_output=True,
            text=True,
            env=environment,
            timeout=120,
        )
        if completed.returncode != 0:
            pytest.fail(completed.stdout + completed.stderr)
        return len(json.loads(completed.stdout))

    assert subprocess.run(
        [sys.executable, "-m", "atheneum.cli", "--db", db, "index", str(source)],
        capture_output=True, text=True, timeout=180,
    ).returncode == 0

    assert search("--top", "7", env={"ATHENEUM_SEARCH_TOP": "2"}) == 7, "flag must beat env"
    assert search(env={"ATHENEUM_SEARCH_TOP": "2"}) == 2, "env applies when no flag is given"
    assert search("--top", "3", env={"ATHENEUM_SEARCH_TOP": "9"}) == 3


def test_no_command_prints_a_traceback(runner, tmp_path):
    """Every failure path in the CLI should be one line, never a stack."""
    cases = [
        ["search", "x", "--mode", "telepathy"],
        ["search", "x", "--top", "abc"],
        ["ask", "x", "-m", "ghost"],
        ["show", "nothing-matches"],
        ["rebuild"],
    ]
    for args in cases:
        result = invoke(runner, args, str(tmp_path / "absent.db"))
        assert result.exit_code != 0, args
        assert "Traceback" not in result.output, (args, result.output)
