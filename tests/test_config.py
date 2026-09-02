from __future__ import annotations

import json

import pytest

from atheneum.config import Config, config_path, default_db_path, load_config, save_config


def test_defaults_are_usable_without_any_configuration():
    config = Config()
    assert config.provider == "offline"
    assert config.fusion == "rrf"
    assert config.fusion_k == 61
    assert config.chunk_overlap < config.chunk_size
    assert config.db.endswith("corpus.db")


def test_config_is_json_serializable():
    assert json.loads(json.dumps(Config().to_dict()))["provider"] == "offline"


def test_merged_ignores_none_so_unset_flags_do_not_override():
    config = Config(provider="openai")
    merged = config.merged(provider=None, top_k=9)
    assert merged.provider == "openai"
    assert merged.top_k == 9


def test_merged_keeps_falsey_but_present_values():
    config = Config(top_k=5)
    assert config.merged(top_k=0).top_k == 0


def test_merged_routes_unknown_keys_to_extra():
    merged = Config().merged(novel_option="x")
    assert merged.extra["novel_option"] == "x"


# -- environment ------------------------------------------------------------
def test_environment_overrides_the_defaults():
    config = load_config(path="/nonexistent/path.json", env={"ATHENEUM_PROVIDER": "openai"})
    assert config.provider == "openai"


def test_environment_typed_coercion():
    config = load_config(
        path="/nope.json",
        env={"ATHENEUM_TOP_K": "12", "ATHENEUM_BM25_B": "0.4", "ATHENEUM_RERANKER": "overlap"},
    )
    assert config.top_k == 12
    assert config.bm25_b == pytest.approx(0.4)
    assert config.reranker == "overlap"


@pytest.mark.parametrize("raw", ["1", "true", "yes", "on", "TRUE"])
def test_boolean_truthy_values(raw: str):
    assert load_config(path="/nope.json", env={"ATHENEUM_EMBEDDER_DIM": "8"}).embedder_dim == 8


def test_invalid_integer_is_reported_clearly():
    with pytest.raises(ValueError, match="must be an integer"):
        load_config(path="/nope.json", env={"ATHENEUM_TOP_K": "many"})


def test_unknown_env_var_is_ignored():
    config = load_config(path="/nope.json", env={"ATHENEUM_NOT_A_SETTING": "x"})
    assert config.provider == "offline"
    assert "not_a_setting" not in config.extra


# -- file -------------------------------------------------------------------
def test_config_file_is_read(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "groq", "top_k": 3}), encoding="utf-8")
    config = load_config(path=path, env={})
    assert config.provider == "groq"
    assert config.top_k == 3


def test_environment_beats_the_config_file(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"provider": "groq"}), encoding="utf-8")
    config = load_config(path=path, env={"ATHENEUM_PROVIDER": "ollama"})
    assert config.provider == "ollama"


def test_unknown_file_keys_survive_in_extra(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"experiment": 1}), encoding="utf-8")
    assert load_config(path=path, env={}).extra["experiment"] == 1


def test_invalid_json_is_reported_with_the_path(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_config(path=path, env={})


def test_non_object_json_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text("[1, 2]", encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON object"):
        load_config(path=path, env={})


def test_save_then_load_round_trips(tmp_path):
    path = tmp_path / "nested" / "config.json"
    original = Config(provider="moonshot", fusion="dbsf", extra={"k": "v"})
    saved = save_config(original, path)
    assert saved == path
    reloaded = load_config(path=path, env={})
    assert reloaded.provider == "moonshot"
    assert reloaded.fusion == "dbsf"
    assert reloaded.extra == {"k": "v"}


def test_save_writes_valid_json(tmp_path):
    path = save_config(Config(), tmp_path / "c.json")
    assert isinstance(json.loads(path.read_text()), dict)


# -- paths ------------------------------------------------------------------
def test_db_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("ATHENEUM_DB", "/tmp/elsewhere.db")
    assert str(default_db_path()) == "/tmp/elsewhere.db"


def test_data_dir_env_var_is_honoured(monkeypatch):
    monkeypatch.delenv("ATHENEUM_DB", raising=False)
    monkeypatch.setenv("ATHENEUM_DATA_DIR", "/tmp/atheneum-data")
    assert str(default_db_path()).startswith("/tmp/atheneum-data")


def test_config_dir_env_var_is_honoured(monkeypatch):
    monkeypatch.setenv("ATHENEUM_CONFIG_DIR", "/tmp/atheneum-conf")
    assert str(config_path()) == "/tmp/atheneum-conf/config.json"


def test_xdg_config_home_is_respected(monkeypatch):
    monkeypatch.delenv("ATHENEUM_CONFIG_DIR", raising=False)
    monkeypatch.setenv("XDG_CONFIG_HOME", "/tmp/xdg")
    assert str(config_path()) == "/tmp/xdg/atheneum/config.json"


def test_config_file_defaults_to_home(monkeypatch):
    monkeypatch.delenv("ATHENEUM_CONFIG_DIR", raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert str(config_path()).endswith(".config/atheneum/config.json")


def test_explicit_db_wins_over_default(tmp_path):
    config = Config(db=str(tmp_path / "x.db"))
    assert config.db == str(tmp_path / "x.db")
