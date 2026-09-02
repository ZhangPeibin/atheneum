from __future__ import annotations

import pytest

from atheneum.providers.base import Generation, GenerationRequest, Provider
from atheneum.providers.offline import OfflineProvider
from atheneum.providers.registry import (
    PROVIDER_PROFILES,
    ProviderRegistry,
    get_provider,
    resolve_provider,
)


def test_offline_is_the_default():
    assert isinstance(get_provider("offline"), OfflineProvider)


def test_unknown_provider_lists_alternatives():
    with pytest.raises(KeyError, match="unknown provider"):
        get_provider("does-not-exist")


@pytest.mark.parametrize(
    "name",
    ["openai", "deepseek", "groq", "mistral", "together", "openrouter", "anthropic", "ollama"],
)
def test_profiles_resolve_to_a_provider(name: str):
    # Construction must not require a key; only a real request does.
    assert isinstance(get_provider(name), Provider)


def test_openai_profile_uses_the_official_base_url():
    provider = get_provider("openai")
    assert provider.base_url == "https://api.openai.com/v1"


def test_profile_base_url_is_applied():
    provider = get_provider("deepseek")
    assert "deepseek.com" in provider.base_url


def test_model_override_wins_over_the_profile_default():
    provider = get_provider("openai", model="gpt-4.1")
    assert provider.model == "gpt-4.1"


def test_base_url_override():
    provider = get_provider("openai", base_url="http://localhost:9999/v1")
    assert provider.base_url == "http://localhost:9999/v1"


def test_api_key_is_taken_from_settings():
    provider = get_provider("openai", api_key="sk-explicit")
    assert provider.api_key == "sk-explicit"


def test_api_key_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gq-test")
    assert get_provider("groq").api_key == "gq-test"


def test_missing_key_is_not_silently_empty(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    assert get_provider("mistral").api_key is None


def test_names_are_sorted_and_include_offline():
    names = ProviderRegistry().names()
    assert names == sorted(names)
    assert "offline" in names


def test_every_profile_declares_a_default_model():
    assert all(profile.default_model for profile in PROVIDER_PROFILES.values())


def test_hosted_local_profiles_need_no_api_key():
    for name in ("lmstudio", "vllm"):
        assert PROVIDER_PROFILES[name].api_key_env == ()


def test_ollama_is_usable_without_a_key(monkeypatch):
    # Ollama may optionally present a key, but a default local server does not.
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    assert get_provider("ollama").api_key is None


def test_has_key_reads_any_of_the_env_names(monkeypatch):
    profile = PROVIDER_PROFILES["openai"]
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert profile.has_key() is False
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert profile.has_key() is True


def test_anonymous_local_profiles_are_always_ready():
    assert PROVIDER_PROFILES["ollama"].has_key({}) is False
    assert PROVIDER_PROFILES["ollama"].capabilities.get("tools") is True


def test_profile_supports_defaults_to_true():
    assert PROVIDER_PROFILES["openai"].supports("anything") is True


# -- custom registration ----------------------------------------------------
class Echo(Provider):
    name = "echo"

    def complete(self, request: GenerationRequest) -> Generation:
        return Generation(text=request.last_user_message())


def test_custom_factory_registration():
    built = ProviderRegistry()
    built.register("echo", "test", lambda **_: Echo())
    assert isinstance(built.create("echo"), Echo)


def test_provider_class_registration():
    built = ProviderRegistry()
    built.register_plugin("echo", Echo)
    assert isinstance(built.create("echo"), Echo)


def test_a_failing_plugin_does_not_break_discovery(monkeypatch):
    """Entry points that raise on load must not take the whole CLI down."""
    import importlib.metadata as md

    class Broken:
        name = "broken"

        def load(self):
            raise RuntimeError("plugin is broken")

    monkeypatch.setattr(md, "entry_points", lambda **kw: [Broken()])
    monkeypatch.delenv("ATHENEUM_NO_PLUGINS", raising=False)
    built = ProviderRegistry()
    built._loaded_entry_points = False
    assert isinstance(built.create("offline"), OfflineProvider)


def test_plugins_are_skipped_when_disabled(monkeypatch):
    monkeypatch.setenv("ATHENEUM_NO_PLUGINS", "1")
    built = ProviderRegistry()
    assert "offline" in built.names()


# -- resolve_provider -------------------------------------------------------
def test_resolve_none_gives_offline():
    assert isinstance(resolve_provider(None), OfflineProvider)


def test_resolve_by_name():
    assert isinstance(resolve_provider("offline"), OfflineProvider)


def test_resolve_from_a_mapping():
    provider = resolve_provider({"name": "openai", "model": "gpt-4.1-mini"})
    assert provider.model == "gpt-4.1-mini"


def test_resolve_passes_through_an_instance():
    instance = OfflineProvider()
    assert resolve_provider(instance) is instance


def test_resolve_with_kind_only():
    assert isinstance(resolve_provider({"kind": "offline"}), OfflineProvider)


def test_registry_create_with_empty_name_uses_offline():
    assert isinstance(ProviderRegistry().create(""), OfflineProvider)
