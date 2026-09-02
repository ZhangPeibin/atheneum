"""Provider registry.

Binds a short name like ``openai`` or ``ollama`` to a configured
:class:`~atheneum.providers.base.Provider`. Two things make this more than a
dictionary: vendor profiles that record each service's defaults and quirks, and
entry-point discovery so a third-party package can register a provider without
atheneum knowing about it.
"""

from __future__ import annotations

import importlib
import logging
import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from atheneum.providers.base import Provider
from atheneum.providers.offline import OfflineProvider

logger = logging.getLogger("atheneum.providers")

__all__ = ["PROVIDER_PROFILES", "ProviderRegistry", "get_provider", "registry", "resolve_provider"]

ENTRY_POINT_GROUP = "atheneum.providers"

Factory = Callable[..., Provider]


@dataclass(frozen=True, slots=True)
class ProviderProfile:
    """Defaults and known behaviour for one vendor.

    ``capabilities`` exists because providers genuinely differ: several
    OpenAI-compatible servers reject ``tools`` or ``stream_options`` outright,
    and encoding that here beats discovering it at runtime through a failed
    request.
    """

    kind: str
    default_model: str
    default_base_url: str | None = None
    api_key_env: tuple[str, ...] = ()
    capabilities: Mapping[str, bool] = field(default_factory=dict)
    label: str = ""

    def supports(self, feature: str) -> bool:
        return bool(self.capabilities.get(feature, True))

    def has_key(self, environ: Mapping[str, str] | None = None) -> bool:
        environment = os.environ if environ is None else environ
        return any(environment.get(name) for name in self.api_key_env)


PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "offline": ProviderProfile(
        kind="offline",
        default_model="offline",
        label="Deterministic extractive engine (no network, no API key)",
        capabilities={"tools": True, "streaming": True},
    ),
    "openai": ProviderProfile(
        kind="openai",
        default_model="gpt-4o-mini",
        default_base_url="https://api.openai.com/v1",
        api_key_env=("OPENAI_API_KEY",),
    ),
    "azure": ProviderProfile(
        kind="openai",
        default_model="gpt-4o-mini",
        api_key_env=("AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"),
    ),
    "deepseek": ProviderProfile(
        kind="openai",
        default_model="deepseek-chat",
        default_base_url="https://api.deepseek.com/v1",
        api_key_env=("DEEPSEEK_API_KEY",),
        # DeepSeek's OpenAI-compatible endpoint does not implement every
        # completion parameter the reference API does.
        capabilities={"response_format": False},
    ),
    "groq": ProviderProfile(
        kind="openai",
        default_model="llama-3.3-70b-versatile",
        default_base_url="https://api.groq.com/openai/v1",
        api_key_env=("GROQ_API_KEY",),
    ),
    "mistral": ProviderProfile(
        kind="openai",
        default_model="mistral-large-latest",
        default_base_url="https://api.mistral.ai/v1",
        api_key_env=("MISTRAL_API_KEY",),
    ),
    "together": ProviderProfile(
        kind="openai",
        default_model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
        default_base_url="https://api.together.xyz/v1",
        api_key_env=("TOGETHER_API_KEY",),
    ),
    "openrouter": ProviderProfile(
        kind="openai",
        default_model="openai/gpt-4o-mini",
        default_base_url="https://openrouter.ai/api/v1",
        api_key_env=("OPENROUTER_API_KEY",),
    ),
    "moonshot": ProviderProfile(
        kind="openai",
        default_model="moonshot-v1-8k",
        default_base_url="https://api.moonshot.cn/v1",
        api_key_env=("MOONSHOT_API_KEY",),
    ),
    "dashscope": ProviderProfile(
        kind="openai",
        default_model="qwen-plus",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key_env=("DASHSCOPE_API_KEY",),
    ),
    "ollama": ProviderProfile(
        kind="openai",
        default_model="qwen2.5:7b",
        default_base_url="http://127.0.0.1:11434/v1",
        api_key_env=("OLLAMA_API_KEY",),
        capabilities={"tools": True},
    ),
    "lmstudio": ProviderProfile(
        kind="openai",
        default_model="local-model",
        default_base_url="http://127.0.0.1:1234/v1",
        api_key_env=(),
    ),
    "vllm": ProviderProfile(
        kind="openai",
        default_model="vllm-model",
        default_base_url="http://127.0.0.1:8000/v1",
        api_key_env=(),
    ),
    "anthropic": ProviderProfile(
        kind="anthropic",
        default_model="claude-sonnet-4-5",
        default_base_url="https://api.anthropic.com/v1",
        api_key_env=("ANTHROPIC_API_KEY",),
    ),
}


class ProviderRegistry:
    """Name -> provider factory, extended at runtime by entry points."""

    def __init__(self) -> None:
        self._factories: dict[str, tuple[str, Factory]] = {}
        self._profiles: dict[str, ProviderProfile] = dict(PROVIDER_PROFILES)
        self._loaded_entry_points = False

    def register(self, name: str, kind: str, factory: Factory, *, profile: ProviderProfile | None = None) -> None:
        self._factories[name] = (kind, factory)
        if profile is not None:
            self._profiles[name] = profile

    def profiles(self) -> dict[str, ProviderProfile]:
        return dict(self._profiles)

    def names(self) -> list[str]:
        self._ensure_entry_points()
        return sorted({*self._profiles, *self._factories})

    def create(self, name: str, **settings: Any) -> Provider:
        """Instantiate the provider registered under ``name``."""
        self._ensure_entry_points()
        key = (name or "offline").strip().lower()
        profile = self._profiles.get(key) or self._profiles.get((settings.get("kind") or "").lower())
        if profile is None and key not in self._factories:
            raise KeyError(
                f"unknown provider {name!r}. Known providers: {', '.join(self.names())}"
            )

        if key in self._factories:
            _, factory = self._factories[key]
            return factory(**_filter_kwargs(factory, settings))

        assert profile is not None
        if profile.kind == "offline":
            return OfflineProvider()
        if profile.kind == "anthropic":
            from atheneum.providers.anthropic import AnthropicProvider

            return AnthropicProvider(
                model=settings.get("model") or profile.default_model,
                api_key=settings.get("api_key") or _first_key(profile),
                base_url=settings.get("base_url") or profile.default_base_url,
            )
        from atheneum.providers.openai_compat import OpenAICompatibleProvider

        # A base URL override is what turns one implementation into every
        # OpenAI-compatible server; DeepSeek, Groq and Ollama differ only here.
        return OpenAICompatibleProvider(
            model=settings.get("model") or profile.default_model,
            api_key=settings.get("api_key") or _first_key(profile),
            base_url=settings.get("base_url") or profile.default_base_url,
            env_prefix=key.upper(),
            supports_tools=profile.supports("tools"),
        )

    def _ensure_entry_points(self) -> None:
        """Load third-party providers once.

        Set ``ATHENEUM_NO_PLUGINS=1`` to skip it, which is how tests get
        reproducible provider lists on a machine that happens to have plugins
        installed. The alternative — a private attribute patched by the test
        suite — is what the design this replaces relied on, and it leaks.
        """
        if self._loaded_entry_points:
            return
        self._loaded_entry_points = True
        if os.environ.get("ATHENEUM_NO_PLUGINS") == "1":
            return
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover
            return
        candidates: Any = entry_points(group=ENTRY_POINT_GROUP)
        for candidate in candidates:
            try:
                loaded = candidate.load()
                if isinstance(loaded, type) and issubclass(loaded, Provider):
                    self.register_plugin(candidate.name, loaded)
                elif callable(loaded):
                    self.register(candidate.name, candidate.name, loaded)  # type: ignore[arg-type]
            except Exception as exc:
                logger.warning("could not load provider plugin %r: %s", candidate.name, exc)

    def register_plugin(self, name: str, cls: type) -> None:
        """Register a Provider subclass exposed by a plugin."""
        module = getattr(cls, "__module__", "")
        target = getattr(cls, "create", None)
        factory: Factory = target if callable(target) else cls  # type: ignore[assignment]
        self._factories[name] = (module or "plugin", factory)


def _first_key(profile: ProviderProfile) -> str | None:
    for name in profile.api_key_env:
        value = os.environ.get(name)
        if value:
            return value
    return None


def _filter_kwargs(factory: Factory, settings: Mapping[str, Any]) -> dict[str, Any]:
    """Pass a plugin factory only the settings it accepts."""
    import inspect

    try:
        signature = inspect.signature(factory)
    except (TypeError, ValueError):  # pragma: no cover - builtins and C callables
        return dict(settings)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()):
        return dict(settings)
    allowed = set(signature.parameters)
    return {k: v for k, v in settings.items() if k in allowed and v is not None}


registry = ProviderRegistry()


def get_provider(name: str, **settings: Any) -> Provider:
    """Build a provider by registered name, e.g. ``get_provider("openai")``."""
    return registry.create(name, **settings)


def resolve_provider(spec: str | Mapping[str, Any] | Provider | None) -> Provider:
    """Accept a name, a settings mapping, or an already-built provider.

    Taking a mapping matters because a config file needs to carry the model and
    base URL alongside the provider name.
    """
    if isinstance(spec, Provider):
        return spec
    if spec is None:
        return OfflineProvider()
    if isinstance(spec, str):
        return get_provider(spec)
    data = dict(spec)
    name = str(data.pop("name", None) or data.pop("provider", None) or "offline")
    return get_provider(name, **data)


def load_provider_module(module: str) -> Any:  # pragma: no cover - plugin helper
    return importlib.import_module(module)
