"""Layered configuration: defaults < config file < environment < explicit.

Precedence is fixed and documented because "which config won" is the single most
common source of confusion in self-hosted tools. Environment variables always
beat the file, and CLI flags always beat both.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

__all__ = ["Config", "config_path", "default_db_path", "load_config", "save_config"]

CONFIG_DIR_ENV = "ATHENEUM_CONFIG_DIR"
DATA_DIR_ENV = "ATHENEUM_DATA_DIR"
DB_ENV = "ATHENEUM_DB"

# Prefix applied to every setting when read from the environment, e.g.
# ATHENEUM_FUSION=rrf. One namespace keeps a generic word like DATA_DIR from
# colliding with another tool's variable of the same name.
ENV_PREFIX = "ATHENEUM_"


@dataclass(slots=True)
class Config:
    """Every tunable setting in one place."""

    provider: str = "offline"
    model: str | None = None
    base_url: str | None = None
    db: str = ""
    config_dir: str = ""

    # Retrieval
    chunk_size: int = 1000
    chunk_overlap: int = 200
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    fusion: str = "rrf"
    fusion_k: int = 61
    reranker: str | None = None
    top_k: int = 5
    embedder: str = "hashing"
    embedder_dim: int = 512

    # Agent
    max_turns: int = 8
    token_budget: int = 8000
    temperature: float = 0.0

    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.db:
            self.db = str(default_db_path())
        if not self.config_dir:
            self.config_dir = str(config_path().parent)
        self.validate()

    def validate(self) -> None:
        """Reject settings that cannot work, at construction rather than at query time.

        Without this, `ATHENEUM_TOP_K=0` or a chunk overlap larger than the chunk
        size were accepted silently and only failed later -- or worse, produced
        empty results that looked like "no matches".
        """
        problems: list[str] = []
        if self.top_k < 1:
            problems.append(f"top_k must be >= 1, got {self.top_k}")
        if self.fusion_k < 1:
            problems.append(f"fusion_k must be >= 1, got {self.fusion_k}")
        if self.chunk_size < 1:
            problems.append(f"chunk_size must be >= 1, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            problems.append(f"chunk_overlap must be >= 0, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            problems.append(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )
        if self.max_turns < 1:
            problems.append(f"max_turns must be >= 1, got {self.max_turns}")
        if self.token_budget < 1:
            problems.append(f"token_budget must be >= 1, got {self.token_budget}")
        if self.embedder_dim < 1:
            problems.append(f"embedder_dim must be >= 1, got {self.embedder_dim}")
        if problems:
            raise ValueError("invalid configuration: " + "; ".join(problems))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def merged(self, **overrides: Any) -> Config:
        """Return a copy with non-None overrides applied.

        Ignoring ``None`` is what makes this usable directly with Click options,
        whose value is ``None`` when the flag was not passed.
        """
        data = self.to_dict()
        for key, value in overrides.items():
            if value is None:
                continue
            if key in data:
                data[key] = value
            else:
                data["extra"][key] = value
        return Config(**data)


def config_path() -> Path:
    override = os.environ.get(CONFIG_DIR_ENV)
    if override:
        return Path(override).expanduser() / "config.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "atheneum" / "config.json"


def default_db_path() -> Path:
    from_env = os.environ.get(DB_ENV)
    if from_env:
        return Path(from_env).expanduser()
    data = os.environ.get(DATA_DIR_ENV) or str(Path.home() / ".local" / "share" / "atheneum")
    return Path(data) / "corpus.db"


def load_config(path: str | Path | None = None, *, env: Mapping[str, str] | None = None) -> Config:
    """Read configuration from file then layer environment variables on top."""
    source = Path(path) if path is not None else config_path()
    data: dict[str, Any] = {}
    if source.is_file():
        try:
            loaded = json.loads(source.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{source} is not valid JSON: {exc}") from exc
        if not isinstance(loaded, dict):
            raise ValueError(f"{source} must contain a JSON object, got {type(loaded).__name__}")
        data.update(loaded)

    fields_by_name = {f.name: f for f in fields(Config) if f.name != "extra"}
    known = set(fields_by_name)
    extra = dict(data.pop("extra", {}) or {})

    environment = os.environ if env is None else env
    for name, spec in fields_by_name.items():
        env_key = ENV_PREFIX + name.upper()
        if env_key not in environment:
            continue
        raw = environment[env_key]
        if not raw.strip():
            # A blank value means "not set". Honouring it literally would put
            # provider="" into the config, which reads as a configured-but-empty
            # provider in `ath config` while actually falling back to offline.
            continue
        data[name] = _convert(name, raw, spec.type)

    unknown_file_keys = set(data) - known
    for key in unknown_file_keys:
        extra[key] = data.pop(key)

    kwargs = {k: v for k, v in data.items() if k in known}
    try:
        return Config(extra=extra, **kwargs)
    except ValueError as exc:
        # Name both sources: "invalid configuration" alone does not tell someone
        # whether to look at their config file or their environment.
        raise ValueError(
            f"{exc} (check the config file and any {ENV_PREFIX}* environment variables)"
        ) from exc


def _convert(name: str, raw: str, declared: Any) -> Any:
    text = raw.strip()
    target = _resolve_type(declared)
    if target is bool:
        lowered = text.lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ValueError(f"{ENV_PREFIX}{name.upper()} must be a boolean, got {raw!r}")
    if target is int:
        try:
            return int(text)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{name.upper()} must be an integer, got {raw!r}") from exc
    if target is float:
        try:
            return float(text)
        except ValueError as exc:
            raise ValueError(f"{ENV_PREFIX}{name.upper()} must be a number, got {raw!r}") from exc
    return text


def _resolve_type(declared: Any) -> type:
    """Reduce an annotation like ``str | None`` to its scalar member."""
    text = declared if isinstance(declared, str) else getattr(declared, "__name__", str(declared))
    for candidate in ("bool", "int", "float", "str"):
        if candidate in text.split("|") or candidate in _members(text):
            return {"bool": bool, "int": int, "float": float, "str": str}[candidate]
    return str


def _members(text: str) -> list[str]:
    return [part.strip() for part in text.replace("[", "|").replace("]", "|").split("|")]


def save_config(config: Config, path: str | Path | None = None) -> Path:
    target = Path(path) if path is not None else config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict()
    # Only write the settings the dataclass knows about; `extra` round-trips as a
    # nested object so unknown keys survive an edit.
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
