"""User configuration: defaults per ADR-8, overridable via a TOML file and
environment variables. Model names are config, never code."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "mustrum" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".mustrum" / "mustrum.db"


@dataclass(frozen=True)
class Config:
    db_path: Path = DEFAULT_DB_PATH
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:30b"  # ADR-8
    embed_model: str = "nomic-embed-text"  # ADR-8
    # generation context window; source texts are truncated to fit (chars)
    max_source_chars: int = 16000


def load_config(path: Path | None = None) -> Config:
    """Defaults ← TOML file (if present) ← environment variables."""
    config = Config()
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.is_file():
        data = tomllib.loads(config_path.read_text())
        known = {f for f in Config.__dataclass_fields__}
        overrides = {k: v for k, v in data.items() if k in known}
        if "db_path" in overrides:
            overrides["db_path"] = Path(overrides["db_path"]).expanduser()
        config = replace(config, **overrides)
    if env_db := os.environ.get("MUSTRUM_DB"):
        config = replace(config, db_path=Path(env_db).expanduser())
    if env_url := os.environ.get("MUSTRUM_OLLAMA_URL"):
        config = replace(config, ollama_url=env_url)
    return config
