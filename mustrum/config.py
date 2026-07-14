"""User configuration: defaults, a global bootstrap file, and a per-library
settings file — overridable by environment variables. Model names are
config, never code.

Two TOML files, in precedence order (later wins):
    1. ~/.config/mustrum/config.toml   — global bootstrap; sets db_path
    2. <db_path's folder>/config.toml  — library settings; travels with the
       library (ADR-16), never sets db_path (that would be self-referential)
    3. MUSTRUM_DB / MUSTRUM_OLLAMA_URL environment variables
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

DEFAULT_CONFIG_PATH = Path.home() / ".config" / "mustrum" / "config.toml"
DEFAULT_DB_PATH = Path.home() / ".mustrum" / "mustrum.db"
LIBRARY_CONFIG_NAME = "config.toml"

# Fields the library-local file (and `config set` / the GUI Settings panel)
# may set. db_path is deliberately excluded from this set (ADR-16).
EDITABLE_FIELDS = (
    "ollama_url",
    "llm_model",
    "embed_model",
    "max_source_chars",
    "num_ctx",
    "unpaywall_email",
    "llm_provider",
    "anthropic_model",
    "anthropic_max_tokens",
)


@dataclass(frozen=True)
class Config:
    db_path: Path = DEFAULT_DB_PATH
    ollama_url: str = "http://localhost:11434"
    llm_model: str = "qwen3:30b"  # ADR-8
    embed_model: str = "nomic-embed-text"  # ADR-8
    # generation context window; source texts are truncated to fit (chars)
    max_source_chars: int = 16000
    # Ollama context window in tokens (must comfortably fit max_source_chars)
    num_ctx: int = 16384
    # contact e-mail for the Unpaywall API (open-access PDF lookup by DOI);
    # empty disables OA lookup for DOI ingestion
    unpaywall_email: str = ""
    # which LLMProvider `_context()` builds (E10-1): "ollama" or "anthropic".
    # Embeddings always come from Ollama regardless — Anthropic has no
    # embeddings endpoint. The API key is never stored here (ADR-4/privacy
    # rule 9): it's resolved from ANTHROPIC_API_KEY / `ant auth login` only.
    llm_provider: str = "ollama"
    anthropic_model: str = "claude-sonnet-5"
    anthropic_max_tokens: int = 8192

    @property
    def files_dir(self) -> Path:
        """Original-file archive: a visible directory next to the database,
        so DB + originals form one backup unit (ADR-13)."""
        return self.db_path.parent / "files"

    @property
    def library_config_path(self) -> Path:
        """Editable settings file next to the database (ADR-16) — travels
        with the library, unlike the global bootstrap file under ~/.config."""
        return self.db_path.parent / LIBRARY_CONFIG_NAME


def _apply_toml(config: Config, path: Path, *, skip: tuple[str, ...] = ()) -> Config:
    data = tomllib.loads(path.read_text())
    known = {f for f in Config.__dataclass_fields__} - set(skip)
    overrides = {k: v for k, v in data.items() if k in known}
    if "db_path" in overrides:
        overrides["db_path"] = Path(overrides["db_path"]).expanduser()
    return replace(config, **overrides)


def load_config(path: Path | None = None) -> Config:
    """Defaults ← global TOML (if present) ← library TOML next to the
    database (if present) ← environment variables. See module docstring."""
    config = Config()
    config_path = path or DEFAULT_CONFIG_PATH
    if config_path.is_file():
        config = _apply_toml(config, config_path)
    if env_db := os.environ.get("MUSTRUM_DB"):
        config = replace(config, db_path=Path(env_db).expanduser())
    if config.library_config_path.is_file():
        config = _apply_toml(config, config.library_config_path, skip=("db_path",))
    if env_url := os.environ.get("MUSTRUM_OLLAMA_URL"):
        config = replace(config, ollama_url=env_url)
    return config


def _toml_literal(value: Any) -> str:
    if isinstance(value, str):
        return json.dumps(value)  # TOML basic-string escaping matches JSON's
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def save_library_config(config: Config, updates: Mapping[str, Any]) -> Config:
    """Merge `updates` (a subset of EDITABLE_FIELDS) into the library config
    file next to config.db_path, creating it if absent, preserving fields not
    being updated. Returns the resulting in-memory Config; does not touch the
    global bootstrap file or db_path, and does not affect an already-running
    process (ADR-16: settings apply on next start)."""
    unknown = set(updates) - set(EDITABLE_FIELDS)
    if unknown:
        raise ValueError(f"not editable: {sorted(unknown)}")
    path = config.library_config_path
    existing: dict[str, Any] = {}
    if path.is_file():
        existing = {
            k: v for k, v in tomllib.loads(path.read_text()).items() if k in EDITABLE_FIELDS
        }
    existing.update(updates)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Mustrum library settings — lives next to mustrum.db, so it travels",
        "# with the library (a synced/backed-up library folder carries its own",
        "# settings). Edit directly, via `mustrum config set`, or the UI Settings",
        "# panel. Restart `mustrum ui` for changes to take effect there.",
        "",
    ]
    lines += [f"{key} = {_toml_literal(value)}" for key, value in existing.items()]
    path.write_text("\n".join(lines) + "\n")
    return replace(config, **existing)
