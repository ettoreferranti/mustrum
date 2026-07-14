from pathlib import Path

import pytest

from mustrum.config import Config, load_config, save_library_config

# Every test that doesn't otherwise pin db_path must set MUSTRUM_DB to a
# tmp_path location: load_config() looks for a library config.toml next to
# db_path, and the untouched default (~/.mustrum) is real state on a
# developer's machine once they've used `config set` — reading it would
# make these tests host-dependent and non-deterministic (CLAUDE.md rule 7).


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSTRUM_DB", str(tmp_path / "test.db"))
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        config = load_config(tmp_path / "missing.toml")
        assert config == Config(db_path=tmp_path / "test.db")
        assert config.llm_model == "qwen3:30b"
        assert config.embed_model == "nomic-embed-text"
        assert config.llm_provider == "ollama"
        assert config.anthropic_model == "claude-sonnet-5"

    def test_toml_overrides(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSTRUM_DB", raising=False)
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        f = tmp_path / "global.toml"
        db_path = tmp_path / "lib" / "x.db"
        f.write_text(f'llm_model = "llama3.1:8b"\ndb_path = "{db_path}"\n')
        config = load_config(f)
        assert config.llm_model == "llama3.1:8b"
        assert config.db_path == db_path
        assert config.embed_model == "nomic-embed-text"  # untouched default

    def test_unknown_toml_keys_ignored(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MUSTRUM_DB", str(tmp_path / "test.db"))
        f = tmp_path / "global.toml"
        f.write_text('nonsense = "value"\n')
        assert load_config(f) == Config(db_path=tmp_path / "test.db")

    def test_files_dir_sits_next_to_db(self):
        config = Config(db_path=Path("/data/mustrum/mustrum.db"))
        assert config.files_dir == Path("/data/mustrum/files")

    def test_library_config_path_sits_next_to_db(self):
        config = Config(db_path=Path("/data/mustrum/mustrum.db"))
        assert config.library_config_path == Path("/data/mustrum/config.toml")

    def test_env_overrides_toml(self, tmp_path, monkeypatch):
        f = tmp_path / "global.toml"
        f.write_text('db_path = "/from/toml.db"\nollama_url = "http://toml:1"\n')
        monkeypatch.setenv("MUSTRUM_DB", "/from/env.db")
        monkeypatch.setenv("MUSTRUM_OLLAMA_URL", "http://env:2")
        config = load_config(f)
        assert config.db_path == Path("/from/env.db")
        assert config.ollama_url == "http://env:2"

    def test_library_config_overrides_global(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lib" / "mustrum.db"
        monkeypatch.setenv("MUSTRUM_DB", str(db_path))
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        g = tmp_path / "global.toml"
        g.write_text('llm_model = "from-global"\nembed_model = "from-global-embed"\n')
        lib = db_path.parent / "config.toml"
        lib.parent.mkdir(parents=True)
        lib.write_text('llm_model = "from-library"\n')
        config = load_config(g)
        assert config.llm_model == "from-library"  # library wins
        assert config.embed_model == "from-global-embed"  # untouched by library

    def test_library_config_cannot_set_db_path(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lib" / "mustrum.db"
        monkeypatch.setenv("MUSTRUM_DB", str(db_path))
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        lib = db_path.parent / "config.toml"
        lib.parent.mkdir(parents=True)
        lib.write_text(f'db_path = "{tmp_path / "elsewhere.db"}"\n')
        assert load_config().db_path == db_path  # self-reference ignored

    def test_env_var_wins_over_library_config(self, tmp_path, monkeypatch):
        db_path = tmp_path / "lib" / "mustrum.db"
        monkeypatch.setenv("MUSTRUM_DB", str(db_path))
        monkeypatch.setenv("MUSTRUM_OLLAMA_URL", "http://env:9")
        lib = db_path.parent / "config.toml"
        lib.parent.mkdir(parents=True)
        lib.write_text('ollama_url = "http://library:1"\n')
        assert load_config().ollama_url == "http://env:9"


class TestSaveLibraryConfig:
    def test_writes_next_to_db_and_returns_updated_config(self, tmp_path):
        config = Config(db_path=tmp_path / "lib" / "mustrum.db")
        updated = save_library_config(config, {"llm_model": "llama3.1:8b", "num_ctx": 8192})
        assert updated.llm_model == "llama3.1:8b"
        assert updated.num_ctx == 8192
        assert updated.db_path == config.db_path  # untouched
        written = config.library_config_path.read_text()
        assert 'llm_model = "llama3.1:8b"' in written
        assert "num_ctx = 8192" in written

    def test_preserves_untouched_fields_across_calls(self, tmp_path):
        config = Config(db_path=tmp_path / "mustrum.db")
        save_library_config(config, {"llm_model": "llama3.1:8b"})
        save_library_config(config, {"num_ctx": 8192})
        written = config.library_config_path.read_text()
        assert 'llm_model = "llama3.1:8b"' in written
        assert "num_ctx = 8192" in written

    def test_empty_string_clears_a_field(self, tmp_path):
        config = Config(db_path=tmp_path / "mustrum.db")
        save_library_config(config, {"unpaywall_email": "me@example.org"})
        save_library_config(config, {"unpaywall_email": ""})
        assert 'unpaywall_email = ""' in config.library_config_path.read_text()

    def test_reload_after_save_round_trips(self, tmp_path, monkeypatch):
        db_path = tmp_path / "mustrum.db"
        monkeypatch.setenv("MUSTRUM_DB", str(db_path))
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        save_library_config(Config(db_path=db_path), {"embed_model": "custom-embed"})
        assert load_config(tmp_path / "missing-global.toml").embed_model == "custom-embed"

    def test_rejects_db_path_update(self, tmp_path):
        config = Config(db_path=tmp_path / "mustrum.db")
        with pytest.raises(ValueError, match="db_path"):
            save_library_config(config, {"db_path": "/elsewhere.db"})

    def test_rejects_unknown_field(self, tmp_path):
        config = Config(db_path=tmp_path / "mustrum.db")
        with pytest.raises(ValueError, match="not_a_field"):
            save_library_config(config, {"not_a_field": "x"})

    def test_anthropic_fields_roundtrip(self, tmp_path):
        """E10-1: llm_provider/anthropic_model/anthropic_max_tokens are
        editable the same way as the Ollama settings."""
        config = Config(db_path=tmp_path / "mustrum.db")
        updated = save_library_config(
            config,
            {
                "llm_provider": "anthropic",
                "anthropic_model": "claude-opus-4-8",
                "anthropic_max_tokens": 4096,
            },
        )
        assert updated.llm_provider == "anthropic"
        assert updated.anthropic_model == "claude-opus-4-8"
        assert updated.anthropic_max_tokens == 4096
