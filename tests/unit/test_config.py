from pathlib import Path

from mustrum.config import Config, load_config


class TestLoadConfig:
    def test_defaults_when_no_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSTRUM_DB", raising=False)
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        config = load_config(tmp_path / "missing.toml")
        assert config == Config()
        assert config.llm_model == "qwen3:30b"
        assert config.embed_model == "nomic-embed-text"

    def test_toml_overrides(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSTRUM_DB", raising=False)
        monkeypatch.delenv("MUSTRUM_OLLAMA_URL", raising=False)
        f = tmp_path / "config.toml"
        f.write_text('llm_model = "llama3.1:8b"\ndb_path = "/tmp/x.db"\n')
        config = load_config(f)
        assert config.llm_model == "llama3.1:8b"
        assert config.db_path == Path("/tmp/x.db")
        assert config.embed_model == "nomic-embed-text"  # untouched default

    def test_unknown_toml_keys_ignored(self, tmp_path, monkeypatch):
        monkeypatch.delenv("MUSTRUM_DB", raising=False)
        f = tmp_path / "config.toml"
        f.write_text('nonsense = "value"\n')
        assert load_config(f) == Config()

    def test_env_overrides_toml(self, tmp_path, monkeypatch):
        f = tmp_path / "config.toml"
        f.write_text('db_path = "/from/toml.db"\nollama_url = "http://toml:1"\n')
        monkeypatch.setenv("MUSTRUM_DB", "/from/env.db")
        monkeypatch.setenv("MUSTRUM_OLLAMA_URL", "http://env:2")
        config = load_config(f)
        assert config.db_path == Path("/from/env.db")
        assert config.ollama_url == "http://env:2"
