"""E10-1: `llm_provider` picks the generation backend, config-switchable,
no core changes — this only exercises the CLI's provider-selection wiring."""

from mustrum.adapters.anthropic import AnthropicLLM
from mustrum.adapters.ollama import OllamaLLM
from mustrum.cli.main import _build_llm
from mustrum.config import Config


def test_defaults_to_ollama():
    llm = _build_llm(Config())
    assert isinstance(llm, OllamaLLM)
    assert llm.model_name == "qwen3:30b"


def test_switches_to_anthropic():
    config = Config(llm_provider="anthropic", anthropic_model="claude-opus-4-8")
    llm = _build_llm(config)
    assert isinstance(llm, AnthropicLLM)
    assert llm.model_name == "claude-opus-4-8"
