"""Live Ollama integration tests. Run explicitly with: pytest -m ollama"""

import pytest

from mustrum.adapters.ollama import OllamaEmbedder, OllamaLLM
from mustrum.config import Config

pytestmark = pytest.mark.ollama


def test_generate_roundtrip():
    config = Config()
    llm = OllamaLLM(config.llm_model, base_url=config.ollama_url)
    reply = llm.generate("Reply with exactly the word: pong")
    assert "pong" in reply.lower()
    assert "<think>" not in reply


def test_embed_roundtrip():
    config = Config()
    embedder = OllamaEmbedder(config.embed_model, base_url=config.ollama_url)
    vectors = embedder.embed(["hello", "world"])
    assert len(vectors) == 2
    assert len(vectors[0]) > 100
