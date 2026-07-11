"""Ollama adapters for the LLMProvider and EmbeddingProvider ports (ADR-4).

Uses the local Ollama HTTP API. `think=False` disables reasoning traces on
models that support it (e.g. qwen3); any `<think>…</think>` block that slips
through is stripped defensively, since downstream verifiers must see only the
final answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

import httpx

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaError(RuntimeError):
    pass


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    data: dict[str, Any] = response.json()
    return data


class OllamaLLM:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 300.0,
        client: httpx.Client | None = None,
        num_ctx: int = 16384,  # Ollama's default (~4k) silently truncates long prompts
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._num_ctx = num_ctx

    @property
    def model_name(self) -> str:
        return self._model

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if system is not None:
            payload["system"] = system
        data = _post(self._client, f"{self._url}/api/generate", payload)
        text = data.get("response")
        if not isinstance(text, str):
            raise OllamaError(f"malformed Ollama response: {data!r}")
        return _THINK_BLOCK.sub("", text).strip()


class OllamaEmbedder:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        client: httpx.Client | None = None,
    ) -> None:
        self._model = model
        self._url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)

    @property
    def model_name(self) -> str:
        return self._model

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        data = _post(
            self._client,
            f"{self._url}/api/embed",
            {"model": self._model, "input": list(texts)},
        )
        vectors = data.get("embeddings")
        if not isinstance(vectors, list) or len(vectors) != len(texts):
            raise OllamaError(f"malformed Ollama embed response: {data!r}")
        return [tuple(v) for v in vectors]
