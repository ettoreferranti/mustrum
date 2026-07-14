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

from mustrum.adapters.errors import ProviderError

_THINK_BLOCK = re.compile(r"<think>.*?</think>\s*", re.DOTALL)


class OllamaError(ProviderError):
    pass


def _post(client: httpx.Client, url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = client.post(url, json=payload)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    data: dict[str, Any] = response.json()
    return data


def _get(client: httpx.Client, url: str) -> dict[str, Any]:
    try:
        response = client.get(url)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    data: dict[str, Any] = response.json()
    return data


def list_models(
    base_url: str, client: httpx.Client | None = None, timeout: float = 10.0
) -> list[str]:
    """Names of models installed on a running Ollama instance (`/api/tags`),
    for populating the model dropdowns in the Settings UI (E12-2)."""
    c = client or httpx.Client(timeout=timeout)
    data = _get(c, f"{base_url.rstrip('/')}/api/tags")
    models = data.get("models")
    if not isinstance(models, list):
        raise OllamaError(f"malformed Ollama tags response: {data!r}")
    names = {m.get("name") or m.get("model") for m in models if isinstance(m, dict)}
    return sorted(n for n in names if isinstance(n, str) and n)


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

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, Any] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_ctx": self._num_ctx},
        }
        if system is not None:
            payload["system"] = system
        if json_schema is not None:
            # structured output (ADR-14): Ollama constrains decoding to the
            # schema, so the reply parses by construction — content is still
            # untrusted and must pass the verifiers
            payload["format"] = json_schema
        data = _post(self._client, f"{self._url}/api/generate", payload)
        text = data.get("response")
        if not isinstance(text, str):
            raise OllamaError(f"malformed Ollama response: {data!r}")
        if data.get("done_reason") == "length":
            # a cut-off reply must fail loudly, not surface as a cryptic
            # parse/grounding failure downstream
            raise OllamaError(
                f"output truncated: the context window filled up "
                f"(num_ctx={self._num_ctx}) — raise num_ctx in the config"
            )
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
