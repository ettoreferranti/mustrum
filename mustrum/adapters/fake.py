"""Deterministic fake providers for the offline test suite (NFR-2).

FakeLLMProvider replays scripted responses; FakeEmbeddingProvider derives a
stable vector from the text itself, so identical texts are identical vectors
and similar-token texts overlap — good enough to exercise matching logic.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence


class FakeLLMProvider:
    """Replays queued responses in order; records every prompt it was given."""

    def __init__(self, responses: Sequence[str] = ()) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str | None]] = []

    @property
    def model_name(self) -> str:
        return "fake-llm"

    def queue(self, *responses: str) -> None:
        self._responses.extend(responses)

    def generate(self, prompt: str, *, system: str | None = None) -> str:
        self.calls.append((prompt, system))
        if not self._responses:
            raise RuntimeError("FakeLLMProvider has no queued response left")
        return self._responses.pop(0)


class FakeEmbeddingProvider:
    """Hash-bucket bag-of-words embedding: deterministic, no model needed.

    Each lowercased token is hashed into one of `dims` buckets; the vector is
    L2-normalised so dot product == cosine similarity.
    """

    def __init__(self, dims: int = 32) -> None:
        self._dims = dims

    @property
    def model_name(self) -> str:
        return "fake-embed"

    def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self._dims
        for token in text.lower().split():
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % self._dims
            vec[bucket] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            return tuple(vec)
        return tuple(v / norm for v in vec)
