import math

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.core.ports import EmbeddingProvider, LLMProvider


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestFakeLLMProvider:
    def test_satisfies_port(self):
        provider: LLMProvider = FakeLLMProvider()
        assert provider.model_name == "fake-llm"

    def test_replays_responses_in_order(self):
        fake = FakeLLMProvider(["one", "two"])
        assert fake.generate("p1") == "one"
        assert fake.generate("p2") == "two"

    def test_queue_appends(self):
        fake = FakeLLMProvider(["one"])
        fake.queue("two")
        fake.generate("x")
        assert fake.generate("y") == "two"

    def test_records_calls(self):
        fake = FakeLLMProvider(["r"])
        fake.generate("prompt", system="sys")
        assert fake.calls == [("prompt", "sys")]

    def test_exhausted_raises(self):
        fake = FakeLLMProvider()
        with pytest.raises(RuntimeError, match="no queued response"):
            fake.generate("p")


class TestFakeEmbeddingProvider:
    def test_satisfies_port(self):
        provider: EmbeddingProvider = FakeEmbeddingProvider()
        assert provider.model_name == "fake-embed"

    def test_deterministic(self):
        fake = FakeEmbeddingProvider()
        assert fake.embed(["hello world"]) == fake.embed(["hello world"])

    def test_batch_matches_single(self):
        fake = FakeEmbeddingProvider()
        batch = fake.embed(["a b", "c d"])
        assert batch == [fake.embed(["a b"])[0], fake.embed(["c d"])[0]]

    def test_vectors_are_unit_length(self):
        (vec,) = FakeEmbeddingProvider().embed(["some interesting text"])
        assert math.isclose(sum(v * v for v in vec), 1.0, rel_tol=1e-9)

    def test_empty_text_gives_zero_vector(self):
        (vec,) = FakeEmbeddingProvider().embed([""])
        assert all(v == 0.0 for v in vec)

    def test_similarity_orders_sensibly(self):
        fake = FakeEmbeddingProvider()
        query, close, far = fake.embed(
            [
                "graph neural networks for molecules",
                "neural networks on molecular graphs",
                "medieval italian poetry archives",
            ]
        )
        assert cosine(query, close) > cosine(query, far)

    def test_case_insensitive(self):
        fake = FakeEmbeddingProvider()
        assert fake.embed(["Hello World"]) == fake.embed(["hello world"])

    def test_respects_dims(self):
        (vec,) = FakeEmbeddingProvider(dims=8).embed(["x"])
        assert len(vec) == 8


class TestFakeLLMDefaultResponse:
    def test_queue_takes_priority_over_default(self):
        fake = FakeLLMProvider(["queued"], default_response="fallback")
        assert fake.generate("p") == "queued"
        assert fake.generate("p") == "fallback"

    def test_default_repeats_indefinitely(self):
        fake = FakeLLMProvider(default_response="always")
        assert [fake.generate("p") for _ in range(3)] == ["always"] * 3

    def test_no_default_still_raises_when_exhausted(self):
        with pytest.raises(RuntimeError):
            FakeLLMProvider().generate("p")
