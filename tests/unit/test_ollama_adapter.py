"""Offline tests for the Ollama adapters using httpx.MockTransport."""

import json
import typing

import httpx
import pytest

from mustrum.adapters.ollama import OllamaEmbedder, OllamaError, OllamaLLM, list_models


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestOllamaLLM:
    def test_sends_model_prompt_system_and_disables_thinking(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "hello"})

        llm = OllamaLLM("qwen3:30b", client=mock_client(handler))
        assert llm.generate("the prompt", system="the system") == "hello"
        assert seen["url"] == "http://localhost:11434/api/generate"
        assert seen["payload"]["model"] == "qwen3:30b"
        assert seen["payload"]["prompt"] == "the prompt"
        assert seen["payload"]["system"] == "the system"
        assert seen["payload"]["stream"] is False
        assert seen["payload"]["think"] is False

    def test_system_omitted_when_not_given(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "ok"})

        OllamaLLM("m", client=mock_client(handler)).generate("p")
        assert "system" not in seen["payload"]

    def test_strips_think_block(self):
        def handler(request):
            return httpx.Response(
                200, json={"response": "<think>step 1... step 2...</think>The answer."}
            )

        assert OllamaLLM("m", client=mock_client(handler)).generate("p") == "The answer."

    def test_http_error_raises_ollama_error(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        with pytest.raises(OllamaError, match="request failed"):
            OllamaLLM("m", client=mock_client(handler)).generate("p")

    def test_malformed_response_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        with pytest.raises(OllamaError, match="malformed"):
            OllamaLLM("m", client=mock_client(handler)).generate("p")

    def test_custom_base_url(self):
        seen = {}

        def handler(request):
            seen["url"] = str(request.url)
            return httpx.Response(200, json={"response": "x"})

        OllamaLLM("m", base_url="http://other:9999/", client=mock_client(handler)).generate("p")
        assert seen["url"] == "http://other:9999/api/generate"

    def test_model_name(self):
        assert OllamaLLM("qwen3:30b").model_name == "qwen3:30b"


class TestStructuredOutput:
    """E3-5 / ADR-14: json_schema becomes Ollama's `format`; truncation is loud."""

    SCHEMA: typing.ClassVar[dict] = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def test_schema_sent_as_format(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": '{"summary": "s"}'})

        OllamaLLM("m", client=mock_client(handler)).generate("p", json_schema=self.SCHEMA)
        assert seen["payload"]["format"] == self.SCHEMA

    def test_format_omitted_without_schema(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "ok"})

        OllamaLLM("m", client=mock_client(handler)).generate("p")
        assert "format" not in seen["payload"]

    def test_truncated_output_raises(self):
        def handler(request):
            return httpx.Response(200, json={"response": "cut off mid", "done_reason": "length"})

        with pytest.raises(OllamaError, match=r"truncated.*num_ctx=16384"):
            OllamaLLM("m", client=mock_client(handler)).generate("p")

    def test_normal_stop_is_fine(self):
        def handler(request):
            return httpx.Response(200, json={"response": "done", "done_reason": "stop"})

        assert OllamaLLM("m", client=mock_client(handler)).generate("p") == "done"


class TestOllamaEmbedder:
    def test_embeds_batch(self):
        def handler(request):
            payload = json.loads(request.content)
            assert payload == {"model": "nomic-embed-text", "input": ["a", "b"]}
            return httpx.Response(200, json={"embeddings": [[1.0, 2.0], [3.0, 4.0]]})

        embedder = OllamaEmbedder("nomic-embed-text", client=mock_client(handler))
        assert embedder.embed(["a", "b"]) == [(1.0, 2.0), (3.0, 4.0)]

    def test_empty_input_short_circuits(self):
        def handler(request):
            raise AssertionError("no request expected")

        assert OllamaEmbedder("m", client=mock_client(handler)).embed([]) == []

    def test_count_mismatch_raises(self):
        def handler(request):
            return httpx.Response(200, json={"embeddings": [[1.0]]})

        with pytest.raises(OllamaError, match="malformed"):
            OllamaEmbedder("m", client=mock_client(handler)).embed(["a", "b"])

    def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(404, text="no model")

        with pytest.raises(OllamaError):
            OllamaEmbedder("m", client=mock_client(handler)).embed(["a"])

    def test_model_name(self):
        assert OllamaEmbedder("nomic-embed-text").model_name == "nomic-embed-text"


class TestContextWindow:
    def test_num_ctx_sent_in_options(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "ok"})

        OllamaLLM("m", client=mock_client(handler)).generate("p")
        assert seen["payload"]["options"] == {"num_ctx": 16384}

    def test_num_ctx_configurable(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return httpx.Response(200, json={"response": "ok"})

        OllamaLLM("m", client=mock_client(handler), num_ctx=32768).generate("p")
        assert seen["payload"]["options"] == {"num_ctx": 32768}


class TestListModels:
    """E12-2: populates the Settings dropdowns / `mustrum config models`."""

    def test_returns_sorted_names(self):
        def handler(request):
            assert request.url.path == "/api/tags"
            return httpx.Response(
                200, json={"models": [{"name": "qwen3:30b"}, {"name": "nomic-embed-text"}]}
            )

        assert list_models("http://localhost:11434", client=mock_client(handler)) == [
            "nomic-embed-text",
            "qwen3:30b",
        ]

    def test_falls_back_to_model_field(self):
        def handler(request):
            return httpx.Response(200, json={"models": [{"model": "llama3.1:8b"}]})

        assert list_models("http://x", client=mock_client(handler)) == ["llama3.1:8b"]

    def test_empty_models_list(self):
        def handler(request):
            return httpx.Response(200, json={"models": []})

        assert list_models("http://x", client=mock_client(handler)) == []

    def test_malformed_response_raises(self):
        def handler(request):
            return httpx.Response(200, json={"unexpected": True})

        with pytest.raises(OllamaError, match="malformed"):
            list_models("http://x", client=mock_client(handler))

    def test_http_error_raises(self):
        def handler(request):
            return httpx.Response(500, text="boom")

        with pytest.raises(OllamaError, match="request failed"):
            list_models("http://x", client=mock_client(handler))

    def test_trailing_slash_in_base_url_handled(self):
        seen = {}

        def handler(request):
            seen["path"] = request.url.path
            return httpx.Response(200, json={"models": []})

        list_models("http://x/", client=mock_client(handler))
        assert seen["path"] == "/api/tags"
