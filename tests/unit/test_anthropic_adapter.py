"""Offline tests for the Anthropic adapter using httpx.MockTransport — no
network access, no API key needed (E10-1)."""

import json
import typing

import anthropic
import httpx
import pytest

from mustrum.adapters.anthropic import AnthropicError, AnthropicLLM


def mock_client(handler) -> anthropic.Anthropic:
    return anthropic.Anthropic(
        api_key="test-key", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    )


def message_response(
    text: str = "hello", stop_reason: str = "end_turn", stop_details: dict | None = None
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-5",
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "stop_details": stop_details,
            "usage": {"input_tokens": 5, "output_tokens": 3},
        },
    )


class TestAnthropicLLM:
    def test_sends_model_prompt_and_system(self):
        seen = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["payload"] = json.loads(request.content)
            return message_response("hello")

        llm = AnthropicLLM("claude-sonnet-5", client=mock_client(handler))
        assert llm.generate("the prompt", system="the system") == "hello"
        assert seen["payload"]["model"] == "claude-sonnet-5"
        assert seen["payload"]["messages"] == [{"role": "user", "content": "the prompt"}]
        assert seen["payload"]["system"] == "the system"

    def test_system_omitted_when_not_given(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return message_response("ok")

        AnthropicLLM("m", client=mock_client(handler)).generate("p")
        assert "system" not in seen["payload"]

    def test_max_tokens_sent(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return message_response("ok")

        AnthropicLLM("m", max_tokens=2048, client=mock_client(handler)).generate("p")
        assert seen["payload"]["max_tokens"] == 2048

    def test_model_name(self):
        assert AnthropicLLM("claude-sonnet-5").model_name == "claude-sonnet-5"

    def test_api_error_wrapped(self):
        def handler(request):
            return httpx.Response(500, json={"error": {"type": "api_error", "message": "boom"}})

        with pytest.raises(AnthropicError, match="request failed"):
            AnthropicLLM("m", client=mock_client(handler)).generate("p")

    def test_missing_credentials_gives_actionable_error(self):
        """The SDK raises a bare TypeError (not an AnthropicError subclass)
        when no key/token/profile can be found at all — this must become a
        clear, actionable AnthropicError, not an unhandled crash."""

        class _RaisingMessages:
            def create(self, **kwargs):
                raise TypeError(
                    '"Could not resolve authentication method. Expected one of '
                    "api_key, auth_token, or credentials to be set. Or for one of "
                    'the `X-Api-Key` or `Authorization` headers to be explicitly '
                    'omitted"'
                )

        class _StubClient:
            messages = _RaisingMessages()

        with pytest.raises(AnthropicError, match="no Anthropic credentials found"):
            AnthropicLLM("m", client=_StubClient()).generate("p")  # type: ignore[arg-type]

    def test_unrelated_type_error_is_not_swallowed(self):
        """Only the specific 'no credentials' TypeError is remapped — any
        other TypeError (a genuine bug) must still propagate as itself."""

        class _RaisingMessages:
            def create(self, **kwargs):
                raise TypeError("unrelated bug, nothing to do with credentials")

        class _StubClient:
            messages = _RaisingMessages()

        with pytest.raises(TypeError, match="unrelated bug"):
            AnthropicLLM("m", client=_StubClient()).generate("p")  # type: ignore[arg-type]

    def test_refusal_raises(self):
        def handler(request):
            return message_response(
                "",
                stop_reason="refusal",
                stop_details={"type": "refusal", "category": "cyber", "explanation": "nope"},
            )

        with pytest.raises(AnthropicError, match=r"declined.*nope"):
            AnthropicLLM("m", client=mock_client(handler)).generate("p")

    def test_refusal_without_details_still_raises(self):
        def handler(request):
            return message_response("", stop_reason="refusal", stop_details=None)

        with pytest.raises(AnthropicError, match="declined"):
            AnthropicLLM("m", client=mock_client(handler)).generate("p")

    def test_max_tokens_stop_reason_raises(self):
        def handler(request):
            return message_response("cut off mid", stop_reason="max_tokens")

        with pytest.raises(AnthropicError, match=r"truncated.*max_tokens=8192"):
            AnthropicLLM("m", client=mock_client(handler)).generate("p")

    def test_malformed_response_raises(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-5",
                    "content": [],
                    "stop_reason": "end_turn",
                    "stop_sequence": None,
                    "usage": {"input_tokens": 1, "output_tokens": 0},
                },
            )

        with pytest.raises(AnthropicError, match="malformed"):
            AnthropicLLM("m", client=mock_client(handler)).generate("p")


class TestStructuredOutput:
    """E10-1 mirrors ADR-14: json_schema becomes output_config.format."""

    SCHEMA: typing.ClassVar[dict] = {
        "type": "object",
        "properties": {"summary": {"type": "string"}},
        "required": ["summary"],
    }

    def test_schema_sent_as_output_config_format(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return message_response('{"summary": "s"}')

        AnthropicLLM("m", client=mock_client(handler)).generate("p", json_schema=self.SCHEMA)
        assert seen["payload"]["output_config"] == {
            "format": {"type": "json_schema", "schema": self.SCHEMA}
        }

    def test_output_config_omitted_without_schema(self):
        seen = {}

        def handler(request):
            seen["payload"] = json.loads(request.content)
            return message_response("ok")

        AnthropicLLM("m", client=mock_client(handler)).generate("p")
        assert "output_config" not in seen["payload"]
