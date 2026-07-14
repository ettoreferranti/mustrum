"""Anthropic adapter for the LLMProvider port (E10-1, config-switchable
alternative to Ollama — ADR-4/ADR-8 pattern, no core changes).

Structured output (`json_schema`) uses `output_config.format` (ADR-14's
"constrained decoding shapes syntax only" rule applies identically here):
the reply parses by construction, but evidence quotes still pass the
GroundingVerifier verbatim like every other provider.
"""

from __future__ import annotations

from typing import Any

import anthropic

from mustrum.adapters.errors import ProviderError


class AnthropicError(ProviderError):
    pass


class AnthropicLLM:
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 8192,
        client: anthropic.Anthropic | None = None,
    ) -> None:
        self._model = model
        self._max_tokens = max_tokens
        self._client = client or anthropic.Anthropic(api_key=api_key)

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
        kwargs: dict[str, Any] = {}
        if system is not None:
            kwargs["system"] = system
        if json_schema is not None:
            kwargs["output_config"] = {"format": {"type": "json_schema", "schema": json_schema}}
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
        except TypeError as exc:
            # the SDK raises a bare TypeError (not an AnthropicError subclass)
            # when it can't find a key/token/profile at all — this is the
            # "no credentials configured" case, not a bug in our request
            if "authentication method" not in str(exc):
                raise
            raise AnthropicError(
                "no Anthropic credentials found — set ANTHROPIC_API_KEY in your "
                "environment, or run `ant auth login`"
            ) from exc
        except anthropic.APIError as exc:
            raise AnthropicError(f"Anthropic request failed: {exc}") from exc
        if response.stop_reason == "refusal":
            detail = getattr(response.stop_details, "explanation", None) or "no reason given"
            raise AnthropicError(f"Anthropic declined to respond: {detail}")
        if response.stop_reason == "max_tokens":
            # a cut-off reply must fail loudly, not surface as a cryptic
            # parse/grounding failure downstream (mirrors OllamaLLM)
            raise AnthropicError(
                f"output truncated: max_tokens={self._max_tokens} was reached — "
                "raise anthropic_max_tokens in the config"
            )
        text: str | None = next((b.text for b in response.content if b.type == "text"), None)
        if text is None:
            raise AnthropicError(f"malformed Anthropic response: no text block in {response!r}")
        return text
