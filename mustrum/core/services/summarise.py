"""Grounded summarisation (FR-3): generate → verify → store, or reject.

The model must return a summary plus verbatim quotes; GroundingVerifier
checks every quote against the stored source text. Unverifiable output is
retried, then rejected with GroundingFailure — never stored (ADR-7).
"""

from __future__ import annotations

import json
from typing import Any

from mustrum.core.models import Summary
from mustrum.core.ports import LLMProvider, StorageRepo
from mustrum.core.verify import GroundingResult, GroundingVerifier

_SYSTEM = (
    "You summarise academic sources with absolute fidelity. Use ONLY the text "
    "provided; never add outside knowledge. Reply with a single JSON object: "
    '{"summary": "<2-4 sentences: what the authors did and found>", '
    '"quotes": ["<verbatim quote from the text supporting the summary>", ...]} '
    "Include 2-4 quotes copied EXACTLY, character for character, from the text."
)


class GroundingFailure(Exception):
    def __init__(self, source_id: int, attempts: int, last_result: GroundingResult | None) -> None:
        detail = "no parsable output"
        if last_result is not None:
            if last_result.empty_evidence:
                detail = "model supplied no evidence quotes"
            else:
                detail = f"quotes not found in source: {list(last_result.missing_quotes)}"
        super().__init__(
            f"summary for source {source_id} failed grounding after {attempts} attempts: {detail}"
        )
        self.source_id = source_id
        self.last_result = last_result


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first JSON object from model output (tolerates code fences)."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


class SummariseService:
    def __init__(
        self,
        repo: StorageRepo,
        llm: LLMProvider,
        verifier: GroundingVerifier | None = None,
        max_source_chars: int = 16000,
        attempts: int = 2,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._verifier = verifier or GroundingVerifier()
        self._max_source_chars = max_source_chars
        self._attempts = attempts

    def summarise(self, source_id: int, force: bool = False) -> Summary:
        existing = self._repo.get_summary(source_id)
        if existing is not None and not force:
            return existing
        source_text = self._repo.get_source_text(source_id)
        if source_text is None:
            raise LookupError(f"source {source_id} has no stored text to summarise")
        excerpt = source_text.text[: self._max_source_chars]
        source = self._repo.get_source(source_id)
        prompt = f"Title: {source.title}\n\nText:\n{excerpt}\n\nSummarise as instructed."

        last_result: GroundingResult | None = None
        for _ in range(self._attempts):
            raw = self._llm.generate(prompt, system=_SYSTEM)
            data = _parse_json_object(raw)
            if data is None:
                continue
            summary_text = data.get("summary")
            quotes = data.get("quotes")
            if not isinstance(summary_text, str) or not isinstance(quotes, list):
                continue
            quotes = [q for q in quotes if isinstance(q, str)]
            # verify against the full stored text, not the excerpt
            last_result = self._verifier.verify(quotes, source_text.text)
            if last_result.ok:
                summary = Summary(
                    source_id=source_id,
                    text=summary_text.strip(),
                    evidence=tuple(quotes),
                    model=self._llm.model_name,
                    verified=True,
                )
                self._repo.set_summary(summary)
                return summary
        raise GroundingFailure(source_id, self._attempts, last_result)

    def override(self, source_id: int, text: str) -> Summary:
        """Store a hand-written summary (FR-3.2); trusted, marked as user's."""
        self._repo.get_source(source_id)
        summary = Summary(
            source_id=source_id,
            text=text.strip(),
            evidence=(),
            model="user",
            verified=True,
            user_override=True,
        )
        self._repo.set_summary(summary)
        return summary
