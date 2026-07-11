"""Grounded summarisation (FR-3): generate → verify → store, or reject.

The model must return a summary plus verbatim quotes; the shared grounded
loop verifies every quote against the stored source text. Unverifiable
output is retried with corrective feedback, then rejected with
GroundingFailure — never stored (ADR-7).
"""

from __future__ import annotations

from mustrum.core.models import Summary
from mustrum.core.ports import LLMProvider, StorageRepo
from mustrum.core.services.grounded import (
    GroundedOutputError,
    describe_failure,
)
from mustrum.core.services.grounded import (
    run_grounded as _run_grounded,
)
from mustrum.core.verify import GroundingResult, GroundingVerifier

_SYSTEM = (
    "You summarise academic sources with absolute fidelity. Use ONLY the text "
    "provided; never add outside knowledge. Reply with a single JSON object: "
    '{"summary": "<2-4 sentences: what the authors did and found>", '
    '"quotes": ["<verbatim quote from the text supporting the summary>", ...]} '
    "Include 2-4 quotes copied EXACTLY, character for character, from the text. "
    "Choose quotes that are complete prose sentences or phrases — avoid "
    "equations, symbol-heavy fragments, tables, and reference lists, because "
    "extracted PDF text often renders those differently than they appear. "
    "Output the JSON object immediately, with no preamble and no reasoning."
)


class GroundingFailure(Exception):
    def __init__(
        self,
        source_id: int,
        attempts: int,
        last_result: GroundingResult | None,
        raw_output: str = "",
    ) -> None:
        super().__init__(
            f"summary for source {source_id} failed grounding after {attempts} "
            f"attempts: {describe_failure(last_result, raw_output)}"
        )
        self.source_id = source_id
        self.last_result = last_result
        self.raw_output = raw_output


class SummariseService:
    def __init__(
        self,
        repo: StorageRepo,
        llm: LLMProvider,
        verifier: GroundingVerifier | None = None,
        max_source_chars: int = 16000,
        attempts: int = 3,  # retries carry corrective feedback, so they're worth having
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
        base_prompt = f"Title: {source.title}\n\nText:\n{excerpt}\n\nSummarise as instructed."
        try:
            # quotes are verified against the full stored text, not the excerpt
            text, quotes = _run_grounded(
                self._llm,
                base_prompt=base_prompt,
                system=_SYSTEM,
                field="summary",
                source_text=source_text.text,
                verifier=self._verifier,
                attempts=self._attempts,
            )
        except GroundedOutputError as exc:
            raise GroundingFailure(
                source_id, exc.attempts, exc.last_result, raw_output=exc.raw_output
            ) from exc
        summary = Summary(
            source_id=source_id,
            text=text,
            evidence=quotes,
            model=self._llm.model_name,
            verified=True,
        )
        self._repo.set_summary(summary)
        return summary

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
