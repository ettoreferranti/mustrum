"""Grounded summarisation (FR-3): generate → verify → store, or reject.

The model must return a summary plus verbatim quotes; GroundingVerifier
checks every quote against the stored source text. Unverifiable output is
retried, then rejected with GroundingFailure — never stored (ADR-7).
"""

from __future__ import annotations

import json
import re
from typing import Any

from mustrum.core.models import Summary
from mustrum.core.ports import LLMProvider, StorageRepo
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
        if last_result is not None:
            if last_result.empty_evidence:
                detail = "model supplied no evidence quotes"
            else:
                detail = f"quotes not found in source: {list(last_result.missing_quotes)}"
        else:
            detail = "no parsable output"
            if raw_output:
                detail += f"; raw reply started with: {raw_output[:200]!r}"
        super().__init__(
            f"summary for source {source_id} failed grounding after {attempts} attempts: {detail}"
        )
        self.source_id = source_id
        self.last_result = last_result
        self.raw_output = raw_output


_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# a backslash not starting a valid JSON escape (LaTeX like \alpha, \cite, ...)
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def _parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output.

    Tolerates code fences, surrounding prose, literal newlines inside strings
    (strict=False), and stray LaTeX-style backslashes (repaired to escaped
    backslashes before a second parse attempt).
    """
    candidates = [m.group(1) for m in _FENCED_BLOCK.finditer(text)]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        for variant in (candidate, _INVALID_ESCAPE.sub(r"\\\\", candidate)):
            try:
                data = json.loads(variant, strict=False)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


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

        # each retry tells the model exactly what was wrong with its last reply
        feedback = ""
        last_result: GroundingResult | None = None
        last_raw = ""
        for _ in range(self._attempts):
            raw = self._llm.generate(base_prompt + feedback, system=_SYSTEM)
            data = _parse_json_object(raw)
            if data is None:
                last_raw = raw
                feedback = (
                    "\n\nYour previous reply could not be parsed. Reply with ONLY the "
                    "JSON object — no prose, no code fences, valid JSON escaping."
                )
                continue
            summary_text = data.get("summary")
            quotes = data.get("quotes")
            if not isinstance(summary_text, str) or not isinstance(quotes, list):
                last_raw = raw
                feedback = (
                    '\n\nYour previous reply had the wrong structure. "summary" must be '
                    'a string and "quotes" a list of strings.'
                )
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
            if last_result.empty_evidence:
                feedback = (
                    "\n\nYour previous reply contained no usable quotes. Include 2-4 "
                    "quotes copied verbatim from the text."
                )
            else:
                missing = ", ".join(repr(q) for q in last_result.missing_quotes)
                feedback = (
                    f"\n\nThese quotes from your previous reply were NOT found verbatim "
                    f"in the text: {missing}. Copy each quote EXACTLY, character for "
                    f"character, from the text above — do not paraphrase, reformat "
                    f"numbers, or fix spacing."
                )
        raise GroundingFailure(source_id, self._attempts, last_result, raw_output=last_raw)

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
