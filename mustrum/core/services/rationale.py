"""Grounded match rationales (FR-4.2/E4-3): explain *why* a source matches a
research idea, backed by verbatim quotes verified against the stored text.

Same rejection discipline as summaries: an unverifiable rationale is never
stored; the match keeps its bare similarity score.
"""

from __future__ import annotations

from mustrum.core.models import Match
from mustrum.core.ports import LLMProvider, StorageRepo
from mustrum.core.services.grounded import (
    GroundedOutputError,
    describe_failure,
    run_grounded,
)
from mustrum.core.verify import GroundingResult, GroundingVerifier

_SYSTEM = (
    "You assess how an academic paper relates to a research idea, with "
    "absolute fidelity to the paper's text. Use ONLY the text provided; never "
    "add outside knowledge. Reply with a single JSON object: "
    '{"rationale": "<1-2 sentences: how this paper is relevant to the idea>", '
    '"quotes": ["<verbatim quote from the paper supporting the rationale>", ...]} '
    "Include 1-3 quotes copied EXACTLY, character for character, from the "
    "paper text. Choose quotes that are complete prose sentences or phrases — "
    "avoid equations, symbol-heavy fragments, and tables. Output the JSON "
    "object immediately, with no preamble and no reasoning."
)


class RationaleFailure(Exception):
    def __init__(
        self,
        match_id: int,
        attempts: int,
        last_result: GroundingResult | None,
        raw_output: str = "",
    ) -> None:
        super().__init__(
            f"rationale for match {match_id} failed grounding after {attempts} "
            f"attempts: {describe_failure(last_result, raw_output)}"
        )
        self.match_id = match_id
        self.last_result = last_result
        self.raw_output = raw_output


class RationaleService:
    def __init__(
        self,
        repo: StorageRepo,
        llm: LLMProvider,
        verifier: GroundingVerifier | None = None,
        max_source_chars: int = 16000,
        attempts: int = 3,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._verifier = verifier or GroundingVerifier()
        self._max_source_chars = max_source_chars
        self._attempts = attempts

    def explain(self, match_id: int, force: bool = False) -> Match:
        match = self._repo.get_match(match_id)
        if match.rationale and not force:
            return match
        source_text = self._repo.get_source_text(match.source_id)
        if source_text is None:
            raise LookupError(
                f"source {match.source_id} has no stored text to ground a rationale in"
            )
        idea = self._repo.get_idea(match.idea_id)
        version = self._repo.latest_idea_version(match.idea_id)
        idea_text = version.text if version else ""
        source = self._repo.get_source(match.source_id)
        excerpt = source_text.text[: self._max_source_chars]
        base_prompt = (
            f"Research idea: {idea.title}\n{idea_text}\n\n"
            f"Paper: {source.title}\n\nPaper text:\n{excerpt}\n\n"
            "Explain the paper's relevance to the research idea as instructed."
        )
        try:
            # quotes are verified against the full stored text, not the excerpt
            rationale, quotes = run_grounded(
                self._llm,
                base_prompt=base_prompt,
                system=_SYSTEM,
                field="rationale",
                source_text=source_text.text,
                verifier=self._verifier,
                attempts=self._attempts,
            )
        except GroundedOutputError as exc:
            raise RationaleFailure(
                match_id, exc.attempts, exc.last_result, raw_output=exc.raw_output
            ) from exc
        self._repo.set_match_rationale(match_id, rationale, quotes)
        return self._repo.get_match(match_id)
