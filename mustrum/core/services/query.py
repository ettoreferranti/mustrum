"""Grounded question-answering over the library (E13-1): retrieve candidate
sources by keyword + embedding similarity, then answer with quotes verified
against the specific source each was drawn from (ADR-7 discipline, extended
to evidence spanning several documents — see `run_grounded_multi`).
"""

from __future__ import annotations

from dataclasses import dataclass

from mustrum.core.models import EntityKind
from mustrum.core.ports import EmbeddingProvider, LLMProvider, StorageRepo
from mustrum.core.services.grounded import (
    Evidence,
    GroundedOutputError,
    describe_failure,
    run_grounded_multi,
)
from mustrum.core.services.match import cosine
from mustrum.core.verify import GroundingResult, GroundingVerifier

_NOT_FOUND = "No sources in your library appear to address this."

_SYSTEM = (
    "You answer questions about the user's personal research library using "
    "ONLY the excerpts below, each tagged with its source_id. Never use "
    "outside knowledge and never invent a source. If none of the excerpts "
    'answer the question, set "found": false. Reply with a single JSON '
    'object: {"found": <bool>, "answer": "<answer text, or empty if not '
    'found>", "evidence": [{"source_id": <int>, "quote": "<verbatim '
    'quote>"}]}. When found is true, every claim in the answer must be '
    "backed by at least one evidence quote copied EXACTLY, character for "
    "character, from that source_id's excerpt — do not paraphrase. Only use "
    "source_id values that appear in the excerpts below. Output the JSON "
    "object immediately, with no preamble and no reasoning."
)


class QueryFailure(Exception):
    def __init__(
        self,
        question: str,
        attempts: int,
        last_result: GroundingResult | None,
        raw_output: str = "",
    ) -> None:
        super().__init__(
            f"query {question!r} failed grounding after {attempts} attempts: "
            f"{describe_failure(last_result, raw_output)}"
        )
        self.question = question
        self.last_result = last_result
        self.raw_output = raw_output


@dataclass(frozen=True)
class QueryAnswer:
    question: str
    answer: str
    found: bool
    evidence: tuple[Evidence, ...]  # empty iff found is False
    considered_source_ids: tuple[int, ...]  # every candidate fed to the LLM


class QueryService:
    def __init__(
        self,
        repo: StorageRepo,
        llm: LLMProvider,
        embedder: EmbeddingProvider,
        embed_model: str,
        verifier: GroundingVerifier | None = None,
        max_source_chars: int = 16000,
        attempts: int = 3,
        top_k: int = 8,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._embedder = embedder
        self._embed_model = embed_model
        self._verifier = verifier or GroundingVerifier()
        self._max_source_chars = max_source_chars
        self._attempts = attempts
        self._top_k = top_k

    def ask(self, question: str) -> QueryAnswer:
        candidate_ids = self._candidate_source_ids(question)
        excerpts: dict[int, str] = {}
        full_texts: dict[int, str] = {}
        for source_id in candidate_ids:
            text = self._repo.get_source_text(source_id)
            if text is None:
                continue
            full_texts[source_id] = text.text
            excerpts[source_id] = text.text[: self._max_source_chars]

        if not excerpts:
            return QueryAnswer(
                question=question,
                answer=_NOT_FOUND,
                found=False,
                evidence=(),
                considered_source_ids=(),
            )

        base_prompt = self._build_prompt(question, excerpts)
        try:
            # quotes are verified against each source's full stored text, not
            # its excerpt — same discipline as run_grounded's single-source callers
            answer, evidence = run_grounded_multi(
                self._llm,
                base_prompt=base_prompt,
                system=_SYSTEM,
                sources=full_texts,
                not_found_message=_NOT_FOUND,
                verifier=self._verifier,
                attempts=self._attempts,
            )
        except GroundedOutputError as exc:
            raise QueryFailure(
                question, exc.attempts, exc.last_result, raw_output=exc.raw_output
            ) from exc

        return QueryAnswer(
            question=question,
            answer=answer,
            found=bool(evidence),
            evidence=evidence,
            considered_source_ids=tuple(excerpts.keys()),
        )

    def _candidate_source_ids(self, question: str) -> list[int]:
        """Union of FTS and embedding-similarity hits, embedding rank first, capped at top_k."""
        query_vector = self._embedder.embed([question])[0]
        best_per_source: dict[int, float] = {}
        for emb in self._repo.embeddings_for(EntityKind.SOURCE, self._embed_model):
            score = cosine(query_vector, emb.vector)
            if score > best_per_source.get(emb.ref_id, float("-inf")):
                best_per_source[emb.ref_id] = score
        ranked_by_embedding = [
            source_id
            for source_id, _ in sorted(best_per_source.items(), key=lambda kv: kv[1], reverse=True)
        ]
        fts_ids = [
            hit.ref_id
            for hit in self._repo.search(question, limit=self._top_k)
            if hit.entity == EntityKind.SOURCE
        ]

        candidate_ids: list[int] = []
        for source_id in ranked_by_embedding + fts_ids:
            if source_id not in candidate_ids:
                candidate_ids.append(source_id)
            if len(candidate_ids) >= self._top_k:
                break
        return candidate_ids

    def _build_prompt(self, question: str, excerpts: dict[int, str]) -> str:
        parts = [f"Question: {question}", ""]
        for source_id, text in excerpts.items():
            title = self._repo.get_source(source_id).title
            parts += [f"[source {source_id}] {title}", text, ""]
        parts.append("Answer as instructed.")
        return "\n".join(parts)
