"""Idea ↔ source matching (FR-4): cosine similarity over stored embeddings.

Scores are the maximum over a source's chunks. Matching only *suggests*;
the user's confirm/reject decision is final (FR-4.3).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from mustrum.core.models import EntityKind, Match, MatchStatus, Source
from mustrum.core.ports import StorageRepo


def cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


@dataclass(frozen=True)
class GapReport:
    unsupported_ideas: tuple[int, ...]  # idea ids with no confirmed matches
    orphan_sources: tuple[int, ...]  # source ids not matched to any idea


class MatchService:
    def __init__(self, repo: StorageRepo, embed_model: str, threshold: float = 0.35) -> None:
        self._repo = repo
        self._embed_model = embed_model
        self._threshold = threshold

    def suggest(self, idea_id: int, limit: int = 20) -> list[Match]:
        """Rank sources against the idea and store new suggestions (FR-4.1)."""
        self._repo.get_idea(idea_id)
        idea_embeddings = [
            e
            for e in self._repo.embeddings_for(EntityKind.IDEA, self._embed_model)
            if e.ref_id == idea_id
        ]
        if not idea_embeddings:
            raise LookupError(
                f"idea {idea_id} has no embedding for model {self._embed_model!r}; "
                "re-save the idea to embed it"
            )
        idea_vector = idea_embeddings[0].vector

        best_per_source: dict[int, float] = {}
        for emb in self._repo.embeddings_for(EntityKind.SOURCE, self._embed_model):
            score = cosine(idea_vector, emb.vector)
            if score > best_per_source.get(emb.ref_id, float("-inf")):
                best_per_source[emb.ref_id] = score

        already_matched = {m.source_id for m in self._repo.list_matches(idea_id)}
        ranked = sorted(best_per_source.items(), key=lambda kv: kv[1], reverse=True)
        created: list[Match] = []
        for source_id, score in ranked:
            if len(created) >= limit:
                break
            if score < self._threshold or source_id in already_matched:
                continue
            created.append(
                self._repo.add_match(Match(idea_id=idea_id, source_id=source_id, score=score))
            )
        return created

    def confirm(self, match_id: int) -> None:
        self._repo.set_match_status(match_id, MatchStatus.CONFIRMED)

    def reject(self, match_id: int) -> None:
        self._repo.set_match_status(match_id, MatchStatus.REJECTED)

    def confirmed_sources(self, idea_id: int) -> list[Source]:
        matches = self._repo.list_matches(idea_id, MatchStatus.CONFIRMED)
        return [self._repo.get_source(m.source_id) for m in matches]

    def gap_report(self) -> GapReport:
        """Ideas without confirmed support; sources not linked to any idea (FR-4.4)."""
        all_matches = self._repo.list_matches()
        confirmed_idea_ids = {m.idea_id for m in all_matches if m.status == MatchStatus.CONFIRMED}
        matched_source_ids = {m.source_id for m in all_matches if m.status != MatchStatus.REJECTED}
        unsupported = tuple(
            idea.id
            for idea in self._repo.list_ideas()
            if idea.id is not None and idea.id not in confirmed_idea_ids
        )
        orphans = tuple(
            source.id
            for source in self._repo.list_sources()
            if source.id is not None and source.id not in matched_source_ids
        )
        return GapReport(unsupported_ideas=unsupported, orphan_sources=orphans)
