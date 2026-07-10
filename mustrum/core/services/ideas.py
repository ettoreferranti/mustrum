"""Idea management (FR-2): versioned ideas with embeddings for matching."""

from __future__ import annotations

from mustrum.core.models import (
    Embedding,
    EntityKind,
    Idea,
    IdeaLink,
    IdeaRelation,
    IdeaVersion,
)
from mustrum.core.ports import EmbeddingProvider, StorageRepo


class IdeaService:
    def __init__(self, repo: StorageRepo, embedder: EmbeddingProvider) -> None:
        self._repo = repo
        self._embedder = embedder

    def create(self, title: str, text: str) -> Idea:
        idea = self._repo.add_idea(Idea(title=title))
        assert idea.id is not None
        self._add_version(idea.id, text)
        return idea

    def revise(self, idea_id: int, text: str) -> IdeaVersion:
        """Append a new version (FR-2.2); older versions are kept forever."""
        self._repo.get_idea(idea_id)
        return self._add_version(idea_id, text)

    def link(self, from_idea_id: int, to_idea_id: int, relation: IdeaRelation) -> None:
        if from_idea_id == to_idea_id:
            raise ValueError("cannot link an idea to itself")
        self._repo.get_idea(from_idea_id)
        self._repo.get_idea(to_idea_id)
        self._repo.add_idea_link(
            IdeaLink(from_idea_id=from_idea_id, to_idea_id=to_idea_id, relation=relation)
        )

    def _add_version(self, idea_id: int, text: str) -> IdeaVersion:
        version = self._repo.add_idea_version(IdeaVersion(idea_id=idea_id, text=text))
        idea = self._repo.get_idea(idea_id)
        (vector,) = self._embedder.embed([f"{idea.title}\n\n{text}"])
        # the idea's embedding always reflects its latest version
        self._repo.store_embeddings(
            [
                Embedding(
                    entity=EntityKind.IDEA,
                    ref_id=idea_id,
                    chunk_index=0,
                    model=self._embedder.model_name,
                    vector=vector,
                )
            ]
        )
        return version
