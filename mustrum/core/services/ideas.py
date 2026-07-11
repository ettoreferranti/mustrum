"""Idea management (FR-2): versioned ideas with embeddings for matching.

Ideas can also be imported in bulk from a Markdown file: every top-level
`# Heading` starts a new idea (heading = title, body until the next heading =
idea text). Import is idempotent — re-running the same file skips unchanged
ideas; `on_existing="revise"` appends changed text as a new version.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mustrum.core.models import (
    Embedding,
    EntityKind,
    Idea,
    IdeaLink,
    IdeaRelation,
    IdeaVersion,
)
from mustrum.core.ports import EmbeddingProvider, StorageRepo

OnExisting = Literal["skip", "revise", "create"]


class IdeaFileError(ValueError):
    """The ideas file does not follow the `# title` + body format."""


def parse_ideas_file(content: str) -> list[tuple[str, str]]:
    """Parse Markdown into (title, text) pairs, one per top-level `# ` heading.

    Rules (violations raise IdeaFileError):
    - at least one `# ` heading; nothing but whitespace may precede the first
    - every idea needs a non-empty title and a non-empty body
    - `##` and deeper headings are ordinary body text
    """
    ideas: list[tuple[str, str]] = []
    title: str | None = None
    body: list[str] = []

    def flush() -> None:
        if title is None:
            return
        text = "\n".join(body).strip()
        if not text:
            raise IdeaFileError(f"idea {title!r} has no body text")
        ideas.append((title, text))

    for line_number, line in enumerate(content.splitlines(), start=1):
        if line.startswith("# "):
            flush()
            title = line[2:].strip()
            body = []
            if not title:
                raise IdeaFileError(f"line {line_number}: heading has no title")
        elif title is None:
            if line.strip():
                raise IdeaFileError(
                    f"line {line_number}: content before the first '# title' heading"
                )
        else:
            body.append(line)
    flush()
    if not ideas:
        raise IdeaFileError("no '# title' headings found — one heading per idea is required")
    return ideas


@dataclass(frozen=True)
class ImportOutcome:
    title: str
    idea_id: int
    action: Literal["created", "revised", "skipped"]


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

    def import_ideas(self, content: str, on_existing: OnExisting = "skip") -> list[ImportOutcome]:
        """Bulk-import ideas from Markdown (see module docstring for format).

        The file is fully parsed (and validated) before anything is stored.
        """
        parsed = parse_ideas_file(content)
        outcomes: list[ImportOutcome] = []
        for title, text in parsed:
            existing = None if on_existing == "create" else self._repo.find_idea_by_title(title)
            if existing is None:
                idea = self.create(title, text)
                assert idea.id is not None
                outcomes.append(ImportOutcome(title=title, idea_id=idea.id, action="created"))
                continue
            assert existing.id is not None
            latest = self._repo.latest_idea_version(existing.id)
            unchanged = latest is not None and latest.text == text
            if on_existing == "revise" and not unchanged:
                self.revise(existing.id, text)
                outcomes.append(ImportOutcome(title=title, idea_id=existing.id, action="revised"))
            else:
                outcomes.append(ImportOutcome(title=title, idea_id=existing.id, action="skipped"))
        return outcomes

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
        embed_idea(self._repo, self._embedder, idea_id, idea.title, text)
        return version


def embed_idea(
    repo: StorageRepo, embedder: EmbeddingProvider, idea_id: int, title: str, text: str
) -> None:
    """Embed an idea from its title + latest version text. Single definition
    so idea edits and backup restore embed identically; the idea's embedding
    always reflects its latest version."""
    (vector,) = embedder.embed([f"{title}\n\n{text}"])
    repo.store_embeddings(
        [
            Embedding(
                entity=EntityKind.IDEA,
                ref_id=idea_id,
                chunk_index=0,
                model=embedder.model_name,
                vector=vector,
            )
        ]
    )
