"""Domain entities. Pure data: no I/O, no adapter imports (ADR-5)."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from datetime import UTC, datetime


def utcnow() -> datetime:
    return datetime.now(UTC)


class SourceKind(enum.StrEnum):
    PAPER = "paper"
    ARTICLE = "article"
    NOTE = "note"


class FieldOrigin(enum.StrEnum):
    """Provenance of a metadata field (FR-1.5)."""

    FETCHED = "fetched"  # from an authoritative service (arXiv/Crossref)
    EXTRACTED = "extracted"  # parsed out of the source file
    USER = "user"  # hand-entered


class ReadingStatus(enum.StrEnum):
    UNREAD = "unread"
    SKIMMED = "skimmed"
    READ = "read"


class MatchStatus(enum.StrEnum):
    SUGGESTED = "suggested"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"


class IdeaRelation(enum.StrEnum):
    BUILDS_ON = "builds-on"
    CONTRASTS_WITH = "contrasts-with"
    RELATED = "related"


class BibOrigin(enum.StrEnum):
    FETCHED = "fetched"  # raw .bib stored byte-exact as received
    DERIVED = "derived"  # rendered from stored metadata


class ContactKind(enum.StrEnum):
    PERSON = "person"
    COMPANY = "company"
    INSTITUTION = "institution"
    UNIVERSITY = "university"


class EntityKind(enum.StrEnum):
    """Discriminator for cross-entity references (search hits, embeddings)."""

    SOURCE = "source"
    IDEA = "idea"
    CONTACT = "contact"


@dataclass(frozen=True)
class Source:
    kind: SourceKind
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    provenance: tuple[tuple[str, FieldOrigin], ...] = ()  # (field_name, origin)
    reading_status: ReadingStatus = ReadingStatus.UNREAD
    notes: str = ""
    created_at: datetime = field(default_factory=utcnow)
    id: int | None = None


@dataclass(frozen=True)
class SourceText:
    """Verbatim ingested text. Immutable after storage (ADR-7)."""

    source_id: int
    text: str
    extraction_method: str  # e.g. "pymupdf", "plaintext"
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class Summary:
    """Machine-generated unless user_override; stored only if verified (FR-3.3)."""

    source_id: int
    text: str
    evidence: tuple[str, ...]  # verbatim quotes from the SourceText
    model: str  # model name, or "user" for hand-written
    verified: bool
    user_override: bool = False
    created_at: datetime = field(default_factory=utcnow)


@dataclass(frozen=True)
class Idea:
    title: str
    created_at: datetime = field(default_factory=utcnow)
    id: int | None = None


@dataclass(frozen=True)
class IdeaVersion:
    """Append-only (FR-2.2): the latest version is the idea's current text."""

    idea_id: int
    text: str
    created_at: datetime = field(default_factory=utcnow)
    id: int | None = None


@dataclass(frozen=True)
class Match:
    idea_id: int
    source_id: int
    score: float
    status: MatchStatus = MatchStatus.SUGGESTED
    rationale: str = ""
    quotes: tuple[str, ...] = ()  # grounding evidence for the rationale
    created_at: datetime = field(default_factory=utcnow)
    id: int | None = None


@dataclass(frozen=True)
class IdeaLink:
    from_idea_id: int
    to_idea_id: int
    relation: IdeaRelation


@dataclass(frozen=True)
class BibEntry:
    source_id: int
    citation_key: str
    raw_bibtex: str
    origin: BibOrigin


@dataclass(frozen=True)
class Contact:
    name: str
    kind: ContactKind
    affiliation: str = ""
    email: str = ""
    url: str = ""
    notes: str = ""
    created_at: datetime = field(default_factory=utcnow)
    id: int | None = None


@dataclass(frozen=True)
class ContactLink:
    """Links a contact to exactly one idea or source, with the reason (FR-7.2)."""

    contact_id: int
    why: str
    idea_id: int | None = None
    source_id: int | None = None

    def __post_init__(self) -> None:
        if (self.idea_id is None) == (self.source_id is None):
            raise ValueError("ContactLink must reference exactly one of idea_id or source_id")


@dataclass(frozen=True)
class FetchedMetadata:
    """What a MetadataFetcher returns. All fields authoritative (FR-1.2)."""

    title: str
    authors: tuple[str, ...]
    year: int | None
    doi: str | None
    arxiv_id: str | None
    raw_bibtex: str
    abstract: str = ""


@dataclass(frozen=True)
class Embedding:
    entity: EntityKind
    ref_id: int  # source id or idea-version id
    chunk_index: int
    model: str
    vector: tuple[float, ...]


@dataclass(frozen=True)
class SearchHit:
    entity: EntityKind
    ref_id: int
    snippet: str
