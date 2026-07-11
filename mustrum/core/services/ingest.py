"""Ingestion pipeline (FR-1): store source + immutable text + embeddings +
BibTeX, with dedup on DOI / arXiv id / normalised title (FR-1.4) and
per-field provenance (FR-1.5).

Core receives already-extracted text and already-fetched metadata — file and
network I/O stay in the adapters/CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mustrum.core.bibtex import extract_citation_key
from mustrum.core.models import (
    BibEntry,
    BibOrigin,
    Embedding,
    EntityKind,
    FetchedMetadata,
    FieldOrigin,
    Source,
    SourceKind,
    SourceText,
)
from mustrum.core.normalize import title_hash
from mustrum.core.ports import EmbeddingProvider, StorageRepo
from mustrum.core.services.chunk import chunk_text

OnDuplicate = Literal["fail", "skip", "merge"]


class DuplicateSourceError(Exception):
    def __init__(self, existing: Source, matched_on: str) -> None:
        super().__init__(
            f"source already in library (matched on {matched_on}): [{existing.id}] {existing.title}"
        )
        self.existing = existing
        self.matched_on = matched_on


@dataclass(frozen=True)
class IngestResult:
    source: Source
    created: bool  # False when deduplicated (skip/merge)
    merged: bool = False


class IngestService:
    def __init__(self, repo: StorageRepo, embedder: EmbeddingProvider) -> None:
        self._repo = repo
        self._embedder = embedder

    def ingest_document(
        self,
        *,
        title: str,
        text: str,
        extraction_method: str,
        kind: SourceKind = SourceKind.PAPER,
        authors: tuple[str, ...] = (),
        year: int | None = None,
        on_duplicate: OnDuplicate = "fail",
    ) -> IngestResult:
        """Ingest a locally-extracted document (PDF / plain text), FR-1.1/1.3."""
        provenance: tuple[tuple[str, FieldOrigin], ...] = (("title", FieldOrigin.USER),)
        if authors:
            provenance += (("authors", FieldOrigin.USER),)
        if year is not None:
            provenance += (("year", FieldOrigin.USER),)
        source = Source(
            kind=kind,
            title=title,
            authors=authors,
            year=year,
            provenance=provenance,
        )
        duplicate = self._find_duplicate(source)
        if duplicate is not None:
            return self._handle_duplicate(source, text, extraction_method, *duplicate, on_duplicate)
        saved = self._repo.add_source(source)
        self._attach_text(saved, text, extraction_method)
        return IngestResult(source=saved, created=True)

    def ingest_fetched(
        self,
        meta: FetchedMetadata,
        kind: SourceKind = SourceKind.PAPER,
        on_duplicate: OnDuplicate = "fail",
        full_text: str = "",
        full_text_method: str = "pdf-download",
    ) -> IngestResult:
        """Ingest from authoritative metadata (arXiv / Crossref), FR-1.2.

        When the caller obtained the paper's full text (e.g. an open-access
        PDF), pass it as `full_text`; it is stored instead of the abstract.
        """
        provenance = tuple(
            (field, FieldOrigin.FETCHED)
            for field, value in (
                ("title", meta.title),
                ("authors", meta.authors),
                ("year", meta.year),
                ("doi", meta.doi),
                ("arxiv_id", meta.arxiv_id),
            )
            if value
        )
        source = Source(
            kind=kind,
            title=meta.title,
            authors=meta.authors,
            year=meta.year,
            doi=meta.doi,
            arxiv_id=meta.arxiv_id,
            provenance=provenance,
        )
        text = full_text or meta.abstract
        method = full_text_method if full_text else "abstract"
        duplicate = self._find_duplicate(source)
        if duplicate is not None:
            result = self._handle_duplicate(source, text, method, *duplicate, on_duplicate)
            if result.merged and self._repo.get_bib_entry(result.source.id) is None:  # type: ignore[arg-type]
                self._attach_fetched_bib(result.source, meta.raw_bibtex)
            return result
        saved = self._repo.add_source(source)
        if text:
            self._attach_text(saved, text, method)
        self._attach_fetched_bib(saved, meta.raw_bibtex)
        return IngestResult(source=saved, created=True)

    # -- internals -----------------------------------------------------------

    def _find_duplicate(self, source: Source) -> tuple[Source, str] | None:
        if source.doi and (existing := self._repo.find_source_by_doi(source.doi)):
            return existing, "doi"
        if source.arxiv_id and (existing := self._repo.find_source_by_arxiv_id(source.arxiv_id)):
            return existing, "arxiv_id"
        if existing := self._repo.find_source_by_title_hash(title_hash(source.title)):
            return existing, "title"
        return None

    def _handle_duplicate(
        self,
        incoming: Source,
        text: str,
        extraction_method: str,
        existing: Source,
        matched_on: str,
        on_duplicate: OnDuplicate,
    ) -> IngestResult:
        if on_duplicate == "fail":
            raise DuplicateSourceError(existing, matched_on)
        if on_duplicate == "skip":
            return IngestResult(source=existing, created=False)
        merged = self._merge(existing, incoming)
        self._repo.update_source(merged)
        if text and self._repo.get_source_text(merged.id) is None:  # type: ignore[arg-type]
            self._attach_text(merged, text, extraction_method)
        return IngestResult(source=merged, created=False, merged=True)

    @staticmethod
    def _merge(existing: Source, incoming: Source) -> Source:
        """Fill gaps in the existing record; never overwrite present fields."""
        provenance = dict(existing.provenance)
        updates: dict[str, object] = {}
        incoming_prov = dict(incoming.provenance)
        for field in ("authors", "year", "doi", "arxiv_id"):
            if not getattr(existing, field) and getattr(incoming, field):
                updates[field] = getattr(incoming, field)
                if field in incoming_prov:
                    provenance[field] = incoming_prov[field]
        return Source(
            kind=existing.kind,
            title=existing.title,
            authors=updates.get("authors", existing.authors),  # type: ignore[arg-type]
            year=updates.get("year", existing.year),  # type: ignore[arg-type]
            doi=updates.get("doi", existing.doi),  # type: ignore[arg-type]
            arxiv_id=updates.get("arxiv_id", existing.arxiv_id),  # type: ignore[arg-type]
            provenance=tuple(provenance.items()),
            reading_status=existing.reading_status,
            notes=existing.notes,
            created_at=existing.created_at,
            id=existing.id,
        )

    def _attach_text(self, source: Source, text: str, extraction_method: str) -> None:
        if not text.strip():
            return
        assert source.id is not None
        self._repo.add_source_text(
            SourceText(source_id=source.id, text=text, extraction_method=extraction_method)
        )
        chunks = chunk_text(text)
        vectors = self._embedder.embed(chunks)
        self._repo.store_embeddings(
            [
                Embedding(
                    entity=EntityKind.SOURCE,
                    ref_id=source.id,
                    chunk_index=i,
                    model=self._embedder.model_name,
                    vector=vector,
                )
                for i, vector in enumerate(vectors)
            ]
        )

    def _attach_fetched_bib(self, source: Source, raw_bibtex: str) -> None:
        assert source.id is not None
        key = extract_citation_key(raw_bibtex)
        if self._repo.get_bib_entry_by_key(key) is not None:
            raise DuplicateSourceError(source, f"citation key {key!r}")
        self._repo.set_bib_entry(
            BibEntry(
                source_id=source.id,
                citation_key=key,
                raw_bibtex=raw_bibtex,
                origin=BibOrigin.FETCHED,
            )
        )
