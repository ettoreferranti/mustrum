"""Ingestion pipeline (FR-1): store source + immutable text + embeddings +
BibTeX, with dedup on DOI / arXiv id / normalised title (FR-1.4) and
per-field provenance (FR-1.5).

Core receives already-extracted text and already-fetched metadata — file and
network I/O stay in the adapters/CLI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mustrum.core.bibtex import extract_citation_key, make_citation_key, render_derived_entry
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
from mustrum.core.refimport import ParsedReference
from mustrum.core.services.chunk import chunk_text

OnDuplicate = Literal["fail", "skip", "merge"]


def embed_source_text(
    repo: StorageRepo, embedder: EmbeddingProvider, source_id: int, text: str
) -> None:
    """Chunk + embed a source's text. Single definition so ingest, text
    upgrades, and backup restore all embed identically."""
    chunks = chunk_text(text)
    vectors = embedder.embed(chunks)
    repo.store_embeddings(
        [
            Embedding(
                entity=EntityKind.SOURCE,
                ref_id=source_id,
                chunk_index=i,
                model=embedder.model_name,
                vector=vector,
            )
            for i, vector in enumerate(vectors)
        ]
    )


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

    def ingest_reference(
        self,
        ref: ParsedReference,
        kind: SourceKind = SourceKind.PAPER,
        on_duplicate: OnDuplicate = "skip",
    ) -> IngestResult:
        """Ingest one entry parsed from a reference-manager export (BibTeX
        or RIS, E9-4). Fields are FieldOrigin.EXTRACTED — parsed out of the
        imported file, not fetched from an authoritative service — and dedup
        reuses the same DOI/arXiv-id/title-hash matching as every other
        ingest path.
        """
        provenance = tuple(
            (field, FieldOrigin.EXTRACTED)
            for field, value in (
                ("title", ref.title),
                ("authors", ref.authors),
                ("year", ref.year),
                ("doi", ref.doi),
                ("arxiv_id", ref.arxiv_id),
            )
            if value
        )
        source = Source(
            kind=kind,
            title=ref.title,
            authors=ref.authors,
            year=ref.year,
            doi=ref.doi,
            arxiv_id=ref.arxiv_id,
            provenance=provenance,
        )
        duplicate = self._find_duplicate(source)
        if duplicate is not None:
            result = self._handle_duplicate(
                source, ref.abstract, "abstract", *duplicate, on_duplicate
            )
            if result.merged and self._repo.get_bib_entry(result.source.id) is None:  # type: ignore[arg-type]
                self._attach_reference_bib(result.source, ref)
            return result
        saved = self._repo.add_source(source)
        if ref.abstract:
            self._attach_text(saved, ref.abstract, "abstract")
        self._attach_reference_bib(saved, ref)
        return IngestResult(source=saved, created=True)

    def attach_full_text(self, source_id: int, text: str, extraction_method: str) -> None:
        """Attach a full text to an existing source, or upgrade an abstract
        (ADR-9). A stored full text is never silently replaced.

        Upgrading invalidates everything derived from the old text: the
        summary is deleted and the embeddings are recomputed, so nothing in
        the library stays grounded against text that is no longer there.
        """
        source = self._repo.get_source(source_id)
        if not text.strip():
            raise ValueError("no text to attach")
        existing = self._repo.get_source_text(source_id)
        if existing is None:
            self._attach_text(source, text, extraction_method)
            return
        if existing.extraction_method != "abstract":
            raise ValueError(
                f"source {source_id} already has full text "
                f"({existing.extraction_method}); refusing to replace it"
            )
        self._repo.replace_source_text(
            SourceText(source_id=source_id, text=text, extraction_method=extraction_method)
        )
        self._repo.delete_summary(source_id)
        self._repo.delete_embeddings(EntityKind.SOURCE, source_id)
        embed_source_text(self._repo, self._embedder, source_id, text)

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
            file_path=existing.file_path,
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
        embed_source_text(self._repo, self._embedder, source.id, text)

    def _attach_fetched_bib(self, source: Source, raw_bibtex: str) -> None:
        assert source.id is not None
        key = extract_citation_key(raw_bibtex)
        if self._repo.get_bib_entry_by_key(key) is not None:
            # publishers derive keys like Author_Year, which collide across
            # papers; duplicate keys are unusable in LaTeX, so de-duplicate by
            # suffix and rewrite exactly the key token in the raw entry
            # (ADR-12: the sole sanctioned amendment to fetched BibTeX)
            existing = self._repo.citation_keys()
            for suffix in "abcdefghijklmnopqrstuvwxyz":
                if (candidate := f"{key}{suffix}") not in existing:
                    break
            else:  # pragma: no cover - 26 collisions on one key
                raise ValueError(f"could not derive a unique citation key from {key!r}")
            raw_bibtex = raw_bibtex.replace(f"{{{key},", f"{{{candidate},", 1)
            key = candidate
        self._repo.set_bib_entry(
            BibEntry(
                source_id=source.id,
                citation_key=key,
                raw_bibtex=raw_bibtex,
                origin=BibOrigin.FETCHED,
            )
        )

    def _attach_reference_bib(self, source: Source, ref: ParsedReference) -> None:
        """A BibTeX import carries its own byte-exact entry (stored FETCHED,
        same as an online fetch); a RIS import has no BibTeX form, so one is
        rendered from the parsed fields (DERIVED) — the same fallback
        `related-work`'s bib export uses for any source with none fetched."""
        assert source.id is not None
        if ref.raw_bibtex is not None:
            self._attach_fetched_bib(source, ref.raw_bibtex)
            return
        key = make_citation_key(source, self._repo.citation_keys())
        self._repo.set_bib_entry(
            BibEntry(
                source_id=source.id,
                citation_key=key,
                raw_bibtex=render_derived_entry(source, key),
                origin=BibOrigin.DERIVED,
            )
        )
