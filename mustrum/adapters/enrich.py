"""Metadata enrichment: complete a bare PDF-ingested source with
authoritative Crossref metadata found by exact-title lookup.

Shared by CLI (`source enrich`) and GUI. Rigour gates: the Crossref match
must have an identical normalised title, and its DOI must not already belong
to a different source — otherwise nothing changes and the reason is
reported. Merging never overwrites fields the source already has (FR-1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from mustrum.adapters.crossref import CrossrefFetcher
from mustrum.core.ports import EmbeddingProvider, StorageRepo
from mustrum.core.services.ingest import IngestService


@dataclass(frozen=True)
class EnrichResult:
    enriched: bool
    message: str


def enrich_source(
    repo: StorageRepo,
    embedder: EmbeddingProvider,
    source_id: int,
    client: httpx.Client | None = None,
) -> EnrichResult:
    source = repo.get_source(source_id)  # raises KeyError for the caller
    if source.doi and repo.get_bib_entry(source_id) is not None:
        return EnrichResult(False, "already has DOI and BibTeX — nothing to enrich")
    meta = CrossrefFetcher(client=client).search_by_title(source.title)
    if meta is None:
        return EnrichResult(
            False,
            "no confident Crossref match — a hit must carry this exact title. "
            "Fix the title or ingest by DOI; if the venue has no DOIs at all "
            "(e.g. CEUR-WS workshop proceedings), set the metadata by hand: "
            'source edit ID --author "..." --year YYYY',
        )
    if meta.doi:
        clash = repo.find_source_by_doi(meta.doi)
        if clash is not None and clash.id != source_id:
            return EnrichResult(
                False,
                f"Crossref resolves this title to DOI {meta.doi}, which already "
                f"belongs to [{clash.id}] {clash.title}",
            )
    result = IngestService(repo, embedder).ingest_fetched(meta, on_duplicate="merge")
    assert result.merged, "exact-title match must merge into the existing source"
    fields = ", ".join(
        name
        for name, value in (
            ("authors", meta.authors),
            ("year", meta.year),
            ("doi", meta.doi),
        )
        if value
    )
    return EnrichResult(True, f"enriched from Crossref ({fields}; BibTeX attached)")
