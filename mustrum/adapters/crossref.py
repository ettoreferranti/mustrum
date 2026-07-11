"""Crossref/DOI MetadataFetcher (FR-1.2): metadata from api.crossref.org and
BibTeX via doi.org content negotiation, stored byte-exact."""

from __future__ import annotations

from typing import Any

import httpx

from mustrum.core.models import FetchedMetadata
from mustrum.core.normalize import normalize_doi


class CrossrefFetcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def fetch(self, identifier: str) -> FetchedMetadata:
        doi = normalize_doi(identifier)
        response = self._client.get(f"https://api.crossref.org/works/{doi}")
        if response.status_code == 404:
            raise LookupError(f"DOI not found: {doi}")
        response.raise_for_status()
        work: dict[str, Any] = response.json()["message"]

        title_parts = work.get("title") or []
        if not title_parts:
            raise LookupError(f"DOI has no title metadata: {doi}")
        title = " ".join(title_parts[0].split())
        authors = tuple(
            " ".join(part for part in (a.get("given"), a.get("family")) if part)
            for a in work.get("author", [])
        )
        year = None
        date_parts = (work.get("issued") or {}).get("date-parts") or [[]]
        if date_parts[0]:
            year = int(date_parts[0][0])
        abstract = " ".join(work.get("abstract", "").split())
        # publisher text-mining links: downloadable where the network has access
        pdf_urls = tuple(
            link["URL"]
            for link in work.get("link", [])
            if isinstance(link.get("URL"), str)
            and link.get("content-type") in ("application/pdf", "unspecified")
        )

        bib_response = self._client.get(
            f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"}
        )
        if bib_response.status_code == 404:
            raise LookupError(f"no BibTeX available for DOI: {doi}")
        bib_response.raise_for_status()
        raw_bibtex = bib_response.text.strip()
        if not raw_bibtex.startswith("@"):
            raise LookupError(f"doi.org returned no BibTeX for {doi}")

        return FetchedMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            arxiv_id=None,
            raw_bibtex=raw_bibtex,
            abstract=abstract,
            pdf_urls=pdf_urls,
        )
