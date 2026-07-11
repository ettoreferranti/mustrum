"""Crossref/DOI MetadataFetcher (FR-1.2): metadata from api.crossref.org and
BibTeX via doi.org content negotiation, stored byte-exact. Also supports
looking a paper up *by title* (search_by_title) — used to enrich sources
ingested from bare PDFs. A search result only counts when its title matches
the query exactly under normalisation: no fuzzy guessing (NFR-1)."""

from __future__ import annotations

from typing import Any

import httpx

from mustrum.core.models import FetchedMetadata
from mustrum.core.normalize import normalize_doi, title_hash


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
        title = self._work_title(work)
        if title is None:
            raise LookupError(f"DOI has no title metadata: {doi}")
        return self._build_metadata(work, title, doi)

    def search_by_title(self, title: str) -> FetchedMetadata | None:
        """Authoritative metadata for a paper found by its exact title.

        Returns None unless one of the top Crossref hits has a title that is
        identical to the query under normalisation (case/punctuation folded).
        """
        response = self._client.get(
            "https://api.crossref.org/works",
            params={"query.title": title, "rows": 5},
        )
        response.raise_for_status()
        wanted = title_hash(title)
        for work in response.json()["message"].get("items", []):
            candidate = self._work_title(work)
            if candidate is None or title_hash(candidate) != wanted:
                continue
            doi = work.get("DOI")
            if not isinstance(doi, str) or not doi:
                continue
            return self._build_metadata(work, candidate, normalize_doi(doi))
        return None

    @staticmethod
    def _work_title(work: dict[str, Any]) -> str | None:
        title_parts = work.get("title") or []
        if not title_parts:
            return None
        return " ".join(title_parts[0].split())

    def _build_metadata(self, work: dict[str, Any], title: str, doi: str) -> FetchedMetadata:
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
        return FetchedMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            arxiv_id=None,
            raw_bibtex=self._fetch_bibtex(doi),
            abstract=abstract,
            pdf_urls=pdf_urls,
        )

    def _fetch_bibtex(self, doi: str) -> str:
        bib_response = self._client.get(
            f"https://doi.org/{doi}", headers={"Accept": "application/x-bibtex"}
        )
        if bib_response.status_code == 404:
            raise LookupError(f"no BibTeX available for DOI: {doi}")
        bib_response.raise_for_status()
        raw_bibtex = bib_response.text.strip()
        if not raw_bibtex.startswith("@"):
            raise LookupError(f"doi.org returned no BibTeX for {doi}")
        return raw_bibtex
