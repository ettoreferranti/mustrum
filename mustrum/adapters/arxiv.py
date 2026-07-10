"""arXiv MetadataFetcher (FR-1.2): authoritative metadata from the arXiv Atom
API and the official BibTeX export, stored byte-exact. Never synthesises."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

import httpx

from mustrum.core.models import FetchedMetadata

_ATOM = "{http://www.w3.org/2005/Atom}"
_ARXIV = "{http://arxiv.org/schemas/atom}"

# accepts "2501.12345", "2501.12345v2", "arXiv:2501.12345", old-style "cs/0112017"
_ID = re.compile(r"^(arxiv:)?(?P<id>(\d{4}\.\d{4,5})(v\d+)?|[a-z-]+(\.[A-Z]{2})?/\d{7})$", re.I)


def normalize_arxiv_id(identifier: str) -> str:
    match = _ID.match(identifier.strip())
    if not match:
        raise ValueError(f"not an arXiv id: {identifier!r}")
    return match.group("id")


class ArxivFetcher:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client(timeout=30.0, follow_redirects=True)

    def fetch(self, identifier: str) -> FetchedMetadata:
        arxiv_id = normalize_arxiv_id(identifier)
        response = self._client.get(
            "https://export.arxiv.org/api/query", params={"id_list": arxiv_id}
        )
        response.raise_for_status()
        root = ET.fromstring(response.text)
        entry = root.find(f"{_ATOM}entry")
        if entry is None or entry.find(f"{_ATOM}title") is None:
            raise LookupError(f"arXiv id not found: {arxiv_id}")
        title = " ".join((entry.findtext(f"{_ATOM}title") or "").split())
        if not title or title.lower() == "error":
            raise LookupError(f"arXiv id not found: {arxiv_id}")
        authors = tuple(
            name.text.strip()
            for author in entry.findall(f"{_ATOM}author")
            if (name := author.find(f"{_ATOM}name")) is not None and name.text
        )
        published = entry.findtext(f"{_ATOM}published") or ""
        year = int(published[:4]) if published[:4].isdigit() else None
        doi = entry.findtext(f"{_ARXIV}doi")
        abstract = " ".join((entry.findtext(f"{_ATOM}summary") or "").split())

        bib_response = self._client.get(f"https://arxiv.org/bibtex/{arxiv_id}")
        bib_response.raise_for_status()
        raw_bibtex = bib_response.text.strip()
        if not raw_bibtex.startswith("@"):
            raise LookupError(f"arXiv returned no BibTeX for {arxiv_id}")

        return FetchedMetadata(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            raw_bibtex=raw_bibtex,
            abstract=abstract,
        )
