"""Open-access full-text retrieval.

For DOIs, the Unpaywall API (api.unpaywall.org) locates a *legal* open-access
PDF if one exists — paywalled papers simply return None. For arXiv ids the
PDF is always available from arxiv.org. Unpaywall requires a contact e-mail
(their fair-use policy), taken from config.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx

from mustrum.adapters.arxiv import normalize_arxiv_id
from mustrum.core.models import FetchedMetadata


@dataclass(frozen=True)
class FullTextResult:
    """Outcome of the PDF hunt: extracted text, human-readable notes, and —
    when a download succeeded — the raw PDF bytes for the file archive
    (E1-11)."""

    text: str = ""
    notes: list[str] = field(default_factory=list)
    pdf_bytes: bytes | None = None


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{normalize_arxiv_id(arxiv_id)}"


class OpenAccessClient:
    def __init__(self, email: str, client: httpx.Client | None = None) -> None:
        if not email:
            raise ValueError("Unpaywall requires a contact e-mail (config: unpaywall_email)")
        self._email = email
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True)

    def find_pdf_url(self, doi: str) -> str | None:
        """Best open-access PDF URL for a DOI, or None if none is known."""
        response = self._client.get(
            f"https://api.unpaywall.org/v2/{doi}", params={"email": self._email}
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        data = response.json()
        location = data.get("best_oa_location") or {}
        url = location.get("url_for_pdf")
        return url if isinstance(url, str) and url else None

    def download_pdf(self, url: str) -> bytes:
        response = self._client.get(url)
        response.raise_for_status()
        content = response.content
        if not content.startswith(b"%PDF"):
            raise ValueError(f"{url} did not return a PDF")
        return content


def fetch_full_text(meta: FetchedMetadata, unpaywall_email: str) -> FullTextResult:
    """Try every candidate PDF URL for fetched metadata; shared by CLI and GUI.

    Candidates, in order: arXiv (always open), an Unpaywall open-access copy,
    then the publisher's Crossref full-text links (succeed only on networks
    with subscription access). `text` is '' when no candidate worked; the
    notes explain what happened.
    """
    from mustrum.adapters.pdf import extract_pdf_bytes

    notes: list[str] = []
    client = OpenAccessClient(email=unpaywall_email or "unused@localhost")
    candidates: list[str] = []
    if meta.arxiv_id:
        candidates.append(arxiv_pdf_url(meta.arxiv_id))
    if meta.doi and not meta.arxiv_id:
        if unpaywall_email:
            try:
                if found := client.find_pdf_url(meta.doi):
                    candidates.append(found)
            except Exception as exc:
                notes.append(f"Unpaywall lookup failed ({exc})")
        else:
            notes.append(
                "no unpaywall_email configured — skipping open-access lookup "
                "(set it in ~/.config/mustrum/config.toml)"
            )
    candidates.extend(meta.pdf_urls)

    for url in candidates:
        try:
            pdf_bytes = client.download_pdf(url)
            text = extract_pdf_bytes(pdf_bytes)
        except Exception as exc:
            notes.append(f"PDF fetch failed from {url} ({exc})")
            continue
        notes.append(f"fetched full text from {url}")
        return FullTextResult(text=text, notes=notes, pdf_bytes=pdf_bytes)
    if candidates or meta.doi:
        notes.append(
            "no downloadable PDF — storing abstract only"
            if meta.abstract
            else "no downloadable PDF and no abstract — stored metadata + BibTeX "
            "only; attach the paper manually (source attach / GUI Add PDF)"
        )
    return FullTextResult(notes=notes)
