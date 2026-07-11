"""Open-access full-text retrieval.

For DOIs, the Unpaywall API (api.unpaywall.org) locates a *legal* open-access
PDF if one exists — paywalled papers simply return None. For arXiv ids the
PDF is always available from arxiv.org. Unpaywall requires a contact e-mail
(their fair-use policy), taken from config.
"""

from __future__ import annotations

import httpx

from mustrum.adapters.arxiv import normalize_arxiv_id


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
