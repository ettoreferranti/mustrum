"""PDF text extraction via PyMuPDF (TextExtractor port, FR-1.1)."""

from __future__ import annotations

import html
from pathlib import Path

import pymupdf


class PdfExtractor:
    extraction_method = "pymupdf"

    def extract(self, path: Path) -> str:
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            return "\n".join(page.get_text() for page in doc)


def _plausible_title(value: str | None) -> str | None:
    """PDF metadata titles are often junk (filenames, 'untitled', tool names);
    accept only strings that look like an actual paper title."""
    if not value:
        return None
    title = " ".join(html.unescape(value).split())
    if len(title) < 8 or len(title) > 300:
        return None
    if " " not in title:  # real titles have spaces; filenames usually don't
        return None
    lowered = title.lower()
    if lowered.startswith(("microsoft word", "untitled", "doi:")) or lowered.endswith(".pdf"):
        return None
    return title


def pdf_metadata_title(path: Path) -> str | None:
    """The paper's title from PDF metadata, if it looks trustworthy."""
    with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
        return _plausible_title(doc.metadata.get("title"))


def pdf_metadata_title_bytes(data: bytes) -> str | None:
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        return _plausible_title(doc.metadata.get("title"))


def extract_pdf_bytes(data: bytes) -> str:
    """Extract text from in-memory PDF bytes (downloaded open-access PDFs)."""
    with pymupdf.open(stream=data, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
        return "\n".join(page.get_text() for page in doc)


class PlainTextExtractor:
    """Passthrough for .txt / .md sources (FR-1.3)."""

    extraction_method = "plaintext"

    def extract(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")


def extractor_for(path: Path) -> PdfExtractor | PlainTextExtractor:
    if path.suffix.lower() == ".pdf":
        return PdfExtractor()
    return PlainTextExtractor()
