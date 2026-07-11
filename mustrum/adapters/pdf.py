"""PDF text extraction via PyMuPDF (TextExtractor port, FR-1.1)."""

from __future__ import annotations

from pathlib import Path

import pymupdf


class PdfExtractor:
    extraction_method = "pymupdf"

    def extract(self, path: Path) -> str:
        with pymupdf.open(path) as doc:  # type: ignore[no-untyped-call]
            return "\n".join(page.get_text() for page in doc)


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
