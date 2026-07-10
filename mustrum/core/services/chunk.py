"""Text chunking for embeddings (E4-1): paragraph-preserving, greedy."""

from __future__ import annotations


def chunk_text(text: str, max_chars: int = 1500) -> list[str]:
    """Split into chunks of at most max_chars, preferring paragraph breaks.

    Paragraphs longer than max_chars are split hard. Empty input → no chunks.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:max_chars])
            paragraph = paragraph[max_chars:].strip()
        if not paragraph:
            continue
        if current and len(current) + 2 + len(paragraph) > max_chars:
            chunks.append(current)
            current = paragraph
        elif current:
            current = f"{current}\n\n{paragraph}"
        else:
            current = paragraph
    if current:
        chunks.append(current)
    return chunks
