"""BibTeX helpers (FR-5.1). Fetched entries are stored byte-exact; these
helpers only read the citation key out of them, or build a key + minimal
entry for sources that were never fetched (origin=derived)."""

from __future__ import annotations

import re

from mustrum.core.models import Source

_ENTRY_KEY = re.compile(r"@\s*[A-Za-z]+\s*\{\s*([^,\s}]+)\s*,")
_KEY_CLEAN = re.compile(r"[^a-z0-9]+")


def extract_citation_key(raw_bibtex: str) -> str:
    """The key of the first entry in a raw BibTeX string."""
    match = _ENTRY_KEY.search(raw_bibtex)
    if not match:
        raise ValueError("no citation key found in BibTeX entry")
    return match.group(1)


def _surname(author: str) -> str:
    """Last whitespace-separated token; handles 'Given Family' and 'Family, Given'."""
    if "," in author:
        return author.split(",")[0].strip()
    parts = author.split()
    return parts[-1] if parts else ""


def make_citation_key(source: Source, existing: set[str]) -> str:
    """surname + year + first significant title word, unique via a/b/c suffix."""
    surname = _KEY_CLEAN.sub("", _surname(source.authors[0]).lower()) if source.authors else "anon"
    year = str(source.year) if source.year else "nd"
    words = [w for w in _KEY_CLEAN.sub(" ", source.title.lower()).split() if len(w) > 3]
    first_word = words[0] if words else "untitled"
    base = f"{surname}{year}{first_word}"
    if base not in existing:
        return base
    for suffix in "abcdefghijklmnopqrstuvwxyz":
        if f"{base}{suffix}" not in existing:
            return f"{base}{suffix}"
    raise ValueError(f"could not find a unique citation key for base {base!r}")


def render_derived_entry(source: Source, citation_key: str) -> str:
    """Minimal BibTeX from stored metadata only — no invented fields (NFR-1)."""
    kind = "article" if source.kind.value == "paper" else "misc"
    lines = [f"@{kind}{{{citation_key},", f"  title = {{{source.title}}},"]
    if source.authors:
        lines.append(f"  author = {{{' and '.join(source.authors)}}},")
    if source.year is not None:
        lines.append(f"  year = {{{source.year}}},")
    if source.doi:
        lines.append(f"  doi = {{{source.doi}}},")
    if source.arxiv_id:
        lines.append(f"  eprint = {{{source.arxiv_id}}},")
        lines.append("  archiveprefix = {arXiv},")
    lines.append("}")
    return "\n".join(lines)
