"""Normalisation helpers used for deduplication (FR-1.4)."""

from __future__ import annotations

import hashlib
import re

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_title(title: str) -> str:
    """Lowercase, strip everything but letters/digits, collapse to single spaces."""
    return _NON_ALNUM.sub(" ", title.lower()).strip()


def title_hash(title: str) -> str:
    """Stable hash of the normalised title, used as a dedup key."""
    return hashlib.sha256(normalize_title(title).encode("utf-8")).hexdigest()


def normalize_doi(doi: str) -> str:
    """Strip URL prefixes and lowercase: DOIs are case-insensitive by spec."""
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix) :]
    return doi
