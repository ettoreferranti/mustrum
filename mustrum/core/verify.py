"""The rigour kernel (ADR-7, NFR-1).

Model output is untrusted input. Nothing citation-bearing is stored or
emitted unless it passes these checks:

- GroundingVerifier: every evidence quote must occur verbatim
  (whitespace-normalised, case-sensitive) in the stored source text, and
  there must be at least one quote — claims without evidence fail.
- CitationVerifier: every citation key used in generated text must exist in
  the database's key set. Supports LaTeX (`\\cite{...}` and biblatex/natbib
  variants) and pandoc-Markdown (`[@key]`, `@key`) citations.

Verification failure is a rejection: callers must discard the artefact,
never repair it silently.
"""

from __future__ import annotations

import re
from collections.abc import Sequence, Set
from dataclasses import dataclass


def _normalize_ws(text: str) -> str:
    """Collapse all whitespace runs (spaces, newlines, tabs) to single spaces."""
    return " ".join(text.split())


@dataclass(frozen=True)
class GroundingResult:
    ok: bool
    missing_quotes: tuple[str, ...]  # quotes not found verbatim in the source
    empty_evidence: bool  # no usable quotes were supplied at all


@dataclass(frozen=True)
class CitationResult:
    ok: bool
    used_keys: tuple[str, ...]  # unique, in order of first appearance
    unknown_keys: tuple[str, ...]  # used but not in the valid key set


class GroundingVerifier:
    """Checks that evidence quotes actually appear in the source text."""

    def verify(self, quotes: Sequence[str], source_text: str) -> GroundingResult:
        usable = [q for q in quotes if _normalize_ws(q)]
        if not usable:
            return GroundingResult(ok=False, missing_quotes=(), empty_evidence=True)
        haystack = _normalize_ws(source_text)
        missing = tuple(q for q in usable if _normalize_ws(q) not in haystack)
        return GroundingResult(ok=not missing, missing_quotes=missing, empty_evidence=False)


# LaTeX: \cite, \citep, \citet, \citealp, \autocite, \parencite, \textcite,
# \footcite, ... — any command whose name contains "cite", with optional
# star and up to two optional [] arguments, e.g. \citep*[see][p. 3]{a,b}.
_LATEX_CITE = re.compile(r"\\[A-Za-z]*[Cc]ite[A-Za-z]*\*?(?:\[[^\]]*\]){0,2}\{([^}]*)\}")

# pandoc-Markdown: @key with punctuation allowed only inside the key, so a
# trailing "." or ";" is not swallowed. The lookbehind rejects keys preceded
# by word characters or '.'/'@' (e-mail addresses, "name@host").
_MD_CITE = re.compile(r"(?<![\w.@])@([A-Za-z0-9_]+(?:[:.#$%&+?<>~/-]+[A-Za-z0-9_]+)*)")


class CitationVerifier:
    """Checks that generated text cites only keys that exist in the library."""

    def extract_keys(self, text: str) -> tuple[str, ...]:
        """All citation keys in order of first appearance, deduplicated."""
        found: list[tuple[int, str]] = []
        for match in _LATEX_CITE.finditer(text):
            for raw in match.group(1).split(","):
                key = raw.strip()
                if key:
                    found.append((match.start(), key))
        for match in _MD_CITE.finditer(text):
            found.append((match.start(), match.group(1)))
        keys: list[str] = []
        for _, key in sorted(found, key=lambda pair: pair[0]):
            if key not in keys:
                keys.append(key)
        return tuple(keys)

    def verify(self, text: str, valid_keys: Set[str]) -> CitationResult:
        used = self.extract_keys(text)
        unknown = tuple(k for k in used if k not in valid_keys)
        return CitationResult(ok=not unknown, used_keys=used, unknown_keys=unknown)
