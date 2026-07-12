"""The rigour kernel (ADR-7, NFR-1).

Model output is untrusted input. Nothing citation-bearing is stored or
emitted unless it passes these checks:

- GroundingVerifier: every evidence quote must occur verbatim in the stored
  source text, and there must be at least one quote — claims without evidence
  fail. "Verbatim" is whitespace- and typography-normalised (Unicode NFKC
  plus quote/dash folding, see _normalize): publisher PDFs use curly quotes,
  ligatures, and typographic dashes that models reproduce as ASCII, and that
  glyph-level variance must not mask genuinely identical wording. Case and
  the words themselves remain strict, with one sanctioned exception
  (ADR-15): the case of a quote's FIRST character is folded, because quoting
  a mid-sentence span as a sentence conventionally recapitalises the first
  word — that is quoting convention, not a wording change.
- CitationVerifier: every citation key used in generated text must exist in
  the database's key set. Supports LaTeX (`\\cite{...}` and biblatex/natbib
  variants) and pandoc-Markdown (`[@key]`, `@key`) citations.

Verification failure is a rejection: callers must discard the artefact,
never repair it silently.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence, Set
from dataclasses import dataclass

# typographic variants folded to ASCII before comparison; NFKC handles
# ligatures (ﬁ→fi) and non-breaking spaces but not quotes/dashes
_TYPOGRAPHY = str.maketrans(
    {
        "‘": "'",  # ‘
        "’": "'",  # ’
        "‚": "'",  # ‚
        "‛": "'",  # ‛
        "“": '"',  # “
        "”": '"',  # ”
        "„": '"',  # „
        "‐": "-",  # ‐ hyphen
        "‑": "-",  # ‑ non-breaking hyphen
        "‒": "-",  # ‒ figure dash
        "–": "-",  # – en dash
        "—": "-",  # — em dash
        "−": "-",  # − minus sign
        "­": None,  # soft hyphen: drop entirely
    }
)


def _normalize(text: str) -> str:
    """Whitespace + typography normalisation (see module docstring)."""
    folded = unicodedata.normalize("NFKC", text).translate(_TYPOGRAPHY)
    return " ".join(folded.split())


def _first_char_variants(quote: str) -> tuple[str, ...]:
    """The quote plus, when it starts with a cased letter, the same quote
    with only that first character's case swapped (ADR-15). Everything after
    the first character stays strict."""
    if quote and quote[0].isalpha():
        head = quote[0]
        swapped = (head.lower() if head.isupper() else head.upper()) + quote[1:]
        if swapped != quote:
            return (quote, swapped)
    return (quote,)


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
        usable = [q for q in quotes if _normalize(q)]
        if not usable:
            return GroundingResult(ok=False, missing_quotes=(), empty_evidence=True)
        haystack = _normalize(source_text)
        missing = tuple(
            q for q in usable if not any(v in haystack for v in _first_char_variants(_normalize(q)))
        )
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
