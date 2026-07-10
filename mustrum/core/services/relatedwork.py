"""Bibliography export and related-work skeleton generation (FR-5).

The skeleton is built deterministically from confirmed matches: citation
keys and summaries come straight from the database; nothing here can invent
a source. As defence in depth, the assembled text still passes through
CitationVerifier before being returned (FR-5.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from mustrum.core.bibtex import make_citation_key, render_derived_entry
from mustrum.core.models import BibEntry, BibOrigin, Match, MatchStatus, Source, Summary
from mustrum.core.ports import StorageRepo
from mustrum.core.verify import CitationVerifier

SkeletonFormat = Literal["markdown", "latex"]


class CitationIntegrityError(Exception):
    """Raised if generated text ever references a key not in the library."""


@dataclass(frozen=True)
class SkeletonEntry:
    source: Source
    citation_key: str
    summary: Summary | None
    match: Match


class RelatedWorkService:
    def __init__(self, repo: StorageRepo) -> None:
        self._repo = repo
        self._verifier = CitationVerifier()

    # -- bibliography (FR-5.1/5.2) -------------------------------------------

    def ensure_bib_entry(self, source_id: int) -> BibEntry:
        """Return the source's BibTeX entry, deriving one from stored metadata
        if it was never fetched (origin=derived, FR-5.1)."""
        existing = self._repo.get_bib_entry(source_id)
        if existing is not None:
            return existing
        source = self._repo.get_source(source_id)
        key = make_citation_key(source, self._repo.citation_keys())
        entry = BibEntry(
            source_id=source_id,
            citation_key=key,
            raw_bibtex=render_derived_entry(source, key),
            origin=BibOrigin.DERIVED,
        )
        self._repo.set_bib_entry(entry)
        return entry

    def export_bib(self, idea_id: int | None = None) -> str:
        """The whole library's .bib, or only sources confirmed for an idea."""
        if idea_id is None:
            sources = self._repo.list_sources()
        else:
            self._repo.get_idea(idea_id)
            matches = self._repo.list_matches(idea_id, MatchStatus.CONFIRMED)
            sources = [self._repo.get_source(m.source_id) for m in matches]
        entries = [self.ensure_bib_entry(s.id) for s in sources if s.id is not None]
        return "\n\n".join(e.raw_bibtex for e in entries) + ("\n" if entries else "")

    # -- related-work skeleton (FR-5.3) ----------------------------------------

    def skeleton(self, idea_id: int, fmt: SkeletonFormat = "markdown") -> str:
        idea = self._repo.get_idea(idea_id)
        version = self._repo.latest_idea_version(idea_id)
        matches = self._repo.list_matches(idea_id, MatchStatus.CONFIRMED)
        entries = [
            SkeletonEntry(
                source=self._repo.get_source(m.source_id),
                citation_key=self.ensure_bib_entry(m.source_id).citation_key,
                summary=self._repo.get_summary(m.source_id),
                match=m,
            )
            for m in matches
        ]
        idea_text = version.text if version else ""
        if fmt == "latex":
            text = self._render_latex(idea.title, idea_text, entries)
        else:
            text = self._render_markdown(idea.title, idea_text, entries)
        result = self._verifier.verify(text, self._repo.citation_keys())
        if not result.ok:
            raise CitationIntegrityError(
                f"generated skeleton cites unknown keys: {list(result.unknown_keys)}"
            )
        return text

    @staticmethod
    def _entry_lines(entry: SkeletonEntry) -> tuple[str, str, str]:
        source = entry.source
        authors = ", ".join(source.authors) if source.authors else "(authors unknown)"
        year = str(source.year) if source.year else "n.d."
        heading = f"{source.title} ({year})"
        if entry.summary is not None:
            origin = "user" if entry.summary.user_override else entry.summary.model
            summary = f"{entry.summary.text} [summary: {origin}, verified]"
        else:
            summary = "TODO: no verified summary stored for this source yet."
        relevance = f"match score {entry.match.score:.2f}" + (
            f" — {entry.match.rationale}" if entry.match.rationale else ""
        )
        return heading, f"{authors}. {summary}", relevance

    def _render_markdown(self, title: str, idea_text: str, entries: list[SkeletonEntry]) -> str:
        lines = [
            f"# Related work — {title}",
            "",
            "> Skeleton assembled from confirmed matches in the library.",
            "> Every citation key below exists in the exported .bib file.",
            "",
        ]
        if idea_text:
            lines += [f"Research idea: {idea_text}", ""]
        if not entries:
            lines += ["*No confirmed matches for this idea yet.*", ""]
        for entry in entries:
            heading, body, relevance = self._entry_lines(entry)
            lines += [
                f"## {heading} [@{entry.citation_key}]",
                "",
                body,
                "",
                f"*Relevance: {relevance}.*",
                "",
                "TODO: relate this work to your contribution.",
                "",
            ]
        return "\n".join(lines)

    def _render_latex(self, title: str, idea_text: str, entries: list[SkeletonEntry]) -> str:
        lines = [
            "\\section{Related Work}",
            f"% skeleton for idea: {title}",
            "% every \\cite key below exists in the exported .bib file",
            "",
        ]
        if idea_text:
            lines += [f"% research idea: {idea_text}", ""]
        if not entries:
            lines += ["% no confirmed matches for this idea yet", ""]
        for entry in entries:
            heading, body, relevance = self._entry_lines(entry)
            lines += [
                f"\\paragraph{{{heading}}}",
                f"\\cite{{{entry.citation_key}}} {body}",
                f"% relevance: {relevance}",
                "% TODO: relate this work to your contribution.",
                "",
            ]
        return "\n".join(lines)
