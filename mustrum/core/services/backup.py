"""Plain-file export/restore (E9-1, NFR-5): the whole library as
human-readable files, so the data outlives the tool and can be git-versioned.

Layout of an export (paths are keys of the bundle mapping):
    manifest.json        format version + counts
    sources.json         all source records (canonical, round-trip)
    ideas.json           ideas + versions + links (canonical)
    matches.json         idea↔source matches incl. rationale/quotes
    contacts.json        contacts + their links
    texts/<id>.txt       verbatim source texts, one file each
    bib/<key>.bib        raw BibTeX, byte-exact, one file each
    ideas.md             generated view — same format `idea import` reads
    LIBRARY.md           generated human-readable overview

Embeddings are derived data and are NOT exported; restore recomputes them.
Restore only targets an empty database — there are no merge semantics.

Archived originals (the `files/` directory next to the DB, ADR-13) are binary
and NOT part of this text export; copy that directory alongside. Source
records keep their `file_path` so a restored DB finds the copied files again.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import datetime
from typing import Any

from mustrum.core.models import (
    BibEntry,
    BibOrigin,
    Contact,
    ContactKind,
    ContactLink,
    EntityKind,
    FieldOrigin,
    Idea,
    IdeaLink,
    IdeaRelation,
    IdeaVersion,
    Match,
    MatchStatus,
    ReadingStatus,
    Source,
    SourceKind,
    SourceText,
    Summary,
)
from mustrum.core.ports import EmbeddingProvider, StorageRepo
from mustrum.core.services.ideas import embed_idea
from mustrum.core.services.ingest import embed_source_text

FORMAT_VERSION = 1

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


class BackupError(Exception):
    pass


def _filename(name: str, taken: set[str]) -> str:
    base = _UNSAFE.sub("_", name) or "entry"
    candidate = base
    counter = 1
    while candidate in taken:
        counter += 1
        candidate = f"{base}-{counter}"
    taken.add(candidate)
    return candidate


def _dumps(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


class BackupService:
    def __init__(self, repo: StorageRepo, embedder: EmbeddingProvider) -> None:
        self._repo = repo
        self._embedder = embedder

    # -- export ---------------------------------------------------------------

    def export_data(self) -> dict[str, str]:
        files: dict[str, str] = {}
        sources = self._export_sources(files)
        ideas, idea_links = self._export_ideas()
        matches = [
            {
                "idea_id": m.idea_id,
                "source_id": m.source_id,
                "score": m.score,
                "status": m.status.value,
                "rationale": m.rationale,
                "quotes": list(m.quotes),
                "created_at": m.created_at.isoformat(),
            }
            for m in self._repo.list_matches()
        ]
        contacts = self._export_contacts()
        files["sources.json"] = _dumps(sources)
        files["ideas.json"] = _dumps({"ideas": ideas, "links": idea_links})
        files["matches.json"] = _dumps(matches)
        files["contacts.json"] = _dumps(contacts)
        files["ideas.md"] = self._render_ideas_md(ideas)
        files["LIBRARY.md"] = self._render_library_md(sources, ideas)
        files["manifest.json"] = _dumps(
            {
                "format": FORMAT_VERSION,
                "counts": {
                    "sources": len(sources),
                    "ideas": len(ideas),
                    "matches": len(matches),
                    "contacts": len(contacts),
                },
            }
        )
        return files

    def _export_sources(self, files: dict[str, str]) -> list[dict[str, Any]]:
        records = []
        bib_names: set[str] = set()
        for source in self._repo.list_sources():
            assert source.id is not None
            record: dict[str, Any] = {
                "id": source.id,
                "kind": source.kind.value,
                "title": source.title,
                "authors": list(source.authors),
                "year": source.year,
                "doi": source.doi,
                "arxiv_id": source.arxiv_id,
                "provenance": {f: o.value for f, o in source.provenance},
                "reading_status": source.reading_status.value,
                "notes": source.notes,
                "file_path": source.file_path,
                "created_at": source.created_at.isoformat(),
                "tags": sorted(self._repo.tags_for(EntityKind.SOURCE, source.id)),
                "text": None,
                "summary": None,
                "bib": None,
            }
            text = self._repo.get_source_text(source.id)
            if text is not None:
                path = f"texts/{source.id:04d}.txt"
                files[path] = text.text
                record["text"] = {
                    "file": path,
                    "extraction_method": text.extraction_method,
                    "created_at": text.created_at.isoformat(),
                }
            summary = self._repo.get_summary(source.id)
            if summary is not None:
                record["summary"] = {
                    "text": summary.text,
                    "evidence": list(summary.evidence),
                    "model": summary.model,
                    "verified": summary.verified,
                    "user_override": summary.user_override,
                    "created_at": summary.created_at.isoformat(),
                }
            bib = self._repo.get_bib_entry(source.id)
            if bib is not None:
                path = f"bib/{_filename(bib.citation_key, bib_names)}.bib"
                files[path] = bib.raw_bibtex
                record["bib"] = {
                    "citation_key": bib.citation_key,
                    "origin": bib.origin.value,
                    "file": path,
                }
            records.append(record)
        return records

    def _export_ideas(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        ideas = []
        for idea in self._repo.list_ideas():
            assert idea.id is not None
            ideas.append(
                {
                    "id": idea.id,
                    "title": idea.title,
                    "created_at": idea.created_at.isoformat(),
                    "tags": sorted(self._repo.tags_for(EntityKind.IDEA, idea.id)),
                    "versions": [
                        {"text": v.text, "created_at": v.created_at.isoformat()}
                        for v in self._repo.get_idea_versions(idea.id)
                    ],
                }
            )
        links = [
            {"from": link.from_idea_id, "to": link.to_idea_id, "relation": link.relation.value}
            for link in self._repo.list_idea_links()
        ]
        return ideas, links

    def _export_contacts(self) -> list[dict[str, Any]]:
        links_by_contact: dict[int, list[dict[str, Any]]] = {}
        for link in self._repo.list_contact_links():
            links_by_contact.setdefault(link.contact_id, []).append(
                {"idea_id": link.idea_id, "source_id": link.source_id, "why": link.why}
            )
        return [
            {
                "id": c.id,
                "name": c.name,
                "kind": c.kind.value,
                "affiliation": c.affiliation,
                "email": c.email,
                "url": c.url,
                "notes": c.notes,
                "created_at": c.created_at.isoformat(),
                "links": links_by_contact.get(c.id, []),  # type: ignore[arg-type]
            }
            for c in self._repo.list_contacts()
        ]

    @staticmethod
    def _render_ideas_md(ideas: list[dict[str, Any]]) -> str:
        blocks = [
            f"# {idea['title']}\n\n{idea['versions'][-1]['text']}\n"
            for idea in ideas
            if idea["versions"]
        ]
        return "\n".join(blocks) if blocks else "(no ideas yet)\n"

    @staticmethod
    def _render_library_md(sources: list[dict[str, Any]], ideas: list[dict[str, Any]]) -> str:
        lines = ["# Mustrum library export", "", "## Sources", ""]
        for s in sources:
            year = f" ({s['year']})" if s["year"] else ""
            key = f" `[@{s['bib']['citation_key']}]`" if s["bib"] else ""
            lines.append(f"- **{s['title']}**{year}{key}")
            if s["summary"]:
                lines.append(f"  - {s['summary']['text']}")
        lines += ["", "## Ideas", ""]
        for idea in ideas:
            lines.append(f"- **{idea['title']}**")
            if idea["versions"]:
                lines.append(f"  - {idea['versions'][-1]['text']}")
        return "\n".join(lines) + "\n"

    # -- restore ---------------------------------------------------------------

    def import_data(self, bundle: Mapping[str, str]) -> dict[str, int]:
        manifest = self._parse(bundle, "manifest.json")
        if manifest.get("format") != FORMAT_VERSION:
            raise BackupError(
                f"unsupported export format {manifest.get('format')!r}; "
                f"this tool reads format {FORMAT_VERSION}"
            )
        if self._repo.list_sources() or self._repo.list_ideas() or self._repo.list_contacts():
            raise BackupError("restore requires an empty database")
        source_map = self._import_sources(bundle)
        idea_map = self._import_ideas(bundle)
        matches = self._import_matches(bundle, idea_map, source_map)
        contacts = self._import_contacts(bundle, idea_map, source_map)
        return {
            "sources": len(source_map),
            "ideas": len(idea_map),
            "matches": matches,
            "contacts": contacts,
        }

    def _parse(self, bundle: Mapping[str, str], path: str) -> Any:
        if path not in bundle:
            raise BackupError(f"export is missing {path}")
        try:
            return json.loads(bundle[path])
        except json.JSONDecodeError as exc:
            raise BackupError(f"{path} is not valid JSON: {exc}") from exc

    def _import_sources(self, bundle: Mapping[str, str]) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for rec in self._parse(bundle, "sources.json"):
            saved = self._repo.add_source(
                Source(
                    kind=SourceKind(rec["kind"]),
                    title=rec["title"],
                    authors=tuple(rec["authors"]),
                    year=rec["year"],
                    doi=rec["doi"],
                    arxiv_id=rec["arxiv_id"],
                    provenance=tuple((f, FieldOrigin(o)) for f, o in rec["provenance"].items()),
                    reading_status=ReadingStatus(rec["reading_status"]),
                    notes=rec["notes"],
                    # optional: absent in pre-E1-11 exports
                    file_path=rec.get("file_path"),
                    created_at=datetime.fromisoformat(rec["created_at"]),
                )
            )
            assert saved.id is not None
            mapping[rec["id"]] = saved.id
            if rec["text"] is not None:
                path = rec["text"]["file"]
                if path not in bundle:
                    raise BackupError(f"export is missing {path}")
                self._repo.add_source_text(
                    SourceText(
                        source_id=saved.id,
                        text=bundle[path],
                        extraction_method=rec["text"]["extraction_method"],
                        created_at=datetime.fromisoformat(rec["text"]["created_at"]),
                    )
                )
                embed_source_text(self._repo, self._embedder, saved.id, bundle[path])
            if rec["summary"] is not None:
                s = rec["summary"]
                self._repo.set_summary(
                    Summary(
                        source_id=saved.id,
                        text=s["text"],
                        evidence=tuple(s["evidence"]),
                        model=s["model"],
                        verified=s["verified"],
                        user_override=s["user_override"],
                        created_at=datetime.fromisoformat(s["created_at"]),
                    )
                )
            if rec["bib"] is not None:
                path = rec["bib"]["file"]
                if path not in bundle:
                    raise BackupError(f"export is missing {path}")
                self._repo.set_bib_entry(
                    BibEntry(
                        source_id=saved.id,
                        citation_key=rec["bib"]["citation_key"],
                        raw_bibtex=bundle[path],
                        origin=BibOrigin(rec["bib"]["origin"]),
                    )
                )
            for tag in rec["tags"]:
                self._repo.tag(EntityKind.SOURCE, saved.id, tag)
        return mapping

    def _import_ideas(self, bundle: Mapping[str, str]) -> dict[int, int]:
        data = self._parse(bundle, "ideas.json")
        mapping: dict[int, int] = {}
        for rec in data["ideas"]:
            saved = self._repo.add_idea(
                Idea(title=rec["title"], created_at=datetime.fromisoformat(rec["created_at"]))
            )
            assert saved.id is not None
            mapping[rec["id"]] = saved.id
            for version in rec["versions"]:
                self._repo.add_idea_version(
                    IdeaVersion(
                        idea_id=saved.id,
                        text=version["text"],
                        created_at=datetime.fromisoformat(version["created_at"]),
                    )
                )
            if rec["versions"]:
                embed_idea(
                    self._repo,
                    self._embedder,
                    saved.id,
                    rec["title"],
                    rec["versions"][-1]["text"],
                )
            for tag in rec["tags"]:
                self._repo.tag(EntityKind.IDEA, saved.id, tag)
        for link in data["links"]:
            self._repo.add_idea_link(
                IdeaLink(
                    from_idea_id=mapping[link["from"]],
                    to_idea_id=mapping[link["to"]],
                    relation=IdeaRelation(link["relation"]),
                )
            )
        return mapping

    def _import_matches(
        self, bundle: Mapping[str, str], idea_map: dict[int, int], source_map: dict[int, int]
    ) -> int:
        count = 0
        for rec in self._parse(bundle, "matches.json"):
            self._repo.add_match(
                Match(
                    idea_id=idea_map[rec["idea_id"]],
                    source_id=source_map[rec["source_id"]],
                    score=rec["score"],
                    status=MatchStatus(rec["status"]),
                    rationale=rec["rationale"],
                    quotes=tuple(rec["quotes"]),
                    created_at=datetime.fromisoformat(rec["created_at"]),
                )
            )
            count += 1
        return count

    def _import_contacts(
        self, bundle: Mapping[str, str], idea_map: dict[int, int], source_map: dict[int, int]
    ) -> int:
        count = 0
        for rec in self._parse(bundle, "contacts.json"):
            saved = self._repo.add_contact(
                Contact(
                    name=rec["name"],
                    kind=ContactKind(rec["kind"]),
                    affiliation=rec["affiliation"],
                    email=rec["email"],
                    url=rec["url"],
                    notes=rec["notes"],
                    created_at=datetime.fromisoformat(rec["created_at"]),
                )
            )
            assert saved.id is not None
            count += 1
            for link in rec["links"]:
                self._repo.add_contact_link(
                    ContactLink(
                        contact_id=saved.id,
                        why=link["why"],
                        idea_id=idea_map[link["idea_id"]] if link["idea_id"] else None,
                        source_id=source_map[link["source_id"]] if link["source_id"] else None,
                    )
                )
        return count
