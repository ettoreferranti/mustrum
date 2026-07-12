"""SQLite implementation of the StorageRepo port (ADR-2).

Storage conventions:
- datetimes stored as ISO-8601 UTC strings
- list/dict fields (authors, provenance, evidence, quotes) stored as JSON
- embedding vectors stored as float64 blobs
- the search_index FTS5 table is rebuilt per entity whenever its text changes
"""

from __future__ import annotations

import json
import sqlite3
import struct
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from mustrum.core.models import (
    BibEntry,
    BibOrigin,
    Contact,
    ContactKind,
    ContactLink,
    Embedding,
    EntityKind,
    FieldOrigin,
    Idea,
    IdeaLink,
    IdeaRelation,
    IdeaVersion,
    Match,
    MatchStatus,
    ReadingStatus,
    SearchHit,
    Source,
    SourceKind,
    SourceText,
    Summary,
)
from mustrum.core.normalize import normalize_doi, title_hash

from .schema import apply_migrations


def _dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _pack(vector: tuple[float, ...]) -> bytes:
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack(blob: bytes) -> tuple[float, ...]:
    return struct.unpack(f"<{len(blob) // 8}d", blob)


def _fts_query(query: str) -> str:
    """Quote each token so user input can't break FTS5 query syntax."""
    tokens = [t.replace('"', "") for t in query.split()]
    return " ".join(f'"{t}"' for t in tokens if t)


class SqliteRepo:
    def __init__(self, path: Path | str) -> None:
        # check_same_thread=False: the web adapter serves from a threadpool;
        # CPython's sqlite3 is compiled serialized (threadsafety==3), so a
        # shared connection is safe for this single-user tool
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        apply_migrations(self._conn)

    def close(self) -> None:
        self._conn.close()

    # -- sources -----------------------------------------------------------

    def add_source(self, source: Source) -> Source:
        cur = self._conn.execute(
            """INSERT INTO sources
               (kind, title, authors, year, doi, arxiv_id, title_hash,
                provenance, reading_status, notes, file_path, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                source.kind.value,
                source.title,
                json.dumps(list(source.authors)),
                source.year,
                normalize_doi(source.doi) if source.doi else None,
                source.arxiv_id,
                title_hash(source.title),
                json.dumps({f: o.value for f, o in source.provenance}),
                source.reading_status.value,
                source.notes,
                source.file_path,
                _dt(source.created_at),
            ),
        )
        self._conn.commit()
        saved = Source(
            kind=source.kind,
            title=source.title,
            authors=source.authors,
            year=source.year,
            doi=normalize_doi(source.doi) if source.doi else None,
            arxiv_id=source.arxiv_id,
            provenance=source.provenance,
            reading_status=source.reading_status,
            notes=source.notes,
            file_path=source.file_path,
            created_at=source.created_at,
            id=cur.lastrowid,
        )
        self._reindex_source(saved.id)  # type: ignore[arg-type]
        return saved

    def update_source(self, source: Source) -> None:
        """Update metadata of an existing source (dedup merge, FR-1.4)."""
        if source.id is None:
            raise ValueError("source has no id")
        self.get_source(source.id)
        self._conn.execute(
            """UPDATE sources SET kind = ?, title = ?, authors = ?, year = ?, doi = ?,
               arxiv_id = ?, title_hash = ?, provenance = ?, reading_status = ?, notes = ?,
               file_path = ?
               WHERE id = ?""",
            (
                source.kind.value,
                source.title,
                json.dumps(list(source.authors)),
                source.year,
                normalize_doi(source.doi) if source.doi else None,
                source.arxiv_id,
                title_hash(source.title),
                json.dumps({f: o.value for f, o in source.provenance}),
                source.reading_status.value,
                source.notes,
                source.file_path,
                source.id,
            ),
        )
        self._conn.commit()
        self._reindex_source(source.id)

    def _row_to_source(self, row: sqlite3.Row) -> Source:
        return Source(
            kind=SourceKind(row["kind"]),
            title=row["title"],
            authors=tuple(json.loads(row["authors"])),
            year=row["year"],
            doi=row["doi"],
            arxiv_id=row["arxiv_id"],
            provenance=tuple((f, FieldOrigin(o)) for f, o in json.loads(row["provenance"]).items()),
            reading_status=ReadingStatus(row["reading_status"]),
            notes=row["notes"],
            file_path=row["file_path"],
            created_at=_parse_dt(row["created_at"]),
            id=row["id"],
        )

    def get_source(self, source_id: int) -> Source:
        row = self._conn.execute("SELECT * FROM sources WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"no source with id {source_id}")
        return self._row_to_source(row)

    def list_sources(self) -> list[Source]:
        rows = self._conn.execute("SELECT * FROM sources ORDER BY id").fetchall()
        return [self._row_to_source(r) for r in rows]

    def find_source_by_doi(self, doi: str) -> Source | None:
        row = self._conn.execute(
            "SELECT * FROM sources WHERE doi = ?", (normalize_doi(doi),)
        ).fetchone()
        return self._row_to_source(row) if row else None

    def find_source_by_arxiv_id(self, arxiv_id: str) -> Source | None:
        row = self._conn.execute("SELECT * FROM sources WHERE arxiv_id = ?", (arxiv_id,)).fetchone()
        return self._row_to_source(row) if row else None

    def find_source_by_title_hash(self, hash_: str) -> Source | None:
        row = self._conn.execute("SELECT * FROM sources WHERE title_hash = ?", (hash_,)).fetchone()
        return self._row_to_source(row) if row else None

    def set_reading_status(self, source_id: int, status: ReadingStatus) -> None:
        self.get_source(source_id)
        self._conn.execute(
            "UPDATE sources SET reading_status = ? WHERE id = ?", (status.value, source_id)
        )
        self._conn.commit()

    def delete_source(self, source_id: int) -> None:
        """Remove a source and everything derived from or attached to it.

        Full-record removal is a user right and distinct from tampering
        (ADR-7/ADR-11): the immutability triggers on source_texts are dropped
        and recreated around the cascade, and no dependent row survives.
        """
        self.get_source(source_id)
        try:
            self._conn.execute("DROP TRIGGER source_texts_immutable_update")
            self._conn.execute("DROP TRIGGER source_texts_immutable_delete")
            for statement in (
                "DELETE FROM matches WHERE source_id = ?",
                "DELETE FROM contact_links WHERE source_id = ?",
                "DELETE FROM tags WHERE entity = 'source' AND ref_id = ?",
                "DELETE FROM embeddings WHERE entity = 'source' AND ref_id = ?",
                "DELETE FROM summaries WHERE source_id = ?",
                "DELETE FROM bib_entries WHERE source_id = ?",
                "DELETE FROM source_texts WHERE source_id = ?",
                "DELETE FROM search_index WHERE entity = 'source' AND ref_id = ?",
                "DELETE FROM sources WHERE id = ?",
            ):
                self._conn.execute(statement, (source_id,))
        finally:
            self._recreate_source_text_triggers()
        self._conn.commit()

    def delete_idea(self, idea_id: int) -> None:
        """Remove an idea with its versions, links, matches, and tags."""
        self.get_idea(idea_id)
        for statement in (
            "DELETE FROM matches WHERE idea_id = ?",
            "DELETE FROM idea_links WHERE from_idea_id = ? OR to_idea_id = ?",
            "DELETE FROM contact_links WHERE idea_id = ?",
            "DELETE FROM tags WHERE entity = 'idea' AND ref_id = ?",
            "DELETE FROM embeddings WHERE entity = 'idea' AND ref_id = ?",
            "DELETE FROM idea_versions WHERE idea_id = ?",
            "DELETE FROM search_index WHERE entity = 'idea' AND ref_id = ?",
            "DELETE FROM ideas WHERE id = ?",
        ):
            params = (idea_id, idea_id) if statement.count("?") == 2 else (idea_id,)
            self._conn.execute(statement, params)
        self._conn.commit()

    def set_source_notes(self, source_id: int, notes: str) -> None:
        self.get_source(source_id)
        self._conn.execute("UPDATE sources SET notes = ? WHERE id = ?", (notes, source_id))
        self._conn.commit()
        self._reindex_source(source_id)

    # -- source texts (immutable, ADR-7) -------------------------------------

    def add_source_text(self, text: SourceText) -> None:
        self._conn.execute(
            """INSERT INTO source_texts (source_id, text, extraction_method, created_at)
               VALUES (?, ?, ?, ?)""",
            (text.source_id, text.text, text.extraction_method, _dt(text.created_at)),
        )
        self._conn.commit()
        self._reindex_source(text.source_id)

    def replace_source_text(self, text: SourceText) -> None:
        """Controlled text upgrade (ADR-9): the immutability triggers are
        dropped and recreated around the swap, atomically. Callers (the ingest
        service) must invalidate summaries/embeddings derived from the old
        text — this method only swaps the text."""
        if self.get_source_text(text.source_id) is None:
            raise KeyError(f"source {text.source_id} has no text to replace")
        try:
            self._conn.execute("DROP TRIGGER source_texts_immutable_update")
            self._conn.execute("DROP TRIGGER source_texts_immutable_delete")
            self._conn.execute("DELETE FROM source_texts WHERE source_id = ?", (text.source_id,))
            self._conn.execute(
                """INSERT INTO source_texts (source_id, text, extraction_method, created_at)
                   VALUES (?, ?, ?, ?)""",
                (text.source_id, text.text, text.extraction_method, _dt(text.created_at)),
            )
        finally:
            self._recreate_source_text_triggers()
        self._conn.commit()
        self._reindex_source(text.source_id)

    def _recreate_source_text_triggers(self) -> None:
        self._conn.executescript(
            """
            CREATE TRIGGER IF NOT EXISTS source_texts_immutable_update
                BEFORE UPDATE ON source_texts
                BEGIN SELECT RAISE(ABORT, 'source_texts is immutable (ADR-7)'); END;
            CREATE TRIGGER IF NOT EXISTS source_texts_immutable_delete
                BEFORE DELETE ON source_texts
                BEGIN SELECT RAISE(ABORT, 'source_texts is immutable (ADR-7)'); END;
            """
        )

    def get_source_text(self, source_id: int) -> SourceText | None:
        row = self._conn.execute(
            "SELECT * FROM source_texts WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return SourceText(
            source_id=row["source_id"],
            text=row["text"],
            extraction_method=row["extraction_method"],
            created_at=_parse_dt(row["created_at"]),
        )

    # -- summaries -------------------------------------------------------------

    def set_summary(self, summary: Summary) -> None:
        self._conn.execute(
            """INSERT INTO summaries
               (source_id, text, evidence, model, verified, user_override, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(source_id) DO UPDATE SET
                 text=excluded.text, evidence=excluded.evidence, model=excluded.model,
                 verified=excluded.verified, user_override=excluded.user_override,
                 created_at=excluded.created_at""",
            (
                summary.source_id,
                summary.text,
                json.dumps(list(summary.evidence)),
                summary.model,
                int(summary.verified),
                int(summary.user_override),
                _dt(summary.created_at),
            ),
        )
        self._conn.commit()
        self._reindex_source(summary.source_id)

    def delete_summary(self, source_id: int) -> None:
        self._conn.execute("DELETE FROM summaries WHERE source_id = ?", (source_id,))
        self._conn.commit()
        self._reindex_source(source_id)

    def get_summary(self, source_id: int) -> Summary | None:
        row = self._conn.execute(
            "SELECT * FROM summaries WHERE source_id = ?", (source_id,)
        ).fetchone()
        if row is None:
            return None
        return Summary(
            source_id=row["source_id"],
            text=row["text"],
            evidence=tuple(json.loads(row["evidence"])),
            model=row["model"],
            verified=bool(row["verified"]),
            user_override=bool(row["user_override"]),
            created_at=_parse_dt(row["created_at"]),
        )

    # -- ideas -------------------------------------------------------------------

    def add_idea(self, idea: Idea) -> Idea:
        cur = self._conn.execute(
            "INSERT INTO ideas (title, created_at) VALUES (?, ?)",
            (idea.title, _dt(idea.created_at)),
        )
        self._conn.commit()
        saved = Idea(title=idea.title, created_at=idea.created_at, id=cur.lastrowid)
        self._reindex_idea(saved.id)  # type: ignore[arg-type]
        return saved

    def get_idea(self, idea_id: int) -> Idea:
        row = self._conn.execute("SELECT * FROM ideas WHERE id = ?", (idea_id,)).fetchone()
        if row is None:
            raise KeyError(f"no idea with id {idea_id}")
        return Idea(title=row["title"], created_at=_parse_dt(row["created_at"]), id=row["id"])

    def find_idea_by_title(self, title: str) -> Idea | None:
        row = self._conn.execute(
            "SELECT * FROM ideas WHERE title = ? ORDER BY id LIMIT 1", (title,)
        ).fetchone()
        if row is None:
            return None
        return Idea(title=row["title"], created_at=_parse_dt(row["created_at"]), id=row["id"])

    def list_ideas(self) -> list[Idea]:
        rows = self._conn.execute("SELECT * FROM ideas ORDER BY id").fetchall()
        return [
            Idea(title=r["title"], created_at=_parse_dt(r["created_at"]), id=r["id"]) for r in rows
        ]

    def add_idea_version(self, version: IdeaVersion) -> IdeaVersion:
        self.get_idea(version.idea_id)
        cur = self._conn.execute(
            "INSERT INTO idea_versions (idea_id, text, created_at) VALUES (?, ?, ?)",
            (version.idea_id, version.text, _dt(version.created_at)),
        )
        self._conn.commit()
        self._reindex_idea(version.idea_id)
        return IdeaVersion(
            idea_id=version.idea_id,
            text=version.text,
            created_at=version.created_at,
            id=cur.lastrowid,
        )

    def _row_to_idea_version(self, row: sqlite3.Row) -> IdeaVersion:
        return IdeaVersion(
            idea_id=row["idea_id"],
            text=row["text"],
            created_at=_parse_dt(row["created_at"]),
            id=row["id"],
        )

    def get_idea_versions(self, idea_id: int) -> list[IdeaVersion]:
        rows = self._conn.execute(
            "SELECT * FROM idea_versions WHERE idea_id = ? ORDER BY id", (idea_id,)
        ).fetchall()
        return [self._row_to_idea_version(r) for r in rows]

    def latest_idea_version(self, idea_id: int) -> IdeaVersion | None:
        row = self._conn.execute(
            "SELECT * FROM idea_versions WHERE idea_id = ? ORDER BY id DESC LIMIT 1",
            (idea_id,),
        ).fetchone()
        return self._row_to_idea_version(row) if row else None

    def add_idea_link(self, link: IdeaLink) -> None:
        self._conn.execute(
            "INSERT INTO idea_links (from_idea_id, to_idea_id, relation) VALUES (?, ?, ?)",
            (link.from_idea_id, link.to_idea_id, link.relation.value),
        )
        self._conn.commit()

    def list_idea_links(self, idea_id: int | None = None) -> list[IdeaLink]:
        if idea_id is None:
            rows = self._conn.execute("SELECT * FROM idea_links").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM idea_links WHERE from_idea_id = ? OR to_idea_id = ?",
                (idea_id, idea_id),
            ).fetchall()
        return [
            IdeaLink(
                from_idea_id=r["from_idea_id"],
                to_idea_id=r["to_idea_id"],
                relation=IdeaRelation(r["relation"]),
            )
            for r in rows
        ]

    # -- matches --------------------------------------------------------------------

    def add_match(self, match: Match) -> Match:
        cur = self._conn.execute(
            """INSERT INTO matches
               (idea_id, source_id, score, status, rationale, quotes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                match.idea_id,
                match.source_id,
                match.score,
                match.status.value,
                match.rationale,
                json.dumps(list(match.quotes)),
                _dt(match.created_at),
            ),
        )
        self._conn.commit()
        return Match(
            idea_id=match.idea_id,
            source_id=match.source_id,
            score=match.score,
            status=match.status,
            rationale=match.rationale,
            quotes=match.quotes,
            created_at=match.created_at,
            id=cur.lastrowid,
        )

    def get_match(self, match_id: int) -> Match:
        row = self._conn.execute("SELECT * FROM matches WHERE id = ?", (match_id,)).fetchone()
        if row is None:
            raise KeyError(f"no match with id {match_id}")
        return self._row_to_match(row)

    def set_match_rationale(self, match_id: int, rationale: str, quotes: tuple[str, ...]) -> None:
        cur = self._conn.execute(
            "UPDATE matches SET rationale = ?, quotes = ? WHERE id = ?",
            (rationale, json.dumps(list(quotes)), match_id),
        )
        if cur.rowcount == 0:
            raise KeyError(f"no match with id {match_id}")
        self._conn.commit()

    def set_match_status(self, match_id: int, status: MatchStatus) -> None:
        cur = self._conn.execute(
            "UPDATE matches SET status = ? WHERE id = ?", (status.value, match_id)
        )
        if cur.rowcount == 0:
            raise KeyError(f"no match with id {match_id}")
        self._conn.commit()

    def list_matches(
        self, idea_id: int | None = None, status: MatchStatus | None = None
    ) -> list[Match]:
        clauses: list[str] = []
        params: list[int | str] = []
        if idea_id is not None:
            clauses.append("idea_id = ?")
            params.append(idea_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status.value)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM matches {where} ORDER BY score DESC", params
        ).fetchall()
        return [self._row_to_match(r) for r in rows]

    def _row_to_match(self, row: sqlite3.Row) -> Match:
        return Match(
            idea_id=row["idea_id"],
            source_id=row["source_id"],
            score=row["score"],
            status=MatchStatus(row["status"]),
            rationale=row["rationale"],
            quotes=tuple(json.loads(row["quotes"])),
            created_at=_parse_dt(row["created_at"]),
            id=row["id"],
        )

    # -- bibliography ------------------------------------------------------------------

    def set_bib_entry(self, entry: BibEntry) -> None:
        self._conn.execute(
            """INSERT INTO bib_entries (source_id, citation_key, raw_bibtex, origin)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(source_id) DO UPDATE SET
                 citation_key=excluded.citation_key, raw_bibtex=excluded.raw_bibtex,
                 origin=excluded.origin""",
            (entry.source_id, entry.citation_key, entry.raw_bibtex, entry.origin.value),
        )
        self._conn.commit()

    def _row_to_bib(self, row: sqlite3.Row) -> BibEntry:
        return BibEntry(
            source_id=row["source_id"],
            citation_key=row["citation_key"],
            raw_bibtex=row["raw_bibtex"],
            origin=BibOrigin(row["origin"]),
        )

    def get_bib_entry(self, source_id: int) -> BibEntry | None:
        row = self._conn.execute(
            "SELECT * FROM bib_entries WHERE source_id = ?", (source_id,)
        ).fetchone()
        return self._row_to_bib(row) if row else None

    def get_bib_entry_by_key(self, citation_key: str) -> BibEntry | None:
        row = self._conn.execute(
            "SELECT * FROM bib_entries WHERE citation_key = ?", (citation_key,)
        ).fetchone()
        return self._row_to_bib(row) if row else None

    def list_bib_entries(self) -> list[BibEntry]:
        rows = self._conn.execute("SELECT * FROM bib_entries ORDER BY source_id").fetchall()
        return [self._row_to_bib(r) for r in rows]

    def citation_keys(self) -> set[str]:
        rows = self._conn.execute("SELECT citation_key FROM bib_entries").fetchall()
        return {r["citation_key"] for r in rows}

    # -- contacts ------------------------------------------------------------------

    def add_contact(self, contact: Contact) -> Contact:
        cur = self._conn.execute(
            """INSERT INTO contacts (name, kind, affiliation, email, url, notes, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                contact.name,
                contact.kind.value,
                contact.affiliation,
                contact.email,
                contact.url,
                contact.notes,
                _dt(contact.created_at),
            ),
        )
        self._conn.commit()
        saved = Contact(
            name=contact.name,
            kind=contact.kind,
            affiliation=contact.affiliation,
            email=contact.email,
            url=contact.url,
            notes=contact.notes,
            created_at=contact.created_at,
            id=cur.lastrowid,
        )
        self._reindex_contact(saved.id)  # type: ignore[arg-type]
        return saved

    def _row_to_contact(self, row: sqlite3.Row) -> Contact:
        return Contact(
            name=row["name"],
            kind=ContactKind(row["kind"]),
            affiliation=row["affiliation"],
            email=row["email"],
            url=row["url"],
            notes=row["notes"],
            created_at=_parse_dt(row["created_at"]),
            id=row["id"],
        )

    def get_contact(self, contact_id: int) -> Contact:
        row = self._conn.execute("SELECT * FROM contacts WHERE id = ?", (contact_id,)).fetchone()
        if row is None:
            raise KeyError(f"no contact with id {contact_id}")
        return self._row_to_contact(row)

    def list_contacts(self) -> list[Contact]:
        rows = self._conn.execute("SELECT * FROM contacts ORDER BY id").fetchall()
        return [self._row_to_contact(r) for r in rows]

    def add_contact_link(self, link: ContactLink) -> None:
        self._conn.execute(
            "INSERT INTO contact_links (contact_id, idea_id, source_id, why) VALUES (?, ?, ?, ?)",
            (link.contact_id, link.idea_id, link.source_id, link.why),
        )
        self._conn.commit()

    def list_contact_links(
        self, idea_id: int | None = None, source_id: int | None = None
    ) -> list[ContactLink]:
        clauses: list[str] = []
        params: list[int] = []
        if idea_id is not None:
            clauses.append("idea_id = ?")
            params.append(idea_id)
        if source_id is not None:
            clauses.append("source_id = ?")
            params.append(source_id)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(f"SELECT * FROM contact_links {where}", params).fetchall()
        return [
            ContactLink(
                contact_id=r["contact_id"],
                idea_id=r["idea_id"],
                source_id=r["source_id"],
                why=r["why"],
            )
            for r in rows
        ]

    # -- tags ------------------------------------------------------------------------

    def tag(self, entity: EntityKind, ref_id: int, tag: str) -> None:
        self._conn.execute(
            "INSERT OR IGNORE INTO tags (entity, ref_id, tag) VALUES (?, ?, ?)",
            (entity.value, ref_id, tag),
        )
        self._conn.commit()

    def untag(self, entity: EntityKind, ref_id: int, tag: str) -> None:
        self._conn.execute(
            "DELETE FROM tags WHERE entity = ? AND ref_id = ? AND tag = ?",
            (entity.value, ref_id, tag),
        )
        self._conn.commit()

    def tags_for(self, entity: EntityKind, ref_id: int) -> set[str]:
        rows = self._conn.execute(
            "SELECT tag FROM tags WHERE entity = ? AND ref_id = ?", (entity.value, ref_id)
        ).fetchall()
        return {r["tag"] for r in rows}

    def entities_with_tag(self, tag: str) -> list[tuple[EntityKind, int]]:
        rows = self._conn.execute(
            "SELECT entity, ref_id FROM tags WHERE tag = ? ORDER BY entity, ref_id", (tag,)
        ).fetchall()
        return [(EntityKind(r["entity"]), r["ref_id"]) for r in rows]

    # -- embeddings ---------------------------------------------------------------------

    def store_embeddings(self, embeddings: Sequence[Embedding]) -> None:
        self._conn.executemany(
            """INSERT INTO embeddings (entity, ref_id, chunk_index, model, vector)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(entity, ref_id, chunk_index, model)
               DO UPDATE SET vector=excluded.vector""",
            [
                (e.entity.value, e.ref_id, e.chunk_index, e.model, _pack(e.vector))
                for e in embeddings
            ],
        )
        self._conn.commit()

    def embeddings_for(self, entity: EntityKind, model: str) -> list[Embedding]:
        rows = self._conn.execute(
            "SELECT * FROM embeddings WHERE entity = ? AND model = ? ORDER BY ref_id, chunk_index",
            (entity.value, model),
        ).fetchall()
        return [
            Embedding(
                entity=EntityKind(r["entity"]),
                ref_id=r["ref_id"],
                chunk_index=r["chunk_index"],
                model=r["model"],
                vector=_unpack(r["vector"]),
            )
            for r in rows
        ]

    def delete_embeddings(self, entity: EntityKind, ref_id: int) -> None:
        self._conn.execute(
            "DELETE FROM embeddings WHERE entity = ? AND ref_id = ?", (entity.value, ref_id)
        )
        self._conn.commit()

    # -- search (FR-8.1) -------------------------------------------------------------------

    def search(self, query: str, limit: int = 20) -> list[SearchHit]:
        fts = _fts_query(query)
        if not fts:
            return []
        rows = self._conn.execute(
            """SELECT entity, ref_id, snippet(search_index, 2, '[', ']', '…', 12) AS snip
               FROM search_index WHERE search_index MATCH ? ORDER BY rank LIMIT ?""",
            (fts, limit),
        ).fetchall()
        return [
            SearchHit(entity=EntityKind(r["entity"]), ref_id=r["ref_id"], snippet=r["snip"])
            for r in rows
        ]

    # -- FTS maintenance ----------------------------------------------------------

    def _set_index(self, entity: EntityKind, ref_id: int, body: str) -> None:
        self._conn.execute(
            "DELETE FROM search_index WHERE entity = ? AND ref_id = ?", (entity.value, ref_id)
        )
        self._conn.execute(
            "INSERT INTO search_index (entity, ref_id, body) VALUES (?, ?, ?)",
            (entity.value, ref_id, body),
        )
        self._conn.commit()

    def _reindex_source(self, source_id: int) -> None:
        source = self.get_source(source_id)
        text = self.get_source_text(source_id)
        summary = self.get_summary(source_id)
        parts = [source.title, " ".join(source.authors), source.notes]
        if text:
            parts.append(text.text)
        if summary:
            parts.append(summary.text)
        self._set_index(EntityKind.SOURCE, source_id, "\n".join(p for p in parts if p))

    def _reindex_idea(self, idea_id: int) -> None:
        idea = self.get_idea(idea_id)
        versions = self.get_idea_versions(idea_id)
        body = "\n".join([idea.title, *(v.text for v in versions)])
        self._set_index(EntityKind.IDEA, idea_id, body)

    def _reindex_contact(self, contact_id: int) -> None:
        c = self.get_contact(contact_id)
        body = "\n".join(p for p in [c.name, c.affiliation, c.notes] if p)
        self._set_index(EntityKind.CONTACT, contact_id, body)
