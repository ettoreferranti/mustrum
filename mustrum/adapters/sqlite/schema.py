"""Versioned schema migrations. PRAGMA user_version tracks the applied version.

Rules:
- Never edit an existing migration; append a new one.
- source_texts is immutable by design (ADR-7): triggers reject UPDATE/DELETE.
"""

from __future__ import annotations

import sqlite3

MIGRATIONS: list[str] = [
    # v1 — initial schema
    """
    CREATE TABLE sources (
        id INTEGER PRIMARY KEY,
        kind TEXT NOT NULL,
        title TEXT NOT NULL,
        authors TEXT NOT NULL DEFAULT '[]',      -- JSON array
        year INTEGER,
        doi TEXT UNIQUE,                          -- normalised
        arxiv_id TEXT UNIQUE,
        title_hash TEXT NOT NULL,
        provenance TEXT NOT NULL DEFAULT '{}',    -- JSON {field: origin}
        reading_status TEXT NOT NULL DEFAULT 'unread',
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_sources_title_hash ON sources(title_hash);

    CREATE TABLE source_texts (
        source_id INTEGER PRIMARY KEY REFERENCES sources(id),
        text TEXT NOT NULL,
        extraction_method TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE TRIGGER source_texts_immutable_update
        BEFORE UPDATE ON source_texts
        BEGIN SELECT RAISE(ABORT, 'source_texts is immutable (ADR-7)'); END;
    CREATE TRIGGER source_texts_immutable_delete
        BEFORE DELETE ON source_texts
        BEGIN SELECT RAISE(ABORT, 'source_texts is immutable (ADR-7)'); END;

    CREATE TABLE summaries (
        source_id INTEGER PRIMARY KEY REFERENCES sources(id),
        text TEXT NOT NULL,
        evidence TEXT NOT NULL DEFAULT '[]',      -- JSON array of quotes
        model TEXT NOT NULL,
        verified INTEGER NOT NULL,
        user_override INTEGER NOT NULL DEFAULT 0,
        created_at TEXT NOT NULL
    );

    CREATE TABLE ideas (
        id INTEGER PRIMARY KEY,
        title TEXT NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE idea_versions (
        id INTEGER PRIMARY KEY,
        idea_id INTEGER NOT NULL REFERENCES ideas(id),
        text TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_idea_versions_idea ON idea_versions(idea_id);

    CREATE TABLE idea_links (
        from_idea_id INTEGER NOT NULL REFERENCES ideas(id),
        to_idea_id INTEGER NOT NULL REFERENCES ideas(id),
        relation TEXT NOT NULL,
        PRIMARY KEY (from_idea_id, to_idea_id, relation)
    );

    CREATE TABLE matches (
        id INTEGER PRIMARY KEY,
        idea_id INTEGER NOT NULL REFERENCES ideas(id),
        source_id INTEGER NOT NULL REFERENCES sources(id),
        score REAL NOT NULL,
        status TEXT NOT NULL DEFAULT 'suggested',
        rationale TEXT NOT NULL DEFAULT '',
        quotes TEXT NOT NULL DEFAULT '[]',        -- JSON array
        created_at TEXT NOT NULL,
        UNIQUE (idea_id, source_id)
    );

    CREATE TABLE bib_entries (
        source_id INTEGER PRIMARY KEY REFERENCES sources(id),
        citation_key TEXT NOT NULL UNIQUE,
        raw_bibtex TEXT NOT NULL,
        origin TEXT NOT NULL
    );

    CREATE TABLE contacts (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        affiliation TEXT NOT NULL DEFAULT '',
        email TEXT NOT NULL DEFAULT '',
        url TEXT NOT NULL DEFAULT '',
        notes TEXT NOT NULL DEFAULT '',
        created_at TEXT NOT NULL
    );

    CREATE TABLE contact_links (
        contact_id INTEGER NOT NULL REFERENCES contacts(id),
        idea_id INTEGER REFERENCES ideas(id),
        source_id INTEGER REFERENCES sources(id),
        why TEXT NOT NULL,
        CHECK ((idea_id IS NULL) != (source_id IS NULL))
    );

    CREATE TABLE tags (
        entity TEXT NOT NULL,
        ref_id INTEGER NOT NULL,
        tag TEXT NOT NULL,
        PRIMARY KEY (entity, ref_id, tag)
    );

    CREATE TABLE embeddings (
        entity TEXT NOT NULL,
        ref_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        model TEXT NOT NULL,
        vector BLOB NOT NULL,
        PRIMARY KEY (entity, ref_id, chunk_index, model)
    );

    CREATE VIRTUAL TABLE search_index USING fts5(
        entity UNINDEXED,
        ref_id UNINDEXED,
        body
    );
    """,
    # v2 — archived original file per source (E1-11 / ADR-13); the value is a
    # file name relative to the `files/` directory next to the database
    """
    ALTER TABLE sources ADD COLUMN file_path TEXT;
    """,
]


def apply_migrations(conn: sqlite3.Connection) -> None:
    current = conn.execute("PRAGMA user_version").fetchone()[0]
    for version, script in enumerate(MIGRATIONS[current:], start=current + 1):
        conn.executescript(script)
        conn.execute(f"PRAGMA user_version = {version}")
    conn.commit()
