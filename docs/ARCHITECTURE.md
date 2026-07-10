# Mustrum — Architecture

> **Living document.** Keep this in sync with the code on every structural
> change (new module, new adapter, schema migration, changed data flow).
> Last updated: 2026-07-10 (Phase 0 implemented: models, ports, SQLite
> adapter, fake providers; services/CLI/graph not yet built).

## 1. Overview

Mustrum is a local-first Python application with a hexagonal (ports &
adapters) architecture. A pure **core** holds domain logic and talks to the
outside world only through narrow interfaces (**ports**). **Adapters**
implement those ports for concrete technology (SQLite, Ollama, arXiv,
Crossref, PyMuPDF). The CLI is just another adapter driving the core.

```
                 ┌─────────────────────────────────────────┐
   CLI (typer)   │                 CORE                    │
  ───────────►   │  domain models · services · verifiers   │
                 │                                         │
  graph HTML ◄── │  ports:                                 │
  exporter       │   StorageRepo · LLMProvider             │
                 │   EmbeddingProvider · MetadataFetcher   │
                 │   TextExtractor                         │
                 └───────┬──────────┬──────────┬───────────┘
                         │          │          │
                   SQLite+FTS5   Ollama     arXiv / Crossref
                   adapter       adapter    clients · PyMuPDF
                                 (later: Anthropic adapter)
```

**Prime directive (NFR-1):** the core is designed so that inventing a
citation is structurally impossible — generated artefacts reference sources
only by database ID, rendering resolves IDs to stored BibTeX, and verifiers
run *after* any model output and reject ungrounded content.

## 2. Package layout

Items marked ⏳ are planned (Phase 1+), everything else exists.

```
mustrum/
  core/
    models.py        # Source, Idea, IdeaVersion, Contact, Match, BibEntry, ...
    normalize.py     # title/DOI normalisation + title_hash (dedup keys)
    ports.py         # Protocol definitions (all ports)
    verify.py        # ⏳ GroundingVerifier, CitationVerifier  ← rigour kernel
    services/        # ⏳ ingest, summarise, match, relatedwork, audit
  adapters/
    sqlite/          # StorageRepo impl: schema.py (migrations), repo.py
    fake.py          # deterministic fake providers for tests
    ollama.py        # ⏳ LLMProvider + EmbeddingProvider via Ollama HTTP API
    arxiv.py         # ⏳ MetadataFetcher for arXiv IDs
    crossref.py      # ⏳ MetadataFetcher for DOIs
    pdf.py           # ⏳ TextExtractor via PyMuPDF
  cli/               # ⏳ typer app: ingest, idea, match, bib, graph, audit
  graph/             # ⏳ self-contained HTML export (embedded Cytoscape.js)
tests/
  unit/  integration/
docs/
```

Implementation notes (v1 schema, `adapters/sqlite/`):
- datetimes ISO-8601 strings; list/dict fields JSON; vectors float64 blobs.
- `source_texts` immutability (ADR-7) is enforced *in the database* by
  BEFORE UPDATE/DELETE triggers that RAISE(ABORT), not just by convention.
- One FTS5 table `search_index(entity, ref_id, body)` covers sources
  (title+authors+notes+text+summary), ideas (title+versions), contacts;
  re-indexed per entity on write. User queries are token-quoted so FTS5
  syntax can't be injected.
- Migrations: append-only list in `schema.py`, tracked via PRAGMA
  user_version.

## 3. Domain model

| Entity | Key fields | Notes |
|---|---|---|
| `Source` | id, kind (paper/article/note), title, authors, year, doi, arxiv_id, provenance per field | metadata record |
| `SourceText` | source_id, verbatim extracted/ingested text, extraction method | **immutable** after ingest |
| `Summary` | source_id, text, evidence quotes[], model, created_at, verified flag, user_override | only stored if verification passes |
| `Idea` | id, title, current_version_id | |
| `IdeaVersion` | idea_id, text, created_at | append-only history |
| `Match` | idea_id, source_id, score, rationale + quotes, status (suggested/confirmed/rejected) | user status overrides machine |
| `IdeaLink` | idea_id ↔ idea_id, relation (builds-on/contrasts/related) | |
| `BibEntry` | source_id, citation_key, raw_bibtex, origin (fetched/derived) | fetched `.bib` stored byte-exact |
| `Contact` | id, name, kind (person/company/institution/university), affiliation, email/url, notes | |
| `ContactLink` | contact_id ↔ idea_id or source_id, why-relevant note | |
| `Tag` | name; many-to-many with sources and ideas | |

Embeddings are stored per idea-version and per source (chunked), in SQLite as
blobs, with the embedding model name — a model change invalidates and triggers
re-embedding.

## 4. Ports (interfaces)

All defined as `typing.Protocol` in `core/ports.py`; core code never imports
an adapter.

- `LLMProvider.generate(task, context) -> str` — plain-text generation.
  Implementations: `OllamaProvider` (phase 1), `AnthropicProvider` (phase 3),
  `FakeProvider` (tests). The interface is deliberately minimal so swapping
  providers is config-only.
- `EmbeddingProvider.embed(texts) -> vectors` — Ollama (`nomic-embed-text`)
  first; same swap story.
- `StorageRepo` — persistence for all entities + FTS queries. SQLite adapter.
- `MetadataFetcher.fetch(identifier) -> SourceMetadata + raw bibtex` — arXiv
  and Crossref implementations. Never called implicitly.
- `TextExtractor.extract(path) -> text` — PyMuPDF for PDFs; passthrough for
  text/Markdown.

## 5. The rigour kernel (`core/verify.py`)

The anti-hallucination guarantees live in two small, heavily-tested classes:

1. **GroundingVerifier** — takes model output that includes evidence quotes
   and the stored `SourceText`; verifies each quote occurs verbatim
   (whitespace-normalised) in the text. Failure ⇒ the artefact is rejected
   and reported; nothing partial is stored.
2. **CitationVerifier** — takes generated text (related-work skeleton) and the
   set of valid citation keys from the DB; any `\cite{key}` / `[@key]` not in
   the set ⇒ hard failure. Also used by the `audit` command on external drafts.

Model output is treated as untrusted input everywhere. These modules have the
strictest test bar in the project (see §7).

## 6. Key flows

- **Ingest PDF:** extract text → dedup check (DOI/arXiv/title-hash) → store
  Source + immutable SourceText → chunk + embed → (optional) grounded
  summarisation → verify → store Summary.
- **Ingest arXiv/DOI:** fetch metadata + BibTeX → same pipeline; fetched
  fields marked authoritative.
- **New idea / new version:** store version → embed → match against all source
  embeddings → present ranked suggestions → user confirms/rejects.
- **Related-work skeleton:** collect confirmed matches for the idea → for each
  source pull citation key + verified summary → LLM arranges/groups (may not
  add sources) → CitationVerifier + GroundingVerifier → render Markdown/LaTeX
  + matching `.bib`.
- **Graph export:** query entities/links → JSON → inline into HTML template
  with embedded Cytoscape.js → single file, no network.

## 7. Testing strategy

- Unit tests against fake providers; adapter integration tests behind markers
  (`-m ollama`, `-m network`) so the default suite is fully offline and
  deterministic.
- **Mutation testing with mutmut** on `mustrum/core/`: overall mutation score
  target ≥ 80%; for `core/verify.py` every surviving mutant must be reviewed
  and either killed or explicitly justified in the PR/commit. `core/ports.py`
  is excluded (Protocol stubs have no behaviour to mutate).
- Justified surviving mutants (keep this list current):
  - `normalize.title_hash`: `"utf-8"` → `"UTF-8"` — Python codec lookup is
    case-insensitive, mutant is semantically equivalent; unkillable.
- mypy strict on `mustrum/core/`; ruff for lint + format.

## 8. Decisions

Architecture decisions and their rationale are recorded in
[DECISIONS.md](DECISIONS.md). Current: ADR-1 Python, ADR-2 SQLite(+FTS5),
ADR-3 CLI + self-contained HTML graph, ADR-4 Ollama for both LLM and
embeddings in phase 1, ADR-5 hexagonal provider interface, ADR-6 mutmut,
ADR-7 immutable source texts + grounded generation.
