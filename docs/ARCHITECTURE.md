# Mustrum — Architecture

> **Living document.** Keep this in sync with the code on every structural
> change (new module, new adapter, schema migration, changed data flow).
> Last updated: 2026-07-15 (security hardening, ADR-25 — see §9: GUI output
> now HTML-escaped before `innerHTML` (graph page reached parity with the
> SPA, closing a stored-XSS-into-the-local-API hole); a cross-origin-write
> guard rejects browser writes from other sites to the unauthenticated
> loopback API; corrupt PDFs / non-UTF-8 files / offline fetches now fail
> cleanly instead of with a raw traceback (CLI) or opaque 500 (GUI). Earlier
> same day E9-4: reference-manager import — `mustrum ingest
> references <path>` bulk-imports a `.bib` or `.ris` export from Zotero,
> Mendeley, or any tool emitting these standard formats; new
> `core/refimport.py` parses either into `ParsedReference` records (one
> parser per format covers both tools); `IngestService.ingest_reference`
> feeds them through the same DOI/arXiv/title-hash dedup as every other
> ingest path with `FieldOrigin.EXTRACTED` provenance; a `.bib` entry's
> byte-exact text becomes its BibEntry, a `.ris` entry gets one rendered
> from parsed fields (reusing `core/bibtex.py`'s existing derived-entry
> fallback); a malformed entry is skipped with a warning, not an aborted
> file; validated against real Zotero and Mendeley exports, which
> surfaced and fixed three bugs — Mendeley's occasional empty BibTeX
> citation key, Zotero's BibLaTeX `date`-not-`year` field, and Zotero's
> RIS abstracts wrapped across untagged continuation lines; ADR-24;
> previous day E9-3: watch-folder auto-ingest — `mustrum watch
> <dir>` polls for new PDFs, ingesting one only once its size/mtime are
> unchanged across two consecutive scans (a download/sync in progress is
> left alone); resolved files move into `ingested/`/`failed/` inside the
> watched folder so re-scans stay bounded; `cli/main.py::_ingest_pdf` is
> shared with the existing `ingest folder` batch command (refactored,
> behavior unchanged) so the two never drift apart, ADR-23, no core
> changes; earlier same day E10-2: provider benchmarking harness —
> `mustrum/benchmark/harness.py::run_benchmark` runs fixed synthetic
> paper/idea fixtures through any `LLMProvider` via the unmodified
> `SummariseService`/`RationaleService` grounding loop and reports a pass
> rate; a provider with no credentials is `unavailable`, never a fabricated
> 0%; `mustrum benchmark --providers fake,ollama[,anthropic]` CLI surface,
> ADR-22, zero core changes; earlier same day E10-1: `AnthropicProvider` —
> `mustrum/adapters/anthropic.py::AnthropicLLM` implements `LLMProvider`
> unchanged, config-switchable via new `Config.llm_provider`/
> `anthropic_model`/`anthropic_max_tokens` + `cli/main.py::_build_llm` on the
> CLI side and `SettingsPayload`/`POST /api/settings` on the GUI side (same
> save-then-restart-notice model as ADR-16), ADR-21, zero core changes;
> plus a graceful-failure fix found via live testing without an API key set
> — new `mustrum/adapters/errors.py::ProviderError` base (`OllamaError`/
> `AnthropicError` both subclass it) lets `cli/main.py::main()` catch any
> provider failure at the real process entry point (one clean line +
> `SystemExit(1)`, not a raw traceback) and lets the GUI's new
> `@app.exception_handler(ProviderError)` turn the same failure into a
> flash-able 502 instead of an opaque 500; earlier same day E11-2: GUI tag editing (add/remove on sources and
> ideas via existing `tag`/`untag`), contact links (`POST`/`GET
> /api/{sources|ideas}/{id}/contacts`, GUI counterpart of `mustrum contact
> link`), and citation audit upload (`POST /api/audit`, GUI counterpart of
> `mustrum audit`) — no core changes, all three reuse existing services/repo
> methods through new thin API endpoints; earlier 2026-07-13: E13-4: MCP
> resources — every source/idea listed
> as an individually-readable `mustrum://sources|ideas/{id}` resource
> alongside E13-3's tools, ADR-20; earlier same day E13-3: MCP server
> adapter — `mustrum/mcp/server.py`, `mustrum mcp` (stdio), read-only
> `search_library`/`get_source`/`get_idea`/`list_citations`, ADR-19, zero
> core changes, no LLM call; earlier same day E13-2: conversational
> grounded chat —
> `core/services/chat.py::ChatSession` + `QueryService.ask()`'s new
> `history`/`extra_candidate_ids` params, ADR-18, `mustrum chat` CLI REPL,
> GUI Chat tab; earlier same day E13-1: grounded library-query core service —
> `core/services/query.py::QueryService` + `core/services/grounded.py::
> run_grounded_multi`, §5/§6/§7; earlier same day a fresh full-`mustrum/core/`
> mutmut run — 2084/2262 killed, 92.1%, verify.py still 100%, one genuine
> test gap found and documented in relatedwork.py; also E11-7 brainstorm
> select-to-save, E12-2 Ollama model dropdowns, E12-1 library-local
> settings file + GUI Settings panel ADR-16, `config` CLI show/init/set).

## 1. Overview

Mustrum is a local-first Python application with a hexagonal (ports &
adapters) architecture. A pure **core** holds domain logic and talks to the
outside world only through narrow interfaces (**ports**). **Adapters**
implement those ports for concrete technology (SQLite, Ollama, arXiv,
Crossref, PyMuPDF). The CLI is just another adapter driving the core.

```
                 ┌─────────────────────────────────────────┐
   CLI (typer)   │                 CORE                    │
   GUI (FastAPI) │  domain models · services · verifiers   │
   MCP (stdio)   │                                         │
  ───────────►   │                                         │
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

Everything below exists; `mustrum/config.py` holds user config: a global
bootstrap TOML (`~/.config/mustrum/config.toml`, sets `db_path`) layered
under a library TOML next to the database itself (`Config.library_config_path`,
everything else — ADR-16), then env vars. `save_library_config` is the one
writer, used by both `mustrum config set` and the GUI Settings panel.

```
mustrum/
  core/
    models.py        # Source, Idea, IdeaVersion, Contact, Match, BibEntry, ...
    normalize.py     # title/DOI normalisation + title_hash (dedup keys)
    bibtex.py        # citation-key derivation + BibTeX rendering for
                     #   sources with no fetched BibTeX (used by
                     #   related-work export and RIS import alike)
    refimport.py     # BibTeX/RIS parsing into ParsedReference (E9-4)
    ports.py         # Protocol definitions (all ports)
    verify.py        # GroundingVerifier, CitationVerifier  ← rigour kernel
    services/        # ingest, summarise, match, rationale, relatedwork,
                     #   audit, chunk, backup (plain-file export/restore),
                     #   brainstorm (quarantined creative mode),
                     #   grounded (shared generate→verify loop; also
                     #   run_grounded_multi, the multi-source variant),
                     #   query (E13-1: grounded Q&A over the library),
                     #   chat (E13-2: in-memory multi-turn ChatSession
                     #   wrapping QueryService, ADR-18)
  adapters/
    sqlite/          # StorageRepo impl: schema.py (migrations), repo.py
    fake.py          # deterministic fake providers for tests
    ollama.py        # OllamaLLM + OllamaEmbedder via Ollama HTTP API
    anthropic.py     # AnthropicLLM (E10-1, ADR-21): LLMProvider over the
                     #   Anthropic Messages API, config-switchable via
                     #   Config.llm_provider; no EmbeddingProvider (Anthropic
                     #   has none) — embeddings stay on Ollama either way
    errors.py        # ProviderError (ADR-21): shared base for OllamaError/
                     #   AnthropicError so CLI/GUI can catch any provider
                     #   failure without importing each adapter's own module
    arxiv.py         # MetadataFetcher for arXiv IDs (Atom API + /bibtex)
    crossref.py      # MetadataFetcher for DOIs (api.crossref.org + doi.org)
    pdf.py           # TextExtractors: PyMuPDF for PDFs, passthrough for text
    archive.py       # original-file archive: visible files/ dir next to the
                     #   DB, one backup unit with it (ADR-13); shared by CLI+GUI
  cli/               # typer app: ingest, source, idea, match, contact,
                     #   summarise, bib, related-work, audit, graph, search,
                     #   chat (E13-2 REPL), mcp (E13-3 stdio server), ui,
                     #   watch (E9-3, ADR-23: continuous folder auto-ingest),
                     #   benchmark (E10-2), ingest references (E9-4, ADR-24:
                     #   bulk BibTeX/RIS reference-manager import)
  web/               # GUI adapter: FastAPI JSON API (api.py) + self-contained
                     #   single-page frontend (static/index.html); a second
                     #   driving adapter beside the CLI — no logic of its own.
                     #   Loopback-only, no auth (single-user), with a
                     #   cross-origin-write guard (ADR-25) — see §9
  mcp/               # MCP server adapter (E13-3, ADR-19): read-only
                     #   search_library/get_source/get_idea/list_citations
                     #   for external MCP clients; no LLM call, no core
                     #   changes — a third driving adapter beside CLI/GUI.
                     #   Also every source/idea as a listable MCP resource
                     #   (E13-4, ADR-20)
  graph/             # self-contained HTML export (vendored Cytoscape.js)
  benchmark/         # provider benchmarking harness (E10-2, ADR-22):
                     #   harness.py::run_benchmark drives SummariseService/
                     #   RationaleService against fixed fixtures for any
                     #   LLMProvider; `mustrum benchmark` CLI command
tests/
  unit/  integration/
docs/
scripts/           # setup.sh (macOS/Linux) + setup.ps1 (Windows), E14-1:
                   #   install uv/Python/Ollama if missing, then `uv sync`
```

Implementation notes (v1 schema, `adapters/sqlite/`):
- datetimes ISO-8601 strings; list/dict fields JSON; vectors float64 blobs.
- `source_texts` immutability (ADR-7) is enforced *in the database* by
  BEFORE UPDATE/DELETE triggers that RAISE(ABORT), not just by convention.
  One sanctioned exception (ADR-9): an abstract may be upgraded to the full
  text via `replace_source_text`, which swaps the row inside a drop/recreate
  of the triggers; the ingest service invalidates the summary and re-embeds
  in the same operation.
- One FTS5 table `search_index(entity, ref_id, body)` covers sources
  (title+authors+notes+text+summary), ideas (title+versions), contacts;
  re-indexed per entity on write. User queries are token-quoted so FTS5
  syntax can't be injected.
- Migrations: append-only list in `schema.py`, tracked via PRAGMA
  user_version. v2 (E1-11) adds `sources.file_path` — the archived original's
  file name relative to the `files/` directory next to the DB.

## 3. Domain model

| Entity | Key fields | Notes |
|---|---|---|
| `Source` | id, kind (paper/article/note), title, authors, year, doi, arxiv_id, provenance per field, file_path (archived original, ADR-13) | metadata record |
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

- `LLMProvider.generate(prompt, system=, json_schema=) -> str` — text
  generation; an optional JSON schema requests structured output (ADR-14):
  the provider constrains decoding so the reply is syntactically valid by
  construction (Ollama: `format`; Anthropic: `output_config.format`).
  Syntax only — content still goes through the verifiers. Implementations:
  `OllamaLLM` (phase 1; raises loudly on `done_reason=length` truncation),
  `AnthropicLLM` (E10-1, ADR-21; raises loudly on `stop_reason=max_tokens`/
  `refusal`), `FakeLLMProvider` (tests). `Config.llm_provider` picks between
  the two live implementations (`cli/main.py::_build_llm`) — the interface
  is deliberately minimal so swapping providers is config-only, no core
  changes.
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
   and the stored `SourceText`; verifies each quote occurs verbatim in the
   text, compared under whitespace + typographic normalisation (Unicode NFKC
   and quote/dash folding, ADR-10); wording, case, and digits stay strict,
   except that the first character's case is folded (sentence-start
   recapitalisation is quoting convention, ADR-15).
   Zero usable quotes is itself a failure (`empty_evidence`) — claims without
   evidence are rejected. Failure ⇒ the artefact is rejected and reported;
   nothing partial is stored.
2. **CitationVerifier** — takes generated text (related-work skeleton) and the
   set of valid citation keys from the DB; any `\cite{key}` / `[@key]` not in
   the set ⇒ hard failure. Also used by the `audit` command on external
   drafts. Recognises LaTeX/natbib/biblatex commands (anything containing
   "cite", incl. starred forms and optional args) and pandoc-Markdown
   (`[@key]`, bare `@key`; e-mail addresses excluded). `extract_keys` returns
   keys deduplicated in order of first appearance.

Model output is treated as untrusted input everywhere. These modules have the
strictest test bar in the project (see §7).

**Multi-source grounding (E13-1, `core/services/grounded.py::run_grounded_multi`):**
answers spanning several candidate sources need a claim to be attributed to
the *specific* source it was quoted from, not just found somewhere in a
concatenated blob (which would let a quote from one paper get miscredited
to another). Each evidence item carries a `source_id`; verification groups
evidence by that id and runs the unchanged `GroundingVerifier` once per
group against that source's own stored text — `verify.py` itself needed no
changes. A `found: bool` field lets the model honestly say "nothing in your
library addresses this" without needing evidence (an empty-evidence claim is
still always a failure per the rule above) — but `found` is a trusted
*classification* signal only; the model's own prose is discarded whenever
`found=false` and replaced with a fixed message, so a "not found" verdict
can never carry unverified freeform text into the answer. A false negative
here is a recall problem, not a grounding violation.

## 6. Key flows

- **Ingest PDF:** extract text → dedup check (DOI/arXiv/title-hash) → store
  Source + immutable SourceText → chunk + embed → (optional) grounded
  summarisation → verify → store Summary. The original file is then archived
  by the driving adapter (CLI/GUI) into `files/` next to the DB and its name
  recorded on the source (`mustrum source open` / GUI "Open PDF" serve it);
  re-ingesting a known paper backfills a missing archive entry, never
  replaces one. Deleting a source removes its archived file with the cascade.
- **Ingest arXiv/DOI:** fetch metadata + BibTeX → same pipeline; fetched
  fields marked authoritative; a downloaded PDF is archived the same way.
- **Ingest reference-manager export (E9-4, `mustrum ingest references`):**
  `core/refimport.py` parses a `.bib` or `.ris` file (Zotero, Mendeley, or
  any tool emitting these standard formats — one parser per format covers
  both, no tool-specific integration) into `ParsedReference` records, each
  fed through `IngestService.ingest_reference` — same dedup (DOI/arXiv/
  title-hash) and abstract-as-text handling as every other ingest path, but
  fields marked `FieldOrigin.EXTRACTED` (parsed out of the imported file,
  not fetched from an authoritative service). A `.bib` entry's byte-exact
  text becomes its `BibEntry` (`origin=fetched`); RIS carries no BibTeX
  form, so one is rendered from the parsed fields (`origin=derived`), the
  same fallback `related-work`'s bib export uses for any source with none
  fetched. A malformed entry (no title) is skipped with a warning rather
  than aborting the whole file. Validated against real Zotero and
  Mendeley exports of both formats (ADR-24): handles Zotero's BibLaTeX
  `date`-instead-of-`year` field and RIS abstracts wrapped across
  untagged continuation lines, and Mendeley's occasional empty BibTeX
  citation key (falls back to a rendered key, same as RIS).
- **New idea / new version:** store version → embed → match against all source
  embeddings → present ranked suggestions → user confirms/rejects. On demand
  (`match explain`), an LLM rationale grounded in verified quotes is attached
  via the shared grounded-generation loop (same rejection rules as summaries).
- **Related-work skeleton:** collect confirmed matches for the idea → for each
  source pull citation key + verified summary → render Markdown/LaTeX +
  matching `.bib`. Assembly is fully deterministic in v1 (no LLM in this
  path); the output still passes CitationVerifier as defence in depth.
  LLM-assisted grouping/prose is a future enhancement and must keep the
  may-not-add-sources rule.
- **Library query / "chat with your knowledge" (E13-1, `QueryService.ask`):**
  retrieve candidate sources by FTS5 keyword search unioned with embedding
  cosine-similarity (reusing `MatchService`'s `cosine` helper), capped at
  `top_k`; sources with no stored text are skipped; zero candidates short-
  circuits to a fixed "nothing found" answer with no LLM call at all. Given
  candidates, one LLM call answers over all of them via
  `run_grounded_multi`; failure to ground raises `QueryFailure`, mirroring
  `GroundingFailure`/`RationaleFailure`. E13-3 (MCP adapter) remains a
  driving adapter planned on top of this core service.
- **Chat (E13-2, `ChatSession` — `mustrum chat` REPL + GUI Chat tab,
  ADR-18):** a thin, purely in-memory stateful wrapper around
  `QueryService.ask`. Every turn is graded identically to a bare `ask()`
  call — quotes verified against real stored text, nothing new in
  `run_grounded_multi`/`verify.py` — but two additive, session-aware inputs
  make it conversational: `history` (the last `history_turns` turns'
  question/answer text, rendered into the prompt as clearly-labelled
  context, truncated per message) lets the model resolve references like
  "it"/"that paper"; `extra_candidate_ids` seeds *only the immediately
  previous turn's* actually-cited source ids into retrieval, so a follow-up
  about the same paper still finds it even when the follow-up's own
  wording wouldn't. Neither input ever reaches the grounding verification
  step. A turn that fails grounding (`QueryFailure`) is not added to
  history. Nothing is persisted — a session lives and dies with the CLI
  process or the running `mustrum ui` server.
- **MCP server (E13-3, `mustrum mcp`, ADR-19):** a third driving adapter,
  read-only, no LLM call. `mustrum/mcp/server.py::create_mcp_server(repo)`
  registers four tools on the `mcp` SDK's `FastMCP` app — `search_library`
  (wraps `StorageRepo.search`), `get_source`/`get_idea` (direct record
  reads, same shape as the GUI's JSON endpoints), `list_citations` (reuses
  `RelatedWorkService.export_bib`) — each a thin wrapper around a plain,
  independently unit-tested function. Runs over stdio, one server process
  per external client connection (e.g. Claude Desktop spawns `mustrum mcp`
  as a subprocess). Every returned field is a direct readout of a stored
  record; nothing is synthesised, so there is nothing to hallucinate.
  **MCP resources (E13-4, ADR-20):** the same `create_mcp_server` also
  registers one `mustrum://sources/{id}` / `mustrum://ideas/{id}` resource
  per row in the repo *at construction time*, so a client can list and
  read them directly (e.g. a resource picker), not only via `get_source`/
  `get_idea`. The list of ids is a startup snapshot (restart to see newly
  ingested sources — same pattern as ADR-16's settings apply model); each
  resource's content is still read fresh from the repo every time.
- **Graph export:** query entities/links → JSON → inline into HTML template
  with embedded Cytoscape.js → single file, no network.
- **Brainstorm (E9-2, the only creative path):** library context → LLM
  proposals → labelled machine-generated output. Produces no citations; the
  only library references ("inspired by") are titles resolved against real
  records, unresolvable mentions dropped. Nothing stored unless the user
  saves, and saved ideas carry the permanent 'brainstorm' tag. The GUI
  generates and saves in two separate calls (E11-7: `POST /api/brainstorm`
  then `POST /api/brainstorm/save`), so the user reviews the whole list
  before deciding which proposals to keep, rather than committing to save
  before seeing them.
- **GUI tags/contact-links/audit (E11-2):** three GUI-only additions, all
  thin adapters over existing core surface — no new core code. Tag editing
  adds `POST`/`DELETE /api/ideas/{id}/tags[/{tag}]` (ideas previously had no
  GUI tagging at all) alongside the existing source-tag endpoints (E11-6),
  both backed by `StorageRepo.tag`/`untag`. Contact links add
  `POST`/`GET /api/sources/{id}/contacts` and the idea equivalent — the GUI
  counterpart of `mustrum contact link`/FR-7.2 — validating the contact and
  idea/source exist (404) before `StorageRepo.add_contact_link`; there is no
  unlink endpoint since the schema has no per-link id to target (matches the
  CLI). Audit upload adds `POST /api/audit` (multipart file), the GUI
  counterpart of `mustrum audit`/FR-5.5, running the uploaded draft's text
  through the existing `AuditService.audit_text`.
- **Backup (NFR-5):** `export` walks the repo into a plain-file bundle
  (canonical JSON + verbatim texts + byte-exact .bib + generated Markdown
  views); `restore` rebuilds an empty DB from it, remapping ids and
  recomputing embeddings. Invariant: export → restore → export is
  byte-identical. Archived originals are binary and stay out of the text
  bundle; `file_path` round-trips so a restored DB finds a copied `files/`
  directory again.
- **Provider benchmark (E10-2, ADR-22):** `mustrum benchmark` runs a fixed
  set of synthetic paper/idea fixtures through `SummariseService`/
  `RationaleService` for each named `LLMProvider` (unmodified — no core
  changes), on a throwaway in-memory `SqliteRepo` with `FakeEmbeddingProvider`
  (embedding quality isn't measured here). Reports a grounding-verification
  pass rate per provider; a provider with no usable credentials is
  `unavailable`, never scored 0% — those mean different things and
  conflating them would defeat the point of comparing providers.
- **Watch-folder auto-ingest (E9-3, ADR-23):** `mustrum watch <dir>` polls
  the folder every `--interval` seconds (default 30) via
  `cli/main.py::_scan_once`, a pure function (no sleeping) so it's directly
  unit-testable. A PDF is ingested only once its `(size, mtime)` is
  unchanged across two consecutive polls — this is how a file still being
  downloaded or synced is told apart from a settled one, without any new
  dependency (no filesystem-event library). `_ingest_pdf` — the same
  dedup-then-archive pipeline `ingest folder` uses — decides the outcome
  (`ingested` / `duplicate_conflict` / `duplicate_skipped` / `failed`);
  resolved files move into an `ingested/` (or `failed/`) subfolder via
  `_move_unique` (never silently overwrites a same-named file already
  there), so re-scans stay bounded as the watched folder accumulates
  papers over weeks/months, and nothing already resolved is ever retried.
  Runs until Ctrl+C, matching `mustrum ui`/`mustrum mcp`'s foreground,
  blocking pattern rather than introducing a new daemon/service concept.

## 7. Testing strategy

- Unit tests against fake providers; adapter integration tests behind markers
  (`-m ollama`, `-m network`) so the default suite is fully offline and
  deterministic.
- **Mutation testing with mutmut** on `mustrum/core/`: overall mutation score
  target ≥ 80%; for `core/verify.py` every surviving mutant must be reviewed
  and either killed or explicitly justified in the PR/commit. `core/ports.py`
  is excluded (Protocol stubs have no behaviour to mutate).
- Justified surviving mutants (keep this list current). As of 2026-07-13
  (full fresh `mustrum/core/` run, not an incremental per-module one — see
  note below) the score is 2084/2262 killed (92.1%), `core/verify.py` still
  at 100% (0 survivors). The mutant count roughly doubled since the
  2026-07-11 baseline (972 total) mainly because E3-5 added two JSON-schema
  dict literals (`grounded.py`, `brainstorm.py`) — schema literals generate
  many mutants per line without changing the overall score. The survivors
  fall into three reviewed classes, accepted as either equivalent or not
  worth a test:
  1. **Human-readable message text** — mutants that only alter error-message
     or rendered-banner wording/case (`"XX…XX"`, upper/lower variants,
     `ValueError(None)`); behaviour and data are unchanged.
  2. **Default-constant tweaks** — changed default parameter values
     (`limit=20→21`, `max_chars=1500→1501`, `attempts=2→3`); behavioural
     defaults that matter (match threshold 0.35, source truncation 16000)
     ARE test-pinned.
  3. **Semantically equivalent code** — e.g. `"utf-8"`→`"UTF-8"` (codec names
     case-insensitive), `float("-inf")`→`float("-INF")`, `>`→`>=` on a
     running-max update, `zip(strict=True)`→`strict=False` behind a length
     pre-check, unreachable guard permutations that fail through the same
     `except` path, an exported-but-never-read-back `"id"` field in the
     backup format (`services/backup.py::_export_contacts` — contacts get
     fresh auto-IDs on restore, so the field is informational only).
  Anything outside these classes must be killed before merging.
  **Known gap (2026-07-13, not yet in a justified class):**
  `services/relatedwork.py::_entry_lines__mutmut_2` survives — mutating
  the `authors` fallback ternary to an unconditional `authors = None`
  isn't caught, because no test asserts on the rendered author byline text
  in the related-work skeleton output (only headings/citation-keys/TODO
  markers are checked). Low severity — the skeleton is deterministic
  assembly, not LLM output, so this can't cause an invented citation, only
  a cosmetic "None." in a draft the user is expected to hand-edit anyway —
  but it should get a test and be moved into the reviewed classes above
  rather than sit here indefinitely.
  Normal story work only reruns mutmut on the specific module it touches
  (see individual PRs), not the whole `mustrum/core/` tree — the aggregate
  score above can go stale after any core/ change and doesn't self-correct;
  don't trust it without a fresh run if it matters for a decision.
  **E13-1 (2026-07-13), scoped run on the two new/touched files only**
  (`services/grounded.py`, `services/query.py`): `run_grounded_multi`
  153/188 killed (81.4%), `query.py` 133/140 killed (95.0%) — both clear the
  bar. Surviving mutants are entirely the accepted classes above (message
  text and default-constant tweaks — the JSON-schema dict literal for
  `run_grounded_multi` alone accounts for most of its survivors, same
  pattern as the E3-5 note above) plus one low-severity known gap:
  `QueryService._candidate_source_ids`'s per-source running-max lookup uses
  `best_per_source.get(emb.ref_id, ...)` as the comparison key; a mutant
  swapping that to `.get(None, ...)` only misranks which of a *single*
  source's own chunks is kept as its best score (never misattributes a
  score to the wrong source, never affects grounding) — untested because
  reproducing it needs hand-crafted embedding vectors across multiple
  chunks of one source, out of proportion to the severity. `run_grounded`
  and its siblings (`parse_json_object`, `describe_failure`,
  `GroundedOutputError.__init__`) were not touched by this story; their
  survivors are pre-existing and unreviewed here.
  **E13-2 (2026-07-13), scoped run on `services/query.py` (extended) +
  `services/chat.py` (new):** `query.py` 152/169 killed (89.9%), `chat.py`
  23/24 killed (95.8%) — both well above the bar. Survivors are entirely
  the same accepted classes (message text, default-constant tweaks) plus
  the identical pre-existing `.get(None, ...)`/sort-key-ordering gaps
  already documented above — no new low-severity gaps introduced.
- mypy strict on `mustrum/core/`; ruff for lint + format.

## 8. Decisions

Architecture decisions and their rationale are recorded in
[DECISIONS.md](DECISIONS.md). Current: ADR-1 Python, ADR-2 SQLite(+FTS5),
ADR-3 CLI + self-contained HTML graph, ADR-4 Ollama for both LLM and
embeddings in phase 1, ADR-5 hexagonal provider interface, ADR-6 mutmut,
ADR-7 immutable source texts + grounded generation, ADR-8 model defaults,
ADR-9 abstract→full-text upgrade, ADR-10 quote normalisation, ADR-11
deletion as a user right, ADR-12 citation-key collision suffixes, ADR-13
original-file archive next to the DB, ADR-14 structured LLM outputs, ADR-15
first-character quote case fold, ADR-16 library settings file next to the DB,
ADR-17 multi-source grounding with a trusted `found` flag, ADR-18 chat
history as interpretive context, never evidence, ADR-19 MCP exposes raw
library data, not a grounded-answer tool, ADR-20 MCP resources are an
eager per-item startup snapshot, not a dynamic listing, ADR-21
config-switchable AnthropicProvider, ADR-22 provider benchmarking harness,
ADR-23 watch-folder auto-ingest, ADR-24 combined BibTeX/RIS reference-manager
import, ADR-25 GUI security hardening (output escaping, cross-origin-write
guard, friendly errors on bad input).

## 9. Security & privacy posture

Mustrum is local-first and single-user; the security model follows from that,
and the hardening below is recorded in ADR-25.

- **Nothing leaves the machine unless you ask.** The default stack is fully
  local (Ollama). The only outbound calls are opt-in: metadata/PDF fetch for
  `ingest arxiv`/`ingest doi` (arXiv, Crossref, doi.org, and — only when you
  set `unpaywall_email` — Unpaywall, which receives that address per their
  fair-use policy), and the Anthropic API *only* if you switch
  `llm_provider` to `anthropic` (then prompts, i.e. source text and ideas, go
  to Anthropic). No telemetry. The API key is never persisted — it is read
  from `ANTHROPIC_API_KEY` at run time (ADR-4, privacy rule 9).
- **The GUI binds loopback only** (`uvicorn host="127.0.0.1"`), disables the
  interactive API docs, and serves a fully self-contained page (no CDNs).
- **No auth, by design** — it is a single-user local app. Because any web
  page the user has open could otherwise reach the unauthenticated API, a
  middleware refuses any state-changing request (POST/PUT/PATCH/DELETE) whose
  `Origin` header is present and not loopback (ADR-25); requests with no
  Origin — curl, scripts, the test client — are not the cross-site threat and
  pass through.
- **Untrusted data is treated as untrusted.** Source/idea/contact fields come
  from PDF metadata, imported `.bib`/`.ris`, Crossref/arXiv, and LLM output,
  so every value rendered into HTML is escaped before it reaches `innerHTML`
  — in the SPA and in the generated graph page alike (ADR-25). FTS5 queries
  are token-quoted (no query-syntax injection); all SQL is parameterised.
  Archived-file lookups reject any stored path that escapes the `files/`
  directory (defence against a tampered DB value).
- **Robustness as a safety property.** Bad input a first-run user will
  realistically hit — a corrupt/encrypted PDF, a non-UTF-8 text file, or being
  offline during a fetch — fails with a clean message (CLI) or a 4xx/502
  (GUI), never a raw traceback or opaque 500 (ADR-25).
- **Known lower-severity items** left as follow-ups: PDF downloads follow
  redirects to publisher-supplied URLs (SSRF surface, negligible on a
  personal machine); the LaTeX skeleton export does not escape LaTeX
  metacharacters in titles/summaries.
