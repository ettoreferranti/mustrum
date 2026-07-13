# Mustrum — Backlog

Priorities: **M** must / **S** should / **C** could (MoSCoW). Status: `todo`,
`in-progress`, `done`. Keep this file updated as stories complete; add new
stories at the bottom of the relevant epic.

## Phase 0 — Foundations

| ID | Story | Prio | Status |
|---|---|---|---|
| E0-1 | Project scaffolding: package layout, pyproject (uv), pytest, ruff, mypy strict on core, mutmut config, CI-ready offline test suite | M | done |
| E0-2 | Domain models (`core/models.py`) + port Protocols (`core/ports.py`) | M | done |
| E0-3 | SQLite adapter: schema v1, migrations, FTS5, StorageRepo | M | done |
| E0-4 | Fake LLM/embedding providers for deterministic tests | M | done |

## Phase 1 — MVP (ingest → match → cite)

### E1 Ingestion
| ID | Story | Prio | Status |
|---|---|---|---|
| E1-1 | PDF ingestion via PyMuPDF; immutable SourceText | M | done |
| E1-2 | arXiv ingestion: fetch metadata + BibTeX by ID | M | done |
| E1-3 | DOI ingestion via Crossref | M | done |
| E1-4 | Plain text / Markdown ingestion | M | done |
| E1-5 | Deduplication (DOI / arXiv ID / title-hash) with merge-or-skip prompt | M | done |
| E1-6 | Provenance tracking per metadata field | S | done |
| E1-9 | `source attach`: attach a manually-downloaded PDF to an existing source; abstract→full-text upgrade invalidates summary + re-embeds (ADR-9) | S | done |
| E1-10 | Docs guard: every CLI command must appear in README (test-enforced) | S | done |
| E1-8 | Automatic full-text PDF fetch on ingest: arXiv PDFs always, DOIs via Unpaywall open-access lookup; abstract fallback | S | done |
| E1-7 | Batch folder ingest (`ingest folder DIR [-r]`): all PDFs, per-file errors don't abort, idempotent re-runs | S | done |
| E1-11 | Original-file archive: copy every ingested/fetched original (PDF/text) into a visible `files/` directory beside the DB so DB + originals back up as one unit; `Source.file_path` (schema migration); persist arXiv/Unpaywall downloads; `source attach` archives too; `source open ID` CLI + GUI "Open PDF" via API file endpoint (ADR-13, [PR #1](https://github.com/ettoreferranti/mustrum/pull/1)) | S | done |

### E2 Ideas
| ID | Story | Prio | Status |
|---|---|---|---|
| E2-1 | Create/list/show ideas; append-only versioning | M | done |
| E2-2 | Idea↔idea links (builds-on / contrasts / related) | S | done |
| E2-3 | Bulk idea import from Markdown file (`# title` + body per idea; idempotent, --on-existing skip/revise/create) | S | done |

### E3 Model layer (Ollama)
| ID | Story | Prio | Status |
|---|---|---|---|
| E3-1 | Ollama adapter: LLMProvider + EmbeddingProvider, model names in config (defaults per ADR-8: nomic-embed-text / qwen3:30b) | M | done |
| E3-2 | GroundingVerifier + CitationVerifier (`core/verify.py`) — 100% mutant review | M | done |
| E3-3 | Grounded summarisation pipeline (generate → verify → store or reject) | M | done |
| E3-4 | Batch summarisation (`summarise --all`): only sources lacking a summary, grounding failures skipped and reported | S | done |
| E3-5 | Structured outputs (ADR-14): `LLMProvider.generate` takes an optional JSON schema, Ollama forwards it as `format` so replies parse by construction (grounded loop + brainstorm); `done_reason=length` raises "output truncated — raise num_ctx" — found when qwen3's untagged reasoning prose caused "no parsable output" ([PR #6](https://github.com/ettoreferranti/mustrum/pull/6)) | S | done |
| E3-6 | Quote verification folds case at the FIRST character only (ADR-15, amends ADR-10): sentence-start recapitalisation is quoting convention, not a wording change; everything else stays strict; verify.py kept at 100% mutation score — found when qwen3 quoted "So far, we have…" as "We have…" ([PR #7](https://github.com/ettoreferranti/mustrum/pull/7)) | S | done |

### E4 Matching
| ID | Story | Prio | Status |
|---|---|---|---|
| E4-1 | Chunked source embeddings + idea embeddings; cosine ranking, threshold | M | done |
| E4-2 | Suggest/confirm/reject workflow; manual match creation | M | done |
| E4-3 | Grounded match rationale (quotes, verified): `match explain`, `match suggest --explain` | S | done |
| E4-4 | Gap analysis report (unsupported ideas, orphan sources) | S | done |

### E5 Bibliography & writing support
| ID | Story | Prio | Status |
|---|---|---|---|
| E5-1 | BibEntry storage: fetched raw `.bib` byte-exact; derived entries for manual sources; unique citation keys | M | done |
| E5-2 | `.bib` export (library / per idea) | M | done |
| E5-3 | Related-work skeleton (Markdown + LaTeX) from confirmed matches, citation-verified | M | done |
| E5-4 | Citation audit of external drafts (`.tex`/`.md`) against the library | S | done |

### E6 Graph
| ID | Story | Prio | Status |
|---|---|---|---|
| E6-1 | Self-contained HTML graph export (Cytoscape.js embedded): idea hubs, source stars, idea links | M | done |
| E6-2 | Filters (tag/idea/type) + node detail panel | S | partial (panel done, filters todo) |
| E6-3 | Contacts as graph nodes | C | done |

### E7 Contacts
| ID | Story | Prio | Status |
|---|---|---|---|
| E7-1 | Contact CRUD (person/company/institution/university) | M | done |
| E7-2 | Contact↔idea / contact↔source links with why-relevant note | S | done |

### E8 Search & organisation
| ID | Story | Prio | Status |
|---|---|---|---|
| E8-1 | Full-text search (FTS5) across sources/ideas/summaries/contacts | M | done |
| E8-2 | Tags + reading status + per-source notes | S | done |
| E8-5 | `source enrich`: complete bare PDF sources via exact-title Crossref lookup; citation-key collision dedup (ADR-12) | S | done |
| E8-4 | Proper source titles: PDF-metadata titles at ingest (HTML entities decoded, junk heuristics), `source rename` + GUI rename, list overflow/meta-line polish | S | done |
| E8-3 | Delete sources/ideas with full cascade (CLI `--yes` guard, GUI confirm, API DELETE) — found via GUI testing | S | done |
| E8-6 | Manual metadata for DOI-less venues (e.g. CEUR-WS, not in Crossref): `source edit ID --author --year` with USER provenance + GUI authors/year button (`POST /api/sources/{id}/metadata`); enrich failure message points at it — found when a PoEM-C 2025 CEUR paper could never match ([PR #4](https://github.com/ettoreferranti/mustrum/pull/4)) | S | done |

## Phase 2 — Longevity & polish

| ID | Story | Prio | Status |
|---|---|---|---|
| E9-1 | Plain-file export/import (JSON + Markdown + .bib) for git-versionable backup: `export` / `restore` | S | done |
| E9-2 | Idea brainstorming mode (explicitly-invoked creative output, clearly labelled, never mixed with citation output): `brainstorm` | S | done |
| E9-3 | Watch-folder auto-ingest for PDFs | C | todo |
| E9-4 | Zotero import | C | todo |
| E9-5 | Contact import (vCard/CSV) — OQ-2 | C | todo |

## E12 Configuration

| ID | Story | Prio | Status |
|---|---|---|---|
| E12-1 | Library-local settings file next to the DB (ADR-16): `config.toml` beside `mustrum.db` holds ollama_url/llm_model/embed_model/max_source_chars/num_ctx/unpaywall_email, so a synced/backed-up library carries its own settings; `mustrum config` split into `show`/`init`/`set` subcommands; GUI Settings tab (`GET`/`POST /api/settings`); save-then-restart-notice apply model — requested to make the whole library portable ([PR #8](https://github.com/ettoreferranti/mustrum/pull/8)) | S | done |
| E12-2 | Model dropdowns fetched from Ollama: `adapters/ollama.list_models` (`/api/tags`), `GET /api/ollama/models` (never raises — returns `{models: [], error}` so the Settings form stays usable when Ollama is down), GUI LLM/embedding `<select>` dropdowns (current value always kept selectable even if not in the fetched list) with a Refresh button, `mustrum config models` CLI bonus — requested to avoid model-name typos and "which models do I even have" ([PR #9](https://github.com/ettoreferranti/mustrum/pull/9)) | S | done |

## GUI

| ID | Story | Prio | Status |
|---|---|---|---|
| E11-1 | Local web GUI (`mustrum ui`): FastAPI JSON adapter + self-contained single-page frontend covering ingest, sources, summaries, matching, related-work, graph, brainstorm, contacts | S | done |
| E11-2 | GUI: tags editing, contact links, audit upload | C | todo |
| E11-3 | GUI "Add PDF": attach a manually-downloaded PDF/text to an existing source from the source panel (`POST /api/sources/{id}/attach`); archives the original (E1-11), invalidates an upgraded abstract's summary — split out of E11-2 after live DOI-ingest testing ([PR #2](https://github.com/ettoreferranti/mustrum/pull/2)) | S | done |
| E11-4 | Ingest feedback: GUI flash surfaces the PDF-fetch notes (failures were CLI-only); "no downloadable PDF" note distinguishes stored-abstract from metadata-only and points at attach — found when an ACM OA PDF 403'd silently in the GUI ([PR #2](https://github.com/ettoreferranti/mustrum/pull/2)) | S | done |
| E11-5 | Readable errors: GUI error flashes persist until dismissed (click ✕; successes still fade); every failed API call leaves a line in the `mustrum ui` terminal — found when a grounding-failure flash vanished before it could be read ([PR #3](https://github.com/ettoreferranti/mustrum/pull/3)) | S | done |
| E11-6 | Source table: sortable columns (title, main-author et al., year), clickable status chips acting in place (text→attach, summary→summarise, meta→enrich/edit, read cycles, file opens), multi-select with bulk summarise-missing / find-metadata / tag / delete (sequential client-side loop, per-source failures collected); new `POST /api/sources/{id}/tags` ([PR #5](https://github.com/ettoreferranti/mustrum/pull/5)) | S | done |
| E11-7 | Brainstorm: post-hoc idea selection — `POST /api/brainstorm` generates only, per-idea checkbox in the results list (checked by default), "Save selected (N)" saves just those via new `POST /api/brainstorm/save`; old pre-generation "save" checkbox removed — requested because deciding to save before seeing the ideas isn't a natural workflow; CLI `brainstorm --save` untouched (independent code path, batch use case) ([PR #10](https://github.com/ettoreferranti/mustrum/pull/10)) | S | done |

## Phase 3 — Anthropic provider

| ID | Story | Prio | Status |
|---|---|---|---|
| E10-1 | AnthropicProvider implementing LLMProvider (config-switchable, no core changes) | M | todo |
| E10-2 | Provider benchmarking harness: same tasks via fake/Ollama/Anthropic, compare grounding-verification pass rates | S | todo |

## Phase 4 — Chat & MCP

| ID | Story | Prio | Status |
|---|---|---|---|
| E13-1 | Core query service (`core/services/query.py::QueryService`, scoped to sources): retrieval = FTS5 search (E8-1) ∪ embedding cosine-similarity, capped at top_k; answer synthesised across all candidates in one LLM call via a new multi-source grounded primitive (`services/grounded.py::run_grounded_multi`, ADR-17) — each claim's quote verified against the *specific* source it's attributed to, `found:bool` a trusted classifier whose prose is discarded when false; `core/verify.py` unchanged. Idea/contact grounding and any CLI/GUI surface deliberately out of scope (E13-2/E13-3) ([PR #12](https://github.com/ettoreferranti/mustrum/pull/12)) | M | done |
| E13-2 | `mustrum chat` — conversational grounded Q&A (CLI REPL + GUI Chat tab, `core/services/chat.py::ChatSession`): purely in-memory multi-turn session; every turn re-grounds via an unchanged `QueryService.ask()` call, now extended with additive `history`/`extra_candidate_ids` params — prior turns shape the prompt (reference resolution) and retrieval seeding (last turn's cited sources) only, never evidence (ADR-18) ([PR #13](https://github.com/ettoreferranti/mustrum/pull/13)) | M | done |
| E13-3 | MCP server adapter (`mustrum mcp`, stdio, `mustrum/mcp/server.py`): read-only tools `search_library`/`get_source`/`get_idea`/`list_citations` — raw structured reads only, deliberately not a grounded-QA tool (ADR-19, user-confirmed); zero core changes, no LLM call anywhere in this story ([PR #14](https://github.com/ettoreferranti/mustrum/pull/14)) | S | done |
| E13-4 | MCP resource exposure: every source/idea listed as an individually-readable `mustrum://sources\|ideas/{id}` resource (`mustrum/mcp/server.py`), not only reachable via `get_source`/`get_idea` tool calls — ids are a startup snapshot, content still read live (ADR-20) ([PR #15](https://github.com/ettoreferranti/mustrum/pull/15)) | C | done |
