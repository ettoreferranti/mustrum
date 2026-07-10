# Mustrum — Backlog

Priorities: **M** must / **S** should / **C** could (MoSCoW). Status: `todo`,
`in-progress`, `done`. Keep this file updated as stories complete; add new
stories at the bottom of the relevant epic.

## Phase 0 — Foundations

| ID | Story | Prio | Status |
|---|---|---|---|
| E0-1 | Project scaffolding: package layout, pyproject (uv), pytest, ruff, mypy strict on core, mutmut config, CI-ready offline test suite | M | todo |
| E0-2 | Domain models (`core/models.py`) + port Protocols (`core/ports.py`) | M | todo |
| E0-3 | SQLite adapter: schema v1, migrations, FTS5, StorageRepo | M | todo |
| E0-4 | Fake LLM/embedding providers for deterministic tests | M | todo |

## Phase 1 — MVP (ingest → match → cite)

### E1 Ingestion
| ID | Story | Prio | Status |
|---|---|---|---|
| E1-1 | PDF ingestion via PyMuPDF; immutable SourceText | M | todo |
| E1-2 | arXiv ingestion: fetch metadata + BibTeX by ID | M | todo |
| E1-3 | DOI ingestion via Crossref | M | todo |
| E1-4 | Plain text / Markdown ingestion | M | todo |
| E1-5 | Deduplication (DOI / arXiv ID / title-hash) with merge-or-skip prompt | M | todo |
| E1-6 | Provenance tracking per metadata field | S | todo |

### E2 Ideas
| ID | Story | Prio | Status |
|---|---|---|---|
| E2-1 | Create/list/show ideas; append-only versioning | M | todo |
| E2-2 | Idea↔idea links (builds-on / contrasts / related) | S | todo |

### E3 Model layer (Ollama)
| ID | Story | Prio | Status |
|---|---|---|---|
| E3-1 | Ollama adapter: LLMProvider + EmbeddingProvider, model names in config; resolve OQ-1 | M | todo |
| E3-2 | GroundingVerifier + CitationVerifier (`core/verify.py`) — 100% mutant review | M | todo |
| E3-3 | Grounded summarisation pipeline (generate → verify → store or reject) | M | todo |

### E4 Matching
| ID | Story | Prio | Status |
|---|---|---|---|
| E4-1 | Chunked source embeddings + idea embeddings; cosine ranking, threshold | M | todo |
| E4-2 | Suggest/confirm/reject workflow; manual match creation | M | todo |
| E4-3 | Grounded match rationale (quotes, verified) | S | todo |
| E4-4 | Gap analysis report (unsupported ideas, orphan sources) | S | todo |

### E5 Bibliography & writing support
| ID | Story | Prio | Status |
|---|---|---|---|
| E5-1 | BibEntry storage: fetched raw `.bib` byte-exact; derived entries for manual sources; unique citation keys | M | todo |
| E5-2 | `.bib` export (library / per idea) | M | todo |
| E5-3 | Related-work skeleton (Markdown + LaTeX) from confirmed matches, citation-verified | M | todo |
| E5-4 | Citation audit of external drafts (`.tex`/`.md`) against the library | S | todo |

### E6 Graph
| ID | Story | Prio | Status |
|---|---|---|---|
| E6-1 | Self-contained HTML graph export (Cytoscape.js embedded): idea hubs, source stars, idea links | M | todo |
| E6-2 | Filters (tag/idea/type) + node detail panel | S | todo |
| E6-3 | Contacts as graph nodes | C | todo |

### E7 Contacts
| ID | Story | Prio | Status |
|---|---|---|---|
| E7-1 | Contact CRUD (person/company/institution/university) | M | todo |
| E7-2 | Contact↔idea / contact↔source links with why-relevant note | S | todo |

### E8 Search & organisation
| ID | Story | Prio | Status |
|---|---|---|---|
| E8-1 | Full-text search (FTS5) across sources/ideas/summaries/contacts | M | todo |
| E8-2 | Tags + reading status + per-source notes | S | todo |

## Phase 2 — Longevity & polish

| ID | Story | Prio | Status |
|---|---|---|---|
| E9-1 | Plain-file export/import (JSON + Markdown + .bib) for git-versionable backup | S | todo |
| E9-2 | Idea brainstorming mode (explicitly-invoked creative output, clearly labelled, never mixed with citation output) | S | todo |
| E9-3 | Watch-folder auto-ingest for PDFs | C | todo |
| E9-4 | Zotero import | C | todo |
| E9-5 | Contact import (vCard/CSV) — OQ-2 | C | todo |

## Phase 3 — Anthropic provider

| ID | Story | Prio | Status |
|---|---|---|---|
| E10-1 | AnthropicProvider implementing LLMProvider (config-switchable, no core changes) | M | todo |
| E10-2 | Provider benchmarking harness: same tasks via fake/Ollama/Anthropic, compare grounding-verification pass rates | S | todo |
