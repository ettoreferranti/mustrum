# Mustrum — Decision log (ADRs)

Short-form architecture decision records. Add new entries at the bottom;
never rewrite history — supersede with a new ADR instead.

## ADR-1 — Python (2026-07-10, accepted)
Chosen over TypeScript/Rust for: mature mutation testing (mutmut), stdlib
SQLite, the local-model ecosystem, and fast iteration. User-confirmed.

## ADR-2 — SQLite with FTS5 (2026-07-10, accepted)
Single-file local DB fits the local-first requirement, needs no server, and
FTS5 gives full-text search for free. Embeddings stored as blobs; at ~5k
sources brute-force cosine similarity is fine (no vector DB needed).

## ADR-3 — CLI + self-contained HTML graph (2026-07-10, accepted)
typer CLI drives everything; the graph is a generated single HTML file with
Cytoscape.js embedded (viewable offline, no server). User-confirmed. A web UI
would be a new adapter, not a rewrite.

## ADR-4 — Ollama for LLM *and* embeddings in phase 1 (2026-07-10, accepted)
User chose a local LLM from day one. Using Ollama for embeddings too
(`nomic-embed-text`) keeps a single runtime dependency instead of adding
sentence-transformers/torch. Model names live in config (OQ-1 finalises
defaults).

## ADR-5 — Hexagonal architecture with Protocol-based ports (2026-07-10, accepted)
Core never imports adapters; `LLMProvider`/`EmbeddingProvider` are minimal
Protocols so the phase-3 Anthropic adapter is config-only. Also enables a
fully offline deterministic test suite via fake providers.

## ADR-6 — mutmut for mutation testing (2026-07-10, accepted)
Actively maintained, pytest-native, simple config. Scope: `mustrum/core/`
(mutating adapters mostly measures third-party behaviour). Score target ≥80%;
`core/verify.py` requires review of every surviving mutant.

## ADR-7 — Immutable source texts + verify-after-generate (2026-07-10, accepted)
The no-invented-citations guarantee is enforced structurally, not by prompt
discipline: source texts are immutable after ingest; model output is untrusted
and must pass GroundingVerifier (evidence quotes verbatim in source) and
CitationVerifier (only DB citation keys) before anything is stored or emitted.
Verification failure rejects the artefact loudly — no partial saves, no silent
repair.
