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

## ADR-8 — Default Ollama models: nomic-embed-text + qwen3:30b (2026-07-10, accepted)
Resolves OQ-1. Embeddings: `nomic-embed-text` (274 MB, strong retrieval
quality, cheap to re-embed the library). Generation: `qwen3:30b` — a
mixture-of-experts model (~3B active parameters), so on the dev machine
(MacBook M5 Pro, 48 GB unified memory) it delivers ~30B-class output quality
at near-8B speed, using ~19 GB. A dense 70B (llama3.3) was rejected: its q4
weights (~40 GB) leave no headroom on 48 GB and inference would be slow.
`llama3.1:8b` remains the documented lightweight fallback for battery/speed —
both names live in config only, so swapping is a one-line change and never
touches code. Grounding verification (ADR-7) makes model choice a prose-quality
concern, not a correctness one.

## ADR-9 — Controlled source-text upgrade (2026-07-11, accepted)
Amends ADR-7. Papers ingested by DOI often start with only their abstract
(publisher PDF not fetchable); when the user later obtains the PDF
(`source attach`), the abstract may be *upgraded* to the full text. Rules:
only `abstract` texts may be replaced, never a stored full text; the swap
happens through a dedicated repo API that drops and atomically recreates the
immutability triggers; and everything derived from the old text is
invalidated in the same operation — the summary is deleted and embeddings are
recomputed — so no stored claim ever remains grounded against text that is
gone. Arbitrary edits to source texts remain impossible.

## ADR-10 — Typographic normalisation in grounding (2026-07-11, accepted)
Refines ADR-7's "verbatim". Publisher PDFs contain typographic glyphs (curly
quotes/apostrophes, en/em dashes, ligatures, non-breaking spaces, soft
hyphens) that LLMs faithfully reproduce as ASCII; strict byte comparison
rejected genuinely identical wording. GroundingVerifier now compares under
Unicode NFKC plus an explicit quote/dash fold (both sides normalised
identically). Wording, casing, digits, and word order remain strict — the
fold cannot mask invented content, only glyph variance introduced by PDF
extraction. Motivating case: a Springer paper whose correct quotes failed on
' vs ' and – vs -.

## ADR-11 — Deletion is a user right, distinct from tampering (2026-07-12, accepted)
ADR-7 guards stored evidence against *alteration*; it does not forbid the
user removing an entire record. `delete_source`/`delete_idea` cascade over
every dependent row (text, summary, BibTeX, matches, tags, contact links,
embeddings, search index) so no grounded claim can dangle against removed
text. The source-text triggers are dropped and recreated around the cascade,
exactly as in ADR-9. Deleting a cited source is allowed — drafts citing its
key will subsequently fail `audit`, which is the correct signal.

## ADR-12 — Citation keys are unique; colliding fetched keys get a suffix (2026-07-12, accepted)
Publishers derive BibTeX keys like Author_Year, which collide across papers
(observed live: two 2025 Mosquera papers both keyed Mosquera_2025). Duplicate
keys are unusable in LaTeX, so on collision the new entry key gains an
a/b/c… suffix, rewritten both in the citation_key column and — the sole
sanctioned amendment to fetched BibTeX — in the key token of the raw entry,
keeping the two byte-identical in what they cite. All other bytes of the
fetched entry remain untouched.

## ADR-13 — Originals archived in a visible files/ directory next to the DB (2026-07-12, accepted)
Extracted text is what the rigour kernel verifies against, but the original
PDF is what a human wants to read. Every ingested or fetched original is
therefore copied into `files/` — a deliberately non-hidden directory that
sits beside the SQLite database, so DB + originals form a single backup/sync
unit. The DB stores only the file name relative to that directory
(`sources.file_path`, schema v2), keeping the pair relocatable as a whole.
Names are `<id>-<title-slug><ext>` (id guarantees uniqueness, slug gives
readability). All archive file I/O lives in `adapters/archive.py` — core
services stay filesystem-free; the CLI and GUI adapters call it after ingest,
attach, and delete. Re-ingesting a known paper backfills a missing archive
entry but never replaces an existing one (attach, which explicitly supplies a
new original, does replace). The plain-file export (E9-1) stays text-only:
originals are not bundled, but `file_path` round-trips so a restored DB finds
a copied `files/` directory again.

## ADR-14 — Structured outputs constrain syntax, never content (2026-07-12, accepted)
qwen3 front-loads untagged reasoning prose despite `think=false`, and a
summarise run failed all three attempts with "no parsable output" (observed
live; Ollama reported `done_reason=stop`, prompt 3.4k tokens of a 16k window
— not truncation). LLMProvider therefore accepts an optional JSON schema and
the Ollama adapter forwards it as `format`, making the reply parse by
construction. This does not weaken the rigour kernel: constrained decoding
shapes *syntax only*; evidence quotes still pass the GroundingVerifier
verbatim, brainstorm's based_on titles are still resolved against real
records, and nothing unverified is stored. Genuine truncation is now loud:
`done_reason=length` raises "output truncated — raise num_ctx" instead of
surfacing as a downstream parse/grounding failure.

## ADR-15 — Quote verification folds case at the first character only (2026-07-12, amends ADR-10)
Live summarising showed qwen3 recapitalising quotes that start mid-sentence
("So far, we have identified 38…" quoted as "We have identified 38…") —
standard quoting convention, rejected by the case-strict verifier as if it
were a wording change. GroundingVerifier now also accepts a quote whose
FIRST character differs from the source only in case (cased letters only;
digits, punctuation, and caseless scripts stay strict). Everything beyond
the first character remains exact — "observe" for "observed" is still
rejected, as is any mid-quote case change. verify.py remains at 100%
mutation score; the variant helper is pinned by direct edge tests.
