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

## ADR-16 — Library settings live next to the database, not just under ~/.config (2026-07-13, accepted)
Requested: an editable settings file that travels with the library, plus a
GUI way to change it. Two files now exist, in precedence order: the global
bootstrap file (`~/.config/mustrum/config.toml`) whose only essential job is
setting `db_path`; and the library file (`<db_path's folder>/config.toml`,
`Config.library_config_path`), which holds everything else — Ollama URL,
model choice, context sizes, the Unpaywall e-mail — and is written by
`mustrum config set` or the GUI Settings panel (`save_library_config`,
`POST /api/settings`). The library file never sets `db_path` itself (that
would be self-referential); env vars (`MUSTRUM_DB`/`MUSTRUM_OLLAMA_URL`)
remain the final, most-explicit override. Backing up or syncing the folder
containing `mustrum.db` now carries data, archived originals (ADR-13), and
settings as one unit. Apply model is save-then-restart-notice, not hot-reload:
a running `mustrum ui` process built its Ollama clients at startup, and
`POST /api/settings` deliberately does not reach into that already-running
process — reconfiguring embed_model mid-session would silently desync
existing embeddings from new ones without a full re-embed, which is exactly
the kind of correctness trap ADR-9's abstract-upgrade handling was designed
to avoid elsewhere; the simpler, safer contract here is "persisted now,
effective on next start". The `config` CLI command became a subgroup
(`show` / `init` / `set`) to make `set` a natural sibling — a documented,
tested break from the single-command form.

## ADR-17 — Multi-source grounding for library Q&A: a trusted `found` flag, not an empty-evidence exception (2026-07-13, accepted)
E13-1 ("chat with your knowledge") needs the model to synthesise one answer
from several candidate sources' excerpts, and to be able to say "nothing in
your library addresses this" — but `GroundingVerifier` already treats zero
evidence quotes as a hard failure whenever a claim is made, and that rule is
load-bearing (NFR-1): weakening it to let a model skip grounding by simply
supplying no quotes would open exactly the hole the rigour kernel exists to
close. Two options were considered: (a) carve an exception into
`GroundingVerifier` for an explicit "not found" case, or (b) keep the
verifier untouched and split the model's output into two channels — a
`found: bool` classification signal, and prose that is only ever trusted
when `found=true` and grounded. Went with (b): `run_grounded_multi`
(`core/services/grounded.py`) treats `found` as a trusted classifier (a
false negative is a recall problem, not a safety violation) but *discards*
the model's own text whenever `found=false`, substituting a fixed message —
so an ungrounded claim can never reach the user regardless of how the model
phrases a "not found" reply. `GroundingVerifier`/`CitationVerifier` needed
no changes and keep their existing mutation-test bar; the new discipline
lives entirely in the calling loop. Evidence for a positive answer is
`{source_id, quote}` pairs, verified per-source-id against that source's
own stored text (not a concatenated blob), so a quote can't be attributed
to the wrong paper. See `core/services/query.py::QueryService` for the
retrieval layer this grounds against (FTS5 ∪ embedding cosine-similarity).

## ADR-18 — Chat history is interpretive context, never evidence (2026-07-13, accepted)
E13-2 makes E13-1's single-turn `QueryService.ask()` conversational
(`mustrum chat` + a GUI Chat tab), which needs follow-ups like "what year
was that published?" to resolve "that" against the previous turn — but
every turn still has to pass the same grounding discipline as a bare
`ask()` call: a claim without a fresh, verified quote is a hard failure
regardless of what was said earlier in the conversation. The fix is
additive, not a new grounding path: `QueryService.ask()` gained two
optional parameters, `history` (prior question/answer-text pairs, rendered
into the prompt under a section explicitly labelled "context only... NOT
evidence", reinforced by one added sentence in the system prompt) and
`extra_candidate_ids` (source ids to also retrieve as candidates,
independent of this turn's own keyword/embedding ranking). Neither ever
reaches the `sources` dict `run_grounded_multi` verifies quotes against —
`core/services/grounded.py` and `core/verify.py` are completely untouched
by this story. The new `ChatSession` (`core/services/chat.py`) is a thin,
purely in-memory stateful wrapper: each turn it feeds `history` (the last
`history_turns` turns, answers truncated to bound growth) and
`extra_candidate_ids` (only the *immediately previous* turn's actual
citations — not accumulated across the whole session, so a conversation
that has moved on to a new topic doesn't keep dragging in stale sources) to
an otherwise-ordinary `ask()` call, and appends the result to its
transcript. A turn that raises `QueryFailure` is never added to history, so
a rejected reply can't poison later turns. Nothing is persisted to the
database — a chat session lives and dies with the CLI process or the
running `mustrum ui` server, matching the backlog's "in-memory per session"
scope.

## ADR-19 — MCP exposes raw library data, not a grounded-answer tool (2026-07-13, accepted)
E13-3 puts the library behind MCP (Model Context Protocol) so external
tools — Claude Desktop, or any other MCP client — can read it. Two shapes
were considered: (a) an `ask_library` tool wrapping `QueryService.ask()`
(E13-1), so an external client gets a grounded prose answer with citations,
same as `mustrum chat`; or (b) plain read-only data-access tools
(`search_library`, `get_source`, `get_idea`, `list_citations`) returning
raw stored records, with zero LLM calls inside the MCP server. Went with
(b) — user-confirmed. The point of MCP here is letting an *external*
assistant read the library directly and do its own reasoning, not routing
every query back through mustrum's own LLM call; wrapping `ask()` would
also mean picking a session model (stateless vs. a `ChatSession` per MCP
connection, mirroring E13-2) for no clear benefit over just handing the
external assistant the data. Consequence: E13-3 makes **zero core changes**
and involves **no LLM call anywhere** — `mustrum/mcp/server.py` is a pure
driving adapter (sibling to `cli/`/`web/`, not under `adapters/`, since it
depends on core the same direction as the CLI/GUI do) reading straight
through `StorageRepo`. "Same grounding guarantees as chat/CLI" (per the
backlog) means every field returned is a direct, faithful readout of a
stored record — nothing is synthesised, so there is nothing to hallucinate.
Built on the official `mcp` Python SDK's high-level `FastMCP` API (decorator
tools, JSON schemas auto-derived from signatures — the same declarative
shape as `web/api.py`'s FastAPI endpoints), stdio transport (the client
spawns `mustrum mcp` as a subprocess per session, the standard way local
MCP servers run — consistent with this being a local-first, single-user
tool). MCP *resources* (direct source/idea reads, not just tool calls) are
the noted follow-up (E13-4), not this story.

## ADR-20 — MCP resources are an eager per-item snapshot at server startup, not a dynamic listing (2026-07-13, accepted)
E13-4 adds every source and idea in the library as an individually
listable MCP *resource* (`mustrum://sources/{id}`, `mustrum://ideas/{id}`)
alongside E13-3's tools, so a client can browse/attach one directly (e.g. a
resource picker) rather than only reaching it through `get_source`/
`get_idea`. The `mcp` SDK's `FastMCP.list_resources()` only enumerates
resources registered at construction time — dynamic per-request listing
isn't part of the high-level API — so `create_mcp_server(repo)` walks
`repo.list_sources()`/`list_ideas()` once at startup and registers one
resource per row. This means the *set* of browsable resources is a
snapshot: a source ingested after `mustrum mcp` started won't appear until
the process restarts. Accepted as consistent with the rest of the app's
existing "changes need a restart to take effect" pattern (ADR-16's
save-then-restart-notice for settings) rather than a new kind of
limitation. Each resource's *content*, however, is still computed fresh on
every read (the registered callback calls `get_source`/`get_idea` live), so
edits to an already-listed record (a new summary, a renamed title) show up
immediately — only the list of *which ids exist* is fixed per process.
Implementation note: the per-item read callback must take zero parameters
— `FastMCP.resource()` treats any function parameter (even one with a
default) as turning the registration into a URI *template* instead of a
concrete resource, which doesn't match a fixed, already-interpolated URI
like `mustrum://sources/3`; each id is captured via closure instead.

## ADR-21 — AnthropicProvider: config-switchable, no core changes (2026-07-14, accepted)

Resolves E10-1. `mustrum/adapters/anthropic.py::AnthropicLLM` implements the
existing `LLMProvider` Protocol unchanged (ADR-4/ADR-8 pattern) — `core/`
never learns a new provider exists. `Config.llm_provider` ("ollama" |
"anthropic", default "ollama") is a new library setting alongside
`anthropic_model` (default `claude-sonnet-5` — near-Opus quality on
summarise/rationale/brainstorm at a fraction of Opus cost, since these run
once per source/match/idea across a whole library) and
`anthropic_max_tokens`; `mustrum/cli/main.py::_build_llm` switches on it.
Embeddings always come from Ollama regardless of `llm_provider` — Anthropic
has no embeddings endpoint, and `EmbeddingProvider` is a separate port. The
API key is never read from config or stored in `config.toml`: `AnthropicLLM`
constructs a bare `anthropic.Anthropic()`, which resolves credentials from
`ANTHROPIC_API_KEY`/`ANTHROPIC_AUTH_TOKEN`/an `ant auth login` profile —
consistent with privacy rule 9 (config.toml must never carry secrets, even
though it isn't committed). `json_schema` structured output reuses
`output_config.format` (Anthropic's equivalent of Ollama's `format`,
ADR-14): syntax is constrained, but evidence quotes still pass
`GroundingVerifier` verbatim like every other provider — the rigour kernel
does not know or care which provider ran. `stop_reason == "max_tokens"`
raises loudly (mirrors Ollama's `done_reason=length` truncation error) and
`stop_reason == "refusal"` raises with Anthropic's `stop_details.explanation`
when present, since a declined generation must never be silently swallowed.
GUI Settings-panel parity (provider dropdown, like E12-2's Ollama model
dropdown) is deliberately out of scope here — CLI/config-file switching
(`mustrum config set --llm-provider anthropic --anthropic-model ...`) is
sufficient for this story; a GUI story can follow if wanted.
