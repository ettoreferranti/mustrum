# Mustrum — Requirements

> Personal knowledge repository for academic research: ingest papers and other
> sources, capture research ideas, match them, and generate rigorously-cited
> related-work material.

Status: **agreed 2026-07-10** (requirements engineering session). Update this
document when requirements change; record the date.

## 1. Vision

A local-first tool that acts as a researcher's external memory. It stores
sources (papers, articles, notes) and research ideas, understands both well
enough to connect them, and produces citation-perfect building blocks
(BibTeX, related-work skeletons) for new papers. It must be **incapable of
inventing a citation**: everything it asserts about a source is traceable to
stored, verbatim material.

## 2. Personas & context

- Single user (the researcher). No multi-user, no auth, no cloud sync in scope.
- Runs on macOS (primary) and Linux. Local-first: works offline except for
  explicit metadata fetches (arXiv/DOI).

## 3. Functional requirements

### FR-1 Source ingestion
- FR-1.1 Ingest a PDF: extract text, store it verbatim and immutably.
- FR-1.2 Ingest by arXiv ID or DOI: fetch authoritative metadata and BibTeX
  from arXiv / Crossref (never synthesise metadata locally).
- FR-1.3 Ingest plain text / Markdown (notes, blog posts, non-paper sources).
- FR-1.4 Deduplicate on DOI, arXiv ID, and normalised-title hash; on conflict,
  offer to merge/enrich the existing record instead of duplicating.
- FR-1.5 Every source records provenance: where it came from, when, and which
  fields are authoritative (fetched) vs. extracted vs. user-entered.

### FR-2 Research ideas
- FR-2.1 Capture an idea as free text with a title.
- FR-2.2 Ideas are versioned: edits create a new version; history is kept.
- FR-2.3 Ideas can be linked to each other (builds-on, contrasts-with, related).

### FR-3 Summarisation
- FR-3.1 Generate a summary per source ("what the authors did") via the model
  provider (phase 1: Ollama).
- FR-3.2 Every summary is stored flagged as machine-generated, with model name
  and date; the user can override with a hand-written summary.
- FR-3.3 Grounding: each summary must carry supporting verbatim quotes from the
  stored source text; a verifier checks the quotes actually appear in the
  source. A summary that fails verification is rejected, not stored.

### FR-4 Idea ↔ source matching
- FR-4.1 Compute semantic matches between ideas and sources (embeddings),
  ranked by score, with a tunable threshold.
- FR-4.2 Each confirmed match stores a rationale grounded in quotes (FR-3.3
  rules apply).
- FR-4.3 The user can confirm, reject, or manually create matches; user
  judgement always overrides the machine's.
- FR-4.4 Gap analysis: report ideas with no/weak supporting sources, and
  sources not connected to any idea.

### FR-5 Bibliography & related-work generation
- FR-5.1 Maintain a BibTeX entry per source (fetched when possible, else
  derived from stored metadata; raw `.bib` text stored as-is when fetched).
- FR-5.2 Export a `.bib` file for the whole library or for the sources matched
  to a given idea.
- FR-5.3 Generate a related-work skeleton for an idea: grouped sources, each
  with its citation key, a brief grounded summary of what the authors did, and
  its relation to the idea. Output in Markdown and LaTeX.
- FR-5.4 **Citation integrity (hard rule):** generated text may only cite keys
  that exist in the database; a post-generation verifier enforces this and
  fails loudly on violation. No silent fixing.
- FR-5.5 Citation audit command: given a draft `.tex`/`.md`, check every
  `\cite{...}` against the library and report unknown/mismatched keys.

### FR-6 Graph visualisation
- FR-6.1 Export an interactive graph as a single self-contained HTML file
  (no network access needed to view it): ideas as hubs, their sources around
  them (star pattern), idea↔idea edges, contact nodes attachable to ideas.
- FR-6.2 Filter by tag, idea, entity type; node click shows details (summary,
  citation key, match rationale).

### FR-7 Contacts
- FR-7.1 Store contacts: people, companies, research institutions,
  universities — name, affiliation, role, email/URL, free-text notes.
- FR-7.2 Link contacts to ideas (and optionally to sources, e.g. authors),
  with a note on *why* they are relevant.

### FR-8 Search & organisation
- FR-8.1 Full-text search across sources, ideas, summaries, contacts.
- FR-8.2 Tags on sources and ideas; reading status per source
  (unread / skimmed / read).
- FR-8.3 Personal notes per source, kept separate from the verbatim text.

## 4. Non-functional requirements

- NFR-1 **Rigour / anti-hallucination** (overrides all other concerns):
  citations and factual claims about sources must be mechanically traceable to
  stored data. Creative output (new idea brainstorming) is allowed only where
  explicitly requested and is labelled as such.
- NFR-2 **Model-provider independence:** all LLM/embedding use goes through a
  narrow provider interface. Phase 1 implements Ollama; an Anthropic adapter
  must be addable without touching core logic. Core tests run with a fake
  provider — no Ollama required in CI.
- NFR-3 **Local-first:** all data in a local SQLite DB + on-disk source texts.
  Network access only for explicit arXiv/DOI fetches.
- NFR-4 **Testing:** unit + integration tests; **mutation testing** (mutmut)
  on core modules with a target mutation score ≥ 80%, enforced for the
  grounding/citation-verification code at 100% surviving-mutant review.
- NFR-5 **Data longevity:** export/import of the whole repository as plain
  files (JSON + Markdown + .bib) so the data outlives the tool and can be
  git-versioned.
- NFR-6 Performance: library of ~5k sources must stay responsive (search
  < 1 s, matching a new idea < 30 s with local models).
- NFR-7 Code quality: typed Python (mypy strict on core), ruff lint/format.

## 5. Explicitly out of scope (for now)

- Multi-user / collaboration / cloud sync.
- Reference-manager integrations (Zotero/Mendeley import) — candidate later.
- Full PDF layout understanding (figures, tables); text extraction only.
- Web crawling / automatic paper discovery.

## 6. Suggested features adopted from the session

These were proposed by Claude and accepted into the backlog (see BACKLOG.md
for priority): deduplication (FR-1.4), provenance tracking (FR-1.5), idea
versioning (FR-2.2), grounded summaries with mechanical verification (FR-3.3),
gap analysis (FR-4.4), citation audit of drafts (FR-5.5), full-text search and
tags (FR-8), plain-file export for longevity (NFR-5).

## 7. Open questions

- ~~OQ-1 Which Ollama models to standardise on?~~ **Resolved 2026-07-10**
  (see ADR-8): `nomic-embed-text` for embeddings, `qwen3:30b` (MoE) for
  generation — the dev machine (M5 Pro, 48 GB) runs it comfortably. Model
  names are config, not code.
- OQ-2 Should contacts also be importable (vCard/CSV)? Deferred.
- OQ-3 LaTeX bibliography style handling (natbib vs biblatex) for the
  related-work skeleton — start with plain `\cite{}` and revisit.
