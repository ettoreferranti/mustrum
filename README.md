# Mustrum

A local-first personal knowledge repository for academic research.

Feed it papers (PDF, arXiv ID, DOI, plain text) and your research ideas; it
stores everything in a local SQLite database, summarises sources with a local
LLM (Ollama), matches ideas to supporting literature, and generates
citation-perfect building blocks for new papers: BibTeX exports, related-work
skeletons (Markdown/LaTeX), and an interactive graph of ideas, sources, and
contacts.

**Core guarantee:** Mustrum never invents a citation. Every claim it makes
about a source is mechanically verified against stored, verbatim text before
it is saved or emitted.

## Quickstart

The fastest way in is the local web GUI — one command, opens in your browser:

```sh
uv run mustrum ui
```

Library browsing, ingestion (arXiv/DOI/file upload), verified summaries,
idea matching with explanations, related-work + BibTeX preview, the knowledge
graph, and brainstorming — all served from localhost, fully self-contained
(no CDNs, nothing leaves your machine). The GUI is a thin adapter over the
same services as the CLI: everything it does has a CLI equivalent below.

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) with `qwen3:30b` and `nomic-embed-text` pulled.

```sh
uv sync                                   # install
uv run mustrum --help

# build the library
uv run mustrum ingest arxiv 1706.03762    # metadata + BibTeX + full-text PDF
uv run mustrum ingest doi 10.1371/journal.pcbi.1003285   # + OA PDF via Unpaywall
uv run mustrum ingest file paper.pdf --title "..." --author "..." --year 2024
uv run mustrum ingest folder ~/papers -r   # batch-import every PDF; re-run safe
uv run mustrum summarise 1                # grounded, verified summary
uv run mustrum summarise --all             # every source still lacking one

# ideas and matching
uv run mustrum idea new "My idea" "one-paragraph description"
uv run mustrum idea import ideas.md      # bulk: each '# Heading' + body = one idea
uv run mustrum match suggest 1            # ranked candidate sources
uv run mustrum match confirm 3            # your judgement is final
uv run mustrum gaps                       # unsupported ideas, orphan sources

# writing support
uv run mustrum related-work 1 --format latex -o related.tex
uv run mustrum bib --idea 1 -o refs.bib
uv run mustrum audit draft.tex            # every \cite must exist in the library

# explore
uv run mustrum graph --open               # interactive offline HTML graph
uv run mustrum search "attention"
uv run mustrum contact add "Prof X" --kind university --affiliation "Unseen University"
```

## Command reference

Every command, grouped by task (`mustrum <command> --help` gives full options).

### Ingesting sources

| Command | Purpose |
|---|---|
| `mustrum ingest arxiv <id>` | Authoritative metadata + BibTeX + full-text PDF |
| `mustrum ingest doi <doi>` | Metadata + BibTeX via Crossref; PDF via Unpaywall open-access lookup, then publisher links (work on subscription networks) |
| `mustrum ingest file <path>` | One PDF or text/Markdown file (`--title`, `--author`, `--year`, `--kind`) |
| `mustrum ingest folder <dir>` | Every PDF in a folder (`-r` recursive); re-run safe |

All accept `--on-duplicate fail|skip|merge`; `merge` enriches an existing
record instead of duplicating. `--no-pdf` skips full-text download.

### Sources

| Command | Purpose |
|---|---|
| `mustrum source list` / `mustrum source show <id>` | Browse the library |
| `mustrum source attach <id> <file>` | Attach a manually-downloaded PDF to an existing source (upgrades an abstract; invalidates its summary) |
| `mustrum source status <id> <unread\|skimmed\|read>` | Reading status |
| `mustrum source tag <id> <tag>` (`--remove`) | Tags |
| `mustrum source note <id> "<text>"` | Personal notes (searchable) |
| `mustrum summarise <id>` | Grounded, verified summary (`--force`, `--override "<text>"`) |
| `mustrum summarise --all` | Every source lacking a summary; failures reported, never stored |

### Ideas

| Command | Purpose |
|---|---|
| `mustrum idea new "<title>" "<text>"` | Capture an idea (embedded immediately) |
| `mustrum idea import <file.md>` | Bulk import: one idea per `# Heading` (`--on-existing skip\|revise\|create`) |
| `mustrum idea revise <id> "<text>"` | New version; history kept forever |
| `mustrum idea list` / `mustrum idea show <id>` (`--history`) | Browse |
| `mustrum idea link <from> <to> --relation <r>` | builds-on / contrasts-with / related |

### Matching

| Command | Purpose |
|---|---|
| `mustrum match suggest <idea-id>` | Ranked source candidates (`--threshold`, `--limit`, `--explain`) |
| `mustrum match explain <match-id>` | Why is this source relevant? Grounded rationale with verified quotes (`--force`) |
| `mustrum match confirm <match-id>` / `mustrum match reject <match-id>` | Your judgement is final |
| `mustrum match add <idea-id> <source-id>` | Manually link a source |
| `mustrum match list <idea-id>` | Review matches (`--status`) |
| `mustrum gaps` | Ideas without confirmed support; orphan sources |

### Writing support & exploration

| Command | Purpose |
|---|---|
| `mustrum related-work <idea-id>` | Citation-verified skeleton (`--format markdown\|latex`, `-o`) |
| `mustrum bib` | BibTeX export (`--idea <id>`, `-o refs.bib`) |
| `mustrum audit <draft>` | Verify every `\\cite`/`[@key]` in a draft against the library |
| `mustrum graph` | Self-contained HTML knowledge graph (`--open`, `-o`, `--no-contacts`) |
| `mustrum search "<query>"` | Full-text search across everything |
| `mustrum contact add "<name>"` | People/institutions (`--kind`, `--affiliation`, `--email`, `--notes`) |
| `mustrum contact link <id> --idea <id>\|--source <id> --why "..."` | Attach contacts with the reason |
| `mustrum contact list` | Browse contacts |
| `mustrum brainstorm` | Creative mode: propose NEW research ideas from your library (`-n`, `--focus`, `--save`). Output is labelled machine-generated, cites nothing, and is quarantined from all citation-bearing features |
| `mustrum ui` | Launch the local web GUI (`--port`, `--no-open`) |
| `mustrum config` | Show effective configuration (`--init` writes a template) |
| `mustrum export <dir>` | Whole library as plain files: JSON + verbatim texts + byte-exact `.bib` + Markdown views (git-friendly backup) |
| `mustrum restore <dir>` | Rebuild the library from an export into an empty database (embeddings recomputed) |

The typical loop: ingest → summarise → capture ideas → match suggest/confirm
→ related-work + bib when writing → audit before submitting → graph to see
the big picture.

## Configuration & syncing (iCloud / OneDrive)

Your entire library — sources, verbatim texts, summaries, ideas, matches,
BibTeX, contacts, embeddings — lives in **one SQLite file**
(`~/.mustrum/mustrum.db` by default). Run `mustrum config --init` to create a
commented config template at `~/.config/mustrum/config.toml`, then point
`db_path` into a synced folder to keep the library in the cloud:

```toml
db_path = "~/Library/Mobile Documents/com~apple~CloudDocs/mustrum/mustrum.db"  # iCloud
# db_path = "~/OneDrive/mustrum/mustrum.db"
unpaywall_email = "you@example.org"   # enables open-access PDF lookup by DOI
```

`mustrum config` shows the effective settings; `MUSTRUM_DB` overrides the
path per invocation. Two rules for synced libraries: never run mustrum on two
machines against the same file simultaneously, and let the sync client finish
before switching machines. The config file (and your e-mail in it) stays on
your machine — nothing personal is ever part of this repository, enforced by
`tests/unit/test_privacy.py`.

## Status

Phase 1 (MVP) complete: ingest → summarise → match → cite, graph export,
contacts, CLI. Next: Phase 2 (export/backup, brainstorming mode) and the
Anthropic provider. See:

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — what it must do
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it is built
- [docs/BACKLOG.md](docs/BACKLOG.md) — prioritised work plan
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log (ADRs)
