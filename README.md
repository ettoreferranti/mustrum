# Mustrum

A local-first personal knowledge repository for academic research.

Feed it papers (PDF, arXiv ID, DOI, plain text) and your research ideas; it
stores everything in a local SQLite database, summarises sources with an LLM
(local via Ollama by default, or Anthropic's API — config-switchable, see
[Configuration](#configuration--syncing-icloud--onedrive)), matches ideas to
supporting literature, and generates citation-perfect building blocks for new
papers: BibTeX exports, related-work skeletons (Markdown/LaTeX), and an
interactive graph of ideas, sources, and contacts.

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
graph, brainstorming, tag editing, contact links, and citation-audit upload —
all served from localhost, fully self-contained (no CDNs, nothing leaves
your machine). The GUI is a thin adapter over the same services as the CLI:
everything it does has a CLI equivalent below.

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) with `qwen3:30b` and `nomic-embed-text` pulled
(embeddings always use Ollama). Generation can instead run on Anthropic's API
(`mustrum config set --llm-provider anthropic`, `ANTHROPIC_API_KEY` in your
environment) — see [Configuration](#configuration--syncing-icloud--onedrive).

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

Every ingested or fetched original (PDF/text) is also archived in a visible
`files/` directory next to the database, so the library and its originals
back up as one unit — open them any time with `mustrum source open <id>`.
Re-running `ingest file`/`ingest folder` on already-known papers backfills
the archive for sources ingested before this feature.

### Sources

| Command | Purpose |
|---|---|
| `mustrum source list` / `mustrum source show <id>` | Browse the library |
| `mustrum source open <id>` | Open the archived original (PDF/text) with the default application |
| `mustrum source attach <id> <file>` | Attach a manually-downloaded PDF to an existing source (upgrades an abstract; invalidates its summary) |
| `mustrum source enrich <id>` / `--all` | Complete bare PDF sources with Crossref metadata found by exact-title lookup (authors, year, DOI, BibTeX) |
| `mustrum source rename <id> "<title>"` | Set a proper title (PDF ingests use PDF-metadata titles automatically when sane) |
| `mustrum source edit <id> --author "<name>" --year <yyyy>` | Set authors/year by hand — for venues Crossref doesn't index, e.g. CEUR-WS (`--author` repeatable) |
| `mustrum source delete <id>` | Remove a source and everything attached to it (`--yes` skips the prompt) |
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
| `mustrum idea delete <id>` | Remove an idea with its history and matches (`--yes`) |
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
| `mustrum chat` | Interactive grounded Q&A over your library — a REPL; every reply is grounded and cites `[source id]`s the same way `summarise`/`match explain` do. Prior turns are used only to resolve references like "it" in a follow-up, never as evidence (ADR-18). `exit`/`quit`/Ctrl+D to leave |
| `mustrum mcp` | Run an MCP server (stdio) for external tools like Claude Desktop: read-only `search_library`/`get_source`/`get_idea`/`list_citations` — raw data, no LLM call, nothing synthesised (ADR-19) |
| `mustrum ui` | Launch the local web GUI (`--port`, `--no-open`) |
| `mustrum config show` | Show the effective configuration and where each setting comes from |
| `mustrum config init` | Write a commented global bootstrap template (sets `db_path`) |
| `mustrum config set --llm-model X --num-ctx N ...` | Edit the library's own settings (model choice, context sizes, Unpaywall e-mail, `--llm-provider ollama\|anthropic` + `--anthropic-model`/`--anthropic-max-tokens`) — same as the UI Settings panel |
| `mustrum config models` | List models installed on the configured Ollama instance (marks the current `llm_model`/`embed_model`) — same list the UI Settings dropdowns fetch |
| `mustrum export <dir>` | Whole library as plain files: JSON + verbatim texts + byte-exact `.bib` + Markdown views (git-friendly backup) |
| `mustrum restore <dir>` | Rebuild the library from an export into an empty database (embeddings recomputed) |

The typical loop: ingest → summarise → capture ideas → match suggest/confirm
→ related-work + bib when writing → audit before submitting → graph to see
the big picture.

## Configuration & syncing (iCloud / OneDrive)

Your entire library — sources, verbatim texts, summaries, ideas, matches,
BibTeX, contacts, embeddings — lives in **one SQLite file**
(`~/.mustrum/mustrum.db` by default), with the archived original files in a
`files/` directory beside it: back up (or sync) the folder containing the DB
and you have everything, including its settings.

There are two config files. The **global bootstrap file**
(`~/.config/mustrum/config.toml`, created by `mustrum config init`) has one
real job: pointing `db_path` at your library, e.g. to keep it in the cloud:

```toml
db_path = "~/Library/Mobile Documents/com~apple~CloudDocs/mustrum/mustrum.db"  # iCloud
# db_path = "~/OneDrive/mustrum/mustrum.db"
```

Everything else — Ollama URL, model choice, context sizes, the Unpaywall
e-mail — belongs in the **library config file**, `config.toml` sitting next
to `mustrum.db` itself. It travels with the library (so a synced/backed-up
library folder carries its own settings, not just data) and is edited with
`mustrum config set --llm-model llama3.1:8b --unpaywall-email you@example.org`
or the **Settings panel in the UI** (`mustrum ui` → Settings tab) — never by
hand-editing required, though it's a plain commented TOML file if you prefer
that. The UI's model fields are dropdowns, fetched live from the configured
Ollama instance (`mustrum config models` on the CLI), so you pick from what's
actually installed instead of typing a name that might have a typo or isn't
pulled; if Ollama is unreachable the dropdown falls back to just the current
value and the rest of the form stays usable. Changes take effect on the next
`mustrum` invocation / `mustrum ui` restart; a running `mustrum ui` process
does not hot-reload them, since its Ollama clients are built once at startup.

**Switching to Anthropic:** `llm_provider` picks the generation backend —
`ollama` (default) or `anthropic`:

```sh
mustrum config set --llm-provider anthropic --anthropic-model claude-sonnet-5
```

Embeddings always stay on Ollama (Anthropic has no embeddings endpoint), so
`embed_model`/`ollama_url` still matter either way. The Anthropic API key is
never stored in `config.toml` — set `ANTHROPIC_API_KEY` in your environment,
or run `ant auth login`; mustrum resolves credentials the same way the
Anthropic SDK/CLI do. `anthropic_max_tokens` (default 8192) caps a single
reply; a cut-off response raises loudly instead of failing silently
downstream, same as Ollama's `num_ctx` truncation guard.

`mustrum config show` prints the effective settings and where each one came
from (defaults ← global file ← library file ← `MUSTRUM_DB`/`MUSTRUM_OLLAMA_URL`
env vars, in that order). Two rules for synced libraries: never run mustrum
on two machines against the same file simultaneously, and let the sync
client finish before switching machines. Both config files stay on your
machine — nothing personal is ever part of this repository, enforced by
`tests/unit/test_privacy.py`.

## Status

Phases 0–2 complete (MVP, GUI, configuration, chat & MCP), plus the
Anthropic provider (E10-1) — see [docs/BACKLOG.md](docs/BACKLOG.md) for the
full, current story-by-story status. Remaining: watch-folder auto-ingest,
Zotero/contact import, and a provider-benchmarking harness. See:

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — what it must do
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it is built
- [docs/BACKLOG.md](docs/BACKLOG.md) — prioritised work plan
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log (ADRs)
