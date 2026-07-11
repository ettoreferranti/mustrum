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
uv run mustrum contact add "Prof X" --kind university --affiliation "ZHAW"
```

Data lives in `~/.mustrum/mustrum.db` (override with `MUSTRUM_DB` or
`~/.config/mustrum/config.toml`). Set `unpaywall_email = "you@example.org"`
in the config to enable open-access PDF lookup for DOI ingestion (paywalled
papers fall back to their abstract).

## Status

Phase 1 (MVP) complete: ingest → summarise → match → cite, graph export,
contacts, CLI. Next: Phase 2 (export/backup, brainstorming mode) and the
Anthropic provider. See:

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — what it must do
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it is built
- [docs/BACKLOG.md](docs/BACKLOG.md) — prioritised work plan
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log (ADRs)
