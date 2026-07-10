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

## Status

Requirements engineering complete; implementation not yet started. See:

- [docs/REQUIREMENTS.md](docs/REQUIREMENTS.md) — what it must do
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — how it is built
- [docs/BACKLOG.md](docs/BACKLOG.md) — prioritised work plan
- [docs/DECISIONS.md](docs/DECISIONS.md) — decision log (ADRs)
