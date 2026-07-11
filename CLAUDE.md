# Mustrum — project instructions

Personal knowledge repository for academic research (papers + ideas +
citations + contacts). Python, hexagonal architecture, local-first.

## Standing rules

1. **Never invent citations or facts about sources.** All generation involving
   sources must go through the verifiers in `mustrum/core/verify.py`
   (grounded quotes, DB-only citation keys). This overrides everything else.
2. **Keep `docs/ARCHITECTURE.md` up to date** — update it in the same commit
   as any structural change (new module/adapter, schema change, changed flow).
3. **Keep `docs/BACKLOG.md` current** — mark stories done as they land; add
   new stories when scope is agreed with the user.
4. **Mutation testing is part of done:** run `mutmut` on changed core modules;
   ≥80% mutation score on `mustrum/core/`, and every surviving mutant in
   `core/verify.py` must be killed or explicitly justified.
5. Record significant technical decisions as ADRs in `docs/DECISIONS.md`
   (append-only; supersede, don't rewrite).
6. Core never imports adapters; all model access via the `LLMProvider` /
   `EmbeddingProvider` ports (Ollama now, Anthropic later, fakes in tests).
7. Default test suite must pass offline with no Ollama running (fake
   providers); Ollama/network integration tests behind pytest markers.
8. **Privacy — this repo is public.** Never commit personal data: no real
   e-mail addresses, affiliations, or machine paths; user settings live only
   in `~/.config/mustrum/config.toml`. Generated artefacts (graph HTML, .bib
   exports, .db files) contain the user's library and must stay gitignored —
   check `git status` before staging; avoid blind `git add -A` when new
   artefact types may exist. `tests/unit/test_privacy.py` enforces the e-mail
   and home-path bans on all tracked files.

## Key docs

- `docs/REQUIREMENTS.md` — agreed requirements (FR/NFR numbering used in code
  review discussion)
- `docs/ARCHITECTURE.md` — living architecture doc
- `docs/BACKLOG.md` — prioritised backlog
- `docs/DECISIONS.md` — ADR log
