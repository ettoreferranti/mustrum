"""Mustrum CLI: the driving adapter. All logic lives in core services."""

from __future__ import annotations

import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

if TYPE_CHECKING:
    from mustrum.adapters.oa import FullTextResult

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.ollama import OllamaEmbedder, OllamaLLM
from mustrum.adapters.pdf import extractor_for
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.config import Config, load_config
from mustrum.core.models import (
    Contact,
    ContactKind,
    ContactLink,
    EntityKind,
    FetchedMetadata,
    IdeaRelation,
    Match,
    MatchStatus,
    ReadingStatus,
    Source,
    SourceKind,
)
from mustrum.core.ports import EmbeddingProvider, LLMProvider, MetadataFetcher
from mustrum.core.services.audit import AuditService
from mustrum.core.services.ideas import IdeaFileError, IdeaService
from mustrum.core.services.ingest import DuplicateSourceError, IngestService
from mustrum.core.services.match import MatchService
from mustrum.core.services.relatedwork import RelatedWorkService
from mustrum.core.services.summarise import GroundingFailure, SummariseService

app = typer.Typer(help="Mustrum — personal knowledge repository for academic research.")
ingest_app = typer.Typer(help="Add sources to the library.")
source_app = typer.Typer(help="Browse and annotate sources.")
idea_app = typer.Typer(help="Capture and evolve research ideas.")
match_app = typer.Typer(help="Match ideas with sources.")
contact_app = typer.Typer(help="People and institutions related to your work.")
app.add_typer(ingest_app, name="ingest")
app.add_typer(source_app, name="source")
app.add_typer(idea_app, name="idea")
app.add_typer(match_app, name="match")
app.add_typer(contact_app, name="contact")


@dataclass
class Context:
    config: Config
    repo: SqliteRepo
    embedder: EmbeddingProvider
    llm: LLMProvider


def _context() -> Context:
    config = load_config()
    config.db_path.parent.mkdir(parents=True, exist_ok=True)
    repo = SqliteRepo(config.db_path)
    if os.environ.get("MUSTRUM_FAKE_PROVIDERS"):  # offline test hook
        canned = os.environ.get("MUSTRUM_FAKE_LLM_RESPONSE")
        return Context(
            config, repo, FakeEmbeddingProvider(), FakeLLMProvider(default_response=canned)
        )
    return Context(
        config,
        repo,
        OllamaEmbedder(config.embed_model, base_url=config.ollama_url),
        OllamaLLM(config.llm_model, base_url=config.ollama_url, num_ctx=config.num_ctx),
    )


def _fail(message: str) -> None:
    typer.secho(message, fg=typer.colors.RED, err=True)
    raise typer.Exit(code=1)


def _print_source(source: Source) -> None:
    year = f" ({source.year})" if source.year else ""
    typer.echo(f"[{source.id}] {source.title}{year} — {source.kind.value}")


# -- ingest ---------------------------------------------------------------------

DupOption = Annotated[str, typer.Option("--on-duplicate", help="fail | skip | merge")]


@ingest_app.command("file")
def ingest_file(
    path: Path,
    title: Annotated[str | None, typer.Option(help="Defaults to the file name.")] = None,
    kind: SourceKind = SourceKind.PAPER,
    author: Annotated[list[str] | None, typer.Option(help="Repeatable.")] = None,
    year: int | None = None,
    on_duplicate: DupOption = "fail",
) -> None:
    """Ingest a PDF or plain-text/Markdown file."""
    if not path.is_file():
        _fail(f"no such file: {path}")
    ctx = _context()
    extractor = extractor_for(path)
    if title is None and path.suffix.lower() == ".pdf":
        from mustrum.adapters.pdf import pdf_metadata_title

        title = pdf_metadata_title(path)
    try:
        result = IngestService(ctx.repo, ctx.embedder).ingest_document(
            title=title or path.stem,
            text=extractor.extract(path),
            extraction_method=extractor.extraction_method,
            kind=kind,
            authors=tuple(author or ()),
            year=year,
            on_duplicate=on_duplicate,  # type: ignore[arg-type]
        )
    except DuplicateSourceError as exc:
        _fail(f"{exc}\nUse --on-duplicate skip|merge to resolve.")
        return
    _archive_ingested(ctx, result.source, path)
    verb = (
        "merged into"
        if result.merged
        else ("already in library:" if not result.created else "ingested")
    )
    typer.echo(f"{verb} ", nl=False)
    _print_source(result.source)


def _archive_ingested(ctx: Context, source: Source, path: Path) -> None:
    """Keep the ingested original in the files dir (E1-11). Fills the gap for
    sources that don't have an archived original yet; never replaces one."""
    from mustrum.adapters.archive import archive_original

    if source.file_path is not None or source.id is None:
        return
    archive_original(
        ctx.repo, ctx.config.files_dir, source, path.read_bytes(), path.suffix or ".txt"
    )


@ingest_app.command("folder")
def ingest_folder(
    directory: Path,
    recursive: Annotated[bool, typer.Option("--recursive", "-r")] = False,
    kind: SourceKind = SourceKind.PAPER,
    on_duplicate: DupOption = "skip",
) -> None:
    """Ingest every PDF in a folder (title = file name). Already-known papers
    are skipped by default, so re-running on the same folder is safe."""
    if not directory.is_dir():
        _fail(f"no such directory: {directory}")
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdfs = sorted(p for p in directory.glob(pattern) if p.is_file())
    if not pdfs:
        typer.echo(f"no PDFs found in {directory}")
        return
    ctx = _context()
    service = IngestService(ctx.repo, ctx.embedder)
    ingested = skipped = failed = 0
    from mustrum.adapters.pdf import pdf_metadata_title

    for pdf in pdfs:
        extractor = extractor_for(pdf)
        try:
            result = service.ingest_document(
                title=pdf_metadata_title(pdf) or pdf.stem,
                text=extractor.extract(pdf),
                extraction_method=extractor.extraction_method,
                kind=kind,
                on_duplicate=on_duplicate,  # type: ignore[arg-type]
            )
        except DuplicateSourceError as exc:
            typer.secho(f"duplicate: {pdf.name} ({exc.matched_on})", fg=typer.colors.YELLOW)
            failed += 1
            continue
        except Exception as exc:  # a corrupt PDF must not abort the batch
            typer.secho(f"failed: {pdf.name}: {exc}", fg=typer.colors.RED, err=True)
            failed += 1
            continue
        _archive_ingested(ctx, result.source, pdf)
        if result.created:
            ingested += 1
            typer.echo(f"ingested [{result.source.id}] {result.source.title}")
        else:
            skipped += 1
            typer.echo(f"skipped (already known): {pdf.name}")
    typer.echo(f"done: {ingested} ingested, {skipped} skipped, {failed} failed")
    if failed:
        raise typer.Exit(code=1)


def _fetch_full_text(ctx: Context, meta: FetchedMetadata) -> FullTextResult:
    """Shared PDF-candidate logic lives in adapters/oa.py; this just reports."""
    from mustrum.adapters.oa import fetch_full_text

    result = fetch_full_text(meta, ctx.config.unpaywall_email)
    for note in result.notes:
        if note.startswith("fetched"):
            typer.echo(note)
        else:
            typer.secho(note, fg=typer.colors.YELLOW)
    return result


def _ingest_fetched(
    identifier: str, fetcher: MetadataFetcher, on_duplicate: str, fetch_pdf: bool
) -> None:
    from mustrum.adapters.archive import archive_original
    from mustrum.adapters.oa import FullTextResult

    ctx = _context()
    try:
        meta = fetcher.fetch(identifier)
    except (LookupError, ValueError) as exc:
        _fail(str(exc))
        return
    full_text = _fetch_full_text(ctx, meta) if fetch_pdf else FullTextResult()
    try:
        result = IngestService(ctx.repo, ctx.embedder).ingest_fetched(
            meta,
            on_duplicate=on_duplicate,  # type: ignore[arg-type]
            full_text=full_text.text,
        )
    except DuplicateSourceError as exc:
        _fail(f"{exc}\nUse --on-duplicate skip|merge to resolve.")
        return
    if full_text.pdf_bytes and result.source.file_path is None and result.source.id is not None:
        archive_original(ctx.repo, ctx.config.files_dir, result.source, full_text.pdf_bytes, ".pdf")
    verb = (
        "merged into"
        if result.merged
        else ("already in library:" if not result.created else "ingested")
    )
    typer.echo(f"{verb} ", nl=False)
    _print_source(result.source)
    if result.source.id is not None:
        bib = ctx.repo.get_bib_entry(result.source.id)
        if bib:
            typer.echo(f"citation key: {bib.citation_key}")


@ingest_app.command("arxiv")
def ingest_arxiv(
    arxiv_id: str,
    on_duplicate: DupOption = "fail",
    pdf: Annotated[bool, typer.Option(help="Also download the PDF full text.")] = True,
) -> None:
    """Fetch authoritative metadata + BibTeX for an arXiv id (and its PDF)."""
    from mustrum.adapters.arxiv import ArxivFetcher

    _ingest_fetched(arxiv_id, ArxivFetcher(), on_duplicate, fetch_pdf=pdf)


@ingest_app.command("doi")
def ingest_doi(
    doi: str,
    on_duplicate: DupOption = "fail",
    pdf: Annotated[
        bool, typer.Option(help="Look up + download an open-access PDF via Unpaywall.")
    ] = True,
) -> None:
    """Fetch authoritative metadata + BibTeX for a DOI via Crossref, plus the
    full-text PDF when a legal open-access copy exists (Unpaywall)."""
    from mustrum.adapters.crossref import CrossrefFetcher

    _ingest_fetched(doi, CrossrefFetcher(), on_duplicate, fetch_pdf=pdf)


# -- sources ---------------------------------------------------------------------


@source_app.command("list")
def source_list() -> None:
    ctx = _context()
    for source in ctx.repo.list_sources():
        _print_source(source)


@source_app.command("show")
def source_show(source_id: int) -> None:
    ctx = _context()
    try:
        source = ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    _print_source(source)
    if source.authors:
        typer.echo(f"authors: {', '.join(source.authors)}")
    for field in ("doi", "arxiv_id"):
        if getattr(source, field):
            typer.echo(f"{field}: {getattr(source, field)}")
    typer.echo(f"status: {source.reading_status.value}")
    if source.file_path:
        typer.echo(f"file: {ctx.config.files_dir / source.file_path}")
    tags = ctx.repo.tags_for(EntityKind.SOURCE, source_id)
    if tags:
        typer.echo(f"tags: {', '.join(sorted(tags))}")
    if source.notes:
        typer.echo(f"notes: {source.notes}")
    summary = ctx.repo.get_summary(source_id)
    if summary:
        origin = "user" if summary.user_override else summary.model
        typer.echo(f"summary ({origin}, verified={summary.verified}): {summary.text}")
    bib = ctx.repo.get_bib_entry(source_id)
    if bib:
        typer.echo(f"citation key: {bib.citation_key} ({bib.origin.value})")


@source_app.command("attach")
def source_attach(source_id: int, path: Path) -> None:
    """Attach a downloaded PDF (or text file) to an existing source — e.g. a
    paper ingested by DOI whose PDF the tool couldn't fetch automatically.
    Upgrading an abstract invalidates the summary (re-run summarise)."""
    if not path.is_file():
        _fail(f"no such file: {path}")
    ctx = _context()
    extractor = extractor_for(path)
    had_summary = ctx.repo.get_summary(source_id) is not None
    try:
        IngestService(ctx.repo, ctx.embedder).attach_full_text(
            source_id, extractor.extract(path), extractor.extraction_method
        )
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
        return
    # the attached file is now the source's original — archive it (replacing
    # any earlier archived original, E1-11)
    from mustrum.adapters.archive import archive_original

    updated = archive_original(
        ctx.repo,
        ctx.config.files_dir,
        ctx.repo.get_source(source_id),
        path.read_bytes(),
        path.suffix or ".txt",
    )
    assert updated.file_path is not None
    typer.echo(f"attached full text to [{source_id}]")
    typer.echo(f"archived original: {ctx.config.files_dir / updated.file_path}")
    if had_summary:
        typer.secho(
            f"summary invalidated — run: mustrum summarise {source_id}", fg=typer.colors.YELLOW
        )


@source_app.command("delete")
def source_delete(
    source_id: int,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete a source and everything attached to it (text, summary, BibTeX,
    matches, tags, contact links). Drafts citing its key will fail audit."""
    ctx = _context()
    try:
        source = ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    matches = [m for m in ctx.repo.list_matches() if m.source_id == source_id]
    bib = ctx.repo.get_bib_entry(source_id)
    if not yes:
        detail = f"{len(matches)} match(es)" + (
            f", citation key '{bib.citation_key}'" if bib else ""
        )
        typer.confirm(f"Delete [{source_id}] {source.title} ({detail})?", abort=True)
    ctx.repo.delete_source(source_id)
    from mustrum.adapters.archive import delete_archived

    delete_archived(ctx.config.files_dir, source)
    typer.echo(f"deleted [{source_id}] {source.title}")


@source_app.command("open")
def source_open(source_id: int) -> None:
    """Open a source's archived original file (PDF/text) with the default
    application. Originals live in a `files/` directory next to the DB."""
    from mustrum.adapters.archive import archived_file

    ctx = _context()
    try:
        source = ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    path = archived_file(ctx.config.files_dir, source)
    if path is None:
        _fail(
            f"no archived file for [{source_id}] {source.title} — "
            f"attach one with: mustrum source attach {source_id} FILE"
        )
        return
    typer.launch(str(path))
    typer.echo(f"opened {path}")


@source_app.command("rename")
def source_rename(source_id: int, title: str) -> None:
    """Set a proper title on a source (e.g. one ingested from an ugly file
    name). Dedup keys and the search index follow the new title."""
    import dataclasses

    from mustrum.core.normalize import title_hash

    ctx = _context()
    try:
        source = ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    clash = ctx.repo.find_source_by_title_hash(title_hash(title))
    if clash is not None and clash.id != source_id:
        _fail(f"another source already has this title: [{clash.id}] {clash.title}")
    ctx.repo.update_source(dataclasses.replace(source, title=title))
    typer.echo(f"renamed [{source_id}] to: {title}")


@source_app.command("enrich")
def source_enrich(
    source_id: Annotated[int | None, typer.Argument()] = None,
    enrich_all: Annotated[
        bool, typer.Option("--all", help="Every source that lacks a DOI.")
    ] = False,
) -> None:
    """Complete a bare PDF-ingested source with authoritative Crossref
    metadata (authors, year, DOI, BibTeX), found by exact-title lookup."""
    from mustrum.adapters.enrich import enrich_source

    if enrich_all == (source_id is not None):
        _fail("give either a SOURCE_ID or --all")
    ctx = _context()
    targets = (
        [source_id]
        if source_id is not None
        else [s.id for s in ctx.repo.list_sources() if not s.doi and s.id is not None]
    )
    if not targets:
        typer.echo("nothing to enrich — every source already has a DOI")
        return
    failed = 0
    for target in targets:
        assert target is not None
        try:
            result = enrich_source(ctx.repo, ctx.embedder, target)
        except KeyError as exc:
            _fail(str(exc))
            return
        except Exception as exc:  # network errors must not abort --all
            typer.secho(f"[{target}] lookup failed: {exc}", fg=typer.colors.RED, err=True)
            failed += 1
            continue
        colour = None if result.enriched else typer.colors.YELLOW
        typer.secho(f"[{target}] {result.message}", fg=colour)
        if not result.enriched:
            failed += 1
    if failed and enrich_all:
        raise typer.Exit(code=1)


@source_app.command("status")
def source_status(source_id: int, status: ReadingStatus) -> None:
    ctx = _context()
    try:
        ctx.repo.set_reading_status(source_id, status)
    except KeyError as exc:
        _fail(str(exc))


@source_app.command("note")
def source_note(source_id: int, text: str) -> None:
    ctx = _context()
    try:
        ctx.repo.set_source_notes(source_id, text)
    except KeyError as exc:
        _fail(str(exc))


@source_app.command("tag")
def source_tag(
    source_id: int,
    tag: str,
    remove: Annotated[bool, typer.Option("--remove")] = False,
) -> None:
    ctx = _context()
    try:
        ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    if remove:
        ctx.repo.untag(EntityKind.SOURCE, source_id, tag)
    else:
        ctx.repo.tag(EntityKind.SOURCE, source_id, tag)


# -- ideas -----------------------------------------------------------------------


@idea_app.command("new")
def idea_new(title: str, text: str) -> None:
    ctx = _context()
    idea = IdeaService(ctx.repo, ctx.embedder).create(title, text)
    typer.echo(f"created idea [{idea.id}] {idea.title}")


@idea_app.command("import")
def idea_import(
    path: Path,
    on_existing: Annotated[
        str, typer.Option("--on-existing", help="skip | revise | create")
    ] = "skip",
) -> None:
    """Bulk-import ideas from a Markdown file: each '# Heading' starts a new
    idea (heading = title, body until the next heading = idea text)."""
    if not path.is_file():
        _fail(f"no such file: {path}")
    if on_existing not in ("skip", "revise", "create"):
        _fail("--on-existing must be skip, revise, or create")
    ctx = _context()
    try:
        outcomes = IdeaService(ctx.repo, ctx.embedder).import_ideas(
            path.read_text(encoding="utf-8"),
            on_existing,  # type: ignore[arg-type]
        )
    except IdeaFileError as exc:
        _fail(f"{path}: {exc}")
        return
    for outcome in outcomes:
        typer.echo(f"{outcome.action} [{outcome.idea_id}] {outcome.title}")


@idea_app.command("revise")
def idea_revise(idea_id: int, text: str) -> None:
    ctx = _context()
    try:
        IdeaService(ctx.repo, ctx.embedder).revise(idea_id, text)
    except KeyError as exc:
        _fail(str(exc))
        return
    versions = ctx.repo.get_idea_versions(idea_id)
    typer.echo(f"idea {idea_id} now has {len(versions)} versions")


@idea_app.command("delete")
def idea_delete(
    idea_id: int,
    yes: Annotated[bool, typer.Option("--yes", help="Skip the confirmation prompt.")] = False,
) -> None:
    """Delete an idea with its whole version history, matches, and links."""
    ctx = _context()
    try:
        idea = ctx.repo.get_idea(idea_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    if not yes:
        versions = len(ctx.repo.get_idea_versions(idea_id))
        typer.confirm(f"Delete idea [{idea_id}] {idea.title} ({versions} version(s))?", abort=True)
    ctx.repo.delete_idea(idea_id)
    typer.echo(f"deleted idea [{idea_id}] {idea.title}")


@idea_app.command("list")
def idea_list() -> None:
    ctx = _context()
    for idea in ctx.repo.list_ideas():
        typer.echo(f"[{idea.id}] {idea.title}")


@idea_app.command("show")
def idea_show(idea_id: int, history: Annotated[bool, typer.Option("--history")] = False) -> None:
    ctx = _context()
    try:
        idea = ctx.repo.get_idea(idea_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    typer.echo(f"[{idea.id}] {idea.title}")
    versions = ctx.repo.get_idea_versions(idea_id)
    if history:
        for v in versions:
            typer.echo(f"  v{v.id} ({v.created_at:%Y-%m-%d}): {v.text}")
    elif versions:
        typer.echo(versions[-1].text)
    for match in ctx.repo.list_matches(idea_id, MatchStatus.CONFIRMED):
        source = ctx.repo.get_source(match.source_id)
        typer.echo(f"  confirmed: [{source.id}] {source.title} (score {match.score:.2f})")


@idea_app.command("link")
def idea_link(from_id: int, to_id: int, relation: IdeaRelation = IdeaRelation.RELATED) -> None:
    ctx = _context()
    try:
        IdeaService(ctx.repo, ctx.embedder).link(from_id, to_id, relation)
    except (KeyError, ValueError) as exc:
        _fail(str(exc))


# -- matching ----------------------------------------------------------------------


def _explain_match(ctx: Context, match_id: int, force: bool = False) -> bool:
    """Generate + print a grounded rationale; returns False on grounding failure."""
    from mustrum.core.services.rationale import RationaleFailure, RationaleService

    service = RationaleService(ctx.repo, ctx.llm, max_source_chars=ctx.config.max_source_chars)
    try:
        match = service.explain(match_id, force=force)
    except (RationaleFailure, LookupError) as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        return False
    typer.echo(f"  why: {match.rationale}")
    for quote in match.quotes:
        typer.echo(f'  evidence: "{quote}"')
    return True


@match_app.command("suggest")
def match_suggest(
    idea_id: int,
    limit: int = 20,
    threshold: float = 0.35,
    explain: Annotated[
        bool, typer.Option("--explain", help="Generate a grounded rationale per suggestion.")
    ] = False,
) -> None:
    ctx = _context()
    service = MatchService(ctx.repo, ctx.embedder.model_name, threshold=threshold)
    try:
        matches = service.suggest(idea_id, limit=limit)
    except (KeyError, LookupError) as exc:
        _fail(str(exc))
        return
    if not matches:
        typer.echo("no new suggestions above threshold")
    for match in matches:
        source = ctx.repo.get_source(match.source_id)
        typer.echo(f"match [{match.id}] score {match.score:.2f}: [{source.id}] {source.title}")
        if explain:
            assert match.id is not None
            _explain_match(ctx, match.id)


@match_app.command("explain")
def match_explain(match_id: int, force: Annotated[bool, typer.Option("--force")] = False) -> None:
    """Explain why a matched source is relevant to its idea, with verified
    quotes from the paper. Unverifiable explanations are rejected, not stored."""
    ctx = _context()
    try:
        ctx.repo.get_match(match_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    if not _explain_match(ctx, match_id, force=force):
        raise typer.Exit(code=1)


@match_app.command("list")
def match_list(idea_id: int, status: MatchStatus | None = None) -> None:
    ctx = _context()
    for match in ctx.repo.list_matches(idea_id, status):
        source = ctx.repo.get_source(match.source_id)
        typer.echo(
            f"[{match.id}] {match.status.value} score {match.score:.2f}: "
            f"[{source.id}] {source.title}"
        )
        if match.rationale:
            typer.echo(f"  why: {match.rationale}")


@match_app.command("confirm")
def match_confirm(match_id: int) -> None:
    ctx = _context()
    try:
        MatchService(ctx.repo, ctx.embedder.model_name).confirm(match_id)
    except KeyError as exc:
        _fail(str(exc))


@match_app.command("reject")
def match_reject(match_id: int) -> None:
    ctx = _context()
    try:
        MatchService(ctx.repo, ctx.embedder.model_name).reject(match_id)
    except KeyError as exc:
        _fail(str(exc))


@match_app.command("add")
def match_add(idea_id: int, source_id: int) -> None:
    """Manually link a source to an idea (confirmed, FR-4.3)."""
    ctx = _context()
    try:
        ctx.repo.get_idea(idea_id)
        ctx.repo.get_source(source_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    match = ctx.repo.add_match(
        Match(
            idea_id=idea_id,
            source_id=source_id,
            score=1.0,
            status=MatchStatus.CONFIRMED,
            rationale="manually added",
        )
    )
    typer.echo(f"confirmed match [{match.id}]")


@app.command("gaps")
def gaps() -> None:
    """Ideas without confirmed sources; sources not matched to any idea."""
    ctx = _context()
    report = MatchService(ctx.repo, ctx.embedder.model_name).gap_report()
    typer.echo("ideas without confirmed support:")
    for idea_id in report.unsupported_ideas:
        typer.echo(f"  [{idea_id}] {ctx.repo.get_idea(idea_id).title}")
    typer.echo("sources not matched to any idea:")
    for source_id in report.orphan_sources:
        typer.echo(f"  [{source_id}] {ctx.repo.get_source(source_id).title}")


# -- summaries, bibliography, writing ------------------------------------------------


@app.command("summarise")
def summarise(
    source_id: Annotated[int | None, typer.Argument()] = None,
    all_sources: Annotated[bool, typer.Option("--all", help="Every source lacking one.")] = False,
    force: Annotated[bool, typer.Option("--force")] = False,
    override: Annotated[str | None, typer.Option(help="Store a hand-written summary.")] = None,
) -> None:
    """Generate a grounded, verified summary of a source (or store your own).

    With --all, summarise every source that has text but no summary yet;
    grounding failures are reported and skipped, never stored.
    """
    if all_sources == (source_id is not None):
        _fail("give either a SOURCE_ID or --all")
    if all_sources and override is not None:
        _fail("--override needs a specific SOURCE_ID")
    ctx = _context()
    service = SummariseService(ctx.repo, ctx.llm, max_source_chars=ctx.config.max_source_chars)
    if not all_sources:
        assert source_id is not None
        try:
            if override is not None:
                summary = service.override(source_id, override)
            else:
                summary = service.summarise(source_id, force=force)
        except (KeyError, LookupError, GroundingFailure) as exc:
            _fail(str(exc))
            return
        typer.echo(summary.text)
        for quote in summary.evidence:
            typer.echo(f'  evidence: "{quote}"')
        return

    done = skipped = failed = 0
    for source in ctx.repo.list_sources():
        assert source.id is not None
        if ctx.repo.get_summary(source.id) is not None and not force:
            skipped += 1
            continue
        if ctx.repo.get_source_text(source.id) is None:
            typer.echo(f"no text stored: [{source.id}] {source.title}")
            skipped += 1
            continue
        try:
            service.summarise(source.id, force=force)
        except GroundingFailure as exc:
            typer.secho(
                f"grounding failed: [{source.id}] {source.title}: {exc}",
                fg=typer.colors.RED,
                err=True,
            )
            failed += 1
            continue
        done += 1
        typer.echo(f"summarised [{source.id}] {source.title}")
    typer.echo(f"done: {done} summarised, {skipped} skipped, {failed} failed")
    if failed:
        raise typer.Exit(code=1)


@app.command("bib")
def bib(
    idea_id: Annotated[int | None, typer.Option("--idea")] = None,
    out: Annotated[Path | None, typer.Option("-o", "--out")] = None,
) -> None:
    """Export BibTeX for the whole library or one idea's confirmed sources."""
    ctx = _context()
    try:
        content = RelatedWorkService(ctx.repo).export_bib(idea_id)
    except KeyError as exc:
        _fail(str(exc))
        return
    if out:
        out.write_text(content)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(content, nl=False)


@app.command("related-work")
def related_work(
    idea_id: int,
    fmt: Annotated[str, typer.Option("--format", help="markdown | latex")] = "markdown",
    out: Annotated[Path | None, typer.Option("-o", "--out")] = None,
) -> None:
    """Citation-verified related-work skeleton for an idea."""
    if fmt not in ("markdown", "latex"):
        _fail("--format must be markdown or latex")
    ctx = _context()
    try:
        text = RelatedWorkService(ctx.repo).skeleton(idea_id, fmt)  # type: ignore[arg-type]
    except KeyError as exc:
        _fail(str(exc))
        return
    if out:
        out.write_text(text)
        typer.echo(f"wrote {out}")
    else:
        typer.echo(text)


@app.command("audit")
def audit(path: Path) -> None:
    """Check every citation in a draft against the library."""
    if not path.is_file():
        _fail(f"no such file: {path}")
    ctx = _context()
    report = AuditService(ctx.repo).audit_text(path.read_text())
    typer.echo(f"{len(report.used_keys)} citation keys used, {len(report.known_keys)} known")
    if report.unknown_keys:
        for key in report.unknown_keys:
            typer.secho(f"UNKNOWN: {key}", fg=typer.colors.RED)
        raise typer.Exit(code=1)
    typer.secho("all citations resolve to the library", fg=typer.colors.GREEN)


@app.command("graph")
def graph(
    out: Annotated[Path, typer.Option("-o", "--out")] = Path("mustrum-graph.html"),
    contacts: Annotated[bool, typer.Option(help="Include contact nodes.")] = True,
    open_browser: Annotated[bool, typer.Option("--open")] = False,
) -> None:
    """Export the knowledge graph as a self-contained HTML file."""
    from mustrum.graph.export import export_graph

    ctx = _context()
    out.write_text(export_graph(ctx.repo, include_contacts=contacts))
    typer.echo(f"wrote {out}")
    if open_browser:
        webbrowser.open(out.resolve().as_uri())


@app.command("ui")
def ui(
    port: Annotated[int, typer.Option("--port")] = 8765,
    open_browser: Annotated[bool, typer.Option("--open/--no-open")] = True,
) -> None:
    """Launch the local web GUI (a second adapter over the same services —
    everything it does is also available as CLI commands)."""
    import uvicorn

    from mustrum.web.api import create_app

    ctx = _context()
    web_app = create_app(ctx.repo, ctx.embedder, ctx.llm, ctx.config)
    url = f"http://127.0.0.1:{port}"
    typer.echo(f"Mustrum UI at {url} (Ctrl+C to stop)")
    if open_browser:
        webbrowser.open(url)
    uvicorn.run(web_app, host="127.0.0.1", port=port, log_level="warning")


@app.command("brainstorm")
def brainstorm(
    count: Annotated[int, typer.Option("--count", "-n")] = 3,
    focus: Annotated[str, typer.Option("--focus", help="Steer towards a topic.")] = "",
    save: Annotated[
        bool, typer.Option("--save", help="Save proposals as ideas (tagged 'brainstorm').")
    ] = False,
) -> None:
    """Generate NEW research idea proposals from your library. Creative mode:
    output is machine-generated and unverified — clearly labelled, never mixed
    with citation-bearing output."""
    from mustrum.core.services.brainstorm import (
        BRAINSTORM_TAG,
        BrainstormFailure,
        BrainstormService,
    )

    ctx = _context()
    service = BrainstormService(ctx.repo, ctx.llm)
    try:
        proposals = service.propose(count=count, focus=focus)
    except (LookupError, BrainstormFailure) as exc:
        _fail(str(exc))
        return
    typer.secho(
        "=== machine-generated brainstorm — creative output, NOT verified, cites nothing ===",
        fg=typer.colors.MAGENTA,
    )
    idea_service = IdeaService(ctx.repo, ctx.embedder)
    for number, proposal in enumerate(proposals, start=1):
        typer.echo(f"\n{number}. {proposal.title}")
        typer.echo(f"   {proposal.description}")
        if proposal.inspirations:
            typer.echo(f"   inspired by: {'; '.join(proposal.inspirations)}")
        if save:
            idea = idea_service.create(proposal.title, proposal.description)
            assert idea.id is not None
            ctx.repo.tag(EntityKind.IDEA, idea.id, BRAINSTORM_TAG)
            typer.echo(f"   saved as idea [{idea.id}] (tagged '{BRAINSTORM_TAG}')")
    if not save:
        typer.echo("\n(re-run with --save to keep them, or: mustrum idea new ...)")


@app.command("export")
def export_cmd(
    directory: Path,
    force: Annotated[
        bool, typer.Option("--force", help="Write into a non-empty directory.")
    ] = False,
) -> None:
    """Export the whole library as plain files (JSON + texts + .bib +
    Markdown views) — git-versionable, tool-independent backup."""
    from mustrum.core.services.backup import BackupService

    if directory.exists() and any(directory.iterdir()) and not force:
        _fail(f"{directory} is not empty — use --force to write into it")
    ctx = _context()
    bundle = BackupService(ctx.repo, ctx.embedder).export_data()
    for relative, content in bundle.items():
        target = directory / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    typer.echo(f"exported {len(bundle)} files to {directory}")


@app.command("restore")
def restore_cmd(directory: Path) -> None:
    """Restore an export into an EMPTY database (set MUSTRUM_DB or db_path
    first). Embeddings are recomputed, so Ollama must be running."""
    from mustrum.core.services.backup import BackupError, BackupService

    if not directory.is_dir():
        _fail(f"no such directory: {directory}")
    bundle = {
        str(path.relative_to(directory)): path.read_text(encoding="utf-8")
        for path in directory.rglob("*")
        if path.is_file()
    }
    ctx = _context()
    try:
        counts = BackupService(ctx.repo, ctx.embedder).import_data(bundle)
    except BackupError as exc:
        _fail(str(exc))
        return
    typer.echo(
        f"restored {counts['sources']} sources, {counts['ideas']} ideas, "
        f"{counts['matches']} matches, {counts['contacts']} contacts"
    )


_CONFIG_TEMPLATE = """\
# Mustrum configuration — local to this machine, never part of any repo.

# Your ENTIRE library (sources, texts, summaries, ideas, matches, BibTeX,
# contacts, embeddings) lives in one SQLite file. Point db_path into a synced
# folder to keep it in iCloud or OneDrive, e.g.:
#   db_path = "~/Library/Mobile Documents/com~apple~CloudDocs/mustrum/mustrum.db"
#   db_path = "~/OneDrive/mustrum/mustrum.db"
# Never run mustrum from two machines against the same synced file at once.
# Original files (ingested/fetched PDFs) are archived in a `files/` directory
# next to the database, so backing up the db_path folder captures everything.
#db_path = "~/.mustrum/mustrum.db"

#ollama_url = "http://localhost:11434"
#llm_model = "qwen3:30b"
#embed_model = "nomic-embed-text"
#max_source_chars = 16000
#num_ctx = 16384

# Contact e-mail for the Unpaywall API — enables open-access PDF download
# when ingesting by DOI. Stays on this machine.
#unpaywall_email = "you@example.org"
"""


@app.command("config")
def config_cmd(
    init: Annotated[
        bool, typer.Option("--init", help="Write a commented template config file.")
    ] = False,
    path: Annotated[
        Path | None, typer.Option("--path", help="Config file location (default: standard path).")
    ] = None,
) -> None:
    """Show the effective configuration (or create a template with --init)."""
    from mustrum.config import DEFAULT_CONFIG_PATH

    config_path = path or DEFAULT_CONFIG_PATH
    if init:
        if config_path.exists():
            _fail(f"{config_path} already exists — edit it directly")
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(_CONFIG_TEMPLATE)
        typer.echo(f"wrote {config_path}")
        return
    config = load_config(path)
    state = "present" if config_path.is_file() else "absent — defaults in effect"
    typer.echo(f"config file:      {config_path} ({state})")
    typer.echo(f"db_path:          {config.db_path}")
    typer.echo(f"files_dir:        {config.files_dir} (originals archive, follows db_path)")
    typer.echo(f"ollama_url:       {config.ollama_url}")
    typer.echo(f"llm_model:        {config.llm_model}")
    typer.echo(f"embed_model:      {config.embed_model}")
    typer.echo(f"max_source_chars: {config.max_source_chars}")
    typer.echo(f"num_ctx:          {config.num_ctx}")
    typer.echo(f"unpaywall_email:  {config.unpaywall_email or '(unset — OA PDF lookup disabled)'}")


@app.command("search")
def search(query: str, limit: int = 20) -> None:
    """Full-text search across sources, ideas, summaries, and contacts."""
    ctx = _context()
    for hit in ctx.repo.search(query, limit=limit):
        typer.echo(f"{hit.entity.value} [{hit.ref_id}]: {hit.snippet}")


# -- contacts -------------------------------------------------------------------------


@contact_app.command("add")
def contact_add(
    name: str,
    kind: ContactKind = ContactKind.PERSON,
    affiliation: str = "",
    email: str = "",
    url: str = "",
    notes: str = "",
) -> None:
    ctx = _context()
    contact = ctx.repo.add_contact(
        Contact(name=name, kind=kind, affiliation=affiliation, email=email, url=url, notes=notes)
    )
    typer.echo(f"created contact [{contact.id}] {contact.name}")


@contact_app.command("list")
def contact_list() -> None:
    ctx = _context()
    for contact in ctx.repo.list_contacts():
        extra = f" — {contact.affiliation}" if contact.affiliation else ""
        typer.echo(f"[{contact.id}] {contact.name} ({contact.kind.value}){extra}")


@contact_app.command("link")
def contact_link(
    contact_id: int,
    why: Annotated[str, typer.Option("--why", help="Why is this contact relevant?")],
    idea_id: Annotated[int | None, typer.Option("--idea")] = None,
    source_id: Annotated[int | None, typer.Option("--source")] = None,
) -> None:
    ctx = _context()
    try:
        ctx.repo.get_contact(contact_id)
        if idea_id is not None:
            ctx.repo.get_idea(idea_id)
        if source_id is not None:
            ctx.repo.get_source(source_id)
        link = ContactLink(contact_id=contact_id, why=why, idea_id=idea_id, source_id=source_id)
    except (KeyError, ValueError) as exc:
        _fail(str(exc))
        return
    ctx.repo.add_contact_link(link)
    typer.echo("linked")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
