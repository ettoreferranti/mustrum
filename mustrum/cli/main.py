"""Mustrum CLI: the driving adapter. All logic lives in core services."""

from __future__ import annotations

import os
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

import typer

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
    verb = (
        "merged into"
        if result.merged
        else ("already in library:" if not result.created else "ingested")
    )
    typer.echo(f"{verb} ", nl=False)
    _print_source(result.source)


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
    for pdf in pdfs:
        extractor = extractor_for(pdf)
        try:
            result = service.ingest_document(
                title=pdf.stem,
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
        if result.created:
            ingested += 1
            typer.echo(f"ingested [{result.source.id}] {result.source.title}")
        else:
            skipped += 1
            typer.echo(f"skipped (already known): {pdf.name}")
    typer.echo(f"done: {ingested} ingested, {skipped} skipped, {failed} failed")
    if failed:
        raise typer.Exit(code=1)


def _fetch_full_text(ctx: Context, meta: FetchedMetadata) -> str:
    """Download and extract the paper's PDF if any candidate URL works.

    Candidates, in order: arXiv (always open), an Unpaywall open-access copy,
    then the publisher's Crossref full-text links — the last succeed only on
    networks with subscription access (e.g. a university network). '' means
    fall back to the abstract.
    """
    from mustrum.adapters.oa import OpenAccessClient, arxiv_pdf_url
    from mustrum.adapters.pdf import extract_pdf_bytes

    client = OpenAccessClient(email=ctx.config.unpaywall_email or "unused@localhost")
    candidates: list[str] = []
    if meta.arxiv_id:
        candidates.append(arxiv_pdf_url(meta.arxiv_id))
    if meta.doi and not meta.arxiv_id:
        if ctx.config.unpaywall_email:
            try:
                if found := client.find_pdf_url(meta.doi):
                    candidates.append(found)
            except Exception as exc:
                typer.secho(f"Unpaywall lookup failed ({exc})", fg=typer.colors.YELLOW)
        else:
            typer.secho(
                "no unpaywall_email configured — skipping open-access lookup "
                "(set it in ~/.config/mustrum/config.toml)",
                fg=typer.colors.YELLOW,
            )
    candidates.extend(meta.pdf_urls)

    for url in candidates:
        try:
            text = extract_pdf_bytes(client.download_pdf(url))
        except Exception as exc:
            typer.secho(f"PDF fetch failed from {url} ({exc})", fg=typer.colors.YELLOW)
            continue
        typer.echo(f"fetched full text from {url}")
        return text
    if candidates or meta.doi:
        typer.secho("no downloadable PDF — storing abstract only", fg=typer.colors.YELLOW)
    return ""


def _ingest_fetched(
    identifier: str, fetcher: MetadataFetcher, on_duplicate: str, fetch_pdf: bool
) -> None:
    ctx = _context()
    try:
        meta = fetcher.fetch(identifier)
    except (LookupError, ValueError) as exc:
        _fail(str(exc))
        return
    full_text = _fetch_full_text(ctx, meta) if fetch_pdf else ""
    try:
        result = IngestService(ctx.repo, ctx.embedder).ingest_fetched(
            meta,
            on_duplicate=on_duplicate,  # type: ignore[arg-type]
            full_text=full_text,
        )
    except DuplicateSourceError as exc:
        _fail(f"{exc}\nUse --on-duplicate skip|merge to resolve.")
        return
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
    typer.echo(f"attached full text to [{source_id}]")
    if had_summary:
        typer.secho(
            f"summary invalidated — run: mustrum summarise {source_id}", fg=typer.colors.YELLOW
        )


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


@match_app.command("suggest")
def match_suggest(idea_id: int, limit: int = 20, threshold: float = 0.35) -> None:
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


@match_app.command("list")
def match_list(idea_id: int, status: MatchStatus | None = None) -> None:
    ctx = _context()
    for match in ctx.repo.list_matches(idea_id, status):
        source = ctx.repo.get_source(match.source_id)
        typer.echo(
            f"[{match.id}] {match.status.value} score {match.score:.2f}: "
            f"[{source.id}] {source.title}"
        )


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


_CONFIG_TEMPLATE = """\
# Mustrum configuration — local to this machine, never part of any repo.

# Your ENTIRE library (sources, texts, summaries, ideas, matches, BibTeX,
# contacts, embeddings) lives in one SQLite file. Point db_path into a synced
# folder to keep it in iCloud or OneDrive, e.g.:
#   db_path = "~/Library/Mobile Documents/com~apple~CloudDocs/mustrum/mustrum.db"
#   db_path = "~/OneDrive/mustrum/mustrum.db"
# Never run mustrum from two machines against the same synced file at once.
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
