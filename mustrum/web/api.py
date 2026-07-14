"""Web GUI adapter: a thin JSON API over the core services (FastAPI).

Strictly a driving adapter like the CLI — no business logic lives here.
Everything the GUI does is also possible via `mustrum <command>`.
"""

from __future__ import annotations

import dataclasses
import sys
from importlib import resources
from typing import Any

from fastapi import FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

from mustrum.adapters.archive import archive_original, archived_file, delete_archived
from mustrum.config import Config, save_library_config
from mustrum.core.models import (
    Contact,
    ContactKind,
    ContactLink,
    EntityKind,
    FieldOrigin,
    MatchStatus,
    ReadingStatus,
    Source,
    SourceKind,
)
from mustrum.core.ports import EmbeddingProvider, LLMProvider, StorageRepo
from mustrum.core.services.audit import AuditService
from mustrum.core.services.brainstorm import BRAINSTORM_TAG, BrainstormFailure, BrainstormService
from mustrum.core.services.chat import ChatSession
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import DuplicateSourceError, IngestService
from mustrum.core.services.match import MatchService
from mustrum.core.services.query import QueryFailure, QueryService
from mustrum.core.services.rationale import RationaleFailure, RationaleService
from mustrum.core.services.relatedwork import RelatedWorkService
from mustrum.core.services.summarise import GroundingFailure, SummariseService


class IngestIdPayload(BaseModel):
    identifier: str


class IdeaPayload(BaseModel):
    title: str
    text: str


class TextPayload(BaseModel):
    text: str


class SuggestPayload(BaseModel):
    threshold: float = 0.35
    limit: int = 20


class BrainstormPayload(BaseModel):
    count: int = 3
    focus: str = ""


class ChatPayload(BaseModel):
    question: str


class BrainstormIdeaPayload(BaseModel):
    title: str
    description: str


class BrainstormSavePayload(BaseModel):
    ideas: list[BrainstormIdeaPayload]


class MetadataPayload(BaseModel):
    authors: list[str] | None = None
    year: int | None = None


class SettingsPayload(BaseModel):
    llm_provider: str | None = None
    ollama_url: str | None = None
    llm_model: str | None = None
    embed_model: str | None = None
    anthropic_model: str | None = None
    anthropic_max_tokens: int | None = None
    max_source_chars: int | None = None
    num_ctx: int | None = None
    unpaywall_email: str | None = None


class ContactPayload(BaseModel):
    name: str
    kind: str = "person"
    affiliation: str = ""
    email: str = ""
    notes: str = ""


class ContactLinkPayload(BaseModel):
    contact_id: int
    why: str


def _settings_json(config: Config) -> dict[str, Any]:
    return {
        "llm_provider": config.llm_provider,
        "ollama_url": config.ollama_url,
        "llm_model": config.llm_model,
        "embed_model": config.embed_model,
        "anthropic_model": config.anthropic_model,
        "anthropic_max_tokens": config.anthropic_max_tokens,
        "max_source_chars": config.max_source_chars,
        "num_ctx": config.num_ctx,
        "unpaywall_email": config.unpaywall_email,
        "db_path": str(config.db_path),
        "files_dir": str(config.files_dir),
        "library_config_path": str(config.library_config_path),
        "library_config_exists": config.library_config_path.is_file(),
    }


def _source_json(repo: StorageRepo, source: Source) -> dict[str, Any]:
    assert source.id is not None
    summary = repo.get_summary(source.id)
    bib = repo.get_bib_entry(source.id)
    text = repo.get_source_text(source.id)
    return {
        "id": source.id,
        "title": source.title,
        "authors": list(source.authors),
        "year": source.year,
        "kind": source.kind.value,
        "doi": source.doi,
        "arxiv_id": source.arxiv_id,
        "reading_status": source.reading_status.value,
        "notes": source.notes,
        "tags": sorted(repo.tags_for(EntityKind.SOURCE, source.id)),
        "citation_key": bib.citation_key if bib else None,
        "has_text": text is not None,
        "text_kind": text.extraction_method if text else None,
        "file_name": source.file_path,
        "summary": (
            {
                "text": summary.text,
                "evidence": list(summary.evidence),
                "model": summary.model,
                "user_override": summary.user_override,
            }
            if summary
            else None
        ),
    }


def _contact_json(contact: Contact) -> dict[str, Any]:
    return {
        "id": contact.id,
        "name": contact.name,
        "kind": contact.kind.value,
        "affiliation": contact.affiliation,
        "email": contact.email,
        "notes": contact.notes,
    }


def create_app(
    repo: StorageRepo,
    embedder: EmbeddingProvider,
    llm: LLMProvider,
    config: Config,
) -> FastAPI:
    app = FastAPI(title="Mustrum", docs_url=None, redoc_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def log_http_errors(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        """Every failed API call leaves a durable line in the `mustrum ui`
        terminal (E11-5) — the GUI flash is no longer the only record."""
        print(
            f"[mustrum ui] {request.method} {request.url.path} -> {exc.status_code}: {exc.detail}",
            file=sys.stderr,
        )
        return JSONResponse({"detail": exc.detail}, status_code=exc.status_code)

    def summariser() -> SummariseService:
        return SummariseService(repo, llm, max_source_chars=config.max_source_chars)

    # one session for the lifetime of this running `mustrum ui` process —
    # unlike the other services above (rebuilt fresh per request), chat
    # state must persist *across* requests, and this is a single-user local
    # app with no auth/multi-session concept anywhere else to hang it off
    chat_session = ChatSession(
        QueryService(
            repo, llm, embedder, config.embed_model, max_source_chars=config.max_source_chars
        )
    )

    # -- pages ------------------------------------------------------------

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return resources.files("mustrum.web").joinpath("static/index.html").read_text()

    @app.get("/graph", response_class=HTMLResponse)
    async def graph_page() -> str:
        from mustrum.graph.export import export_graph

        return export_graph(repo)

    # -- sources -------------------------------------------------------------

    @app.get("/api/sources")
    async def list_sources() -> list[dict[str, Any]]:
        return [_source_json(repo, s) for s in repo.list_sources()]

    @app.get("/api/sources/{source_id}")
    async def get_source(source_id: int) -> dict[str, Any]:
        try:
            return _source_json(repo, repo.get_source(source_id))
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.delete("/api/sources/{source_id}")
    async def delete_source(source_id: int) -> dict[str, Any]:
        try:
            source = repo.get_source(source_id)
            repo.delete_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        delete_archived(config.files_dir, source)
        return {"ok": True}

    @app.get("/api/sources/{source_id}/file")
    async def source_file(source_id: int) -> FileResponse:
        try:
            source = repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        path = archived_file(config.files_dir, source)
        if path is None:
            raise HTTPException(404, f"no archived file for source {source_id}")
        media = "application/pdf" if path.suffix == ".pdf" else "text/plain"
        # inline: the GUI's "Open PDF" opens a tab that should display, not download
        return FileResponse(
            path, media_type=media, filename=path.name, content_disposition_type="inline"
        )

    @app.post("/api/sources/{source_id}/attach")
    async def attach_file(source_id: int, file: UploadFile) -> dict[str, Any]:
        """GUI counterpart of `source attach`: store the full text of a
        manually-downloaded original and archive the file (E1-11/E11-3)."""
        from mustrum.adapters.pdf import extract_pdf_bytes

        name = file.filename or "upload"
        data = await file.read()
        if name.lower().endswith(".pdf"):
            text = extract_pdf_bytes(data)
            method = "pymupdf"
        else:
            text = data.decode("utf-8", errors="replace")
            method = "plaintext"
        had_summary = repo.get_summary(source_id) is not None
        try:
            IngestService(repo, embedder).attach_full_text(source_id, text, method)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        suffix = "." + name.rsplit(".", 1)[1].lower() if "." in name else ".txt"
        archive_original(repo, config.files_dir, repo.get_source(source_id), data, suffix)
        return {"ok": True, "summary_invalidated": had_summary}

    @app.post("/api/sources/{source_id}/status/{status}")
    async def set_status(source_id: int, status: str) -> dict[str, Any]:
        try:
            repo.set_reading_status(source_id, ReadingStatus(status))
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/sources/{source_id}/title")
    async def rename_source(source_id: int, payload: TextPayload) -> dict[str, Any]:
        import dataclasses

        from mustrum.core.normalize import title_hash

        title = payload.text.strip()
        if not title:
            raise HTTPException(400, "title must not be empty")
        try:
            source = repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        clash = repo.find_source_by_title_hash(title_hash(title))
        if clash is not None and clash.id != source_id:
            raise HTTPException(409, f"another source already has this title: {clash.title}")
        repo.update_source(dataclasses.replace(source, title=title))
        return {"ok": True}

    @app.post("/api/sources/{source_id}/metadata")
    async def edit_metadata(source_id: int, payload: MetadataPayload) -> dict[str, Any]:
        """GUI counterpart of `source edit` (E8-6): manual authors/year for
        papers whose venue has no DOIs (e.g. CEUR-WS)."""
        if payload.authors is None and payload.year is None:
            raise HTTPException(400, "nothing to change — give authors and/or year")
        try:
            source = repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        provenance = dict(source.provenance)
        if payload.authors is not None:
            authors = tuple(a.strip() for a in payload.authors if a.strip())
            source = dataclasses.replace(source, authors=authors)
            provenance["authors"] = FieldOrigin.USER
        if payload.year is not None:
            source = dataclasses.replace(source, year=payload.year)
            provenance["year"] = FieldOrigin.USER
        repo.update_source(dataclasses.replace(source, provenance=tuple(provenance.items())))
        return {"ok": True}

    @app.post("/api/sources/{source_id}/tags")
    async def add_tag(source_id: int, payload: TextPayload) -> dict[str, Any]:
        tag = payload.text.strip()
        if not tag:
            raise HTTPException(400, "tag must not be empty")
        try:
            repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        repo.tag(EntityKind.SOURCE, source_id, tag)
        return {"ok": True}

    @app.delete("/api/sources/{source_id}/tags/{tag}")
    async def remove_source_tag(source_id: int, tag: str) -> dict[str, Any]:
        try:
            repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        repo.untag(EntityKind.SOURCE, source_id, tag)
        return {"ok": True}

    @app.post("/api/sources/{source_id}/contacts")
    async def link_source_contact(source_id: int, payload: ContactLinkPayload) -> dict[str, Any]:
        try:
            repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _add_contact_link(source_id=source_id, idea_id=None, payload=payload)

    @app.get("/api/sources/{source_id}/contacts")
    async def source_contacts(source_id: int) -> list[dict[str, Any]]:
        try:
            repo.get_source(source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _contact_links_json(repo.list_contact_links(source_id=source_id))

    @app.post("/api/sources/{source_id}/enrich")
    async def enrich(source_id: int) -> dict[str, Any]:
        from mustrum.adapters.enrich import enrich_source

        try:
            result = enrich_source(repo, embedder, source_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        except Exception as exc:
            raise HTTPException(502, f"Crossref lookup failed: {exc}") from exc
        return {"enriched": result.enriched, "message": result.message}

    @app.post("/api/sources/{source_id}/notes")
    async def set_notes(source_id: int, payload: TextPayload) -> dict[str, Any]:
        try:
            repo.set_source_notes(source_id, payload.text)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/sources/{source_id}/summarise")
    async def summarise(source_id: int, force: bool = False) -> dict[str, Any]:
        try:
            summary = summariser().summarise(source_id, force=force)
        except (KeyError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc
        except GroundingFailure as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"text": summary.text, "evidence": list(summary.evidence)}

    # -- ingestion ---------------------------------------------------------------

    def _ingest_result(result: Any) -> dict[str, Any]:
        return {
            "created": result.created,
            "merged": result.merged,
            "source": _source_json(repo, result.source),
        }

    def _ingest_fetched(kind: str, identifier: str) -> dict[str, Any]:
        from mustrum.adapters.oa import fetch_full_text

        try:
            if kind == "arxiv":
                from mustrum.adapters.arxiv import ArxivFetcher

                meta = ArxivFetcher().fetch(identifier)
            else:
                from mustrum.adapters.crossref import CrossrefFetcher

                meta = CrossrefFetcher().fetch(identifier)
        except (LookupError, ValueError) as exc:
            raise HTTPException(404, str(exc)) from exc
        full_text = fetch_full_text(meta, config.unpaywall_email)
        try:
            result = IngestService(repo, embedder).ingest_fetched(
                meta, on_duplicate="merge", full_text=full_text.text
            )
        except DuplicateSourceError as exc:
            raise HTTPException(409, str(exc)) from exc
        if full_text.pdf_bytes and result.source.file_path is None and result.source.id is not None:
            archived = archive_original(
                repo, config.files_dir, result.source, full_text.pdf_bytes, ".pdf"
            )
            result = dataclasses.replace(result, source=archived)
        return {**_ingest_result(result), "notes": full_text.notes}

    @app.post("/api/ingest/arxiv")
    async def ingest_arxiv(payload: IngestIdPayload) -> dict[str, Any]:
        return _ingest_fetched("arxiv", payload.identifier)

    @app.post("/api/ingest/doi")
    async def ingest_doi(payload: IngestIdPayload) -> dict[str, Any]:
        return _ingest_fetched("doi", payload.identifier)

    @app.post("/api/ingest/file")
    async def ingest_file(file: UploadFile) -> dict[str, Any]:
        from mustrum.adapters.pdf import extract_pdf_bytes, pdf_metadata_title_bytes

        name = file.filename or "upload"
        data = await file.read()
        title = None
        if name.lower().endswith(".pdf"):
            text = extract_pdf_bytes(data)
            method = "pymupdf"
            title = pdf_metadata_title_bytes(data)
        else:
            text = data.decode("utf-8", errors="replace")
            method = "plaintext"
        title = title or name.rsplit(".", 1)[0]
        try:
            result = IngestService(repo, embedder).ingest_document(
                title=title,
                text=text,
                extraction_method=method,
                kind=SourceKind.PAPER,
                on_duplicate="skip",
            )
        except DuplicateSourceError as exc:  # pragma: no cover - skip mode
            raise HTTPException(409, str(exc)) from exc
        if result.source.file_path is None and result.source.id is not None:
            suffix = "." + name.rsplit(".", 1)[1].lower() if "." in name else ".txt"
            archived = archive_original(repo, config.files_dir, result.source, data, suffix)
            result = dataclasses.replace(result, source=archived)
        return _ingest_result(result)

    # -- ideas ---------------------------------------------------------------------

    @app.get("/api/ideas")
    async def list_ideas() -> list[dict[str, Any]]:
        out = []
        for idea in repo.list_ideas():
            assert idea.id is not None
            version = repo.latest_idea_version(idea.id)
            out.append(
                {
                    "id": idea.id,
                    "title": idea.title,
                    "text": version.text if version else "",
                    "versions": len(repo.get_idea_versions(idea.id)),
                    "tags": sorted(repo.tags_for(EntityKind.IDEA, idea.id)),
                }
            )
        return out

    @app.post("/api/ideas")
    async def create_idea(payload: IdeaPayload) -> dict[str, Any]:
        idea = IdeaService(repo, embedder).create(payload.title, payload.text)
        return {"id": idea.id, "title": idea.title}

    @app.delete("/api/ideas/{idea_id}")
    async def delete_idea(idea_id: int) -> dict[str, Any]:
        try:
            repo.delete_idea(idea_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/ideas/{idea_id}/revise")
    async def revise_idea(idea_id: int, payload: TextPayload) -> dict[str, Any]:
        try:
            IdeaService(repo, embedder).revise(idea_id, payload.text)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"ok": True}

    @app.post("/api/ideas/{idea_id}/tags")
    async def add_idea_tag(idea_id: int, payload: TextPayload) -> dict[str, Any]:
        tag = payload.text.strip()
        if not tag:
            raise HTTPException(400, "tag must not be empty")
        try:
            repo.get_idea(idea_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        repo.tag(EntityKind.IDEA, idea_id, tag)
        return {"ok": True}

    @app.delete("/api/ideas/{idea_id}/tags/{tag}")
    async def remove_idea_tag(idea_id: int, tag: str) -> dict[str, Any]:
        try:
            repo.get_idea(idea_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        repo.untag(EntityKind.IDEA, idea_id, tag)
        return {"ok": True}

    @app.post("/api/ideas/{idea_id}/contacts")
    async def link_idea_contact(idea_id: int, payload: ContactLinkPayload) -> dict[str, Any]:
        try:
            repo.get_idea(idea_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _add_contact_link(source_id=None, idea_id=idea_id, payload=payload)

    @app.get("/api/ideas/{idea_id}/contacts")
    async def idea_contacts(idea_id: int) -> list[dict[str, Any]]:
        try:
            repo.get_idea(idea_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return _contact_links_json(repo.list_contact_links(idea_id=idea_id))

    # -- matching -------------------------------------------------------------------

    def _match_json(match: Any) -> dict[str, Any]:
        source = repo.get_source(match.source_id)
        return {
            "id": match.id,
            "source_id": match.source_id,
            "source_title": source.title,
            "score": round(match.score, 3),
            "status": match.status.value,
            "rationale": match.rationale,
            "quotes": list(match.quotes),
        }

    @app.get("/api/ideas/{idea_id}/matches")
    async def list_matches(idea_id: int) -> list[dict[str, Any]]:
        return [_match_json(m) for m in repo.list_matches(idea_id)]

    @app.post("/api/ideas/{idea_id}/suggest")
    async def suggest(idea_id: int, payload: SuggestPayload) -> list[dict[str, Any]]:
        service = MatchService(repo, embedder.model_name, threshold=payload.threshold)
        try:
            return [_match_json(m) for m in service.suggest(idea_id, limit=payload.limit)]
        except (KeyError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/matches/{match_id}/{action}")
    async def match_action(match_id: int, action: str) -> dict[str, Any]:
        service = MatchService(repo, embedder.model_name)
        try:
            if action == "confirm":
                service.confirm(match_id)
            elif action == "reject":
                service.reject(match_id)
            elif action == "explain":
                rationale = RationaleService(repo, llm, max_source_chars=config.max_source_chars)
                return _match_json(rationale.explain(match_id))
            else:
                raise HTTPException(400, f"unknown action {action!r}")
        except (KeyError, LookupError) as exc:
            raise HTTPException(404, str(exc)) from exc
        except RationaleFailure as exc:
            raise HTTPException(422, str(exc)) from exc
        return {"ok": True}

    # -- writing, search, misc ---------------------------------------------------------

    @app.get("/api/ideas/{idea_id}/related-work")
    async def related_work(idea_id: int, fmt: str = "markdown") -> dict[str, Any]:
        if fmt not in ("markdown", "latex"):
            raise HTTPException(400, "fmt must be markdown or latex")
        try:
            text = RelatedWorkService(repo).skeleton(idea_id, fmt)  # type: ignore[arg-type]
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"text": text}

    @app.get("/api/bib")
    async def bib(idea_id: int | None = None) -> dict[str, Any]:
        try:
            return {"text": RelatedWorkService(repo).export_bib(idea_id)}
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc

    @app.post("/api/audit")
    async def audit_draft(file: UploadFile) -> dict[str, Any]:
        """GUI counterpart of `mustrum audit` (FR-5.5): upload a draft
        .tex/.md and check every citation key against the library."""
        data = await file.read()
        text = data.decode("utf-8", errors="replace")
        report = AuditService(repo).audit_text(text)
        return {
            "ok": report.ok,
            "used_keys": list(report.used_keys),
            "unknown_keys": list(report.unknown_keys),
            "known_keys": list(report.known_keys),
        }

    @app.get("/api/search")
    async def search(q: str) -> list[dict[str, Any]]:
        return [
            {"entity": hit.entity.value, "ref_id": hit.ref_id, "snippet": hit.snippet}
            for hit in repo.search(q)
        ]

    @app.get("/api/gaps")
    async def gaps() -> dict[str, Any]:
        report = MatchService(repo, embedder.model_name).gap_report()
        return {
            "unsupported_ideas": [
                {"id": i, "title": repo.get_idea(i).title} for i in report.unsupported_ideas
            ],
            "orphan_sources": [
                {"id": s, "title": repo.get_source(s).title} for s in report.orphan_sources
            ],
        }

    @app.post("/api/brainstorm")
    async def brainstorm(payload: BrainstormPayload) -> dict[str, Any]:
        """Generate proposals only — nothing is stored here (E11-7). The
        user reviews the list in the GUI and picks which to keep via
        POST /api/brainstorm/save."""
        service = BrainstormService(repo, llm)
        try:
            proposals = service.propose(count=payload.count, focus=payload.focus)
        except LookupError as exc:
            raise HTTPException(404, str(exc)) from exc
        except BrainstormFailure as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "proposals": [
                {
                    "title": p.title,
                    "description": p.description,
                    "inspirations": list(p.inspirations),
                }
                for p in proposals
            ]
        }

    @app.post("/api/brainstorm/save")
    async def save_brainstorm(payload: BrainstormSavePayload) -> dict[str, Any]:
        """Save the proposals the user picked after reviewing the generated
        list (E11-7), tagged 'brainstorm' same as before."""
        if not payload.ideas:
            raise HTTPException(400, "no ideas selected")
        idea_service = IdeaService(repo, embedder)
        saved = []
        for item in payload.ideas:
            idea = idea_service.create(item.title, item.description)
            assert idea.id is not None
            repo.tag(EntityKind.IDEA, idea.id, BRAINSTORM_TAG)
            saved.append({"id": idea.id, "title": idea.title})
        return {"saved": saved}

    @app.post("/api/chat")
    async def chat(payload: ChatPayload) -> dict[str, Any]:
        """One grounded turn in the running GUI chat session (E13-2). Prior
        turns shape retrieval/interpretation only — see ADR-18 — every
        answer is grounded exactly like a single QueryService.ask() call."""
        try:
            answer = chat_session.ask(payload.question)
        except QueryFailure as exc:
            raise HTTPException(422, str(exc)) from exc
        return {
            "answer": answer.answer,
            "found": answer.found,
            "evidence": [{"source_id": e.source_id, "quote": e.quote} for e in answer.evidence],
            "considered_source_ids": list(answer.considered_source_ids),
        }

    @app.post("/api/chat/reset")
    async def reset_chat() -> dict[str, Any]:
        chat_session.reset()
        return {"reset": True}

    def _contact_links_json(links: list[Any]) -> list[dict[str, Any]]:
        return [
            {**_contact_json(repo.get_contact(link.contact_id)), "why": link.why} for link in links
        ]

    def _add_contact_link(
        *, source_id: int | None, idea_id: int | None, payload: ContactLinkPayload
    ) -> dict[str, Any]:
        """Shared by the source- and idea-contact endpoints (GUI counterpart
        of `mustrum contact link`, FR-7.2)."""
        why = payload.why.strip()
        if not why:
            raise HTTPException(400, "why must not be empty")
        try:
            repo.get_contact(payload.contact_id)
        except KeyError as exc:
            raise HTTPException(404, str(exc)) from exc
        repo.add_contact_link(
            ContactLink(
                contact_id=payload.contact_id, why=why, idea_id=idea_id, source_id=source_id
            )
        )
        return {"ok": True}

    @app.get("/api/contacts")
    async def contacts() -> list[dict[str, Any]]:
        return [_contact_json(c) for c in repo.list_contacts()]

    @app.post("/api/contacts")
    async def add_contact(payload: ContactPayload) -> dict[str, Any]:
        try:
            kind = ContactKind(payload.kind)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        contact = repo.add_contact(
            Contact(
                name=payload.name,
                kind=kind,
                affiliation=payload.affiliation,
                email=payload.email,
                notes=payload.notes,
            )
        )
        return {"id": contact.id}

    @app.get("/api/ollama/models")
    async def ollama_models(url: str | None = None) -> dict[str, Any]:
        """Installed models at `url` (defaults to the library's configured
        ollama_url), for the Settings dropdowns (E12-2). Never raises: the
        settings form must stay usable when Ollama is unreachable — that's
        often exactly what the user is here to fix."""
        from mustrum.adapters.ollama import list_models

        target = url or config.ollama_url
        try:
            return {"models": list_models(target), "error": None}
        except Exception as exc:
            return {"models": [], "error": str(exc)}

    @app.get("/api/settings")
    async def get_settings() -> dict[str, Any]:
        """The settings this *running* process is actually using — may lag
        a hand-edited file on disk until the next `mustrum ui` start."""
        return _settings_json(config)

    @app.post("/api/settings")
    async def update_settings(payload: SettingsPayload) -> dict[str, Any]:
        """Writes the library config next to the database (ADR-16). Does
        NOT reconfigure this running process — the Ollama/Anthropic clients
        and max_source_chars were built at startup; restart to pick this up."""
        if payload.llm_provider is not None and payload.llm_provider not in (
            "ollama",
            "anthropic",
        ):
            raise HTTPException(
                400, f"llm_provider must be 'ollama' or 'anthropic', got {payload.llm_provider!r}"
            )
        updates = {k: v for k, v in payload.model_dump().items() if v is not None}
        if not updates:
            raise HTTPException(400, "nothing to update — give at least one field")
        updated = save_library_config(config, updates)
        return {**_settings_json(updated), "restart_required": True}

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        matches = repo.list_matches()
        return {
            "sources": len(repo.list_sources()),
            "ideas": len(repo.list_ideas()),
            "matches": len(matches),
            "confirmed": sum(1 for m in matches if m.status == MatchStatus.CONFIRMED),
            "contacts": len(repo.list_contacts()),
            "llm_model": llm.model_name,
            "embed_model": embedder.model_name,
        }

    return app
