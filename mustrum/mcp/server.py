"""MCP server adapter (E13-3, ADR-19): read-only library access for
external MCP clients (e.g. Claude Desktop). A driving adapter like `cli/`
and `web/` — depends on core/StorageRepo, not the other way round.

Deliberately exposes raw, faithful readouts of stored records only — no
LLM call anywhere in this module. "Same grounding guarantees as chat/CLI"
here means nothing is ever synthesised, so there is nothing to hallucinate.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from mustrum.core.models import EntityKind
from mustrum.core.ports import StorageRepo
from mustrum.core.services.relatedwork import RelatedWorkService


def search_library(repo: StorageRepo, query: str, limit: int = 20) -> list[dict[str, Any]]:
    """FTS5 search across sources/ideas/contacts — same index as `mustrum
    search` / `GET /api/search`. No new retrieval logic."""
    return [
        {"entity": hit.entity.value, "ref_id": hit.ref_id, "snippet": hit.snippet}
        for hit in repo.search(query, limit=limit)
    ]


def get_source(repo: StorageRepo, source_id: int) -> dict[str, Any]:
    """Full source record: metadata, verified summary (if any), citation
    key, tags — same shape as web/api.py's `_source_json`."""
    try:
        source = repo.get_source(source_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    assert source.id is not None
    summary = repo.get_summary(source.id)
    bib = repo.get_bib_entry(source.id)
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


def get_idea(repo: StorageRepo, idea_id: int) -> dict[str, Any]:
    """Idea title, current (latest) version text, tags, links to other
    ideas."""
    try:
        idea = repo.get_idea(idea_id)
    except KeyError as exc:
        raise ValueError(str(exc)) from exc
    assert idea.id is not None
    version = repo.latest_idea_version(idea.id)
    links = repo.list_idea_links(idea.id)
    return {
        "id": idea.id,
        "title": idea.title,
        "text": version.text if version else "",
        "tags": sorted(repo.tags_for(EntityKind.IDEA, idea.id)),
        "links": [
            {
                "from_idea_id": link.from_idea_id,
                "to_idea_id": link.to_idea_id,
                "relation": link.relation.value,
            }
            for link in links
        ],
    }


def list_citations(repo: StorageRepo, idea_id: int | None = None) -> str:
    """The library's `.bib` text, optionally scoped to one idea's confirmed
    matches — exactly what `mustrum bib [--idea]` produces (E5-2/E5-3).
    Reuses RelatedWorkService.export_bib; no new core logic."""
    return RelatedWorkService(repo).export_bib(idea_id)


def create_mcp_server(repo: StorageRepo) -> FastMCP:
    app = FastMCP("mustrum")

    @app.tool(name="search_library")
    def _search_library_tool(query: str, limit: int = 20) -> list[dict[str, Any]]:
        """Full-text search across sources, ideas, and contacts. Returns
        entity type, id, and a matching snippet for each hit."""
        return search_library(repo, query, limit)

    @app.tool(name="get_source")
    def _get_source_tool(source_id: int) -> dict[str, Any]:
        """Fetch one source's full record by id: metadata, verified
        summary (if any), citation key, tags."""
        return get_source(repo, source_id)

    @app.tool(name="get_idea")
    def _get_idea_tool(idea_id: int) -> dict[str, Any]:
        """Fetch one idea's full record by id: title, current text, tags,
        links to other ideas."""
        return get_idea(repo, idea_id)

    @app.tool(name="list_citations")
    def _list_citations_tool(idea_id: int | None = None) -> str:
        """The library's BibTeX, optionally scoped to one idea's confirmed
        source matches."""
        return list_citations(repo, idea_id)

    return app
