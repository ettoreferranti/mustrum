"""MCP protocol-level tests: the real FastMCP server over an in-memory
client/server session (no subprocess/stdio needed) — proves tool
registration and wiring, mirroring how test_web_api.py/test_cli.py test
the real adapter rather than just the underlying functions."""

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.services.ingest import IngestService
from mustrum.mcp.server import create_mcp_server

SOLAR_TEXT = (
    "We evaluate photovoltaic panel efficiency under variable cloud cover. "
    "Renewable energy generation from solar arrays improves with tracking mounts."
)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


def ingest(repo, embedder, title, text):
    return (
        IngestService(repo, embedder)
        .ingest_document(title=title, text=text, extraction_method="plaintext")
        .source
    )


class TestMcpServer:
    @pytest.mark.anyio
    async def test_tools_registered(self, repo):
        server = create_mcp_server(repo)
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            tools = {t.name for t in (await client.list_tools()).tools}
            assert tools == {"search_library", "get_source", "get_idea", "list_citations"}

    @pytest.mark.anyio
    async def test_search_library_returns_hit(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        server = create_mcp_server(repo)
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            result = await client.call_tool("search_library", {"query": "solar"})
            assert result.isError is False
            hits = result.structuredContent["result"]
            assert hits[0]["ref_id"] == source.id
            assert hits[0]["entity"] == "source"

    @pytest.mark.anyio
    async def test_get_source_returns_full_record(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        server = create_mcp_server(repo)
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            result = await client.call_tool("get_source", {"source_id": source.id})
            assert result.isError is False
            assert result.structuredContent["title"] == "Solar PV study"

    @pytest.mark.anyio
    async def test_get_source_missing_id_is_tool_error(self, repo):
        server = create_mcp_server(repo)
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            result = await client.call_tool("get_source", {"source_id": 999})
            assert result.isError is True
            assert "no source with id 999" in result.content[0].text

    @pytest.mark.anyio
    async def test_list_citations_matches_bib_export(self, repo, embedder):
        ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        server = create_mcp_server(repo)
        async with create_connected_server_and_client_session(server) as client:
            await client.initialize()
            result = await client.call_tool("list_citations", {})
            assert result.isError is False
            assert result.content[0].text.strip() != ""
