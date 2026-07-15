import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import Contact, ContactKind, ContactLink, IdeaRelation, MatchStatus
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.match import MatchService
from mustrum.graph.export import build_graph_data, export_graph


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def world(repo):
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    ideas = IdeaService(repo, embedder)
    source = ingest.ingest_document(
        title="Graph networks", text="graph networks molecules", extraction_method="plaintext"
    ).source
    idea_a = ideas.create("molecular ML", "graph networks molecules")
    idea_b = ideas.create("spinoff", "another angle")
    ideas.link(idea_b.id, idea_a.id, IdeaRelation.BUILDS_ON)
    matcher = MatchService(repo, "fake-embed", threshold=0.01)
    matches = matcher.suggest(idea_a.id)
    matcher.confirm(matches[0].id)
    contact = repo.add_contact(Contact(name="Prof X", kind=ContactKind.UNIVERSITY))
    repo.add_contact_link(ContactLink(contact_id=contact.id, why="expert", idea_id=idea_a.id))
    return repo, idea_a, idea_b, source, contact


class TestBuildGraphData:
    def test_nodes_for_all_entity_types(self, world):
        repo, idea_a, idea_b, source, contact = world
        data = build_graph_data(repo)
        ids = {n["data"]["id"] for n in data["nodes"]}
        assert {
            f"idea-{idea_a.id}",
            f"idea-{idea_b.id}",
            f"source-{source.id}",
            f"contact-{contact.id}",
        } <= ids

    def test_confirmed_match_edge_with_status_and_score(self, world):
        repo, idea_a, _, source, _ = world
        data = build_graph_data(repo)
        match_edges = [e["data"] for e in data["edges"] if e["data"]["type"] == "match"]
        assert len(match_edges) == 1
        edge = match_edges[0]
        assert edge["source"] == f"idea-{idea_a.id}"
        assert edge["target"] == f"source-{source.id}"
        assert edge["status"] == "confirmed"
        assert 0 < edge["score"] <= 1

    def test_rejected_matches_excluded(self, world):
        repo, _idea_a, idea_b, _source, _ = world
        matcher = MatchService(repo, "fake-embed", threshold=0.0)
        for m in matcher.suggest(idea_b.id):
            matcher.reject(m.id)
        data = build_graph_data(repo)
        statuses = {e["data"].get("status") for e in data["edges"]}
        assert MatchStatus.REJECTED.value not in statuses

    def test_idea_link_edge(self, world):
        repo, _idea_a, _idea_b, *_ = world
        data = build_graph_data(repo)
        link_edges = [e["data"] for e in data["edges"] if e["data"]["type"] == "idea-link"]
        assert link_edges[0]["relation"] == "builds-on"

    def test_contacts_can_be_excluded(self, world):
        repo, *_ = world
        data = build_graph_data(repo, include_contacts=False)
        assert not any(n["data"]["type"] == "contact" for n in data["nodes"])
        assert not any(e["data"]["type"] == "contact-link" for e in data["edges"])

    def test_unlinked_contacts_not_drawn(self, repo):
        repo.add_contact(Contact(name="Lonely", kind=ContactKind.PERSON))
        data = build_graph_data(repo)
        assert not any(n["data"]["type"] == "contact" for n in data["nodes"])

    def test_source_node_carries_citation_key_and_summary(self, world):
        repo, _, _, source, _ = world
        from mustrum.core.services.relatedwork import RelatedWorkService

        RelatedWorkService(repo).ensure_bib_entry(source.id)
        data = build_graph_data(repo)
        node = next(n["data"] for n in data["nodes"] if n["data"]["id"] == f"source-{source.id}")
        assert node["citation_key"]


class TestRenderHtml:
    def test_self_contained_html(self, world):
        repo, *_ = world
        page = export_graph(repo)
        assert page.startswith("<!DOCTYPE html>")
        assert "cytoscape" in page.lower()
        # no external references: everything must be inline
        assert 'src="http' not in page
        assert "href=" not in page

    def test_contains_all_elements(self, world):
        repo, idea_a, *_ = world
        page = export_graph(repo)
        assert f"idea-{idea_a.id}" in page
        assert "molecular ML" in page

    def test_script_breakout_escaped(self, repo):
        embedder = FakeEmbeddingProvider()
        IngestService(repo, embedder).ingest_document(
            title="Evil </script><script>alert(1)</script> title",
            text="x",
            extraction_method="plaintext",
        )
        page = export_graph(repo)
        assert "</script><script>alert(1)" not in page

    def test_detail_panel_escapes_untrusted_values(self, repo):
        """The tap-handler builds the detail panel with innerHTML; every
        node-data value it splices in originates from untrusted sources
        (titles, summaries, imported metadata), so each must be routed
        through the esc() helper — otherwise a crafted title runs arbitrary
        script against the unauthenticated local API served by `mustrum ui`."""
        page = export_graph(repo)
        assert "function esc(value)" in page
        # every field spliced into the panel is escaped ...
        for escaped in (
            "esc(d.label)",
            "esc(d.type)",
            "esc(d.kind)",
            "esc(d.detail)",
            "esc(d.authors.join",
            "esc(d.year)",
            "esc(d.citation_key)",
            "esc(d.tags.join",
        ):
            assert escaped in page, escaped
        # ... and no field reaches innerHTML raw (regression guard)
        for raw in ('" + d.label + "', '" + d.detail + "', '" + d.type + "'):
            assert raw not in page, raw

    def test_empty_library_still_renders(self, repo):
        page = export_graph(repo)
        assert page.startswith("<!DOCTYPE html>")
