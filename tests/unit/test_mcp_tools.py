import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import EntityKind, IdeaRelation, Match, MatchStatus
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.relatedwork import RelatedWorkService
from mustrum.mcp.server import get_idea, get_source, list_citations, search_library

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


class TestSearchLibrary:
    def test_returns_matching_hits(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        hits = search_library(repo, "solar")
        assert hits == [
            {
                "entity": "source",
                "ref_id": source.id,
                "snippet": hits[0]["snippet"],
            }
        ]
        assert "solar" in hits[0]["snippet"].lower()

    def test_no_match_returns_empty(self, repo, embedder):
        ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        assert search_library(repo, "nonexistent zzz") == []

    def test_limit_respected(self, repo, embedder):
        for i in range(3):
            ingest(repo, embedder, f"Solar paper {i}", SOLAR_TEXT + f" variant {i}")
        assert len(search_library(repo, "solar", limit=2)) == 2


class TestGetSource:
    def test_full_record_shape(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        repo.tag(EntityKind.SOURCE, source.id, "energy")
        data = get_source(repo, source.id)
        assert data["id"] == source.id
        assert data["title"] == "Solar PV study"
        assert data["tags"] == ["energy"]
        assert data["summary"] is None
        assert data["citation_key"] is None

    def test_missing_source_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="no source with id 999"):
            get_source(repo, 999)


class TestGetIdea:
    def test_full_record_shape(self, repo, embedder):
        idea_service = IdeaService(repo, embedder)
        idea = idea_service.create("molecular ML", "predict molecule properties")
        idea2 = idea_service.create("unrelated idea", "something else")
        idea_service.link(idea.id, idea2.id, IdeaRelation.RELATED)
        data = get_idea(repo, idea.id)
        assert data["id"] == idea.id
        assert data["title"] == "molecular ML"
        assert data["text"] == "predict molecule properties"
        assert data["links"] == [
            {"from_idea_id": idea.id, "to_idea_id": idea2.id, "relation": "related"}
        ]

    def test_missing_idea_raises_value_error(self, repo):
        with pytest.raises(ValueError, match="no idea with id 999"):
            get_idea(repo, 999)


class TestListCitations:
    def test_matches_related_work_service_export(self, repo, embedder):
        ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        assert list_citations(repo) == RelatedWorkService(repo).export_bib()

    def test_scoped_to_idea_matches_related_work_service(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        idea = IdeaService(repo, embedder).create("energy", "renewable energy research")
        match = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.9))
        repo.set_match_status(match.id, MatchStatus.CONFIRMED)
        assert list_citations(repo, idea.id) == RelatedWorkService(repo).export_bib(idea.id)
        assert list_citations(repo, idea.id) != ""

    def test_empty_library_returns_empty_string(self, repo):
        assert list_citations(repo) == ""
