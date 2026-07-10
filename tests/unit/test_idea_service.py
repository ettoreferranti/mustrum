import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import EntityKind, IdeaRelation
from mustrum.core.services.ideas import IdeaService


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def service(repo):
    return IdeaService(repo, FakeEmbeddingProvider())


class TestIdeaService:
    def test_create_stores_idea_version_and_embedding(self, repo, service):
        idea = service.create("Grounded RAG", "verify quotes against sources")
        assert repo.get_idea(idea.id).title == "Grounded RAG"
        versions = repo.get_idea_versions(idea.id)
        assert [v.text for v in versions] == ["verify quotes against sources"]
        embeddings = repo.embeddings_for(EntityKind.IDEA, "fake-embed")
        assert [e.ref_id for e in embeddings] == [idea.id]

    def test_revise_appends_version_and_reembeds(self, repo, service):
        idea = service.create("t", "first")
        before = repo.embeddings_for(EntityKind.IDEA, "fake-embed")[0].vector
        service.revise(idea.id, "completely different focus on quantum chemistry")
        versions = repo.get_idea_versions(idea.id)
        assert [v.text for v in versions] == [
            "first",
            "completely different focus on quantum chemistry",
        ]
        after = repo.embeddings_for(EntityKind.IDEA, "fake-embed")
        assert len(after) == 1  # replaced, not accumulated
        assert after[0].vector != before

    def test_revise_missing_idea_raises(self, service):
        with pytest.raises(KeyError):
            service.revise(99, "x")

    def test_link(self, repo, service):
        a = service.create("a", "ta")
        b = service.create("b", "tb")
        service.link(a.id, b.id, IdeaRelation.BUILDS_ON)
        links = repo.list_idea_links(a.id)
        assert links[0].relation == IdeaRelation.BUILDS_ON

    def test_self_link_rejected(self, service):
        a = service.create("a", "t")
        with pytest.raises(ValueError, match="itself"):
            service.link(a.id, a.id, IdeaRelation.RELATED)

    def test_link_missing_idea_raises(self, service):
        a = service.create("a", "t")
        with pytest.raises(KeyError):
            service.link(a.id, 999, IdeaRelation.RELATED)
