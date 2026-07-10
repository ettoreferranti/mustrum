import math

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import MatchStatus
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.match import GapReport, MatchService, cosine


class TestCosine:
    def test_identical_vectors(self):
        assert math.isclose(cosine((1.0, 2.0), (1.0, 2.0)), 1.0)

    def test_orthogonal(self):
        assert cosine((1.0, 0.0), (0.0, 1.0)) == 0.0

    def test_opposite(self):
        assert math.isclose(cosine((1.0, 0.0), (-1.0, 0.0)), -1.0)

    def test_zero_vector_is_zero(self):
        assert cosine((0.0, 0.0), (1.0, 1.0)) == 0.0

    def test_dimension_mismatch_raises(self):
        with pytest.raises(ValueError, match="dimension"):
            cosine((1.0,), (1.0, 2.0))

    def test_scale_invariant(self):
        assert math.isclose(cosine((1.0, 2.0), (10.0, 20.0)), 1.0)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def world(repo):
    """A small library: two on-topic sources, one off-topic, one idea."""
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    ideas = IdeaService(repo, embedder)
    on_topic = ingest.ingest_document(
        title="Graph neural networks for molecular property prediction",
        text="graph neural networks predict molecular properties from molecule graphs",
        extraction_method="plaintext",
    ).source
    also_on_topic = ingest.ingest_document(
        title="Message passing on molecules",
        text="message passing networks operate on molecular graphs and molecules",
        extraction_method="plaintext",
    ).source
    off_topic = ingest.ingest_document(
        title="Medieval poetry archives",
        text="digitising medieval italian poetry manuscripts in monastic archives",
        extraction_method="plaintext",
    ).source
    idea = ideas.create(
        "molecule property prediction",
        "use graph neural networks on molecular graphs to predict properties of molecules",
    )
    service = MatchService(repo, "fake-embed", threshold=0.1)
    return repo, service, idea, on_topic, also_on_topic, off_topic


class TestSuggest:
    def test_ranks_on_topic_sources_first(self, world):
        _repo, service, idea, on_topic, _also_on_topic, off_topic = world
        matches = service.suggest(idea.id)
        assert matches, "expected suggestions"
        suggested_ids = [m.source_id for m in matches]
        assert on_topic.id in suggested_ids
        assert matches == sorted(matches, key=lambda m: m.score, reverse=True)
        if off_topic.id in suggested_ids:
            assert suggested_ids.index(on_topic.id) < suggested_ids.index(off_topic.id)

    def test_suggestions_are_persisted_as_suggested(self, world):
        repo, service, idea, *_ = world
        service.suggest(idea.id)
        stored = repo.list_matches(idea.id, MatchStatus.SUGGESTED)
        assert stored

    def test_rerun_does_not_duplicate(self, world):
        repo, service, idea, *_ = world
        first = service.suggest(idea.id)
        second = service.suggest(idea.id)
        assert second == []
        assert len(repo.list_matches(idea.id)) == len(first)

    def test_threshold_filters(self, world):
        repo, _, idea, *_ = world
        strict = MatchService(repo, "fake-embed", threshold=0.999)
        assert strict.suggest(idea.id) == []

    def test_limit_respected(self, world):
        repo, _, idea, *_ = world
        limited = MatchService(repo, "fake-embed", threshold=0.0)
        assert len(limited.suggest(idea.id, limit=1)) == 1

    def test_missing_idea_raises(self, world):
        _, service, *_ = world
        with pytest.raises(KeyError):
            service.suggest(999)

    def test_unembedded_idea_raises_lookup(self, repo):
        idea = repo.add_idea(__import__("mustrum.core.models", fromlist=["Idea"]).Idea(title="x"))
        service = MatchService(repo, "fake-embed")
        with pytest.raises(LookupError, match="no embedding"):
            service.suggest(idea.id)


class TestWorkflow:
    def test_confirm_and_confirmed_sources(self, world):
        _repo, service, idea, on_topic, *_ = world
        matches = service.suggest(idea.id)
        target = next(m for m in matches if m.source_id == on_topic.id)
        service.confirm(target.id)
        sources = service.confirmed_sources(idea.id)
        assert [s.id for s in sources] == [on_topic.id]

    def test_reject(self, world):
        repo, service, idea, *_ = world
        matches = service.suggest(idea.id)
        service.reject(matches[0].id)
        assert repo.list_matches(idea.id, MatchStatus.REJECTED)


class TestGapReport:
    def test_reports_unsupported_and_orphans(self, world):
        _repo, service, idea, on_topic, _also_on_topic, off_topic = world
        report = service.gap_report()
        assert idea.id in report.unsupported_ideas  # nothing confirmed yet
        assert off_topic.id in report.orphan_sources or on_topic.id in report.orphan_sources

    def test_confirmed_idea_not_unsupported(self, world):
        _repo, service, idea, _on_topic, *_ = world
        matches = service.suggest(idea.id)
        service.confirm(matches[0].id)
        report = service.gap_report()
        assert idea.id not in report.unsupported_ideas

    def test_rejected_source_is_orphan(self, world):
        _repo, service, idea, *_ = world
        matches = service.suggest(idea.id)
        rejected_source = matches[0].source_id
        for m in matches:
            service.reject(m.id)
        report = service.gap_report()
        assert rejected_source in report.orphan_sources

    def test_empty_library(self, repo):
        service = MatchService(repo, "fake-embed")
        assert service.gap_report() == GapReport(unsupported_ideas=(), orphan_sources=())
