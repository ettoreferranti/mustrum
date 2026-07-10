import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import BibOrigin, FetchedMetadata
from mustrum.core.services.audit import AuditService
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.match import MatchService
from mustrum.core.services.relatedwork import RelatedWorkService
from mustrum.core.services.summarise import SummariseService


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def world(repo):
    """Idea with two confirmed sources: one fetched (bib), one manual (no bib)."""
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    ideas = IdeaService(repo, embedder)
    fetched = ingest.ingest_fetched(
        FetchedMetadata(
            title="Graph networks for molecules",
            authors=("Ada Ideasmith",),
            year=2021,
            doi="10.1/graphs",
            arxiv_id=None,
            raw_bibtex="@article{ideasmith2021graph,\n title={Graph networks for molecules}\n}",
            abstract="graph networks molecules chemistry",
        )
    ).source
    manual = ingest.ingest_document(
        title="Molecule property prediction with message passing",
        text="message passing molecules graph property prediction",
        extraction_method="plaintext",
        authors=("Bob Manualson",),
        year=2019,
    ).source
    idea = ideas.create("molecular ML", "graph networks molecules property prediction")
    matcher = MatchService(repo, "fake-embed", threshold=0.01)
    for m in matcher.suggest(idea.id):
        matcher.confirm(m.id)
    service = RelatedWorkService(repo)
    return repo, service, idea, fetched, manual


class TestEnsureBibEntry:
    def test_fetched_entry_returned_as_is(self, world):
        repo, service, idea, fetched, manual = world
        entry = service.ensure_bib_entry(fetched.id)
        assert entry.origin == BibOrigin.FETCHED
        assert entry.citation_key == "ideasmith2021graph"

    def test_derived_entry_created_for_manual_source(self, world):
        repo, service, idea, fetched, manual = world
        entry = service.ensure_bib_entry(manual.id)
        assert entry.origin == BibOrigin.DERIVED
        assert entry.citation_key == "manualson2019molecule"
        assert "@article{manualson2019molecule," in entry.raw_bibtex
        # persisted: second call returns the same entry
        assert service.ensure_bib_entry(manual.id) == entry

    def test_missing_source_raises(self, world):
        _, service, *_ = world
        with pytest.raises(KeyError):
            service.ensure_bib_entry(999)


class TestExportBib:
    def test_library_export_contains_all(self, world):
        repo, service, idea, fetched, manual = world
        bib = service.export_bib()
        assert "ideasmith2021graph" in bib
        assert "manualson2019molecule" in bib
        assert bib.endswith("\n")

    def test_idea_export_only_confirmed(self, repo, world):
        _, service, idea, fetched, manual = world
        ingest = IngestService(repo, FakeEmbeddingProvider())
        ingest.ingest_document(
            title="Unrelated interpretive dance", text="dance", extraction_method="plaintext"
        )
        bib = service.export_bib(idea.id)
        assert "ideasmith2021graph" in bib
        assert "dance" not in bib

    def test_empty_library(self, repo):
        assert RelatedWorkService(repo).export_bib() == ""

    def test_missing_idea_raises(self, world):
        _, service, *_ = world
        with pytest.raises(KeyError):
            service.export_bib(999)


class TestSkeleton:
    def test_markdown_contains_keys_summaries_and_relevance(self, repo, world):
        _, service, idea, fetched, manual = world
        # give the fetched source a verified summary
        reply = json.dumps(
            {"summary": "They apply graph networks to molecules.", "quotes": ["graph networks"]}
        )
        SummariseService(repo, FakeLLMProvider([reply])).summarise(fetched.id)
        text = service.skeleton(idea.id, "markdown")
        assert "# Related work — molecular ML" in text
        assert "[@ideasmith2021graph]" in text
        assert "[@manualson2019molecule]" in text
        assert "They apply graph networks to molecules." in text
        assert "[summary: fake-llm, verified]" in text
        assert "TODO: no verified summary stored" in text  # manual source has none
        assert "match score" in text

    def test_latex_format(self, world):
        _, service, idea, *_ = world
        text = service.skeleton(idea.id, "latex")
        assert text.startswith("\\section{Related Work}")
        assert "\\cite{ideasmith2021graph}" in text
        assert "\\paragraph{" in text

    def test_no_confirmed_matches(self, repo):
        embedder = FakeEmbeddingProvider()
        idea = IdeaService(repo, embedder).create("lonely idea", "no sources yet")
        text = RelatedWorkService(repo).skeleton(idea.id)
        assert "No confirmed matches" in text

    def test_missing_idea_raises(self, world):
        _, service, *_ = world
        with pytest.raises(KeyError):
            service.skeleton(999)

    def test_skeleton_passes_citation_verifier(self, world):
        repo, service, idea, *_ = world
        text = service.skeleton(idea.id)
        report = AuditService(repo).audit_text(text)
        assert report.ok is True
        assert set(report.used_keys) == {"ideasmith2021graph", "manualson2019molecule"}


class TestAuditService:
    def test_clean_draft(self, world):
        repo, service, idea, fetched, manual = world
        report = AuditService(repo).audit_text(r"we build on \cite{ideasmith2021graph}.")
        assert report.ok is True
        assert report.known_keys == ("ideasmith2021graph",)

    def test_unknown_key_flagged(self, world):
        repo, *_ = world
        report = AuditService(repo).audit_text(r"\cite{ideasmith2021graph} \cite{phantom2024}")
        assert report.ok is False
        assert report.unknown_keys == ("phantom2024",)
        assert report.known_keys == ("ideasmith2021graph",)

    def test_draft_without_citations(self, repo):
        report = AuditService(repo).audit_text("no citations at all")
        assert report.ok is True
        assert report.used_keys == ()
