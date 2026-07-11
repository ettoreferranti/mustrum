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
        _repo, service, _idea, fetched, _manual = world
        entry = service.ensure_bib_entry(fetched.id)
        assert entry.origin == BibOrigin.FETCHED
        assert entry.citation_key == "ideasmith2021graph"

    def test_derived_entry_created_for_manual_source(self, world):
        _repo, service, _idea, _fetched, manual = world
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
        _repo, service, _idea, _fetched, _manual = world
        bib = service.export_bib()
        assert "ideasmith2021graph" in bib
        assert "manualson2019molecule" in bib
        assert bib.endswith("\n")

    def test_idea_export_only_confirmed(self, repo, world):
        _, service, idea, _fetched, _manual = world
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
        _, service, idea, fetched, _manual = world
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
        repo, _service, _idea, _fetched, _manual = world
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


class TestExportBibScoping:
    def test_other_ideas_confirmed_sources_excluded(self, repo, world):
        _, service, idea, *_ = world
        other_source = (
            IngestService(repo, FakeEmbeddingProvider())
            .ingest_fetched(
                FetchedMetadata(
                    title="Interpretive dance studies",
                    authors=(),
                    year=2018,
                    doi="10.1/dance",
                    arxiv_id=None,
                    raw_bibtex="@misc{dance2018,}",
                    abstract="dance",
                )
            )
            .source
        )
        other_idea = IdeaService(repo, FakeEmbeddingProvider()).create("dance", "dancing research")
        from mustrum.core.models import Match, MatchStatus

        m = repo.add_match(Match(idea_id=other_idea.id, source_id=other_source.id, score=0.9))
        repo.set_match_status(m.id, MatchStatus.CONFIRMED)
        bib = service.export_bib(idea.id)
        assert "dance2018" not in bib
        assert "ideasmith2021graph" in bib

    def test_suggested_matches_excluded_from_export_and_skeleton(self, repo, world):
        _, service, idea, *_ = world
        suggested_source = (
            IngestService(repo, FakeEmbeddingProvider())
            .ingest_fetched(
                FetchedMetadata(
                    title="Only suggested so far",
                    authors=(),
                    year=2022,
                    doi="10.1/sugg",
                    arxiv_id=None,
                    raw_bibtex="@misc{suggested2022,}",
                    abstract="x",
                )
            )
            .source
        )
        from mustrum.core.models import Match

        repo.add_match(Match(idea_id=idea.id, source_id=suggested_source.id, score=0.4))
        assert "suggested2022" not in service.export_bib(idea.id)
        assert "suggested2022" not in service.skeleton(idea.id)


class TestSkeletonContent:
    def test_markdown_includes_idea_text_and_year(self, world):
        _, service, idea, *_ = world
        text = service.skeleton(idea.id, "markdown")
        assert "Research idea: graph networks molecules property prediction" in text
        assert "(2021)" in text

    def test_latex_includes_idea_title_and_text_comments(self, world):
        _, service, idea, *_ = world
        text = service.skeleton(idea.id, "latex")
        assert "% skeleton for idea: molecular ML" in text
        assert "% research idea: graph networks molecules property prediction" in text

    def test_latex_empty_idea_notes_no_matches(self, repo):
        idea = IdeaService(repo, FakeEmbeddingProvider()).create("lonely", "text")
        text = RelatedWorkService(repo).skeleton(idea.id, "latex")
        assert "% no confirmed matches for this idea yet" in text

    def test_malformed_stored_citation_key_fails_loudly(self, repo, world):
        _, service, idea, _fetched, manual = world
        from mustrum.core.models import BibEntry, BibOrigin
        from mustrum.core.services.relatedwork import CitationIntegrityError

        repo.set_bib_entry(
            BibEntry(
                source_id=manual.id,
                citation_key="bad key",
                raw_bibtex="@misc{bad key,}",
                origin=BibOrigin.DERIVED,
            )
        )
        with pytest.raises(CitationIntegrityError, match="unknown keys"):
            service.skeleton(idea.id)

    def test_skeleton_excludes_other_ideas_confirmed_sources(self, repo, world):
        _, service, idea, *_ = world
        other_source = (
            IngestService(repo, FakeEmbeddingProvider())
            .ingest_fetched(
                FetchedMetadata(
                    title="Underwater basket weaving",
                    authors=(),
                    year=2015,
                    doi="10.1/baskets",
                    arxiv_id=None,
                    raw_bibtex="@misc{baskets2015,}",
                    abstract="baskets",
                )
            )
            .source
        )
        other_idea = IdeaService(repo, FakeEmbeddingProvider()).create("baskets", "weaving")
        from mustrum.core.models import Match, MatchStatus

        m = repo.add_match(Match(idea_id=other_idea.id, source_id=other_source.id, score=0.9))
        repo.set_match_status(m.id, MatchStatus.CONFIRMED)
        text = service.skeleton(idea.id)
        assert "baskets2015" not in text
        assert "Underwater basket weaving" not in text

    def test_empty_markdown_skeleton_keeps_header(self, repo):
        idea = IdeaService(repo, FakeEmbeddingProvider()).create("lonely", "text")
        text = RelatedWorkService(repo).skeleton(idea.id, "markdown")
        assert "# Related work — lonely" in text
        assert "No confirmed matches" in text
