"""Enrichment: exact-title Crossref lookup completing bare PDF sources."""

import httpx
import pytest

from mustrum.adapters.crossref import CrossrefFetcher
from mustrum.adapters.enrich import enrich_source
from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import FieldOrigin
from mustrum.core.services.ingest import IngestService

TITLE = "Optimizing Pid Parameters in Mechatronics Using Particle Swarm Optimization"


def crossref_search_client(items, bib="@inproceedings{key2020, title={x}}"):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.crossref.org" and request.url.path == "/works":
            assert request.url.params["query.title"]
            return httpx.Response(200, json={"message": {"items": items}})
        if request.url.host == "doi.org":
            return httpx.Response(200, text=bib)
        raise AssertionError(f"unexpected url {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


def work(title=TITLE, doi="10.1109/pid.2020.1", authors=True):
    return {
        "title": [title],
        "DOI": doi,
        "author": [{"given": "Pia", "family": "Controller"}] if authors else [],
        "issued": {"date-parts": [[2020]]},
    }


class TestSearchByTitle:
    def test_exact_normalised_match_wins(self):
        fetcher = CrossrefFetcher(
            client=crossref_search_client([work(title="A Wrong Paper Here Instead"), work()])
        )
        meta = fetcher.search_by_title(TITLE.upper())  # case must not matter
        assert meta is not None
        assert meta.doi == "10.1109/pid.2020.1"
        assert meta.authors == ("Pia Controller",)
        assert meta.year == 2020

    def test_no_exact_match_returns_none(self):
        fetcher = CrossrefFetcher(
            client=crossref_search_client([work(title=TITLE + " Extended Journal Version")])
        )
        assert fetcher.search_by_title(TITLE) is None

    def test_hit_without_doi_skipped(self):
        fetcher = CrossrefFetcher(client=crossref_search_client([work(doi=None)]))
        assert fetcher.search_by_title(TITLE) is None

    def test_empty_results(self):
        fetcher = CrossrefFetcher(client=crossref_search_client([]))
        assert fetcher.search_by_title(TITLE) is None


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def bare_source(repo):
    return (
        IngestService(repo, FakeEmbeddingProvider())
        .ingest_document(
            title=TITLE, text="full pdf text of the paper", extraction_method="pymupdf"
        )
        .source
    )


class TestEnrichSource:
    def test_enriches_bare_source(self, repo, bare_source):
        result = enrich_source(
            repo,
            FakeEmbeddingProvider(),
            bare_source.id,
            client=crossref_search_client([work()]),
        )
        assert result.enriched is True
        merged = repo.get_source(bare_source.id)
        assert merged.authors == ("Pia Controller",)
        assert merged.year == 2020
        assert merged.doi == "10.1109/pid.2020.1"
        assert dict(merged.provenance)["doi"] == FieldOrigin.FETCHED
        assert repo.get_bib_entry(bare_source.id).citation_key == "key2020"
        # existing full text is untouched
        assert repo.get_source_text(bare_source.id).text == "full pdf text of the paper"

    def test_no_match_changes_nothing(self, repo, bare_source):
        result = enrich_source(
            repo,
            FakeEmbeddingProvider(),
            bare_source.id,
            client=crossref_search_client([]),
        )
        assert result.enriched is False
        assert "no confident Crossref match" in result.message
        assert repo.get_source(bare_source.id).doi is None

    def test_doi_clash_with_other_source_refused(self, repo, bare_source):
        other = (
            IngestService(repo, FakeEmbeddingProvider())
            .ingest_document(
                title="A Different Paper Entirely", text="t", extraction_method="plaintext"
            )
            .source
        )
        import dataclasses

        repo.update_source(dataclasses.replace(other, doi="10.1109/pid.2020.1"))
        result = enrich_source(
            repo,
            FakeEmbeddingProvider(),
            bare_source.id,
            client=crossref_search_client([work()]),
        )
        assert result.enriched is False
        assert "already" in result.message
        assert repo.get_source(bare_source.id).doi is None

    def test_already_complete_source_skipped(self, repo):
        from mustrum.core.models import FetchedMetadata

        complete = (
            IngestService(repo, FakeEmbeddingProvider())
            .ingest_fetched(
                FetchedMetadata(
                    title="Complete Paper",
                    authors=("A",),
                    year=2021,
                    doi="10.1/done",
                    arxiv_id=None,
                    raw_bibtex="@misc{done2021,}",
                    abstract="a",
                )
            )
            .source
        )
        result = enrich_source(repo, FakeEmbeddingProvider(), complete.id)
        assert result.enriched is False
        assert "nothing to enrich" in result.message

    def test_missing_source_raises(self, repo):
        with pytest.raises(KeyError):
            enrich_source(repo, FakeEmbeddingProvider(), 99)
