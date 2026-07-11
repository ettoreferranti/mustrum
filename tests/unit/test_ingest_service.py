import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import BibOrigin, EntityKind, FetchedMetadata, FieldOrigin, SourceKind
from mustrum.core.services.ingest import DuplicateSourceError, IngestService


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def service(repo):
    return IngestService(repo, FakeEmbeddingProvider())


META = FetchedMetadata(
    title="Attention Is All You Need",
    authors=("Ashish Vaswani", "Noam Shazeer"),
    year=2017,
    doi="10.48550/arXiv.1706.03762",
    arxiv_id="1706.03762",
    raw_bibtex="@misc{vaswani2017attention,\n  title={Attention Is All You Need}\n}",
    abstract="The dominant sequence transduction models are based on recurrence.",
)


class TestIngestDocument:
    def test_creates_source_text_and_embeddings(self, repo, service):
        result = service.ingest_document(
            title="My Notes",
            text="para one\n\npara two",
            extraction_method="plaintext",
            kind=SourceKind.NOTE,
        )
        assert result.created is True
        stored = repo.get_source_text(result.source.id)
        assert stored.text == "para one\n\npara two"
        assert stored.extraction_method == "plaintext"
        embeddings = repo.embeddings_for(EntityKind.SOURCE, "fake-embed")
        assert len(embeddings) == 1
        assert embeddings[0].ref_id == result.source.id

    def test_provenance_marks_user_fields(self, service):
        result = service.ingest_document(
            title="T",
            text="x",
            extraction_method="plaintext",
            authors=("A",),
            year=2020,
        )
        assert dict(result.source.provenance) == {
            "title": FieldOrigin.USER,
            "authors": FieldOrigin.USER,
            "year": FieldOrigin.USER,
        }

    def test_duplicate_title_fails_by_default(self, service):
        service.ingest_document(title="Same Title", text="a", extraction_method="plaintext")
        with pytest.raises(DuplicateSourceError) as exc:
            service.ingest_document(title="same title!", text="b", extraction_method="plaintext")
        assert exc.value.matched_on == "title"

    def test_duplicate_skip_returns_existing(self, service):
        first = service.ingest_document(title="T", text="a", extraction_method="plaintext")
        result = service.ingest_document(
            title="T", text="b", extraction_method="plaintext", on_duplicate="skip"
        )
        assert result.created is False
        assert result.merged is False
        assert result.source.id == first.source.id

    def test_empty_text_stores_no_source_text(self, repo, service):
        result = service.ingest_document(title="T", text="   ", extraction_method="plaintext")
        assert repo.get_source_text(result.source.id) is None


class TestIngestFetched:
    def test_creates_source_bib_and_abstract(self, repo, service):
        result = service.ingest_fetched(META)
        assert result.created is True
        source = result.source
        assert source.doi == "10.48550/arxiv.1706.03762"
        assert source.arxiv_id == "1706.03762"
        assert dict(source.provenance)["title"] == FieldOrigin.FETCHED
        bib = repo.get_bib_entry(source.id)
        assert bib.citation_key == "vaswani2017attention"
        assert bib.raw_bibtex == META.raw_bibtex  # byte-exact
        assert bib.origin == BibOrigin.FETCHED
        assert repo.get_source_text(source.id).extraction_method == "abstract"

    def test_duplicate_by_doi_detected(self, service):
        service.ingest_fetched(META)
        with pytest.raises(DuplicateSourceError) as exc:
            service.ingest_fetched(META)
        assert exc.value.matched_on == "doi"

    def test_merge_enriches_manual_source(self, repo, service):
        manual = service.ingest_document(
            title="Attention Is All You Need",
            text="full pdf text",
            extraction_method="pymupdf",
        )
        result = service.ingest_fetched(META, on_duplicate="merge")
        assert result.merged is True
        assert result.source.id == manual.source.id
        merged = repo.get_source(manual.source.id)
        assert merged.doi == "10.48550/arxiv.1706.03762"
        assert merged.arxiv_id == "1706.03762"
        assert merged.authors == META.authors
        prov = dict(merged.provenance)
        assert prov["title"] == FieldOrigin.USER  # kept
        assert prov["doi"] == FieldOrigin.FETCHED  # gained
        # existing full text is never replaced by the abstract
        assert repo.get_source_text(manual.source.id).text == "full pdf text"
        # but the bib entry is gained
        assert repo.get_bib_entry(manual.source.id).citation_key == "vaswani2017attention"

    def test_merge_never_overwrites_existing_fields(self, repo, service):
        first = service.ingest_document(
            title="Attention Is All You Need",
            text="t",
            extraction_method="plaintext",
            authors=("Original Author",),
            year=2016,
        )
        result = service.ingest_fetched(META, on_duplicate="merge")
        merged = repo.get_source(first.source.id)
        assert merged.authors == ("Original Author",)
        assert merged.year == 2016
        assert result.merged is True

    def test_clashing_citation_key_from_another_source_rejected(self, repo, service):
        service.ingest_fetched(META)
        other = FetchedMetadata(
            title="A Different Paper",
            authors=("X",),
            year=2020,
            doi="10.1/other",
            arxiv_id=None,
            raw_bibtex="@misc{vaswani2017attention,\n  title={A Different Paper}\n}",
            abstract="",
        )
        with pytest.raises(DuplicateSourceError, match="citation key"):
            service.ingest_fetched(other)

    def test_no_abstract_means_no_text(self, repo, service):
        meta = FetchedMetadata(
            title="No Abstract Paper",
            authors=(),
            year=None,
            doi="10.1/na",
            arxiv_id=None,
            raw_bibtex="@misc{na2020,}",
            abstract="",
        )
        result = service.ingest_fetched(meta)
        assert repo.get_source_text(result.source.id) is None


class TestDedupAndMergeEdges:
    def test_duplicate_by_arxiv_id_detected(self, service):
        first = FetchedMetadata(
            title="Paper A",
            authors=(),
            year=2020,
            doi=None,
            arxiv_id="2001.00001",
            raw_bibtex="@misc{a2020,}",
            abstract="",
        )
        second = FetchedMetadata(
            title="Paper A prime",
            authors=(),
            year=2020,
            doi=None,
            arxiv_id="2001.00001",
            raw_bibtex="@misc{a2020b,}",
            abstract="",
        )
        service.ingest_fetched(first)
        with pytest.raises(DuplicateSourceError) as exc:
            service.ingest_fetched(second)
        assert exc.value.matched_on == "arxiv_id"

    def test_fetched_provenance_covers_all_present_fields(self, service):
        result = service.ingest_fetched(META)
        assert dict(result.source.provenance) == {
            "title": FieldOrigin.FETCHED,
            "authors": FieldOrigin.FETCHED,
            "year": FieldOrigin.FETCHED,
            "doi": FieldOrigin.FETCHED,
            "arxiv_id": FieldOrigin.FETCHED,
        }
        assert result.source.year == 2017
        assert result.source.authors == META.authors

    def test_merge_attaches_abstract_when_existing_has_no_text(self, repo, service):
        manual = service.ingest_document(
            title="Attention Is All You Need", text="", extraction_method="plaintext"
        )
        assert repo.get_source_text(manual.source.id) is None
        service.ingest_fetched(META, on_duplicate="merge")
        stored = repo.get_source_text(manual.source.id)
        assert stored.text == META.abstract
        assert stored.extraction_method == "abstract"

    def test_document_merge_enriches_existing(self, repo, service):
        service.ingest_document(title="Shared Title", text="v1", extraction_method="plaintext")
        result = service.ingest_document(
            title="shared title",
            text="v2",
            extraction_method="plaintext",
            authors=("New Author",),
            year=2022,
            on_duplicate="merge",
        )
        assert result.merged is True
        merged = repo.get_source(result.source.id)
        assert merged.authors == ("New Author",)
        assert merged.year == 2022
        # existing text is never replaced
        assert repo.get_source_text(result.source.id).text == "v1"


class TestMergeFieldRetention:
    def test_skip_does_not_attach_bib_to_existing(self, repo, service):
        manual = service.ingest_document(
            title="Attention Is All You Need", text="t", extraction_method="plaintext"
        )
        service.ingest_fetched(META, on_duplicate="skip")
        assert repo.get_bib_entry(manual.source.id) is None

    def test_refetch_with_merge_is_idempotent(self, repo, service):
        first = service.ingest_fetched(META)
        result = service.ingest_fetched(META, on_duplicate="merge")
        assert result.created is False
        assert result.merged is True
        assert result.source.created_at == first.source.created_at
        assert repo.get_bib_entry(first.source.id).citation_key == "vaswani2017attention"

    def test_merge_keeps_existing_doi_when_matched_on_arxiv(self, repo, service):
        service.ingest_fetched(META)
        incoming = FetchedMetadata(
            title="Different Title",
            authors=("X",),
            year=2018,
            doi="10.9/other",
            arxiv_id=META.arxiv_id,
            raw_bibtex="@misc{otherkey2018,}",
            abstract="",
        )
        result = service.ingest_fetched(incoming, on_duplicate="merge")
        merged = repo.get_source(result.source.id)
        assert merged.doi == "10.48550/arxiv.1706.03762"  # retained, not overwritten

    def test_merge_keeps_existing_arxiv_when_matched_on_doi(self, repo, service):
        service.ingest_fetched(META)
        incoming = FetchedMetadata(
            title="Different Title",
            authors=("X",),
            year=2018,
            doi=META.doi,
            arxiv_id="9999.00001",
            raw_bibtex="@misc{otherkey2018b,}",
            abstract="",
        )
        result = service.ingest_fetched(incoming, on_duplicate="merge")
        merged = repo.get_source(result.source.id)
        assert merged.arxiv_id == "1706.03762"  # retained

    def test_document_merge_attaches_text_when_existing_has_none(self, repo, service):
        service.ingest_document(title="Bare", text="", extraction_method="plaintext")
        result = service.ingest_document(
            title="bare",
            text="now with text",
            extraction_method="pymupdf",
            on_duplicate="merge",
        )
        stored = repo.get_source_text(result.source.id)
        assert stored.text == "now with text"
        assert stored.extraction_method == "pymupdf"

    def test_duplicate_error_carries_existing_source(self, service):
        first = service.ingest_document(title="T", text="a", extraction_method="plaintext")
        with pytest.raises(DuplicateSourceError) as exc:
            service.ingest_document(title="T", text="b", extraction_method="plaintext")
        assert exc.value.existing.id == first.source.id

    def test_merge_preserves_reading_status_and_notes(self, repo, service):
        from mustrum.core.models import ReadingStatus

        manual = service.ingest_document(
            title="Attention Is All You Need", text="t", extraction_method="plaintext"
        )
        repo.set_reading_status(manual.source.id, ReadingStatus.READ)
        repo.set_source_notes(manual.source.id, "great related work section")
        service.ingest_fetched(META, on_duplicate="merge")
        merged = repo.get_source(manual.source.id)
        assert merged.reading_status == ReadingStatus.READ
        assert merged.notes == "great related work section"


class TestIngestFetchedFullText:
    def test_full_text_stored_instead_of_abstract(self, repo, service):
        result = service.ingest_fetched(META, full_text="the complete paper body")
        stored = repo.get_source_text(result.source.id)
        assert stored.text == "the complete paper body"
        assert stored.extraction_method == "pdf-download"

    def test_empty_full_text_falls_back_to_abstract(self, repo, service):
        result = service.ingest_fetched(META, full_text="")
        stored = repo.get_source_text(result.source.id)
        assert stored.text == META.abstract
        assert stored.extraction_method == "abstract"

    def test_merge_offers_full_text_when_existing_has_none(self, repo, service):
        service.ingest_document(
            title="Attention Is All You Need", text="", extraction_method="plaintext"
        )
        result = service.ingest_fetched(
            META, on_duplicate="merge", full_text="full body from OA pdf"
        )
        stored = repo.get_source_text(result.source.id)
        assert stored.text == "full body from OA pdf"
        assert stored.extraction_method == "pdf-download"
