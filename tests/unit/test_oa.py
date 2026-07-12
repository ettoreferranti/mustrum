"""Offline tests for open-access PDF retrieval (Unpaywall + arXiv)."""

import httpx
import pymupdf
import pytest

from mustrum.adapters.oa import OpenAccessClient, arxiv_pdf_url, fetch_full_text
from mustrum.adapters.pdf import extract_pdf_bytes
from mustrum.core.models import FetchedMetadata


def pdf_bytes(text="hello from a pdf"):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    data = doc.tobytes()
    doc.close()
    return data


def mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


class TestArxivPdfUrl:
    def test_builds_url(self):
        assert arxiv_pdf_url("arXiv:1706.03762v5") == "https://arxiv.org/pdf/1706.03762v5"

    def test_rejects_non_arxiv(self):
        with pytest.raises(ValueError):
            arxiv_pdf_url("10.1000/xyz")


class TestOpenAccessClient:
    def test_requires_email(self):
        with pytest.raises(ValueError, match="e-mail"):
            OpenAccessClient(email="")

    def test_finds_pdf_url(self):
        def handler(request):
            assert request.url.host == "api.unpaywall.org"
            assert request.url.params["email"] == "me@example.org"
            return httpx.Response(
                200, json={"best_oa_location": {"url_for_pdf": "https://oa.host/x.pdf"}}
            )

        client = OpenAccessClient("me@example.org", client=mock_client(handler))
        assert client.find_pdf_url("10.1/x") == "https://oa.host/x.pdf"

    def test_paywalled_returns_none(self):
        def handler(request):
            return httpx.Response(200, json={"best_oa_location": None})

        client = OpenAccessClient("me@example.org", client=mock_client(handler))
        assert client.find_pdf_url("10.1/x") is None

    def test_unknown_doi_returns_none(self):
        def handler(request):
            return httpx.Response(404, json={"error": True})

        client = OpenAccessClient("me@example.org", client=mock_client(handler))
        assert client.find_pdf_url("10.1/none") is None

    def test_oa_location_without_pdf_returns_none(self):
        def handler(request):
            return httpx.Response(
                200, json={"best_oa_location": {"url_for_pdf": None, "url": "https://page"}}
            )

        client = OpenAccessClient("me@example.org", client=mock_client(handler))
        assert client.find_pdf_url("10.1/x") is None

    def test_download_pdf_checks_magic_bytes(self):
        def handler(request):
            if request.url.path.endswith("real.pdf"):
                return httpx.Response(200, content=pdf_bytes())
            return httpx.Response(200, content=b"<html>login required</html>")

        client = OpenAccessClient("me@example.org", client=mock_client(handler))
        assert client.download_pdf("https://oa.host/real.pdf").startswith(b"%PDF")
        with pytest.raises(ValueError, match="did not return a PDF"):
            client.download_pdf("https://oa.host/fake.pdf")


class TestExtractPdfBytes:
    def test_extracts_text_from_stream(self):
        assert "hello from a pdf" in extract_pdf_bytes(pdf_bytes())


def make_meta(**overrides):
    defaults = dict(
        title="T", authors=("A",), year=2020, doi=None, arxiv_id=None, raw_bibtex="@misc{t,}"
    )
    defaults.update(overrides)
    return FetchedMetadata(**defaults)


class TestFetchFullText:
    """fetch_full_text keeps the raw PDF bytes for the file archive (E1-11)."""

    def _patch_client(self, monkeypatch, data):
        class FakeClient:
            def __init__(self, email, client=None):
                pass

            def find_pdf_url(self, doi):
                return None

            def download_pdf(self, url):
                return data

        monkeypatch.setattr("mustrum.adapters.oa.OpenAccessClient", FakeClient)

    def test_success_returns_text_and_pdf_bytes(self, monkeypatch):
        data = pdf_bytes()
        self._patch_client(monkeypatch, data)
        result = fetch_full_text(make_meta(arxiv_id="1706.03762"), "")
        assert "hello from a pdf" in result.text
        assert result.pdf_bytes == data
        assert any(note.startswith("fetched") for note in result.notes)

    def test_no_candidates_returns_empty_result(self, monkeypatch):
        self._patch_client(monkeypatch, b"never used")
        result = fetch_full_text(make_meta(), "")
        assert result.text == ""
        assert result.pdf_bytes is None
