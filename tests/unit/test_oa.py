"""Offline tests for open-access PDF retrieval (Unpaywall + arXiv)."""

import httpx
import pymupdf
import pytest

from mustrum.adapters.oa import OpenAccessClient, arxiv_pdf_url
from mustrum.adapters.pdf import extract_pdf_bytes


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
