"""Offline tests for arXiv/Crossref fetchers via httpx.MockTransport."""

import httpx
import pytest

from mustrum.adapters.arxiv import ArxivFetcher, normalize_arxiv_id
from mustrum.adapters.crossref import CrossrefFetcher

ATOM_OK = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <title>Attention Is All
     You Need</title>
    <published>2017-06-12T17:57:34Z</published>
    <summary>The dominant sequence
      transduction models.</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:doi>10.48550/arXiv.1706.03762</arxiv:doi>
  </entry>
</feed>"""

ATOM_EMPTY = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"></feed>"""

BIBTEX = "@misc{vaswani2017attentionisallyouneed,\n title={Attention Is All You Need}\n}"


def arxiv_client(atom=ATOM_OK, bibtex=BIBTEX):
    def handler(request: httpx.Request) -> httpx.Response:
        if "export.arxiv.org" in request.url.host:
            return httpx.Response(200, text=atom)
        if request.url.path.startswith("/bibtex/"):
            return httpx.Response(200, text=bibtex)
        raise AssertionError(f"unexpected url {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestNormalizeArxivId:
    def test_modern_id(self):
        assert normalize_arxiv_id("1706.03762") == "1706.03762"

    def test_with_version_and_prefix(self):
        assert normalize_arxiv_id("arXiv:1706.03762v5") == "1706.03762v5"

    def test_old_style(self):
        assert normalize_arxiv_id("cs/0112017") == "cs/0112017"

    def test_rejects_doi(self):
        with pytest.raises(ValueError):
            normalize_arxiv_id("10.1000/xyz")


class TestArxivFetcher:
    def test_parses_metadata_and_bibtex(self):
        meta = ArxivFetcher(client=arxiv_client()).fetch("arXiv:1706.03762")
        assert meta.title == "Attention Is All You Need"  # whitespace collapsed
        assert meta.authors == ("Ashish Vaswani", "Noam Shazeer")
        assert meta.year == 2017
        assert meta.doi == "10.48550/arXiv.1706.03762"
        assert meta.arxiv_id == "1706.03762"
        assert meta.raw_bibtex == BIBTEX
        assert meta.abstract == "The dominant sequence transduction models."

    def test_unknown_id_raises_lookup_error(self):
        with pytest.raises(LookupError):
            ArxivFetcher(client=arxiv_client(atom=ATOM_EMPTY)).fetch("9999.99999")

    def test_non_bibtex_reply_raises(self):
        client = arxiv_client(bibtex="<html>rate limited</html>")
        with pytest.raises(LookupError, match="BibTeX"):
            ArxivFetcher(client=client).fetch("1706.03762")


CROSSREF_JSON = {
    "message": {
        "title": ["Deep  Residual Learning"],
        "author": [
            {"given": "Kaiming", "family": "He"},
            {"family": "Anonymous Collective"},
        ],
        "issued": {"date-parts": [[2016, 6]]},
        "abstract": "<jats:p>Deeper networks.</jats:p>",
    }
}

CROSSREF_BIB = "@inproceedings{He_2016, title={Deep Residual Learning}}"


def crossref_client(json_body=CROSSREF_JSON, bib=CROSSREF_BIB, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "api.crossref.org":
            return httpx.Response(status, json=json_body)
        if request.url.host == "doi.org":
            assert request.headers["accept"] == "application/x-bibtex"
            return httpx.Response(200, text=bib)
        raise AssertionError(f"unexpected url {request.url}")

    return httpx.Client(transport=httpx.MockTransport(handler))


class TestCrossrefFetcher:
    def test_parses_metadata_and_bibtex(self):
        meta = CrossrefFetcher(client=crossref_client()).fetch("10.1109/CVPR.2016.90")
        assert meta.title == "Deep Residual Learning"
        assert meta.authors == ("Kaiming He", "Anonymous Collective")
        assert meta.year == 2016
        assert meta.doi == "10.1109/cvpr.2016.90"
        assert meta.arxiv_id is None
        assert meta.raw_bibtex == CROSSREF_BIB

    def test_doi_not_found(self):
        client = crossref_client(json_body={}, status=404)
        with pytest.raises(LookupError, match="not found"):
            CrossrefFetcher(client=client).fetch("10.9999/none")

    def test_missing_title_raises(self):
        client = crossref_client(json_body={"message": {"title": []}})
        with pytest.raises(LookupError, match="title"):
            CrossrefFetcher(client=client).fetch("10.1/x")
