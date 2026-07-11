import pymupdf

from mustrum.adapters.pdf import PdfExtractor, PlainTextExtractor, extractor_for


def make_pdf(path, texts):
    doc = pymupdf.open()
    for text in texts:
        page = doc.new_page()
        page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestPdfExtractor:
    def test_extracts_text_across_pages(self, tmp_path):
        pdf = tmp_path / "paper.pdf"
        make_pdf(pdf, ["First page content.", "Second page content."])
        text = PdfExtractor().extract(pdf)
        assert "First page content." in text
        assert "Second page content." in text
        assert text.index("First") < text.index("Second")

    def test_method_name(self):
        assert PdfExtractor().extraction_method == "pymupdf"


class TestPlainTextExtractor:
    def test_reads_utf8(self, tmp_path):
        f = tmp_path / "notes.md"
        f.write_text("# Idée\n\ncontenu", encoding="utf-8")
        assert PlainTextExtractor().extract(f) == "# Idée\n\ncontenu"


class TestExtractorFor:
    def test_pdf_by_suffix(self, tmp_path):
        assert isinstance(extractor_for(tmp_path / "x.PDF"), PdfExtractor)

    def test_everything_else_plaintext(self, tmp_path):
        assert isinstance(extractor_for(tmp_path / "x.md"), PlainTextExtractor)
        assert isinstance(extractor_for(tmp_path / "x.txt"), PlainTextExtractor)


def make_pdf_with_meta(path, text, title=None):
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    if title is not None:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


class TestPdfMetadataTitle:
    def test_plausible_title_accepted(self, tmp_path):
        from mustrum.adapters.pdf import pdf_metadata_title

        pdf = tmp_path / "x.pdf"
        make_pdf_with_meta(pdf, "body", title="Optimizing PID Parameters in Mechatronics")
        assert pdf_metadata_title(pdf) == "Optimizing PID Parameters in Mechatronics"

    def test_whitespace_collapsed(self, tmp_path):
        from mustrum.adapters.pdf import pdf_metadata_title

        pdf = tmp_path / "x.pdf"
        make_pdf_with_meta(pdf, "body", title="A  Title\nWith   Breaks")
        assert pdf_metadata_title(pdf) == "A Title With Breaks"

    def test_junk_titles_rejected(self, tmp_path):
        from mustrum.adapters.pdf import pdf_metadata_title

        for junk in [
            None,
            "",
            "short",
            "no_spaces_in_this_one_at_all",
            "Microsoft Word - draft final v3",
            "untitled document",
            "paper final version.pdf",
            "x" * 301,
        ]:
            pdf = tmp_path / "j.pdf"
            make_pdf_with_meta(pdf, "body", title=junk)
            assert pdf_metadata_title(pdf) is None, repr(junk)

    def test_bytes_variant(self, tmp_path):
        from mustrum.adapters.pdf import pdf_metadata_title_bytes

        pdf = tmp_path / "x.pdf"
        make_pdf_with_meta(pdf, "body", title="A Proper Paper Title")
        assert pdf_metadata_title_bytes(pdf.read_bytes()) == "A Proper Paper Title"

    def test_html_entities_decoded(self, tmp_path):
        from mustrum.adapters.pdf import pdf_metadata_title

        pdf = tmp_path / "x.pdf"
        make_pdf_with_meta(pdf, "body", title="SBFT Tool Competition 2025 &#x2013; UAV Track")
        assert pdf_metadata_title(pdf) == "SBFT Tool Competition 2025 \u2013 UAV Track"
