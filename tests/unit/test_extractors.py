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
