from mustrum.core.normalize import normalize_doi, normalize_title, title_hash


class TestNormalizeTitle:
    def test_lowercases(self):
        assert normalize_title("Attention Is All You Need") == "attention is all you need"

    def test_strips_punctuation_and_collapses_spaces(self):
        assert (
            normalize_title("BERT: Pre-training of  Deep --- Bidirectional Transformers!")
            == "bert pre training of deep bidirectional transformers"
        )

    def test_strips_leading_and_trailing_separators(self):
        assert normalize_title("  (A Survey) ") == "a survey"

    def test_keeps_digits(self):
        assert normalize_title("GPT-4 Technical Report") == "gpt 4 technical report"

    def test_empty_title(self):
        assert normalize_title("") == ""


class TestTitleHash:
    def test_equal_for_formatting_variants(self):
        assert title_hash("Attention is all you need.") == title_hash("ATTENTION IS ALL — YOU NEED")

    def test_differs_for_different_titles(self):
        assert title_hash("Paper One") != title_hash("Paper Two")

    def test_is_sha256_hex(self):
        h = title_hash("x")
        assert len(h) == 64
        int(h, 16)  # parses as hex


class TestNormalizeDoi:
    def test_strips_https_prefix(self):
        assert normalize_doi("https://doi.org/10.1000/XYZ") == "10.1000/xyz"

    def test_strips_http_prefix(self):
        assert normalize_doi("http://doi.org/10.1000/xyz") == "10.1000/xyz"

    def test_strips_doi_colon_prefix(self):
        assert normalize_doi("doi:10.1000/xyz") == "10.1000/xyz"

    def test_lowercases_and_trims(self):
        assert normalize_doi("  10.1000/AbC  ") == "10.1000/abc"

    def test_bare_doi_unchanged(self):
        assert normalize_doi("10.1000/abc") == "10.1000/abc"
