import pytest

from mustrum.core.services.chunk import chunk_text


class TestChunkText:
    def test_short_text_single_chunk(self):
        assert chunk_text("hello world") == ["hello world"]

    def test_empty_text_no_chunks(self):
        assert chunk_text("") == []
        assert chunk_text("\n\n  \n\n") == []

    def test_paragraphs_grouped_within_limit(self):
        text = "para one\n\npara two\n\npara three"
        assert chunk_text(text, max_chars=100) == [text]

    def test_split_on_paragraph_boundary(self):
        text = "a" * 60 + "\n\n" + "b" * 60
        chunks = chunk_text(text, max_chars=100)
        assert chunks == ["a" * 60, "b" * 60]

    def test_oversized_paragraph_hard_split(self):
        chunks = chunk_text("x" * 250, max_chars=100)
        assert chunks == ["x" * 100, "x" * 100, "x" * 50]

    def test_no_chunk_exceeds_limit(self):
        text = ("word " * 50 + "\n\n") * 10
        assert all(len(c) <= 120 for c in chunk_text(text, max_chars=120))

    def test_all_content_preserved(self):
        text = "alpha beta\n\ngamma delta\n\n" + "z" * 300
        joined = " ".join(chunk_text(text, max_chars=80))
        for token in ["alpha", "beta", "gamma", "delta"]:
            assert token in joined
        assert joined.count("z") == 300

    def test_invalid_max_chars(self):
        with pytest.raises(ValueError):
            chunk_text("x", max_chars=0)
