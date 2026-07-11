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


class TestChunkBoundaries:
    def test_max_chars_one_is_valid(self):
        assert chunk_text("ab", max_chars=1) == ["a", "b"]

    def test_exact_fit_join_kept_together(self):
        # "aa\n\nbb" is exactly 6 chars: must remain one chunk
        assert chunk_text("aa\n\nbb", max_chars=6) == ["aa\n\nbb"]

    def test_join_that_would_overflow_by_one_splits(self):
        # 3 + 2 + 3 = 8 > 7: must split
        assert chunk_text("aaa\n\nbbb", max_chars=7) == ["aaa", "bbb"]

    def test_pending_chunk_flushed_before_oversized_paragraph(self):
        assert chunk_text("bb\n\naaaa", max_chars=3) == ["bb", "aaa", "a"]

    def test_paragraph_exactly_divisible_then_next_paragraph_kept(self):
        assert chunk_text("aaaa\n\nbb", max_chars=2) == ["aa", "aa", "bb"]
