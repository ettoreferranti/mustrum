import pytest

from mustrum.core.bibtex import extract_citation_key, make_citation_key, render_derived_entry
from mustrum.core.models import Source, SourceKind


def make_source(**overrides) -> Source:
    defaults = dict(
        kind=SourceKind.PAPER,
        title="Attention Is All You Need",
        authors=("Ashish Vaswani", "Noam Shazeer"),
        year=2017,
    )
    defaults.update(overrides)
    return Source(**defaults)


class TestExtractCitationKey:
    def test_simple_entry(self):
        assert extract_citation_key("@article{vaswani2017,\n title={X}\n}") == "vaswani2017"

    def test_whitespace_tolerant(self):
        assert extract_citation_key("@ misc { my-key ,\n}") == "my-key"

    def test_key_with_colon_and_dots(self):
        assert extract_citation_key("@inproceedings{DBLP:conf/nips/17,}") == "DBLP:conf/nips/17"

    def test_no_key_raises(self):
        with pytest.raises(ValueError, match="no citation key"):
            extract_citation_key("this is not bibtex")


class TestMakeCitationKey:
    def test_basic(self):
        assert make_citation_key(make_source(), set()) == "vaswani2017attention"

    def test_family_comma_given_format(self):
        source = make_source(authors=("Vaswani, Ashish",))
        assert make_citation_key(source, set()) == "vaswani2017attention"

    def test_short_title_words_skipped(self):
        source = make_source(title="On the Use of Grounding")
        assert make_citation_key(source, set()) == "vaswani2017grounding"

    def test_no_authors(self):
        source = make_source(authors=())
        assert make_citation_key(source, set()) == "anon2017attention"

    def test_no_year(self):
        source = make_source(year=None)
        assert make_citation_key(source, set()) == "vaswanindattention"

    def test_unicode_surname_sanitised(self):
        source = make_source(authors=("Rémi Müller-Straße",))
        key = make_citation_key(source, set())
        assert key.startswith("mllerstrae") or key.isascii()

    def test_collision_gets_suffix(self):
        existing = {"vaswani2017attention"}
        assert make_citation_key(make_source(), existing) == "vaswani2017attentiona"

    def test_second_collision(self):
        existing = {"vaswani2017attention", "vaswani2017attentiona"}
        assert make_citation_key(make_source(), existing) == "vaswani2017attentionb"


class TestRenderDerivedEntry:
    def test_full_entry(self):
        source = make_source(doi="10.1/x", arxiv_id="1706.03762")
        entry = render_derived_entry(source, "vaswani2017attention")
        assert entry.startswith("@article{vaswani2017attention,")
        assert "  title = {Attention Is All You Need}," in entry
        assert "  author = {Ashish Vaswani and Noam Shazeer}," in entry
        assert "  year = {2017}," in entry
        assert "  doi = {10.1/x}," in entry
        assert "  eprint = {1706.03762}," in entry
        assert entry.endswith("}")

    def test_note_kind_is_misc(self):
        source = make_source(kind=SourceKind.NOTE)
        assert render_derived_entry(source, "k").startswith("@misc{k,")

    def test_absent_fields_not_rendered(self):
        source = make_source(authors=(), year=None)
        entry = render_derived_entry(source, "k")
        assert "author" not in entry
        assert "year" not in entry
        assert "doi" not in entry
        assert "eprint" not in entry
