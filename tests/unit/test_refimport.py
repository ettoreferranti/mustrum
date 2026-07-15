"""Parser tests for E9-4 reference-manager import. Fixtures below mirror the
real quirks of Zotero's and Mendeley's exports (tab vs flush-left
indentation, single- vs double-braced titles, mendeley-tags field, T2/AB/DO
tag choices) so the "one parser covers both tools" claim is exercised
against both, not just one."""

from mustrum.core.refimport import (
    _ris_record_to_reference,
    _split_bibtex_entries,
    _split_bibtex_fields,
    parse_bibtex,
    parse_ris,
)

# -- BibTeX -------------------------------------------------------------------

ZOTERO_BIB = """\
@article{doe_deep_2021,
\ttitle = {Deep {Learning} for Graphs: {A} Survey},
\tvolume = {34},
\tissn = {1041-4347},
\tdoi = {10.1109/tkde.2020.2981333},
\tjournal = {IEEE Transactions on Knowledge and Data Engineering},
\tauthor = {Doe, Jane and Smith, John},
\tyear = {2021},
\tpages = {249--270},
\tabstract = {Deep learning has been shown to be successful in a number of
\tdomains, ranging from acoustics to image processing.}
}
"""

MENDELEY_BIB = """\
@article{Smith2019Attention,
abstract = {We propose a new architecture for sequence modeling.},
author = {Smith, John and Doe, Jane},
doi = {10.1000/182},
issn = {0028-0836},
journal = {Nature},
keywords = {deep learning,neural networks},
mendeley-tags = {deep learning,neural networks},
number = {1},
pages = {123--135},
title = {{Attention mechanisms for sequence modeling}},
volume = {500},
year = {2019}
}
"""

ARXIV_BIB = """\
@misc{doe_arxiv_2022,
\ttitle = {A Preprint on the arXiv},
\tdoi = {10.48550/arXiv.2201.00001},
\tauthor = {Doe, Jane},
\tyear = {2022}
}
"""


class TestParseBibtexZotero:
    def test_fields(self):
        result = parse_bibtex(ZOTERO_BIB)
        assert result.warnings == ()
        assert len(result.references) == 1
        ref = result.references[0]
        assert ref.title == "Deep Learning for Graphs: A Survey"
        assert ref.authors == ("Doe, Jane", "Smith, John")
        assert ref.year == 2021
        assert ref.doi == "10.1109/tkde.2020.2981333"
        assert ref.arxiv_id is None
        assert "acoustics to image processing" in ref.abstract
        assert ref.raw_bibtex == ZOTERO_BIB.strip()


class TestParseBibtexMendeley:
    def test_fields(self):
        result = parse_bibtex(MENDELEY_BIB)
        assert len(result.references) == 1
        ref = result.references[0]
        # double-braced title unwraps to plain text
        assert ref.title == "Attention mechanisms for sequence modeling"
        assert ref.authors == ("Smith, John", "Doe, Jane")
        assert ref.year == 2019
        assert ref.doi == "10.1000/182"
        assert ref.abstract == "We propose a new architecture for sequence modeling."


class TestParseBibtexArxivDoi:
    def test_arxiv_id_from_doi(self):
        ref = parse_bibtex(ARXIV_BIB).references[0]
        assert ref.arxiv_id == "2201.00001"


class TestParseBibtexMultipleEntries:
    def test_both_entries_parsed(self):
        result = parse_bibtex(ZOTERO_BIB + "\n" + MENDELEY_BIB)
        assert len(result.references) == 2
        assert result.references[0].title.startswith("Deep Learning")
        assert result.references[1].title.startswith("Attention mechanisms")


class TestParseBibtexMalformed:
    def test_missing_title_warns_and_is_skipped(self):
        raw = "@article{no_title_2020,\n  author = {Doe, Jane},\n  year = {2020}\n}"
        result = parse_bibtex(raw)
        assert result.references == ()
        assert "no_title_2020" in result.warnings[0]

    def test_one_bad_entry_does_not_lose_the_others(self):
        bad = "@article{bad,\n  author = {Doe, Jane}\n}\n"
        result = parse_bibtex(bad + ZOTERO_BIB)
        assert len(result.references) == 1
        assert len(result.warnings) == 1


class TestParseBibtexEmptyCitationKey:
    """Regression test: a real Mendeley export (library.bib, validated
    2026-07-15) contained an entry with an empty citation key —
    '@techReport{,' — whose key-generation template evidently produced
    nothing. The original `[^,\\s}]+` key pattern required at least one
    character, so `_split_bibtex_entries` never matched this entry at all
    and it vanished from the import with no warning."""

    RAW = "@techReport{,\n  author = {Jane Doe},\n  title = {A Report With No Citation Key}\n}\n"

    def test_entry_is_not_silently_dropped(self):
        result = parse_bibtex(self.RAW)
        assert len(result.references) == 1
        assert result.references[0].title == "A Report With No Citation Key"

    def test_warns_that_a_key_will_be_generated(self):
        result = parse_bibtex(self.RAW)
        assert "no citation key" in result.warnings[0]
        assert "A Report With No Citation Key" in result.warnings[0]

    def test_raw_bibtex_is_none_so_ingest_renders_a_derived_entry(self):
        # a keyless entry has nothing citable in its own text, so it must
        # take the same derived-bib path as a RIS import rather than being
        # stored byte-exact with a blank key
        ref = parse_bibtex(self.RAW).references[0]
        assert ref.raw_bibtex is None


# -- RIS ------------------------------------------------------------------------

ZOTERO_RIS = """\
TY  - JOUR
TI  - Deep Learning for Graphs: A Survey
AU  - Doe, Jane
AU  - Smith, John
T2  - IEEE Transactions on Knowledge and Data Engineering
AB  - Deep learning has been shown to be successful in a number of domains.
DA  - 2021
PY  - 2021
VL  - 34
SP  - 249
EP  - 270
DO  - 10.1109/tkde.2020.2981333
ER  -
"""

MENDELEY_RIS = """\
TY  - JOUR
AU  - Smith, John
AU  - Doe, Jane
TI  - Attention mechanisms for sequence modeling
T2  - Nature
AB  - We propose a new architecture for sequence modeling.
DO  - 10.1000/182
IS  - 1
PY  - 2019
SP  - 123
EP  - 135
VL  - 500
ER  -
"""

ARXIV_RIS = """\
TY  - JOUR
TI  - A Preprint on the arXiv
AU  - Doe, Jane
UR  - https://arxiv.org/abs/2201.00001
PY  - 2022
ER  -
"""


class TestParseRisZotero:
    def test_fields(self):
        result = parse_ris(ZOTERO_RIS)
        assert result.warnings == ()
        assert len(result.references) == 1
        ref = result.references[0]
        assert ref.title == "Deep Learning for Graphs: A Survey"
        assert ref.authors == ("Doe, Jane", "Smith, John")
        assert ref.year == 2021
        assert ref.doi == "10.1109/tkde.2020.2981333"
        assert ref.raw_bibtex is None  # RIS carries no BibTeX form
        assert "successful in a number of domains" in ref.abstract


class TestParseRisMendeley:
    def test_fields(self):
        result = parse_ris(MENDELEY_RIS)
        assert len(result.references) == 1
        ref = result.references[0]
        assert ref.title == "Attention mechanisms for sequence modeling"
        assert ref.authors == ("Smith, John", "Doe, Jane")
        assert ref.year == 2019
        assert ref.doi == "10.1000/182"


class TestParseRisArxivUrl:
    def test_arxiv_id_from_url(self):
        ref = parse_ris(ARXIV_RIS).references[0]
        assert ref.arxiv_id == "2201.00001"


class TestParseRisMultipleRecords:
    def test_both_records_parsed(self):
        result = parse_ris(ZOTERO_RIS + "\n" + MENDELEY_RIS)
        assert len(result.references) == 2


class TestParseRisMalformed:
    def test_missing_title_warns_and_is_skipped(self):
        raw = "TY  - JOUR\nAU  - Doe, Jane\nPY  - 2020\nER  -\n"
        result = parse_ris(raw)
        assert result.references == ()
        assert "no TI/T1" in result.warnings[0]

    def test_one_bad_record_does_not_lose_the_others(self):
        bad = "TY  - JOUR\nAU  - Doe, Jane\nER  -\n"
        result = parse_ris(bad + ZOTERO_RIS)
        assert len(result.references) == 1
        assert len(result.warnings) == 1

    def test_blank_lines_and_unknown_tags_ignored(self):
        raw = "TY  - JOUR\nTI  - T\n\nXY  - unsupported\nER  -\n"
        result = parse_ris(raw)
        assert result.references[0].title == "T"

    def test_exact_line_number_in_warning(self):
        raw = "TY  - JOUR\nAU  - Doe, Jane\nPY  - 2020\nER  -\n"
        result = parse_ris(raw)
        assert "line 4" in result.warnings[0]  # the ER line

    def test_stray_tag_before_first_ty_does_not_crash(self):
        raw = "XY  - stray preamble\nTY  - JOUR\nTI  - T\nER  -\n"
        result = parse_ris(raw)
        assert result.references[0].title == "T"

    def test_indented_tag_line_is_silently_ignored(self):
        # a known limitation (docs/DECISIONS.md ADR-24): only tag lines
        # anchored at column 0 are recognised
        raw = "TY  - JOUR\n  TI  - Indented Title\nER  -\n"
        result = parse_ris(raw)
        assert result.references == ()
        assert "no TI/T1" in result.warnings[0]

    def test_non_tag_continuation_line_mid_record_does_not_abort_remaining_input(self):
        # a wrapped-abstract continuation line with no "XX  - " tag prefix at
        # all (not just an unrecognised tag) — must be skipped, not treated
        # as a reason to stop reading the rest of the file
        raw = (
            "TY  - JOUR\nstray continuation line, no tag prefix\nTI  - First\nER  -\n"
            "TY  - JOUR\nTI  - Second\nER  -\n"
        )
        result = parse_ris(raw)
        assert [r.title for r in result.references] == ["First", "Second"]

    def test_unsupported_tag_mid_record_does_not_abort_remaining_input(self):
        raw = "TY  - JOUR\nXY  - unsupported\nTI  - First\nER  -\nTY  - JOUR\nTI  - Second\nER  -\n"
        result = parse_ris(raw)
        assert [r.title for r in result.references] == ["First", "Second"]

    def test_stray_tag_after_last_er_does_not_crash(self):
        raw = "TY  - JOUR\nTI  - T\nER  -\nXY  - trailing stray\n"
        result = parse_ris(raw)
        assert len(result.references) == 1
        assert result.references[0].title == "T"


class TestParseRisRecordIsolation:
    def test_second_record_does_not_inherit_first_records_authors(self):
        raw = (
            "TY  - JOUR\nAU  - Alpha, A\nTI  - First\nER  -\n"
            "TY  - JOUR\nAU  - Beta, B\nTI  - Second\nER  -\n"
        )
        result = parse_ris(raw)
        assert result.references[0].authors == ("Alpha, A",)
        assert result.references[1].authors == ("Beta, B",)

    def test_ty_resets_a_record_abandoned_without_er(self):
        # a truncated/corrupted record (missing its ER terminator) directly
        # followed by the next record's TY: the new TY must still clear
        # whatever fields leaked in from the abandoned one
        raw = "TY  - JOUR\nAU  - Alpha, A\nTY  - JOUR\nTI  - Real Title\nER  -\n"
        result = parse_ris(raw)
        assert len(result.references) == 1
        assert result.references[0].title == "Real Title"
        assert result.references[0].authors == ()


class TestRisRecordToReferenceDirect:
    def test_t1_alias_used_when_ti_absent(self):
        ref, warning = _ris_record_to_reference({"T1": ["Alt Title"]}, 1)
        assert warning is None
        assert ref.title == "Alt Title"

    def test_y1_alias_used_when_py_absent(self):
        ref, _ = _ris_record_to_reference({"TI": ["T"], "Y1": ["2015"]}, 1)
        assert ref.year == 2015

    def test_n2_alias_used_when_ab_absent(self):
        ref, _ = _ris_record_to_reference({"TI": ["T"], "N2": ["An abstract."]}, 1)
        assert ref.abstract == "An abstract."

    def test_multiple_ab_lines_joined_with_space(self):
        ref, _ = _ris_record_to_reference({"TI": ["T"], "AB": ["Line one", "line two"]}, 1)
        assert ref.abstract == "Line one line two"

    def test_arxiv_id_from_doi_when_no_ur_field(self):
        ref, _ = _ris_record_to_reference({"TI": ["T"], "DO": ["10.48550/arXiv.9999.00001"]}, 1)
        assert ref.arxiv_id == "9999.00001"


# -- BibTeX: private helpers and edge cases ------------------------------------


class TestSplitBibtexFieldsDirect:
    def test_single_field_no_leading_whitespace(self):
        assert _split_bibtex_fields("title={T}") == {"title": "T"}

    def test_quoted_value_with_comma_is_not_split(self):
        fields = _split_bibtex_fields('title = "A, B", year = {2020}')
        assert fields["title"] == "A, B"
        assert fields["year"] == "2020"

    def test_unclosed_quote_trailing_comma_is_stripped(self):
        # malformed (never-closed) quoted value at the end of the body: the
        # comma is swallowed into the value (still "in quote"), so it falls
        # to this defensive rstrip rather than the split boundary
        fields = _split_bibtex_fields('title = "Bar,')
        assert fields["title"] == '"Bar'

    def test_unclosed_quote_trailing_comma_strip_does_not_eat_x(self):
        fields = _split_bibtex_fields('note = "EndsWithX,')
        assert fields["note"] == '"EndsWithX'

    def test_adjacent_fields_no_space_after_comma(self):
        fields = _split_bibtex_fields("title={T},year={2020}")
        assert fields == {"title": "T", "year": "2020"}

    def test_part_without_equals_is_skipped_not_aborted(self):
        fields = _split_bibtex_fields("garbage,title={T}")
        assert fields == {"title": "T"}

    def test_value_containing_equals_sign_kept_whole(self):
        fields = _split_bibtex_fields("url={http://x.example/a=b}")
        assert fields["url"] == "http://x.example/a=b"

    def test_double_braced_title_unwraps_fully(self):
        fields = _split_bibtex_fields("title={{T}}")
        assert fields["title"] == "T"

    def test_nested_protective_braces_removed(self):
        fields = _split_bibtex_fields("title={Deep {Learning} for {NLP}}")
        assert fields["title"] == "Deep Learning for NLP"

    def test_whitespace_collapsed_to_single_spaces(self):
        fields = _split_bibtex_fields("abstract={line one\n\tline   two}")
        assert fields["abstract"] == "line one line two"


class TestSplitBibtexEntriesDirect:
    def test_unclosed_entry_does_not_crash(self):
        entries = _split_bibtex_entries("@article{key, title={Unclosed")
        assert entries == ["@article{"]


class TestParseBibtexArxivViaEprint:
    def test_explicit_eprint_and_archiveprefix_no_doi(self):
        raw = (
            "@misc{doe2022,\n"
            "  title = {A Preprint},\n"
            "  author = {Doe, Jane},\n"
            "  eprint = {2202.00002},\n"
            "  archiveprefix = {arXiv}\n"
            "}\n"
        )
        ref = parse_bibtex(raw).references[0]
        assert ref.arxiv_id == "2202.00002"

    def test_archiveprefix_without_eprint_does_not_crash(self):
        raw = "@misc{doe2022b,\n  title = {A Preprint},\n  archiveprefix = {arXiv}\n}\n"
        ref = parse_bibtex(raw).references[0]
        assert ref.arxiv_id is None

    def test_arxiv_id_from_url_field(self):
        raw = (
            "@misc{doe2022c,\n"
            "  title = {A Preprint},\n"
            "  url = {https://arxiv.org/abs/2203.00003}\n"
            "}\n"
        )
        ref = parse_bibtex(raw).references[0]
        assert ref.arxiv_id == "2203.00003"

    def test_arxiv_id_from_journal_field(self):
        raw = (
            "@misc{doe2022d,\n"
            "  title = {A Preprint},\n"
            "  journal = {see https://arxiv.org/abs/2204.00004}\n"
            "}\n"
        )
        ref = parse_bibtex(raw).references[0]
        assert ref.arxiv_id == "2204.00004"


class TestParseBibtexAbsentFields:
    def test_no_author_field_does_not_crash(self):
        raw = "@misc{k,\n  title = {T}\n}\n"
        ref = parse_bibtex(raw).references[0]
        assert ref.authors == ()

    def test_no_year_field_does_not_crash(self):
        raw = "@misc{k,\n  title = {T},\n  author = {A}\n}\n"
        ref = parse_bibtex(raw).references[0]
        assert ref.year is None

    def test_no_abstract_field_defaults_to_empty_string(self):
        raw = "@misc{k,\n  title = {T}\n}\n"
        ref = parse_bibtex(raw).references[0]
        assert ref.abstract == ""
