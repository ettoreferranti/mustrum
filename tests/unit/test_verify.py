"""Tests for the rigour kernel. This module has the strictest bar in the
project: every mutmut survivor must be killed or justified (NFR-4)."""

from mustrum.core.verify import CitationVerifier, GroundingVerifier

SOURCE = (
    "We propose the Transformer, a model architecture eschewing recurrence and\n"
    "instead relying entirely on an attention mechanism to draw global\n"
    "dependencies between input and output.\tExperiments on two machine\n"
    "translation tasks show these models to be superior in quality."
)


class TestGroundingVerifier:
    def setup_method(self):
        self.v = GroundingVerifier()

    def test_exact_quote_passes(self):
        result = self.v.verify(["We propose the Transformer"], SOURCE)
        assert result.ok is True
        assert result.missing_quotes == ()
        assert result.empty_evidence is False

    def test_quote_spanning_linebreak_passes(self):
        # source has a newline between "and" / "instead"
        result = self.v.verify(["eschewing recurrence and instead relying"], SOURCE)
        assert result.ok

    def test_quote_with_extra_whitespace_passes(self):
        result = self.v.verify(["We  propose   the\tTransformer"], SOURCE)
        assert result.ok

    def test_quote_with_tab_in_source_passes(self):
        result = self.v.verify(["output. Experiments on two machine"], SOURCE)
        assert result.ok

    def test_fabricated_quote_fails(self):
        result = self.v.verify(["we achieve state of the art on ImageNet"], SOURCE)
        assert result.ok is False
        assert result.missing_quotes == ("we achieve state of the art on ImageNet",)

    def test_case_mismatch_fails(self):
        result = self.v.verify(["we propose the transformer"], SOURCE)
        assert not result.ok

    def test_punctuation_mismatch_fails(self):
        result = self.v.verify(["We propose the Transformer;"], SOURCE)
        assert not result.ok

    def test_partial_word_overlap_fails(self):
        result = self.v.verify(["attention mechanisms to draw"], SOURCE)
        assert not result.ok

    def test_only_missing_quotes_reported(self):
        good, bad = "an attention mechanism", "quantum entanglement"
        result = self.v.verify([good, bad], SOURCE)
        assert result.ok is False
        assert result.missing_quotes == (bad,)

    def test_all_missing_reported_in_order(self):
        result = self.v.verify(["zzz first", "zzz second"], SOURCE)
        assert result.missing_quotes == ("zzz first", "zzz second")

    def test_no_quotes_is_empty_evidence_failure(self):
        result = self.v.verify([], SOURCE)
        assert result.ok is False
        assert result.empty_evidence is True
        assert result.missing_quotes == ()

    def test_whitespace_only_quotes_are_empty_evidence(self):
        result = self.v.verify(["   ", "\n\t"], SOURCE)
        assert result.ok is False
        assert result.empty_evidence is True

    def test_blank_quotes_ignored_but_real_quote_still_checked(self):
        result = self.v.verify(["", "an attention mechanism"], SOURCE)
        assert result.ok is True

    def test_duplicate_quotes_pass(self):
        q = "an attention mechanism"
        assert self.v.verify([q, q], SOURCE).ok

    def test_quote_equal_to_whole_source_passes(self):
        assert self.v.verify([SOURCE], SOURCE).ok

    def test_empty_source_fails_any_quote(self):
        result = self.v.verify(["anything"], "")
        assert not result.ok
        assert result.missing_quotes == ("anything",)

    def test_junk_quote_never_matches_normalisation_artifacts(self):
        # guards the whitespace-normalisation delimiter itself: a single-token
        # quote absent from the source must not match whatever separator the
        # normalisation inserts between source words
        result = self.v.verify(["XX"], "a b")
        assert not result.ok
        assert result.missing_quotes == ("XX",)


class TestCitationExtraction:
    def setup_method(self):
        self.v = CitationVerifier()

    def test_basic_latex_cite(self):
        assert self.v.extract_keys(r"as shown in \cite{vaswani2017}") == ("vaswani2017",)

    def test_multiple_keys_in_one_cite(self):
        assert self.v.extract_keys(r"\cite{a, b,c}") == ("a", "b", "c")

    def test_keys_in_one_cite_keep_written_order_not_alphabetical(self):
        assert self.v.extract_keys(r"\cite{zeta, alpha}") == ("zeta", "alpha")

    def test_natbib_and_biblatex_variants(self):
        text = (
            r"\citep{k1} \citet{k2} \citealp{k3} \autocite{k4}"
            r" \parencite{k5} \textcite{k6} \footcite{k7}"
        )
        assert self.v.extract_keys(text) == ("k1", "k2", "k3", "k4", "k5", "k6", "k7")

    def test_starred_variant(self):
        assert self.v.extract_keys(r"\citet*{key}") == ("key",)

    def test_optional_arguments(self):
        assert self.v.extract_keys(r"\citep[see][p.~3]{key}") == ("key",)
        assert self.v.extract_keys(r"\cite[p. 7]{key2}") == ("key2",)

    def test_uppercase_variant(self):
        assert self.v.extract_keys(r"\Citep{key}") == ("key",)

    def test_empty_braces_yield_nothing(self):
        assert self.v.extract_keys(r"\cite{}") == ()
        assert self.v.extract_keys(r"\cite{ , }") == ()

    def test_markdown_bracketed(self):
        assert self.v.extract_keys("as shown [@smith2020]") == ("smith2020",)

    def test_markdown_multiple_in_brackets(self):
        assert self.v.extract_keys("[@a2020; @b2021]") == ("a2020", "b2021")

    def test_markdown_bare_key(self):
        assert self.v.extract_keys("@smith2020 showed that") == ("smith2020",)

    def test_markdown_trailing_punctuation_not_swallowed(self):
        assert self.v.extract_keys("see @smith2020.") == ("smith2020",)
        assert self.v.extract_keys("(@a2019;") == ("a2019",)

    def test_markdown_internal_punctuation_kept(self):
        assert self.v.extract_keys("[@doe:2021/a]") == ("doe:2021/a",)

    def test_email_not_a_citation(self):
        assert self.v.extract_keys("contact user@example.com please") == ()

    def test_dedup_preserves_first_appearance_order(self):
        assert self.v.extract_keys(r"\cite{b} \cite{a} \cite{b}") == ("b", "a")

    def test_mixed_syntax_document_order(self):
        assert self.v.extract_keys(r"[@md1] then \cite{tex1} then @md2") == (
            "md1",
            "tex1",
            "md2",
        )

    def test_plain_text_has_no_keys(self):
        assert self.v.extract_keys("no citations here, just prose.") == ()

    def test_empty_text(self):
        assert self.v.extract_keys("") == ()


class TestCitationVerify:
    def setup_method(self):
        self.v = CitationVerifier()

    def test_all_known_passes(self):
        result = self.v.verify(r"\cite{a} and [@b]", {"a", "b", "c"})
        assert result.ok is True
        assert result.used_keys == ("a", "b")
        assert result.unknown_keys == ()

    def test_unknown_key_fails(self):
        result = self.v.verify(r"\cite{a} \cite{ghost}", {"a"})
        assert result.ok is False
        assert result.unknown_keys == ("ghost",)

    def test_all_unknown_reported_in_order(self):
        result = self.v.verify(r"\cite{g1} \cite{g2}", set())
        assert result.unknown_keys == ("g1", "g2")

    def test_keys_are_case_sensitive(self):
        result = self.v.verify(r"\cite{Smith2020}", {"smith2020"})
        assert not result.ok

    def test_text_without_citations_passes(self):
        result = self.v.verify("plain prose", set())
        assert result.ok is True
        assert result.used_keys == ()


class TestTypographyNormalisation:
    """Publisher PDFs use typographic glyphs; models answer in ASCII. The
    grounding check must fold those — and nothing else."""

    def setup_method(self):
        self.v = GroundingVerifier()

    def _grounded(self, quote, source):
        return self.v.verify([quote], source).ok

    def test_curly_apostrophe_in_source(self):
        assert self._grounded("the subjects' precision", "the subjects’ precision")

    def test_curly_apostrophe_in_quote(self):
        assert self._grounded("the subjects’ precision", "the subjects' precision")

    def test_single_quote_variants(self):
        for glyph in "‘’‚‛":
            assert self._grounded("it's fine", f"it{glyph}s fine"), repr(glyph)

    def test_double_quote_variants(self):
        for glyph_pair in ["“”", "„“"]:
            source = f"they said {glyph_pair[0]}yes{glyph_pair[1]} loudly"
            assert self._grounded('they said "yes" loudly', source), repr(glyph_pair)

    def test_dash_variants(self):
        for glyph in "‐‑‒–—−":
            assert self._grounded("state-of-the-art", f"state{glyph}of{glyph}the{glyph}art"), repr(
                glyph
            )

    def test_soft_hyphen_dropped(self):
        assert self._grounded("discovery", "discov­ery")

    def test_ligatures_folded_by_nfkc(self):
        assert self._grounded("efficient classification", "eﬃcient classiﬁcation")

    def test_non_breaking_space(self):
        assert self._grounded("7 percent", "7 percent")

    def test_wording_changes_still_fail(self):
        assert not self._grounded("median precision increased", "median precision rose")

    def test_missing_words_still_fail(self):
        assert not self._grounded("precision increased greatly", "precision increased")

    def test_case_still_strict(self):
        assert not self._grounded("Median Precision", "median precision")

    def test_digits_still_strict(self):
        assert not self._grounded("recall of 7%", "recall of 8%")
