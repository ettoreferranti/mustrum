import sqlite3

import pytest

from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import (
    BibEntry,
    BibOrigin,
    Contact,
    ContactKind,
    ContactLink,
    Embedding,
    EntityKind,
    FieldOrigin,
    Idea,
    IdeaLink,
    IdeaRelation,
    IdeaVersion,
    Match,
    MatchStatus,
    ReadingStatus,
    Source,
    SourceKind,
    SourceText,
    Summary,
)
from mustrum.core.normalize import title_hash


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


def make_source(**overrides) -> Source:
    defaults = dict(
        kind=SourceKind.PAPER,
        title="Attention Is All You Need",
        authors=("Vaswani", "Shazeer"),
        year=2017,
        doi="10.1000/example",
        arxiv_id="1706.03762",
        provenance=(("title", FieldOrigin.FETCHED), ("year", FieldOrigin.USER)),
    )
    defaults.update(overrides)
    return Source(**defaults)


class TestSources:
    def test_add_and_get_roundtrip(self, repo):
        saved = repo.add_source(make_source())
        assert saved.id is not None
        assert repo.get_source(saved.id) == saved

    def test_get_missing_raises(self, repo):
        with pytest.raises(KeyError):
            repo.get_source(999)

    def test_list_sources_ordered(self, repo):
        a = repo.add_source(make_source(title="A", doi=None, arxiv_id=None))
        b = repo.add_source(make_source(title="B", doi=None, arxiv_id=None))
        assert repo.list_sources() == [a, b]

    def test_find_by_doi_normalises(self, repo):
        saved = repo.add_source(make_source(doi="https://doi.org/10.1000/Example"))
        assert saved.doi == "10.1000/example"
        assert repo.find_source_by_doi("DOI:10.1000/EXAMPLE") == saved
        assert repo.find_source_by_doi("10.9999/other") is None

    def test_find_by_arxiv_id(self, repo):
        saved = repo.add_source(make_source())
        assert repo.find_source_by_arxiv_id("1706.03762") == saved
        assert repo.find_source_by_arxiv_id("0000.00000") is None

    def test_find_by_title_hash(self, repo):
        saved = repo.add_source(make_source())
        assert repo.find_source_by_title_hash(title_hash("attention is ALL you need!")) == saved
        assert repo.find_source_by_title_hash(title_hash("other")) is None

    def test_duplicate_doi_rejected(self, repo):
        repo.add_source(make_source())
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_source(make_source(title="Different", arxiv_id=None))

    def test_duplicate_arxiv_id_rejected(self, repo):
        repo.add_source(make_source())
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_source(make_source(title="Different", doi=None))

    def test_reading_status_update(self, repo):
        saved = repo.add_source(make_source())
        repo.set_reading_status(saved.id, ReadingStatus.READ)
        assert repo.get_source(saved.id).reading_status == ReadingStatus.READ

    def test_reading_status_missing_source_raises(self, repo):
        with pytest.raises(KeyError):
            repo.set_reading_status(999, ReadingStatus.READ)

    def test_notes_update(self, repo):
        saved = repo.add_source(make_source())
        repo.set_source_notes(saved.id, "re-read section 3")
        assert repo.get_source(saved.id).notes == "re-read section 3"


class TestSourceTextImmutability:
    def test_roundtrip(self, repo):
        s = repo.add_source(make_source())
        st = SourceText(source_id=s.id, text="full text", extraction_method="plaintext")
        repo.add_source_text(st)
        assert repo.get_source_text(s.id) == st

    def test_missing_returns_none(self, repo):
        s = repo.add_source(make_source())
        assert repo.get_source_text(s.id) is None

    def test_second_insert_for_same_source_rejected(self, repo):
        s = repo.add_source(make_source())
        repo.add_source_text(SourceText(source_id=s.id, text="v1", extraction_method="x"))
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_source_text(SourceText(source_id=s.id, text="v2", extraction_method="x"))

    def test_sql_update_blocked_by_trigger(self, repo):
        s = repo.add_source(make_source())
        repo.add_source_text(SourceText(source_id=s.id, text="v1", extraction_method="x"))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            repo._conn.execute("UPDATE source_texts SET text = 'tampered'")

    def test_sql_delete_blocked_by_trigger(self, repo):
        s = repo.add_source(make_source())
        repo.add_source_text(SourceText(source_id=s.id, text="v1", extraction_method="x"))
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            repo._conn.execute("DELETE FROM source_texts")


class TestSummaries:
    def test_roundtrip(self, repo):
        s = repo.add_source(make_source())
        summary = Summary(
            source_id=s.id,
            text="The authors propose the Transformer.",
            evidence=("we propose the Transformer",),
            model="qwen3:30b",
            verified=True,
        )
        repo.set_summary(summary)
        assert repo.get_summary(s.id) == summary

    def test_missing_returns_none(self, repo):
        s = repo.add_source(make_source())
        assert repo.get_summary(s.id) is None

    def test_set_replaces_existing(self, repo):
        s = repo.add_source(make_source())
        first = Summary(source_id=s.id, text="v1", evidence=(), model="m", verified=True)
        second = Summary(
            source_id=s.id, text="v2", evidence=(), model="user", verified=True, user_override=True
        )
        repo.set_summary(first)
        repo.set_summary(second)
        got = repo.get_summary(s.id)
        assert got.text == "v2"
        assert got.user_override is True


class TestIdeas:
    def test_add_get_list(self, repo):
        idea = repo.add_idea(Idea(title="Graph-based related work"))
        assert repo.get_idea(idea.id) == idea
        assert repo.list_ideas() == [idea]

    def test_get_missing_raises(self, repo):
        with pytest.raises(KeyError):
            repo.get_idea(42)

    def test_versions_are_append_only_and_ordered(self, repo):
        idea = repo.add_idea(Idea(title="t"))
        v1 = repo.add_idea_version(IdeaVersion(idea_id=idea.id, text="first draft"))
        v2 = repo.add_idea_version(IdeaVersion(idea_id=idea.id, text="refined"))
        assert repo.get_idea_versions(idea.id) == [v1, v2]
        assert repo.latest_idea_version(idea.id) == v2

    def test_latest_version_none_when_no_versions(self, repo):
        idea = repo.add_idea(Idea(title="t"))
        assert repo.latest_idea_version(idea.id) is None

    def test_version_for_missing_idea_raises(self, repo):
        with pytest.raises(KeyError):
            repo.add_idea_version(IdeaVersion(idea_id=99, text="x"))

    def test_idea_links(self, repo):
        a = repo.add_idea(Idea(title="a"))
        b = repo.add_idea(Idea(title="b"))
        c = repo.add_idea(Idea(title="c"))
        link_ab = IdeaLink(from_idea_id=a.id, to_idea_id=b.id, relation=IdeaRelation.BUILDS_ON)
        link_bc = IdeaLink(from_idea_id=b.id, to_idea_id=c.id, relation=IdeaRelation.RELATED)
        repo.add_idea_link(link_ab)
        repo.add_idea_link(link_bc)
        assert set(repo.list_idea_links()) == {link_ab, link_bc}
        assert repo.list_idea_links(a.id) == [link_ab]
        assert set(repo.list_idea_links(b.id)) == {link_ab, link_bc}


class TestMatches:
    def _setup(self, repo):
        idea = repo.add_idea(Idea(title="t"))
        source = repo.add_source(make_source())
        return idea, source

    def test_add_and_list(self, repo):
        idea, source = self._setup(repo)
        m = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.9))
        assert repo.list_matches(idea.id) == [m]

    def test_list_ordered_by_score_desc(self, repo):
        idea, source = self._setup(repo)
        s2 = repo.add_source(make_source(title="Other", doi=None, arxiv_id=None))
        low = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.2))
        high = repo.add_match(Match(idea_id=idea.id, source_id=s2.id, score=0.8))
        assert repo.list_matches(idea.id) == [high, low]

    def test_status_workflow_and_filter(self, repo):
        idea, source = self._setup(repo)
        m = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.5))
        repo.set_match_status(m.id, MatchStatus.CONFIRMED)
        assert repo.list_matches(idea.id, MatchStatus.CONFIRMED)[0].status == MatchStatus.CONFIRMED
        assert repo.list_matches(idea.id, MatchStatus.REJECTED) == []

    def test_set_status_missing_match_raises(self, repo):
        with pytest.raises(KeyError):
            repo.set_match_status(123, MatchStatus.CONFIRMED)

    def test_duplicate_idea_source_pair_rejected(self, repo):
        idea, source = self._setup(repo)
        repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.5))
        with pytest.raises(sqlite3.IntegrityError):
            repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.6))

    def test_quotes_roundtrip(self, repo):
        idea, source = self._setup(repo)
        m = repo.add_match(
            Match(
                idea_id=idea.id,
                source_id=source.id,
                score=0.7,
                rationale="uses attention",
                quotes=("attention mechanism", "sequence transduction"),
            )
        )
        got = repo.list_matches(idea.id)[0]
        assert got.quotes == m.quotes
        assert got.rationale == m.rationale


class TestBibliography:
    def test_roundtrip_byte_exact(self, repo):
        s = repo.add_source(make_source())
        raw = "@article{vaswani2017attention,\n  title = {Attention Is All You Need}\n}"
        entry = BibEntry(
            source_id=s.id,
            citation_key="vaswani2017attention",
            raw_bibtex=raw,
            origin=BibOrigin.FETCHED,
        )
        repo.set_bib_entry(entry)
        assert repo.get_bib_entry(s.id).raw_bibtex == raw
        assert repo.get_bib_entry_by_key("vaswani2017attention") == entry

    def test_citation_keys(self, repo):
        s = repo.add_source(make_source())
        repo.set_bib_entry(
            BibEntry(
                source_id=s.id, citation_key="k1", raw_bibtex="@misc{k1,}", origin=BibOrigin.DERIVED
            )
        )
        assert repo.citation_keys() == {"k1"}

    def test_duplicate_citation_key_rejected(self, repo):
        s1 = repo.add_source(make_source())
        s2 = repo.add_source(make_source(title="Other", doi=None, arxiv_id=None))
        repo.set_bib_entry(
            BibEntry(source_id=s1.id, citation_key="k", raw_bibtex="x", origin=BibOrigin.DERIVED)
        )
        with pytest.raises(sqlite3.IntegrityError):
            repo.set_bib_entry(
                BibEntry(
                    source_id=s2.id, citation_key="k", raw_bibtex="y", origin=BibOrigin.DERIVED
                )
            )

    def test_missing_lookups_return_none(self, repo):
        assert repo.get_bib_entry(1) is None
        assert repo.get_bib_entry_by_key("nope") is None


class TestContacts:
    def test_roundtrip(self, repo):
        c = repo.add_contact(
            Contact(name="Ada Lovelace", kind=ContactKind.PERSON, affiliation="Analytical Engines")
        )
        assert repo.get_contact(c.id) == c
        assert repo.list_contacts() == [c]

    def test_get_missing_raises(self, repo):
        with pytest.raises(KeyError):
            repo.get_contact(7)

    def test_links_to_idea_and_source(self, repo):
        c = repo.add_contact(Contact(name="X", kind=ContactKind.UNIVERSITY))
        idea = repo.add_idea(Idea(title="t"))
        source = repo.add_source(make_source())
        li = ContactLink(contact_id=c.id, why="potential collaborator", idea_id=idea.id)
        ls = ContactLink(contact_id=c.id, why="author", source_id=source.id)
        repo.add_contact_link(li)
        repo.add_contact_link(ls)
        assert repo.list_contact_links(idea_id=idea.id) == [li]
        assert repo.list_contact_links(source_id=source.id) == [ls]
        assert set(repo.list_contact_links()) == {li, ls}


class TestTags:
    def test_tag_untag(self, repo):
        s = repo.add_source(make_source())
        repo.tag(EntityKind.SOURCE, s.id, "transformers")
        repo.tag(EntityKind.SOURCE, s.id, "nlp")
        assert repo.tags_for(EntityKind.SOURCE, s.id) == {"transformers", "nlp"}
        repo.untag(EntityKind.SOURCE, s.id, "nlp")
        assert repo.tags_for(EntityKind.SOURCE, s.id) == {"transformers"}

    def test_tag_is_idempotent(self, repo):
        s = repo.add_source(make_source())
        repo.tag(EntityKind.SOURCE, s.id, "x")
        repo.tag(EntityKind.SOURCE, s.id, "x")
        assert repo.tags_for(EntityKind.SOURCE, s.id) == {"x"}

    def test_entities_with_tag_spans_kinds(self, repo):
        s = repo.add_source(make_source())
        idea = repo.add_idea(Idea(title="t"))
        repo.tag(EntityKind.SOURCE, s.id, "ml")
        repo.tag(EntityKind.IDEA, idea.id, "ml")
        assert set(repo.entities_with_tag("ml")) == {
            (EntityKind.SOURCE, s.id),
            (EntityKind.IDEA, idea.id),
        }


class TestEmbeddings:
    def test_roundtrip_exact(self, repo):
        s = repo.add_source(make_source())
        emb = Embedding(
            entity=EntityKind.SOURCE,
            ref_id=s.id,
            chunk_index=0,
            model="nomic-embed-text",
            vector=(0.1, -2.5, 3.25),
        )
        repo.store_embeddings([emb])
        assert repo.embeddings_for(EntityKind.SOURCE, "nomic-embed-text") == [emb]

    def test_upsert_replaces_vector(self, repo):
        s = repo.add_source(make_source())
        first = Embedding(EntityKind.SOURCE, s.id, 0, "m", (1.0,))
        second = Embedding(EntityKind.SOURCE, s.id, 0, "m", (2.0,))
        repo.store_embeddings([first])
        repo.store_embeddings([second])
        assert repo.embeddings_for(EntityKind.SOURCE, "m") == [second]

    def test_filter_by_model(self, repo):
        s = repo.add_source(make_source())
        repo.store_embeddings([Embedding(EntityKind.SOURCE, s.id, 0, "old-model", (1.0,))])
        assert repo.embeddings_for(EntityKind.SOURCE, "new-model") == []

    def test_delete(self, repo):
        s = repo.add_source(make_source())
        repo.store_embeddings([Embedding(EntityKind.SOURCE, s.id, 0, "m", (1.0,))])
        repo.delete_embeddings(EntityKind.SOURCE, s.id)
        assert repo.embeddings_for(EntityKind.SOURCE, "m") == []


class TestSearch:
    def test_finds_source_by_title(self, repo):
        s = repo.add_source(make_source())
        hits = repo.search("attention")
        assert [(h.entity, h.ref_id) for h in hits] == [(EntityKind.SOURCE, s.id)]

    def test_finds_source_by_full_text(self, repo):
        s = repo.add_source(make_source())
        repo.add_source_text(
            SourceText(source_id=s.id, text="scaled dot-product kernels", extraction_method="x")
        )
        assert repo.search("kernels")[0].ref_id == s.id

    def test_finds_source_by_summary(self, repo):
        s = repo.add_source(make_source())
        repo.set_summary(
            Summary(
                source_id=s.id,
                text="introduces multihead layers",
                evidence=(),
                model="m",
                verified=True,
            )
        )
        assert repo.search("multihead")[0].ref_id == s.id

    def test_finds_idea_by_version_text(self, repo):
        idea = repo.add_idea(Idea(title="untitled"))
        repo.add_idea_version(IdeaVersion(idea_id=idea.id, text="federated distillation"))
        hits = repo.search("federated")
        assert (hits[0].entity, hits[0].ref_id) == (EntityKind.IDEA, idea.id)

    def test_finds_contact_by_affiliation(self, repo):
        c = repo.add_contact(Contact(name="N", kind=ContactKind.PERSON, affiliation="ZHAW"))
        hits = repo.search("zhaw")
        assert (hits[0].entity, hits[0].ref_id) == (EntityKind.CONTACT, c.id)

    def test_multiword_query_is_anded(self, repo):
        repo.add_source(make_source(title="Attention Networks", doi=None, arxiv_id=None))
        repo.add_source(make_source(title="Graph Networks", doi=None, arxiv_id=None))
        hits = repo.search("attention networks")
        assert len(hits) == 1

    def test_malicious_fts_syntax_is_neutralised(self, repo):
        repo.add_source(make_source())
        # raw FTS5 operators/quotes must not raise
        assert repo.search('"unbalanced AND NEAR(') == []

    def test_empty_query_returns_nothing(self, repo):
        repo.add_source(make_source())
        assert repo.search("   ") == []

    def test_no_stale_index_after_notes_update(self, repo):
        s = repo.add_source(make_source())
        repo.set_source_notes(s.id, "quantum")
        repo.set_source_notes(s.id, "classical")
        assert repo.search("quantum") == []
        assert repo.search("classical")[0].ref_id == s.id


class TestMigrations:
    def test_user_version_set(self, repo):
        assert repo._conn.execute("PRAGMA user_version").fetchone()[0] == 1

    def test_reopening_existing_db_is_idempotent(self, tmp_path):
        db = tmp_path / "test.db"
        r1 = SqliteRepo(db)
        s = r1.add_source(make_source())
        r1.close()
        r2 = SqliteRepo(db)
        assert r2.get_source(s.id) == s
        r2.close()


class TestUpdateSource:
    def test_updates_metadata(self, repo):
        saved = repo.add_source(make_source(doi=None, arxiv_id=None))
        import dataclasses

        updated = dataclasses.replace(saved, year=2020, doi="10.1/new")
        repo.update_source(updated)
        got = repo.get_source(saved.id)
        assert got.year == 2020
        assert got.doi == "10.1/new"

    def test_update_without_id_raises(self, repo):
        with pytest.raises(ValueError):
            repo.update_source(make_source())

    def test_update_missing_source_raises(self, repo):
        import dataclasses

        ghost = dataclasses.replace(make_source(), id=999)
        with pytest.raises(KeyError):
            repo.update_source(ghost)

    def test_update_reindexes_search(self, repo):
        import dataclasses

        saved = repo.add_source(make_source(doi=None, arxiv_id=None))
        repo.update_source(dataclasses.replace(saved, title="Zephyr Quantum Widgets"))
        assert repo.search("zephyr")[0].ref_id == saved.id
        assert repo.search("attention") == []


class TestFindIdeaByTitle:
    def test_finds_exact_title(self, repo):
        idea = repo.add_idea(Idea(title="Grounded RAG"))
        assert repo.find_idea_by_title("Grounded RAG") == idea

    def test_title_is_case_sensitive_and_missing_returns_none(self, repo):
        repo.add_idea(Idea(title="Grounded RAG"))
        assert repo.find_idea_by_title("grounded rag") is None
        assert repo.find_idea_by_title("nope") is None

    def test_duplicate_titles_return_oldest(self, repo):
        first = repo.add_idea(Idea(title="Same"))
        repo.add_idea(Idea(title="Same"))
        assert repo.find_idea_by_title("Same") == first
