import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import (
    Contact,
    ContactKind,
    ContactLink,
    EntityKind,
    FetchedMetadata,
    IdeaRelation,
    Match,
    MatchStatus,
    ReadingStatus,
    Summary,
)
from mustrum.core.services.backup import FORMAT_VERSION, BackupError, BackupService
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


def build_rich_library(repo):
    """Sources (fetched + manual), summary, ideas w/ history, links, matches
    with rationale, contacts, tags, notes — everything export must carry."""
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    ideas = IdeaService(repo, embedder)
    manual = ingest.ingest_document(
        title="Manual notes on distillation",
        text="para one\n\npara two about distillation",
        extraction_method="plaintext",
        authors=("Carol Writer",),
        year=2019,
    ).source
    fetched = ingest.ingest_fetched(
        FetchedMetadata(
            title="Graph networks for molécules",
            authors=("Ada Ideasmith", "Bob Co"),
            year=2021,
            doi="10.1/graphs",
            arxiv_id="2101.00001",
            raw_bibtex="@article{DBLP:conf/x/21,\n title={Graph networks}\n}",
            abstract="graph networks molecules chemistry",
        )
    ).source
    repo.set_summary(
        Summary(
            source_id=fetched.id,
            text="They apply graph networks.",
            evidence=("graph networks",),
            model="qwen3:30b",
            verified=True,
        )
    )
    repo.set_reading_status(fetched.id, ReadingStatus.READ)
    repo.set_source_notes(fetched.id, "great baselines")
    repo.tag(EntityKind.SOURCE, fetched.id, "gnn")
    repo.tag(EntityKind.SOURCE, fetched.id, "chemistry")
    idea_a = ideas.create("molecular ML", "first draft")
    ideas.revise(idea_a.id, "refined: graph networks for molecules")
    idea_b = ideas.create("spinoff", "another direction")
    ideas.link(idea_b.id, idea_a.id, IdeaRelation.BUILDS_ON)
    repo.tag(EntityKind.IDEA, idea_a.id, "active")
    match = repo.add_match(
        Match(
            idea_id=idea_a.id,
            source_id=fetched.id,
            score=0.82,
            status=MatchStatus.CONFIRMED,
            rationale="directly relevant",
            quotes=("graph networks",),
        )
    )
    repo.set_summary(
        Summary(
            source_id=manual.id,
            text="My own words.",
            evidence=(),
            model="user",
            verified=True,
            user_override=True,
        )
    )
    repo.add_match(
        Match(idea_id=idea_b.id, source_id=manual.id, score=0.4, status=MatchStatus.SUGGESTED)
    )
    ingest.ingest_document(title="Metadata-only record", text="", extraction_method="plaintext")
    contact = repo.add_contact(
        Contact(
            name="Prof X",
            kind=ContactKind.UNIVERSITY,
            affiliation="Unseen University",
            email="profx@example.org",
            url="https://uu.example.org/profx",
            notes="met at conference",
        )
    )
    repo.add_contact(Contact(name="Acme Labs", kind=ContactKind.COMPANY))
    repo.add_contact_link(ContactLink(contact_id=contact.id, why="expert", idea_id=idea_a.id))
    repo.add_contact_link(ContactLink(contact_id=contact.id, why="author", source_id=manual.id))
    return fetched, manual, idea_a, idea_b, match, contact


@pytest.fixture
def bundle(repo):
    build_rich_library(repo)
    return BackupService(repo, FakeEmbeddingProvider()).export_data()


class TestExport:
    def test_manifest_and_counts(self, bundle):
        manifest = json.loads(bundle["manifest.json"])
        assert manifest["format"] == FORMAT_VERSION
        assert manifest["counts"] == {"sources": 3, "ideas": 2, "matches": 2, "contacts": 2}

    def test_bibtex_byte_exact(self, bundle):
        assert (
            bundle["bib/DBLP_conf_x_21.bib"]
            == "@article{DBLP:conf/x/21,\n title={Graph networks}\n}"
        )

    def test_texts_exported_verbatim(self, bundle):
        texts = [v for k, v in bundle.items() if k.startswith("texts/")]
        assert "para one\n\npara two about distillation" in texts

    def test_ideas_md_is_idea_import_compatible(self, bundle):
        from mustrum.core.services.ideas import parse_ideas_file

        parsed = parse_ideas_file(bundle["ideas.md"])
        assert ("molecular ML", "refined: graph networks for molecules") in parsed

    def test_library_md_overview(self, bundle):
        md = bundle["LIBRARY.md"]
        assert "Graph networks for molécules" in md
        assert "They apply graph networks." in md
        assert "molecular ML" in md

    def test_no_embeddings_in_export(self, bundle):
        assert not any("embed" in k for k in bundle)

    def test_json_is_stable_sorted(self, bundle):
        assert (
            bundle["sources.json"]
            == json.dumps(
                json.loads(bundle["sources.json"]), indent=2, ensure_ascii=False, sort_keys=True
            )
            + "\n"
        )


class TestRoundTrip:
    def test_full_round_trip(self, repo, bundle):
        target = SqliteRepo(":memory:")
        counts = BackupService(target, FakeEmbeddingProvider()).import_data(bundle)
        assert counts == {"sources": 3, "ideas": 2, "matches": 2, "contacts": 2}

        # sources with all attributes
        sources = {s.title: s for s in target.list_sources()}
        fetched = sources["Graph networks for molécules"]
        assert fetched.authors == ("Ada Ideasmith", "Bob Co")
        assert fetched.doi == "10.1/graphs"
        assert fetched.reading_status == ReadingStatus.READ
        assert fetched.notes == "great baselines"
        assert dict(fetched.provenance)["title"].value == "fetched"
        assert target.tags_for(EntityKind.SOURCE, fetched.id) == {"gnn", "chemistry"}
        # verbatim text + summary + bib
        assert target.get_source_text(fetched.id).text == "graph networks molecules chemistry"
        summary = target.get_summary(fetched.id)
        assert summary.text == "They apply graph networks."
        assert summary.verified is True
        bib = target.get_bib_entry(fetched.id)
        assert bib.citation_key == "DBLP:conf/x/21"
        assert bib.raw_bibtex.startswith("@article{DBLP:conf/x/21,")
        # ideas with history + links + tags
        ideas = {i.title: i for i in target.list_ideas()}
        versions = target.get_idea_versions(ideas["molecular ML"].id)
        assert [v.text for v in versions] == [
            "first draft",
            "refined: graph networks for molecules",
        ]
        links = target.list_idea_links()
        assert links[0].relation == IdeaRelation.BUILDS_ON
        assert target.tags_for(EntityKind.IDEA, ideas["molecular ML"].id) == {"active"}
        # matches with rationale
        matches = target.list_matches()
        assert len(matches) == 2
        confirmed = next(m for m in matches if m.status == MatchStatus.CONFIRMED)
        assert confirmed.rationale == "directly relevant"
        assert confirmed.quotes == ("graph networks",)
        # user-override summary survives with its flags
        manual_restored = sources["Manual notes on distillation"]
        override = target.get_summary(manual_restored.id)
        assert override.user_override is True
        assert override.model == "user"
        # the bib belongs to the *second* source — a mis-attached bib cannot hide
        assert target.get_bib_entry(manual_restored.id) is None
        # contacts + both link kinds
        contacts = {c.name: c for c in target.list_contacts()}
        assert contacts["Prof X"].email == "profx@example.org"
        assert contacts["Prof X"].url == "https://uu.example.org/profx"
        assert contacts["Prof X"].notes == "met at conference"
        # a text-less source restores as text-less
        bare = sources["Metadata-only record"]
        assert target.get_source_text(bare.id) is None
        assert contacts["Acme Labs"].kind == ContactKind.COMPANY
        contact_links = target.list_contact_links()
        assert {link.why for link in contact_links} == {"expert", "author"}
        # embeddings recomputed, idea embedding composed from title + latest text
        assert target.embeddings_for(EntityKind.SOURCE, "fake-embed")
        idea_embeddings = {
            e.ref_id: e.vector for e in target.embeddings_for(EntityKind.IDEA, "fake-embed")
        }
        expected = FakeEmbeddingProvider().embed(
            ["molecular ML\n\nrefined: graph networks for molecules"]
        )[0]
        assert idea_embeddings[ideas["molecular ML"].id] == expected
        target.close()

    def test_timestamps_preserved(self, repo, bundle):
        target = SqliteRepo(":memory:")
        BackupService(target, FakeEmbeddingProvider()).import_data(bundle)
        original = {s.title: s.created_at for s in repo.list_sources()}
        restored = {s.title: s.created_at for s in target.list_sources()}
        assert original == restored
        target.close()

    def test_reexport_is_identical(self, repo, bundle):
        """export → restore → export must be byte-identical (modulo ids)."""
        target = SqliteRepo(":memory:")
        BackupService(target, FakeEmbeddingProvider()).import_data(bundle)
        second = BackupService(target, FakeEmbeddingProvider()).export_data()
        assert second == bundle
        target.close()


class TestRestoreGuards:
    def test_refuses_non_empty_database(self, repo, bundle):
        with pytest.raises(BackupError, match="empty database"):
            BackupService(repo, FakeEmbeddingProvider()).import_data(bundle)

    def test_rejects_unknown_format_version(self, bundle):
        target = SqliteRepo(":memory:")
        broken = dict(bundle)
        broken["manifest.json"] = json.dumps({"format": 99})
        with pytest.raises(BackupError, match="unsupported export format 99"):
            BackupService(target, FakeEmbeddingProvider()).import_data(broken)
        target.close()

    def test_missing_manifest(self):
        target = SqliteRepo(":memory:")
        with pytest.raises(BackupError, match=r"missing manifest\.json"):
            BackupService(target, FakeEmbeddingProvider()).import_data({})
        target.close()

    def test_missing_referenced_text_file(self, bundle):
        target = SqliteRepo(":memory:")
        broken = {k: v for k, v in bundle.items() if not k.startswith("texts/")}
        with pytest.raises(BackupError, match="missing texts/"):
            BackupService(target, FakeEmbeddingProvider()).import_data(broken)
        target.close()

    def test_corrupt_json(self, bundle):
        target = SqliteRepo(":memory:")
        broken = dict(bundle)
        broken["sources.json"] = "{not json"
        with pytest.raises(BackupError, match="not valid JSON"):
            BackupService(target, FakeEmbeddingProvider()).import_data(broken)
        target.close()


class TestFilenameSanitisation:
    def test_unsafe_citation_key_gets_safe_filename(self, bundle):
        assert "bib/DBLP_conf_x_21.bib" in bundle  # DBLP:conf/x/21 sanitised


class TestGuardsAndEncoding:
    def test_restore_refused_when_only_ideas_exist(self, bundle):
        target = SqliteRepo(":memory:")
        IdeaService(target, FakeEmbeddingProvider()).create("pre-existing", "text")
        with pytest.raises(BackupError, match="empty database"):
            BackupService(target, FakeEmbeddingProvider()).import_data(bundle)
        target.close()

    def test_unicode_kept_readable_in_json(self, bundle):
        assert "molécules" in bundle["sources.json"]  # not \u00e9-escaped

    def test_colliding_citation_key_filenames_disambiguated(self, repo):
        from mustrum.core.models import BibEntry, BibOrigin, Source, SourceKind

        a = repo.add_source(Source(kind=SourceKind.PAPER, title="A"))
        b = repo.add_source(Source(kind=SourceKind.PAPER, title="B"))
        repo.set_bib_entry(
            BibEntry(
                source_id=a.id,
                citation_key="x:y",
                raw_bibtex="@misc{x:y,}",
                origin=BibOrigin.DERIVED,
            )
        )
        repo.set_bib_entry(
            BibEntry(
                source_id=b.id,
                citation_key="x/y",
                raw_bibtex="@misc{x/y,}",
                origin=BibOrigin.DERIVED,
            )
        )
        files = BackupService(repo, FakeEmbeddingProvider()).export_data()
        assert "bib/x_y.bib" in files
        assert "bib/x_y-2.bib" in files
        assert files["bib/x_y.bib"] != files["bib/x_y-2.bib"]

    def test_fully_unsafe_key_falls_back_to_entry(self, repo):
        from mustrum.core.models import BibEntry, BibOrigin, Source, SourceKind

        a = repo.add_source(Source(kind=SourceKind.PAPER, title="A"))
        repo.set_bib_entry(
            BibEntry(
                source_id=a.id, citation_key="", raw_bibtex="@misc{k,}", origin=BibOrigin.DERIVED
            )
        )
        files = BackupService(repo, FakeEmbeddingProvider()).export_data()
        assert "bib/entry.bib" in files
