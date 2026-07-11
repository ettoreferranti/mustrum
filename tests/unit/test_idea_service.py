import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import EntityKind, IdeaRelation
from mustrum.core.services.ideas import (
    IdeaFileError,
    IdeaService,
    parse_ideas_file,
)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def service(repo):
    return IdeaService(repo, FakeEmbeddingProvider())


class TestIdeaService:
    def test_create_stores_idea_version_and_embedding(self, repo, service):
        idea = service.create("Grounded RAG", "verify quotes against sources")
        assert repo.get_idea(idea.id).title == "Grounded RAG"
        versions = repo.get_idea_versions(idea.id)
        assert [v.text for v in versions] == ["verify quotes against sources"]
        embeddings = repo.embeddings_for(EntityKind.IDEA, "fake-embed")
        assert [e.ref_id for e in embeddings] == [idea.id]

    def test_revise_appends_version_and_reembeds(self, repo, service):
        idea = service.create("t", "first")
        before = repo.embeddings_for(EntityKind.IDEA, "fake-embed")[0].vector
        service.revise(idea.id, "completely different focus on quantum chemistry")
        versions = repo.get_idea_versions(idea.id)
        assert [v.text for v in versions] == [
            "first",
            "completely different focus on quantum chemistry",
        ]
        after = repo.embeddings_for(EntityKind.IDEA, "fake-embed")
        assert len(after) == 1  # replaced, not accumulated
        assert after[0].vector != before

    def test_revise_missing_idea_raises(self, service):
        with pytest.raises(KeyError):
            service.revise(99, "x")

    def test_link(self, repo, service):
        a = service.create("a", "ta")
        b = service.create("b", "tb")
        service.link(a.id, b.id, IdeaRelation.BUILDS_ON)
        links = repo.list_idea_links(a.id)
        assert links[0].relation == IdeaRelation.BUILDS_ON

    def test_self_link_rejected(self, service):
        a = service.create("a", "t")
        with pytest.raises(ValueError, match="itself"):
            service.link(a.id, a.id, IdeaRelation.RELATED)

    def test_link_missing_idea_raises(self, service):
        a = service.create("a", "t")
        with pytest.raises(KeyError):
            service.link(a.id, 999, IdeaRelation.RELATED)


class TestEmbeddingContract:
    def test_idea_embedding_uses_chunk_zero(self, repo, service):
        service.create("t", "text")
        (embedding,) = repo.embeddings_for(EntityKind.IDEA, "fake-embed")
        assert embedding.chunk_index == 0


class TestParseIdeasFile:
    def test_single_idea(self):
        assert parse_ideas_file("# My Idea\nSome body text.") == [("My Idea", "Some body text.")]

    def test_multiple_ideas(self):
        content = "# First\nbody one\n\n# Second\nbody two\nmore text"
        assert parse_ideas_file(content) == [
            ("First", "body one"),
            ("Second", "body two\nmore text"),
        ]

    def test_leading_blank_lines_allowed(self):
        assert parse_ideas_file("\n\n# T\nbody") == [("T", "body")]

    def test_subheadings_belong_to_body(self):
        content = "# T\nintro\n## Details\nmore"
        assert parse_ideas_file(content) == [("T", "intro\n## Details\nmore")]

    def test_body_whitespace_trimmed(self):
        assert parse_ideas_file("# T\n\n  body  \n\n") == [("T", "body")]

    def test_content_before_first_heading_rejected(self):
        with pytest.raises(IdeaFileError, match=r"line 1.*before the first"):
            parse_ideas_file("stray text\n# T\nbody")

    def test_empty_body_rejected(self):
        with pytest.raises(IdeaFileError, match="'Empty' has no body"):
            parse_ideas_file("# Empty\n\n# Next\nbody")

    def test_empty_body_at_end_rejected(self):
        with pytest.raises(IdeaFileError, match="'Last' has no body"):
            parse_ideas_file("# First\nbody\n# Last\n")

    def test_heading_without_title_rejected(self):
        with pytest.raises(IdeaFileError, match=r"line 1.*no title"):
            parse_ideas_file("# \nbody")

    def test_no_headings_rejected(self):
        with pytest.raises(IdeaFileError, match="no '# title' headings"):
            parse_ideas_file("")

    def test_hash_without_space_is_not_a_heading(self):
        with pytest.raises(IdeaFileError, match="before the first"):
            parse_ideas_file("#NotAHeading\nbody")


IDEAS_MD = "# Alpha\nfirst idea text\n\n# Beta\nsecond idea text"


class TestImportIdeas:
    def test_creates_all_ideas_with_versions_and_embeddings(self, repo, service):
        outcomes = service.import_ideas(IDEAS_MD)
        assert [(o.title, o.action) for o in outcomes] == [
            ("Alpha", "created"),
            ("Beta", "created"),
        ]
        assert [i.title for i in repo.list_ideas()] == ["Alpha", "Beta"]
        alpha_id = outcomes[0].idea_id
        assert repo.latest_idea_version(alpha_id).text == "first idea text"
        embedded_ids = {e.ref_id for e in repo.embeddings_for(EntityKind.IDEA, "fake-embed")}
        assert embedded_ids == {o.idea_id for o in outcomes}

    def test_reimport_same_file_skips_everything(self, repo, service):
        service.import_ideas(IDEAS_MD)
        outcomes = service.import_ideas(IDEAS_MD)
        assert all(o.action == "skipped" for o in outcomes)
        assert len(repo.list_ideas()) == 2

    def test_revise_appends_version_only_when_text_changed(self, repo, service):
        first = service.import_ideas(IDEAS_MD)
        changed = IDEAS_MD.replace("first idea text", "sharpened idea text")
        outcomes = service.import_ideas(changed, on_existing="revise")
        assert [(o.title, o.action) for o in outcomes] == [
            ("Alpha", "revised"),
            ("Beta", "skipped"),
        ]
        alpha_id = first[0].idea_id
        assert [v.text for v in repo.get_idea_versions(alpha_id)] == [
            "first idea text",
            "sharpened idea text",
        ]

    def test_create_mode_allows_duplicate_titles(self, repo, service):
        service.import_ideas(IDEAS_MD)
        outcomes = service.import_ideas(IDEAS_MD, on_existing="create")
        assert all(o.action == "created" for o in outcomes)
        assert len(repo.list_ideas()) == 4

    def test_invalid_file_stores_nothing(self, repo, service):
        with pytest.raises(IdeaFileError):
            service.import_ideas("# Good\nbody\n# Bad\n")
        assert repo.list_ideas() == []

    def test_outcome_ids_match_existing_on_skip(self, service):
        created = service.import_ideas(IDEAS_MD)
        skipped = service.import_ideas(IDEAS_MD)
        assert [o.idea_id for o in skipped] == [o.idea_id for o in created]
