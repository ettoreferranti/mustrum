import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.services.brainstorm import BrainstormFailure, BrainstormService
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.summarise import SummariseService


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def library(repo):
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    source = ingest.ingest_document(
        title="Graph networks for molecules",
        text="message passing on molecular graphs predicts properties",
        extraction_method="plaintext",
    ).source
    SummariseService(repo, FakeLLMProvider()).override(source.id, "GNNs predict properties.")
    IdeaService(repo, embedder).create("molecular ML", "apply graph networks to chemistry")
    return repo


def reply(entries):
    return json.dumps({"ideas": entries})


GOOD = [
    {
        "title": "Uncertainty-aware GNNs",
        "description": "Combine graph networks with calibrated uncertainty for chemistry.",
        "based_on": ["Graph networks for molecules"],
    },
    {
        "title": "Cross-domain transfer",
        "description": "Transfer molecular representations to materials science.",
        "based_on": [],
    },
]


class TestPropose:
    def test_returns_proposals_with_resolved_inspirations(self, library):
        service = BrainstormService(library, FakeLLMProvider([reply(GOOD)]))
        proposals = service.propose(count=2)
        assert [p.title for p in proposals] == ["Uncertainty-aware GNNs", "Cross-domain transfer"]
        assert proposals[0].inspirations == ("Graph networks for molecules",)
        assert proposals[1].inspirations == ()

    def test_invented_inspiration_titles_are_dropped(self, library):
        entries = [
            {
                "title": "T",
                "description": "D.",
                "based_on": ["A Paper That Does Not Exist", "Graph networks for molecules"],
            }
        ]
        service = BrainstormService(library, FakeLLMProvider([reply(entries)]))
        (proposal,) = service.propose(count=1)
        assert proposal.inspirations == ("Graph networks for molecules",)

    def test_inspiration_matching_is_case_insensitive_but_returns_real_title(self, library):
        entries = [
            {"title": "T", "description": "D.", "based_on": ["graph NETWORKS for molecules"]}
        ]
        service = BrainstormService(library, FakeLLMProvider([reply(entries)]))
        (proposal,) = service.propose(count=1)
        assert proposal.inspirations == ("Graph networks for molecules",)

    def test_existing_idea_titles_resolve_too(self, library):
        entries = [{"title": "T", "description": "D.", "based_on": ["molecular ML"]}]
        service = BrainstormService(library, FakeLLMProvider([reply(entries)]))
        (proposal,) = service.propose(count=1)
        assert proposal.inspirations == ("molecular ML",)

    def test_invalid_entries_dropped_valid_kept(self, library):
        entries = [
            {"title": "", "description": "no title"},
            {"title": "No description", "description": "   "},
            "not even a dict",
            {"title": "Valid", "description": "Keeps this one.", "based_on": "not-a-list"},
        ]
        service = BrainstormService(library, FakeLLMProvider([reply(entries)]))
        proposals = service.propose(count=5)
        assert [p.title for p in proposals] == ["Valid"]

    def test_count_caps_proposals(self, library):
        service = BrainstormService(library, FakeLLMProvider([reply(GOOD)]))
        assert len(service.propose(count=1)) == 1

    def test_prompt_contains_library_and_focus(self, library):
        llm = FakeLLMProvider([reply(GOOD)])
        BrainstormService(library, llm).propose(count=2, focus="sustainability")
        prompt, system = llm.calls[0]
        assert "Graph networks for molecules" in prompt
        assert "GNNs predict properties." in prompt
        assert "molecular ML" in prompt
        assert "Focus area: sustainability" in prompt
        assert "never invent" in system

    def test_requests_structured_output_schema(self, library):
        """E3-5/ADR-14: shape is constrained (pinned in full — every key
        steers constrained sampling); the based_on quarantine stays."""
        llm = FakeLLMProvider([reply(GOOD)])
        BrainstormService(library, llm).propose()
        (schema,) = llm.schemas
        assert schema == {
            "type": "object",
            "properties": {
                "ideas": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "based_on": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["title", "description"],
                    },
                }
            },
            "required": ["ideas"],
        }

    def test_unparsable_reply_retries_then_fails(self, library):
        llm = FakeLLMProvider(["junk", "more junk", "still junk"])
        service = BrainstormService(library, llm, attempts=3)
        with pytest.raises(BrainstormFailure, match=r"after 3 attempts.*junk"):
            service.propose()
        assert len(llm.calls) == 3
        assert "not usable" in llm.calls[1][0]  # corrective feedback on retry

    def test_bad_then_good_reply_succeeds(self, library):
        llm = FakeLLMProvider(["junk", reply(GOOD)])
        assert BrainstormService(library, llm, attempts=2).propose(count=2)

    def test_empty_library_raises(self, repo):
        service = BrainstormService(repo, FakeLLMProvider())
        with pytest.raises(LookupError, match="library is empty"):
            service.propose()

    def test_nothing_is_stored(self, library):
        before = [i.title for i in library.list_ideas()]
        BrainstormService(library, FakeLLMProvider([reply(GOOD)])).propose(count=2)
        assert [i.title for i in library.list_ideas()] == before
