import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import Match
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.rationale import RationaleFailure, RationaleService

PAPER_TEXT = (
    "We propose message passing networks that operate on molecular graphs. "
    "Our method predicts quantum chemical properties with high accuracy."
)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def match(repo):
    embedder = FakeEmbeddingProvider()
    source = (
        IngestService(repo, embedder)
        .ingest_document(
            title="Neural message passing", text=PAPER_TEXT, extraction_method="plaintext"
        )
        .source
    )
    idea = IdeaService(repo, embedder).create(
        "molecular ML", "predict molecule properties with graph networks"
    )
    return repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.8))


def good_reply(rationale="Directly applies graph networks to molecule property prediction."):
    return json.dumps({"rationale": rationale, "quotes": ["operate on molecular graphs"]})


class TestExplain:
    def test_verified_rationale_stored(self, repo, match):
        service = RationaleService(repo, FakeLLMProvider([good_reply()]))
        result = service.explain(match.id)
        assert result.rationale == (
            "Directly applies graph networks to molecule property prediction."
        )
        assert result.quotes == ("operate on molecular graphs",)
        assert repo.get_match(match.id).rationale == result.rationale

    def test_prompt_contains_idea_and_paper(self, repo, match):
        llm = FakeLLMProvider([good_reply()])
        RationaleService(repo, llm).explain(match.id)
        prompt, system = llm.calls[0]
        assert "molecular ML" in prompt
        assert "predict molecule properties with graph networks" in prompt
        assert "Neural message passing" in prompt
        assert "We propose message passing networks" in prompt
        assert "EXACTLY" in system

    def test_existing_rationale_returned_without_llm_call(self, repo, match):
        RationaleService(repo, FakeLLMProvider([good_reply()])).explain(match.id)
        llm2 = FakeLLMProvider()  # would raise if called
        again = RationaleService(repo, llm2).explain(match.id)
        assert again.rationale
        assert llm2.calls == []

    def test_force_regenerates(self, repo, match):
        service = RationaleService(
            repo, FakeLLMProvider([good_reply("first"), good_reply("second")])
        )
        service.explain(match.id)
        assert service.explain(match.id, force=True).rationale == "second"

    def test_fabricated_quotes_rejected_nothing_stored(self, repo, match):
        bad = json.dumps({"rationale": "r", "quotes": ["totally invented span"]})
        service = RationaleService(repo, FakeLLMProvider([bad]), attempts=1)
        with pytest.raises(
            RationaleFailure,
            match=(
                f"match {match.id} failed grounding after 1 attempts: quotes not found in source"
            ),
        ) as exc:
            service.explain(match.id)
        assert exc.value.last_result is not None
        assert exc.value.raw_output == ""
        assert repo.get_match(match.id).rationale == ""

    def test_retry_feedback_names_missing_quotes(self, repo, match):
        bad = json.dumps({"rationale": "r", "quotes": ["invented span"]})
        llm = FakeLLMProvider([bad, good_reply()])
        RationaleService(repo, llm, attempts=2).explain(match.id)
        second_prompt, _ = llm.calls[1]
        assert "invented span" in second_prompt

    def test_missing_match_raises(self, repo):
        with pytest.raises(KeyError):
            RationaleService(repo, FakeLLMProvider()).explain(999)

    def test_source_without_text_raises_lookup(self, repo):
        embedder = FakeEmbeddingProvider()
        source = (
            IngestService(repo, embedder)
            .ingest_document(title="No text", text="", extraction_method="plaintext")
            .source
        )
        idea = IdeaService(repo, embedder).create("i", "t")
        m = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.5))
        with pytest.raises(LookupError, match="no stored text"):
            RationaleService(repo, FakeLLMProvider()).explain(m.id)
