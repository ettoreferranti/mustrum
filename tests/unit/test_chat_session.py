import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.services.chat import ChatSession
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.query import QueryFailure, QueryService

SOLAR_TEXT = (
    "We evaluate photovoltaic panel efficiency under variable cloud cover. "
    "Renewable energy generation from solar arrays improves with tracking mounts."
)
MOLECULE_TEXT = (
    "We propose message passing networks that operate on molecular graphs. "
    "Our method predicts quantum chemical properties with high accuracy."
)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def embedder():
    return FakeEmbeddingProvider()


def ingest(repo, embedder, title, text):
    return (
        IngestService(repo, embedder)
        .ingest_document(title=title, text=text, extraction_method="plaintext")
        .source
    )


def found_reply(source_id, quote, answer="Yes, one source covers this."):
    return json.dumps(
        {"found": True, "answer": answer, "evidence": [{"source_id": source_id, "quote": quote}]}
    )


def not_found_reply(answer="I found nothing."):
    return json.dumps({"found": False, "answer": answer, "evidence": []})


class TestChatSession:
    def test_first_turn_behaves_like_bare_ask(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        llm = FakeLLMProvider(
            [found_reply(source.id, "Renewable energy generation from solar arrays")]
        )
        service = QueryService(repo, llm, embedder, embedder.model_name)
        session = ChatSession(service)
        answer = session.ask("solar")
        assert answer.found
        prompt, _ = llm.calls[0]
        assert "Recent conversation" not in prompt
        assert session.turns == (session.turns[0],)
        assert session.turns[0].question == "solar"
        assert session.turns[0].answer is answer

    def test_second_turn_gets_history_and_sticky_candidate(self, repo, embedder):
        solar = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        # "graphs" is a literal token only in MOLECULE_TEXT, so FTS matches
        # molecule alone on turn 2 — any solar.id present must come from
        # sticky seeding, not natural retrieval
        molecule = ingest(repo, embedder, "Molecular graph networks", MOLECULE_TEXT)
        llm = FakeLLMProvider(
            [
                found_reply(solar.id, "Renewable energy generation from solar arrays", "Yes."),
                found_reply(molecule.id, "operate on molecular graphs"),
            ]
        )
        service = QueryService(repo, llm, embedder, embedder.model_name)
        session = ChatSession(service)
        session.ask("solar")
        session.ask("graphs")

        second_prompt, _ = llm.calls[1]
        assert "Recent conversation" in second_prompt
        assert "Q: solar" in second_prompt
        assert "A: Yes." in second_prompt
        # solar.id was the previous turn's citation — seeded as a sticky
        # candidate even though this question is about something else
        assert solar.id in session.turns[1].answer.considered_source_ids
        assert molecule.id in session.turns[1].answer.considered_source_ids

    def test_failed_turn_not_added_to_history(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = found_reply(source.id, "totally invented span")
        llm = FakeLLMProvider([bad])
        service = QueryService(repo, llm, embedder, embedder.model_name, attempts=1)
        session = ChatSession(service)
        with pytest.raises(QueryFailure):
            session.ask("solar")
        assert session.turns == ()

    def test_history_truncated_to_history_turns(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        # "solar" is a guaranteed FTS hit every turn, so every turn actually
        # calls the LLM regardless of what came before — keeps this test
        # about history-window truncation, not retrieval relevance
        llm = FakeLLMProvider(
            [found_reply(source.id, "Renewable energy generation from solar arrays")] * 5
        )
        service = QueryService(repo, llm, embedder, embedder.model_name)
        session = ChatSession(service, history_turns=2)
        for i in range(4):
            session.ask(f"solar {i}")
        session.ask("solar")

        last_prompt, _ = llm.calls[4]
        # only the last 2 turns (2, 3) should appear; turns 0 and 1 are gone
        assert "Q: solar 2" in last_prompt
        assert "Q: solar 3" in last_prompt
        assert "Q: solar 0" not in last_prompt
        assert "Q: solar 1" not in last_prompt

    def test_reset_clears_turns_and_stops_seeding_history(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        llm = FakeLLMProvider(
            [found_reply(source.id, "Renewable energy generation from solar arrays")] * 2
        )
        service = QueryService(repo, llm, embedder, embedder.model_name)
        session = ChatSession(service)
        session.ask("solar")
        session.reset()
        assert session.turns == ()
        session.ask("solar")
        second_prompt, _ = llm.calls[1]
        assert "Recent conversation" not in second_prompt

    def test_sticky_candidate_only_from_immediately_previous_turn(self, repo, embedder):
        solar = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        molecule = ingest(repo, embedder, "Molecular graph networks", MOLECULE_TEXT)
        llm = FakeLLMProvider(
            [
                found_reply(solar.id, "Renewable energy generation from solar arrays"),
                found_reply(molecule.id, "operate on molecular graphs"),
                not_found_reply(),
            ]
        )
        service = QueryService(repo, llm, embedder, embedder.model_name)
        session = ChatSession(service)
        session.ask("solar")
        session.ask("graphs")
        # "an unrelated third question" matches neither source via FTS
        # (no shared tokens) nor embedding (both well under embed_threshold)
        session.ask("an unrelated third question")

        # molecule.id (the immediately previous turn's citation) is seeded;
        # solar.id (two turns back) is not carried forward at all
        assert session.turns[2].answer.considered_source_ids == (molecule.id,)
