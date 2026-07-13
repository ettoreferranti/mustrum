import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import Embedding, EntityKind, Source, SourceKind
from mustrum.core.services.grounded import Evidence
from mustrum.core.services.ideas import IdeaService
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


def not_found_reply(answer="I found nothing, please ignore this text."):
    return json.dumps({"found": False, "answer": answer, "evidence": []})


class TestAsk:
    def test_fts_match_grounded_answer(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        llm = FakeLLMProvider(
            [found_reply(source.id, "Renewable energy generation from solar arrays")]
        )
        answer = QueryService(repo, llm, embedder, embedder.model_name).ask("renewable energy")
        assert answer.found
        assert answer.answer == "Yes, one source covers this."
        assert answer.evidence == (
            Evidence(source.id, "Renewable energy generation from solar arrays"),
        )
        assert answer.considered_source_ids == (source.id,)
        assert answer.question == "renewable energy"
        prompt, system = llm.calls[0]
        assert prompt == (
            f"Question: renewable energy\n\n[source {source.id}] Solar PV study\n"
            f"{SOLAR_TEXT}\n\nAnswer as instructed."
        )
        assert "EXACTLY" in system
        assert llm.schemas[0] == {
            "type": "object",
            "properties": {
                "found": {"type": "boolean"},
                "answer": {"type": "string"},
                "evidence": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "integer"},
                            "quote": {"type": "string"},
                        },
                        "required": ["source_id", "quote"],
                    },
                },
            },
            "required": ["found", "answer", "evidence"],
        }

    def test_quotes_verified_against_full_text_not_excerpt(self, repo, embedder):
        # quote from beyond the excerpt boundary must still verify
        long_text = ("x" * 200 + " ") * 20 + "renewable energy final unique sentence"
        source = ingest(repo, embedder, "Long paper", long_text)
        llm = FakeLLMProvider([found_reply(source.id, "renewable energy final unique sentence")])
        answer = QueryService(repo, llm, embedder, embedder.model_name, max_source_chars=100).ask(
            "renewable energy"
        )
        assert answer.found
        # the excerpt in the prompt was truncated, but verification wasn't
        prompt, _ = llm.calls[0]
        assert "renewable energy final unique sentence" not in prompt

    def test_candidate_without_text_is_skipped_not_short_circuited(self, repo, embedder):
        # a bare/metadata-only candidate (no stored text) ranks first by
        # embedding; it must be skipped, not abort retrieval for the rest
        real = ingest(repo, embedder, "Real paper", SOLAR_TEXT)
        bare = repo.add_source(Source(kind=SourceKind.PAPER, title="Bare metadata source"))
        query_vector = embedder.embed(["renewable energy solar"])[0]
        repo.store_embeddings(
            [
                Embedding(
                    entity=EntityKind.SOURCE,
                    ref_id=bare.id,
                    chunk_index=0,
                    model=embedder.model_name,
                    vector=query_vector,  # cosine == 1.0, guaranteed to rank first
                )
            ]
        )
        llm = FakeLLMProvider(
            [found_reply(real.id, "Renewable energy generation from solar arrays")]
        )
        answer = QueryService(repo, llm, embedder, embedder.model_name).ask(
            "renewable energy solar"
        )
        assert answer.found
        assert real.id in answer.considered_source_ids

    def test_fts_candidates_capped_at_top_k(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        calls: list[int] = []
        original_search = repo.search

        def spy_search(query: str, limit: int = 20):
            calls.append(limit)
            return original_search(query, limit=limit)

        repo.search = spy_search  # type: ignore[method-assign]
        llm = FakeLLMProvider(
            [found_reply(source.id, "Renewable energy generation from solar arrays")]
        )
        QueryService(repo, llm, embedder, embedder.model_name, top_k=3).ask("solar")
        assert calls == [3]

    def test_embedding_only_recall(self, repo, embedder):
        # search()/_fts_query ANDs every token; "wind" isn't in SOLAR_TEXT at
        # all, so FTS returns nothing for this query. The embedding path
        # (token-overlap cosine over the *other* four shared tokens) must be
        # the one that surfaces the source.
        source = ingest(repo, embedder, "Photovoltaic tracking mounts", SOLAR_TEXT)
        ingest(repo, embedder, "Molecular graph networks", MOLECULE_TEXT)
        query = "solar panel tracking mounts wind"
        assert repo.search(query) == []
        llm = FakeLLMProvider([found_reply(source.id, "improves with tracking mounts")])
        answer = QueryService(repo, llm, embedder, embedder.model_name, top_k=1).ask(query)
        assert answer.found
        assert answer.considered_source_ids == (source.id,)

    def test_no_candidates_llm_never_called(self, repo, embedder):
        llm = FakeLLMProvider()  # would raise if called
        answer = QueryService(repo, llm, embedder, embedder.model_name).ask("anything at all")
        assert answer.found is False
        assert answer.question == "anything at all"
        assert answer.answer == "No sources in your library appear to address this."
        assert answer.evidence == ()
        assert answer.considered_source_ids == ()
        assert llm.calls == []

    def test_fts_candidates_exclude_non_source_entities(self, repo, embedder):
        # no sources exist at all, so ranked_by_embedding is empty; the only
        # FTS hit for "renewable" is the idea, which must not leak through
        # as a (numerically coincidental) source candidate id.
        IdeaService(repo, embedder).create("Renewable energy idea", "notes about renewables")
        hits = repo.search("renewable")
        assert [h.entity for h in hits] == [EntityKind.IDEA]
        service = QueryService(repo, FakeLLMProvider(), embedder, embedder.model_name)
        assert service._candidate_source_ids("renewable") == []

    def test_llm_output_never_parsable_raises_with_last_result_none(self, repo, embedder):
        ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        garbage = "not json at all"
        llm = FakeLLMProvider([garbage])
        with pytest.raises(QueryFailure, match="no parsable output") as exc:
            QueryService(repo, llm, embedder, embedder.model_name, attempts=1).ask("solar")
        assert exc.value.question == "solar"
        assert exc.value.last_result is None
        assert exc.value.raw_output == garbage
        assert garbage in str(exc.value)

    def test_non_str_answer_triggers_structure_error_not_a_crash(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = json.dumps(
            {
                "found": True,
                "answer": 42,  # wrong type: must be a string
                "evidence": [
                    {
                        "source_id": source.id,
                        "quote": "Renewable energy generation from solar arrays",
                    }
                ],
            }
        )
        llm = FakeLLMProvider([bad])
        with pytest.raises(QueryFailure):
            QueryService(repo, llm, embedder, embedder.model_name, attempts=1).ask("solar")

    def test_evidence_item_with_wrong_type_source_id_retried_as_malformed(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = json.dumps(
            {
                "found": True,
                "answer": "a",
                "evidence": [
                    {
                        "source_id": "not-an-int",
                        "quote": "Renewable energy generation from solar arrays",
                    }
                ],
            }
        )
        good = found_reply(source.id, "Renewable energy generation from solar arrays")
        llm = FakeLLMProvider([bad, good])
        answer = QueryService(repo, llm, embedder, embedder.model_name, attempts=2).ask("solar")
        assert answer.found
        second_prompt, _ = llm.calls[1]
        assert "malformed evidence items" in second_prompt

    def test_non_dict_evidence_item_retried_as_malformed(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = json.dumps({"found": True, "answer": "a", "evidence": ["not-a-dict-item"]})
        good = found_reply(source.id, "Renewable energy generation from solar arrays")
        llm = FakeLLMProvider([bad, good])
        answer = QueryService(repo, llm, embedder, embedder.model_name, attempts=2).ask("solar")
        assert answer.found
        second_prompt, _ = llm.calls[1]
        assert "malformed evidence items" in second_prompt

    def test_found_true_with_empty_evidence_retried_then_rejected(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        empty = json.dumps({"found": True, "answer": "a", "evidence": []})
        good = found_reply(source.id, "Renewable energy generation from solar arrays")
        llm = FakeLLMProvider([empty, good])
        answer = QueryService(repo, llm, embedder, embedder.model_name, attempts=2).ask("solar")
        assert answer.found
        second_prompt, _ = llm.calls[1]
        assert "supplied no evidence" in second_prompt

        llm2 = FakeLLMProvider([empty])
        with pytest.raises(QueryFailure) as exc:
            QueryService(repo, llm2, embedder, embedder.model_name, attempts=1).ask("solar")
        assert exc.value.last_result is not None
        assert exc.value.last_result.empty_evidence is True

    def test_unknown_source_id_before_valid_group_both_reported(self, repo, embedder):
        # the unknown-source group must not short-circuit — a later group's
        # own (unrelated) grounding failure must still surface
        solar = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        reply = json.dumps(
            {
                "found": True,
                "answer": "a",
                "evidence": [
                    {"source_id": solar.id + 999, "quote": "ghost quote"},
                    {"source_id": solar.id, "quote": "totally invented span"},
                ],
            }
        )
        llm = FakeLLMProvider([reply])
        with pytest.raises(QueryFailure) as exc:
            QueryService(repo, llm, embedder, embedder.model_name, attempts=1).ask("solar")
        assert exc.value.last_result is not None
        assert exc.value.last_result.empty_evidence is False
        missing = exc.value.last_result.missing_quotes
        assert any("ghost quote" in m for m in missing)
        assert any("totally invented span" in m for m in missing)
        assert exc.value.raw_output == ""

    def test_model_not_found_prose_discarded(self, repo, embedder):
        ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        llm = FakeLLMProvider([not_found_reply("some untrusted free-form claim")])
        answer = QueryService(repo, llm, embedder, embedder.model_name).ask("solar")
        assert not answer.found
        assert answer.answer == "No sources in your library appear to address this."
        assert answer.evidence == ()

    def test_hallucinated_source_id_rejected(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = found_reply(source.id + 999, "Renewable energy generation from solar arrays")
        llm = FakeLLMProvider([bad])
        with pytest.raises(QueryFailure, match="failed grounding after 1 attempts"):
            QueryService(repo, llm, embedder, embedder.model_name, attempts=1).ask("solar")

    def test_unverifiable_quote_retried_then_rejected(self, repo, embedder):
        source = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        bad = found_reply(source.id, "totally invented span")
        llm = FakeLLMProvider([bad, bad])
        with pytest.raises(
            QueryFailure, match=r"failed grounding after 2 attempts.*totally invented span"
        ):
            QueryService(repo, llm, embedder, embedder.model_name, attempts=2).ask("solar")
        second_prompt, _ = llm.calls[1]
        assert "totally invented span" in second_prompt

    def test_multi_source_partial_failure_rejected(self, repo, embedder):
        solar = ingest(repo, embedder, "Solar PV study", SOLAR_TEXT)
        molecule = ingest(repo, embedder, "Molecular graph networks", MOLECULE_TEXT)
        reply = json.dumps(
            {
                "found": True,
                "answer": "Two sources are relevant.",
                "evidence": [
                    {
                        "source_id": solar.id,
                        "quote": "Renewable energy generation from solar arrays",
                    },
                    {"source_id": molecule.id, "quote": "invented molecule claim"},
                ],
            }
        )
        llm = FakeLLMProvider([reply])
        with pytest.raises(QueryFailure, match="failed grounding after 1 attempts"):
            QueryService(repo, llm, embedder, embedder.model_name, attempts=1).ask(
                "solar and molecules"
            )
