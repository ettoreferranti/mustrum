import json

import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.summarise import GroundingFailure, SummariseService

TEXT = (
    "We propose the Transformer, a model architecture eschewing recurrence. "
    "Experiments on two machine translation tasks show these models to be "
    "superior in quality while being more parallelizable."
)


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


@pytest.fixture
def source_id(repo):
    ingest = IngestService(repo, FakeEmbeddingProvider())
    return ingest.ingest_document(
        title="Attention Is All You Need", text=TEXT, extraction_method="plaintext"
    ).source.id


def good_reply(summary="The authors propose the Transformer."):
    return json.dumps(
        {"summary": summary, "quotes": ["We propose the Transformer", "superior in quality"]}
    )


class TestSummarise:
    def test_verified_summary_stored(self, repo, source_id):
        llm = FakeLLMProvider([good_reply()])
        service = SummariseService(repo, llm)
        summary = service.summarise(source_id)
        assert summary.verified is True
        assert summary.model == "fake-llm"
        assert summary.evidence == ("We propose the Transformer", "superior in quality")
        assert repo.get_summary(source_id) == summary

    def test_prompt_contains_title_and_text(self, repo, source_id):
        llm = FakeLLMProvider([good_reply()])
        SummariseService(repo, llm).summarise(source_id)
        prompt, system = llm.calls[0]
        assert "Attention Is All You Need" in prompt
        assert "We propose the Transformer" in prompt
        assert "verbatim" in system or "EXACTLY" in system

    def test_existing_summary_returned_without_llm_call(self, repo, source_id):
        service = SummariseService(repo, FakeLLMProvider([good_reply()]))
        first = service.summarise(source_id)
        llm2 = FakeLLMProvider()  # would raise if called
        again = SummariseService(repo, llm2).summarise(source_id)
        assert again == first
        assert llm2.calls == []

    def test_force_regenerates(self, repo, source_id):
        service = SummariseService(repo, FakeLLMProvider([good_reply("v1"), good_reply("v2")]))
        service.summarise(source_id)
        regenerated = service.summarise(source_id, force=True)
        assert regenerated.text == "v2"

    def test_fabricated_quote_rejected_nothing_stored(self, repo, source_id):
        bad = json.dumps({"summary": "s", "quotes": ["we invented ImageNet"]})
        service = SummariseService(repo, FakeLLMProvider([bad, bad]), attempts=2)
        with pytest.raises(GroundingFailure, match="not found in source"):
            service.summarise(source_id)
        assert repo.get_summary(source_id) is None

    def test_no_quotes_rejected(self, repo, source_id):
        bad = json.dumps({"summary": "s", "quotes": []})
        service = SummariseService(repo, FakeLLMProvider([bad]), attempts=1)
        with pytest.raises(GroundingFailure, match="no evidence"):
            service.summarise(source_id)

    def test_retry_succeeds_after_bad_first_attempt(self, repo, source_id):
        bad = json.dumps({"summary": "s", "quotes": ["fabricated"]})
        service = SummariseService(repo, FakeLLMProvider([bad, good_reply()]), attempts=2)
        assert service.summarise(source_id).verified is True

    def test_unparsable_output_rejected(self, repo, source_id):
        service = SummariseService(
            repo, FakeLLMProvider(["not json at all", "{broken"]), attempts=2
        )
        with pytest.raises(GroundingFailure, match="no parsable output"):
            service.summarise(source_id)

    def test_json_in_code_fence_accepted(self, repo, source_id):
        fenced = f"```json\n{good_reply()}\n```"
        service = SummariseService(repo, FakeLLMProvider([fenced]))
        assert service.summarise(source_id).verified is True

    def test_source_without_text_raises(self, repo):
        ingest = IngestService(repo, FakeEmbeddingProvider())
        bare = ingest.ingest_document(title="No text", text="", extraction_method="plaintext")
        service = SummariseService(repo, FakeLLMProvider())
        with pytest.raises(LookupError, match="no stored text"):
            service.summarise(bare.source.id)

    def test_quotes_verified_against_full_text_not_excerpt(self, repo):
        # quote from beyond the excerpt boundary must still verify
        long_text = ("x" * 200 + " ") * 20 + "the final unique sentence"
        ingest = IngestService(repo, FakeEmbeddingProvider())
        sid = ingest.ingest_document(
            title="Long", text=long_text, extraction_method="plaintext"
        ).source.id
        reply = json.dumps({"summary": "s", "quotes": ["the final unique sentence"]})
        service = SummariseService(repo, FakeLLMProvider([reply]), max_source_chars=100)
        assert service.summarise(sid).verified is True


class TestOverride:
    def test_user_override_stored(self, repo, source_id):
        service = SummariseService(repo, FakeLLMProvider())
        summary = service.override(source_id, "  My own summary. ")
        assert summary.user_override is True
        assert summary.model == "user"
        assert repo.get_summary(source_id).text == "My own summary."

    def test_override_missing_source_raises(self, repo):
        service = SummariseService(repo, FakeLLMProvider())
        with pytest.raises(KeyError):
            service.override(999, "x")
