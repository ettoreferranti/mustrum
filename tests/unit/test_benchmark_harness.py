"""E10-2: the benchmark harness itself, offline — a FakeLLMProvider stands
in for a real provider that always grounds correctly (a self-test of the
harness's pass/fail bookkeeping), plus the "provider unavailable" path
that lets Ollama-down/Anthropic-no-key be reported instead of crashing or
faking a 0% score."""

import json

from mustrum.adapters.errors import ProviderError
from mustrum.adapters.fake import FakeLLMProvider
from mustrum.benchmark.harness import GOOD_FAKE_RESPONSE, TASKS, ProviderReport, run_benchmark

_GOOD_RESPONSE = GOOD_FAKE_RESPONSE


class TestRunBenchmark:
    def test_perfect_provider_scores_100_percent(self):
        llm = FakeLLMProvider(default_response=_GOOD_RESPONSE)
        (report,) = run_benchmark({"fake": llm})
        assert report.available is True
        assert report.pass_rate == 1.0
        assert all(r.passed for r in report.results)
        # 2 fixtures x (1 summarise + 1 rationale) x 1 repeat
        assert len(report.results) == len(TASKS) * 2

    def test_ungrounded_provider_scores_below_100_percent(self):
        llm = FakeLLMProvider(default_response=json.dumps({"summary": "x", "quotes": ["nonsense"]}))
        (report,) = run_benchmark({"fake": llm})
        assert report.available is True
        assert report.pass_rate == 0.0
        assert all(not r.passed for r in report.results)
        assert all(r.detail for r in report.results)

    def test_repeats_multiplies_attempts(self):
        llm = FakeLLMProvider(default_response=_GOOD_RESPONSE)
        (report,) = run_benchmark({"fake": llm}, repeats=3)
        assert len(report.results) == len(TASKS) * 2 * 3

    def test_unavailable_provider_reported_not_crashed(self):
        class _AlwaysDown:
            model_name = "down"

            def generate(self, prompt, *, system=None, json_schema=None):
                raise ProviderError("no Anthropic credentials found")

        (report,) = run_benchmark({"anthropic": _AlwaysDown()})
        assert report.available is False
        assert report.pass_rate is None
        assert report.results == ()
        assert "no Anthropic credentials found" in report.unavailable_reason

    def test_one_unavailable_provider_does_not_stop_the_others(self):
        class _AlwaysDown:
            model_name = "down"

            def generate(self, prompt, *, system=None, json_schema=None):
                raise ProviderError("boom")

        good = FakeLLMProvider(default_response=_GOOD_RESPONSE)
        reports = run_benchmark({"broken": _AlwaysDown(), "fake": good})
        by_name = {r.provider: r for r in reports}
        assert by_name["broken"].available is False
        assert by_name["fake"].available is True
        assert by_name["fake"].pass_rate == 1.0

    def test_empty_results_pass_rate_is_none(self):
        assert ProviderReport(provider="x").pass_rate is None
