"""Provider benchmarking harness (E10-2): run the same fixed summarise/
rationale tasks through different LLMProvider implementations and compare
grounding-verification pass rates.

This never invents a result — every attempt either passes real
GroundingFailure/RationaleFailure verification against stored source text
(ADR-7) or is counted as a failure. A provider with no usable credentials
(e.g. Anthropic with no API key) is reported as unavailable rather than
given a fabricated 0% score, since that would conflate "couldn't run" with
"ran and produced ungrounded output" — two different things.

A throwaway in-memory SqliteRepo is used per provider; nothing here ever
touches the user's real library. Embeddings use FakeEmbeddingProvider
regardless of which LLM is under test — matching/embedding quality isn't
what this harness measures, only generation grounding is.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from mustrum.adapters.errors import ProviderError
from mustrum.adapters.fake import FakeEmbeddingProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import Match, SourceKind
from mustrum.core.ports import LLMProvider
from mustrum.core.services.ideas import IdeaService
from mustrum.core.services.ingest import IngestService
from mustrum.core.services.rationale import RationaleFailure, RationaleService
from mustrum.core.services.summarise import GroundingFailure, SummariseService

# Fixed benchmark fixtures — short, self-contained synthetic "papers" and
# "ideas" chosen only to exercise the grounded generate->verify loop
# identically across providers, not to represent real research. Every
# source text shares one boilerplate sentence, letting a single
# FakeLLMProvider default_response ground against any of them (see
# tests/unit/test_benchmark_harness.py).
_ANCHOR = "All measurements were repeated three times and the mean value is reported."

TASKS: tuple[tuple[str, str, str, str], ...] = (
    (
        "Photovoltaic tracking",
        "We evaluate photovoltaic panel efficiency under variable cloud cover. "
        "A two-axis tracking mount increases average daily energy yield by 18% "
        "compared to a fixed-tilt mount across six months of field data. "
        "Efficiency losses under diffuse light remain the dominant limiting "
        f"factor on overcast days. {_ANCHOR}",
        "Improve tracking-mount control under variable irradiance",
        "Explore adaptive control strategies for solar tracking mounts that "
        "respond to short-term irradiance changes, aiming to reduce the "
        "efficiency losses observed under diffuse cloud cover.",
    ),
    (
        "Graph neural networks for molecules",
        "Graph neural networks are applied to molecular property prediction "
        "across a dataset of 130,000 small organic molecules. Message-passing "
        "layers with attention pooling outperform baseline fingerprint models "
        f"on solubility and toxicity prediction tasks by a wide margin. {_ANCHOR}",
        "Molecular property prediction with attention",
        "Investigate whether attention-based pooling in graph neural networks "
        "generalises to larger, more diverse molecular datasets than those "
        "used in prior molecular property prediction work.",
    ),
)

# A FakeLLMProvider using this as its default_response grounds against every
# TASKS fixture (all share _ANCHOR) — the CLI's `--providers fake` baseline,
# so `mustrum benchmark` works out of the box with no setup at all.
GOOD_FAKE_RESPONSE = json.dumps(
    {
        "summary": "Fake grounded summary.",
        "rationale": "Fake grounded rationale.",
        "quotes": [_ANCHOR],
    }
)


@dataclass(frozen=True)
class TaskResult:
    task: str  # "summarise" | "rationale"
    label: str  # which fixture, e.g. the source title
    passed: bool
    detail: str = ""  # failure message; empty on success


@dataclass(frozen=True)
class ProviderReport:
    provider: str
    results: tuple[TaskResult, ...] = ()
    unavailable_reason: str | None = None

    @property
    def available(self) -> bool:
        return self.unavailable_reason is None

    @property
    def pass_rate(self) -> float | None:
        """None when unavailable or no attempts ran — never a fabricated 0%."""
        if not self.available or not self.results:
            return None
        return sum(1 for r in self.results if r.passed) / len(self.results)


def run_benchmark(providers: dict[str, LLMProvider], repeats: int = 1) -> list[ProviderReport]:
    """Run TASKS through each named provider `repeats` times. A provider
    that fails with ProviderError (no credentials, unreachable, ...) is
    reported unavailable and the rest of the run continues unaffected."""
    reports = []
    for name, llm in providers.items():
        repo = SqliteRepo(":memory:")
        try:
            try:
                results = _run_tasks(repo, llm, repeats)
            except ProviderError as exc:
                reports.append(ProviderReport(provider=name, unavailable_reason=str(exc)))
                continue
            reports.append(ProviderReport(provider=name, results=tuple(results)))
        finally:
            repo.close()
    return reports


def _run_tasks(repo: SqliteRepo, llm: LLMProvider, repeats: int) -> list[TaskResult]:
    embedder = FakeEmbeddingProvider()
    ingest = IngestService(repo, embedder)
    ideas = IdeaService(repo, embedder)
    summariser = SummariseService(repo, llm)
    rationale_service = RationaleService(repo, llm)

    results: list[TaskResult] = []
    for title, text, idea_title, idea_text in TASKS:
        source = ingest.ingest_document(
            title=title,
            text=text,
            extraction_method="plaintext",
            kind=SourceKind.PAPER,
            on_duplicate="skip",
        ).source
        idea = ideas.create(idea_title, idea_text)
        assert source.id is not None and idea.id is not None
        match = repo.add_match(Match(idea_id=idea.id, source_id=source.id, score=0.9))
        assert match.id is not None

        for _ in range(repeats):
            try:
                summariser.summarise(source.id, force=True)
                results.append(TaskResult("summarise", title, True))
            except GroundingFailure as exc:
                results.append(TaskResult("summarise", title, False, str(exc)))

        for _ in range(repeats):
            try:
                rationale_service.explain(match.id, force=True)
                results.append(TaskResult("rationale", title, True))
            except RationaleFailure as exc:
                results.append(TaskResult("rationale", title, False, str(exc)))
    return results
