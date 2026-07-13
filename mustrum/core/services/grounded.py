"""Shared grounded-generation loop (ADR-7).

Every LLM output that will be stored alongside a source follows the same
discipline: generate → parse JSON → verify quotes against the stored text →
retry with corrective feedback → reject loudly if it never verifies. This
module owns that loop; summarisation and match rationales both use it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from mustrum.core.ports import LLMProvider
from mustrum.core.verify import GroundingResult, GroundingVerifier

_FENCED_BLOCK = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
# a backslash not starting a valid JSON escape (LaTeX like \alpha, \cite, ...)
_INVALID_ESCAPE = re.compile(r'\\(?!["\\/bfnrtu])')


def parse_json_object(text: str) -> dict[str, Any] | None:
    """Extract a JSON object from model output.

    Tolerates code fences, surrounding prose, literal newlines inside strings
    (strict=False), and stray LaTeX-style backslashes (repaired to escaped
    backslashes before a second parse attempt).
    """
    candidates = [m.group(1) for m in _FENCED_BLOCK.finditer(text)]
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        for variant in (candidate, _INVALID_ESCAPE.sub(r"\\\\", candidate)):
            try:
                data = json.loads(variant, strict=False)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return None


def describe_failure(last_result: GroundingResult | None, raw_output: str) -> str:
    if last_result is not None:
        if last_result.empty_evidence:
            return "model supplied no evidence quotes"
        return f"quotes not found in source: {list(last_result.missing_quotes)}"
    detail = "no parsable output"
    if raw_output:
        detail += f"; raw reply started with: {raw_output[:200]!r}"
    return detail


class GroundedOutputError(Exception):
    """The model never produced verifiable output; nothing may be stored."""

    def __init__(self, attempts: int, last_result: GroundingResult | None, raw_output: str) -> None:
        super().__init__(
            f"failed grounding after {attempts} attempts: "
            f"{describe_failure(last_result, raw_output)}"
        )
        self.attempts = attempts
        self.last_result = last_result
        self.raw_output = raw_output


def run_grounded(
    llm: LLMProvider,
    *,
    base_prompt: str,
    system: str,
    field: str,
    source_text: str,
    verifier: GroundingVerifier,
    attempts: int,
) -> tuple[str, tuple[str, ...]]:
    """Generate `{field, quotes}` JSON whose quotes verify against source_text.

    Returns (field_text, quotes) on success; raises GroundedOutputError after
    exhausting the attempts. Each retry tells the model exactly what was
    wrong with its previous reply.
    """
    # structured output (ADR-14): guarantees the reply is syntactically valid
    # JSON of this shape; the quotes are still verified against source_text
    schema = {
        "type": "object",
        "properties": {
            field: {"type": "string"},
            "quotes": {"type": "array", "items": {"type": "string"}},
        },
        "required": [field, "quotes"],
    }
    feedback = ""
    last_result: GroundingResult | None = None
    last_raw = ""
    for _ in range(attempts):
        raw = llm.generate(base_prompt + feedback, system=system, json_schema=schema)
        data = parse_json_object(raw)
        if data is None:
            last_raw = raw
            feedback = (
                "\n\nYour previous reply could not be parsed. Reply with ONLY the "
                "JSON object — no prose, no code fences, valid JSON escaping."
            )
            continue
        field_text = data.get(field)
        quotes = data.get("quotes")
        if not isinstance(field_text, str) or not isinstance(quotes, list):
            last_raw = raw
            feedback = (
                f'\n\nYour previous reply had the wrong structure. "{field}" must be '
                'a string and "quotes" a list of strings.'
            )
            continue
        quotes = [q for q in quotes if isinstance(q, str)]
        last_result = verifier.verify(quotes, source_text)
        if last_result.ok:
            return field_text.strip(), tuple(quotes)
        if last_result.empty_evidence:
            feedback = (
                "\n\nYour previous reply contained no usable quotes. Include 2-4 "
                "quotes copied verbatim from the text."
            )
        else:
            missing = ", ".join(repr(q) for q in last_result.missing_quotes)
            feedback = (
                f"\n\nThese quotes from your previous reply were NOT found verbatim "
                f"in the text: {missing}. Copy each quote EXACTLY, character for "
                f"character, from the text above — do not paraphrase, reformat "
                f"numbers, or fix spacing."
            )
    raise GroundedOutputError(attempts, last_result, last_raw)


@dataclass(frozen=True)
class Evidence:
    """One verbatim quote, attributed to the specific source it was drawn from."""

    source_id: int
    quote: str


def run_grounded_multi(
    llm: LLMProvider,
    *,
    base_prompt: str,
    system: str,
    sources: dict[int, str],
    not_found_message: str,
    verifier: GroundingVerifier,
    attempts: int,
) -> tuple[str, tuple[Evidence, ...]]:
    """Generate an answer grounded across several candidate source texts.

    Unlike `run_grounded` (one field, one source text), each evidence item
    names the `source_id` it was quoted from; a quote is only accepted if it
    verifies against *that* source's text. `found=false` in the model's
    reply is trusted as a classification signal (a false "not found" is a
    recall problem, not a rigour violation) — but the model's own prose for
    that case is discarded in favour of `not_found_message`, so an
    unverified claim can never reach the answer text.

    Returns `(answer_text, evidence)`; `evidence` is empty iff nothing was
    found. Raises GroundedOutputError after exhausting `attempts`.
    """
    schema = {
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
    feedback = ""
    last_result: GroundingResult | None = None
    last_raw = ""
    for _ in range(attempts):
        raw = llm.generate(base_prompt + feedback, system=system, json_schema=schema)
        data = parse_json_object(raw)
        if data is None:
            last_raw = raw
            feedback = (
                "\n\nYour previous reply could not be parsed. Reply with ONLY the "
                "JSON object — no prose, no code fences, valid JSON escaping."
            )
            continue
        found = data.get("found")
        answer = data.get("answer")
        raw_evidence = data.get("evidence")
        if (
            not isinstance(found, bool)
            or not isinstance(answer, str)
            or not isinstance(raw_evidence, list)
        ):
            last_raw = raw
            feedback = (
                '\n\nYour previous reply had the wrong structure. "found" must be a '
                'boolean, "answer" a string, and "evidence" a list of '
                '{"source_id": <int>, "quote": <string>} objects.'
            )
            continue
        if not found:
            return not_found_message, ()
        items: list[tuple[int, str]] = [
            (item["source_id"], item["quote"])
            for item in raw_evidence
            if isinstance(item, dict)
            and isinstance(item.get("source_id"), int)
            and isinstance(item.get("quote"), str)
        ]
        if len(items) != len(raw_evidence):
            last_raw = raw
            feedback = (
                "\n\nYour previous reply had malformed evidence items. Every entry "
                'must be {"source_id": <int>, "quote": <string>}, with source_id one '
                "of the ids listed above."
            )
            continue
        by_source: dict[int, list[str]] = {}
        for source_id, quote in items:
            by_source.setdefault(source_id, []).append(quote)
        missing: list[str] = []
        for source_id, quotes in by_source.items():
            source_text = sources.get(source_id)
            if source_text is None:
                missing.extend(f"[source {source_id}] {q}" for q in quotes)
                continue
            group_result = verifier.verify(quotes, source_text)
            missing.extend(f"[source {source_id}] {q}" for q in group_result.missing_quotes)
        empty_evidence = not items
        last_result = GroundingResult(
            ok=not missing and not empty_evidence,
            missing_quotes=tuple(missing),
            empty_evidence=empty_evidence,
        )
        if last_result.ok:
            return answer.strip(), tuple(Evidence(sid, q) for sid, q in items)
        if empty_evidence:
            feedback = (
                "\n\nYour previous reply set found=true but supplied no evidence. "
                "Include at least one quote, copied verbatim, attributed to the "
                "source_id it came from — or set found=false if nothing answers the "
                "question."
            )
        else:
            feedback = (
                f"\n\nThese quotes from your previous reply were NOT found verbatim "
                f"in their claimed source (or named a source_id not listed above): "
                f"{', '.join(repr(m) for m in missing)}. Copy each quote EXACTLY from "
                f"the excerpt of the source_id you attribute it to."
            )
    raise GroundedOutputError(attempts, last_result, last_raw)
