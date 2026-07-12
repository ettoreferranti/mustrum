"""Shared grounded-generation loop (ADR-7).

Every LLM output that will be stored alongside a source follows the same
discipline: generate → parse JSON → verify quotes against the stored text →
retry with corrective feedback → reject loudly if it never verifies. This
module owns that loop; summarisation and match rationales both use it.
"""

from __future__ import annotations

import json
import re
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
