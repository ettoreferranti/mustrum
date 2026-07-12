"""Brainstorming mode (E9-2): the only *creative* feature of the tool.

Quarantine rules (NFR-1): brainstorming is explicitly invoked, its output is
labelled machine-generated creative content, it never produces citations or
keys, and the only library references it may carry ("inspired by") are titles
resolved against real records — unresolvable mentions are dropped, so an
invented reference can never surface. Proposals are not stored unless the
user saves them, and saved ideas carry the 'brainstorm' tag permanently.
"""

from __future__ import annotations

from dataclasses import dataclass

from mustrum.core.ports import LLMProvider, StorageRepo
from mustrum.core.services.grounded import parse_json_object

BRAINSTORM_TAG = "brainstorm"

_SYSTEM = (
    "You are in BRAINSTORM mode: propose novel research ideas that build on, "
    "combine, or extend the user's library. Be creative — but never invent "
    "citations, paper titles, or references. Reply with a single JSON object: "
    '{"ideas": [{"title": "<short idea title>", '
    '"description": "<2-4 sentences: the idea and why it is promising>", '
    '"based_on": ["<EXACT title copied from the library below>", ...]}]} '
    "based_on may ONLY contain titles copied exactly from the library; leave "
    "it empty if the idea is not tied to specific entries. Output the JSON "
    "object immediately, with no preamble and no reasoning."
)


@dataclass(frozen=True)
class IdeaProposal:
    title: str
    description: str
    inspirations: tuple[str, ...]  # titles resolved against the library only


class BrainstormFailure(Exception):
    def __init__(self, attempts: int, raw_output: str) -> None:
        detail = f"; raw reply started with: {raw_output[:200]!r}" if raw_output else ""
        super().__init__(
            f"brainstorm produced no usable proposals after {attempts} attempts{detail}"
        )
        self.raw_output = raw_output


class BrainstormService:
    def __init__(
        self,
        repo: StorageRepo,
        llm: LLMProvider,
        attempts: int = 3,
        max_context_chars: int = 12000,
    ) -> None:
        self._repo = repo
        self._llm = llm
        self._attempts = attempts
        self._max_context_chars = max_context_chars

    def propose(self, count: int = 3, focus: str = "") -> list[IdeaProposal]:
        context, known_titles = self._library_context()
        if not context:
            raise LookupError("the library is empty — nothing to brainstorm from")
        prompt = (
            f"Library:\n{context}\n\n"
            + (f"Focus area: {focus}\n\n" if focus else "")
            + f"Propose {count} new research ideas as instructed."
        )
        # structured output (ADR-14): syntactic shape only — the based_on
        # quarantine (titles resolved against real records) stays in _parse
        schema = {
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
        feedback = ""
        last_raw = ""
        for _ in range(self._attempts):
            raw = self._llm.generate(prompt + feedback, system=_SYSTEM, json_schema=schema)
            proposals = self._parse(raw, known_titles)
            if proposals:
                return proposals[:count]
            last_raw = raw
            feedback = (
                "\n\nYour previous reply was not usable. Reply with ONLY the JSON "
                'object {"ideas": [...]} where every entry has a non-empty '
                '"title" and "description".'
            )
        raise BrainstormFailure(self._attempts, last_raw)

    def _library_context(self) -> tuple[str, dict[str, str]]:
        """Compact library listing + case-insensitive title lookup table."""
        lines: list[str] = []
        known: dict[str, str] = {}
        for source in self._repo.list_sources():
            assert source.id is not None
            summary = self._repo.get_summary(source.id)
            gist = summary.text if summary else ""
            if not gist:
                text = self._repo.get_source_text(source.id)
                gist = text.text[:200] if text else ""
            lines.append(f"- [paper] {source.title}: {gist[:300]}")
            known[source.title.strip().lower()] = source.title
        for idea in self._repo.list_ideas():
            assert idea.id is not None
            version = self._repo.latest_idea_version(idea.id)
            gist = version.text[:300] if version else ""
            lines.append(f"- [existing idea] {idea.title}: {gist}")
            known[idea.title.strip().lower()] = idea.title
        return "\n".join(lines)[: self._max_context_chars], known

    def _parse(self, raw: str, known_titles: dict[str, str]) -> list[IdeaProposal]:
        data = parse_json_object(raw)
        if data is None:
            return []
        entries = data.get("ideas")
        if not isinstance(entries, list):
            return []
        proposals: list[IdeaProposal] = []
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            description = entry.get("description")
            if not isinstance(title, str) or not title.strip():
                continue
            if not isinstance(description, str) or not description.strip():
                continue
            based_on = entry.get("based_on")
            mentions = based_on if isinstance(based_on, list) else []
            # only titles that resolve to real records may surface (NFR-1)
            inspirations = tuple(
                known_titles[m.strip().lower()]
                for m in mentions
                if isinstance(m, str) and m.strip().lower() in known_titles
            )
            proposals.append(
                IdeaProposal(
                    title=title.strip(),
                    description=description.strip(),
                    inspirations=inspirations,
                )
            )
        return proposals
