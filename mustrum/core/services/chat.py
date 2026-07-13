"""Multi-turn grounded chat over the library (E13-2, ADR-18): a thin,
in-memory stateful wrapper around QueryService. Every turn runs the exact
same grounded call as a bare `QueryService.ask()` — history only ever
widens what QueryService is given to work with (prompt context for
resolving references, and sticky candidate ids), never what counts as
evidence.
"""

from __future__ import annotations

from dataclasses import dataclass

from mustrum.core.services.query import QueryAnswer, QueryService


@dataclass(frozen=True)
class ChatTurn:
    question: str
    answer: QueryAnswer


class ChatSession:
    def __init__(self, query_service: QueryService, history_turns: int = 3) -> None:
        self._query_service = query_service
        self._history_turns = history_turns
        self._turns: list[ChatTurn] = []

    @property
    def turns(self) -> tuple[ChatTurn, ...]:
        return tuple(self._turns)

    def ask(self, question: str) -> QueryAnswer:
        """Run one grounded turn. Raises QueryFailure (from QueryService) on
        ungroundable output — a failed turn is never added to history, so a
        rejected reply can't corrupt the session's context for later turns."""
        recent = self._turns[-self._history_turns :]
        history = [(turn.question, turn.answer.answer) for turn in recent]
        # sticky candidates: only the immediately previous turn's actual
        # citations, not accumulated across the whole session — bounds
        # context growth and avoids dragging stale relevance into a
        # conversation that has moved on to a new topic
        extra_candidate_ids = (
            list(dict.fromkeys(ev.source_id for ev in self._turns[-1].answer.evidence))
            if self._turns
            else []
        )
        answer = self._query_service.ask(
            question, history=history, extra_candidate_ids=extra_candidate_ids
        )
        self._turns.append(ChatTurn(question=question, answer=answer))
        return answer

    def reset(self) -> None:
        self._turns.clear()
