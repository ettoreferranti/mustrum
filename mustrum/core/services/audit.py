"""Citation audit of external drafts (FR-5.5): every \\cite / [@key] in a
draft must resolve to a library entry."""

from __future__ import annotations

from dataclasses import dataclass

from mustrum.core.ports import StorageRepo
from mustrum.core.verify import CitationVerifier


@dataclass(frozen=True)
class AuditReport:
    ok: bool
    used_keys: tuple[str, ...]
    unknown_keys: tuple[str, ...]
    known_keys: tuple[str, ...]


class AuditService:
    def __init__(self, repo: StorageRepo) -> None:
        self._repo = repo
        self._verifier = CitationVerifier()

    def audit_text(self, text: str) -> AuditReport:
        result = self._verifier.verify(text, self._repo.citation_keys())
        known = tuple(k for k in result.used_keys if k not in result.unknown_keys)
        return AuditReport(
            ok=result.ok,
            used_keys=result.used_keys,
            unknown_keys=result.unknown_keys,
            known_keys=known,
        )
