import dataclasses
from datetime import UTC

import pytest

from mustrum.core.models import (
    ContactLink,
    Idea,
    Source,
    SourceKind,
    utcnow,
)


class TestContactLink:
    def test_link_to_idea_only_is_valid(self):
        link = ContactLink(contact_id=1, why="expert on topic", idea_id=2)
        assert link.source_id is None

    def test_link_to_source_only_is_valid(self):
        link = ContactLink(contact_id=1, why="paper author", source_id=3)
        assert link.idea_id is None

    def test_link_to_both_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            ContactLink(contact_id=1, why="x", idea_id=2, source_id=3)

    def test_link_to_neither_rejected(self):
        with pytest.raises(ValueError, match="exactly one"):
            ContactLink(contact_id=1, why="x")


class TestEntityBasics:
    def test_entities_are_immutable(self):
        source = Source(kind=SourceKind.PAPER, title="T")
        with pytest.raises(dataclasses.FrozenInstanceError):
            source.title = "changed"  # type: ignore[misc]

    def test_created_at_is_timezone_aware_utc(self):
        idea = Idea(title="x")
        assert idea.created_at.tzinfo == UTC

    def test_utcnow_is_utc(self):
        assert utcnow().tzinfo == UTC
