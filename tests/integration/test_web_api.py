"""Web GUI adapter tests: the JSON API over real SQLite + fake providers."""

import json

import pytest
from fastapi.testclient import TestClient

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.config import Config
from mustrum.web.api import create_app

NOTE_TEXT = "graph neural networks for molecular property prediction on molecules"


@pytest.fixture
def llm():
    return FakeLLMProvider()


@pytest.fixture
def client(tmp_path, llm):
    repo = SqliteRepo(tmp_path / "web.db")
    app = create_app(repo, FakeEmbeddingProvider(), llm, Config(db_path=tmp_path / "web.db"))
    with TestClient(app) as c:
        yield c
    repo.close()


def ingest_note(client, title="Graph networks"):
    response = client.post(
        "/api/ingest/file",
        files={"file": (f"{title}.md", NOTE_TEXT.encode(), "text/markdown")},
    )
    assert response.status_code == 200, response.text
    return response.json()["source"]["id"]


class TestPagesAndStatus:
    def test_index_serves_gui(self, client):
        response = client.get("/")
        assert response.status_code == 200
        assert "<title>Mustrum</title>" in response.text
        assert 'src="http' not in response.text  # self-contained, no CDNs

    def test_graph_page(self, client):
        assert client.get("/graph").text.startswith("<!DOCTYPE html>")

    def test_status_counts(self, client):
        ingest_note(client)
        status = client.get("/api/status").json()
        assert status["sources"] == 1
        assert status["llm_model"] == "fake-llm"


class TestSources:
    def test_upload_and_get(self, client):
        source_id = ingest_note(client)
        data = client.get(f"/api/sources/{source_id}").json()
        assert data["title"] == "Graph networks"
        assert data["has_text"] is True
        assert data["text_kind"] == "plaintext"

    def test_missing_source_404(self, client):
        assert client.get("/api/sources/99").status_code == 404

    def test_status_notes_roundtrip(self, client):
        source_id = ingest_note(client)
        assert client.post(f"/api/sources/{source_id}/status/read").status_code == 200
        client.post(f"/api/sources/{source_id}/notes", json={"text": "solid baselines"})
        data = client.get(f"/api/sources/{source_id}").json()
        assert data["reading_status"] == "read"
        assert data["notes"] == "solid baselines"

    def test_invalid_status_400(self, client):
        source_id = ingest_note(client)
        assert client.post(f"/api/sources/{source_id}/status/devoured").status_code == 400

    def test_summarise_verified(self, client, llm):
        source_id = ingest_note(client)
        llm.queue(json.dumps({"summary": "GNNs for chemistry.", "quotes": [NOTE_TEXT[:30]]}))
        response = client.post(f"/api/sources/{source_id}/summarise")
        assert response.status_code == 200
        assert response.json()["text"] == "GNNs for chemistry."
        assert client.get(f"/api/sources/{source_id}").json()["summary"]["text"]

    def test_summarise_grounding_failure_422(self, client, llm):
        source_id = ingest_note(client)
        bad = json.dumps({"summary": "s", "quotes": ["fabricated"]})
        llm.queue(bad, bad, bad)
        assert client.post(f"/api/sources/{source_id}/summarise").status_code == 422


class TestIdeasAndMatching:
    def _idea(self, client):
        response = client.post(
            "/api/ideas",
            json={"title": "molecular ML", "text": "graph networks molecule prediction"},
        )
        return response.json()["id"]

    def test_create_list_revise(self, client):
        idea_id = self._idea(client)
        client.post(f"/api/ideas/{idea_id}/revise", json={"text": "sharper focus"})
        (idea,) = client.get("/api/ideas").json()
        assert idea["text"] == "sharper focus"
        assert idea["versions"] == 2

    def test_suggest_confirm_flow(self, client):
        ingest_note(client)
        idea_id = self._idea(client)
        suggestions = client.post(f"/api/ideas/{idea_id}/suggest", json={"threshold": 0.05}).json()
        assert suggestions
        match_id = suggestions[0]["id"]
        assert client.post(f"/api/matches/{match_id}/confirm").status_code == 200
        matches = client.get(f"/api/ideas/{idea_id}/matches").json()
        assert matches[0]["status"] == "confirmed"

    def test_explain_returns_rationale(self, client, llm):
        ingest_note(client)
        idea_id = self._idea(client)
        match_id = client.post(f"/api/ideas/{idea_id}/suggest", json={"threshold": 0.05}).json()[0][
            "id"
        ]
        llm.queue(json.dumps({"rationale": "Relevant.", "quotes": [NOTE_TEXT[:20]]}))
        data = client.post(f"/api/matches/{match_id}/explain").json()
        assert data["rationale"] == "Relevant."

    def test_unknown_match_action_400(self, client):
        ingest_note(client)
        idea_id = self._idea(client)
        match_id = client.post(f"/api/ideas/{idea_id}/suggest", json={"threshold": 0.05}).json()[0][
            "id"
        ]
        assert client.post(f"/api/matches/{match_id}/frobnicate").status_code == 400

    def test_related_work_and_bib(self, client):
        ingest_note(client)
        idea_id = self._idea(client)
        match_id = client.post(f"/api/ideas/{idea_id}/suggest", json={"threshold": 0.05}).json()[0][
            "id"
        ]
        client.post(f"/api/matches/{match_id}/confirm")
        text = client.get(f"/api/ideas/{idea_id}/related-work").json()["text"]
        assert "# Related work — molecular ML" in text
        bib = client.get(f"/api/bib?idea_id={idea_id}").json()["text"]
        assert "@" in bib

    def test_gaps_and_search(self, client):
        ingest_note(client)
        self._idea(client)
        gaps = client.get("/api/gaps").json()
        assert gaps["unsupported_ideas"][0]["title"] == "molecular ML"
        hits = client.get("/api/search", params={"q": "molecular"}).json()
        assert hits


class TestBrainstormAndContacts:
    def test_brainstorm_save_tags(self, client, llm):
        ingest_note(client)
        llm.queue(
            json.dumps(
                {
                    "ideas": [
                        {
                            "title": "New direction",
                            "description": "Combine things.",
                            "based_on": ["Graph networks"],
                        }
                    ]
                }
            )
        )
        data = client.post("/api/brainstorm", json={"count": 1, "save": True}).json()
        proposal = data["proposals"][0]
        assert proposal["inspirations"] == ["Graph networks"]
        assert proposal["saved_id"] is not None
        (idea,) = client.get("/api/ideas").json()
        assert "brainstorm" in idea["tags"]

    def test_brainstorm_empty_library_404(self, client):
        assert client.post("/api/brainstorm", json={}).status_code == 404

    def test_contacts_roundtrip(self, client):
        response = client.post(
            "/api/contacts",
            json={"name": "Prof X", "kind": "university", "affiliation": "Unseen University"},
        )
        assert response.status_code == 200
        (contact,) = client.get("/api/contacts").json()
        assert contact["name"] == "Prof X"

    def test_bad_contact_kind_400(self, client):
        assert client.post("/api/contacts", json={"name": "X", "kind": "wizard"}).status_code == 400


class TestDelete:
    def test_delete_source_cascades(self, client):
        source_id = ingest_note(client)
        idea_id = client.post(
            "/api/ideas", json={"title": "i", "text": "graph networks molecules"}
        ).json()["id"]
        client.post(f"/api/ideas/{idea_id}/suggest", json={"threshold": 0.05})
        assert client.delete(f"/api/sources/{source_id}").status_code == 200
        assert client.get(f"/api/sources/{source_id}").status_code == 404
        assert client.get(f"/api/ideas/{idea_id}/matches").json() == []
        assert client.get("/api/status").json()["sources"] == 0

    def test_delete_idea(self, client):
        idea_id = client.post("/api/ideas", json={"title": "i", "text": "t"}).json()["id"]
        assert client.delete(f"/api/ideas/{idea_id}").status_code == 200
        assert client.get("/api/ideas").json() == []

    def test_delete_missing_404(self, client):
        assert client.delete("/api/sources/9").status_code == 404
        assert client.delete("/api/ideas/9").status_code == 404
