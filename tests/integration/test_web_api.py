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


class TestFileArchive:
    """E1-11: uploads are archived next to the DB and served back."""

    def test_upload_archives_and_serves_file(self, client, tmp_path):
        source_id = ingest_note(client)
        data = client.get(f"/api/sources/{source_id}").json()
        assert data["file_name"] == "0001-Graph-networks.md"
        assert (tmp_path / "files" / "0001-Graph-networks.md").is_file()
        response = client.get(f"/api/sources/{source_id}/file")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        assert response.content == NOTE_TEXT.encode()

    def test_pdf_upload_served_as_pdf(self, client):
        import pymupdf

        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "content")
        data = doc.tobytes()
        doc.close()
        response = client.post(
            "/api/ingest/file", files={"file": ("paper.pdf", data, "application/pdf")}
        )
        source_id = response.json()["source"]["id"]
        served = client.get(f"/api/sources/{source_id}/file")
        assert served.headers["content-type"] == "application/pdf"
        assert served.content.startswith(b"%PDF")

    def test_file_endpoint_404s(self, client, tmp_path):
        assert client.get("/api/sources/99/file").status_code == 404
        source_id = ingest_note(client)
        (tmp_path / "files" / "0001-Graph-networks.md").unlink()
        assert client.get(f"/api/sources/{source_id}/file").status_code == 404

    def test_delete_removes_archived_file(self, client, tmp_path):
        source_id = ingest_note(client)
        assert client.delete(f"/api/sources/{source_id}").status_code == 200
        assert list((tmp_path / "files").iterdir()) == []


class TestAttach:
    """E11-3: GUI 'Add PDF' — attach a manually-downloaded original."""

    def _bare_source(self, client, title="Metadata Only"):
        """A source without stored text (like a DOI ingest whose PDF 403'd)."""
        response = client.post(
            "/api/ingest/file", files={"file": (f"{title}.md", b"", "text/markdown")}
        )
        return response.json()["source"]["id"]

    def test_attach_pdf_stores_text_and_archives(self, client, tmp_path):
        import pymupdf

        source_id = self._bare_source(client)
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "the manually downloaded full text")
        data = doc.tobytes()
        doc.close()
        response = client.post(
            f"/api/sources/{source_id}/attach",
            files={"file": ("downloaded.pdf", data, "application/pdf")},
        )
        assert response.status_code == 200, response.text
        assert response.json()["summary_invalidated"] is False
        source = client.get(f"/api/sources/{source_id}").json()
        assert source["has_text"] is True
        assert source["text_kind"] == "pymupdf"
        assert source["file_name"].endswith(".pdf")
        archived = tmp_path / "files" / source["file_name"]
        assert archived.read_bytes() == data

    def test_attach_refuses_replacing_full_text_409(self, client):
        source_id = ingest_note(client)  # already has full plaintext
        response = client.post(
            f"/api/sources/{source_id}/attach",
            files={"file": ("x.md", b"other text", "text/markdown")},
        )
        assert response.status_code == 409
        assert "refusing to replace" in response.json()["detail"]

    def test_attach_missing_source_404(self, client):
        response = client.post(
            "/api/sources/99/attach", files={"file": ("x.md", b"t", "text/markdown")}
        )
        assert response.status_code == 404


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


class TestTags:
    """E11-6: bulk tag runs through this per-source endpoint."""

    def test_add_tag_roundtrip(self, client):
        source_id = ingest_note(client)
        assert (
            client.post(f"/api/sources/{source_id}/tags", json={"text": " gnn "}).status_code == 200
        )
        assert client.get(f"/api/sources/{source_id}").json()["tags"] == ["gnn"]

    def test_empty_tag_400_and_missing_source_404(self, client):
        source_id = ingest_note(client)
        assert client.post(f"/api/sources/{source_id}/tags", json={"text": "  "}).status_code == 400
        assert client.post("/api/sources/99/tags", json={"text": "x"}).status_code == 404


class TestEditMetadata:
    """E8-6: GUI counterpart of `source edit` for DOI-less venues."""

    def test_metadata_roundtrip(self, client):
        source_id = ingest_note(client)
        response = client.post(
            f"/api/sources/{source_id}/metadata",
            json={"authors": ["Ada Smith", " Bob Jones "], "year": 2025},
        )
        assert response.status_code == 200
        data = client.get(f"/api/sources/{source_id}").json()
        assert data["authors"] == ["Ada Smith", "Bob Jones"]
        assert data["year"] == 2025

    def test_partial_update_keeps_other_field(self, client):
        source_id = ingest_note(client)
        client.post(f"/api/sources/{source_id}/metadata", json={"authors": ["A"], "year": 2024})
        client.post(f"/api/sources/{source_id}/metadata", json={"year": 2025})
        data = client.get(f"/api/sources/{source_id}").json()
        assert data["authors"] == ["A"]
        assert data["year"] == 2025

    def test_empty_payload_400_and_missing_404(self, client):
        source_id = ingest_note(client)
        assert client.post(f"/api/sources/{source_id}/metadata", json={}).status_code == 400
        assert client.post("/api/sources/99/metadata", json={"year": 2020}).status_code == 404


class TestSettings:
    """E12-1: library-local settings, editable from the GUI (or `config set`)."""

    def test_get_reflects_startup_config(self, client, tmp_path):
        # /api/settings reports config.embed_model (what Ollama model name to
        # request) — distinct from embedder.model_name ("fake-embed" in
        # tests), which is what's actually wired in and shown by /api/status
        data = client.get("/api/settings").json()
        assert data["llm_model"] == "qwen3:30b"
        assert data["embed_model"] == "nomic-embed-text"
        assert data["db_path"] == str(tmp_path / "web.db")
        assert data["library_config_path"] == str(tmp_path / "config.toml")
        assert data["library_config_exists"] is False

    def test_post_writes_library_config_and_reports_new_values(self, client, tmp_path):
        response = client.post("/api/settings", json={"llm_model": "llama3.1:8b", "num_ctx": 8192})
        assert response.status_code == 200
        data = response.json()
        assert data["llm_model"] == "llama3.1:8b"
        assert data["num_ctx"] == 8192
        assert data["restart_required"] is True
        content = (tmp_path / "config.toml").read_text()
        assert 'llm_model = "llama3.1:8b"' in content
        assert "num_ctx = 8192" in content

    def test_running_process_config_unaffected_until_restart(self, client):
        """Save+restart-notice (not hot-apply): a second GET must still show
        the process's original startup values, not what was just written."""
        client.post("/api/settings", json={"llm_model": "llama3.1:8b"})
        assert client.get("/api/settings").json()["llm_model"] == "qwen3:30b"

    def test_empty_payload_400(self, client):
        assert client.post("/api/settings", json={}).status_code == 400

    def test_partial_update_preserves_other_fields(self, client, tmp_path):
        client.post("/api/settings", json={"llm_model": "llama3.1:8b"})
        client.post("/api/settings", json={"num_ctx": 8192})
        content = (tmp_path / "config.toml").read_text()
        assert 'llm_model = "llama3.1:8b"' in content
        assert "num_ctx = 8192" in content


class TestErrorLogging:
    """E11-5: failed API calls leave a durable record in the ui terminal."""

    def test_failed_call_logged_to_stderr_and_detail_preserved(self, client, capsys):
        response = client.get("/api/sources/99")
        assert response.status_code == 404
        assert "no source with id 99" in response.json()["detail"]
        assert "[mustrum ui] GET /api/sources/99 -> 404" in capsys.readouterr().err

    def test_successful_call_not_logged(self, client, capsys):
        assert client.get("/api/sources").status_code == 200
        assert "[mustrum ui]" not in capsys.readouterr().err


class TestRename:
    def test_rename_endpoint(self, client):
        source_id = ingest_note(client)
        response = client.post(f"/api/sources/{source_id}/title", json={"text": "Proper Title"})
        assert response.status_code == 200
        assert client.get(f"/api/sources/{source_id}").json()["title"] == "Proper Title"

    def test_rename_collision_409(self, client):
        first = ingest_note(client, "First")
        ingest_note(client, "Second")
        response = client.post("/api/sources/2/title", json={"text": "first"})
        assert response.status_code == 409
        assert first == 1

    def test_rename_empty_400(self, client):
        source_id = ingest_note(client)
        assert (
            client.post(f"/api/sources/{source_id}/title", json={"text": "  "}).status_code == 400
        )

    def test_upload_uses_pdf_metadata_title(self, client, tmp_path):
        import pymupdf

        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "content")
        doc.set_metadata({"title": "Metadata Title Wins"})
        data = doc.tobytes()
        doc.close()
        response = client.post(
            "/api/ingest/file", files={"file": ("ugly-name.pdf", data, "application/pdf")}
        )
        assert response.json()["source"]["title"] == "Metadata Title Wins"
