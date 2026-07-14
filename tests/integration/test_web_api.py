"""Web GUI adapter tests: the JSON API over real SQLite + fake providers."""

import json

import pytest
from fastapi.testclient import TestClient

from mustrum.adapters.errors import ProviderError
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
    def test_brainstorm_generates_without_saving(self, client, llm):
        """E11-7: generation never persists — the user reviews the list and
        saves selected proposals via a separate call."""
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
        data = client.post("/api/brainstorm", json={"count": 1}).json()
        proposal = data["proposals"][0]
        assert proposal["inspirations"] == ["Graph networks"]
        assert "saved_id" not in proposal
        assert client.get("/api/ideas").json() == []

    def test_brainstorm_empty_library_404(self, client):
        assert client.post("/api/brainstorm", json={}).status_code == 404

    def test_brainstorm_save_selected_creates_and_tags_only_those(self, client):
        """Only the proposals the user picked get created — the third,
        unpicked one must not appear."""
        response = client.post(
            "/api/brainstorm/save",
            json={
                "ideas": [
                    {"title": "Kept idea one", "description": "First kept."},
                    {"title": "Kept idea two", "description": "Second kept."},
                ]
            },
        )
        assert response.status_code == 200
        saved = response.json()["saved"]
        assert [s["title"] for s in saved] == ["Kept idea one", "Kept idea two"]
        titles = {i["title"] for i in client.get("/api/ideas").json()}
        assert titles == {"Kept idea one", "Kept idea two"}
        for idea in client.get("/api/ideas").json():
            assert "brainstorm" in idea["tags"]

    def test_brainstorm_save_empty_list_400(self, client):
        assert client.post("/api/brainstorm/save", json={"ideas": []}).status_code == 400

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

    def test_remove_tag_roundtrip(self, client):
        """E11-2: source tags can be edited, not just added."""
        source_id = ingest_note(client)
        client.post(f"/api/sources/{source_id}/tags", json={"text": "gnn"})
        response = client.delete(f"/api/sources/{source_id}/tags/gnn")
        assert response.status_code == 200
        assert client.get(f"/api/sources/{source_id}").json()["tags"] == []

    def test_remove_tag_missing_source_404(self, client):
        assert client.delete("/api/sources/99/tags/gnn").status_code == 404

    def test_remove_unknown_tag_is_a_noop(self, client):
        """untag() has no notion of 'not tagged' — removing a tag that was
        never applied just leaves the (empty) set unchanged."""
        source_id = ingest_note(client)
        assert client.delete(f"/api/sources/{source_id}/tags/never-applied").status_code == 200


class TestIdeaTags:
    """E11-2: idea tags get the same add/remove treatment as source tags."""

    def _idea(self, client):
        return client.post("/api/ideas", json={"title": "i", "text": "t"}).json()["id"]

    def test_add_and_remove_tag_roundtrip(self, client):
        idea_id = self._idea(client)
        response = client.post(f"/api/ideas/{idea_id}/tags", json={"text": " grant "})
        assert response.status_code == 200
        (idea,) = [i for i in client.get("/api/ideas").json() if i["id"] == idea_id]
        assert idea["tags"] == ["grant"]
        assert client.delete(f"/api/ideas/{idea_id}/tags/grant").status_code == 200
        (idea,) = [i for i in client.get("/api/ideas").json() if i["id"] == idea_id]
        assert idea["tags"] == []

    def test_empty_tag_400_and_missing_idea_404(self, client):
        idea_id = self._idea(client)
        assert client.post(f"/api/ideas/{idea_id}/tags", json={"text": "  "}).status_code == 400
        assert client.post("/api/ideas/99/tags", json={"text": "x"}).status_code == 404
        assert client.delete("/api/ideas/99/tags/x").status_code == 404


class TestContactLinks:
    """E11-2: GUI counterpart of `mustrum contact link` (FR-7.2)."""

    def _contact(self, client, name="Prof X"):
        return client.post("/api/contacts", json={"name": name, "kind": "person"}).json()["id"]

    def _idea(self, client):
        return client.post("/api/ideas", json={"title": "i", "text": "t"}).json()["id"]

    def test_link_source_roundtrip(self, client):
        source_id = ingest_note(client)
        contact_id = self._contact(client)
        response = client.post(
            f"/api/sources/{source_id}/contacts",
            json={"contact_id": contact_id, "why": "co-author"},
        )
        assert response.status_code == 200
        (link,) = client.get(f"/api/sources/{source_id}/contacts").json()
        assert link["name"] == "Prof X"
        assert link["why"] == "co-author"

    def test_link_idea_roundtrip(self, client):
        idea_id = self._idea(client)
        contact_id = self._contact(client)
        response = client.post(
            f"/api/ideas/{idea_id}/contacts",
            json={"contact_id": contact_id, "why": "suggested this direction"},
        )
        assert response.status_code == 200
        (link,) = client.get(f"/api/ideas/{idea_id}/contacts").json()
        assert link["name"] == "Prof X"
        assert link["why"] == "suggested this direction"

    def test_link_empty_why_400(self, client):
        source_id = ingest_note(client)
        contact_id = self._contact(client)
        response = client.post(
            f"/api/sources/{source_id}/contacts", json={"contact_id": contact_id, "why": "  "}
        )
        assert response.status_code == 400

    def test_link_missing_contact_404(self, client):
        source_id = ingest_note(client)
        response = client.post(
            f"/api/sources/{source_id}/contacts", json={"contact_id": 99, "why": "x"}
        )
        assert response.status_code == 404

    def test_link_missing_source_or_idea_404(self, client):
        contact_id = self._contact(client)
        assert (
            client.post(
                "/api/sources/99/contacts", json={"contact_id": contact_id, "why": "x"}
            ).status_code
            == 404
        )
        assert (
            client.post(
                "/api/ideas/99/contacts", json={"contact_id": contact_id, "why": "x"}
            ).status_code
            == 404
        )
        assert client.get("/api/sources/99/contacts").status_code == 404
        assert client.get("/api/ideas/99/contacts").status_code == 404


class TestAudit:
    """E11-2: GUI counterpart of `mustrum audit` (FR-5.5)."""

    def test_audit_upload_all_known(self, client):
        source_id = ingest_note(client)
        key = client.get(f"/api/sources/{source_id}").json()["citation_key"]
        assert key is None  # not derived until something asks for BibTeX
        client.get("/api/bib")  # derives + stores a citation key for every source
        key = client.get(f"/api/sources/{source_id}").json()["citation_key"]
        draft = f"See \\cite{{{key}}} for details.".encode()
        response = client.post("/api/audit", files={"file": ("draft.tex", draft, "text/plain")})
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is True
        assert data["used_keys"] == [key]
        assert data["unknown_keys"] == []

    def test_audit_upload_unknown_key(self, client):
        draft = b"See [@nonexistent2024] for details."
        response = client.post("/api/audit", files={"file": ("draft.md", draft, "text/plain")})
        assert response.status_code == 200
        data = response.json()
        assert data["ok"] is False
        assert data["unknown_keys"] == ["nonexistent2024"]


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


class TestOllamaModels:
    """E12-2: populates the Settings model dropdowns."""

    def test_success_uses_configured_url_by_default(self, client, monkeypatch):
        seen = {}

        def fake_list_models(url, **kw):
            seen["url"] = url
            return ["qwen3:30b", "nomic-embed-text"]

        monkeypatch.setattr("mustrum.adapters.ollama.list_models", fake_list_models)
        response = client.get("/api/ollama/models")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == ["qwen3:30b", "nomic-embed-text"]
        assert data["error"] is None
        assert seen["url"] == "http://localhost:11434"  # Config() default

    def test_url_query_param_overrides_configured_url(self, client, monkeypatch):
        seen = {}

        def fake_list_models(url, **kw):
            seen["url"] = url
            return []

        monkeypatch.setattr("mustrum.adapters.ollama.list_models", fake_list_models)
        client.get("/api/ollama/models?url=http://other-host:1234")
        assert seen["url"] == "http://other-host:1234"

    def test_unreachable_ollama_returns_200_with_error_not_500(self, client):
        # nothing listens on port 1 — deterministic, host-independent failure
        response = client.get("/api/ollama/models?url=http://127.0.0.1:1")
        assert response.status_code == 200
        data = response.json()
        assert data["models"] == []
        assert data["error"]


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

    def test_get_reflects_anthropic_defaults(self, client):
        """E10-1: GUI settings surface the provider switch too."""
        data = client.get("/api/settings").json()
        assert data["llm_provider"] == "ollama"
        assert data["anthropic_model"] == "claude-sonnet-5"
        assert data["anthropic_max_tokens"] == 8192

    def test_switch_to_anthropic_via_settings(self, client, tmp_path):
        response = client.post(
            "/api/settings",
            json={
                "llm_provider": "anthropic",
                "anthropic_model": "claude-opus-4-8",
                "anthropic_max_tokens": 4096,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["llm_provider"] == "anthropic"
        assert data["anthropic_model"] == "claude-opus-4-8"
        content = (tmp_path / "config.toml").read_text()
        assert 'llm_provider = "anthropic"' in content
        assert 'anthropic_model = "claude-opus-4-8"' in content
        assert "anthropic_max_tokens = 4096" in content

    def test_invalid_llm_provider_400(self, client):
        response = client.post("/api/settings", json={"llm_provider": "openai"})
        assert response.status_code == 400


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

    def test_provider_error_gives_502_flash_and_stderr_line(self, tmp_path, capsys):
        """A provider adapter (Ollama down, Anthropic misconfigured, ...) can
        fail deep inside a service call with no HTTPException around it —
        must still reach the GUI as a flash-able detail + stderr line, not
        FastAPI's default opaque 500."""

        class _FailingLLM:
            model_name = "failing-llm"

            def generate(self, prompt, *, system=None, json_schema=None):
                raise ProviderError("no Anthropic credentials found")

        repo = SqliteRepo(tmp_path / "web.db")
        app = create_app(
            repo, FakeEmbeddingProvider(), _FailingLLM(), Config(db_path=tmp_path / "web.db")
        )
        with TestClient(app) as failing_client:
            source_id = ingest_note(failing_client)
            response = failing_client.post(f"/api/sources/{source_id}/summarise")
            assert response.status_code == 502
            assert "no Anthropic credentials found" in response.json()["detail"]
            assert "[mustrum ui]" in capsys.readouterr().err
        repo.close()


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


def _found_reply(source_id, quote, answer="Grounded chat answer."):
    return json.dumps(
        {"found": True, "answer": answer, "evidence": [{"source_id": source_id, "quote": quote}]}
    )


class TestChat:
    def test_no_candidates_found_false_no_llm_call(self, client, llm):
        response = client.post("/api/chat", json={"question": "anything at all"})
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is False
        assert data["evidence"] == []
        assert data["considered_source_ids"] == []
        assert llm.calls == []

    def test_grounded_turn_shape(self, client, llm):
        source_id = ingest_note(client)
        llm.queue(_found_reply(source_id, "graph neural networks"))
        response = client.post("/api/chat", json={"question": "graph"})
        assert response.status_code == 200
        data = response.json()
        assert data["found"] is True
        assert data["answer"] == "Grounded chat answer."
        assert data["evidence"] == [{"source_id": source_id, "quote": "graph neural networks"}]
        assert data["considered_source_ids"] == [source_id]

    def test_second_turn_carries_history(self, client, llm):
        source_id = ingest_note(client)
        llm.queue(
            _found_reply(source_id, "graph neural networks", "First answer."),
            _found_reply(source_id, "molecular property prediction", "Second answer."),
        )
        client.post("/api/chat", json={"question": "graph"})
        client.post("/api/chat", json={"question": "molecules"})
        second_prompt, _ = llm.calls[1]
        assert "Recent conversation" in second_prompt
        assert "Q: graph" in second_prompt
        assert "A: First answer." in second_prompt

    def test_reset_clears_session(self, client, llm):
        source_id = ingest_note(client)
        llm.queue(
            _found_reply(source_id, "graph neural networks"),
            _found_reply(source_id, "graph neural networks"),
        )
        client.post("/api/chat", json={"question": "graph"})
        assert client.post("/api/chat/reset").json() == {"reset": True}
        client.post("/api/chat", json={"question": "graph"})
        second_prompt, _ = llm.calls[1]
        assert "Recent conversation" not in second_prompt

    def test_ungroundable_reply_returns_422(self, client, llm):
        source_id = ingest_note(client)
        # QueryService retries up to its default `attempts` (3); queue a bad
        # reply for every attempt so the fake provider never runs dry
        llm.queue(*[_found_reply(source_id, "totally invented span")] * 3)
        response = client.post("/api/chat", json={"question": "graph"})
        assert response.status_code == 422
