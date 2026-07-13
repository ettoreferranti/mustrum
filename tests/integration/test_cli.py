"""End-to-end CLI tests: real SQLite on disk, fake model providers
(MUSTRUM_FAKE_PROVIDERS=1), no network."""

import pytest
from typer.testing import CliRunner

from mustrum.cli.main import app

runner = CliRunner()


@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setenv("MUSTRUM_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("MUSTRUM_FAKE_PROVIDERS", "1")


def invoke(*args, expect_exit=0, input=None):
    result = runner.invoke(app, list(args), input=input)
    assert result.exit_code == expect_exit, result.output
    return result.output


@pytest.fixture
def note(tmp_path):
    f = tmp_path / "graph-networks.md"
    f.write_text("graph neural networks for molecular property prediction on molecules")
    return f


class TestIngestAndSources:
    def test_ingest_file_and_show(self, note):
        out = invoke(
            "ingest",
            "file",
            str(note),
            "--title",
            "Graph Networks Survey",
            "--author",
            "Ada Smith",
            "--year",
            "2021",
        )
        assert "ingested" in out
        out = invoke("source", "show", "1")
        assert "Graph Networks Survey (2021)" in out
        assert "Ada Smith" in out

    def test_duplicate_fails_then_skips(self, note):
        invoke("ingest", "file", str(note), "--title", "T")
        invoke("ingest", "file", str(note), "--title", "T", expect_exit=1)
        invoke("ingest", "file", str(note), "--title", "T", "--on-duplicate", "skip")

    def test_tag_status_note_search(self, note):
        invoke("ingest", "file", str(note), "--title", "Molecules")
        invoke("source", "tag", "1", "gnn")
        invoke("source", "status", "1", "read")
        invoke("source", "note", "1", "excellent baseline section")
        out = invoke("source", "show", "1")
        assert "gnn" in out and "read" in out and "excellent baseline" in out
        assert "source [1]" in invoke("search", "baseline")

    def test_missing_source_errors(self):
        invoke("source", "show", "42", expect_exit=1)

    def test_edit_sets_authors_and_year(self, note):
        """E8-6: manual metadata for papers whose venue has no DOIs."""
        invoke("ingest", "file", str(note), "--title", "CEUR Workshop Paper")
        out = invoke(
            "source",
            "edit",
            "1",
            "--author",
            "Ada Smith",
            "--author",
            "Bob Jones",
            "--year",
            "2025",
        )
        assert "CEUR Workshop Paper (2025)" in out
        assert "Ada Smith, Bob Jones" in out
        out = invoke("source", "show", "1")
        assert "Ada Smith, Bob Jones" in out

    def test_edit_requires_a_change(self, note):
        invoke("ingest", "file", str(note), "--title", "T")
        invoke("source", "edit", "1", expect_exit=1)
        invoke("source", "edit", "42", "--year", "2020", expect_exit=1)


class TestFileArchive:
    """E1-11: originals live in a visible files/ dir next to the DB."""

    def test_ingest_archives_original_next_to_db(self, note, tmp_path):
        invoke("ingest", "file", str(note), "--title", "Molecules")
        files = list((tmp_path / "files").iterdir())
        assert [f.name for f in files] == ["0001-Molecules.md"]
        assert files[0].read_bytes() == note.read_bytes()
        assert str(files[0]) in invoke("source", "show", "1")

    def test_folder_ingest_archives_each_pdf(self, tmp_path):
        import pymupdf

        folder = tmp_path / "papers"
        folder.mkdir()
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "pdf body text")
        doc.save(folder / "paper.pdf")
        doc.close()
        invoke("ingest", "folder", str(folder))
        assert [f.name for f in (tmp_path / "files").iterdir()] == ["0001-paper.pdf"]

    def test_source_open_launches_archived_file(self, note, monkeypatch):
        invoke("ingest", "file", str(note), "--title", "Molecules")
        opened = []
        monkeypatch.setattr("typer.launch", lambda target: opened.append(target) or 0)
        out = invoke("source", "open", "1")
        assert opened and opened[0].endswith("0001-Molecules.md")
        assert "opened" in out

    def test_source_open_errors_without_archived_file(self, note, tmp_path):
        invoke("ingest", "file", str(note), "--title", "Molecules")
        (tmp_path / "files" / "0001-Molecules.md").unlink()
        invoke("source", "open", "1", expect_exit=1)
        invoke("source", "open", "42", expect_exit=1)

    def test_delete_source_removes_archived_file(self, note, tmp_path):
        invoke("ingest", "file", str(note), "--title", "Molecules")
        invoke("source", "delete", "1", "--yes")
        assert list((tmp_path / "files").iterdir()) == []

    def test_attach_archives_the_new_original(self, note, tmp_path):
        empty = tmp_path / "empty.md"
        empty.write_text("")
        invoke("ingest", "file", str(empty), "--title", "Bare")
        out = invoke("source", "attach", "1", str(note))
        assert "archived original" in out
        (archived,) = (tmp_path / "files").iterdir()
        assert archived.read_bytes() == note.read_bytes()


class TestIdeaAndMatchFlow:
    def test_full_flow_to_related_work(self, note, tmp_path):
        invoke(
            "ingest",
            "file",
            str(note),
            "--title",
            "Graph networks for molecules",
            "--author",
            "Ada Smith",
            "--year",
            "2021",
        )
        invoke(
            "idea",
            "new",
            "molecular ML",
            "use graph neural networks to predict molecular properties of molecules",
        )
        out = invoke("match", "suggest", "1", "--threshold", "0.05")
        assert "match [1]" in out
        invoke("match", "confirm", "1")
        out = invoke("idea", "show", "1")
        assert "confirmed: [1] Graph networks for molecules" in out
        # skeleton + bib export
        skeleton = invoke("related-work", "1")
        assert "[@smith2021graph]" in skeleton
        bib_out = tmp_path / "refs.bib"
        invoke("bib", "--idea", "1", "-o", str(bib_out))
        assert "smith2021graph" in bib_out.read_text()
        # audit a draft citing that key plus a phantom
        draft = tmp_path / "draft.tex"
        draft.write_text(r"\cite{smith2021graph} and \cite{phantom2020}")
        out = invoke("audit", str(draft), expect_exit=1)
        assert "UNKNOWN: phantom2020" in out

    def test_idea_revise_and_history(self):
        invoke("idea", "new", "t", "first version")
        invoke("idea", "revise", "1", "second version")
        out = invoke("idea", "show", "1", "--history")
        assert "first version" in out and "second version" in out

    def test_manual_match_add(self, note):
        invoke("ingest", "file", str(note), "--title", "S")
        invoke("idea", "new", "unrelated", "completely different topic")
        invoke("match", "add", "1", "1")
        assert "confirmed" in invoke("match", "list", "1")

    def test_gaps(self):
        invoke("idea", "new", "lonely", "no sources")
        out = invoke("gaps")
        assert "lonely" in out


class TestSummarise:
    def test_override_summary(self, note):
        invoke("ingest", "file", str(note), "--title", "S")
        out = invoke("summarise", "1", "--override", "A survey of GNNs for chemistry.")
        assert "survey of GNNs" in out
        assert "user, verified=True" in invoke("source", "show", "1")


class TestGraphAndContacts:
    def test_contact_add_link_and_graph(self, note, tmp_path):
        invoke("ingest", "file", str(note), "--title", "S")
        invoke("idea", "new", "i", "text about molecules and graphs")
        invoke(
            "contact",
            "add",
            "Unseen University",
            "--kind",
            "university",
            "--affiliation",
            "Ankh-Morpork",
        )
        invoke("contact", "link", "1", "--idea", "1", "--why", "potential collaboration")
        out = invoke("contact", "list")
        assert "Unseen University (university) — Ankh-Morpork" in out
        graph_file = tmp_path / "g.html"
        out = invoke("graph", "-o", str(graph_file))
        page = graph_file.read_text()
        assert "contact-1" in page
        assert 'src="http' not in page

    def test_contact_link_requires_exactly_one_target(self):
        invoke("contact", "add", "X")
        invoke("contact", "link", "1", "--why", "w", expect_exit=1)


class TestIdeaImport:
    def test_import_creates_then_skips_then_revises(self, tmp_path):
        f = tmp_path / "ideas.md"
        f.write_text("# Alpha\nfirst text\n\n# Beta\nsecond text")
        out = invoke("idea", "import", str(f))
        assert "created [1] Alpha" in out and "created [2] Beta" in out
        out = invoke("idea", "import", str(f))
        assert "skipped [1] Alpha" in out
        f.write_text("# Alpha\nrewritten text\n\n# Beta\nsecond text")
        out = invoke("idea", "import", str(f), "--on-existing", "revise")
        assert "revised [1] Alpha" in out and "skipped [2] Beta" in out
        assert "rewritten text" in invoke("idea", "show", "1")

    def test_import_invalid_format_fails(self, tmp_path):
        f = tmp_path / "bad.md"
        f.write_text("no heading here")
        result = runner.invoke(app, ["idea", "import", str(f)])
        assert result.exit_code == 1

    def test_import_missing_file_fails(self):
        invoke("idea", "import", "/nonexistent/ideas.md", expect_exit=1)


def make_pdf(path, text):
    import pymupdf

    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((72, 72), text)
    doc.save(path)
    doc.close()


class TestIngestFolder:
    def test_batch_ingest_and_rerun_skips(self, tmp_path):
        folder = tmp_path / "papers"
        folder.mkdir()
        make_pdf(folder / "attention-survey.pdf", "a survey of attention mechanisms")
        make_pdf(folder / "gnn-molecules.pdf", "graph networks for molecules")
        (folder / "notes.txt").write_text("not a pdf, must be ignored")
        out = invoke("ingest", "folder", str(folder))
        assert "ingested [1] attention-survey" in out
        assert "ingested [2] gnn-molecules" in out
        assert "2 ingested, 0 skipped, 0 failed" in out
        assert "notes" not in out
        # re-running is idempotent
        out = invoke("ingest", "folder", str(folder))
        assert "0 ingested, 2 skipped, 0 failed" in out

    def test_recursive_flag(self, tmp_path):
        folder = tmp_path / "papers"
        (folder / "sub").mkdir(parents=True)
        make_pdf(folder / "sub" / "deep.pdf", "nested paper text")
        out = invoke("ingest", "folder", str(folder))
        assert "no PDFs found" in out  # non-recursive misses the subfolder
        out = invoke("ingest", "folder", str(folder), "--recursive")
        assert "ingested [1] deep" in out

    def test_corrupt_pdf_does_not_abort_batch(self, tmp_path):
        folder = tmp_path / "papers"
        folder.mkdir()
        (folder / "broken.pdf").write_bytes(b"this is not a real pdf")
        make_pdf(folder / "good.pdf", "valid content")
        out = invoke("ingest", "folder", str(folder), expect_exit=1)
        assert "failed: broken.pdf" in out or "1 failed" in out
        assert "ingested" in out and "good" in out

    def test_missing_directory(self):
        invoke("ingest", "folder", "/nonexistent/dir", expect_exit=1)

    def test_empty_directory(self, tmp_path):
        folder = tmp_path / "empty"
        folder.mkdir()
        assert "no PDFs found" in invoke("ingest", "folder", str(folder))


class TestSummariseAll:
    def _reply(self, quote):
        import json

        return json.dumps({"summary": "Grounded batch summary.", "quotes": [quote]})

    def test_summarises_missing_then_skips_done(self, tmp_path, monkeypatch):
        a = tmp_path / "a.md"
        a.write_text("shared corpus phrase in paper alpha")
        b = tmp_path / "b.md"
        b.write_text("shared corpus phrase in paper beta")
        invoke("ingest", "file", str(a), "--title", "Alpha")
        invoke("ingest", "file", str(b), "--title", "Beta")
        monkeypatch.setenv("MUSTRUM_FAKE_LLM_RESPONSE", self._reply("shared corpus phrase"))
        out = invoke("summarise", "--all")
        assert "summarised [1] Alpha" in out and "summarised [2] Beta" in out
        assert "2 summarised, 0 skipped, 0 failed" in out
        out = invoke("summarise", "--all")
        assert "0 summarised, 2 skipped, 0 failed" in out

    def test_grounding_failure_reported_not_stored(self, tmp_path, monkeypatch):
        f = tmp_path / "a.md"
        f.write_text("actual source content")
        invoke("ingest", "file", str(f), "--title", "Alpha")
        monkeypatch.setenv("MUSTRUM_FAKE_LLM_RESPONSE", self._reply("fabricated quote"))
        out = invoke("summarise", "--all", expect_exit=1)
        assert "0 summarised, 0 skipped, 1 failed" in out
        assert "summary" not in invoke("source", "show", "1")

    def test_source_without_text_skipped(self, tmp_path, monkeypatch):
        empty = tmp_path / "empty.md"
        empty.write_text("")
        invoke("ingest", "file", str(empty), "--title", "Metadata Only")
        monkeypatch.setenv("MUSTRUM_FAKE_LLM_RESPONSE", self._reply("x"))
        out = invoke("summarise", "--all")
        assert "no text stored: [1] Metadata Only" in out
        assert "0 summarised, 1 skipped, 0 failed" in out

    def test_id_and_all_are_mutually_exclusive(self):
        invoke("summarise", expect_exit=1)
        invoke("summarise", "1", "--all", expect_exit=1)


class TestConfigCommand:
    """isolated_db pins MUSTRUM_DB to tmp_path/test.db, so the library
    config lives at tmp_path/config.toml — a global bootstrap file under
    test must use a different directory to avoid colliding with it."""

    def test_show_defaults(self, tmp_path):
        out = invoke("config", "show", "--path", str(tmp_path / "global" / "absent.toml"))
        assert "defaults in effect" in out
        assert "qwen3:30b" in out
        assert "OA PDF lookup disabled" in out
        assert str(tmp_path / "config.toml") in out  # library config path shown

    def test_init_writes_template_then_show_reads_it(self, tmp_path):
        target = tmp_path / "global" / "config.toml"
        out = invoke("config", "init", "--path", str(target))
        assert str(target) in out
        content = target.read_text()
        assert "db_path" in content and "iCloud" in content and "OneDrive" in content
        # activate a value and confirm the effective config reflects it
        target.write_text('llm_model = "llama3.1:8b"\n')
        assert "llama3.1:8b" in invoke("config", "show", "--path", str(target))

    def test_init_refuses_to_overwrite(self, tmp_path):
        target = tmp_path / "global" / "config.toml"
        target.parent.mkdir()
        target.write_text("# mine")
        invoke("config", "init", "--path", str(target), expect_exit=1)
        assert target.read_text() == "# mine"

    def test_set_writes_library_config_next_to_db(self, tmp_path):
        out = invoke("config", "set", "--llm-model", "llama3.1:8b", "--num-ctx", "8192")
        lib_config = tmp_path / "config.toml"
        assert str(lib_config) in out
        assert 'llm_model = "llama3.1:8b"' in lib_config.read_text()
        assert "num_ctx = 8192" in lib_config.read_text()
        shown = invoke("config", "show", "--path", str(tmp_path / "global" / "absent.toml"))
        assert "llama3.1:8b" in shown and "8192" in shown

    def test_set_preserves_untouched_fields(self, tmp_path):
        invoke("config", "set", "--llm-model", "llama3.1:8b")
        invoke("config", "set", "--num-ctx", "8192")
        content = (tmp_path / "config.toml").read_text()
        assert 'llm_model = "llama3.1:8b"' in content  # first call's value survives
        assert "num_ctx = 8192" in content

    def test_set_clears_unpaywall_email_with_empty_string(self, tmp_path):
        invoke("config", "set", "--unpaywall-email", "me@example.org")
        invoke("config", "set", "--unpaywall-email", "")
        assert 'unpaywall_email = ""' in (tmp_path / "config.toml").read_text()

    def test_set_requires_at_least_one_option(self):
        invoke("config", "set", expect_exit=1)

    def test_set_rejects_non_numeric_num_ctx(self):
        invoke("config", "set", "--num-ctx", "not-a-number", expect_exit=2)

    def test_models_lists_installed_and_marks_current(self, monkeypatch):
        """E12-2: same list the GUI Settings dropdowns fetch."""
        monkeypatch.setattr(
            "mustrum.adapters.ollama.list_models",
            lambda url, **kw: ["qwen3:30b", "nomic-embed-text", "llama3.1:8b"],
        )
        out = invoke("config", "models")
        assert "qwen3:30b  (llm_model)" in out
        assert "nomic-embed-text  (embed_model)" in out
        assert "llama3.1:8b" in out and "llama3.1:8b  (" not in out

    def test_models_empty_list(self, monkeypatch):
        monkeypatch.setattr("mustrum.adapters.ollama.list_models", lambda url, **kw: [])
        assert "no models found" in invoke("config", "models")

    def test_models_unreachable_ollama_fails_cleanly(self, monkeypatch):
        def raiser(url, **kw):
            from mustrum.adapters.ollama import OllamaError

            raise OllamaError("boom")

        monkeypatch.setattr("mustrum.adapters.ollama.list_models", raiser)
        invoke("config", "models", expect_exit=1)


class TestSourceAttach:
    def test_attach_pdf_to_abstract_only_source(self, tmp_path, monkeypatch):
        stub = tmp_path / "abstract-only.md"
        stub.write_text("")
        invoke("ingest", "file", str(stub), "--title", "Paywalled Paper")
        pdf = tmp_path / "downloaded.pdf"
        make_pdf(pdf, "the full downloaded paper body")
        out = invoke("source", "attach", "1", str(pdf))
        assert "attached full text to [1]" in out
        assert "source [1]" in invoke("search", "downloaded")

    def test_attach_refuses_existing_full_text(self, tmp_path, note):
        invoke("ingest", "file", str(note), "--title", "Complete")
        pdf = tmp_path / "other.pdf"
        make_pdf(pdf, "different content")
        invoke("source", "attach", "1", str(pdf), expect_exit=1)

    def test_attach_missing_file(self):
        invoke("contact", "add", "pad")
        invoke("source", "attach", "1", "/nonexistent.pdf", expect_exit=1)


class TestMatchExplain:
    def _setup_match(self, note, monkeypatch, quote="molecular property prediction"):
        import json

        invoke("ingest", "file", str(note), "--title", "Graph networks for molecules")
        invoke("idea", "new", "molecular ML", "graph networks for molecule property prediction")
        invoke("match", "suggest", "1", "--threshold", "0.05")
        monkeypatch.setenv(
            "MUSTRUM_FAKE_LLM_RESPONSE",
            json.dumps({"rationale": "Applies GNNs to molecules.", "quotes": [quote]}),
        )

    def test_explain_stores_and_prints_rationale(self, note, monkeypatch):
        self._setup_match(note, monkeypatch)
        out = invoke("match", "explain", "1")
        assert "why: Applies GNNs to molecules." in out
        assert 'evidence: "molecular property prediction"' in out
        assert "why: Applies GNNs" in invoke("match", "list", "1")

    def test_suggest_with_explain_flag(self, note, monkeypatch):
        import json

        invoke("ingest", "file", str(note), "--title", "Graph networks for molecules")
        monkeypatch.setenv(
            "MUSTRUM_FAKE_LLM_RESPONSE",
            json.dumps({"rationale": "Relevant.", "quotes": ["molecular property prediction"]}),
        )
        invoke("idea", "new", "molecular ML", "graph networks molecule property prediction")
        out = invoke("match", "suggest", "1", "--threshold", "0.05", "--explain")
        assert "why: Relevant." in out

    def test_unverifiable_rationale_fails_loudly(self, note, monkeypatch):
        self._setup_match(note, monkeypatch, quote="fabricated nonsense span")
        out = invoke("match", "explain", "1", expect_exit=1)
        assert "failed grounding" in out
        assert "why:" not in invoke("match", "list", "1")

    def test_explain_missing_match(self):
        invoke("contact", "add", "pad")
        invoke("match", "explain", "9", expect_exit=1)


class TestExportRestore:
    def test_export_restore_cycle(self, tmp_path, note, monkeypatch):
        invoke("ingest", "file", str(note), "--title", "Graph networks", "--year", "2021")
        invoke("idea", "new", "molecular ML", "graph networks molecule prediction")
        invoke("match", "suggest", "1", "--threshold", "0.05")
        invoke("match", "confirm", "1")
        export_dir = tmp_path / "backup"
        out = invoke("export", str(export_dir))
        assert "exported" in out
        assert (export_dir / "manifest.json").is_file()
        assert (export_dir / "LIBRARY.md").is_file()
        # restore into a fresh database
        monkeypatch.setenv("MUSTRUM_DB", str(tmp_path / "restored.db"))
        out = invoke("restore", str(export_dir))
        assert "restored 1 sources, 1 ideas, 1 matches, 0 contacts" in out
        assert "Graph networks (2021)" in invoke("source", "show", "1")
        assert "confirmed: [1] Graph networks" in invoke("idea", "show", "1")
        assert "source [1]" in invoke("search", "molecular")

    def test_restore_into_non_empty_db_fails(self, tmp_path, note):
        invoke("ingest", "file", str(note), "--title", "T")
        export_dir = tmp_path / "backup"
        invoke("export", str(export_dir))
        invoke("restore", str(export_dir), expect_exit=1)

    def test_export_refuses_non_empty_dir_without_force(self, tmp_path, note):
        invoke("ingest", "file", str(note), "--title", "T")
        target = tmp_path / "occupied"
        target.mkdir()
        (target / "existing.txt").write_text("data")
        invoke("export", str(target), expect_exit=1)
        invoke("export", str(target), "--force")

    def test_restore_missing_directory(self):
        invoke("restore", "/nonexistent/backup", expect_exit=1)


class TestBrainstorm:
    def _reply(self):
        import json

        return json.dumps(
            {
                "ideas": [
                    {
                        "title": "Spin-off direction",
                        "description": "Combine the graph work with new evaluation.",
                        "based_on": ["Graph networks"],
                    }
                ]
            }
        )

    def test_brainstorm_prints_labelled_output(self, note, monkeypatch):
        invoke("ingest", "file", str(note), "--title", "Graph networks")
        monkeypatch.setenv("MUSTRUM_FAKE_LLM_RESPONSE", self._reply())
        out = invoke("brainstorm", "-n", "1")
        assert "machine-generated brainstorm" in out
        assert "NOT verified" in out
        assert "Spin-off direction" in out
        assert "inspired by: Graph networks" in out
        assert "--save" in out
        # nothing stored without --save
        assert invoke("idea", "list") == ""

    def test_brainstorm_save_tags_ideas(self, note, monkeypatch):
        invoke("ingest", "file", str(note), "--title", "Graph networks")
        monkeypatch.setenv("MUSTRUM_FAKE_LLM_RESPONSE", self._reply())
        out = invoke("brainstorm", "-n", "1", "--save")
        assert "saved as idea [1] (tagged 'brainstorm')" in out
        assert "Spin-off direction" in invoke("idea", "list")

    def test_brainstorm_empty_library_fails(self):
        invoke("brainstorm", expect_exit=1)


class TestDelete:
    def test_source_delete_with_yes(self, note):
        invoke("ingest", "file", str(note), "--title", "Doomed")
        out = invoke("source", "delete", "1", "--yes")
        assert "deleted [1] Doomed" in out
        assert invoke("source", "list") == ""
        assert invoke("search", "molecular") == ""

    def test_source_delete_missing(self):
        invoke("source", "delete", "9", "--yes", expect_exit=1)

    def test_idea_delete_with_yes(self):
        invoke("idea", "new", "Doomed idea", "text")
        out = invoke("idea", "delete", "1", "--yes")
        assert "deleted idea [1] Doomed idea" in out
        assert invoke("idea", "list") == ""

    def test_delete_without_yes_prompts_and_aborts(self, note):
        invoke("ingest", "file", str(note), "--title", "Kept")
        result = runner.invoke(app, ["source", "delete", "1"], input="n\n")
        assert result.exit_code != 0
        assert "Kept" in invoke("source", "list")


class TestTitles:
    def test_folder_ingest_uses_pdf_metadata_title(self, tmp_path):
        import pymupdf

        folder = tmp_path / "papers"
        folder.mkdir()
        doc = pymupdf.open()
        doc.new_page().insert_text((72, 72), "actual content of the paper")
        doc.set_metadata({"title": "A Very Proper Paper Title"})
        doc.save(folder / "1-s2.0-S0164121225001979-main.pdf")
        doc.close()
        make_pdf(folder / "no-meta.pdf", "content without metadata")
        out = invoke("ingest", "folder", str(folder))
        assert "ingested [1] A Very Proper Paper Title" in out
        assert "ingested [2] no-meta" in out  # falls back to file name

    def test_source_rename(self, note):
        invoke("ingest", "file", str(note), "--title", "ugly_file_name_main")
        out = invoke("source", "rename", "1", "Graph Networks: A Survey")
        assert "renamed [1]" in out
        assert "Graph Networks: A Survey" in invoke("source", "show", "1")
        assert "source [1]" in invoke("search", "survey")

    def test_rename_collision_rejected(self, tmp_path, note):
        invoke("ingest", "file", str(note), "--title", "First")
        other = tmp_path / "other.md"
        other.write_text("different content entirely")
        invoke("ingest", "file", str(other), "--title", "Second")
        invoke("source", "rename", "2", "first", expect_exit=1)  # title-hash clash

    def test_rename_missing_source(self):
        invoke("source", "rename", "9", "X Y", expect_exit=1)


class TestChat:
    def _reply(self, source_id, quote, answer="Grounded chat answer."):
        import json

        return json.dumps(
            {
                "found": True,
                "answer": answer,
                "evidence": [{"source_id": source_id, "quote": quote}],
            }
        )

    def test_empty_library_no_llm_call_needed(self):
        # no candidates at all short-circuits before ever calling the LLM,
        # so this works even with no MUSTRUM_FAKE_LLM_RESPONSE configured
        out = invoke("chat", input="anything at all\nexit\n")
        assert "No sources in your library appear to address this." in out
        assert "bye." in out

    def test_grounded_turn_prints_answer_and_sources(self, note, monkeypatch):
        invoke("ingest", "file", str(note), "--title", "Graph Networks Survey")
        monkeypatch.setenv(
            "MUSTRUM_FAKE_LLM_RESPONSE",
            self._reply(1, "graph neural networks for molecular property prediction"),
        )
        out = invoke("chat", input="graph\nexit\n")
        assert "Grounded chat answer." in out
        assert "sources: [1]" in out

    def test_exits_cleanly_on_eof(self):
        out = invoke("chat", input="")
        assert "bye." in out
