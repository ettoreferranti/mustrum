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


def invoke(*args, expect_exit=0):
    result = runner.invoke(app, list(args))
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
    def test_show_defaults(self, tmp_path):
        out = invoke("config", "--path", str(tmp_path / "absent.toml"))
        assert "defaults in effect" in out
        assert "qwen3:30b" in out
        assert "OA PDF lookup disabled" in out

    def test_init_writes_template_then_show_reads_it(self, tmp_path):
        target = tmp_path / "config.toml"
        out = invoke("config", "--init", "--path", str(target))
        assert str(target) in out
        content = target.read_text()
        assert "db_path" in content and "iCloud" in content and "OneDrive" in content
        # activate a value and confirm the effective config reflects it
        target.write_text('llm_model = "llama3.1:8b"\n')
        assert "llama3.1:8b" in invoke("config", "--path", str(target))

    def test_init_refuses_to_overwrite(self, tmp_path):
        target = tmp_path / "config.toml"
        target.write_text("# mine")
        invoke("config", "--init", "--path", str(target), expect_exit=1)
        assert target.read_text() == "# mine"


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
