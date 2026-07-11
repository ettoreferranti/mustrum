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
        invoke("contact", "add", "ZHAW InIT", "--kind", "university", "--affiliation", "Zurich")
        invoke("contact", "link", "1", "--idea", "1", "--why", "potential collaboration")
        out = invoke("contact", "list")
        assert "ZHAW InIT (university) — Zurich" in out
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
