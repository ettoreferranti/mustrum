"""E9-3: watch-folder auto-ingest — `_scan_once`/`_move_unique` are pure
functions (no sleeping, no Typer), so the whole scan/settle/move lifecycle
is testable by calling `_scan_once` directly across simulated polls."""

import pymupdf
import pytest

from mustrum.adapters.fake import FakeEmbeddingProvider, FakeLLMProvider
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.cli.main import Context, _move_unique, _scan_once
from mustrum.config import Config
from mustrum.core.models import SourceKind
from mustrum.core.services.ingest import IngestService


def make_pdf(path, text, title=None):
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), text)
    if title is not None:
        doc.set_metadata({"title": title})
    doc.save(path)
    doc.close()


@pytest.fixture
def ctx(tmp_path):
    repo = SqliteRepo(tmp_path / "test.db")
    context = Context(
        Config(db_path=tmp_path / "test.db"), repo, FakeEmbeddingProvider(), FakeLLMProvider()
    )
    yield context
    repo.close()


@pytest.fixture
def scan(ctx):
    """scan(folder, seen={}, recursive=False) -> next_seen, wrapping
    _scan_once with ctx/service and folder/ingested/folder/failed dirs."""
    service = IngestService(ctx.repo, ctx.embedder)

    def _run(folder, seen=None, recursive=False):
        return _scan_once(
            folder,
            recursive,
            folder / "ingested",
            folder / "failed",
            seen or {},
            ctx,
            service,
            SourceKind.PAPER,
        )

    return _run


class TestScanOnce:
    def test_stable_file_ingested_after_two_scans(self, tmp_path, ctx, scan):
        folder = tmp_path / "papers"
        folder.mkdir()
        make_pdf(folder / "paper.pdf", "graph networks for molecules")

        seen = scan(folder)
        assert ctx.repo.list_sources() == []  # first scan only records the stamp

        scan(folder, seen)
        (source,) = ctx.repo.list_sources()
        assert source.title == "paper"

    def test_ingested_file_moved_out_of_the_watched_folder(self, tmp_path, ctx, scan):
        folder = tmp_path / "papers"
        folder.mkdir()
        pdf = folder / "paper.pdf"
        make_pdf(pdf, "graph networks for molecules")

        scan(folder, scan(folder))

        assert not pdf.exists()
        assert (folder / "ingested" / "paper.pdf").exists()

    def test_changing_file_is_not_ingested_yet(self, tmp_path, ctx, scan):
        """A file still being written/synced changes on every poll, so it
        must never be ingested mid-write."""
        folder = tmp_path / "papers"
        folder.mkdir()
        pdf = folder / "paper.pdf"
        make_pdf(pdf, "graph networks for molecules")

        seen = scan(folder)
        # simulate the file still being written: rewrite with different content
        make_pdf(pdf, "graph networks for molecules and much more extra text")
        seen = scan(folder, seen)

        assert ctx.repo.list_sources() == []
        assert pdf.exists()  # still in the watched folder, not moved anywhere
        scan(folder, seen)  # now it settles: one more stable scan ingests it
        assert len(ctx.repo.list_sources()) == 1

    def test_corrupt_pdf_moved_to_failed_dir(self, tmp_path, ctx, scan):
        folder = tmp_path / "papers"
        folder.mkdir()
        (folder / "broken.pdf").write_bytes(b"not a real pdf")

        scan(folder, scan(folder))

        assert ctx.repo.list_sources() == []
        assert (folder / "failed" / "broken.pdf").exists()
        assert not (folder / "broken.pdf").exists()

    def test_already_ingested_paper_is_skipped_and_moved(self, tmp_path, ctx, scan):
        """Re-ingesting the same paper under a different filename (e.g.
        after a restart re-scans a stray copy) is a safe no-op — dedup is
        by title, same as `ingest folder`."""
        folder = tmp_path / "papers"
        folder.mkdir()
        make_pdf(folder / "paper.pdf", "graph networks for molecules", title="Same Paper")
        make_pdf(folder / "duplicate.pdf", "graph networks for molecules", title="Same Paper")

        scan(folder, scan(folder))

        assert len(ctx.repo.list_sources()) == 1  # not duplicated
        assert (folder / "ingested" / "duplicate.pdf").exists()  # moved, not retried forever

    def test_moved_files_are_never_rescanned(self, tmp_path, ctx, scan):
        folder = tmp_path / "papers"
        folder.mkdir()
        make_pdf(folder / "paper.pdf", "graph networks for molecules")

        seen = scan(folder, recursive=True)
        seen = scan(folder, seen, recursive=True)
        assert len(ctx.repo.list_sources()) == 1

        # a third, recursive scan must not pick the moved file back up from ingested/
        scan(folder, seen, recursive=True)
        assert len(ctx.repo.list_sources()) == 1

    def test_non_pdf_files_ignored(self, tmp_path, ctx, scan):
        folder = tmp_path / "papers"
        folder.mkdir()
        (folder / "notes.txt").write_text("not a pdf")

        scan(folder, scan(folder))

        assert ctx.repo.list_sources() == []
        assert (folder / "notes.txt").exists()


class TestMoveUnique:
    def test_moves_into_dest(self, tmp_path):
        src = tmp_path / "a.pdf"
        src.write_bytes(b"x")
        dest_dir = tmp_path / "dest"
        dest = _move_unique(src, dest_dir)
        assert dest == dest_dir / "a.pdf"
        assert dest.read_bytes() == b"x"
        assert not src.exists()

    def test_never_overwrites_a_same_named_file(self, tmp_path):
        dest_dir = tmp_path / "dest"
        first = tmp_path / "a.pdf"
        first.write_bytes(b"first")
        _move_unique(first, dest_dir)

        second = tmp_path / "sub" / "a.pdf"
        second.parent.mkdir()
        second.write_bytes(b"second")
        moved = _move_unique(second, dest_dir)

        assert moved != dest_dir / "a.pdf"
        assert (dest_dir / "a.pdf").read_bytes() == b"first"  # untouched
        assert moved.read_bytes() == b"second"
