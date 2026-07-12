"""Original-file archive (E1-11 / ADR-13): files land next to the DB and the
source record points at them by relative name."""

import dataclasses

import pytest

from mustrum.adapters.archive import archive_original, archived_file, delete_archived
from mustrum.adapters.sqlite.repo import SqliteRepo
from mustrum.core.models import Source, SourceKind


@pytest.fixture
def repo():
    r = SqliteRepo(":memory:")
    yield r
    r.close()


def add_source(repo, title="Attention Is All You Need"):
    return repo.add_source(Source(kind=SourceKind.PAPER, title=title))


class TestArchiveOriginal:
    def test_writes_file_and_records_relative_name(self, repo, tmp_path):
        files_dir = tmp_path / "files"
        source = add_source(repo)
        updated = archive_original(repo, files_dir, source, b"%PDF-fake", ".pdf")
        assert updated.file_path == "0001-Attention-Is-All-You-Need.pdf"
        assert (files_dir / updated.file_path).read_bytes() == b"%PDF-fake"
        assert repo.get_source(source.id).file_path == updated.file_path

    def test_suffix_is_lowercased_and_title_slugged(self, repo, tmp_path):
        source = add_source(repo, title="Weird: title/with — spaces?")
        updated = archive_original(repo, tmp_path / "files", source, b"x", ".PDF")
        assert updated.file_path == "0001-Weird-title-with-spaces.pdf"

    def test_replaces_previous_archived_file(self, repo, tmp_path):
        files_dir = tmp_path / "files"
        source = add_source(repo)
        first = archive_original(repo, files_dir, source, b"old", ".txt")
        renamed = dataclasses.replace(first, title="New Title")
        repo.update_source(renamed)
        second = archive_original(repo, files_dir, renamed, b"new", ".pdf")
        assert not (files_dir / str(first.file_path)).exists()
        assert (files_dir / str(second.file_path)).read_bytes() == b"new"

    def test_source_without_id_rejected(self, repo, tmp_path):
        with pytest.raises(ValueError):
            archive_original(repo, tmp_path, Source(kind=SourceKind.PAPER, title="t"), b"x", ".pdf")


class TestArchivedFile:
    def test_resolves_existing_file(self, repo, tmp_path):
        files_dir = tmp_path / "files"
        updated = archive_original(repo, files_dir, add_source(repo), b"x", ".pdf")
        path = archived_file(files_dir, updated)
        assert path is not None and path.read_bytes() == b"x"

    def test_none_when_nothing_recorded(self, tmp_path):
        assert archived_file(tmp_path, Source(kind=SourceKind.PAPER, title="t", id=1)) is None

    def test_none_when_file_missing_on_disk(self, tmp_path):
        source = Source(kind=SourceKind.PAPER, title="t", file_path="gone.pdf", id=1)
        assert archived_file(tmp_path, source) is None

    def test_none_when_recorded_name_escapes_archive_dir(self, tmp_path):
        # a tampered DB value must not let the web endpoint serve arbitrary files
        outside = tmp_path / "secret.txt"
        outside.write_text("secret")
        files_dir = tmp_path / "files"
        files_dir.mkdir()
        source = Source(kind=SourceKind.PAPER, title="t", file_path="../secret.txt", id=1)
        assert archived_file(files_dir, source) is None


class TestDeleteArchived:
    def test_removes_file(self, repo, tmp_path):
        files_dir = tmp_path / "files"
        updated = archive_original(repo, files_dir, add_source(repo), b"x", ".pdf")
        delete_archived(files_dir, updated)
        assert not (files_dir / str(updated.file_path)).exists()

    def test_tolerates_missing_file(self, tmp_path):
        source = Source(kind=SourceKind.PAPER, title="t", file_path="gone.pdf", id=1)
        delete_archived(tmp_path, source)  # must not raise
