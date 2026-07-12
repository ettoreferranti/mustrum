"""Original-file archive (E1-11 / ADR-13).

The original of every ingested source (PDF, plain text, Markdown) is kept as
a plain file in a visible `files/` directory next to the SQLite database, so
database + originals form one backup unit. The database stores only the file
name relative to that directory (`Source.file_path`); all file I/O lives here
in the adapter layer — core services never touch the filesystem.
"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path

from mustrum.core.models import Source
from mustrum.core.ports import StorageRepo

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(title: str, max_len: int = 60) -> str:
    slug = _UNSAFE.sub("-", title).strip("-.")[:max_len].rstrip("-.")
    return slug or "source"


def archive_original(
    repo: StorageRepo, files_dir: Path, source: Source, data: bytes, suffix: str
) -> Source:
    """Store the original bytes as `<id>-<title-slug><suffix>` in files_dir
    and record the name on the source. A previously archived file is
    replaced — the new original is now the source's original."""
    if source.id is None:
        raise ValueError("source has no id")
    files_dir.mkdir(parents=True, exist_ok=True)
    name = f"{source.id:04d}-{_slug(source.title)}{suffix.lower()}"
    (files_dir / name).write_bytes(data)
    if source.file_path and source.file_path != name:
        (files_dir / source.file_path).unlink(missing_ok=True)
    updated = dataclasses.replace(source, file_path=name)
    repo.update_source(updated)
    return updated


def archived_file(files_dir: Path, source: Source) -> Path | None:
    """Absolute path of the archived original, or None when the source has
    none recorded, the file has gone missing on disk, or the recorded name
    escapes the archive directory (defence against a tampered DB value)."""
    if not source.file_path:
        return None
    path = files_dir / source.file_path
    if not path.is_file():
        return None
    if not path.resolve().is_relative_to(files_dir.resolve()):
        return None
    return path


def delete_archived(files_dir: Path, source: Source) -> None:
    """Remove the archived original (part of the source-delete cascade)."""
    path = archived_file(files_dir, source)
    if path is not None:
        path.unlink(missing_ok=True)
