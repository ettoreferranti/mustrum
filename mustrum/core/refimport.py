"""Reference-manager import (E9-4): parses a BibTeX (`.bib`) or RIS (`.ris`)
library export into `ParsedReference` records ready for
`IngestService.ingest_reference`. Zotero and Mendeley both emit these two
standard formats, so one parser per format covers both tools rather than a
tool-specific API/DB integration (docs/REQUIREMENTS.md's "Zotero/Mendeley
import" framing).

Only reads what's on the page: a field absent from an entry is left absent,
never invented (NFR-1). A malformed entry is skipped with a warning rather
than aborting the whole file, so one bad record in a 500-entry export
doesn't lose the other 499.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_ARXIV_DOI = re.compile(r"10\.48550/arxiv\.(.+)", re.IGNORECASE)
_ARXIV_URL = re.compile(r"arxiv\.org/abs/([\w.\-/]+)", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedReference:
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    abstract: str = ""
    # Byte-exact source entry text; only ever set for BibTeX imports. RIS
    # has no BibTeX form, so those entries get a bib entry rendered from
    # these fields at ingest time, same as any other source with no
    # fetched BibTeX (core/bibtex.py's render_derived_entry).
    raw_bibtex: str | None = None


@dataclass(frozen=True)
class ParseResult:
    references: tuple[ParsedReference, ...]
    warnings: tuple[str, ...] = ()


def _find_arxiv_id(*candidates: str | None) -> str | None:
    for value in candidates:
        if not value:
            continue
        if match := _ARXIV_DOI.search(value):
            return match.group(1)
        if match := _ARXIV_URL.search(value):
            return match.group(1)
    return None


def _year_from(text: str) -> int | None:
    match = re.search(r"\d{4}", text)
    return int(match.group()) if match else None


# -- BibTeX -------------------------------------------------------------------

_ENTRY_START = re.compile(r"@\s*[A-Za-z]+\s*\{\s*([^,\s}]+)\s*,")


def _split_bibtex_entries(raw: str) -> list[str]:
    """Whole `@type{key, ... }` spans, found by brace-depth counting so a
    nested brace protecting capitalisation inside a field value (e.g.
    `{A} Survey`) never truncates the entry early."""
    entries = []
    for match in _ENTRY_START.finditer(raw):
        start = match.start()
        depth = 0
        brace_start = raw.index("{", start)
        end = brace_start
        for pos in range(brace_start, len(raw)):
            if raw[pos] == "{":
                depth += 1
            elif raw[pos] == "}":
                depth -= 1
                if depth == 0:
                    end = pos
                    break
        entries.append(raw[start : end + 1])
    return entries


def _split_bibtex_fields(body: str) -> dict[str, str]:
    """`body` is entry text after the outer braces and the `key,` prefix.
    Splits on top-level commas (outside any brace or quoted value) so a
    comma inside a field value never splits the field."""
    depth = 0
    in_quote = False
    start = 0
    parts = []
    for i, ch in enumerate(body):
        if ch == '"' and depth == 0:
            in_quote = not in_quote
        elif ch == "{" and not in_quote:
            depth += 1
        elif ch == "}" and not in_quote:
            depth -= 1
        elif ch == "," and depth == 0 and not in_quote:
            parts.append(body[start:i])
            start = i + 1
    tail = body[start:].strip()
    if tail:
        parts.append(tail)
    fields: dict[str, str] = {}
    for part in parts:
        if "=" not in part:
            continue
        name, _, value = part.partition("=")
        name = name.strip().lower()
        value = value.strip().rstrip(",").strip()
        if value[:1] in '{"' and value[-1:] in '}"':
            value = value[1:-1]
        # a field's outer wrapper may itself be doubled, e.g. Mendeley's
        # '{{Title}}' — strip once more if still fully braced
        if value[:1] == "{" and value[-1:] == "}":
            value = value[1:-1]
        # remaining braces are LaTeX case-protection, not content
        value = value.replace("{", "").replace("}", "")
        fields[name] = re.sub(r"\s+", " ", value).strip()
    return fields


def parse_bibtex(raw: str) -> ParseResult:
    references = []
    warnings = []
    for entry in _split_bibtex_entries(raw):
        key_match = _ENTRY_START.match(entry)
        assert key_match  # entry was located by this same regex
        body = entry[entry.index("{") + 1 : -1]
        _, _, rest = body.partition(",")
        fields = _split_bibtex_fields(rest)
        title = fields.get("title", "")
        if not title:
            warnings.append(f"skipped entry {key_match.group(1)!r}: no title field")
            continue
        authors = tuple(a.strip() for a in fields.get("author", "").split(" and ") if a.strip())
        year = int(fields["year"]) if fields.get("year", "").isdigit() else None
        doi = fields.get("doi") or None
        arxiv_id = None
        if fields.get("archiveprefix", "").lower() == "arxiv" and fields.get("eprint"):
            arxiv_id = fields["eprint"]
        arxiv_id = arxiv_id or _find_arxiv_id(doi, fields.get("url"), fields.get("journal"))
        references.append(
            ParsedReference(
                title=title,
                authors=authors,
                year=year,
                doi=doi,
                arxiv_id=arxiv_id,
                abstract=fields.get("abstract", ""),
                raw_bibtex=entry,
            )
        )
    return ParseResult(tuple(references), tuple(warnings))


# -- RIS ------------------------------------------------------------------------

_RIS_TAG = re.compile(r"^([A-Za-z0-9]{2})\s{0,2}-\s?(.*)$")


def parse_ris(raw: str) -> ParseResult:
    references = []
    warnings = []
    record: dict[str, list[str]] = {}
    for lineno, line in enumerate(raw.splitlines(), start=1):
        line = line.rstrip()
        if not line.strip():
            continue
        match = _RIS_TAG.match(line)
        if not match:
            continue  # an unsupported continuation line; ignore, don't fail the record
        tag, value = match.group(1).upper(), match.group(2).strip()
        if tag == "TY":
            record = {}
        elif tag == "ER":
            ref, warning = _ris_record_to_reference(record, lineno)
            if ref is not None:
                references.append(ref)
            if warning:
                warnings.append(warning)
            record = {}
        else:
            record.setdefault(tag, []).append(value)
    return ParseResult(tuple(references), tuple(warnings))


def _ris_record_to_reference(
    record: dict[str, list[str]], lineno: int
) -> tuple[ParsedReference | None, str | None]:
    title = (record.get("TI") or record.get("T1") or [""])[0]
    if not title:
        return None, f"skipped RIS record ending at line {lineno}: no TI/T1 title tag"
    authors = tuple(record.get("AU", ()))
    year_field = (record.get("PY") or record.get("Y1") or [""])[0]
    year = _year_from(year_field) if year_field else None
    do_values = record.get("DO")
    doi = do_values[0] if do_values else None
    abstract = " ".join(record.get("AB") or record.get("N2") or ())
    arxiv_id = _find_arxiv_id(doi, *(record.get("UR") or ()))
    return (
        ParsedReference(
            title=title,
            authors=authors,
            year=year,
            doi=doi,
            arxiv_id=arxiv_id,
            abstract=abstract,
        ),
        None,
    )
