"""Privacy guard: this repo is public, so no real personal data may ever be
tracked. Fails the suite if an email-like string (outside documentation
placeholders) or a user home path appears in any tracked file."""

import re
import subprocess
from pathlib import Path

# resolved through git so the test scans the real repository even when the
# suite runs from a copied tree (mutmut's mutants/ sandbox)
REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
)

EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
ALLOWED_EMAIL_DOMAINS = {"example.com", "example.org", "anthropic.com"}
HOME_PATH = re.compile(r"/(?:Users|home)/[a-z0-9_-]+/")

# third-party or machine-generated files exempt from scanning
SKIP_PREFIXES = ("mustrum/graph/vendor/",)
SKIP_FILES = {"uv.lock"}


def tracked_text_files():
    output = subprocess.run(
        ["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout
    for name in output.splitlines():
        if name in SKIP_FILES or name.startswith(SKIP_PREFIXES):
            continue
        path = REPO_ROOT / name
        if not path.is_file():
            continue
        try:
            yield name, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue  # binary


class TestNoPersonalData:
    def test_no_real_email_addresses_in_tracked_files(self):
        offenders = []
        for name, text in tracked_text_files():
            for match in EMAIL.finditer(text):
                domain = match.group(0).rsplit("@", 1)[1].lower()
                if domain not in ALLOWED_EMAIL_DOMAINS:
                    offenders.append(f"{name}: {match.group(0)}")
        assert not offenders, f"real email addresses in tracked files: {offenders}"

    def test_no_user_home_paths_in_tracked_files(self):
        offenders = [
            f"{name}: {HOME_PATH.search(text).group(0)}"
            for name, text in tracked_text_files()
            if HOME_PATH.search(text)
        ]
        assert not offenders, f"user home paths in tracked files: {offenders}"

    def test_generated_artefacts_are_gitignored(self):
        gitignore = (REPO_ROOT / ".gitignore").read_text()
        for pattern in ("*.db", "mustrum-graph.html", "*.bib"):
            assert pattern in gitignore
