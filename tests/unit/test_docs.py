"""Documentation guard: every CLI command must appear in the README, so the
docs can never drift from the implementation."""

import subprocess
from pathlib import Path

from mustrum.cli.main import app

REPO_ROOT = Path(
    subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
)


def all_cli_invocations() -> list[str]:
    """Full command strings, e.g. 'mustrum ingest doi', from the typer app."""
    invocations = []
    for command in app.registered_commands:
        assert command.name, f"command {command.callback} needs an explicit name"
        invocations.append(f"mustrum {command.name}")
    for group in app.registered_groups:
        assert group.name, "sub-app needs an explicit name"
        assert group.typer_instance is not None
        for command in group.typer_instance.registered_commands:
            assert command.name, f"command {command.callback} needs an explicit name"
            invocations.append(f"mustrum {group.name} {command.name}")
    return invocations


class TestReadmeCoversCli:
    def test_every_command_is_documented_in_readme(self):
        readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
        missing = [inv for inv in all_cli_invocations() if inv not in readme]
        assert not missing, (
            f"commands missing from README.md (document them in the command reference): {missing}"
        )

    def test_the_cli_actually_has_commands(self):
        # guards the guard: an import/introspection failure must not pass silently
        assert len(all_cli_invocations()) >= 20
