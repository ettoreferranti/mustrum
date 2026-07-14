"""main()'s top-level ProviderError handler: a provider failure deep inside
a command (Ollama down, Anthropic misconfigured, ...) must print one clean
line and exit(1), not a raw traceback.

Click's `CliRunner` (used throughout tests/integration/test_cli.py) already
catches any exception raised during `app()` and reports it via
`result.exception`/`result.exit_code` — so it can't reproduce or verify this
bug, which is specifically about what happens *outside* Click's own
exception handling, at the real `mustrum` process's actual entry point.
Testing `main()` directly, with `app()` stubbed to raise, exercises exactly
that boundary without the cost/flakiness of a real subprocess."""

import pytest

from mustrum.adapters.errors import ProviderError
from mustrum.cli import main as main_module


def test_provider_error_prints_one_line_and_exits_1(monkeypatch, capsys):
    def boom() -> None:
        raise ProviderError("no Anthropic credentials found — set ANTHROPIC_API_KEY ...")

    monkeypatch.setattr(main_module, "app", boom)
    with pytest.raises(SystemExit) as exc_info:
        main_module.main()
    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "no Anthropic credentials found" in captured.err
    assert "Traceback" not in captured.err
    assert "Traceback" not in captured.out


def test_non_provider_errors_still_propagate(monkeypatch):
    """Only ProviderError is turned into a clean message — anything else
    (a genuine bug) must still surface loudly, not be silently swallowed."""

    def boom() -> None:
        raise ValueError("something else broke")

    monkeypatch.setattr(main_module, "app", boom)
    with pytest.raises(ValueError, match="something else broke"):
        main_module.main()
