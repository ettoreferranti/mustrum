"""Shared base for provider-adapter failures (Ollama, Anthropic, ...).

Exists so callers — the CLI's top-level handler, the GUI's exception
handler — can catch any provider's failure by this one class without
eagerly importing each adapter's (and its SDK's) module.
"""

from __future__ import annotations


class ProviderError(RuntimeError):
    pass
