"""Shared scaffolding for the runnable examples.

Every example follows the same shape so the focus stays on the one concept it
teaches rather than on plumbing:

* It runs against a **real provider** when a credential (or explicit opt-in) is
  present, selected with :func:`select_adapter`.
* It falls back to a small **deterministic offline adapter** otherwise, so the
  example runs with no credentials and doubles as a CI smoke test.
* It exposes ``main(adapter=None)`` returning its headline numbers, so the test
  suite can pin the offline behavior.

Provider selection is driven by environment variables:

* ``PROMPTLENS_EXAMPLE_PROVIDER`` — which provider to use; defaults to the
  example's own ``prefer`` (usually ``openai``).
* ``PROMPTLENS_EXAMPLE_MODEL`` — optional model id override.

A provider only goes live when its **activation signal** is present, so nothing
makes a surprise network call:

* keyed providers (openai, anthropic, gemini, grok, bedrock, copilot) activate
  when their API-key / credential environment variable is set;
* keyless local providers (ollama, openai-compatible) activate only when you
  explicitly select them via ``PROMPTLENS_EXAMPLE_PROVIDER``.
"""

from __future__ import annotations

import os
from pathlib import Path

from promptlens.core.base import Adapter

# Provider -> the environment variables whose presence means "you can run this
# provider for real". ``None`` marks a keyless provider that activates only on
# an explicit PROMPTLENS_EXAMPLE_PROVIDER opt-in (it has no credential to probe).
_ACTIVATION_ENVS: dict[str, tuple[str, ...] | None] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    "copilot": ("GITHUB_COPILOT_TOKEN", "COPILOT_API_KEY", "GITHUB_TOKEN"),
    "ollama": None,
    "openai-compatible": ("OPENAI_COMPATIBLE_BASE_URL", "OPENAI_BASE_URL"),
}


def load_text(example_file: str, name: str) -> str:
    """Read a sibling data file (prompt, instructions) next to an example script."""
    return (Path(example_file).resolve().parent / name).read_text(encoding="utf-8")


def select_adapter(
    offline: Adapter, *, prefer: str = "openai", temperature: float = 0.0
) -> Adapter:
    """Return a real provider adapter when one is configured, else ``offline``.

    ``prefer`` is the example's default provider, overridable per run with
    ``PROMPTLENS_EXAMPLE_PROVIDER``. ``offline`` is the example's deterministic
    simulated adapter, used whenever the chosen provider is not activated so the
    demo (and the CI smoke test) keeps working without network access.
    """
    requested = os.environ.get("PROMPTLENS_EXAMPLE_PROVIDER", "").strip().lower()
    provider = requested or prefer
    if provider not in _ACTIVATION_ENVS:
        supported = ", ".join(sorted(_ACTIVATION_ENVS))
        print(
            f"[promptlens] Unknown PROMPTLENS_EXAMPLE_PROVIDER='{provider}' "
            f"(supported: {supported}); using the offline simulated adapter."
        )
        return offline
    if not _is_activated(provider, explicit=bool(requested)):
        print(_inactive_notice(provider))
        return offline
    # Imported lazily so the offline path never needs a provider SDK installed.
    from promptlens.cli.factories import build_adapter

    model = os.environ.get("PROMPTLENS_EXAMPLE_MODEL") or None
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    adapter = build_adapter(
        provider,
        model,
        temperature=temperature,
        base_url=base_url if provider == "openai-compatible" else None,
    )
    print(f"[promptlens] Using real provider '{provider}' (model: {adapter.model}).")
    return adapter


def _is_activated(provider: str, *, explicit: bool) -> bool:
    envs = _ACTIVATION_ENVS[provider]
    if envs is None:
        # Keyless local provider: only when the user explicitly asked for it.
        return explicit
    return any(os.environ.get(name) for name in envs)


def _inactive_notice(provider: str) -> str:
    envs = _ACTIVATION_ENVS[provider]
    if envs is None:
        return (
            f"[promptlens] Using the offline simulated adapter. Set "
            f"PROMPTLENS_EXAMPLE_PROVIDER={provider} (with a local server running) "
            f"to run this example against '{provider}'."
        )
    hint = " or ".join(envs)
    return (
        f"[promptlens] No {hint} set; using the offline simulated adapter. Export "
        f"{envs[0]} (and optionally PROMPTLENS_EXAMPLE_MODEL) to run against '{provider}'."
    )


def print_footer(message: str) -> None:
    """Print the shared 'lens, not oracle' caveat that closes every example."""
    print(f"\nLens, not oracle: {message}")
