"""Shared scaffolding for the runnable examples.

Every example runs against a **real provider** — there are no simulated
models. ``require_adapter`` resolves the provider from your environment and
exits with setup instructions when no credential is present.

* ``PROMPTLENS_EXAMPLE_PROVIDER`` — which provider to use; defaults to the
  example's own ``prefer`` (usually ``openai``).
* ``PROMPTLENS_EXAMPLE_MODEL`` — optional model id override.

Each example exposes ``main(adapter=None)`` so the test suite can inject a
deterministic stub adapter and pin the documented behavior without network
access; the examples themselves always call a real model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel

from promptlens.core.base import Adapter

# Provider -> credential env vars that must be present to run it. ``()`` marks
# a local provider that needs a running server rather than a key.
_CREDENTIAL_ENVS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    "copilot": ("GITHUB_COPILOT_TOKEN", "COPILOT_API_KEY", "GITHUB_TOKEN"),
    "ollama": (),
    "openai-compatible": (),
}


def load_text(example_file: str, name: str) -> str:
    """Read a sibling data file (prompt, instructions) next to an example script."""
    return (Path(example_file).resolve().parent / name).read_text(encoding="utf-8")


def require_adapter(*, prefer: str = "openai", temperature: float = 0.0) -> Adapter:
    """Build a real provider adapter, or exit with setup instructions.

    ``prefer`` is the example's default provider, overridable per run with
    ``PROMPTLENS_EXAMPLE_PROVIDER``.
    """
    console = Console()
    requested = os.environ.get("PROMPTLENS_EXAMPLE_PROVIDER", "").strip().lower()
    provider = requested or prefer
    if provider not in _CREDENTIAL_ENVS:
        supported = ", ".join(sorted(_CREDENTIAL_ENVS))
        console.print(
            f"[red]Unknown PROMPTLENS_EXAMPLE_PROVIDER='{provider}'.[/red] "
            f"Supported: {supported}"
        )
        sys.exit(1)
    envs = _CREDENTIAL_ENVS[provider]
    if envs and not any(os.environ.get(name) for name in envs):
        hint = " or ".join(envs)
        console.print(
            Panel(
                f"This example makes real calls to [bold]{provider}[/bold] and "
                f"needs {hint} set.\n\n"
                f"  export {envs[0]}=...\n"
                f"  python {sys.argv[0]}\n\n"
                "Or pick another provider with PROMPTLENS_EXAMPLE_PROVIDER="
                f"{'|'.join(sorted(_CREDENTIAL_ENVS))}.",
                title="[bold red]Missing credentials[/bold red]",
                border_style="red",
            )
        )
        sys.exit(1)
    # Imported lazily so reading this module never needs a provider SDK installed.
    from promptlens.cli.factories import build_adapter

    model = os.environ.get("PROMPTLENS_EXAMPLE_MODEL") or None
    base_url = os.environ.get("OPENAI_COMPATIBLE_BASE_URL") or os.environ.get("OPENAI_BASE_URL")
    adapter = build_adapter(
        provider,
        model,
        temperature=temperature,
        base_url=base_url if provider == "openai-compatible" else None,
    )
    console.print(
        f"[dim]Provider: [bold]{provider}[/bold] · model: [bold]{adapter.model}[/bold][/dim]"
    )
    return adapter


def print_footer(message: str) -> None:
    """Print the shared 'lens, not oracle' caveat that closes every example."""
    Console().print(f"\n[dim]Lens, not oracle: {message}[/dim]")
