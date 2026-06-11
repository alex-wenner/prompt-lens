"""Shared scaffolding for the runnable examples.

Every example runs against a **real provider** — there are no simulated models
and no offline fallbacks. Export an API key before running:

* ``OPENAI_API_KEY`` (the default provider for most examples)
* or any other supported provider, selected with environment variables:

  * ``PROMPTLENS_EXAMPLE_PROVIDER`` — openai, anthropic, gemini, grok, bedrock,
    copilot, ollama, or openai-compatible.
  * ``PROMPTLENS_EXAMPLE_MODEL`` — optional model id override.

The test suite injects its own stub adapters through each example's
``main(adapter=...)`` parameter, so CI never makes network calls.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from promptlens.core.base import Adapter, CompletionOutput, Tool, ToolDefinitions, normalize_tool

console = Console()

# Provider -> the environment variables whose presence supplies its credential.
_CREDENTIAL_ENVS: dict[str, tuple[str, ...]] = {
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "grok": ("XAI_API_KEY", "GROK_API_KEY"),
    "bedrock": ("AWS_ACCESS_KEY_ID", "AWS_PROFILE"),
    "copilot": ("GITHUB_COPILOT_TOKEN", "COPILOT_API_KEY", "GITHUB_TOKEN"),
    # Keyless local providers: a running server is the only requirement.
    "ollama": (),
    "openai-compatible": (),
}


def load_text(example_file: str, name: str) -> str:
    """Read a sibling data file (prompt, instructions) next to an example script."""
    return (Path(example_file).resolve().parent / name).read_text(encoding="utf-8")


def get_adapter(*, prefer: str = "openai", temperature: float = 0.0) -> Adapter:
    """Build a real provider adapter, exiting with guidance when no key is set.

    ``prefer`` is the example's default provider; override it per run with
    ``PROMPTLENS_EXAMPLE_PROVIDER`` (and optionally ``PROMPTLENS_EXAMPLE_MODEL``).
    """
    provider = os.environ.get("PROMPTLENS_EXAMPLE_PROVIDER", "").strip().lower() or prefer
    if provider not in _CREDENTIAL_ENVS:
        supported = ", ".join(sorted(_CREDENTIAL_ENVS))
        console.print(
            f"[bold red]Unknown PROMPTLENS_EXAMPLE_PROVIDER='{provider}'[/bold red] "
            f"(supported: {supported})."
        )
        raise SystemExit(1)
    envs = _CREDENTIAL_ENVS[provider]
    if envs and not any(os.environ.get(name) for name in envs):
        hint = " or ".join(envs)
        console.print(
            Panel(
                Text.from_markup(
                    f"This example makes real [bold]{provider}[/bold] calls and needs a "
                    f"credential.\n\nExport [bold cyan]{envs[0]}[/bold cyan]"
                    + (f" (or {hint})" if len(envs) > 1 else "")
                    + " and re-run, or pick another provider with "
                    "[bold cyan]PROMPTLENS_EXAMPLE_PROVIDER[/bold cyan].",
                ),
                title="[bold red]Missing API key[/bold red]",
                border_style="red",
            )
        )
        raise SystemExit(1)
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
        f"[dim]Provider:[/dim] [bold magenta]{provider}[/bold magenta] "
        f"[dim]model:[/dim] [bold magenta]{adapter.model}[/bold magenta]\n"
    )
    return adapter


def print_tools(tools: ToolDefinitions) -> None:
    """Render every tool the model sees: name, docstring, parameters, and types."""
    table = Table(
        title="Tools exposed to the model",
        title_style="bold cyan",
        border_style="cyan",
        show_lines=True,
    )
    table.add_column("Tool", style="bold magenta", no_wrap=True)
    table.add_column("Description")
    table.add_column("Parameters")
    for item in tools:
        if isinstance(item, Tool):
            name, description = item.name, item.description
            schema = item.json_schema()
        else:
            normalized = normalize_tool(item)
            name = str(normalized.get("name", "?"))
            description = str(normalized.get("description", ""))
            schema = normalized.get("parameters") or {}
        required = set(schema.get("required") or [])
        parameters = Text()
        for parameter_name, spec in (schema.get("properties") or {}).items():
            requirement = "required" if parameter_name in required else "optional"
            parameters.append(f"{parameter_name}: ", style="bold")
            parameters.append(f"{spec.get('type', 'string')} ({requirement})\n", style="green")
            if spec.get("description"):
                parameters.append(f"  {spec['description']}\n", style="dim")
        table.add_row(Text(name), Text(description), parameters)
    console.print(table)


def print_completion(title: str, output: CompletionOutput) -> None:
    """Show a model response in full: text, tool calls, and token usage."""
    body = Text(output.text.strip() or "(no text output)")
    subtitle = (
        f"usage: {output.usage.input_tokens} in / {output.usage.output_tokens} out"
        if output.usage
        else None
    )
    console.print(
        Panel(body, title=f"[bold]{title}[/bold]", subtitle=subtitle, border_style="green")
    )
    if output.tool_calls:
        table = Table(
            title=f"{title} — tool calls", title_style="bold", border_style="green"
        )
        table.add_column("Tool", style="bold magenta")
        table.add_column("Arguments")
        for call in output.tool_calls:
            arguments = call.get("arguments")
            rendered = arguments if isinstance(arguments, str) else json.dumps(arguments)
            table.add_row(Text(str(call.get("name"))), Text(str(rendered)))
        console.print(table)


def complete_tool_round_trip(
    adapter: Adapter,
    prompt: str,
    output: CompletionOutput,
    implementations: dict[str, Any],
    tools: ToolDefinitions,
) -> CompletionOutput | None:
    """Execute the model's tool calls and ask it to respond to the results.

    This closes the agent loop the attribution sweep observes indirectly: the
    tools run (against the example's stub implementations), their outputs are
    appended to the conversation, and the model produces its final reply.
    Returns ``None`` when the model called no tools.
    """
    if not output.tool_calls:
        return None
    results: list[str] = []
    for call in output.tool_calls:
        name = str(call.get("name"))
        arguments = call.get("arguments")
        if isinstance(arguments, str):
            arguments = json.loads(arguments) if arguments.strip() else {}
        implementation = implementations.get(name)
        result = (
            implementation(**(arguments or {}))
            if implementation is not None
            else f"(no implementation for {name})"
        )
        results.append(f"- {name}({json.dumps(arguments)}) -> {result}")
    follow_up = (
        f"{prompt}\n\nTool results:\n" + "\n".join(results) + "\n\n"
        "Use these tool results to give the user a final answer."
    )
    return adapter.complete(follow_up, tools=tools)


def print_footer(message: str) -> None:
    """Print the shared 'lens, not oracle' caveat that closes every example."""
    console.print(f"\n[bold cyan]Lens, not oracle:[/bold cyan] {message}")


def confirm_spend(estimate_total_usd: float, evaluations: int) -> None:
    """Ask before the sweep when running interactively; proceed in pipes/CI."""
    if not sys.stdin.isatty():
        return
    from rich.prompt import Confirm

    if not Confirm.ask(
        f"Run [bold]{evaluations}[/bold] more provider calls "
        f"(~[bold green]${estimate_total_usd:.4f}[/bold green] total)?",
        default=True,
    ):
        raise SystemExit(0)
