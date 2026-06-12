"""Rich rendering helpers shared by the CLI commands and the wizard."""

from __future__ import annotations

from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from promptlens.core.base import CompletionOutput, ToolDefinitions, normalize_tool
from promptlens.core.result import format_tool_call

_MAX_TEXT = 600


def render_tools(tools: ToolDefinitions | None, console: Console | None = None) -> None:
    """Show every tool the model sees: name, docstring, and typed parameters.

    Attribution over tool-using prompts is only interpretable if you can see
    what the model could call, so the CLI prints this before the run.
    """
    if not tools:
        return
    console = console or Console()
    tables: list[Any] = []
    for definition in tools:
        normalized = normalize_tool(definition)
        name = str(normalized.get("name", "?"))
        description = str(normalized.get("description", "") or "").strip()
        header = Text(name, style="bold cyan")
        if description:
            header.append("  —  ", style="dim")
            header.append(description.splitlines()[0])
        table = Table(
            show_header=True, header_style="dim", box=None, padding=(0, 1), title=None
        )
        table.add_column("parameter", style="green")
        table.add_column("type", style="magenta")
        table.add_column("required", justify="center")
        table.add_column("description")
        schema = normalized.get("parameters") or {}
        required = set(schema.get("required") or [])
        for parameter_name, parameter in (schema.get("properties") or {}).items():
            table.add_row(
                Text(parameter_name),
                Text(str(parameter.get("type", "string"))),
                "[green]yes[/green]" if parameter_name in required else "[dim]no[/dim]",
                Text(str(parameter.get("description", "") or "")),
            )
        tables.append(Group(header, table))
    console.print(
        Panel(
            Group(*tables),
            title=f"[bold]Tools the model sees ({len(tools)})[/bold]",
            border_style="magenta",
        )
    )


def render_baseline(output: CompletionOutput, console: Console | None = None) -> None:
    """Show the baseline run: the tool trajectory and the model's reply.

    For agent-style runs the reply is the model's response *after* tool results
    came back, so this panel captures both halves of the behavior attribution
    will measure drift against.
    """
    console = console or Console()
    body: list[Any] = []
    if output.tool_calls:
        trace = Table(show_header=False, box=None, padding=(0, 1))
        trace.add_column(style="dim", justify="right")
        trace.add_column()
        for index, call in enumerate(output.tool_calls, start=1):
            trace.add_row(f"{index}.", Text(format_tool_call(call), style="cyan"))
        body.append(Text("Tool trajectory", style="bold"))
        body.append(trace)
        body.append(Text("Model reply (after tool results)", style="bold"))
    else:
        body.append(Text("Model reply", style="bold"))
    text = output.text.strip() or "(empty)"
    if len(text) > _MAX_TEXT:
        text = text[:_MAX_TEXT] + " …"
    body.append(Text(text))
    if output.usage is not None:
        body.append(
            Text(
                f"usage: {output.usage.input_tokens} input / "
                f"{output.usage.output_tokens} output tokens",
                style="dim",
            )
        )
    console.print(
        Panel(Group(*body), title="[bold]Baseline run[/bold]", border_style="green")
    )
