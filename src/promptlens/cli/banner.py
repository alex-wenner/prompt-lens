"""The promptlens banner and shared CLI console."""

from __future__ import annotations

from rich.console import Console

# "promptlens" in a compact box-drawing face, with a lens over the prompt.
_BANNER = """\
┌─┐┬─┐┌─┐┌┬┐┌─┐┌┬┐┬  ┌─┐┌┐┌┌─┐
├─┘├┬┘│ ││││├─┘ │ │  ├┤ │││└─┐
┴  ┴└─└─┘┴ ┴┴   ┴ ┴─┘└─┘┘└┘└─┘"""

_TAGLINE = "see which parts of your prompt actually matter"


def print_banner(console: Console | None = None) -> None:
    """Print the promptlens banner and tagline."""
    console = console or Console()
    console.print(f"[bold cyan]{_BANNER}[/bold cyan]")
    console.print(f"[dim]{_TAGLINE}[/dim]\n")
