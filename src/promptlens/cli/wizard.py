"""Interactive guided attribution runs.

``promptlens wizard`` walks through every choice an attribution run needs —
provider, model, segmentation, drill-down, scorer, masking, scale, synopsis —
with explanations and sensible defaults at each step, shows the cost estimate
before any provider call, runs the experiment, and finally prints the
equivalent non-interactive ``promptlens explain`` command so the configured
run can be scripted or shared.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

from promptlens import AttributionHarness, explain_drilldown
from promptlens.cli.banner import print_banner
from promptlens.cli.factories import build_adapter, build_masker, build_sampler, build_scorer
from promptlens.core.base import Scorer
from promptlens.reporters import LLMSynopsisWriter
from promptlens.scorers import ToolAccuracyScorer

_PROVIDERS: list[tuple[str, str]] = [
    ("echo", "offline, free — returns the prompt; great for a first try"),
    ("openai", "OpenAI via OPENAI_API_KEY"),
    ("anthropic", "Anthropic via ANTHROPIC_API_KEY"),
    ("bedrock", "Amazon Bedrock via AWS credentials"),
    ("copilot", "GitHub Copilot via GITHUB_COPILOT_TOKEN"),
    ("grok", "xAI Grok via XAI_API_KEY"),
    ("gemini", "Google Gemini via GEMINI_API_KEY"),
    ("ollama", "local models, free — defaults to localhost:11434"),
    ("openai-compatible", "any OpenAI-compatible endpoint (vLLM, gateways)"),
]

_SCORERS: list[tuple[str, str]] = [
    ("length", "output-length drift; offline, works with any provider"),
    ("embedding-local", "deterministic text-shape drift; offline smoke runs only"),
    ("logprob", "token log-probability drift; needs a logprobs-capable model"),
    ("tool-sequence", "did the model call different tools, in a different order?"),
    ("tool-args", "tool-sequence drift plus weighted tool-argument changes"),
    ("tool-call", "objective: did it call the expected tool with required args?"),
]

_SEGMENTERS: list[tuple[str, str]] = [
    ("auto", "pick from the prompt's shape: headings > paragraphs > sentences"),
    ("sentences", "one feature per sentence (finest, most provider calls)"),
    ("paragraphs", "one feature per blank-line block"),
    ("sections", "one feature per markdown heading section"),
    ("tools", "one feature per tool-schema parameter"),
]

_MASKERS: list[tuple[str, str]] = [
    ("placeholder", "replace masked text with [...] — keeps structure visible"),
    ("drop", "remove masked text entirely"),
    ("filler", "replace with neutral filler of the same length"),
]


def run_wizard() -> None:
    """Interactively configure and run an attribution experiment."""
    console = Console()
    print_banner(console)

    from promptlens.cli.main import _read_prompt, _read_tools, _segmenter

    prompt_text = _read_prompt(Prompt.ask("[bold]Prompt text or path to a prompt file[/bold]"))

    provider = _choice(console, "Provider", _PROVIDERS, default="echo")
    model = Prompt.ask("Model id", default="auto")
    model_id = None if model == "auto" else model
    base_url = _ask_base_url(provider)

    segmenter_name = _choice(console, "Segmenter", _SEGMENTERS, default="auto")
    drilldown, drilldown_top = _ask_drilldown(console, prompt_text, segmenter_name)

    scorer_name = _choice(console, "Scorer", _SCORERS, default="length")
    scorer, scorer_config = _build_scorer(scorer_name)

    tools_path = Prompt.ask("Tool schema JSON file (optional)", default="none")
    tools = _read_tools(None if tools_path == "none" else tools_path)

    masker_name = _choice(console, "Masker", _MASKERS, default="placeholder")
    scale = Prompt.ask(
        "Perturbation scale (repeats per sweep: quick=1, standard=3, full=5)",
        choices=["quick", "standard", "full"],
        default="quick",
    )
    synopsis, synopsis_provider, synopsis_model = _ask_synopsis(provider)

    try:
        harness = AttributionHarness(
            adapter=build_adapter(
                provider=provider, model=model_id, temperature=0.0, base_url=base_url
            ),
            segmenter=_segmenter(segmenter_name, prompt_text),
            scorer=scorer,
            masker=build_masker(masker_name),
            sampler=build_sampler("leave-one-out", scale=scale),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print()
    harness.estimate(prompt_text, tools=tools).print()
    if drilldown:
        console.print(
            "[dim]The estimate covers the overview pass; each refined section "
            "adds roughly one sentence sweep on top.[/dim]"
        )
    if not Confirm.ask("\n[bold]Run attribution now?[/bold]", default=True):
        raise typer.Exit()

    if drilldown:
        result: Any = explain_drilldown(harness, prompt_text, tools=tools, top_k=drilldown_top)
        synopsis_target = result.overview
    else:
        result = harness.explain(prompt_text, tools=tools)
        synopsis_target = result
    if synopsis:
        writer = LLMSynopsisWriter(
            build_adapter(
                provider=synopsis_provider,
                model=synopsis_model,
                temperature=0.0,
                base_url=base_url if synopsis_provider == provider else None,
            )
        )
        enriched = synopsis_target.with_synopsis(writer.summarize(prompt_text, synopsis_target))
        result = (
            result.model_copy(update={"overview": enriched}) if drilldown else enriched
        )

    console.print()
    result.print()

    if Confirm.ask("\nSave the full result as JSON?", default=False):
        path = Prompt.ask("Output path", default="attribution.json")
        Path(path).write_text(result.to_json(), encoding="utf-8")
        console.print(f"[green]Saved {path}[/green]")

    _print_equivalent_command(
        console,
        provider=provider,
        model=model_id,
        base_url=base_url,
        segmenter=segmenter_name,
        scorer=scorer_name,
        scorer_config=scorer_config,
        masker=masker_name,
        scale=scale,
        drilldown=drilldown,
        drilldown_top=drilldown_top,
        tools_path=None if tools_path == "none" else tools_path,
        synopsis=synopsis,
        synopsis_provider=synopsis_provider,
        synopsis_model=synopsis_model,
    )


def _choice(
    console: Console, title: str, options: list[tuple[str, str]], *, default: str
) -> str:
    table = Table(title=title, show_header=False, title_justify="left")
    table.add_column(style="bold cyan")
    table.add_column(style="dim")
    for name, note in options:
        table.add_row(name, note)
    console.print(table)
    return Prompt.ask(
        f"[bold]{title}[/bold]", choices=[name for name, _ in options], default=default
    )


def _ask_base_url(provider: str) -> str | None:
    if provider == "ollama":
        return Prompt.ask("Ollama endpoint", default="http://localhost:11434/v1")
    if provider == "openai-compatible":
        return Prompt.ask("Endpoint base URL (e.g. http://localhost:8000/v1)")
    return None


def _ask_drilldown(console: Console, prompt_text: str, segmenter_name: str) -> tuple[bool, int]:
    if segmenter_name in {"sentences", "tools"}:
        return False, 0
    structured = bool(re.search(r"(?m)^#{1,6}\s+", prompt_text)) or "\n\n" in prompt_text
    if structured:
        console.print(
            "[dim]Drill-down attributes coarse sections first, then re-attributes "
            "only the hottest ones sentence by sentence — far fewer provider calls "
            "than masking every sentence of a long prompt.[/dim]"
        )
    drilldown = Confirm.ask("Use coarse-to-fine drill-down?", default=structured)
    if not drilldown:
        return False, 0
    top_k = IntPrompt.ask("How many top sections to refine?", default=2)
    return True, max(0, top_k)


def _build_scorer(scorer_name: str) -> tuple[Scorer, dict[str, Any] | None]:
    """Build the chosen scorer, collecting its config interactively when needed."""
    if scorer_name == "tool-call":
        expected_tool = Prompt.ask("Expected tool name")
        raw_args = Prompt.ask("Required argument names (comma-separated)", default="none")
        required = (
            [item.strip() for item in raw_args.split(",") if item.strip()]
            if raw_args != "none"
            else []
        )
        config: dict[str, Any] = {"expected_tool": expected_tool, "required_args": required}
        return ToolAccuracyScorer(expected_tool=expected_tool, required_args=required), config
    return build_scorer(scorer_name), None


def _ask_synopsis(provider: str) -> tuple[bool, str, str | None]:
    if not Confirm.ask(
        "Generate an LLM synopsis of the evidence (one extra call)?", default=False
    ):
        return False, provider, None
    synopsis_provider = Prompt.ask(
        "Synopsis provider (ollama keeps this step free and local)",
        choices=[name for name, _ in _PROVIDERS],
        default=provider,
    )
    synopsis_model = Prompt.ask("Synopsis model id", default="auto")
    return True, synopsis_provider, None if synopsis_model == "auto" else synopsis_model


def _print_equivalent_command(console: Console, **config: Any) -> None:
    parts = ["promptlens explain", "--prompt <your-prompt>"]
    parts.append(f"--provider {config['provider']}")
    if config["model"]:
        parts.append(f"--model {config['model']}")
    if config["base_url"]:
        parts.append(f"--base-url {config['base_url']}")
    parts.append(f"--segmenter {config['segmenter']}")
    parts.append(f"--scorer {config['scorer']}")
    if config["scorer_config"]:
        parts.append("--scorer-config scorer.json")
    parts.append(f"--masker {config['masker']}")
    parts.append(f"--scale {config['scale']}")
    if config["drilldown"]:
        parts.append(f"--drilldown --drilldown-top {config['drilldown_top']}")
    if config["tools_path"]:
        parts.append(f"--tools {config['tools_path']}")
    if config["synopsis"]:
        parts.append(f"--synopsis --synopsis-provider {config['synopsis_provider']}")
        if config["synopsis_model"]:
            parts.append(f"--synopsis-model {config['synopsis_model']}")
    command = " \\\n  ".join(parts)
    body = command
    if config["scorer_config"]:
        body += "\n\n# scorer.json\n" + json.dumps(config["scorer_config"], indent=2)
    console.print(
        Panel(body, title="Run this again without the wizard", border_style="cyan")
    )
