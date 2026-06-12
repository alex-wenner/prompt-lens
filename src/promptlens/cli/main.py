"""promptlens command line interface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.table import Table

from promptlens import (
    Adapter,
    AttributionHarness,
    AttributionResult,
    CompletionOutput,
    DrilldownResult,
    Segmenter,
    explain_drilldown,
)
from promptlens.cli.banner import print_banner
from promptlens.cli.factories import build_adapter, build_masker, build_sampler, build_scorer
from promptlens.cli.render import render_baseline, render_tools
from promptlens.core.base import ToolDefinitions
from promptlens.core.pricing import MODEL_PRICING_USD_PER_MTOK
from promptlens.mutators import LLMRewriteMutator
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.reporters import LLMSynopsisWriter
from promptlens.segmenters import (
    MarkdownSectionSegmenter,
    ParagraphSegmenter,
    SentenceSegmenter,
    ToolSegmenter,
)

app = typer.Typer(help="Black-box prompt attribution for LLM prompts.")

_SCALE_HELP = "Perturbation scale: quick, standard, full, or an integer repeat count."


@app.callback(invoke_without_command=True)
def _entry(ctx: typer.Context) -> None:
    """Black-box prompt attribution for LLM prompts."""
    if ctx.invoked_subcommand is None:
        print_banner()
        typer.echo("Try `promptlens wizard` for a guided run, or `promptlens --help`.")


def _read_prompt(prompt: str) -> str:
    path = Path(prompt)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return prompt


def _read_tools(path: str | None) -> ToolDefinitions | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = "Tools file must contain a JSON list"
        raise typer.BadParameter(msg)
    tools: ToolDefinitions = [dict(item) for item in data]
    return tools


def _segmenter(name: str, prompt: str = "") -> Segmenter:
    if name == "auto":
        return _auto_segmenter(prompt)
    if name == "sentences":
        return SentenceSegmenter()
    if name == "paragraphs":
        return ParagraphSegmenter()
    if name == "sections":
        return MarkdownSectionSegmenter()
    if name == "tools":
        return ToolSegmenter(granularity="parameter")
    msg = f"Unsupported segmenter: {name}"
    raise typer.BadParameter(msg)


def _auto_segmenter(prompt: str) -> Segmenter:
    """Pick a segmenter from the prompt's shape: headings, blank lines, sentences."""
    if re.search(r"(?m)^#{1,6}\s+", prompt):
        return MarkdownSectionSegmenter()
    if "\n\n" in prompt:
        return ParagraphSegmenter()
    return SentenceSegmenter()


def _baseline_gate(
    harness: AttributionHarness,
    prompt_text: str,
    tools: ToolDefinitions | None,
    *,
    dry_run: bool,
    yes: bool,
    console: Console | None = None,
    extra_calls_note: str | None = None,
    compare_models: list[str] | None = None,
) -> CompletionOutput:
    """Run the real baseline, project the sweep cost from its metered usage, and ask.

    This is the only honest way to estimate: one real call, the provider's own
    token counts, multiplied by the number of planned perturbations. The
    returned baseline is reused by the sweep so it is never paid for twice.
    """
    console = console or Console()
    with console.status("[cyan]Running the baseline completion…[/cyan]"):
        baseline = harness.run_baseline(prompt_text, tools=tools)
    render_baseline(baseline, console)
    estimate = harness.estimate_from_baseline(
        prompt_text, baseline, tools=tools, compare_models=compare_models
    )
    estimate.print(console)
    if extra_calls_note:
        console.print(f"[dim]{extra_calls_note}[/dim]")
    if dry_run:
        console.print("[yellow]Dry run: stopping after the baseline call.[/yellow]")
        raise typer.Exit()
    if not yes and not Confirm.ask(
        f"\n[bold]Run the remaining {estimate.evaluations} provider calls?[/bold]",
        default=True,
        console=console,
    ):
        console.print("[yellow]Stopped at the cost gate; only the baseline ran.[/yellow]")
        raise typer.Exit(1)
    return baseline


def _harness(
    *,
    provider: str,
    model: str | None,
    segmenter_name: str,
    prompt_text: str,
    scale: str | int,
    temperature: float,
    base_url: str | None,
    sampler_name: str,
    scorer_name: str,
    scorer_config: str | None,
    supplementary_rewrites: int,
    masker_name: str = "placeholder",
    samples_per_coalition: int = 1,
    use_batch_api: bool = False,
    with_optimizer: bool = False,
) -> AttributionHarness:
    try:
        adapter = build_adapter(
            provider=provider,
            model=model,
            temperature=temperature,
            base_url=base_url,
            use_batch_api=use_batch_api,
        )
        sampler = build_sampler(sampler_name, scale=scale)
        scorer = build_scorer(scorer_name, config_path=scorer_config)
        masker = build_masker(masker_name)
        if supplementary_rewrites < 0:
            msg = "The --supplementary-rewrites value must be non-negative"
            raise ValueError(msg)
        if samples_per_coalition < 1:
            msg = "The --samples-per-coalition value must be a positive integer"
            raise ValueError(msg)
        supplementary_mutator = (
            LLMRewriteMutator(adapter, rewrites_per_feature=supplementary_rewrites)
            if supplementary_rewrites
            else None
        )
        optimizer = LLMPromptOptimizer(adapter) if with_optimizer else None
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return AttributionHarness(
        adapter=adapter,
        segmenter=_segmenter(segmenter_name, prompt_text),
        scorer=scorer,
        sampler=sampler,
        masker=masker,
        supplementary_mutator=supplementary_mutator,
        optimizer=optimizer,
        perturbation_scale=_parse_scale(scale),
        samples_per_coalition=samples_per_coalition,
    )


def _parse_scale(scale: str | int) -> str | int:
    if isinstance(scale, str) and scale.isdigit():
        return int(scale)
    return scale


@app.command()
def estimate(
    prompt: Annotated[str, typer.Option(help="Prompt text or path to a prompt file.")],
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider type: openai, anthropic, bedrock, copilot, grok, "
                "gemini, ollama (local), or openai-compatible (with --base-url)."
            )
        ),
    ] = "openai",
    model: Annotated[
        str | None,
        typer.Option(help="Model id. Defaults to provider-specific environment/default model."),
    ] = None,
    temperature: Annotated[float, typer.Option(help="Provider sampling temperature.")] = 0.0,
    base_url: Annotated[
        str | None,
        typer.Option(help="Base URL for the ollama or openai-compatible providers."),
    ] = None,
    segmenter: Annotated[
        str, typer.Option(help="auto, sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    compare: Annotated[
        str | None, typer.Option(help="Comma-separated model names to compare.")
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
) -> None:
    """Run the baseline call for real and project the full sweep cost from its usage.

    Costs exactly one provider call. The provider's own metered token counts for
    that call are multiplied across the planned perturbations — promptlens never
    estimates with a local tokenizer or heuristic.
    """
    console = Console()
    prompt_text = _read_prompt(prompt)
    tool_definitions = _read_tools(tools)
    harness = _harness(
        provider=provider,
        model=model,
        segmenter_name=segmenter,
        prompt_text=prompt_text,
        scale=scale,
        temperature=temperature,
        base_url=base_url,
        sampler_name="leave-one-out",
        scorer_name="length",
        scorer_config=None,
        supplementary_rewrites=0,
    )
    compare_models = [item.strip() for item in compare.split(",")] if compare else None
    render_tools(tool_definitions, console)
    with console.status("[cyan]Running the baseline completion…[/cyan]"):
        baseline = harness.run_baseline(prompt_text, tools=tool_definitions)
    render_baseline(baseline, console)
    harness.estimate_from_baseline(
        prompt_text, baseline, tools=tool_definitions, compare_models=compare_models
    ).print(console)


@app.command()
def explain(
    prompt: Annotated[str, typer.Option(help="Prompt text or path to a prompt file.")],
    output: Annotated[str | None, typer.Option(help="Optional JSON output path.")] = None,
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider type: openai, anthropic, bedrock, copilot, grok, "
                "gemini, ollama (local), openai-compatible (with --base-url), "
                "or echo (offline smoke runs)."
            )
        ),
    ] = "openai",
    model: Annotated[
        str | None,
        typer.Option(help="Model id. Defaults to provider-specific environment/default model."),
    ] = None,
    temperature: Annotated[float, typer.Option(help="Provider sampling temperature.")] = 0.0,
    base_url: Annotated[
        str | None,
        typer.Option(help="Base URL for the ollama or openai-compatible providers."),
    ] = None,
    segmenter: Annotated[
        str, typer.Option(help="auto, sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    sampler: Annotated[str, typer.Option(help="leave-one-out or random.")] = "leave-one-out",
    masker: Annotated[
        str,
        typer.Option(help="Masking strategy: placeholder, drop, or filler."),
    ] = "placeholder",
    scorer: Annotated[
        str,
        typer.Option(
            help=(
                "length, embedding (semantic; huggingface/openai via config), "
                "embedding-local (local huggingface), logprob, tool-call, "
                "tool-sequence, or tool-args."
            )
        ),
    ] = "length",
    scorer_config: Annotated[
        str | None,
        typer.Option(help="Optional JSON scorer config path."),
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
    drilldown: Annotated[
        bool,
        typer.Option(
            "--drilldown",
            help=(
                "Coarse-to-fine attribution: attribute sections first, then refine "
                "only the top sections sentence by sentence — far fewer provider "
                "calls than a flat sentence sweep on long prompts. Uses the auto "
                "segmenter for the coarse pass unless --segmenter overrides it. "
                "The cost estimate covers the overview pass; each refined section "
                "adds roughly one sentence sweep."
            ),
        ),
    ] = False,
    drilldown_top: Annotated[
        int,
        typer.Option(help="How many top coarse features to refine with --drilldown."),
    ] = 2,
    samples_per_coalition: Annotated[
        int,
        typer.Option(
            help="Evaluations per coalition for distributional attribution at temperature > 0."
        ),
    ] = 1,
    batch_api: Annotated[
        bool,
        typer.Option(
            help="Use the provider native batch API (openai, anthropic) for cheaper async runs."
        ),
    ] = False,
    supplementary_rewrites: Annotated[
        int,
        typer.Option(
            help="Optional LLM prompt rewrites per feature to evaluate as supplementary analysis."
        ),
    ] = 0,
    synopsis: Annotated[
        bool,
        typer.Option(
            "--synopsis",
            help=(
                "After attribution, make one extra LLM call that turns the full "
                "evidence into a plain-language synopsis."
            ),
        ),
    ] = False,
    synopsis_provider: Annotated[
        str | None,
        typer.Option(
            help=(
                "Provider for the synopsis call; defaults to the run provider. "
                "Point at ollama or openai-compatible to summarize on a local model."
            )
        ),
    ] = None,
    synopsis_model: Annotated[
        str | None,
        typer.Option(help="Model id for the synopsis call."),
    ] = None,
    synopsis_base_url: Annotated[
        str | None,
        typer.Option(help="Base URL for an ollama/openai-compatible synopsis provider."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run only the baseline call, print the projected sweep cost, and exit."
            ),
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-gate confirmation and run the sweep immediately.",
        ),
    ] = False,
) -> None:
    """Run attribution using the SDK pipeline.

    The baseline always runs first; its real metered usage projects the sweep
    cost, which is shown before the remaining provider calls are made.
    """
    console = Console()
    prompt_text = _read_prompt(prompt)
    tool_definitions = _read_tools(tools)
    if drilldown and segmenter == "sentences":
        # A sentence-grained coarse pass leaves nothing to refine; drill-down
        # wants sections or paragraphs first, which "auto" picks from the shape.
        segmenter = "auto"
    harness = _harness(
        provider=provider,
        model=model,
        segmenter_name=segmenter,
        prompt_text=prompt_text,
        scale=scale,
        temperature=temperature,
        base_url=base_url,
        sampler_name=sampler,
        scorer_name=scorer,
        scorer_config=scorer_config,
        supplementary_rewrites=supplementary_rewrites,
        masker_name=masker,
        samples_per_coalition=samples_per_coalition,
        use_batch_api=batch_api,
    )
    render_tools(tool_definitions, console)
    baseline = _baseline_gate(
        harness,
        prompt_text,
        tool_definitions,
        dry_run=dry_run,
        yes=yes,
        console=console,
        extra_calls_note=(
            "The projection covers the overview pass; each refined section adds "
            "roughly one sentence sweep."
            if drilldown
            else None
        ),
    )
    result: AttributionResult | DrilldownResult
    if drilldown:
        drilldown_result = explain_drilldown(
            harness,
            prompt_text,
            tools=tool_definitions,
            top_k=drilldown_top,
            baseline=baseline,
        )
        result = drilldown_result
        synopsis_target = drilldown_result.overview
    else:
        result = harness.explain(prompt_text, tools=tool_definitions, baseline=baseline)
        synopsis_target = result
    if synopsis:
        writer = LLMSynopsisWriter(
            _synopsis_adapter(
                run_provider=provider,
                run_model=model,
                run_base_url=base_url,
                synopsis_provider=synopsis_provider,
                synopsis_model=synopsis_model,
                synopsis_base_url=synopsis_base_url,
                temperature=temperature,
            )
        )
        enriched = synopsis_target.with_synopsis(
            writer.summarize(prompt_text, synopsis_target)
        )
        result = result.model_copy(update={"overview": enriched}) if drilldown else enriched
    if output:
        Path(output).write_text(result.to_json(), encoding="utf-8")
    result.print()


def _synopsis_adapter(
    *,
    run_provider: str,
    run_model: str | None,
    run_base_url: str | None,
    synopsis_provider: str | None,
    synopsis_model: str | None,
    synopsis_base_url: str | None,
    temperature: float,
) -> Adapter:
    """Build the adapter for the synopsis call.

    With no synopsis overrides this reuses the attribution run's provider
    settings. Overriding the provider drops back to that provider's own
    defaults for model and base URL unless they are overridden too, so
    ``--synopsis-provider ollama`` alone summarizes on the local default model.
    """
    provider = synopsis_provider or run_provider
    same_provider = provider == run_provider
    try:
        return build_adapter(
            provider=provider,
            model=synopsis_model or (run_model if same_provider else None),
            temperature=temperature,
            base_url=synopsis_base_url or (run_base_url if same_provider else None),
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def optimize(
    prompt: Annotated[str, typer.Option(help="Prompt text or path to a prompt file.")],
    output: Annotated[str | None, typer.Option(help="Optional JSON output path.")] = None,
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider type: openai, anthropic, bedrock, copilot, grok, "
                "gemini, ollama (local), openai-compatible (with --base-url), "
                "or echo (offline smoke runs)."
            )
        ),
    ] = "openai",
    model: Annotated[
        str | None,
        typer.Option(help="Model id. Defaults to provider-specific environment/default model."),
    ] = None,
    temperature: Annotated[float, typer.Option(help="Provider sampling temperature.")] = 0.0,
    base_url: Annotated[
        str | None,
        typer.Option(help="Base URL for the ollama or openai-compatible providers."),
    ] = None,
    segmenter: Annotated[
        str, typer.Option(help="auto, sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    sampler: Annotated[str, typer.Option(help="leave-one-out or random.")] = "leave-one-out",
    scorer: Annotated[
        str,
        typer.Option(
            help=(
                "length, embedding (semantic; huggingface/openai via config), "
                "embedding-local (local huggingface), logprob, tool-call, "
                "tool-sequence, or tool-args."
            )
        ),
    ] = "length",
    scorer_config: Annotated[
        str | None,
        typer.Option(help="Optional JSON scorer config path."),
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
    batch_api: Annotated[
        bool,
        typer.Option(
            help="Use the provider native batch API (openai, anthropic) for cheaper async runs."
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help=(
                "Run only the baseline call, print the projected sweep cost, and exit."
            ),
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip the cost-gate confirmation and run the sweep immediately.",
        ),
    ] = False,
) -> None:
    """Run attribution, then propose an attribution-informed prompt rewrite."""
    console = Console()
    prompt_text = _read_prompt(prompt)
    tool_definitions = _read_tools(tools)
    harness = _harness(
        provider=provider,
        model=model,
        segmenter_name=segmenter,
        prompt_text=prompt_text,
        scale=scale,
        temperature=temperature,
        base_url=base_url,
        sampler_name=sampler,
        scorer_name=scorer,
        scorer_config=scorer_config,
        supplementary_rewrites=0,
        use_batch_api=batch_api,
        with_optimizer=True,
    )
    render_tools(tool_definitions, console)
    baseline = _baseline_gate(
        harness,
        prompt_text,
        tool_definitions,
        dry_run=dry_run,
        yes=yes,
        console=console,
        extra_calls_note="The final rewrite adds one call on top of the projection.",
    )
    result = harness.optimize(prompt_text, tools=tool_definitions, baseline=baseline)
    if output:
        Path(output).write_text(result.to_json(), encoding="utf-8")
    result.print()


@app.command("models")
def list_models() -> None:
    """List built-in pricing entries."""
    console = Console()
    table = Table(title="Built-in pricing", title_style="bold")
    table.add_column("Model", style="cyan")
    table.add_column("Input $/MTok", justify="right")
    table.add_column("Output $/MTok", justify="right")
    for model, (input_rate, output_rate) in sorted(MODEL_PRICING_USD_PER_MTOK.items()):
        table.add_row(model, f"${input_rate:.2f}", f"${output_rate:.2f}")
    console.print(table)


# Registered late so the wizard can lazily import this module's helpers without
# a circular import at load time.
from promptlens.cli.wizard import run_wizard  # noqa: E402

app.command("wizard")(run_wizard)
