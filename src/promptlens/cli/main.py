"""promptlens command line interface."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Annotated

import typer

from promptlens import (
    Adapter,
    AttributionHarness,
    AttributionResult,
    DrilldownResult,
    Segmenter,
    explain_drilldown,
)
from promptlens.adapters import EchoAdapter
from promptlens.cli.banner import print_banner
from promptlens.cli.factories import build_adapter, build_masker, build_sampler, build_scorer
from promptlens.core.base import ToolDefinitions
from promptlens.core.pricing import MODEL_PRICING_USD_PER_MTOK
from promptlens.mutators import LLMRewriteMutator
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.reporters import LLMSynopsisWriter
from promptlens.scorers import LengthDriftScorer
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


def _offline_harness(
    model: str, segmenter_name: str, scale: str | int = "quick", prompt_text: str = ""
) -> AttributionHarness:
    return AttributionHarness(
        adapter=EchoAdapter(model=model),
        segmenter=_segmenter(segmenter_name, prompt_text),
        scorer=LengthDriftScorer(),
        perturbation_scale=_parse_scale(scale),
    )


def _cost_gate(
    harness: AttributionHarness,
    prompt_text: str,
    tools: ToolDefinitions | None,
    *,
    dry_run: bool,
    confirm: bool,
    exact_tokens: bool = False,
) -> None:
    """Print the cost estimate and stop (dry run) or ask before inference calls."""
    if not dry_run and not confirm:
        return
    harness.estimate(prompt_text, tools=tools, exact_tokens=exact_tokens).print()
    if dry_run:
        raise typer.Exit()
    typer.confirm("Run attribution with these provider calls?", abort=True)


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
    model: Annotated[str, typer.Option(help="Provider/model name.")] = "openai/gpt-4o-mini",
    segmenter: Annotated[
        str, typer.Option(help="auto, sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    compare: Annotated[
        str | None, typer.Option(help="Comma-separated model names to compare.")
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
) -> None:
    """Preview attribution cost without provider calls."""
    prompt_text = _read_prompt(prompt)
    harness = _offline_harness(
        model=model, segmenter_name=segmenter, scale=scale, prompt_text=prompt_text
    )
    compare_models = [item.strip() for item in compare.split(",")] if compare else None
    harness.estimate(prompt_text, tools=_read_tools(tools), compare_models=compare_models).print()


@app.command()
def explain(
    prompt: Annotated[str, typer.Option(help="Prompt text or path to a prompt file.")],
    output: Annotated[str | None, typer.Option(help="Optional JSON output path.")] = None,
    provider: Annotated[
        str,
        typer.Option(
            help=(
                "Provider type: echo, openai, anthropic, bedrock, copilot, grok, "
                "gemini, ollama (local), or openai-compatible (with --base-url)."
            )
        ),
    ] = "echo",
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
                "length, embedding, embedding-local, logprob, tool-call, "
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
        typer.Option("--dry-run", help="Print the cost estimate and exit without provider calls."),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm", help="Show the cost estimate and ask before making provider calls."
        ),
    ] = False,
    exact_tokens: Annotated[
        bool,
        typer.Option(
            "--exact-tokens",
            help=(
                "Count estimate tokens with the provider's free count_tokens endpoint "
                "when the adapter supports it (anthropic). Needs credentials and network "
                "but runs no inference; other providers fall back to local counting."
            ),
        ),
    ] = False,
) -> None:
    """Run attribution using the SDK pipeline."""
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
    _cost_gate(
        harness,
        prompt_text,
        tool_definitions,
        dry_run=dry_run,
        confirm=confirm,
        exact_tokens=exact_tokens,
    )
    result: AttributionResult | DrilldownResult
    if drilldown:
        drilldown_result = explain_drilldown(
            harness, prompt_text, tools=tool_definitions, top_k=drilldown_top
        )
        result = drilldown_result
        synopsis_target = drilldown_result.overview
    else:
        result = harness.explain(prompt_text, tools=tool_definitions)
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
                "Provider type: echo, openai, anthropic, bedrock, copilot, grok, "
                "gemini, ollama (local), or openai-compatible (with --base-url)."
            )
        ),
    ] = "echo",
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
                "length, embedding, embedding-local, logprob, tool-call, "
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
        typer.Option("--dry-run", help="Print the cost estimate and exit without provider calls."),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option(
            "--confirm",
            help=(
                "Show the cost estimate and ask before making provider calls. "
                "The final rewrite adds one call on top of the estimate."
            ),
        ),
    ] = False,
    exact_tokens: Annotated[
        bool,
        typer.Option(
            "--exact-tokens",
            help=(
                "Count estimate tokens with the provider's free count_tokens endpoint "
                "when the adapter supports it (anthropic). Needs credentials and network "
                "but runs no inference; other providers fall back to local counting."
            ),
        ),
    ] = False,
) -> None:
    """Run attribution, then propose an attribution-informed prompt rewrite."""
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
    _cost_gate(
        harness,
        prompt_text,
        tool_definitions,
        dry_run=dry_run,
        confirm=confirm,
        exact_tokens=exact_tokens,
    )
    result = harness.optimize(prompt_text, tools=tool_definitions)
    if output:
        Path(output).write_text(result.to_json(), encoding="utf-8")
    result.print()


@app.command("models")
def list_models() -> None:
    """List built-in pricing entries."""
    for model, (input_rate, output_rate) in sorted(MODEL_PRICING_USD_PER_MTOK.items()):
        typer.echo(f"{model}\tinput=${input_rate}/MTok\toutput=${output_rate}/MTok")


# Registered late so the wizard can lazily import this module's helpers without
# a circular import at load time.
from promptlens.cli.wizard import run_wizard  # noqa: E402

app.command("wizard")(run_wizard)
