"""promptlens command line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from promptlens import AttributionHarness, Segmenter
from promptlens.adapters import EchoAdapter
from promptlens.cli.factories import build_adapter, build_sampler, build_scorer
from promptlens.core.pricing import MODEL_PRICING_USD_PER_MTOK
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import (
    MarkdownSectionSegmenter,
    ParagraphSegmenter,
    SentenceSegmenter,
    ToolSegmenter,
)

app = typer.Typer(help="Black-box prompt attribution for LLM prompts.")

_SCALE_HELP = "Perturbation scale: quick, standard, full, or an integer repeat count."


def _read_prompt(prompt: str) -> str:
    path = Path(prompt)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return prompt


def _read_tools(path: str | None) -> list[dict[str, object]] | None:
    if path is None:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = "Tools file must contain a JSON list"
        raise typer.BadParameter(msg)
    return [dict(item) for item in data]


def _segmenter(name: str) -> Segmenter:
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


def _offline_harness(
    model: str, segmenter_name: str, scale: str | int = "quick"
) -> AttributionHarness:
    return AttributionHarness(
        adapter=EchoAdapter(model=model),
        segmenter=_segmenter(segmenter_name),
        scorer=LengthDriftScorer(),
        perturbation_scale=_parse_scale(scale),
    )


def _harness(
    *,
    provider: str,
    model: str | None,
    segmenter_name: str,
    scale: str | int,
    temperature: float,
    base_url: str | None,
    sampler_name: str,
    scorer_name: str,
    scorer_config: str | None,
) -> AttributionHarness:
    try:
        adapter = build_adapter(
            provider=provider,
            model=model,
            temperature=temperature,
            base_url=base_url,
        )
        sampler = build_sampler(sampler_name, scale=scale)
        scorer = build_scorer(scorer_name, config_path=scorer_config)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    return AttributionHarness(
        adapter=adapter,
        segmenter=_segmenter(segmenter_name),
        scorer=scorer,
        sampler=sampler,
        perturbation_scale=_parse_scale(scale),
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
        str, typer.Option(help="sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    compare: Annotated[
        str | None, typer.Option(help="Comma-separated model names to compare.")
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
) -> None:
    """Preview attribution cost without provider calls."""
    prompt_text = _read_prompt(prompt)
    harness = _offline_harness(model=model, segmenter_name=segmenter, scale=scale)
    compare_models = [item.strip() for item in compare.split(",")] if compare else None
    harness.estimate(prompt_text, tools=_read_tools(tools), compare_models=compare_models).print()


@app.command()
def explain(
    prompt: Annotated[str, typer.Option(help="Prompt text or path to a prompt file.")],
    output: Annotated[str | None, typer.Option(help="Optional JSON output path.")] = None,
    provider: Annotated[
        str,
        typer.Option(help="echo, openai, anthropic, bedrock, or openai-compatible."),
    ] = "echo",
    model: Annotated[
        str | None,
        typer.Option(help="Model id. Defaults to provider-specific environment/default model."),
    ] = None,
    temperature: Annotated[float, typer.Option(help="Provider sampling temperature.")] = 0.0,
    base_url: Annotated[
        str | None,
        typer.Option(help="Base URL for openai-compatible providers."),
    ] = None,
    segmenter: Annotated[
        str, typer.Option(help="sentences, paragraphs, sections, or tools.")
    ] = "sentences",
    tools: Annotated[str | None, typer.Option(help="Optional JSON tool schema file.")] = None,
    sampler: Annotated[str, typer.Option(help="leave-one-out.")] = "leave-one-out",
    scorer: Annotated[
        str,
        typer.Option(help="length, embedding, logprob, or tool-call."),
    ] = "length",
    scorer_config: Annotated[
        str | None,
        typer.Option(help="Optional JSON scorer config path."),
    ] = None,
    scale: Annotated[str, typer.Option(help=_SCALE_HELP)] = "quick",
) -> None:
    """Run attribution using the SDK pipeline."""
    prompt_text = _read_prompt(prompt)
    result = _harness(
        provider=provider,
        model=model,
        segmenter_name=segmenter,
        scale=scale,
        temperature=temperature,
        base_url=base_url,
        sampler_name=sampler,
        scorer_name=scorer,
        scorer_config=scorer_config,
    ).explain(prompt_text, tools=_read_tools(tools))
    if output:
        Path(output).write_text(result.to_json(), encoding="utf-8")
    result.print()


@app.command("models")
def list_models() -> None:
    """List built-in pricing entries."""
    for model, (input_rate, output_rate) in sorted(MODEL_PRICING_USD_PER_MTOK.items()):
        typer.echo(f"{model}\tinput=${input_rate}/MTok\toutput=${output_rate}/MTok")
