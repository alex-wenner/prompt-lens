"""Price an attribution run across providers from one measured baseline call.

Attribution multiplies provider calls by feature count, so the same experiment
can cost very different amounts depending on the model — and running it on a
local model costs nothing. promptlens never guesses token counts: this example
runs the **baseline completion once for real**, reads the provider-reported
input/output usage, and projects the sweep cost across a frontier model, a
mid-tier model, a cheap model, and a free local model.

This example makes **one real provider call** (export ``OPENAI_API_KEY``, or
pick another provider with ``PROMPTLENS_EXAMPLE_PROVIDER``). Config
permutations it pins: ``estimate`` with ``compare_models`` (one measured
baseline, many price tables) and how the **perturbation scale** multiplies the
bill — ``quick`` versus ``full`` change the evaluation count, and the estimate
tracks it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import console, get_adapter, load_text, print_completion, print_footer  # noqa: E402

PROMPT = load_text(__file__, "prompt.md").strip()

# Comparison models, spanning a frontier model down to a free local one. All
# must be keys in promptlens' built-in pricing table.
COMPARE_MODELS = [
    "anthropic/claude-opus-4-8",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4-5",
    "ollama/llama3.2",
]


def _harness(adapter: Adapter, scale: str) -> AttributionHarness:
    return AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        perturbation_scale=scale,
    )


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run one real baseline and project the sweep cost across providers."""
    adapter = adapter if adapter is not None else get_adapter()

    # One real call; the 'full' estimate reuses the same measured baseline.
    baseline, quick = _harness(adapter, "quick").estimate(
        PROMPT, compare_models=COMPARE_MODELS
    )
    _, full = _harness(adapter, "full").estimate(
        PROMPT, compare_models=COMPARE_MODELS, baseline=baseline
    )

    print_completion("Baseline output (the one real call)", baseline)
    console.print(
        "\n[bold]Projected attribution cost across providers "
        "(from the measured baseline):[/bold]\n"
    )
    quick.print()
    console.print(
        f"\nAt 'quick' scale that is {quick.evaluations} evaluations; 'full' scale "
        f"runs {full.evaluations} and costs [bold green]${full.total_usd:.6f}[/bold green] "
        f"on {adapter.model}."
    )
    console.print(
        f"The same run on a local model (ollama/llama3.2) is "
        f"[bold green]${quick.comparisons['ollama/llama3.2']:.2f}[/bold green] — the cost "
        "case for local inference in one line."
    )
    print_footer(
        "projections multiply the measured baseline usage by the call count and "
        "use built-in pricing; check live provider pricing before budgeting."
    )
    return {
        "model": adapter.model,
        "quick_total_usd": quick.total_usd,
        "full_total_usd": full.total_usd,
        "comparisons": quick.comparisons,
        "quick_evaluations": quick.evaluations,
        "full_evaluations": full.evaluations,
    }


if __name__ == "__main__":
    main()
