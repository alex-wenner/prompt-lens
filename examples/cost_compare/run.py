"""Price an attribution run across providers from one real baseline call.

Attribution multiplies provider calls by feature count, so the same experiment
can cost very different amounts depending on the model — and running it on a
local model costs nothing. This example runs the **baseline completion once**
on a real provider, reads the provider's own metered token usage off that
response, and projects the sweep cost across a frontier model, a mid-tier
model, a cheap model, and a free local model. One call, real numbers, no
tokenizer guesswork — the same flow the CLI's cost gate uses before every run.

Config permutations this example pins: ``estimate_from_baseline`` with
``compare_models`` (one metered baseline, many price tables), and how the
**perturbation scale** multiplies the bill — ``quick`` versus ``full`` change
the evaluation count, and the projection tracks it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter
from promptlens.core.result import CostEstimate
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer, require_adapter  # noqa: E402

PROMPT = load_text(__file__, "prompt.md").strip()

# Comparison models, spanning a frontier model down to a free local one. All
# are keys in promptlens' built-in pricing table.
COMPARE_MODELS = [
    "anthropic/claude-opus-4-8",
    "openai/gpt-5.4",
    "anthropic/claude-haiku-4-5",
    "ollama/llama3.2",
]


def _project(adapter: Adapter, baseline: Any, scale: str) -> CostEstimate:
    """Project the sweep cost at one perturbation scale from the shared baseline."""
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        perturbation_scale=scale,
    )
    return harness.estimate_from_baseline(PROMPT, baseline, compare_models=COMPARE_MODELS)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run one baseline call, then project the cost at two perturbation scales."""
    adapter = adapter if adapter is not None else require_adapter()
    baseline = adapter.complete(PROMPT)  # the only provider call this example makes

    quick = _project(adapter, baseline, "quick")
    full = _project(adapter, baseline, "full")

    print("Projected attribution cost from one real baseline call:\n")
    quick.print()
    print(
        f"\nAt 'quick' scale that is {quick.evaluations} evaluations; 'full' scale "
        f"runs {full.evaluations} and projects to ${full.total_usd:.4f} on {quick.model}."
    )
    print(
        f"The same run priced on a local model (ollama/llama3.2) is "
        f"${quick.comparisons['ollama/llama3.2']:.2f} — the cost case for local "
        f"inference in one line."
    )
    print_footer(
        "projections use the provider's real metered usage for this baseline and "
        "promptlens' built-in pricing table; check live provider pricing before "
        "budgeting."
    )
    return {
        "model": quick.model,
        "quick_total_usd": quick.total_usd,
        "full_total_usd": full.total_usd,
        "comparisons": quick.comparisons,
        "quick_evaluations": quick.evaluations,
        "full_evaluations": full.evaluations,
        "baseline_input_tokens": quick.baseline_input_tokens,
    }


if __name__ == "__main__":
    main()
