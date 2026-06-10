"""Price an attribution run across providers before spending a cent.

Attribution multiplies provider calls by feature count, so the same experiment
can cost very different amounts depending on the model — and running it on a
local model costs nothing. This example estimates the spend for one prompt
across a frontier model, a mid-tier model, a cheap model, and a local model,
entirely offline (no provider calls, no credentials), using the same cost
estimator the CLI's ``--dry-run`` flag uses.

Config permutations this example pins: the ``estimate`` path with
``compare_models`` (one token count, many price tables), and how the
**perturbation scale** multiplies the bill — ``quick`` versus ``full`` change
the evaluation count, and the estimate tracks it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.adapters import EchoAdapter
from promptlens.core.base import Adapter
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer  # noqa: E402

PROMPT = load_text(__file__, "prompt.md").strip()

# Primary plus comparison models, spanning a frontier model down to a free
# local one. All must be keys in promptlens' built-in pricing table.
PRIMARY_MODEL = "anthropic/claude-opus-4-8"
COMPARE_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "anthropic/claude-haiku-4-5",
    "ollama/llama3.2",
]


def _estimate(scale: str) -> Any:
    """Estimate the attribution cost for the prompt at one perturbation scale.

    Uses the offline echo adapter: segmentation and masking are local, so the
    estimate is exact about token counts and call volume without any inference.
    """
    harness = AttributionHarness(
        adapter=EchoAdapter(model=PRIMARY_MODEL),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        perturbation_scale=scale,
    )
    return harness.estimate(PROMPT, compare_models=COMPARE_MODELS)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Estimate and print the per-provider cost at two perturbation scales."""
    del adapter  # this example is inherently offline; the parameter keeps the shared shape
    quick = _estimate("quick")
    full = _estimate("full")

    print("Estimated attribution cost across providers (no provider calls made):\n")
    quick.print()
    print(
        f"\nAt 'quick' scale that is {quick.evaluations} evaluations; 'full' scale "
        f"runs {full.evaluations} and costs ${full.total_usd:.6f} on {PRIMARY_MODEL}."
    )
    print(
        f"The same run on a local model (ollama/llama3.2) is "
        f"${quick.comparisons['ollama/llama3.2']:.2f} — the cost case for local "
        f"inference in one line."
    )
    print_footer(
        "estimates use built-in pricing and a conservative token count; check "
        "live provider pricing before budgeting, and use --exact-tokens on the "
        "CLI for an exact (still inference-free) Anthropic count."
    )
    return {
        "primary_model": PRIMARY_MODEL,
        "quick_total_usd": quick.total_usd,
        "full_total_usd": full.total_usd,
        "comparisons": quick.comparisons,
        "quick_evaluations": quick.evaluations,
        "full_evaluations": full.evaluations,
    }


if __name__ == "__main__":
    main()
