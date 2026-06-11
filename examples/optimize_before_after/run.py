"""Turn attribution evidence into a concrete before/after prompt rewrite.

Scenario: you have a verbose prompt and want a tighter version that keeps the
load-bearing instruction and prunes filler. ``AttributionHarness.optimize`` runs
a leave-one-out attribution sweep, then hands that evidence to an
``LLMPromptOptimizer`` which proposes a whole-prompt rewrite for review.

This example makes **real provider calls** (export ``OPENAI_API_KEY``, or pick
another provider with ``PROMPTLENS_EXAMPLE_PROVIDER``): the same model both
answers the attribution sweep and proposes the rewrite.

The proposed rewrite is returned for review and is **never** adopted
automatically.

Attribution is a lens, not an oracle: a proposed rewrite is a candidate, not a
verified improvement. Re-run attribution and task checks before shipping it.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter
from promptlens.optimizers import LLMPromptOptimizer
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import console, get_adapter, print_completion, print_footer  # noqa: E402

ORIGINAL_PROMPT = (
    "You are an extremely helpful, friendly, and knowledgeable assistant. "
    "Summarize the input text in exactly three bullet points. "
    "Feel free to be as detailed and thorough as you possibly can. "
    "Thanks so much for your hard work on this important task."
)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the rewrite for inspection and tests."""
    adapter = adapter if adapter is not None else get_adapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
        optimizer=LLMPromptOptimizer(adapter),
    )

    baseline, estimate = harness.estimate(ORIGINAL_PROMPT)
    print_completion("Baseline output", baseline)
    estimate.print()

    console.print("\n[bold]Attribution-informed prompt rewrite[/bold]\n")
    result = harness.optimize(ORIGINAL_PROMPT, baseline=baseline)
    result.print()
    print_footer(
        "the proposed rewrite is a candidate, not a verified improvement. Re-run "
        "attribution and a task metric before adopting it."
    )
    return {
        "original_prompt": result.original_prompt,
        "proposed_prompt": result.proposed_prompt,
        "rationale": result.rationale,
        "top_feature": result.metadata.get("top_feature"),
    }


if __name__ == "__main__":
    main()
