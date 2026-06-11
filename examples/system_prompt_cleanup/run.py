"""Separate the load-bearing instructions in a long system prompt from dead weight.

Scenario: a system prompt has grown to a wall of instructions and nobody knows
which lines still matter. promptlens segments it into sentences, masks each one,
and scores how much the model output drifts — so the inert "dead weight" lines
fall to ~0% while the lines that actually shape the output rise to the top.

This example makes **real provider calls** (export ``OPENAI_API_KEY``, or pick
another provider with ``PROMPTLENS_EXAMPLE_PROVIDER``). It first runs the real
baseline, shows the measured cost projection, then sweeps the prompt.

Attribution is a lens, not an oracle: low attribution means "no measured effect
under this scorer", not "safe to delete". Confirm before trimming a prompt.
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
from _shared import (  # noqa: E402
    console,
    get_adapter,
    load_text,
    print_completion,
    print_footer,
)

# Single source of truth: the same file the README points the CLI at.
SYSTEM_PROMPT = load_text(__file__, "system_prompt.txt").strip()


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the per-feature shares for inspection and tests."""
    adapter = adapter if adapter is not None else get_adapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    baseline, estimate = harness.estimate(SYSTEM_PROMPT)
    print_completion("Baseline output", baseline)
    estimate.print()

    console.print("\n[bold]Attribution over the system prompt (drift: output length)[/bold]\n")
    result = harness.explain(SYSTEM_PROMPT, baseline=baseline)
    result.print()
    ranked = result.ranked()

    shares = {attribution.feature.text: share for attribution, share in ranked}
    load_bearing = [text for text, share in shares.items() if share > 0.01]
    dead_weight = [text for text, share in shares.items() if share <= 0.01]

    console.print("\n[bold green]Load-bearing lines (keep and tighten):[/bold green]")
    for text in load_bearing:
        console.print(f"  - {text}")
    console.print(
        "\n[bold yellow]Dead-weight under this scorer "
        "(candidates to review, not auto-delete):[/bold yellow]"
    )
    for text in dead_weight:
        console.print(f"  - {text}")
    print_footer(
        "'no measured effect' under a length/drift scorer is not proof a line is "
        "useless. Verify with a task metric before trimming."
    )
    return {"shares": shares, "load_bearing": load_bearing, "dead_weight": dead_weight}


if __name__ == "__main__":
    main()
