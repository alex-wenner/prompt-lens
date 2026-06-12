"""Separate the load-bearing instructions in a long system prompt from dead weight.

Scenario: a system prompt has grown to a wall of instructions and nobody knows
which lines still matter. promptlens segments it into sentences, masks each one,
and scores how much the model output drifts — so the inert "dead weight" lines
fall to ~0% while the lines that actually shape the output rise to the top.

This runs against a **real provider** (set ``OPENAI_API_KEY``, or pick another
provider via ``PROMPTLENS_EXAMPLE_PROVIDER``; see ``examples/_shared.py``).

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
from _shared import load_text, print_footer, require_adapter  # noqa: E402

# Single source of truth: the same file the README points the CLI at.
SYSTEM_PROMPT = load_text(__file__, "system_prompt.txt").strip()


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the per-feature shares for inspection and tests."""
    adapter = adapter if adapter is not None else require_adapter()
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )
    result = harness.explain(SYSTEM_PROMPT)
    ranked = result.ranked()

    print("Attribution over a long system prompt (drift: output length):\n")
    result.print()

    shares = {attribution.feature.text: share for attribution, share in ranked}
    load_bearing = [text for text, share in shares.items() if share > 0.01]
    dead_weight = [text for text, share in shares.items() if share <= 0.01]

    print("\nLoad-bearing lines (keep and tighten):")
    for text in load_bearing:
        print(f"  - {text}")
    print("\nDead-weight under this scorer (candidates to review, not auto-delete):")
    for text in dead_weight:
        print(f"  - {text}")
    print_footer(
        "'no measured effect' under a length/drift scorer is not proof a line is "
        "useless. Verify with a task metric before trimming."
    )
    return {"shares": shares, "load_bearing": load_bearing, "dead_weight": dead_weight}


if __name__ == "__main__":
    main()
