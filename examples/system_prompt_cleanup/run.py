"""Separate the load-bearing instructions in a long system prompt from dead weight.

Scenario: a system prompt has grown to a wall of instructions and nobody knows
which lines still matter. promptlens segments it into sentences, masks each one,
and scores how much the model output drifts — so the inert "dead weight" lines
fall to ~0% while the lines that actually shape the output rise to the top.

By default this runs against a **real provider** (set ``OPENAI_API_KEY`` or
``ANTHROPIC_API_KEY``; see ``examples/_shared.py`` for the env vars). When
no credential is available it falls back to a deterministic offline adapter whose
output shape depends only on two instructions:

* "Always respond in valid JSON." controls the output envelope.
* "Include a confidence score…" appends a confidence field.

so the example still runs end-to-end and doubles as a CI smoke test.

Attribution is a lens, not an oracle: low attribution means "no measured effect
under this scorer", not "safe to delete". Confirm before trimming a prompt.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer, select_adapter  # noqa: E402

JSON_SIGNAL = "valid json"
CONFIDENCE_SIGNAL = "confidence score"

# Single source of truth: the same file the README points the CLI at.
SYSTEM_PROMPT = load_text(__file__, "system_prompt.txt").strip()


class SimulatedFormatter(Adapter):
    """Offline fallback: output whose shape depends only on the two load-bearing lines."""

    def __init__(self) -> None:
        self.model = "simulated-formatter"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        answer = "Order shipped."
        if JSON_SIGNAL in lowered:
            text = '{"status": "Order shipped.", "format": "json-mode-enabled"}'
        else:
            text = answer
        if CONFIDENCE_SIGNAL in lowered:
            text = f"{text} [confidence=0.90 high-certainty]"
        return CompletionOutput(text=text)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the per-feature shares for inspection and tests."""
    adapter = adapter if adapter is not None else select_adapter(SimulatedFormatter())
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
