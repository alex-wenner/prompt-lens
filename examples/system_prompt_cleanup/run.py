"""Separate the load-bearing instructions in a long system prompt from dead weight.

Scenario: a system prompt has grown to a wall of instructions and nobody knows
which lines still matter. promptlens segments it into sentences, masks each one,
and scores how much the model output drifts — so the inert "dead weight" lines
fall to ~0% while the lines that actually shape the output rise to the top.

This runs entirely offline with a deterministic simulated adapter whose output
depends only on two instructions:

* "Always respond in valid JSON." controls the output envelope.
* "Include a confidence score…" appends a confidence field.

Every other line is polite boilerplate that does not change the simulated
output, so it should attract almost no attribution. Swap in a real provider
adapter and a semantic scorer to run the same workflow against a live model.

Attribution is a lens, not an oracle: low attribution means "no measured effect
under this scorer", not "safe to delete". Confirm before trimming a prompt.
"""

from __future__ import annotations

from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.scorers import LengthDriftScorer
from promptlens.segmenters import SentenceSegmenter

JSON_SIGNAL = "valid json"
CONFIDENCE_SIGNAL = "confidence score"

SYSTEM_PROMPT = (
    "You are a friendly and helpful customer support assistant. "
    "Always respond in valid JSON. "
    "Be polite and empathetic with every customer. "
    "Include a confidence score between 0 and 1 for your answer. "
    "Never share internal company secrets. "
    "Remember that the customer is always the hero of their own story."
)


class SimulatedFormatter(Adapter):
    """Produce output whose shape depends only on the two load-bearing lines."""

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


def main() -> dict[str, Any]:
    """Run the demo and return the per-feature shares for inspection and tests."""
    harness = AttributionHarness(
        adapter=SimulatedFormatter(),
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
    print(
        "\nLens, not oracle: 'no measured effect' under a length/drift scorer is "
        "not proof a line is useless. Verify with a task metric before trimming."
    )
    return {"shares": shares, "load_bearing": load_bearing, "dead_weight": dead_weight}


if __name__ == "__main__":
    main()
