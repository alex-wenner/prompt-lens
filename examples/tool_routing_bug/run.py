"""Find the instruction that decides a routing call, then prove the fix.

Scenario: a support agent has two tools and must answer "where is my recent
purchase?" by calling ``lookup_order``. The routing rules — including the
description of the ``order_reference`` parameter — live in the agent's
instruction prompt. A simulated model routes correctly only when it can still
see the order-id hint. Corrupt that one sentence and the agent silently routes
to the wrong tool (``search_catalog``).

This runs entirely offline with a deterministic simulated adapter, so it needs
no API keys. The point is the *workflow*, not the toy model:

1. Run promptlens with the objective ``ToolAccuracyScorer`` over the healthy
   prompt. The order-id sentence ranks #1 because masking it collapses tool
   accuracy from 1.0 to 0.0, while masking dead-weight sentences changes
   nothing.
2. Confirm the diagnosis: swap in the misleading description and watch baseline
   accuracy drop to 0.0; restore it and accuracy returns to 1.0.

Attribution is a lens, not an oracle: it points at the load-bearing text so you
know *where* to look, and the before/after task metric is what proves the fix.
"""

from __future__ import annotations

from typing import Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.scorers import ToolAccuracyScorer
from promptlens.segmenters import SentenceSegmenter

# The simulated model routes to lookup_order only when this hint survives in the
# prompt. It appears in exactly one sentence: the order_reference description.
ROUTING_SIGNAL = "order id"

GOOD_DESCRIPTION = "The order_reference parameter is the customer's order ID, for example #1234."
MISLEADING_DESCRIPTION = "The order_reference parameter is a product search keyword."

_PROMPT_TEMPLATE = (
    "You are a support agent with two tools. "
    "Call lookup_order when the user asks about an existing purchase. "
    "{description} "
    "Call search_catalog only when the user wants to buy a new product. "
    "User message: 'Where is my recent purchase?'"
)

GOOD_PROMPT = _PROMPT_TEMPLATE.format(description=GOOD_DESCRIPTION)
MISLEADING_PROMPT = _PROMPT_TEMPLATE.format(description=MISLEADING_DESCRIPTION)


class SimulatedRoutingAgent(Adapter):
    """Route to ``lookup_order`` only when the order-id hint is visible.

    The decision is made from the prompt text the harness actually sends, so when
    promptlens masks the order-id sentence the hint disappears and routing flips,
    exactly as a real model would lose the cue.
    """

    def __init__(self) -> None:
        self.model = "simulated-routing-agent"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if ROUTING_SIGNAL in prompt.lower():
            tool_call: dict[str, Any] = {
                "name": "lookup_order",
                "arguments": {"order_reference": "#1234"},
            }
        else:
            tool_call = {"name": "search_catalog", "arguments": {"query": "recent purchase"}}
        return CompletionOutput(text="", tool_calls=[tool_call])


def _accuracy(prompt: str) -> float:
    scorer = ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"])
    output = SimulatedRoutingAgent().complete(prompt)
    return scorer.score(CompletionOutput(text=""), output)


def main() -> dict[str, Any]:
    """Run the demo and return the headline numbers for inspection and tests."""
    harness = AttributionHarness(
        adapter=SimulatedRoutingAgent(),
        segmenter=SentenceSegmenter(),
        scorer=ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"]),
    )
    result = harness.explain(GOOD_PROMPT)
    ranked = result.ranked()
    top_feature, top_share = ranked[0][0], ranked[0][1]

    before = _accuracy(MISLEADING_PROMPT)
    after = _accuracy(GOOD_PROMPT)

    print("Attribution over the healthy prompt (objective: tool accuracy):\n")
    result.print()
    print(
        f"\nMost load-bearing feature: {top_feature.feature.name} "
        f"({top_share * 100:.0f}% of attribution mass)\n"
        f"  -> {top_feature.feature.text}"
    )
    print(
        "\nDiagnosis confirmed by the task metric:\n"
        f"  tool accuracy with the misleading description: {before:.2f}\n"
        f"  tool accuracy after restoring the description: {after:.2f}"
    )
    print(
        "\nLens, not oracle: attribution located the sentence that drives "
        "routing; the before/after accuracy is what proves the fix."
    )
    return {
        "top_feature": top_feature.feature.name,
        "top_feature_text": top_feature.feature.text,
        "accuracy_before": before,
        "accuracy_after": after,
    }


if __name__ == "__main__":
    main()
