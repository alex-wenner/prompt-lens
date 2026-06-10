"""Find the instruction that decides a routing call, then prove the fix.

Scenario: a support agent has two tools and must answer "where is my recent
purchase?" by calling ``lookup_order``. The routing rules — including the
description of the ``order_reference`` parameter — live in the agent's
instruction prompt. promptlens masks each sentence in turn and measures how the
agent's tool choice changes, surfacing the one sentence that actually drives the
routing decision.

By default this runs against a **real provider** (set ``OPENAI_API_KEY`` or
``ANTHROPIC_API_KEY``; see ``examples/_shared.py`` for the env vars). When
no credential is available it falls back to a deterministic offline adapter that
routes to ``lookup_order`` only while the order-id hint survives, so the example
still runs end-to-end and doubles as a CI smoke test.

Attribution is a lens, not an oracle: it points at the load-bearing text so you
know *where* to look, and the before/after task metric is what proves the fix.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

from promptlens import AttributionHarness
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, tool
from promptlens.scorers import ToolAccuracyScorer
from promptlens.segmenters import SentenceSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import print_footer, select_adapter  # noqa: E402

# The offline fallback model routes to lookup_order only when this hint survives
# in the prompt. It appears in exactly one sentence: the order_reference
# description.
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


# Tools are declared once with the provider-neutral ``@tool`` decorator; the
# active adapter coerces each ``Tool`` into its provider's schema. The offline
# adapter ignores them and routes from the prompt text alone.
@tool
def lookup_order(
    order_reference: Annotated[str, "Identifier for the customer's existing purchase."],
) -> str:
    """Look up the status of an existing customer order."""


@tool
def search_catalog(query: Annotated[str, "What the customer wants to buy."]) -> str:
    """Search the product catalog for items to buy."""


TOOLS: ToolDefinitions = [lookup_order, search_catalog]


class SimulatedRoutingAgent(Adapter):
    """Offline fallback: route to ``lookup_order`` only when the order-id hint is visible.

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


def _accuracy(prompt: str, adapter: Adapter) -> float:
    scorer = ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"])
    output = adapter.complete(prompt, tools=TOOLS)
    return scorer.score(CompletionOutput(text=""), output)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the headline numbers for inspection and tests."""
    adapter = adapter if adapter is not None else select_adapter(SimulatedRoutingAgent())
    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"]),
    )
    result = harness.explain(GOOD_PROMPT, tools=TOOLS)
    ranked = result.ranked()
    top_feature, top_share = ranked[0][0], ranked[0][1]

    before = _accuracy(MISLEADING_PROMPT, adapter)
    after = _accuracy(GOOD_PROMPT, adapter)

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
    print_footer(
        "attribution located the sentence that drives routing; the before/after "
        "accuracy is what proves the fix."
    )
    return {
        "top_feature": top_feature.feature.name,
        "top_feature_text": top_feature.feature.text,
        "accuracy_before": before,
        "accuracy_after": after,
    }


if __name__ == "__main__":
    main()
