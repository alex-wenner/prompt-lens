"""Find the instruction that decides a routing call, then prove the fix.

Scenario: a support agent has two tools and must answer "where is my recent
purchase?" by calling ``lookup_order``. The routing rules — including the
description of the ``order_reference`` parameter — live in the agent's
instruction prompt. promptlens masks each sentence in turn and measures how the
agent's tool choice changes, surfacing the one sentence that actually drives the
routing decision.

This example makes **real provider calls** (export ``OPENAI_API_KEY``, or pick
another provider with ``PROMPTLENS_EXAMPLE_PROVIDER``). It prints the full tool
schemas the model sees, the model's tool calls, and — after executing the chosen
tool — the model's final answer to the tool result, so the whole agent loop is
visible.

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
from _shared import (  # noqa: E402
    complete_tool_round_trip,
    console,
    get_adapter,
    print_completion,
    print_footer,
    print_tools,
)

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
# active adapter coerces each ``Tool`` into its provider's schema.
@tool
def lookup_order(
    order_reference: Annotated[str, "Identifier for the customer's existing purchase."],
) -> str:
    """Look up the status of an existing customer order."""


@tool
def search_catalog(query: Annotated[str, "What the customer wants to buy."]) -> str:
    """Search the product catalog for items to buy."""


TOOLS: ToolDefinitions = [lookup_order, search_catalog]

# Stub backends so the model's tool calls can actually execute, letting the
# example show the model's final response to the tool outcome.
TOOL_IMPLEMENTATIONS: dict[str, Any] = {
    "lookup_order": lambda order_reference="": (
        f"Order {order_reference or '#1234'}: shipped 2 days ago via UPS, "
        "arriving tomorrow. Tracking 1Z999AA10123456784."
    ),
    "search_catalog": lambda query="": f"3 catalog matches for '{query}'.",
}


def _accuracy(prompt: str, adapter: Adapter) -> float:
    scorer = ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"])
    output = adapter.complete(prompt, tools=TOOLS)
    return scorer.score(CompletionOutput(text=""), output)


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the headline numbers for inspection and tests."""
    adapter = adapter if adapter is not None else get_adapter()
    print_tools(TOOLS)

    harness = AttributionHarness(
        adapter=adapter,
        segmenter=SentenceSegmenter(),
        scorer=ToolAccuracyScorer(expected_tool="lookup_order", required_args=["order_reference"]),
    )

    # One real call: the baseline shows the routing decision being attributed.
    baseline, estimate = harness.estimate(GOOD_PROMPT, tools=TOOLS)
    print_completion("Baseline routing decision", baseline)
    estimate.print()

    # Close the loop: execute the chosen tool and show the model's final answer.
    final = complete_tool_round_trip(adapter, GOOD_PROMPT, baseline, TOOL_IMPLEMENTATIONS, TOOLS)
    if final is not None:
        print_completion("Model's answer after the tool result", final)

    console.print("\n[bold]Attribution over the healthy prompt (objective: tool accuracy)[/bold]\n")
    result = harness.explain(GOOD_PROMPT, tools=TOOLS, baseline=baseline)
    result.print()
    ranked = result.ranked()
    top_feature, top_share = ranked[0][0], ranked[0][1]

    before = _accuracy(MISLEADING_PROMPT, adapter)
    after = _accuracy(GOOD_PROMPT, adapter)

    console.print(
        f"\nMost load-bearing feature: [bold magenta]{top_feature.feature.name}[/bold magenta] "
        f"({top_share * 100:.0f}% of attribution mass)\n"
        f"  -> {top_feature.feature.text}"
    )
    console.print(
        "\nDiagnosis confirmed by the task metric:\n"
        f"  tool accuracy with the misleading description: [red]{before:.2f}[/red]\n"
        f"  tool accuracy after restoring the description: [green]{after:.2f}[/green]"
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
        "final_answer": final.text if final else "",
    }


if __name__ == "__main__":
    main()
