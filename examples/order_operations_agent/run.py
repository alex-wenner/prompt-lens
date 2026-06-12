"""Drill into a production-sized operations prompt without a flat sentence sweep.

Scenario: "Atlas" is an order-operations agent working a real ticket — an
account manager wants a $182.40 refund on a damaged order. Its instruction set
(``instructions.md``) is the realistic kind: eight markdown sections tying the
agent to business objects (orders, RMAs, credit memos), a refund policy with a
dollar threshold, tool usage rules, an escalation matrix, and an output
contract. Masking it one sentence at a time costs ~30 provider calls per sweep,
most of which would confirm that the glossary is a glossary.

Drill-down spends the calls where they matter: attribute the eight *sections*
first, then re-attribute only the top two — sentence by sentence, with the rest
of the prompt intact. Trajectory drift is scored with the argument-aware tool
scorer (a free-text ``summary`` parameter is weighted to zero so rephrasing
never counts as drift) blended with output-length drift for the reply envelope.

This runs against a **real provider** (set ``OPENAI_API_KEY``, or pick another
provider via ``PROMPTLENS_EXAMPLE_PROVIDER``; see ``examples/_shared.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Annotated, Any

from promptlens import AttributionHarness, explain_drilldown
from promptlens.cli.render import render_tools
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, tool
from promptlens.scorers import CompositeScorer, LengthDriftScorer, ToolArgumentDriftScorer
from promptlens.segmenters import MarkdownSectionSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer, require_adapter  # noqa: E402

INSTRUCTIONS = load_text(__file__, "instructions.md")

TICKET = (
    "Account manager Dana Whitfield: customer Helio Manufacturing requests a "
    "refund of $182.40 on order ORD-7421 — the item arrived damaged. Order "
    "status is delivered; account tier is preferred."
)

@tool
def lookup_order(order_id: Annotated[str, "Order identifier, for example ORD-7421."]) -> str:
    """Fetch an order's status, line items, and payment state."""


@tool
def create_rma(
    order_id: Annotated[str, "Order the items are coming back from."],
    reason_code: Annotated[str, "Reason for the return, in the customer's words."],
) -> str:
    """Open a return-merchandise authorization for an order."""


@tool
def issue_refund(
    order_id: Annotated[str, "Order to refund."],
    amount: Annotated[float, "Refund amount in USD."],
) -> str:
    """Refund the customer for an order, within policy limits."""


@tool
def escalate_to_human(
    order_id: Annotated[str, "Order the escalation concerns."],
    reason_code: Annotated[str, "Machine-readable escalation reason."],
    summary: Annotated[str, "One-sentence summary for the reviewer."],
) -> str:
    """Hand the ticket to a human reviewer."""


TOOLS: ToolDefinitions = [lookup_order, create_rma, issue_refund, escalate_to_human]


class TicketedAdapter(Adapter):
    """Append the fixed ticket to every (masked) instruction set.

    The harness masks the *instructions*; the ticket is the held-fixed task, so
    it rides outside the attribution surface — the same shape as
    ``AgentAdapter``'s system-prompt/task split, in single-completion form.
    """

    def __init__(self, inner: Adapter) -> None:
        self.inner = inner
        self.model = inner.model

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        return self.inner.complete(f"{prompt}\n\n# Ticket\n\n{TICKET}", tools=tools)


def build_scorer() -> CompositeScorer:
    """Trajectory drift first, reply-envelope drift second.

    The free-text ``summary`` parameter is weighted to zero: a reviewer note
    phrased differently is not a behavior change, and without the zero weight
    every rewording would register as argument drift.
    """
    return CompositeScorer(
        [
            (ToolArgumentDriftScorer(param_weights={"summary": 0.0}), 0.7),
            (LengthDriftScorer(), 0.3),
        ]
    )


def main(adapter: Adapter | None = None) -> dict[str, Any]:
    """Run the demo and return the headline numbers for inspection and tests."""
    inner = adapter if adapter is not None else require_adapter()
    render_tools(TOOLS)
    harness = AttributionHarness(
        adapter=TicketedAdapter(inner),
        segmenter=MarkdownSectionSegmenter(),
        scorer=build_scorer(),
    )
    result = explain_drilldown(harness, INSTRUCTIONS, tools=TOOLS, top_k=2)

    print("Coarse-to-fine attribution over the Atlas instruction set:\n")
    result.print()

    headings = {
        attribution.feature.name: attribution.feature.text.splitlines()[0]
        for attribution in result.overview.attributions
    }
    shares = {
        attribution.feature.name: share for attribution, share in result.overview.ranked()
    }
    refined = {
        refinement.feature.name: refinement.result.ranked()[0][0].feature.text
        for refinement in result.refinements
    }
    print("\nWhat drill-down found:")
    for name, top_sentence in refined.items():
        print(f"  {headings[name]} -> {top_sentence}")
    print(
        f"\nProvider calls: {result.provider_calls_used} with drill-down vs "
        f"~{result.flat_sweep_provider_calls} for a flat sentence sweep."
    )
    print_footer(
        "drill-down finds the sentences that drive the trajectory; confirm policy "
        "edits with a task-level metric before shipping."
    )
    return {
        "headings": headings,
        "shares": shares,
        "refined": refined,
        "calls_used": result.provider_calls_used,
        "flat_calls": result.flat_sweep_provider_calls,
    }


if __name__ == "__main__":
    main()
