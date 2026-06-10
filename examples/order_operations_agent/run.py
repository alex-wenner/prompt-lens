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

By default this runs against a **real provider** (set ``OPENAI_API_KEY`` or
``ANTHROPIC_API_KEY``; see ``examples/_shared.py``). With no credential it
falls back to a deterministic offline agent whose tool trajectory is governed by
the same policy sentences a real model would key on, so the example runs
end-to-end and doubles as a CI smoke test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any

from promptlens import AttributionHarness, explain_drilldown
from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, tool
from promptlens.scorers import CompositeScorer, LengthDriftScorer, ToolArgumentDriftScorer
from promptlens.segmenters import MarkdownSectionSegmenter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _shared import load_text, print_footer, select_adapter  # noqa: E402

INSTRUCTIONS = load_text(__file__, "instructions.md")

TICKET = (
    "Account manager Dana Whitfield: customer Helio Manufacturing requests a "
    "refund of $182.40 on order ORD-7421 — the item arrived damaged. Order "
    "status is delivered; account tier is preferred."
)

# Policy sentences the offline agent keys on. Each lives in exactly one
# sentence of instructions.md, so masking that sentence flips the behavior —
# the same causal link attribution should surface for a real model.
ESCALATE_SIGNAL = "must be escalated"
RMA_SIGNAL = "require an rma"
LOOKUP_SIGNAL = "call lookup_order first"
AUDIT_SIGNAL = "order_id on every tool call"
JSON_SIGNAL = "reply to the account manager in json"


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


class SimulatedOrderOpsAgent(Adapter):
    """Offline fallback: a trajectory governed by the visible policy sentences.

    Reads the masked instruction text the harness actually sends and follows
    whatever policy survives — exactly the dependency structure attribution is
    supposed to recover. The ticket is a $182.40 damaged-item refund, so the
    escalation threshold, the RMA prerequisite, the lookup-first rule, the
    order_id audit rule, and the JSON output contract each control one
    observable piece of the run.
    """

    def __init__(self) -> None:
        self.model = "simulated-order-ops-agent"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        visible = prompt.lower()
        audited = AUDIT_SIGNAL in visible

        def arguments(**kwargs: Any) -> dict[str, Any]:
            return {"order_id": "ORD-7421", **kwargs} if audited else kwargs

        calls: list[dict[str, Any]] = []
        if LOOKUP_SIGNAL in visible:
            calls.append({"name": "lookup_order", "arguments": arguments()})
        if RMA_SIGNAL in visible:
            calls.append(
                {"name": "create_rma", "arguments": arguments(reason_code="damaged_in_transit")}
            )
        if ESCALATE_SIGNAL in visible:
            calls.append(
                {
                    "name": "escalate_to_human",
                    "arguments": arguments(
                        reason_code="refund_over_limit",
                        summary="Refund of $182.40 exceeds the $100 direct-refund limit.",
                    ),
                }
            )
            action, status = "escalated_to_reviewer", "pending_review"
        else:
            calls.append({"name": "issue_refund", "arguments": arguments(amount=182.40)})
            action, status = "refund_issued", "resolved"
        if JSON_SIGNAL in visible:
            text = json.dumps(
                {
                    "status": status,
                    "action_taken": action,
                    "next_step": "Reply to Dana with the outcome.",
                }
            )
        else:
            text = f"Handled the Helio Manufacturing ticket: {action}."
        return CompletionOutput(text=text, tool_calls=calls)


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
    inner = adapter if adapter is not None else select_adapter(SimulatedOrderOpsAgent())
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
