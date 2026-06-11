"""Smoke tests that keep the examples runnable in CI.

The examples themselves only make real provider calls — they contain no
simulated models. Each exposes ``main(adapter=...)``, so these tests inject
deterministic stub adapters (defined here, not in the examples) to assert the
documented behavior without network access or provider credentials.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions, Usage

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_example(name: str) -> ModuleType:
    path = _EXAMPLES_DIR / name / "run.py"
    spec = importlib.util.spec_from_file_location(f"promptlens_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _usage(prompt: str, text: str) -> Usage:
    return Usage(
        input_tokens=max(1, len(prompt.split())), output_tokens=max(1, len(text.split()))
    )


class StubRoutingAgent(Adapter):
    """Route to lookup_order only while the order-id hint is visible."""

    model = "stub-routing-agent"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if "order id" in prompt.lower():
            tool_call: dict[str, Any] = {
                "name": "lookup_order",
                "arguments": {"order_reference": "#1234"},
            }
        else:
            tool_call = {"name": "search_catalog", "arguments": {"query": "recent purchase"}}
        text = "Your order shipped." if "tool results" in prompt.lower() else ""
        return CompletionOutput(
            text=text, tool_calls=[tool_call], usage=_usage(prompt, text)
        )


def test_tool_routing_bug_example() -> None:
    module = _load_example("tool_routing_bug")
    result = module.main(adapter=StubRoutingAgent())

    assert result["top_feature"] == "sentence_3"
    assert "order ID" in result["top_feature_text"]
    assert result["accuracy_before"] == 0.0
    assert result["accuracy_after"] == 1.0
    assert result["final_answer"]


class StubFormatter(Adapter):
    """Output whose shape depends only on the two load-bearing lines."""

    model = "stub-formatter"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        if "valid json" in lowered:
            text = '{"status": "Order shipped.", "format": "json-mode-enabled"}'
        else:
            text = "Order shipped."
        if "confidence score" in lowered:
            text = f"{text} [confidence=0.90 high-certainty]"
        return CompletionOutput(text=text, usage=_usage(prompt, text))


def test_system_prompt_cleanup_example() -> None:
    module = _load_example("system_prompt_cleanup")
    result = module.main(adapter=StubFormatter())

    load_bearing = result["load_bearing"]
    assert any("valid JSON" in text for text in load_bearing)
    assert any("confidence score" in text for text in load_bearing)
    # Polite boilerplate carries no measured drift.
    assert all("hero of their own story" not in text for text in load_bearing)


class StubOptimizerAdapter(Adapter):
    """Echo during attribution; return a scripted rewrite for the optimizer brief."""

    model = "stub-optimizer"
    rewrite = "Summarize the input text in exactly three bullet points."
    rationale = (
        "Kept the only load-bearing instruction (the three-bullet constraint) and "
        "pruned the inert pleasantries that carried no attribution."
    )

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        if "REWRITTEN PROMPT:" in prompt:
            text = f"REWRITTEN PROMPT:\n{self.rewrite}\n\nRATIONALE:\n{self.rationale}"
        else:
            text = prompt
        return CompletionOutput(text=text, usage=_usage(prompt, text))


def test_optimize_before_after_example() -> None:
    module = _load_example("optimize_before_after")
    result = module.main(adapter=StubOptimizerAdapter())

    assert result["proposed_prompt"] == StubOptimizerAdapter.rewrite
    assert result["proposed_prompt"] != result["original_prompt"]
    assert result["rationale"]


class StubOrderOpsAgent(Adapter):
    """A trajectory governed by the policy sentences visible in the masked prompt."""

    model = "stub-order-ops-agent"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        visible = prompt.lower()
        audited = "order_id on every tool call" in visible

        def arguments(**kwargs: Any) -> dict[str, Any]:
            return {"order_id": "ORD-7421", **kwargs} if audited else kwargs

        calls: list[dict[str, Any]] = []
        if "call lookup_order first" in visible:
            calls.append({"name": "lookup_order", "arguments": arguments()})
        if "require an rma" in visible:
            calls.append(
                {"name": "create_rma", "arguments": arguments(reason_code="damaged_in_transit")}
            )
        if "must be escalated" in visible:
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
        if "reply to the account manager in json" in visible:
            text = json.dumps(
                {
                    "status": status,
                    "action_taken": action,
                    "next_step": "Reply to Dana with the outcome.",
                }
            )
        else:
            text = f"Handled the Helio Manufacturing ticket: {action}."
        return CompletionOutput(text=text, tool_calls=calls, usage=_usage(prompt, text))


def test_order_operations_agent_example() -> None:
    module = _load_example("order_operations_agent")
    result = module.main(adapter=StubOrderOpsAgent())

    refined = result["refined"]
    headings = result["headings"]
    refund_sections = [
        name for name in refined if headings[name] == "# Refund policy"
    ]
    assert refund_sections, "the refund policy section should be refined"
    assert "must be escalated" in refined[refund_sections[0]]
    tool_sections = [
        name for name in refined if headings[name] == "# Tool usage rules"
    ]
    assert tool_sections, "the tool usage rules section should be refined"
    # Glossary and tone are dead weight for this ticket.
    inert = [
        name
        for name, heading in headings.items()
        if heading in {"# Business objects", "# Tone and communication"}
    ]
    assert all(result["shares"][name] < 0.05 for name in inert)
    assert result["calls_used"] < result["flat_calls"]


class StubLocalModel(Adapter):
    """Review text shaped by the two formatting paragraphs; canned synopsis."""

    model = "stub-local-model"
    synopsis = (
        "The severity-tagging and markdown-checklist paragraphs carry the output; "
        "together they account for nearly all measured drift."
    )

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        if "attribution evidence" in lowered:
            return CompletionOutput(text=self.synopsis, usage=_usage(prompt, self.synopsis))
        review = "Found a possible off-by-one in the pagination helper."
        if "severity of blocker" in lowered:
            review += " [blocker] fails test_pagination_last_page; [nit] rename idx."
        if "markdown checklist" in lowered:
            review = "## pagination.py\n- " + review.replace("; ", "\n- ")
        return CompletionOutput(text=review, usage=_usage(prompt, review))


def test_local_inference_example() -> None:
    module = _load_example("local_inference")
    result = module.main(adapter=StubLocalModel())

    # The two formatting paragraphs carry the output; role/tone are inert.
    assert result["top_feature"].startswith("paragraph_")
    assert result["synopsis"]
    assert result["synopsis_model"] == "stub-local-model"


class StubHelpdeskAgent(Adapter):
    """Two redundant instructions both drive reply length."""

    model = "stub-helpdesk-agent"
    long_reply = (
        "Here is what is happening: your order cleared payment, entered fulfillment, "
        "and shipped this morning. For example, order #4821 followed the same path "
        "yesterday and arrived within a day. Tracking is on its way to your inbox now."
    )
    terse_reply = "Your order shipped today."

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        lowered = prompt.lower()
        verbose = "reasoning step by step" in lowered or "worked example" in lowered
        text = self.long_reply if verbose else self.terse_reply
        return CompletionOutput(text=text, usage=_usage(prompt, text))


def test_interaction_effects_example() -> None:
    module = _load_example("interaction_effects")
    result = module.main(adapter=StubHelpdeskAgent())

    # Leave-one-out misses both redundant drivers; Banzhaf recovers both.
    assert abs(result["leave_one_out"]["reasoning"]) < 1e-9
    assert abs(result["leave_one_out"]["example"]) < 1e-9
    assert result["banzhaf"]["reasoning"] > 0.1
    assert result["banzhaf"]["example"] > 0.1


class StubPricedModel(Adapter):
    """Fixed-usage stub so cost projections are deterministic."""

    model = "openai/gpt-5.4"

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        del tools
        return CompletionOutput(
            text="Summary of the brief.", usage=Usage(input_tokens=500, output_tokens=120)
        )


def test_cost_compare_example() -> None:
    module = _load_example("cost_compare")
    result = module.main(adapter=StubPricedModel())

    assert result["full_total_usd"] > result["quick_total_usd"]
    assert result["full_evaluations"] > result["quick_evaluations"]
    # Local inference is the free option in the comparison.
    assert result["comparisons"]["ollama/llama3.2"] == 0.0
