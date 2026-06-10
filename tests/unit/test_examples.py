"""Smoke tests that keep the examples runnable in CI.

Each example exposes a ``main(adapter=...)`` that returns its headline numbers.
The examples default to a real provider, so these tests pass each example's
deterministic offline adapter explicitly to assert the documented behavior
without network access or provider credentials.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _load_example(name: str) -> ModuleType:
    path = _EXAMPLES_DIR / name / "run.py"
    spec = importlib.util.spec_from_file_location(f"promptlens_example_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_routing_bug_example() -> None:
    module = _load_example("tool_routing_bug")
    result = module.main(adapter=module.SimulatedRoutingAgent())

    assert result["top_feature"] == "sentence_3"
    assert "order ID" in result["top_feature_text"]
    assert result["accuracy_before"] == 0.0
    assert result["accuracy_after"] == 1.0


def test_system_prompt_cleanup_example() -> None:
    module = _load_example("system_prompt_cleanup")
    result = module.main(adapter=module.SimulatedFormatter())

    load_bearing = result["load_bearing"]
    assert any("valid JSON" in text for text in load_bearing)
    assert any("confidence score" in text for text in load_bearing)
    # Polite boilerplate carries no measured drift.
    assert all("hero of their own story" not in text for text in load_bearing)


def test_optimize_before_after_example() -> None:
    module = _load_example("optimize_before_after")
    result = module.main(adapter=module.ScriptedOptimizerAdapter())

    assert result["proposed_prompt"] == "Summarize the input text in exactly three bullet points."
    assert result["proposed_prompt"] != result["original_prompt"]
    assert result["rationale"]


def test_order_operations_agent_example() -> None:
    module = _load_example("order_operations_agent")
    result = module.main(adapter=module.SimulatedOrderOpsAgent())

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
