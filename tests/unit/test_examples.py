"""Smoke tests that keep the examples runnable in CI.

Each example exposes a ``main()`` that runs offline with a deterministic
simulated adapter and returns its headline numbers, so we assert the documented
behavior here rather than just importing the modules.
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
    result = _load_example("tool_routing_bug").main()

    assert result["top_feature"] == "sentence_3"
    assert "order ID" in result["top_feature_text"]
    assert result["accuracy_before"] == 0.0
    assert result["accuracy_after"] == 1.0


def test_system_prompt_cleanup_example() -> None:
    result = _load_example("system_prompt_cleanup").main()

    load_bearing = result["load_bearing"]
    assert any("valid JSON" in text for text in load_bearing)
    assert any("confidence score" in text for text in load_bearing)
    # Polite boilerplate carries no measured drift.
    assert all("hero of their own story" not in text for text in load_bearing)


def test_optimize_before_after_example() -> None:
    result = _load_example("optimize_before_after").main()

    assert result["proposed_prompt"] == "Summarize the input text in exactly three bullet points."
    assert result["proposed_prompt"] != result["original_prompt"]
    assert result["rationale"]
