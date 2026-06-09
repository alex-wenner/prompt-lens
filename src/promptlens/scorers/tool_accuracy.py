"""Tool-call accuracy scorer."""

from __future__ import annotations

import json
from typing import Any

from promptlens.core.base import CompletionOutput, Scorer


class ToolAccuracyScorer(Scorer):
    """Score whether a completion selected the expected tool and required arguments.

    This is an ``"objective"`` (task-quality) scorer, not a drift scorer: it
    ignores the baseline and returns higher values when ``candidate`` picked the
    expected tool with its required arguments. The harness converts this into an
    attribution signal by measuring how much the objective drops when a feature
    is masked, so a feature whose removal still yields the correct tool call
    correctly receives *low* attribution.
    """

    orientation = "objective"

    def __init__(self, expected_tool: str, required_args: list[str] | None = None) -> None:
        self.expected_tool = expected_tool
        self.required_args = required_args or []

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        del baseline
        for tool_call in candidate.tool_calls:
            name = str(tool_call.get("name") or tool_call.get("function", {}).get("name") or "")
            if name != self.expected_tool:
                continue
            arguments = _parse_arguments(tool_call.get("arguments") or tool_call.get("input"))
            if not isinstance(arguments, dict):
                return 0.5
            present = sum(
                1 for arg in self.required_args if arg in arguments and arguments[arg] is not None
            )
            if not self.required_args:
                return 1.0
            return 0.5 + 0.5 * (present / len(self.required_args))
        return 0.0


def _parse_arguments(arguments: Any) -> Any:
    """Decode tool-call arguments, which OpenAI delivers as a JSON string."""
    if arguments is None:
        return {}
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return arguments
    return arguments
