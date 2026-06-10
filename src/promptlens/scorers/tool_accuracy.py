"""Tool-call accuracy scorer."""

from __future__ import annotations

from promptlens.core.base import CompletionOutput, Scorer
from promptlens.scorers._tool_calls import tool_call_arguments, tool_call_name


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
            if tool_call_name(tool_call) != self.expected_tool:
                continue
            arguments = tool_call_arguments(tool_call)
            if not isinstance(arguments, dict):
                return 0.5
            present = sum(
                1 for arg in self.required_args if arg in arguments and arguments[arg] is not None
            )
            if not self.required_args:
                return 1.0
            return 0.5 + 0.5 * (present / len(self.required_args))
        return 0.0
