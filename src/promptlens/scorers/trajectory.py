"""Trajectory scorers for agent runs."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from promptlens.core.base import CompletionOutput, Scorer
from promptlens.scorers._tool_calls import tool_call_arguments, tool_call_name

# Sentinel distinguishing "parameter absent" from "parameter present with None".
_ABSENT = object()


class ToolSequenceDriftScorer(Scorer):
    """Score drift between baseline and candidate tool-call sequences.

    Compares the *ordered* tool names of the two outputs with a normalized edit
    distance: ``0.0`` means the candidate called the same tools in the same
    order, ``1.0`` means the sequences share nothing. For single completions
    this measures tool-choice drift; for :class:`~promptlens.adapters.AgentAdapter`
    runs it measures how much an agent's tool path changed when part of its
    instructions was masked.

    Only tool *names* are compared, so a run that calls the same tools with
    different arguments scores ``0.0``; use
    :class:`ToolArgumentDriftScorer` when the arguments passed to those tools
    should also count, or combine with a text scorer over the final answer via
    :class:`~promptlens.scorers.CompositeScorer` when answer drift matters too.
    """

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        base = _tool_names(baseline.tool_calls)
        cand = _tool_names(candidate.tool_calls)
        if not base and not cand:
            return 0.0
        distance = _edit_distance(
            base, cand, lambda left, right: float(left != right)
        )
        return distance / max(len(base), len(cand))


class ToolArgumentDriftScorer(Scorer):
    """Argument-aware tool-sequence drift with explicit per-parameter weights.

    Like :class:`ToolSequenceDriftScorer`, this compares the ordered tool calls
    of two outputs with a normalized edit distance — but calls to the *same*
    tool are compared parameter by parameter instead of being treated as
    identical. Calling a different tool still costs a full ``1.0`` per
    position; calling the same tool with different arguments costs at most
    ``argument_weight``, so argument churn can never swing the score more than
    an outright tool-choice change.

    Per-parameter weights decide how much each argument matters:

    * ``param_weights`` maps parameter names to relative weights. Weight ``0.0``
      makes a parameter inert — a noisy free-text ``reason`` field, say — while
      a heavy weight makes a critical parameter (``account_id``) dominate.
      Unlisted parameters get ``default_param_weight``.
    * ``none_is_missing`` (default ``True``) decides the "agent passed
      undefined" question: with the default, an explicit ``None``/null argument
      is treated exactly like omitting the parameter, so an agent that pads
      calls with nulls does not register drift against one that leaves them
      out. Set it to ``False`` when a null is meaningful to the tool (e.g. it
      clears a value server-side) and *should* count as a different call.

    The per-call argument drift is the weighted fraction of parameters (over
    the union of both calls' parameter names) whose values differ, so it stays
    in ``[0, 1]`` regardless of the configured weights.
    """

    def __init__(
        self,
        *,
        argument_weight: float = 0.5,
        param_weights: Mapping[str, float] | None = None,
        default_param_weight: float = 1.0,
        none_is_missing: bool = True,
    ) -> None:
        if not 0.0 <= argument_weight <= 1.0:
            msg = f"argument_weight must be in [0, 1], got {argument_weight}"
            raise ValueError(msg)
        if default_param_weight < 0:
            msg = f"default_param_weight must be >= 0, got {default_param_weight}"
            raise ValueError(msg)
        weights = dict(param_weights or {})
        for name, weight in weights.items():
            if weight < 0:
                msg = f"param_weights[{name!r}] must be >= 0, got {weight}"
                raise ValueError(msg)
        self.argument_weight = argument_weight
        self.param_weights = weights
        self.default_param_weight = default_param_weight
        self.none_is_missing = none_is_missing

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        base = [self._normalize(call) for call in baseline.tool_calls]
        cand = [self._normalize(call) for call in candidate.tool_calls]
        if not base and not cand:
            return 0.0
        distance = _edit_distance(base, cand, self._substitution_cost)
        return distance / max(len(base), len(cand))

    def _normalize(self, call: dict[str, Any]) -> tuple[str, Any]:
        arguments = tool_call_arguments(call)
        if isinstance(arguments, dict) and self.none_is_missing:
            arguments = {key: value for key, value in arguments.items() if value is not None}
        return tool_call_name(call), arguments

    def _substitution_cost(self, left: tuple[str, Any], right: tuple[str, Any]) -> float:
        left_name, left_args = left
        right_name, right_args = right
        if left_name != right_name:
            return 1.0
        return self.argument_weight * self._argument_drift(left_args, right_args)

    def _argument_drift(self, left: Any, right: Any) -> float:
        if not isinstance(left, dict) or not isinstance(right, dict):
            return 0.0 if left == right else 1.0
        names = set(left) | set(right)
        if not names:
            return 0.0
        total = sum(self._param_weight(name) for name in names)
        if total == 0:
            return 0.0
        changed = sum(
            self._param_weight(name)
            for name in names
            if left.get(name, _ABSENT) != right.get(name, _ABSENT)
        )
        return changed / total

    def _param_weight(self, name: str) -> float:
        return self.param_weights.get(name, self.default_param_weight)


def _tool_names(tool_calls: Sequence[dict[str, Any]]) -> list[str]:
    return [tool_call_name(call) for call in tool_calls]


def _edit_distance(
    left: Sequence[Any],
    right: Sequence[Any],
    substitution_cost: Callable[[Any, Any], float],
) -> float:
    """Levenshtein distance with a pluggable (possibly fractional) substitution cost."""
    if not left:
        return float(len(right))
    if not right:
        return float(len(left))
    previous = [float(index) for index in range(len(right) + 1)]
    for row, left_item in enumerate(left, start=1):
        current = [float(row)]
        for column, right_item in enumerate(right, start=1):
            substitution = previous[column - 1] + substitution_cost(left_item, right_item)
            current.append(min(previous[column] + 1.0, current[-1] + 1.0, substitution))
        previous = current
    return previous[-1]
