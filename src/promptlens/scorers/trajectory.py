"""Trajectory scorers for agent runs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from promptlens.core.base import CompletionOutput, Scorer


class ToolSequenceDriftScorer(Scorer):
    """Score drift between baseline and candidate tool-call sequences.

    Compares the *ordered* tool names of the two outputs with a normalized edit
    distance: ``0.0`` means the candidate called the same tools in the same
    order, ``1.0`` means the sequences share nothing. For single completions
    this measures tool-choice drift; for :class:`~promptlens.adapters.AgentAdapter`
    runs it measures how much an agent's tool path changed when part of its
    instructions was masked.

    Only tool *names* are compared, so a run that calls the same tools with
    different arguments scores ``0.0``; combine with a text scorer over the
    final answer via :class:`~promptlens.scorers.CompositeScorer` when argument
    or answer drift also matters.
    """

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        base = _tool_names(baseline.tool_calls)
        cand = _tool_names(candidate.tool_calls)
        if not base and not cand:
            return 0.0
        return _edit_distance(base, cand) / max(len(base), len(cand))


def _tool_names(tool_calls: Sequence[dict[str, Any]]) -> list[str]:
    return [
        str(call.get("name") or call.get("function", {}).get("name") or "")
        for call in tool_calls
    ]


def _edit_distance(left: Sequence[str], right: Sequence[str]) -> int:
    """Levenshtein distance between two sequences of tool names."""
    if not left:
        return len(right)
    if not right:
        return len(left)
    previous = list(range(len(right) + 1))
    for row, left_item in enumerate(left, start=1):
        current = [row]
        for column, right_item in enumerate(right, start=1):
            substitution = previous[column - 1] + (left_item != right_item)
            current.append(min(previous[column] + 1, current[-1] + 1, substitution))
        previous = current
    return previous[-1]
