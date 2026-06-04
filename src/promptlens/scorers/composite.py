"""Composite scorer combining weighted sub-scorers."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import CompletionOutput, Scorer


class CompositeScorer(Scorer):
    """Combine several scorers into one weighted scalar signal.

    Each component contributes ``weight * scorer.score(...)`` and the results are
    summed. Useful when "what changed" is best captured by more than one signal,
    e.g. ``0.7`` embedding drift plus ``0.3`` tool accuracy.
    """

    def __init__(self, scorers: Sequence[tuple[Scorer, float]]) -> None:
        if not scorers:
            msg = "CompositeScorer requires at least one (scorer, weight) pair"
            raise ValueError(msg)
        self.scorers: list[tuple[Scorer, float]] = list(scorers)

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        return sum(
            weight * scorer.score(baseline, candidate) for scorer, weight in self.scorers
        )
