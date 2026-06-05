"""Composite scorer combining weighted sub-scorers."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import CompletionOutput, Scorer


class CompositeScorer(Scorer):
    """Combine several scorers into one weighted scalar signal.

    Each component contributes ``weight * scorer.score(...)`` and the results are
    summed. Useful when "what changed" is best captured by more than one signal,
    e.g. ``0.7`` embedding drift plus ``0.3`` length drift.

    All components must share the same :attr:`Scorer.orientation`; the composite
    adopts it. Mixing ``"drift"`` and ``"objective"`` scorers in one weighted sum
    is rejected because the two point in opposite directions for attribution, so
    summing them produces a value that is neither a drift nor a quality signal.
    """

    def __init__(self, scorers: Sequence[tuple[Scorer, float]]) -> None:
        if not scorers:
            msg = "CompositeScorer requires at least one (scorer, weight) pair"
            raise ValueError(msg)
        orientations = {scorer.orientation for scorer, _ in scorers}
        if len(orientations) > 1:
            msg = (
                "CompositeScorer components must share one orientation; got "
                f"{sorted(orientations)}. Combine drift scorers with drift scorers and "
                "objective scorers with objective scorers."
            )
            raise ValueError(msg)
        self.orientation = orientations.pop()
        self.scorers: list[tuple[Scorer, float]] = list(scorers)

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        return sum(
            weight * scorer.score(baseline, candidate) for scorer, weight in self.scorers
        )
