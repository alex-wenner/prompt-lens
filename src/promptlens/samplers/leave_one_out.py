"""Leave-one-out coalition sampler."""

from __future__ import annotations

from collections.abc import Iterator

from promptlens.core.base import Coalition, Sampler


class LeaveOneOutSampler(Sampler):
    """Mask exactly one feature per coalition.

    Each feature is occluded on its own so its attribution is an unconfounded
    marginal effect relative to the full prompt. ``repeats`` re-runs the full
    sweep, which reduces variance for non-deterministic providers (temperature
    above zero) by averaging multiple samples per feature.
    """

    def __init__(self, repeats: int = 1) -> None:
        if repeats < 1:
            msg = f"repeats must be >= 1, got {repeats}"
            raise ValueError(msg)
        self.repeats = repeats

    def sample(self, n_features: int) -> Iterator[Coalition]:
        for _ in range(self.repeats):
            for index in range(n_features):
                yield tuple(position != index for position in range(n_features))

    def estimate_evaluations(self, n_features: int) -> int:
        return n_features * self.repeats
