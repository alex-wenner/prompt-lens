"""Leave-one-out coalition sampler."""

from __future__ import annotations

from collections.abc import Iterator

from promptlens.core.base import Coalition, Sampler


class LeaveOneOutSampler(Sampler):
    """Generate one coalition per feature with exactly that feature masked."""

    def sample(self, n_features: int) -> Iterator[Coalition]:
        for index in range(n_features):
            yield tuple(position != index for position in range(n_features))

    def estimate_evaluations(self, n_features: int) -> int:
        return n_features
