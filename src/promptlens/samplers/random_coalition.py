"""Random coalition sampler for interaction-aware attribution."""

from __future__ import annotations

import random
from collections.abc import Iterator

from promptlens.core.base import Coalition, Sampler

_MAX_ATTEMPTS_PER_COALITION = 100
_MAX_ATTEMPTS_BASE = 100


class RandomCoalitionSampler(Sampler):
    """Sample random feature coalitions instead of masking one feature at a time.

    Each coalition independently includes every feature with probability
    ``inclusion_probability`` (default ``0.5``, the SHAP-style uniform-over-subsets
    expectation). Because several features are typically masked together, the
    averaged drift attributed to a feature reflects its effect across many
    contexts rather than the single full-prompt context that leave-one-out uses.
    That makes the sampler sensitive to interactions a pure leave-one-out sweep
    misses, at the cost of being an approximation that needs enough samples to
    stabilize.

    ``seed`` makes runs reproducible. Coalitions that include all or no features
    are resampled so every evaluation call carries signal.

    Because several features are masked per coalition, the harness attributes
    the masked-vs-kept contrast (``mean(score | masked) - mean(score | kept)``)
    rather than the raw mean over masked coalitions, cancelling the shared
    co-masking offset; at ``inclusion_probability=0.5`` this is a Monte-Carlo
    Banzhaf-value estimate. Skipping the degenerate coalitions couples features
    slightly, which tends to push inert features just below zero rather than to
    exactly zero — a conservative bias, since only positive attribution mass is
    distributed as share.
    """

    def __init__(
        self,
        n_coalitions: int = 50,
        *,
        inclusion_probability: float = 0.5,
        seed: int | None = None,
    ) -> None:
        if n_coalitions < 1:
            msg = f"n_coalitions must be >= 1, got {n_coalitions}"
            raise ValueError(msg)
        if not 0.0 < inclusion_probability < 1.0:
            msg = f"inclusion_probability must be in (0, 1), got {inclusion_probability}"
            raise ValueError(msg)
        self.n_coalitions = n_coalitions
        self.inclusion_probability = inclusion_probability
        self.seed = seed

    def sample(self, n_features: int) -> Iterator[Coalition]:
        if n_features <= 0:
            return
        rng = random.Random(self.seed)
        emitted = 0
        # Bound the resampling loop so a degenerate configuration cannot spin
        # forever; mixed coalitions are overwhelmingly likely for n_features >= 2.
        attempts = 0
        max_attempts = self.n_coalitions * _MAX_ATTEMPTS_PER_COALITION + _MAX_ATTEMPTS_BASE
        while emitted < self.n_coalitions and attempts < max_attempts:
            attempts += 1
            coalition = tuple(
                rng.random() < self.inclusion_probability for _ in range(n_features)
            )
            # Skip all-included (nothing masked) and, when more than one feature
            # exists, all-excluded coalitions, which carry no attribution signal.
            if all(coalition):
                continue
            if n_features > 1 and not any(coalition):
                continue
            emitted += 1
            yield coalition

    def estimate_evaluations(self, n_features: int) -> int:
        if n_features <= 0:
            return 0
        return self.n_coalitions
