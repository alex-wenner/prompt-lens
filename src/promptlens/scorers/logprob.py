"""Log probability scorer."""

from __future__ import annotations

from promptlens.core.base import CompletionOutput, Scorer


class LogprobScorer(Scorer):
    """Score candidates by relative loss of average token log probability."""

    def score(self, baseline: CompletionOutput, candidate: CompletionOutput) -> float:
        if not baseline.logprobs or not candidate.logprobs:
            return 0.0
        baseline_avg = sum(baseline.logprobs) / len(baseline.logprobs)
        candidate_avg = sum(candidate.logprobs) / len(candidate.logprobs)
        return max(0.0, baseline_avg - candidate_avg)
