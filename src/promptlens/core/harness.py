"""Attribution harness orchestration."""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable

from promptlens.core.base import (
    Adapter,
    CompletionOutput,
    Feature,
    Masker,
    PromptMutator,
    PromptOptimizer,
    Sampler,
    Scorer,
    Segmenter,
    ToolDefinitions,
)
from promptlens.core.pricing import estimate_cost
from promptlens.core.result import (
    AttributionResult,
    CoalitionEvaluation,
    CostEstimate,
    FeatureAttribution,
    OptimizationResult,
    SupplementaryEvaluation,
)
from promptlens.maskers import PlaceholderMasker
from promptlens.samplers import LeaveOneOutSampler


class AttributionHarness:
    """Orchestrate segmentation, masking, model calls, scoring, and result assembly."""

    def __init__(
        self,
        *,
        adapter: Adapter,
        segmenter: Segmenter,
        scorer: Scorer,
        masker: Masker | None = None,
        sampler: Sampler | None = None,
        supplementary_mutator: PromptMutator | None = None,
        optimizer: PromptOptimizer | None = None,
        perturbation_scale: str | int = "quick",
        samples_per_coalition: int = 1,
        expected_output_tokens: int = 300,
    ) -> None:
        if samples_per_coalition < 1:
            msg = f"samples_per_coalition must be >= 1, got {samples_per_coalition}"
            raise ValueError(msg)
        self.adapter = adapter
        self.segmenter = segmenter
        self.scorer = scorer
        self.masker = masker or PlaceholderMasker()
        self.sampler = sampler or _sampler_from_scale(perturbation_scale)
        self.supplementary_mutator = supplementary_mutator
        self.optimizer = optimizer
        self.perturbation_scale = perturbation_scale
        self.samples_per_coalition = samples_per_coalition
        self.expected_output_tokens = expected_output_tokens

    def estimate(
        self,
        prompt: str,
        tools: ToolDefinitions | None = None,
        compare_models: list[str] | None = None,
        exact_tokens: bool = False,
    ) -> CostEstimate:
        """Preview attribution cost without making inference calls.

        Segmentation and masking run offline, so input tokens are counted per
        actual masked prompt rather than assuming every call resends the full
        prompt. Supplementary rewrites and the optimizer's rewrite call are not
        included; they add adapter calls on top of this estimate.

        ``exact_tokens=True`` uses the adapter's exact token counter when it has
        one (``AnthropicAdapter`` exposes the provider's free ``count_tokens``
        metering endpoint). That makes one free network call per prompt counted
        — no inference, no token charges — and falls back silently to the local
        counters for adapters without exact counting.
        """
        features = self.segmenter.segment(prompt, tools=tools)
        masked_prompts = [
            self.masker.mask(features, coalition, prompt=prompt)
            for coalition in self.sampler.sample(len(features))
        ]
        evaluations = len(masked_prompts) * self.samples_per_coalition
        token_count_fn: Callable[[str], int] | None = None
        adapter_counter = getattr(self.adapter, "count_tokens", None)
        if exact_tokens and callable(adapter_counter):

            def _exact_count(text: str) -> int:
                return int(adapter_counter(text, tools=tools))

            token_count_fn = _exact_count
        return estimate_cost(
            model=self.adapter.model,
            prompt=prompt,
            features=len(features),
            evaluations=evaluations,
            expected_output_tokens=self.expected_output_tokens,
            compare_models=compare_models,
            evaluation_prompts=masked_prompts,
            token_count_fn=token_count_fn,
        )

    def explain(self, prompt: str, tools: ToolDefinitions | None = None) -> AttributionResult:
        features = self.segmenter.segment(prompt, tools=tools)
        baseline = self.adapter.complete(prompt, tools=tools)
        coalitions = list(self.sampler.sample(len(features)))
        samples = self.samples_per_coalition
        masked_prompts = [
            self.masker.mask(features, coalition, prompt=prompt) for coalition in coalitions
        ]
        # Objective (task-quality) scorers ignore the baseline and return higher
        # values when the candidate did the desired thing. To turn that into an
        # attribution signal we measure how far the objective drops when a feature
        # is masked, relative to the baseline's own objective. Drift scorers are
        # already attribution signals, so their raw score is used directly.
        objective = self.scorer.orientation == "objective"
        baseline_objective = self.scorer.score(baseline, baseline) if objective else 0.0
        # Evaluate every coalition ``samples`` times so non-deterministic providers
        # yield a distribution per coalition rather than a single draw. Prompts are
        # expanded contiguously (coalition-major) so each group maps back cleanly.
        batch_prompts = [
            masked for masked in masked_prompts for _ in range(samples)
        ]
        candidates = self.adapter.complete_batch(batch_prompts, tools=tools)
        expected = len(coalitions) * samples
        if len(candidates) != expected:
            msg = (
                f"Adapter.complete_batch returned {len(candidates)} outputs for "
                f"{expected} prompts"
            )
            raise ValueError(msg)
        evaluations: list[CoalitionEvaluation] = []
        attributions: list[FeatureAttribution] = []
        masked_scores: dict[int, list[float]] = {index: [] for index in range(len(features))}
        kept_scores: dict[int, list[float]] = {index: [] for index in range(len(features))}
        for c_index, (coalition, masked_prompt) in enumerate(
            zip(coalitions, masked_prompts, strict=True)
        ):
            group = candidates[c_index * samples : (c_index + 1) * samples]
            sample_scores = [self.scorer.score(baseline, candidate) for candidate in group]
            mean_score = sum(sample_scores) / len(sample_scores)
            # For objective scorers the stored coalition score remains the raw
            # task-quality value (transparent), while attribution accumulates the
            # drop from the baseline objective.
            signal_samples = (
                [baseline_objective - score for score in sample_scores]
                if objective
                else sample_scores
            )
            evaluations.append(
                CoalitionEvaluation(
                    coalition=coalition,
                    prompt=masked_prompt,
                    output=group[0],
                    score=mean_score,
                )
            )
            for index, included in enumerate(coalition):
                (kept_scores if included else masked_scores)[index].extend(signal_samples)
        # When every coalition masks exactly one feature (leave-one-out), the mean
        # score over a feature's masked coalitions is its exact marginal effect.
        # When coalitions mask several features at once (random coalitions), that
        # mean confounds the feature's own effect with the average effect of
        # whatever was co-masked, biasing every attribution toward the overall
        # mean drift. The masked-vs-kept contrast removes that shared offset; at
        # inclusion probability 0.5 it is a Monte-Carlo Banzhaf-value estimate.
        contrast = any(coalition.count(False) > 1 for coalition in coalitions)
        for index, feature in enumerate(features):
            masked = masked_scores[index]
            kept = kept_scores[index]
            if not masked:
                value, stderr = 0.0, None
            elif contrast and kept:
                value = _mean(masked) - _mean(kept)
                stderr = _difference_stderr(masked, kept)
            else:
                value = _mean(masked)
                stderr = None
                if len(masked) > 1:
                    stderr = statistics.stdev(masked) / math.sqrt(len(masked))
            attributions.append(FeatureAttribution(feature=feature, value=value, stderr=stderr))
        supplementary_evaluations = self._run_supplementary_mutations(
            prompt=prompt,
            features=features,
            baseline=baseline,
            tools=tools,
        )
        return AttributionResult(
            baseline_output=baseline,
            attributions=attributions,
            evaluations=evaluations,
            cost_estimate=self.estimate(prompt, tools=tools),
            supplementary_evaluations=supplementary_evaluations,
        )

    def optimize(
        self,
        prompt: str,
        tools: ToolDefinitions | None = None,
        result: AttributionResult | None = None,
    ) -> OptimizationResult:
        """Propose an attribution-informed prompt rewrite.

        Runs :meth:`explain` to gather attribution evidence when ``result`` is not
        supplied, then hands that evidence to the configured ``optimizer``. The
        proposed rewrite is returned for review and is never adopted automatically.
        """
        if self.optimizer is None:
            msg = "AttributionHarness.optimize requires an optimizer"
            raise ValueError(msg)
        attribution_result = result if result is not None else self.explain(prompt, tools=tools)
        optimization = self.optimizer.optimize(prompt, attribution_result)
        if not isinstance(optimization, OptimizationResult):
            msg = "Optimizer must return an OptimizationResult"
            raise TypeError(msg)
        return optimization

    def _run_supplementary_mutations(
        self,
        *,
        prompt: str,
        features: list[Feature],
        baseline: CompletionOutput,
        tools: ToolDefinitions | None,
    ) -> list[SupplementaryEvaluation]:
        if self.supplementary_mutator is None:
            return []
        mutations = self.supplementary_mutator.mutate(prompt, features, tools=tools)
        mutated_prompts = [mutation.prompt for mutation in mutations]
        outputs = self.adapter.complete_batch(mutated_prompts, tools=tools)
        if len(outputs) != len(mutations):
            msg = (
                f"Adapter.complete_batch returned {len(outputs)} outputs for "
                f"{len(mutations)} supplementary prompts"
            )
            raise ValueError(msg)
        return [
            SupplementaryEvaluation(
                kind="prompt-mutation",
                feature=mutation.feature,
                prompt=mutation.prompt,
                output=output,
                score=self.scorer.score(baseline, output),
                metadata=mutation.metadata,
            )
            for mutation, output in zip(mutations, outputs, strict=True)
        ]


def _mean(scores: list[float]) -> float:
    return sum(scores) / len(scores)


def _difference_stderr(masked: list[float], kept: list[float]) -> float | None:
    """Standard error of mean(masked) - mean(kept) from per-side sample variances."""
    terms = [
        statistics.variance(scores) / len(scores)
        for scores in (masked, kept)
        if len(scores) > 1
    ]
    if not terms:
        return None
    return math.sqrt(sum(terms))


def _sampler_from_scale(scale: str | int) -> Sampler:
    if isinstance(scale, bool):
        msg = f"Unsupported perturbation scale: {scale}"
        raise ValueError(msg)
    if isinstance(scale, int):
        return LeaveOneOutSampler(repeats=scale)
    repeats = {"quick": 1, "standard": 3, "full": 5}.get(scale)
    if repeats is None:
        msg = f"Unsupported perturbation scale: {scale}"
        raise ValueError(msg)
    return LeaveOneOutSampler(repeats=repeats)
