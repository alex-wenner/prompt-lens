"""Attribution harness orchestration."""

from __future__ import annotations

import math
import statistics

from promptlens.core.base import (
    Adapter,
    CompletionOutput,
    Feature,
    Masker,
    PromptMutator,
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
        perturbation_scale: str | int = "quick",
        expected_output_tokens: int = 300,
    ) -> None:
        self.adapter = adapter
        self.segmenter = segmenter
        self.scorer = scorer
        self.masker = masker or PlaceholderMasker()
        self.sampler = sampler or _sampler_from_scale(perturbation_scale)
        self.supplementary_mutator = supplementary_mutator
        self.perturbation_scale = perturbation_scale
        self.expected_output_tokens = expected_output_tokens

    def estimate(
        self,
        prompt: str,
        tools: ToolDefinitions | None = None,
        compare_models: list[str] | None = None,
    ) -> CostEstimate:
        features = self.segmenter.segment(prompt, tools=tools)
        evaluations = self.sampler.estimate_evaluations(len(features))
        return estimate_cost(
            model=self.adapter.model,
            prompt=prompt,
            features=len(features),
            evaluations=evaluations,
            expected_output_tokens=self.expected_output_tokens,
            compare_models=compare_models,
        )

    def explain(self, prompt: str, tools: ToolDefinitions | None = None) -> AttributionResult:
        features = self.segmenter.segment(prompt, tools=tools)
        baseline = self.adapter.complete(prompt, tools=tools)
        coalitions = list(self.sampler.sample(len(features)))
        masked_prompts = [self.masker.mask(features, coalition) for coalition in coalitions]
        candidates = self.adapter.complete_batch(masked_prompts, tools=tools)
        if len(candidates) != len(coalitions):
            msg = (
                f"Adapter.complete_batch returned {len(candidates)} outputs for "
                f"{len(coalitions)} prompts"
            )
            raise ValueError(msg)
        evaluations: list[CoalitionEvaluation] = []
        attributions: list[FeatureAttribution] = []
        feature_scores: dict[int, list[float]] = {index: [] for index in range(len(features))}
        for coalition, masked_prompt, candidate in zip(
            coalitions, masked_prompts, candidates, strict=True
        ):
            score = self.scorer.score(baseline, candidate)
            evaluations.append(
                CoalitionEvaluation(
                    coalition=coalition,
                    prompt=masked_prompt,
                    output=candidate,
                    score=score,
                )
            )
            for index, included in enumerate(coalition):
                if not included:
                    feature_scores[index].append(score)
        for index, feature in enumerate(features):
            scores = feature_scores[index]
            value = sum(scores) / len(scores) if scores else 0.0
            stderr = None
            if len(scores) > 1:
                stderr = statistics.stdev(scores) / math.sqrt(len(scores))
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
