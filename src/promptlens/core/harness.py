"""Attribution harness orchestration."""

from __future__ import annotations

from promptlens.core.base import Adapter, Masker, Sampler, Scorer, Segmenter, ToolDefinitions
from promptlens.core.pricing import estimate_cost
from promptlens.core.result import (
    AttributionResult,
    CoalitionEvaluation,
    CostEstimate,
    FeatureAttribution,
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
        perturbation_scale: str | int = "quick",
        expected_output_tokens: int = 300,
    ) -> None:
        self.adapter = adapter
        self.segmenter = segmenter
        self.scorer = scorer
        self.masker = masker or PlaceholderMasker()
        self.sampler = sampler or _sampler_from_scale(perturbation_scale)
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
        evaluations: list[CoalitionEvaluation] = []
        attributions: list[FeatureAttribution] = []
        feature_scores: dict[int, list[float]] = {index: [] for index in range(len(features))}
        for coalition in self.sampler.sample(len(features)):
            masked_prompt = self.masker.mask(features, coalition)
            candidate = self.adapter.complete(masked_prompt, tools=tools)
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
            attributions.append(FeatureAttribution(feature=feature, value=value))
        return AttributionResult(
            baseline_output=baseline,
            attributions=attributions,
            evaluations=evaluations,
            cost_estimate=self.estimate(prompt, tools=tools),
        )


def _sampler_from_scale(scale: str | int) -> Sampler:
    if isinstance(scale, int):
        return LeaveOneOutSampler()
    if scale == "quick":
        return LeaveOneOutSampler()
    if scale in {"standard", "full"}:
        return LeaveOneOutSampler()
    msg = f"Unsupported perturbation scale: {scale}"
    raise ValueError(msg)
