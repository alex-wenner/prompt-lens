"""Coarse-to-fine drill-down attribution.

Sentence-level masking over a real instruction set gets expensive fast: a
60-sentence operations prompt is 60+ provider calls per leave-one-out sweep,
most of them spent confirming that boilerplate is boilerplate.
:func:`explain_drilldown` spends those calls where they matter instead:

1. **Overview pass** — attribute the prompt at the harness's own (coarse)
   granularity, typically markdown sections or paragraphs.
2. **Refinement passes** — take the ``top_k`` highest-attribution coarse
   features and re-attribute each one sentence by sentence, keeping the rest
   of the prompt byte-for-byte intact so every refined evaluation still sees
   the full instruction set.

The result keeps both grains: which *sections* carry the behavior, and which
*sentences inside them* do the carrying — at a fraction of the flat-sweep
cost, which the result reports explicitly.
"""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import (
    Coalition,
    Feature,
    Masker,
    Segmenter,
    ToolDefinitions,
)
from promptlens.core.harness import AttributionHarness
from promptlens.core.result import DrilldownRefinement, DrilldownResult
from promptlens.segmenters import SentenceSegmenter


def explain_drilldown(
    harness: AttributionHarness,
    prompt: str,
    tools: ToolDefinitions | None = None,
    *,
    top_k: int = 2,
    fine_segmenter: Segmenter | None = None,
) -> DrilldownResult:
    """Run coarse attribution, then refine the hottest features sentence by sentence.

    The harness's own segmenter provides the coarse grain (use sections or
    paragraphs; sentence-level coarse features cannot be refined further).
    ``fine_segmenter`` defaults to :class:`~promptlens.segmenters.SentenceSegmenter`.

    Refinement candidates are the ``top_k`` coarse features with positive
    attribution. A candidate is skipped when it cannot be located in the prompt
    (no span metadata, e.g. the synthetic tools feature) or when the fine
    segmenter cannot split it into at least two parts (nothing to refine).
    Each refinement masks one fine feature at a time *inside the full prompt*,
    so refined scores stay comparable to the overview's.
    """
    if top_k < 0:
        msg = f"top_k must be >= 0, got {top_k}"
        raise ValueError(msg)
    fine = fine_segmenter or SentenceSegmenter()
    overview = harness.explain(prompt, tools=tools)
    refinements: list[DrilldownRefinement] = []
    for attribution, _ in overview.ranked():
        if len(refinements) >= top_k:
            break
        if attribution.value <= 0:
            break
        refinement = _refine_feature(
            harness, prompt, attribution.feature, fine, tools=tools
        )
        if refinement is not None:
            refinements.append(refinement)
    samples = harness.samples_per_coalition
    used = _provider_calls(overview, samples) + sum(
        _provider_calls(refinement.result, samples) for refinement in refinements
    )
    flat_features = len(fine.segment(prompt, tools=tools))
    flat = harness.sampler.estimate_evaluations(flat_features) * samples + 1
    return DrilldownResult(
        overview=overview,
        refinements=refinements,
        provider_calls_used=used,
        flat_sweep_provider_calls=flat,
    )


def _refine_feature(
    harness: AttributionHarness,
    prompt: str,
    feature: Feature,
    fine: Segmenter,
    *,
    tools: ToolDefinitions | None,
) -> DrilldownRefinement | None:
    if feature.start is None or feature.end is None:
        return None
    span = prompt[feature.start : feature.end]
    core = span.strip()
    if not core:
        return None
    # Fine-segment only the feature's own text; tools are deliberately not
    # re-appended because they already had their shot in the overview pass.
    fine_features = [
        item.model_copy(update={"name": f"{feature.name}.{item.name}"})
        for item in fine.segment(core)
        if not item.metadata.get("kind")
    ]
    if len(fine_features) < 2:
        return None
    lead = span[: len(span) - len(span.lstrip())]
    trail = span[len(span.rstrip()) :]
    stage = AttributionHarness(
        adapter=harness.adapter,
        segmenter=_FixedFeatures(fine_features),
        scorer=harness.scorer,
        masker=_InContextMasker(
            harness.masker,
            prefix=prompt[: feature.start] + lead,
            suffix=trail + prompt[feature.end :],
        ),
        sampler=harness.sampler,
        samples_per_coalition=harness.samples_per_coalition,
    )
    return DrilldownRefinement(feature=feature, result=stage.explain(prompt, tools=tools))


def _provider_calls(result: object, samples: int) -> int:
    evaluations = getattr(result, "evaluations", [])
    return len(evaluations) * samples + 1  # +1 for the baseline completion


class _FixedFeatures(Segmenter):
    """Return a precomputed feature list regardless of the prompt passed in."""

    def __init__(self, features: Sequence[Feature]) -> None:
        self.features = list(features)

    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        del prompt, tools
        return list(self.features)


class _InContextMasker(Masker):
    """Mask fine features with the base strategy, then restore the surrounding prompt.

    The base masker only ever sees the refined feature's sentences; the prefix
    and suffix (everything before and after the coarse feature, including its
    original surrounding whitespace) are reattached verbatim so each refined
    evaluation runs against the full instruction set.
    """

    def __init__(self, base: Masker, *, prefix: str, suffix: str) -> None:
        self.base = base
        self.prefix = prefix
        self.suffix = suffix

    def mask(self, features: Sequence[Feature], coalition: Coalition) -> str:
        return self.prefix + self.base.mask(features, coalition) + self.suffix
