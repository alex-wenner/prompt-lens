"""LLM-backed prompt optimization driven by attribution evidence."""

from __future__ import annotations

from promptlens.core.base import Adapter, PromptOptimizer, ToolDefinitions
from promptlens.core.result import AttributionResult, OptimizationResult

_DEFAULT_INSTRUCTION = (
    "You are improving an LLM prompt using black-box attribution evidence. "
    "The evidence shows, per feature, how much removing that text changed the "
    "model output (its attribution share), plus concrete examples of how the "
    "output drifted when text was masked. Rewrite the prompt to strengthen and "
    "clarify the load-bearing instructions (high share) and prune or tighten "
    "inert dead-weight text (near-zero share), while preserving the prompt's "
    "original intent and the model's observed behavior."
)

_OUTPUT_CONTRACT = (
    "Return your answer in exactly this format:\n"
    "REWRITTEN PROMPT:\n"
    "<the full rewritten prompt>\n\n"
    "RATIONALE:\n"
    "<one short paragraph explaining the changes>"
)

_PROMPT_MARKER = "REWRITTEN PROMPT:"
_RATIONALE_MARKER = "RATIONALE:"

_VALIDATION_CAVEAT = (
    "Proposed rewrite is a candidate, not a verified improvement. Embedding and "
    "length scores can hide precision-critical changes (flipped numbers, "
    "negations, broken JSON); re-run attribution and task-level checks before "
    "adopting it."
)


class LLMPromptOptimizer(PromptOptimizer):
    """Use an adapter to propose a whole-prompt rewrite from attribution evidence."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        instruction: str = _DEFAULT_INSTRUCTION,
        max_permutations: int = 8,
        max_output_chars: int = 280,
        tools: ToolDefinitions | None = None,
    ) -> None:
        if max_permutations < 0:
            msg = f"max_permutations must be >= 0, got {max_permutations}"
            raise ValueError(msg)
        if max_output_chars < 0:
            msg = f"max_output_chars must be >= 0, got {max_output_chars}"
            raise ValueError(msg)
        self.adapter = adapter
        self.instruction = instruction
        self.max_permutations = max_permutations
        self.max_output_chars = max_output_chars
        self.tools = tools

    def optimize(self, prompt: str, result: AttributionResult) -> OptimizationResult:
        brief = self.build_brief(prompt, result)
        output = self.adapter.complete(brief, tools=self.tools)
        proposed_prompt, rationale = _parse_response(output.text, fallback=prompt)
        ranked = result.ranked()
        top_feature = ranked[0][0].feature.name if ranked else None
        metadata = {
            "optimizer": self.__class__.__name__,
            "rewrite_model": self.adapter.model,
            "evaluations": len(result.evaluations),
            "features": len(result.attributions),
            "top_feature": top_feature,
            "caveat": _VALIDATION_CAVEAT,
        }
        return OptimizationResult(
            original_prompt=prompt,
            proposed_prompt=proposed_prompt,
            rationale=rationale,
            metadata=metadata,
        )

    def build_brief(self, prompt: str, result: AttributionResult) -> str:
        sections = [
            self.instruction,
            f"Original prompt:\n{prompt}",
            f"Baseline output:\n{_truncate(result.baseline_output.text, self.max_output_chars)}",
            self._features_section(result),
        ]
        permutations = self._permutations_section(result)
        if permutations:
            sections.append(permutations)
        sections.append(_OUTPUT_CONTRACT)
        return "\n\n".join(sections)

    def _features_section(self, result: AttributionResult) -> str:
        lines = ["Ranked features (most load-bearing first):"]
        for attribution, share in result.ranked():
            feature = attribution.feature
            text = feature.text.replace("\n", " ")
            lines.append(
                f"- {feature.name}: share={share * 100:.1f}% value={attribution.value:.4f} "
                f"| {_truncate(text, 120)}"
            )
        return "\n".join(lines)

    def _permutations_section(self, result: AttributionResult) -> str:
        if self.max_permutations == 0 or not result.evaluations:
            return ""
        feature_names = [attribution.feature.name for attribution in result.attributions]
        top = sorted(result.evaluations, key=lambda item: item.score, reverse=True)
        top = top[: self.max_permutations]
        lines = ["Masked-prompt permutations (largest output drift first):"]
        for evaluation in top:
            removed = [
                feature_names[index]
                for index, included in enumerate(evaluation.coalition)
                if not included and index < len(feature_names)
            ]
            removed_label = ", ".join(removed) if removed else "(none)"
            output_text = _truncate(
                evaluation.output.text.replace("\n", " "), self.max_output_chars
            )
            lines.append(
                f"- removed [{removed_label}]: drift={evaluation.score:.4f} -> {output_text}"
            )
        return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"


def _parse_response(text: str, *, fallback: str) -> tuple[str, str]:
    stripped = text.strip()
    lowered = stripped.lower()
    prompt_index = lowered.find(_PROMPT_MARKER.lower())
    rationale_index = lowered.find(_RATIONALE_MARKER.lower())
    if prompt_index == -1:
        return (stripped or fallback), ""
    prompt_start = prompt_index + len(_PROMPT_MARKER)
    if rationale_index != -1 and rationale_index > prompt_index:
        proposed = stripped[prompt_start:rationale_index]
        rationale = stripped[rationale_index + len(_RATIONALE_MARKER) :]
    else:
        proposed = stripped[prompt_start:]
        rationale = ""
    proposed_clean = proposed.strip()
    return (proposed_clean or fallback), rationale.strip()
