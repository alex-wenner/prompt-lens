"""LLM-written synopsis of attribution evidence.

Attribution results carry a lot of structure — ranked features, per-coalition
drifts, supplementary mutations — and the table view answers "what moved" but
not "so what". :class:`LLMSynopsisWriter` hands the whole result to a model and
asks for a short narrative: which instructions carry the output, what is dead
weight, anything surprising, and what to try next.

The writer takes any :class:`~promptlens.core.base.Adapter`, so the synopsis
does not have to run on the model under attribution. Summarizing structured
evidence is well within reach of small open-weight models, so pointing this at
a local endpoint (the ``ollama`` or ``openai-compatible`` providers) keeps the
narrative step free even when the attribution run itself used a paid provider.
"""

from __future__ import annotations

from typing import Any

from promptlens.core.base import Adapter
from promptlens.core.result import AttributionResult, Synopsis

_DEFAULT_INSTRUCTION = (
    "You are an analyst summarizing black-box prompt attribution evidence for "
    "the engineer who owns the prompt. The evidence shows, per feature, how "
    "much the model's output changed when that feature was masked (its "
    "attribution share), plus the concrete outputs produced without each "
    "feature. Write a short plain-language synopsis covering: (1) which "
    "features carry the output and what they appear to control, (2) which "
    "features are dead weight, (3) anything surprising such as negative "
    "attribution or large drift from small text, and (4) two or three "
    "concrete next steps. Cite feature names. Do not invent evidence that is "
    "not shown. Keep it under 250 words and respond with prose only."
)


class LLMSynopsisWriter:
    """Use an adapter to write a narrative synopsis of an attribution result."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        instruction: str = _DEFAULT_INSTRUCTION,
        max_examples: int = 6,
        max_output_chars: int = 280,
    ) -> None:
        if max_examples < 0:
            msg = f"max_examples must be >= 0, got {max_examples}"
            raise ValueError(msg)
        if max_output_chars < 0:
            msg = f"max_output_chars must be >= 0, got {max_output_chars}"
            raise ValueError(msg)
        self.adapter = adapter
        self.instruction = instruction
        self.max_examples = max_examples
        self.max_output_chars = max_output_chars

    def summarize(self, prompt: str, result: AttributionResult) -> Synopsis:
        """Make one adapter call and return the model's synopsis of ``result``."""
        brief = self.build_brief(prompt, result)
        output = self.adapter.complete(brief)
        return Synopsis(
            text=output.text.strip(),
            model=self.adapter.model,
            metadata={
                "writer": self.__class__.__name__,
                "features": len(result.attributions),
                "evaluations": len(result.evaluations),
            },
        )

    def build_brief(self, prompt: str, result: AttributionResult) -> str:
        sections = [
            self.instruction,
            f"Prompt under attribution:\n{prompt}",
            f"Baseline output:\n{_truncate(result.baseline_output.text, self.max_output_chars)}",
        ]
        baseline_tools = _tool_sequence(result.baseline_output.tool_calls)
        if baseline_tools:
            sections.append(f"Baseline tool calls (in order): {baseline_tools}")
        sections.append(self._features_section(result))
        examples = self._examples_section(result)
        if examples:
            sections.append(examples)
        supplementary = self._supplementary_section(result)
        if supplementary:
            sections.append(supplementary)
        return "\n\n".join(sections)

    def _features_section(self, result: AttributionResult) -> str:
        lines = ["Ranked features (most load-bearing first):"]
        for attribution, share in result.ranked():
            feature = attribution.feature
            stderr = f" stderr={attribution.stderr:.4f}" if attribution.stderr is not None else ""
            text = _truncate(feature.text.replace("\n", " "), 120)
            lines.append(
                f"- {feature.name}: share={share * 100:.1f}% "
                f"value={attribution.value:.4f}{stderr} | {text}"
            )
        return "\n".join(lines)

    def _examples_section(self, result: AttributionResult) -> str:
        highlights = result.drift_highlights(limit=self.max_examples)
        if not highlights:
            return ""
        lines = ["Largest output drifts (what the model did without these features):"]
        for highlight in highlights:
            removed = ", ".join(highlight["removed"]) or "(none)"
            output_text = _truncate(
                highlight["output_text"].replace("\n", " "), self.max_output_chars
            )
            lines.append(f"- removed [{removed}]: drift={highlight['score']:.4f} -> {output_text}")
        return "\n".join(lines)

    def _supplementary_section(self, result: AttributionResult) -> str:
        if not result.supplementary_evaluations:
            return ""
        lines = ["Supplementary prompt mutations:"]
        for evaluation in result.supplementary_evaluations[: self.max_examples]:
            feature = evaluation.feature.name if evaluation.feature else "(whole prompt)"
            output_text = _truncate(
                evaluation.output.text.replace("\n", " "), self.max_output_chars
            )
            lines.append(
                f"- {evaluation.kind} on {feature}: score={evaluation.score:.4f} -> {output_text}"
            )
        return "\n".join(lines)


def _tool_sequence(tool_calls: list[dict[str, Any]]) -> str:
    names = [
        str(call.get("name") or call.get("function", {}).get("name") or "?")
        for call in tool_calls
    ]
    return " -> ".join(names)


def _truncate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[:limit].rstrip() + "…"
