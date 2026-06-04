"""LLM-backed supplementary prompt mutations."""

from __future__ import annotations

from collections.abc import Sequence

from promptlens.core.base import Adapter, Feature, PromptMutation, PromptMutator, ToolDefinitions

_DEFAULT_INSTRUCTION = (
    "Rewrite the prompt by changing only the text between <mutate> and </mutate>. "
    "Preserve the prompt's intent, keep unrelated text unchanged, and return only the full "
    "rewritten prompt without the mutation tags."
)


class LLMRewriteMutator(PromptMutator):
    """Use an adapter to generate feature-level prompt rewrites."""

    def __init__(
        self,
        adapter: Adapter,
        *,
        rewrites_per_feature: int = 1,
        instruction: str = _DEFAULT_INSTRUCTION,
    ) -> None:
        if rewrites_per_feature < 1:
            msg = f"rewrites_per_feature must be >= 1, got {rewrites_per_feature}"
            raise ValueError(msg)
        self.adapter = adapter
        self.rewrites_per_feature = rewrites_per_feature
        self.instruction = instruction

    def mutate(
        self,
        prompt: str,
        features: Sequence[Feature],
        tools: ToolDefinitions | None = None,
    ) -> list[PromptMutation]:
        mutations: list[PromptMutation] = []
        for feature in features:
            marked_prompt = _mark_feature(prompt, features, feature)
            rewrite_prompt = f"{self.instruction}\n\nPrompt:\n{marked_prompt}"
            for repeat in range(self.rewrites_per_feature):
                output = self.adapter.complete(rewrite_prompt, tools=tools)
                mutations.append(
                    PromptMutation(
                        prompt=output.text.strip(),
                        feature=feature,
                        metadata={
                            "mutator": self.__class__.__name__,
                            "rewrite_model": self.adapter.model,
                            "repeat": repeat,
                        },
                    )
                )
        return mutations


def _mark_feature(prompt: str, features: Sequence[Feature], feature: Feature) -> str:
    if feature.start is not None and feature.end is not None:
        return (
            f"{prompt[: feature.start]}<mutate>{prompt[feature.start:feature.end]}"
            f"</mutate>{prompt[feature.end:]}"
        )
    parts = [
        f"<mutate>{item.text}</mutate>" if item is feature else item.text
        for item in features
    ]
    return " ".join(parts)
