"""Prompt text segmenters."""

from __future__ import annotations

import re

from promptlens.core.base import Feature, Segmenter, ToolDefinitions

_SENTENCE_RE = re.compile(r"[^.!?]+(?:[.!?]+|$)", re.MULTILINE)
_HEADING_RE = re.compile(r"(?m)^#{1,6}\s+.*$")


class SentenceSegmenter(Segmenter):
    """Split prompt text into sentence-like spans and append tools as opaque features."""

    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        features = [
            Feature(
                name=f"sentence_{index + 1}",
                text=match.group().strip(),
                start=match.start(),
                end=match.end(),
            )
            for index, match in enumerate(_SENTENCE_RE.finditer(prompt))
            if match.group().strip()
        ]
        if not features and prompt:
            features.append(Feature(name="prompt", text=prompt, start=0, end=len(prompt)))
        if tools:
            features.append(_tools_feature(tools))
        return features


class ParagraphSegmenter(Segmenter):
    """Split prompt text on blank lines."""

    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        features: list[Feature] = []
        offset = 0
        for index, paragraph in enumerate(part for part in prompt.split("\n\n") if part.strip()):
            start = prompt.find(paragraph, offset)
            end = start + len(paragraph)
            offset = end
            features.append(
                Feature(name=f"paragraph_{index + 1}", text=paragraph.strip(), start=start, end=end)
            )
        if tools:
            features.append(_tools_feature(tools))
        return features


class MarkdownSectionSegmenter(Segmenter):
    """Split markdown into heading-delimited sections."""

    def segment(self, prompt: str, tools: ToolDefinitions | None = None) -> list[Feature]:
        matches = list(_HEADING_RE.finditer(prompt))
        if not matches:
            return ParagraphSegmenter().segment(prompt, tools=tools)
        features: list[Feature] = []
        for index, match in enumerate(matches):
            end = matches[index + 1].start() if index + 1 < len(matches) else len(prompt)
            text = prompt[match.start() : end].strip()
            features.append(
                Feature(name=f"section_{index + 1}", text=text, start=match.start(), end=end)
            )
        if tools:
            features.append(_tools_feature(tools))
        return features


def _tools_feature(tools: ToolDefinitions) -> Feature:
    return Feature(name="tools", text=str(tools), metadata={"kind": "tools", "tools": tools})
