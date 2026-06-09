import pytest

from promptlens import AttributionHarness
from promptlens.adapters import AgentAdapter, messages_to_output
from promptlens.core.base import CompletionOutput, ToolDefinitions
from promptlens.scorers import ToolSequenceDriftScorer
from promptlens.segmenters import SentenceSegmenter

_TRIGGER = "Always check the knowledge base first."


def _fake_agent_run(
    system_prompt: str, task: str, tools: ToolDefinitions | None
) -> CompletionOutput:
    """Simulate a two-step agent whose tool path depends on its instructions."""
    del tools
    if _TRIGGER in system_prompt:
        messages = [
            {"role": "user", "content": [{"text": task}]},
            {
                "role": "assistant",
                "content": [
                    {"text": "Checking the KB."},
                    {"toolUse": {"toolUseId": "t1", "name": "search_kb", "input": {"q": task}}},
                ],
            },
            {"role": "user", "content": [{"toolResult": {"toolUseId": "t1"}}]},
            {
                "role": "assistant",
                "content": [
                    {"toolUse": {"toolUseId": "t2", "name": "answer", "input": {}}},
                    {"text": "Grounded answer."},
                ],
            },
        ]
    else:
        messages = [
            {"role": "user", "content": [{"text": task}]},
            {"role": "assistant", "content": [{"text": "Answering from memory."}]},
        ]
    return messages_to_output(messages)


def test_messages_to_output_collects_trajectory() -> None:
    output = _fake_agent_run(_TRIGGER, "What is our refund policy?", None)

    assert output.text == "Grounded answer."
    assert [call["name"] for call in output.tool_calls] == ["search_kb", "answer"]
    assert output.tool_calls[0]["arguments"] == {"q": "What is our refund policy?"}


def test_messages_to_output_handles_string_content() -> None:
    output = messages_to_output(
        [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )

    assert output.text == "hello"
    assert output.tool_calls == []


def test_agent_adapter_requires_completion_output() -> None:
    adapter = AgentAdapter(lambda prompt, task, tools: "not an output", task="t")  # type: ignore[arg-type, return-value]

    with pytest.raises(TypeError, match="CompletionOutput"):
        adapter.complete("system prompt")


def test_agent_attribution_finds_trajectory_driver() -> None:
    # The harness masks pieces of the *system prompt*; the task stays fixed.
    # Masking the trigger sentence collapses the agent's tool path, so the
    # trajectory scorer hands that sentence all of the attribution mass.
    harness = AttributionHarness(
        adapter=AgentAdapter(_fake_agent_run, task="What is our refund policy?"),
        segmenter=SentenceSegmenter(),
        scorer=ToolSequenceDriftScorer(),
    )

    result = harness.explain(f"{_TRIGGER} Be concise.")
    by_name = {a.feature.name: a for a in result.attributions}

    assert by_name["sentence_1"].value == 1.0
    assert by_name["sentence_2"].value == 0.0
    assert result.ranked()[0][0].feature.name == "sentence_1"
