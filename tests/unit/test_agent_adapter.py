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


def _routing_agent_run(
    system_prompt: str, task: str, tools: ToolDefinitions | None
) -> CompletionOutput:
    """Each instruction drives the tool path for a different kind of question."""
    del tools
    tool_calls = []
    if "refund" in task and "knowledge base" in system_prompt:
        tool_calls.append({"name": "search_kb", "arguments": {}})
    if "total" in task and "calculator" in system_prompt:
        tool_calls.append({"name": "calculator", "arguments": {}})
    return CompletionOutput(text="answer", tool_calls=tool_calls)


_ROUTING_PROMPT = "Check the knowledge base for policy questions. Use the calculator for math."


def _routing_harness() -> AttributionHarness:
    from promptlens.scorers import ToolSequenceDriftScorer

    return AttributionHarness(
        adapter=AgentAdapter(_routing_agent_run, task="unused"),
        segmenter=SentenceSegmenter(),
        scorer=ToolSequenceDriftScorer(),
    )


def test_explain_per_question_separates_drivers() -> None:
    from promptlens.adapters import explain_per_question

    questions = ["What is the refund policy?", "What is the total of 2 and 3?"]
    result = explain_per_question(_routing_harness(), _ROUTING_PROMPT, questions)

    assert result.questions == questions
    matrix = result.share_matrix()
    # The KB instruction carries the refund question only; the calculator
    # instruction carries the math question only.
    assert matrix["sentence_1"] == [1.0, 0.0]
    assert matrix["sentence_2"] == [0.0, 1.0]


def test_explain_per_question_restores_task() -> None:
    from promptlens.adapters import explain_per_question

    harness = _routing_harness()
    explain_per_question(harness, _ROUTING_PROMPT, ["What is the refund policy?"])

    assert harness.adapter.task == "unused"  # type: ignore[attr-defined]


def test_explain_per_question_requires_agent_adapter() -> None:
    from promptlens.adapters import EchoAdapter, explain_per_question
    from promptlens.scorers import LengthDriftScorer

    harness = AttributionHarness(
        adapter=EchoAdapter(),
        segmenter=SentenceSegmenter(),
        scorer=LengthDriftScorer(),
    )

    with pytest.raises(TypeError, match="AgentAdapter"):
        explain_per_question(harness, "A sentence.", ["q"])


def test_explain_per_question_requires_questions() -> None:
    from promptlens.adapters import explain_per_question

    with pytest.raises(ValueError, match="at least one question"):
        explain_per_question(_routing_harness(), _ROUTING_PROMPT, [])


def test_per_question_to_dict_includes_matrix() -> None:
    from promptlens.adapters import explain_per_question

    result = explain_per_question(
        _routing_harness(), _ROUTING_PROMPT, ["What is the refund policy?"]
    )
    data = result.to_dict()

    assert data["questions"] == ["What is the refund policy?"]
    assert "sentence_1" in data["share_matrix"]
    assert len(data["results"]) == 1
