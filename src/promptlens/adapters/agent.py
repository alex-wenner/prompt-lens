"""Attribution over whole agent runs instead of single completions.

The harness never inspects what an adapter does to produce its output, so the
unit under attribution can be an entire agent loop — multiple model turns and
tool executions — rather than one completion. :class:`AgentAdapter` makes that
explicit: the *prompt* the harness segments and masks is the agent's system
prompt, while the task (the user question or questions) is held fixed across
every coalition. Each evaluation answers "how does the agent's trajectory
change when this piece of its instructions is hidden?".

This keeps promptlens decoupled from any specific agent runtime (Strands
Agents, LangGraph, a hand-rolled loop): the runtime-specific part is a single
callable the user supplies.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

from promptlens.core.base import Adapter, CompletionOutput, ToolDefinitions
from promptlens.core.result import PerQuestionAttribution

if TYPE_CHECKING:
    from promptlens.core.harness import AttributionHarness
    from promptlens.core.result import AttributionResult

# One full, stateless agent run: (system_prompt, task, tools) -> CompletionOutput.
AgentRunner = Callable[[str, str, "ToolDefinitions | None"], CompletionOutput]


class AgentAdapter(Adapter):
    """Treat one full agent run as the completion under attribution.

    ``run_agent(system_prompt, task, tools)`` must execute a **fresh, stateless**
    agent run — build the agent from scratch each call so coalitions never share
    conversation memory — and return a :class:`CompletionOutput` whose ``text``
    is the final answer and whose ``tool_calls`` lists every tool invocation the
    agent made, in order. Use :func:`messages_to_output` to build that from a
    Bedrock/Strands-style message history.

    Score trajectories with :class:`~promptlens.scorers.ToolSequenceDriftScorer`
    (did the agent call different tools?), a text scorer over the final answer,
    or both combined via :class:`~promptlens.scorers.CompositeScorer`.

    Agent runs cost one provider call *per agent step*, not one per coalition,
    so :meth:`AttributionHarness.estimate` undercounts them — its input-token
    arithmetic assumes a single completion per evaluation. Budget accordingly.
    """

    def __init__(
        self,
        run_agent: AgentRunner,
        task: str,
        *,
        model: str = "agent",
        max_concurrency: int = 1,
    ) -> None:
        if max_concurrency < 1:
            msg = f"max_concurrency must be >= 1, got {max_concurrency}"
            raise ValueError(msg)
        self.run_agent = run_agent
        self.task = task
        self.model = model
        # Raise above 1 only when run_agent is thread-safe: each batch worker
        # invokes it concurrently.
        self.max_concurrency = max_concurrency

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        output = self.run_agent(prompt, self.task, tools)
        if not isinstance(output, CompletionOutput):
            msg = (
                "run_agent must return a CompletionOutput; use "
                "promptlens.adapters.messages_to_output to build one from an "
                "agent message history"
            )
            raise TypeError(msg)
        return output


def explain_per_question(
    harness: AttributionHarness,
    prompt: str,
    questions: Sequence[str],
    tools: ToolDefinitions | None = None,
) -> PerQuestionAttribution:
    """Attribute the system prompt separately for each question of a task.

    This is the finer "turn" granularity that is statistically sound for agent
    runs: the questions are fixed *inputs*, so answer ``i`` corresponds to
    question ``i`` in every coalition by construction — unlike model-generated
    turns, which stop being comparable once trajectories diverge. The result is
    a feature × question view: which instruction carries which question.

    Each question runs as its own independent agent run per coalition (cost
    scales with ``len(questions)``). Conversation-dependent behavior — where a
    question's answer should depend on earlier questions — belongs inside the
    ``run_agent`` callable (e.g. fold prior turns into the task string).

    ``harness.adapter`` must be an :class:`AgentAdapter`; its ``task`` is
    swapped per question and restored afterwards.
    """
    adapter = harness.adapter
    if not isinstance(adapter, AgentAdapter):
        msg = "explain_per_question requires the harness adapter to be an AgentAdapter"
        raise TypeError(msg)
    if not questions:
        msg = "explain_per_question requires at least one question"
        raise ValueError(msg)
    original_task = adapter.task
    results: list[AttributionResult] = []
    try:
        for question in questions:
            adapter.task = question
            results.append(harness.explain(prompt, tools=tools))
    finally:
        adapter.task = original_task
    return PerQuestionAttribution(questions=list(questions), results=results)


def messages_to_output(
    messages: Sequence[Any], *, raw: Any | None = None
) -> CompletionOutput:
    """Collapse an agent's message history into a :class:`CompletionOutput`.

    Accepts Bedrock Converse-style message lists — the format Strands Agents
    exposes as ``agent.messages`` — where each message is a mapping with a
    ``role`` and either a string ``content`` or a list of content blocks
    (``{"text": ...}``, ``{"toolUse": {"toolUseId", "name", "input"}}``).

    The returned ``text`` is the text of the *final* assistant message (the
    agent's answer) and ``tool_calls`` lists every ``toolUse`` block across the
    whole trajectory, in order, so trajectory scorers see the full tool path.
    """
    tool_calls: list[dict[str, Any]] = []
    final_text = ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "assistant":
            continue
        content = message.get("content", [])
        if isinstance(content, str):
            final_text = content
            continue
        text_parts: list[str] = []
        for block in content:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text_parts.append(str(block["text"]))
            if "toolUse" in block:
                tool_use = block["toolUse"]
                tool_calls.append(
                    {
                        "id": tool_use.get("toolUseId"),
                        "name": tool_use.get("name"),
                        "arguments": tool_use.get("input", {}),
                    }
                )
        if text_parts:
            final_text = "".join(text_parts)
    return CompletionOutput(
        text=final_text, tool_calls=tool_calls, raw=raw if raw is not None else list(messages)
    )
