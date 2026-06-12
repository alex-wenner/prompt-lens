"""OpenAI adapter using the official SDK."""

from __future__ import annotations

import io
import json
import time
from collections.abc import Sequence
from typing import Any

from promptlens.adapters.models import supports_logprobs
from promptlens.core.base import (
    Adapter,
    CompletionOutput,
    TokenUsage,
    ToolDefinitions,
    coerce_tools,
)

_CHAT_COMPLETIONS_ENDPOINT = "/v1/chat/completions"


class OpenAIAdapter(Adapter):
    """Thin wrapper around OpenAI chat completions.

    Set ``use_batch_api=True`` to route :meth:`complete_batch` through the
    OpenAI Batch API (50% cheaper, asynchronous with polling). The synchronous
    :meth:`complete` path is unchanged.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        logprobs: bool = False,
        client: Any | None = None,
        use_batch_api: bool = False,
        poll_interval_seconds: float = 5.0,
    ) -> None:
        if poll_interval_seconds <= 0:
            msg = f"poll_interval_seconds must be > 0, got {poll_interval_seconds}"
            raise ValueError(msg)
        if logprobs and not supports_logprobs(model):
            msg = (
                f"Model {model!r} does not support logprobs. Construct the adapter "
                "with logprobs=False, or choose a model that returns log "
                "probabilities (e.g. gpt-4o or gpt-4.1)."
            )
            raise ValueError(msg)
        self.model = model
        self.temperature = temperature
        self.logprobs = logprobs
        self._client = client
        self.use_batch_api = use_batch_api
        self.poll_interval_seconds = poll_interval_seconds

    def _request_body(self, prompt: str, tools: ToolDefinitions | None) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.temperature,
        }
        if tools:
            body["tools"] = coerce_tools(tools, "openai")
        if self.logprobs:
            body["logprobs"] = True
        return body

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        client = self._client or _default_client()
        response = client.chat.completions.create(**self._request_body(prompt, tools))
        choice = response.choices[0]
        message = choice.message
        text = message.content or ""
        tool_calls = [_openai_tool_call_to_dict(call) for call in (message.tool_calls or [])]
        logprobs = _extract_logprobs(choice)
        return CompletionOutput(
            text=text,
            tool_calls=tool_calls,
            logprobs=logprobs,
            usage=_extract_usage(getattr(response, "usage", None)),
            raw=response,
        )

    def complete_batch(
        self, prompts: Sequence[str], tools: ToolDefinitions | None = None
    ) -> list[CompletionOutput]:
        if not self.use_batch_api or len(prompts) <= 1:
            return super().complete_batch(prompts, tools=tools)
        client = self._client or _default_client()
        payload = "\n".join(
            json.dumps(
                {
                    "custom_id": _custom_id(index),
                    "method": "POST",
                    "url": _CHAT_COMPLETIONS_ENDPOINT,
                    "body": self._request_body(prompt, tools),
                }
            )
            for index, prompt in enumerate(prompts)
        )
        upload = client.files.create(
            file=io.BytesIO(payload.encode("utf-8")), purpose="batch"
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint=_CHAT_COMPLETIONS_ENDPOINT,
            completion_window="24h",
        )
        completed = self._await_batch(client, batch.id)
        outputs_by_id = self._parse_output_file(client, completed.output_file_id)
        return [outputs_by_id[_custom_id(index)] for index in range(len(prompts))]

    def _await_batch(self, client: Any, batch_id: str) -> Any:
        while True:
            current = client.batches.retrieve(batch_id)
            if current.status == "completed":
                return current
            if current.status in {"failed", "expired", "cancelled", "cancelling"}:
                msg = f"OpenAI batch {batch_id} ended as {current.status}"
                raise RuntimeError(msg)
            time.sleep(self.poll_interval_seconds)

    def _parse_output_file(self, client: Any, output_file_id: str) -> dict[str, CompletionOutput]:
        content = client.files.content(output_file_id)
        text = content.text if hasattr(content, "text") else content.read().decode("utf-8")
        outputs: dict[str, CompletionOutput] = {}
        for line in text.splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            response = record.get("response") or {}
            if response.get("status_code") != 200:
                msg = f"OpenAI batch request {record.get('custom_id')} failed"
                raise RuntimeError(msg)
            outputs[record["custom_id"]] = _completion_body_to_output(response.get("body", {}))
        return outputs


def _default_client() -> Any:
    try:
        from openai import OpenAI
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[openai] to use OpenAIAdapter"
        raise RuntimeError(msg) from exc
    return OpenAI()


def _custom_id(index: int) -> str:
    return f"req-{index}"


def _openai_tool_call_to_dict(call: Any) -> dict[str, Any]:
    function = getattr(call, "function", None)
    return {
        "id": getattr(call, "id", None),
        "name": getattr(function, "name", None),
        "arguments": getattr(function, "arguments", None),
    }


def _completion_body_to_output(body: dict[str, Any]) -> CompletionOutput:
    """Map a JSON chat-completion body (from the Batch API) to CompletionOutput."""
    choices = body.get("choices") or [{}]
    choice = choices[0]
    message = choice.get("message", {})
    tool_calls = [
        {
            "id": call.get("id"),
            "name": call.get("function", {}).get("name"),
            "arguments": call.get("function", {}).get("arguments"),
        }
        for call in (message.get("tool_calls") or [])
    ]
    logprobs_content = (choice.get("logprobs") or {}).get("content")
    logprobs = (
        [float(item["logprob"]) for item in logprobs_content] if logprobs_content else None
    )
    usage = body.get("usage") or {}
    return CompletionOutput(
        text=message.get("content") or "",
        tool_calls=tool_calls,
        logprobs=logprobs,
        usage=(
            TokenUsage(
                input_tokens=int(usage["prompt_tokens"]),
                output_tokens=int(usage["completion_tokens"]),
            )
            if "prompt_tokens" in usage and "completion_tokens" in usage
            else None
        ),
        raw=body,
    )


def _extract_usage(usage: Any) -> TokenUsage | None:
    """Map the SDK usage object (prompt/completion token counts) to TokenUsage."""
    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if prompt_tokens is None or completion_tokens is None:
        return None
    return TokenUsage(input_tokens=int(prompt_tokens), output_tokens=int(completion_tokens))


def _extract_logprobs(choice: Any) -> list[float] | None:
    logprobs = getattr(choice, "logprobs", None)
    content = getattr(logprobs, "content", None) if logprobs is not None else None
    if not content:
        return None
    return [float(item.logprob) for item in content]
