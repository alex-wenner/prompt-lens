"""GitHub Copilot adapter using the official ``github-copilot-sdk``.

Unlike the generic OpenAI-compatible path, GitHub Copilot is driven through its
own SDK, which talks to the bundled Copilot CLI runtime over JSON-RPC rather than
a plain HTTP Chat Completions endpoint. This adapter wraps that asynchronous,
session-based SDK behind promptlens's synchronous :class:`Adapter` interface.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Coroutine
from typing import Any

from promptlens.core.base import Adapter, CompletionOutput, TokenUsage, ToolDefinitions


class CopilotAdapter(Adapter):
    """Thin wrapper around the official GitHub Copilot SDK.

    Each :meth:`complete` call runs in a fresh, stateless Copilot session so
    attribution coalitions never share conversation memory. The SDK is
    asynchronous, so the adapter owns a private event loop on a background thread
    and reuses a single Copilot CLI runtime across calls.

    The Copilot CLI controls sampling, so ``temperature`` is accepted for
    interface parity but is not forwarded to the runtime. OpenAI-style tool
    schemas are likewise not forwarded, because the Copilot SDK exposes a
    different custom-tool model; tool *requests* the assistant makes are still
    captured on the returned :class:`CompletionOutput`.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0.0,
        github_token: str | None = None,
        timeout: float = 120.0,
        client: Any | None = None,
    ) -> None:
        if timeout <= 0:
            msg = f"timeout must be > 0, got {timeout}"
            raise ValueError(msg)
        self.model = model
        self.temperature = temperature
        self.github_token = github_token
        self.timeout = timeout
        self._client = client
        self._owns_client = client is None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    def complete(self, prompt: str, tools: ToolDefinitions | None = None) -> CompletionOutput:
        result = self._run(self._acomplete(prompt, tools))
        assert isinstance(result, CompletionOutput)
        return result

    async def _acomplete(
        self, prompt: str, tools: ToolDefinitions | None
    ) -> CompletionOutput:
        client = self._client or _default_client(self.github_token)
        self._client = client
        session = await client.create_session(
            model=self.model,
            on_permission_request=_approve_all_handler(),
        )
        try:
            event = await session.send_and_wait(prompt, timeout=self.timeout)
        finally:
            await session.disconnect()
        return _event_to_output(event)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._lock:
            if self._loop is None:
                loop = asyncio.new_event_loop()
                thread = threading.Thread(
                    target=loop.run_forever,
                    name="promptlens-copilot",
                    daemon=True,
                )
                thread.start()
                self._loop = loop
                self._thread = thread
            return self._loop

    def _run(self, coro: Coroutine[Any, Any, Any]) -> Any:
        if self._closed:
            coro.close()
            msg = "CopilotAdapter has been closed"
            raise RuntimeError(msg)
        loop = self._ensure_loop()
        return asyncio.run_coroutine_threadsafe(coro, loop).result()

    def close(self) -> None:
        """Stop the Copilot runtime and tear down the background event loop."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            loop = self._loop
            client = self._client
        if loop is None:
            return
        try:
            if self._owns_client and client is not None:
                asyncio.run_coroutine_threadsafe(client.stop(), loop).result(
                    timeout=self.timeout
                )
        finally:
            loop.call_soon_threadsafe(loop.stop)
            if self._thread is not None:
                self._thread.join(timeout=self.timeout)
            loop.close()

    def __del__(self) -> None:  # pragma: no cover - best-effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _default_client(github_token: str | None) -> Any:
    try:
        from copilot import CopilotClient
    except ImportError as exc:  # pragma: no cover - exercised without optional extra
        msg = "Install promptlens[copilot] to use CopilotAdapter"
        raise RuntimeError(msg) from exc
    if github_token:
        return CopilotClient(github_token=github_token)
    return CopilotClient()


def _approve_all_handler() -> Any:
    try:
        from copilot.session import PermissionHandler
    except ImportError:  # pragma: no cover - tests inject a fake client
        return None
    return PermissionHandler.approve_all


def _event_to_output(event: Any) -> CompletionOutput:
    if event is None:
        return CompletionOutput(text="")
    data = getattr(event, "data", None)
    text = getattr(data, "content", "") or ""
    tool_requests = getattr(data, "tool_requests", None) or []
    tool_calls = [
        {
            "id": getattr(request, "tool_call_id", None),
            "name": getattr(request, "name", None),
            "arguments": getattr(request, "arguments", None),
        }
        for request in tool_requests
    ]
    return CompletionOutput(
        text=str(text),
        tool_calls=tool_calls,
        usage=_extract_usage(getattr(data, "usage", None)),
        raw=event,
    )


def _extract_usage(usage: Any) -> TokenUsage | None:
    """Best-effort usage mapping; the Copilot runtime does not always meter."""
    input_tokens = getattr(usage, "input_tokens", None)
    output_tokens = getattr(usage, "output_tokens", None)
    if input_tokens is None or output_tokens is None:
        return None
    return TokenUsage(input_tokens=int(input_tokens), output_tokens=int(output_tokens))
