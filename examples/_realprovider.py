"""Shared helper: run the examples against a real provider when one is configured.

By default the examples talk to a real model so they demonstrate genuine
attribution rather than a hand-rigged simulation. The provider is selected from
environment variables:

* ``PROMPTLENS_EXAMPLE_PROVIDER`` — ``openai`` (default) or ``anthropic``.
* ``PROMPTLENS_EXAMPLE_MODEL`` — optional model id override.

A real call only happens when the matching credential is present
(``OPENAI_API_KEY`` or ``ANTHROPIC_API_KEY``). When no credential is available —
as in CI — :func:`select_adapter` returns the deterministic offline adapter the
caller passes in, so the examples still run end-to-end and double as smoke tests
without network access.
"""

from __future__ import annotations

import os

from promptlens.core.base import Adapter

# Provider -> environment variable that signals a usable credential.
_PROVIDER_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_DEFAULT_PROVIDER = "openai"


def select_adapter(offline: Adapter, *, temperature: float = 0.0) -> Adapter:
    """Return a real provider adapter when configured, else ``offline``.

    ``offline`` is the example's deterministic simulated adapter, used whenever no
    provider credential is available so the demo (and the CI smoke test) keeps
    working without network access.
    """
    provider = os.environ.get("PROMPTLENS_EXAMPLE_PROVIDER", _DEFAULT_PROVIDER).strip().lower()
    model = os.environ.get("PROMPTLENS_EXAMPLE_MODEL") or None
    key_env = _PROVIDER_KEY_ENV.get(provider)
    if key_env is None:
        supported = ", ".join(sorted(_PROVIDER_KEY_ENV))
        print(
            f"[promptlens] Unknown PROMPTLENS_EXAMPLE_PROVIDER='{provider}' "
            f"(supported: {supported}); using the offline simulated adapter."
        )
        return offline
    if not os.environ.get(key_env):
        print(
            f"[promptlens] No {key_env} set; using the offline simulated adapter. "
            f"Export {key_env} (and optionally PROMPTLENS_EXAMPLE_MODEL) to run this "
            f"example against the real '{provider}' provider."
        )
        return offline
    # Imported lazily so the offline path never needs the provider SDK installed.
    from promptlens.cli.factories import build_adapter

    adapter = build_adapter(provider, model, temperature=temperature, base_url=None)
    print(f"[promptlens] Using real provider '{provider}' (model: {adapter.model}).")
    return adapter
