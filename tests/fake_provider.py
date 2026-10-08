"""Mocked LLM provider for tests: a scripted async call function.

The service's test seam is ``_call_fn``: an async ``(system, user) -> str``
injected via ``TestGenerationService(settings, _call_fn=...)``.
"""

from __future__ import annotations

import json


def scripted_call(response, calls=None):
    """Build an async (system, user) -> str that always returns *response*.

    - dict/list  -> serialized as JSON (valid structured output)
    - str        -> returned verbatim (use for malformed output)
    - Exception  -> raised on every call (use for provider failures)
    """

    async def fn(system: str, user: str) -> str:
        if calls is not None:
            calls.append(user)
        if isinstance(response, Exception):
            raise response
        return response if isinstance(response, str) else json.dumps(response)

    return fn
