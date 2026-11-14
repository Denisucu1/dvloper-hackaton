"""Gateway LLM client for AI Harness Arena (Team adn).

Owner: adn-1 (Architecture & Infra).
Strictly uses the arena's OpenAI-compatible Haiku gateway and always switches to
the newest credentials supplied in `prep` or `start`.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from openai import OpenAI

DEFAULT_THINKING_DISABLED = {"type": "disabled"}


def log(*parts: Any) -> None:
    """Write diagnostic output exclusively to stderr (stdout is reserved for JSON lines)."""
    print(*parts, file=sys.stderr, flush=True)


class GatewayLLMClient:
    """Manages connection state, key rotation, thinking config, and strict timeouts."""

    def __init__(self) -> None:
        self.base_url: str = os.environ.get("LLM_BASE_URL", "http://gateway:4000/v1")
        self.api_key: str = os.environ.get("LLM_API_KEY", "missing-key")
        self.model: str = os.environ.get("LLM_MODEL", "haiku")
        self._client: OpenAI | None = None
        self._client_creds: tuple[str, str] | None = None

    def update_from_message(self, msg: dict[str, Any]) -> None:
        """Rotate gateway URL, API key, and model from the latest `prep` or `start` message."""
        info = msg.get("llm") or {}
        if info.get("base_url"):
            self.base_url = str(info["base_url"])
        if info.get("api_key"):
            self.api_key = str(info["api_key"])
        if info.get("model"):
            self.model = str(info["model"])

    def _get_client(self) -> OpenAI:
        creds = (self.base_url, self.api_key)
        if self._client is None or self._client_creds != creds:
            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
            self._client_creds = creds
        return self._client

    def ask(
        self,
        system: str,
        user: str,
        timeout_seconds: float,
        *,
        max_tokens: int = 400,
        thinking_budget: int = 0,
        temperature: float | None = None,
    ) -> str:
        """Send one chat-completion request to Haiku 5.5 within `timeout_seconds`.

        When `thinking_budget` >= 1024, thinking is enabled with generous max_tokens
        headroom (`budget + 1500`). Otherwise thinking is explicitly disabled for speed.
        """
        extra_body: dict[str, Any]
        effective_max_tokens = max_tokens
        if thinking_budget >= 1024:
            extra_body = {"thinking": {"type": "enabled", "budget_tokens": thinking_budget}}
            effective_max_tokens = max(max_tokens, thinking_budget + 1500)
        else:
            extra_body = {"thinking": DEFAULT_THINKING_DISABLED}

        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "timeout": max(2.0, float(timeout_seconds)),
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "extra_body": extra_body,
        }
        if temperature is not None and thinking_budget < 1024:
            kwargs["temperature"] = temperature

        try:
            resp = self._get_client().chat.completions.create(**kwargs)
            return resp.choices[0].message.content or ""
        except Exception as exc:
            log("LLM error:", exc)
            return ""
