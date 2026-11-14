"""Minimal OpenAI-compatible client for the arena gateway, stdlib only.

Why not the `openai` package: one less dependency in the image, no version drift, and we need exact
control over the `thinking` field, the timeout and the budget (rulebook 7.3-7.5).

Hard rules encoded here:
- only `llm.base_url` + `llm.model` from the latest `prep` / `start` (rule 7.2);
- thinking is disabled by default (fast, cheap, no empty answers);
- every call has a timeout strictly inside the clock we were given;
- a failed call returns "" instead of raising: a broken model must never cost a game.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Any

from . import config
from .log import log

_EMPTY = ""


class Gateway:
    """One gateway client per container; `configure()` on every prep/start (the key changes each match)."""

    def __init__(self) -> None:
        self.base_url = str(config.LLM["base_url"]).rstrip("/")
        self.api_key = str(config.LLM["api_key"])
        self.model = str(config.LLM["model"])
        self.spend = 0.0
        self.cap = config.PREP_BUDGET
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.failures = 0

    # ---------------------------------------------------------------- config
    def configure(self, info: dict | None, cap: float | None = None) -> None:
        """Adopt the key from the latest message. The arena deletes old keys, so this is mandatory."""
        info = info or {}
        for key in ("base_url", "api_key", "model"):
            value = info.get(key)
            if value:
                setattr(self, key, str(value))
        self.base_url = self.base_url.rstrip("/")
        if cap is not None:
            self.cap = float(cap)

    @property
    def available(self) -> bool:
        return bool(self.base_url and self.api_key and self.api_key != "missing-key")

    @property
    def affordable(self) -> bool:
        return self.available and self.spend < self.cap

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "spend_usd": round(self.spend, 4),
            "cap_usd": self.cap,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "failures": self.failures,
        }

    # ---------------------------------------------------------------- calls
    def ask(
        self,
        system: str,
        user: str,
        max_tokens: int = 600,
        timeout: float = 15.0,
        temperature: float = 0.5,
        thinking: str = "disabled",
        budget_tokens: int | None = None,
    ) -> str:
        """One chat completion. Returns "" on any failure (timeout, budget, HTTP error, empty answer)."""
        if not self.available or not self.affordable or timeout <= 0.5:
            return _EMPTY

        effective_max_tokens = max(64, int(max_tokens))
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": effective_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        # Rulebook 7.3: `extra_body={"thinking": ...}` in the SDK is a top-level field on the wire.
        # When thinking is enabled, omit temperature and give max_tokens headroom above budget_tokens.
        if thinking == "enabled":
            budget = max(1024, int(budget_tokens or 1024))
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
            body["max_tokens"] = max(effective_max_tokens, budget + 1500)
        else:
            body["thinking"] = {"type": "disabled"}
            body["temperature"] = float(temperature)

        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = json.loads(response.read().decode("utf-8", "replace"))
        except urllib.error.HTTPError as exc:
            self.failures += 1
            detail = ""
            try:
                detail = exc.read().decode("utf-8", "replace")[:200]
            except Exception:  # noqa: BLE001 - best effort diagnostics only
                pass
            log(f"llm http {exc.code}: {detail}")
            return _EMPTY
        except Exception as exc:  # noqa: BLE001 - timeouts, proxy, DNS, bad JSON: all non-fatal
            self.failures += 1
            log(f"llm error: {type(exc).__name__}: {exc}")
            return _EMPTY

        self.calls += 1
        self._meter(payload.get("usage") or {})
        try:
            content = payload["choices"][0]["message"].get("content") or ""
        except (KeyError, IndexError, TypeError):
            log("llm: unexpected response shape")
            return _EMPTY
        if not content.strip():
            # Rulebook 7.3: thinking can eat the whole max_tokens and return nothing.
            log("llm: empty answer (thinking or max_tokens)")
            return _EMPTY
        return content

    def ask_json(self, system: str, user: str, **kwargs: Any) -> dict | None:
        text = self.ask(system, user, **kwargs)
        data = extract_json(text)
        if data is None and text.strip():
            # The call succeeded but the answer was not JSON: keep a short trace so prep can explain it.
            log(f"llm: json parse failed ({len(text)} chars): {text.strip()[:140]!r}")
        return data

    def key_info(self) -> dict:
        """`GET <base without /v1>/key/info` - what this key has spent (rulebook 7.5)."""
        root = re.sub(r"/v1/?$", "", self.base_url)
        request = urllib.request.Request(
            f"{root}/key/info", headers={"Authorization": f"Bearer {self.api_key}"}, method="GET"
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return json.loads(response.read().decode("utf-8", "replace"))
        except Exception as exc:  # noqa: BLE001
            log("key_info failed:", type(exc).__name__)
            return {}

    def _meter(self, usage: dict) -> None:
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        cached = int((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        self.input_tokens += prompt
        self.output_tokens += completion
        fresh = max(0, prompt - cached)
        self.spend += (
            fresh * config.PRICE_INPUT
            + cached * config.PRICE_CACHE_READ
            + completion * config.PRICE_OUTPUT
        ) / 1_000_000


def extract_json(text: str) -> dict | None:
    """Tolerant JSON extraction: models like to wrap objects in prose or code fences."""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    try:
        value = json.loads(text)
        return value if isinstance(value, dict) else None
    except ValueError:
        pass
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            value = json.loads(text[start : end + 1])
            return value if isinstance(value, dict) else None
        except ValueError:
            return None
    return None
