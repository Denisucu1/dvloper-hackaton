"""stderr-only logging. stdout belongs to the protocol (one JSON object per line, nothing else)."""
from __future__ import annotations

import sys
import traceback


def log(*parts: object) -> None:
    print("[bot]", *parts, file=sys.stderr, flush=True)


def log_exc(prefix: str, exc: BaseException) -> None:
    log(prefix, f"{type(exc).__name__}: {exc}")
    # Full traceback on stderr is free and makes local debugging humane.
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()


def kv(**fields: object) -> str:
    return " ".join(f"{k}={v}" for k, v in fields.items() if v is not None)
